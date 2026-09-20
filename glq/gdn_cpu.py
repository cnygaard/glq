"""Route GatedDeltaNet's decode step to GLQ's fused CPU kernel.

On CPU, 36 of Qwen3.8-Flash-Next's 48 layers run transformers' reference PyTorch rule --
every CPU log says so ("falling back to its reference PyTorch implementation"). That
reference walks the K x V recurrent state five times per token and allocates four
temporaries of that size; ``glq_gdn_recurrent_step_cpu`` does it in two passes with none.
Measured 34.8x on the kernel at 48 heads x 128 x 128 (4-core AVX2 machine).

**The seam, and why it is a module global.** Qwen4Exp does not bind the rule as an instance
attribute the way qwen3_next does. ``torch_recurrent_gated_delta_rule`` is a module-level
function decorated with ``@use_kernel_func_from_hub_with_fallback("...", "fla")`` and the
forward calls it by global name, so replacing the global covers every layer at once.

**Why we wrap instead of replace.** That decorator resolves its implementation at IMPORT
time -- priority Hub kernel, then the original package, then the torch path -- and its body
ends in ``implementation(*args, **kwargs)`` with no device check. Two consequences worth
knowing: installing ``causal-conv1d`` or ``flash-linear-attention`` on a CPU-only box binds
a CUDA implementation and breaks the model rather than merely failing to help; and whatever
is already bound may not be the reference. So the shim keeps a reference to the existing
global and delegates to it for everything it does not handle -- prefill, CUDA tensors, no
cache state, unsupported dims. The failure mode is "as fast as before", never "wrong".

Off by default behind ``GLQ_CPU_GDN`` while the end-to-end win is measured, matching how
the fused CPU MoE path shipped.
"""
from __future__ import annotations

import os
import sys
import warnings

import torch

#: One-shot, so a 36-layer model does not repeat itself once per layer per token.
_WARNED_GDN_CPU = False

#: Module globals we wrap, in the modules that define a GatedDeltaNet.
_TARGET = "torch_recurrent_gated_delta_rule"


def _enabled() -> bool:
    return os.environ.get("GLQ_CPU_GDN", "0") != "0"


def _ext():
    """The CPU extension carrying the entry, or None."""
    from . import inference_kernel_cpu as _ikc
    if not _ikc._try_load_cpu_ext():
        return None
    ext = _ikc._glq_cpu
    return ext if hasattr(ext, "glq_gdn_recurrent_step_cpu") else None


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """The reference's own formula, not ``F.normalize``.

    ``F.normalize`` clamps the denominator (``max(||x||, eps)``) where the reference adds
    eps under the sqrt (``rsqrt(sum(x*x) + eps)``). They differ, and on a normalised vector
    the gap lands right where the delta rule is most sensitive.
    """
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _warn(why: str) -> None:
    global _WARNED_GDN_CPU
    if _WARNED_GDN_CPU:
        return
    _WARNED_GDN_CPU = True
    warnings.warn(
        f"GLQ_CPU_GDN is set but GatedDeltaNet is using the reference PyTorch rule: {why}. "
        f"Correct, but materially slower.", RuntimeWarning)


def _make_shim(orig):
    """Wrap ``orig`` so eligible decode steps take the kernel and everything else does not."""

    def _shim(query, key, value, g=None, beta=None, initial_state=None,
              output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        ext = None
        why = None
        if not _enabled():
            why = "skip"                       # not a refusal; the flag is simply off
        elif query.is_cuda:
            why = "skip"                       # CUDA has its own fused path
        elif query.dim() != 4 or query.shape[1] != 1:
            why = "skip"                       # prefill: the chunked rule, not this one
        elif initial_state is None:
            why = "skip"                       # no cache state -> first token, take the ref
        else:
            ext = _ext()
            if ext is None:
                why = "the CPU extension has no glq_gdn_recurrent_step_cpu entry"
        if ext is None:
            if why and why != "skip":
                _warn(why)
            return orig(query, key, value, g=g, beta=beta, initial_state=initial_state,
                        output_final_state=output_final_state,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)

        initial_dtype = query.dtype
        k_head_dim = key.shape[-1]

        # (B, S=1, H, D) -> (B, H, D), fp32, exactly as the reference prepares them.
        def _prep(x):
            return x.transpose(1, 2).to(torch.float32).contiguous().squeeze(2)

        q, k, v = _prep(query), _prep(key), _prep(value)
        if use_qk_l2norm_in_kernel:
            q, k = _l2norm(q), _l2norm(k)
        q = q / (k_head_dim ** 0.5)

        # g and beta arrive (B, S, H); the reference transposes then indexes the token.
        gg = g.transpose(1, 2).to(torch.float32).contiguous().squeeze(-1)
        bb = beta.transpose(1, 2).to(torch.float32).contiguous().squeeze(-1)

        # The kernel writes through this tensor, which is what removes the reference's four
        # temporaries. If the cache holds it in another dtype or layout we must copy, and
        # then the caller's write-back of the returned object is what keeps it correct.
        state = initial_state
        if state.dtype != torch.float32 or not state.is_contiguous():
            state = state.to(torch.float32).contiguous()

        out = ext.glq_gdn_recurrent_step_cpu(state, q, k, v, gg, bb)
        core_attn_out = out.unsqueeze(1).to(initial_dtype)        # (B, 1, H, V)
        return core_attn_out, (state if output_final_state else None)

    _shim._glq_gdn_orig = orig          # so install() is idempotent and reversible
    return _shim


def install(model) -> int:
    """Wrap the recurrent rule in every module that defines a GatedDeltaNet. Idempotent.

    Returns the number of modeling modules patched. Resolves the target by walking the
    model's own classes rather than importing a specific modeling module, because the
    architectures differ: Qwen4Exp keeps this as a decorated module global, qwen3_next binds
    an instance attribute.
    """
    patched = 0
    seen: set[str] = set()
    for mod in model.modules():
        cls = type(mod)
        if "GatedDeltaNet" not in cls.__name__ or cls.__module__ in seen:
            continue
        seen.add(cls.__module__)
        mm = sys.modules.get(cls.__module__)
        fn = getattr(mm, _TARGET, None)
        if fn is None or getattr(fn, "_glq_gdn_orig", None) is not None:
            continue                       # not this shape, or already wrapped
        setattr(mm, _TARGET, _make_shim(fn))
        patched += 1
    return patched


def uninstall(model) -> int:
    """Restore the original globals. Exists so tests can A/B in one process."""
    restored = 0
    for mod in model.modules():
        cls = type(mod)
        if "GatedDeltaNet" not in cls.__name__:
            continue
        mm = sys.modules.get(cls.__module__)
        fn = getattr(mm, _TARGET, None)
        orig = getattr(fn, "_glq_gdn_orig", None) if fn is not None else None
        if orig is not None:
            setattr(mm, _TARGET, orig)
            restored += 1
    return restored
