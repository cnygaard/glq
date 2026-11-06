"""Pure GLQ MoE dispatch-policy helpers.

Kept dependency-free (no torch / no vllm / no env reads) so the kernel-selection
policy is unit-testable on a CPU box without a GPU or vLLM installed. The methods
in ``fused_moe_method`` read the env and tensors, then delegate the *decision* here.
"""


def _grouped_enabled(env_val, grouped_min: int, num_tokens: int) -> bool:
    """Tri-state decision for the Stage-3 grouped-GEMM MoE path (GLQ_MOE_GROUPED).

    The grouped path sorts tokens by expert and runs one batched tensor-core GEMM
    per expert — the batched-decode throughput win (validated 26B-A4B: b32 6.9-8.4x
    over the per-(token,expert) block-diag matvec; b1 ~1.15x). It is numerically
    equivalent to but not bit-identical with the matvec oracle (TC accumulation),
    so by default b1 stays on the bit-exact block-diag path.

      "0"/"off"/"false"/"no"  -> never grouped (force block-diag; A/B isolation)
      "1"/"on"/"true"/"yes"   -> force grouped whenever the rest of the gate holds
                                 (including b1)
      None / "" / "auto"/else -> grouped only for batched MoE (num_tokens >=
                                 grouped_min); b1 keeps the block-diag matvec.
    """
    v = (env_val or "auto").strip().lower()
    if v in ("0", "off", "false", "no"):
        return False
    if v in ("1", "on", "true", "yes"):
        return True
    return num_tokens >= grouped_min


#: Codebooks whose MoE experts can run on the CPU platform. Only trellis: its per-expert
#: fallback (fused_moe_method._apply_trellis) reaches E8RHTLinear._trellis_linear_apply,
#: which has a CPU branch, and every tensor it passes follows the packed buffers' device.
#: e8p and shell have no CPU expert path at all — their per-expert dequant asserts
#: sv.is_cuda — so they must refuse rather than fail deep inside a forward pass.
_CPU_MOE_CODEBOOKS = frozenset({"trellis"})


def moe_cpu_refusal(codebook: str) -> str | None:
    """Why this MoE cannot serve on the CPU platform, or None if it can.

    Unknown codebooks are refused: opting in is a decision that needs a CPU expert path
    behind it, and a wrong "yes" here surfaces as garbage output rather than an error.
    """
    if codebook in _CPU_MOE_CODEBOOKS:
        return None
    return (f"GLQ MoE on the CPU platform is supported for trellis checkpoints only; this "
            f"one is {codebook}, whose expert decode kernels are CUDA-only. Serve it on a "
            f"GPU, or pick a trellis checkpoint for CPU serving.")


# The fused-CPU-MoE gate lives in `glq.moe_cpu_gate` because HF's
# `GLQStackedGatedExperts` reaches the same op and `glq` must not import `glq_vllm`.
# Re-exported here so this module stays the one place vLLM asks about dispatch.
from glq.moe_cpu_gate import moe_cpu_fused_refusal  # noqa: E402,F401


#: Activation ids the fused CUDA trellis MoE op implements: 0/1/2 the GATED
#: silu / gelu-tanh / relu2, and 3/4/5 the same three NON-GATED. NemotronH is 5
#: (``up -> relu**2 -> down``, no gate projection).
#:
#: This exists as a named predicate rather than an inline ``< 3`` because getting it wrong
#: is expensive and silent in one direction and loud in the other. Too narrow, and the layer
#: falls back to ``fused_moe_method._apply_trellis`` — correct, but its ``topk_ids.unique()``
#: is a host sync, so CUDA-graph capture dies with
#: ``cudaErrorStreamCaptureUnsupported`` at *every* batch size and the model cannot serve
#: without ``--enforce-eager``. Too wide, and an id outside the table reaches a kernel that
#: selects the function with ``% 3`` and would quietly compute silu.
_TRELLIS_MOE_ACTIVATIONS = frozenset(range(6))


def trellis_moe_activation_ok(activation_type: int | None) -> bool:
    """Can the fused CUDA trellis MoE op evaluate this activation id?

    ``None`` means the layer never declared one, which is not something to guess at: the
    fallback loop reads ``layer.activation`` itself and applies whatever it finds.
    """
    return activation_type in _TRELLIS_MOE_ACTIVATIONS


#: vLLM activation name -> GLQ kernel activation id. 0/1/2 gated, 3/4/5 the same three
#: non-gated, matching the kernel's ``act_type % 3`` selection and its ``act_type < 3``
#: up-multiply. Lives beside ``trellis_moe_activation_ok`` on purpose: the table and the
#: predicate that validates it drifting apart is the failure this module exists to prevent,
#: and keeping it here (dependency-free) makes both testable without vLLM installed.
_ACTIVATION_IDS = {
    "silu": 0,
    "gelu": 1, "gelu_tanh": 1, "gelu_pytorch_tanh": 1,
    "relu2": 2,
    "silu_no_mul": 3,
    "gelu_no_mul": 4,
    "relu2_no_mul": 5,
}

#: Returned for an activation the kernels do not implement. Must be **>= 6**, not -1 or
#: None, because three separate guards have to refuse it:
#:
#:   * ``trellis_moe_activation_ok`` — ``in frozenset(range(6))``
#:   * the e8p and shell paths — ``_activation_type(...) < 3``
#:   * ``moe_cpu_fused_refusal`` — ``0 <= activation_type < 3``
#:
#: A negative sentinel passes the second one and reaches a kernel whose ``act_type % 3``
#: is -1 in C++ (not 2), falling through to relu**2 with the gated up-multiply applied.
#: A value >= 6 fails all three, and the kernel's own
#: ``TORCH_CHECK(0 <= activation_type < 6)`` then fails loudly if one is ever missed.
ACTIVATION_UNSUPPORTED = 99


def activation_id(name) -> int:
    """GLQ kernel activation id for a vLLM activation, or ``ACTIVATION_UNSUPPORTED``.

    **Unnamed activations are refused, never guessed.** This used to default to 5 — the
    non-gated relu**2 id — which was harmless only while 5 was itself refused everywhere.
    Once the fused path accepted the non-gated ids, that default became silent
    miscomputation for five activations vLLM actually ships: ``swigluoai``, ``situ``,
    ``swigluoai_uninterleave`` and ``swiglustep`` are all **gated**, so computing them as
    non-gated relu**2 also reads the wrong number of w13 halves; ``gelu_tanh_no_mul`` is
    non-gated but is not relu**2. None of them raise, none of them slow down, and nothing
    downstream can detect the result.

    Refusing costs a fall back to the per-expert loop, which is correct on every shape.
    """
    act = name.value if hasattr(name, "value") else str(name) if name else "silu"
    return _ACTIVATION_IDS.get(act, ACTIVATION_UNSUPPORTED)


def moe_swiglu_limit(moe, layer) -> float:
    """The clamp vLLM applies inside this layer's SwiGLU, or 0.0 for none.

    GLM-5.3 computes ``silu(min(gate, L)) * clamp(up, -L, L)``, and vLLM passes ``L`` to
    FusedMoE while the activation stays named ``silu`` — so the activation id alone cannot
    see it. vLLM 0.32 stores it on ``FusedMoEConfig`` and again on ``RoutedExperts``; the
    config is read first. Only a positive limit clamps, as in vLLM's ``_swiglu_limit_torch``.
    """
    limit = getattr(moe, "swiglu_limit", None)
    if limit is None:
        limit = getattr(layer, "swiglu_limit", None)
    return float(limit) if limit is not None and limit > 0 else 0.0


def swiglu_limit_refusal(codebook: str, activation_type: int, limit: float) -> str | None:
    """Why this MoE cannot apply its SwiGLU clamp, or None if it can (or has none).

    Only the trellis path implements the clamp: its fused CUDA op, its per-expert loop and
    the CPU loop all do. The e8p and shell kernels do not, and a fallback would not help —
    their loops would compute the same unclamped function. So this is a load-time refusal,
    not a dispatch preference.
    """
    if limit <= 0:
        return None
    if codebook != "trellis":
        return (f"this MoE clamps its SwiGLU (swiglu_limit={limit:g}), which GLQ implements "
                f"for trellis checkpoints only; this one is {codebook}. Serving it would "
                f"compute the unclamped activation. Re-quantize with --codebook trellis.")
    if activation_type != 0:
        return (f"swiglu_limit={limit:g} with activation id {activation_type}: GLQ applies the "
                f"clamp to gated silu only, as vLLM does.")
    return None
