"""vLLM CPU-platform integration (phase 2 of the CPU fused-decode work).

These tests need an importable vllm (any install, including the +cpu wheel or a source
checkout) but NO GPU and NO CUDA extension — that is the point: they pin the behaviors
that make `--quantization glq` viable on vLLM's CPU backend:

* `embedding_dequant` registers under the CPU dispatch key even when the CUDA extension
  is absent (its real impl is pure torch) — the gemma-4 PLE path.
* The MoE quant method refuses loudly on the CPU platform (every GLQ MoE path bottoms
  out in CUDA kernels; a mid-load failure would be far harder to read).
* fp32 joins the supported activation dtypes (the CPU platform's list is bf16-first).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# importorskip on the SUBMODULE: a bare source checkout at the repo root shadows `vllm`
# as an empty namespace package, so `import vllm` succeeding proves nothing.
pytest.importorskip("vllm.platforms", reason="no installed vllm (wheel needed)")

from vllm.platforms import current_platform  # noqa: E402

pytestmark = pytest.mark.skipif(
    getattr(current_platform, "device_type", "") != "cpu",
    reason="these gates target the CPU platform (dispatch key CPU)")


def test_embedding_dequant_registers_without_the_cuda_ext():
    from glq_vllm import custom_ops
    custom_ops._ensure_registered()
    assert hasattr(torch.ops, "glq") and hasattr(torch.ops.glq, "embedding_dequant"), \
        "embedding_dequant must register on the CPU platform without the CUDA extension"


def test_embedding_dequant_cpu_matches_the_pure_torch_impl():
    from glq.quantized_linear import _dequant_embedding_rows
    from glq_vllm import custom_ops
    custom_ops._ensure_registered()

    torch.manual_seed(0)
    vocab, n_pad, dim = 32, 64, 64
    qidxs = torch.randint(0, 255, (vocab, n_pad // 8), dtype=torch.uint8)
    sv = torch.where(torch.rand(n_pad) < 0.5, -1.0, 1.0).half()
    wscale = torch.full((vocab,), 0.5)
    codebook = torch.randn(256, 8).half()
    ids = torch.tensor([3, 7, 1])

    ref = _dequant_embedding_rows(ids, qidxs, sv, wscale, codebook, None, None, None,
                                  n_pad, dim, 1.0, torch.float16)
    out = torch.ops.glq.embedding_dequant(ids, qidxs, sv, wscale, codebook, None, None,
                                          None, n_pad, dim, 1.0, torch.float16)
    assert torch.equal(out, ref)


def test_trellis_moe_is_accepted_on_cpu():
    """The per-expert fallback reaches the CPU decode branch, so a trellis MoE serves —
    unfused and slow, but correct. This is what lets the 26B-A4B run without a GPU."""
    from glq_vllm.config import GLQvLLMConfig
    try:
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            RoutedExperts as _MoELayer)
    except ImportError:
        try:
            from vllm.model_executor.layers.fused_moe.layer import FusedMoE as _MoELayer
        except ImportError:
            pytest.skip("no MoE layer class in this vllm")

    cfg = GLQvLLMConfig(bpw=4, codebook="trellis", variant="3inst",
                        trellis_layout="kernel")
    layer = _MoELayer.__new__(_MoELayer)
    layer.moe_config = None
    method = GLQvLLMConfig.get_quant_method(cfg, layer, prefix="model.layers.0.mlp.experts")
    assert method is not None and "MoE" in type(method).__name__


def test_moe_quant_method_refuses_on_cpu():
    from glq_vllm.config import GLQvLLMConfig
    try:
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            RoutedExperts as _MoELayer)
    except ImportError:
        try:
            from vllm.model_executor.layers.fused_moe.layer import FusedMoE as _MoELayer
        except ImportError:
            pytest.skip("no MoE layer class in this vllm")

    cfg = GLQvLLMConfig(bpw=4, codebook="e8p")
    layer = _MoELayer.__new__(_MoELayer)          # isinstance without __init__ plumbing
    with pytest.raises(NotImplementedError, match="trellis checkpoints only"):
        GLQvLLMConfig.get_quant_method(cfg, layer, prefix="model.layers.0.mlp.experts")


# ---- the fused CPU MoE op is what actually serves ----------------------------------------
#
# `_apply_trellis` (the per-expert loop) and the fused CPU op produce nearly the same
# numbers, so comparing outputs cannot tell you which one ran. These assert the MECHANISM:
# the extension entry is called, exactly once, for a layer inside the gate — and not at all
# for one outside it.

sys.path.insert(0, os.path.dirname(__file__))


def _cpu_ext_or_skip():
    from glq import inference_kernel_cpu as ikc
    if not (ikc._try_load_cpu_ext()
            and hasattr(ikc._glq_cpu, "glq_fused_moe_trellis_3inst_cpu")):
        pytest.skip("CPU extension without the fused MoE entry")
    return ikc._glq_cpu


def _trellis_moe_layer(w, activation="gelu_pytorch_tanh"):
    """A layer carrying exactly what ``apply()`` reads, laid out as the loader leaves it:
    stacked per-expert buffers, a shared SV, and the block metadata `_process_trellis`
    computes. Built from the same synthetic weights the kernel suite uses."""
    from glq.hadamard import _block_decompose as _bd

    layer = torch.nn.Module()
    layer.glq_is_trellis, layer.glq_is_e8p = True, False
    layer.glq_hidden_size = w.hidden
    layer.glq_intermediate_size = w.inter
    layer.glq_w13_out = w.w13_out
    layer.activation = activation
    layer.glq_trellis_fused_ok = True
    sentinel = torch.empty(w.E, 1, 1, dtype=torch.int16)      # "no stage 2", as the loader
    for pfx, packed, su, sv, ws, n_pad, m_pad, mn, mm in (
            ("w13", w.w13_packed, w.w13_SU, w.w13_SV, w.w13_Wscale, w.hidden, w.w13_out,
             w.meta_n_w13, w.meta_m_w13),
            ("w2", w.w2_packed, w.w2_SU, w.w2_SV, w.w2_Wscale, w.inter, w.hidden,
             w.meta_n_w2, w.meta_m_w2)):
        setattr(layer, f"{pfx}_trellis_packed", packed)
        setattr(layer, f"{pfx}_trellis_packed2", sentinel)
        setattr(layer, f"{pfx}_SU", su)
        setattr(layer, f"{pfx}_SV", sv)
        setattr(layer, f"{pfx}_Wscale", ws)
        setattr(layer, f"{pfx}_inv_resid_scale2", torch.zeros(w.E))
        layer.__dict__.setdefault("_glq_trellis_moe_meta", {})[pfx] = {
            'has_s2': False, 'n_pad': n_pad, 'm_pad': m_pad,
            '_bn': torch.tensor(_bd(n_pad), dtype=torch.int64),
            '_bm': torch.tensor(_bd(m_pad), dtype=torch.int64),
            '_bnm': mn, '_bmm': mm,
        }
    return layer


@pytest.fixture
def moe_case():
    from test_moe_cpu_kernel import MoEWeights, _route
    from glq_vllm.fused_moe_method import GLQFusedMoEMethod

    _cpu_ext_or_skip()
    torch.manual_seed(0)
    w = MoEWeights(E=4)
    ids, wts = _route(3, w.E, 2)
    method = GLQFusedMoEMethod.__new__(GLQFusedMoEMethod)     # apply() needs no __init__ state
    return method, _trellis_moe_layer(w), torch.randn(3, w.hidden), ids, wts


def _counting(monkeypatch):
    """Wrap the extension entry so a call is observable. Returns the call list."""
    ext = _cpu_ext_or_skip()
    calls, orig = [], ext.glq_fused_moe_trellis_3inst_cpu

    def counted(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(ext, "glq_fused_moe_trellis_3inst_cpu", counted)
    return calls


def test_a_trellis_moe_forward_calls_the_fused_cpu_op(monkeypatch, moe_case):
    method, layer, x, ids, wts = moe_case
    calls = _counting(monkeypatch)
    out = method.apply(layer, x, wts, ids)
    assert len(calls) == 1, "the fused CPU MoE op must serve the block, once"
    assert out.shape == x.shape and out.dtype == x.dtype


def test_the_output_dtype_follows_the_activations(moe_case):
    """vLLM's CPU backend runs bf16 activations; the op works in fp32 internally."""
    method, layer, x, ids, wts = moe_case
    out = method.apply(layer, x.bfloat16(), wts, ids)
    assert out.dtype == torch.bfloat16


def test_a_layer_outside_the_gate_falls_back_without_calling_the_op(monkeypatch, moe_case):
    method, layer, x, ids, wts = moe_case
    layer.glq_trellis_fused_ok = False            # e.g. m_pad % 32 != 0
    calls = _counting(monkeypatch)
    with pytest.warns(RuntimeWarning, match="per-expert loop"):
        out = method.apply(layer, x, wts, ids)
    assert not calls, "a refused layer must not reach the fused op"
    assert out.shape == x.shape


def test_fused_and_the_per_expert_loop_agree(monkeypatch, moe_case):
    """The A/B that makes GLQ_MOE_FORCE_FALLBACK a usable isolation switch: same layer, both
    paths. Not bit-exact by construction — the fused op is fp32 end to end while the loop
    runs fp16 intermediates — so this is a closeness check. Bit-exactness against a torch
    oracle is pinned in tests/test_moe_cpu_kernel.py."""
    method, layer, x, ids, wts = moe_case
    fused = method.apply(layer, x, wts, ids)

    monkeypatch.setenv("GLQ_MOE_FORCE_FALLBACK", "1")
    calls = _counting(monkeypatch)
    loop = method.apply(layer, x, wts, ids)
    assert not calls, "GLQ_MOE_FORCE_FALLBACK must reach the loop, not the fused op"

    rel = (fused - loop).norm() / loop.norm().clamp_min(1e-12)
    assert rel < 1e-2, f"fused and loop disagree by {rel:.3e} — more than fp16 rounding"


def test_fp32_is_a_supported_act_dtype():
    from glq_vllm.config import GLQvLLMConfig
    cfg = GLQvLLMConfig(bpw=4, codebook="trellis", variant="3inst", trellis_layout="kernel")
    assert torch.float32 in cfg.get_supported_act_dtypes()
