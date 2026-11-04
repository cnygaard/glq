"""The GLQ vLLM MoE applies GLM-5.3's SwiGLU clamp on every path that can serve it.

vLLM passes ``swiglu_limit`` to FusedMoE and keeps the activation named ``silu``, so the
activation id alone cannot see the clamp. The decision helpers are pinned on CPU in
tests/test_moe_swiglu_limit.py; this file covers what needs vLLM, and on a GPU the fused op:

* the per-expert loop's activation against vLLM's own ``SiluAndMulWithClamp``;
* the load-time refusal on codebooks that have no clamp;
* the new op's schema against its fake, without the extension;
* on CUDA, the fused op against the loop, with the clamp shown to engage and the loop
  shown not to run — equal outputs alone would pass if apply() quietly took the loop.

Run in its own process on the box, like tests/test_glq_vllm.py.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("vllm.model_executor.layers.fused_moe",
                    reason="no installed vllm (wheel needed)")

import torch.nn.functional as F  # noqa: E402

from glq_vllm import fused_moe_method as fmm  # noqa: E402

LIMIT = 10.0


def _wide(rows=64, d=32, dtype=torch.float32):
    """Gate and up spread over ±40, so both clamps engage."""
    g = torch.Generator().manual_seed(0)
    return (torch.rand(rows, 2 * d, generator=g) * 80 - 40).to(dtype)


# ---- the per-expert loop's activation ----------------------------------------------------

def test_the_loop_activation_matches_vllms_clamp():
    """Oracle: vLLM's own clamped SwiGLU. Its native form writes silu as
    ``g * sigmoid(g)``, so this is a closeness check; the clamp itself is exact."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    h = _wide()
    ours = fmm._apply_activation(h, "silu", swiglu_limit=LIMIT)
    with set_current_vllm_config(VllmConfig()):     # a CustomOp reads it when built
        op = SiluAndMulWithClamp(swiglu_limit=LIMIT, compile_native=False)
    ref = op.forward_native(h)
    torch.testing.assert_close(ours, ref, rtol=1e-6, atol=1e-6)


def test_the_loop_activation_is_exactly_vllms_torch_reference():
    """``_swiglu_limit_torch`` (vLLM's MoE reference) is ``F.silu(clamped) * clamped``."""
    h = _wide()
    gate, up = h.chunk(2, dim=-1)
    ref = F.silu(gate.clamp(max=LIMIT)) * up.clamp(min=-LIMIT, max=LIMIT)
    assert torch.equal(fmm._apply_activation(h, "silu", swiglu_limit=LIMIT), ref)


def test_no_limit_leaves_the_activation_unchanged():
    h = _wide()
    gate, up = h.chunk(2, dim=-1)
    assert torch.equal(fmm._apply_activation(h, "silu"), F.silu(gate) * up)
    assert torch.equal(fmm._apply_activation(h, "silu", swiglu_limit=0.0), F.silu(gate) * up)


# ---- load time ---------------------------------------------------------------------------

def _method(codebook, limit):
    m = fmm.GLQFusedMoEMethod.__new__(fmm.GLQFusedMoEMethod)
    m.quant_config = SimpleNamespace(codebook=codebook, variant="3inst", bpw=4)
    m.codebook_type = codebook
    m.moe = SimpleNamespace(swiglu_limit=limit, is_act_and_mul=True, activation="silu")
    return m


def _create(method, layer):
    method.create_weights(layer, num_experts=2, hidden_size=64,
                          intermediate_size_per_partition=64, params_dtype=torch.float16,
                          weight_loader=None)


@pytest.mark.parametrize("codebook", ["e8p", "e8_shell"])
def test_a_codebook_without_a_clamp_refuses_at_load(codebook):
    with pytest.raises(ValueError, match="swiglu_limit"):
        _create(_method(codebook, LIMIT), torch.nn.Module())


def test_a_trellis_layer_records_its_limit():
    layer = torch.nn.Module()
    _create(_method("trellis", LIMIT), layer)
    assert layer.glq_swiglu_limit == LIMIT


def test_a_trellis_layer_without_a_limit_records_zero():
    layer = torch.nn.Module()
    _create(_method("trellis", None), layer)
    assert layer.glq_swiglu_limit == 0.0


# ---- the op's registration ---------------------------------------------------------------

def test_the_swiglu_op_schema_matches_its_fake():
    """Same check as test_fused_moe_trellis_3inst_schema_matches_fake, for the clamped op:
    names, not only the count, because a reorder binds weights to the wrong parameter."""
    import inspect

    from glq_vllm.custom_ops import (FUSED_MOE_TRELLIS_3INST_SCHEMA,
                                     FUSED_MOE_TRELLIS_3INST_SWIGLU_SCHEMA,
                                     _fused_moe_trellis_3inst_swiglu_fake)

    def names(schema):
        args = schema.split("(", 1)[1].rsplit(")", 1)[0]
        return [a.strip().split()[-1] for a in args.split(",")]

    swiglu = names(FUSED_MOE_TRELLIS_3INST_SWIGLU_SCHEMA)
    assert swiglu == list(inspect.signature(_fused_moe_trellis_3inst_swiglu_fake).parameters)
    # The unclamped op's arguments, in order, then the limit.
    assert swiglu == names(FUSED_MOE_TRELLIS_3INST_SCHEMA) + ["swiglu_limit"]
    assert FUSED_MOE_TRELLIS_3INST_SWIGLU_SCHEMA.startswith("fused_moe_trellis_3inst_swiglu(")


# ---- CUDA: the fused op ------------------------------------------------------------------

HIDDEN, INTER = 128, 64
W13_OUT = 2 * INTER
N_EXPERTS, BPW = 8, 4


def _cuda_ops_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from glq import inference_kernel as ik
    from glq_vllm.custom_ops import _ensure_registered
    if not ik._try_load_cuda_ext():
        pytest.skip("glq CUDA extension not built")
    _ensure_registered()
    for op in ("fused_moe_trellis_3inst", "fused_moe_trellis_3inst_swiglu"):
        assert hasattr(torch.ops.glq, op), f"torch.ops.glq.{op} not registered"


def _cuda_layer(limit, seed=0):
    """A trellis MoE layer as apply() reads it, with random packed bytes.

    The decode need not be meaningful: both arms decode the same bytes. Shapes and metadata
    are real, built the way `_process_trellis` builds them.
    """
    from glq.hadamard import _block_decompose as _bd
    from glq.quantized_linear import _pack_block_meta as _pbm
    from glq_vllm.moe_shapes import trellis_moe_shapes

    g = torch.Generator().manual_seed(seed)
    layer = torch.nn.Module()
    for name, shape in trellis_moe_shapes(N_EXPERTS, HIDDEN, INTER, W13_OUT, BPW).items():
        if "trellis_packed" in name:
            t = torch.randint(-32768, 32767, shape, dtype=torch.int16, generator=g)
        elif name.endswith(("SU", "SV")):
            t = torch.randint(0, 2, shape, generator=g).to(torch.float16) * 2 - 1
        elif name.endswith("Wscale"):
            t = torch.rand(shape, generator=g) * 0.02 + 0.01
        else:                                           # inv_resid_scale2: 0 at 4 bpw
            t = torch.zeros(shape, dtype=torch.float32)
        setattr(layer, name, t.cuda())
    meta = {}
    for pfx, n_pad, m_pad in (("w13", HIDDEN, W13_OUT), ("w2", INTER, HIDDEN)):
        bn, bm = _bd(n_pad), _bd(m_pad)
        meta[pfx] = {'has_s2': False, 'n_pad': n_pad, 'm_pad': m_pad,
                     '_bn': torch.tensor(bn, dtype=torch.int64),
                     '_bm': torch.tensor(bm, dtype=torch.int64),
                     '_bnm': _pbm(bn).cuda(), '_bmm': _pbm(bm).cuda()}
    layer.__dict__["_glq_trellis_moe_meta"] = meta
    layer.glq_is_trellis, layer.glq_is_e8p = True, False
    layer.glq_trellis_fused_ok = True
    layer.glq_hidden_size, layer.glq_intermediate_size = HIDDEN, INTER
    layer.glq_w13_out = W13_OUT
    layer.activation = "silu"
    layer.glq_arena = None                              # no arena: not what is under test
    layer.glq_swiglu_limit = limit
    return layer


def _inputs(T=12, top_k=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, HIDDEN, generator=g).to(torch.bfloat16).cuda()
    ids = torch.stack([torch.randperm(N_EXPERTS, generator=g)[:top_k] for _ in range(T)])
    w = torch.rand(T, top_k, generator=g)
    return x, ids.cuda(), (w / w.sum(dim=1, keepdim=True)).cuda()


def _method_on_cuda():
    m = fmm.GLQFusedMoEMethod.__new__(fmm.GLQFusedMoEMethod)
    m.prefix = "test.mlp.experts"
    return m


def _median_gate_up(method, layer, x, ids, w, monkeypatch):
    """|w13 output| median, read from the loop's own activation input, so the limit sits
    inside the data and both sides of the clamp are exercised."""
    seen = []
    orig = fmm._apply_activation

    def spy(h, activation, swiglu_limit=0.0):
        seen.append(h.detach().float().abs().flatten())
        return orig(h, activation, swiglu_limit=swiglu_limit)

    monkeypatch.setattr(fmm, "_apply_activation", spy)
    monkeypatch.setenv("GLQ_MOE_FORCE_FALLBACK", "1")
    method.apply(layer, x, w, ids)
    monkeypatch.undo()
    return float(torch.cat(seen).median())


def _fused_only(method, layer, x, ids, w, monkeypatch):
    """apply() with the loop made to fail, so the result can only come from the fused op."""
    def no_loop(*a, **k):
        raise AssertionError("apply() took the per-expert loop; the fused op must serve")
    monkeypatch.setattr(method, "_apply_trellis", no_loop)
    monkeypatch.setenv("GLQ_MOE_FORCE_FALLBACK", "0")
    out = method.apply(layer, x, w, ids)
    monkeypatch.undo()
    return out


def _loop_only(method, layer, x, ids, w, monkeypatch):
    monkeypatch.setenv("GLQ_MOE_FORCE_FALLBACK", "1")
    out = method.apply(layer, x, w, ids)
    monkeypatch.undo()
    return out


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def test_the_fused_op_applies_the_clamp(monkeypatch):
    _cuda_ops_or_skip()
    method, (x, ids, w) = _method_on_cuda(), _inputs()
    limit = _median_gate_up(method, _cuda_layer(0.0), x, ids, w, monkeypatch)
    assert limit > 0

    fused = _fused_only(method, _cuda_layer(limit), x, ids, w, monkeypatch)
    loop = _loop_only(method, _cuda_layer(limit), x, ids, w, monkeypatch)
    unclamped = _fused_only(method, _cuda_layer(0.0), x, ids, w, monkeypatch)

    assert torch.isfinite(fused).all()
    # Both run fp16 intermediates on different schedules: a closeness check, as in
    # test_fused_and_the_per_expert_loop_agree.
    assert _rel(fused, loop) < 1e-2, f"fused vs loop {_rel(fused, loop):.3e}"
    # The clamp engaged: half the values sit beyond the limit.
    assert _rel(fused, unclamped) > 5e-2, f"clamp had no effect ({_rel(fused, unclamped):.3e})"


def test_a_zero_limit_is_the_unclamped_op_bit_for_bit():
    """The kernel now takes a limit. At 0 its branch is skipped, so the clamped op must equal
    the unclamped one exactly — which is what keeps the e8p and shell callers unchanged."""
    _cuda_ops_or_skip()
    layer, (x, ids, w) = _cuda_layer(0.0), _inputs()
    m13, m2 = layer._glq_trellis_moe_meta["w13"], layer._glq_trellis_moe_meta["w2"]
    args = (x.half().contiguous(), ids, w.float().contiguous(),
            layer.w13_trellis_packed, layer.w13_trellis_packed2, layer.w13_SU,
            layer.w13_SV, layer.w13_Wscale, layer.w13_inv_resid_scale2,
            layer.w2_trellis_packed, layer.w2_trellis_packed2, layer.w2_SU,
            layer.w2_SV, layer.w2_Wscale, layer.w2_inv_resid_scale2,
            HIDDEN, INTER, W13_OUT,
            m13['n_pad'], m13['m_pad'], m2['n_pad'], m2['m_pad'],
            m13['_bn'], m13['_bm'], m13['_bnm'], m13['_bmm'],
            m2['_bn'], m2['_bm'], m2['_bnm'], m2['_bmm'], 0)
    plain = torch.ops.glq.fused_moe_trellis_3inst(*args)
    clamped_off = torch.ops.glq.fused_moe_trellis_3inst_swiglu(*args, 0.0)
    assert torch.equal(plain, clamped_off)


def test_the_clamped_op_refuses_a_non_silu_activation():
    _cuda_ops_or_skip()
    layer, (x, ids, w) = _cuda_layer(LIMIT), _inputs()
    m13, m2 = layer._glq_trellis_moe_meta["w13"], layer._glq_trellis_moe_meta["w2"]
    with pytest.raises(RuntimeError, match="silu"):
        torch.ops.glq.fused_moe_trellis_3inst_swiglu(
            x.half().contiguous(), ids, w.float().contiguous(),
            layer.w13_trellis_packed, layer.w13_trellis_packed2, layer.w13_SU,
            layer.w13_SV, layer.w13_Wscale, layer.w13_inv_resid_scale2,
            layer.w2_trellis_packed, layer.w2_trellis_packed2, layer.w2_SU,
            layer.w2_SV, layer.w2_Wscale, layer.w2_inv_resid_scale2,
            HIDDEN, INTER, W13_OUT,
            m13['n_pad'], m13['m_pad'], m2['n_pad'], m2['m_pad'],
            m13['_bn'], m13['_bm'], m13['_bnm'], m13['_bmm'],
            m2['_bn'], m2['_bm'], m2['_bnm'], m2['_bmm'], 1, LIMIT)
