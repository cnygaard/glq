"""Gate+up MoE fusion: loader reassembly is lossless.

Routed MoE experts are quantized JOINTLY as one ``[gate; up]`` matrix (one
Wscale/SU/SV) then stored split back under the ``gate_proj``/``up_proj`` keys
(Qidxs/Qidxs2/SU row-split; SV/Wscale/inv_resid_scale shared -> duplicated).
vLLM's native expert mapping feeds the loader shard "w1" (gate) / "w3" (up),
and ``_glq_weight_loader`` must place them into the two halves of ``w13`` so the
fused dequant kernel sees an exact reassembly. This test verifies that split +
loader-reassemble reproduces the original fused tensors bit-for-bit, and that
down_proj (shard "w2") is a full-width copy.
"""

import types

import pytest
import torch

try:
    from glq_vllm.linear_method import _glq_weight_loader
    _HAS = True
except (ImportError, ModuleNotFoundError):
    _HAS = False

requires_glq_vllm = pytest.mark.skipif(not _HAS, reason="glq_vllm/vllm not installed")


def _param(t):
    """Minimal stand-in for an nn.Parameter: the loader only touches `.data`."""
    return types.SimpleNamespace(data=t)


@requires_glq_vllm
def test_w13_gate_up_halves_reassemble():
    E, m_pad, nblk = 4, 2048, 512  # m_pad = 2*intermediate (gate+up), nblk = hidden/8
    half = m_pad // 2
    torch.manual_seed(0)

    # A "jointly quantized" w13 for each expert (what the kernel must end up with).
    fused_Qidxs = torch.randint(-32768, 32767, (E, m_pad, nblk), dtype=torch.int16)
    fused_SU = torch.randn(E, m_pad, dtype=torch.float16)

    # Allocated w13 buffers (gated -> full 2*intermediate rows), as create_weights does.
    w13_Q = _param(torch.zeros(E, m_pad, nblk, dtype=torch.int16))
    w13_SU = _param(torch.zeros(E, m_pad, dtype=torch.float16))

    for e in range(E):
        # split (as _store_artifacts does), then load via the real loader
        gate_Q, up_Q = fused_Qidxs[e, :half].clone(), fused_Qidxs[e, half:].clone()
        gate_SU, up_SU = fused_SU[e, :half].clone(), fused_SU[e, half:].clone()
        _glq_weight_loader(w13_Q, gate_Q, "n", shard_id="w1", expert_id=e)
        _glq_weight_loader(w13_Q, up_Q, "n", shard_id="w3", expert_id=e)
        _glq_weight_loader(w13_SU, gate_SU, "n", shard_id="w1", expert_id=e)
        _glq_weight_loader(w13_SU, up_SU, "n", shard_id="w3", expert_id=e)

    assert torch.equal(w13_Q.data, fused_Qidxs), "w13 Qidxs reassembly is not lossless"
    assert torch.equal(w13_SU.data, fused_SU), "w13 SU reassembly is not lossless"


@requires_glq_vllm
def test_w2_down_is_full_width_copy():
    E, m_pad_w2, nblk_w2 = 4, 4096, 128  # down: hidden rows, intermediate/8 blocks
    torch.manual_seed(1)
    down = torch.randint(-32768, 32767, (E, m_pad_w2, nblk_w2), dtype=torch.int16)
    w2_Q = _param(torch.zeros(E, m_pad_w2, nblk_w2, dtype=torch.int16))
    for e in range(E):
        _glq_weight_loader(w2_Q, down[e].clone(), "n", shard_id="w2", expert_id=e)
    assert torch.equal(w2_Q.data, down), "w2 (down) full-width copy failed"


@requires_glq_vllm
def test_shared_sv_and_scalars():
    E, n_pad = 4, 4096
    torch.manual_seed(2)
    sv = torch.sign(torch.randn(n_pad)).to(torch.float16)   # deterministic ±1, shared
    wscale = torch.tensor(0.0088, dtype=torch.float32)
    w13_SV = _param(torch.zeros(n_pad, dtype=torch.float16))
    w13_Wscale = _param(torch.zeros(E, dtype=torch.float32))
    for e in range(E):
        # gate (w1) and up (w3) of each expert carry the SAME shared SV/Wscale
        _glq_weight_loader(w13_SV, sv.clone(), "n", shard_id="w1", expert_id=e)
        _glq_weight_loader(w13_SV, sv.clone(), "n", shard_id="w3", expert_id=e)
        _glq_weight_loader(w13_Wscale, wscale.clone(), "n", shard_id="w1", expert_id=e)
        _glq_weight_loader(w13_Wscale, wscale.clone(), "n", shard_id="w3", expert_id=e)
    assert torch.equal(w13_SV.data, sv), "shared SV not loaded"
    assert torch.allclose(w13_Wscale.data, torch.full((E,), 0.0088)), "per-expert Wscale not loaded"


# --------------------------------------------------------------------------- #
# Gemma-4 MoE: the routed experts are stored STACKED + pre-fused as
# experts.gate_up_proj [E, 2*inter, hidden]. The quantizer slices each expert,
# quantizes the (already fused) gate_up jointly, then ROW-SPLITS the artifacts
# into per-expert gate_proj/up_proj exactly like sarvam — so the same loader
# reassembly applies. These pin both halves of that contract.
# --------------------------------------------------------------------------- #

# Must match _ROW_ARTS / _SHARED_ARTS in glq/quantize_model.py:_store_artifacts.
_ROW_ARTS = {"Qidxs", "Qidxs2", "Qidxs3", "SU"}
_SHARED_ARTS = {"SV", "Wscale", "inv_resid_scale", "inv_resid_scale2"}


def _split_gate_up(arts: dict, inter: int):
    """Mirror of _store_artifacts' moe_gate_up branch: row-split row-indexed
    artifacts into gate (rows [0:inter]) + up (rows [inter:2*inter]); duplicate
    the shared input-side artifacts."""
    gate, up = {}, {}
    for k, v in arts.items():
        if k in _ROW_ARTS:
            gate[k] = v[:inter].clone()
            up[k] = v[inter:2 * inter].clone()
        elif k in _SHARED_ARTS:
            gate[k] = v.clone()
            up[k] = v.clone()
        else:
            raise AssertionError(f"unhandled artifact {k!r}")
    return gate, up


def test_gemma4_expert_gate_up_split_roundtrip():
    """Pure-logic (no GPU/glq_vllm): row-splitting a jointly-quantized gate_up
    and re-concatenating reproduces the original — and shared artifacts dup."""
    inter, hidden_blk = 704, 512  # gemma-4-26B-A4B: expert inter 704, hidden 2816->/8 pad
    m_pad = 2 * inter             # 1408 fused gate||up rows
    torch.manual_seed(3)
    # 3-stage (5/6 bpw) artifacts: Qidxs3 is row-indexed (like Qidxs/Qidxs2),
    # inv_resid_scale2 is shared (like inv_resid_scale). Both must survive the split.
    arts = {
        "Qidxs": torch.randint(-32768, 32767, (m_pad, hidden_blk), dtype=torch.int16),
        "Qidxs2": torch.randint(-128, 127, (m_pad, hidden_blk), dtype=torch.int16),
        "Qidxs3": torch.randint(-32768, 32767, (m_pad, hidden_blk), dtype=torch.int16),
        "SU": torch.randn(m_pad, dtype=torch.float16),
        "SV": torch.sign(torch.randn(4096)).to(torch.float16),
        "Wscale": torch.tensor(0.0091, dtype=torch.float32),
        "inv_resid_scale": torch.tensor(7.5, dtype=torch.float32),
        "inv_resid_scale2": torch.tensor(31.0, dtype=torch.float32),
    }
    gate, up = _split_gate_up(arts, inter)
    for k in _ROW_ARTS & set(arts):
        recon = torch.cat([gate[k], up[k]], dim=0)
        assert torch.equal(recon, arts[k]), f"{k} row-split not lossless"
        assert gate[k].shape[0] == inter and up[k].shape[0] == inter
    for k in _SHARED_ARTS & set(arts):
        assert torch.equal(gate[k], arts[k]) and torch.equal(up[k], arts[k])


@requires_glq_vllm
def test_gemma4_expert_split_then_loader_reassemble():
    """End-to-end on real gemma-4-26B-A4B expert dims: store-split then the vLLM
    loader reassembles an exact w13; down is a full-width per-expert copy."""
    E, inter, hidden_blk, inter_blk = 4, 704, 512, 128
    m_pad_w13, m_pad_w2 = 2 * inter, 2816  # gate||up rows; down hidden rows
    torch.manual_seed(4)
    fused_Q = torch.randint(-32768, 32767, (E, m_pad_w13, hidden_blk), dtype=torch.int16)
    fused_SU = torch.randn(E, m_pad_w13, dtype=torch.float16)
    down_Q = torch.randint(-32768, 32767, (E, m_pad_w2, inter_blk), dtype=torch.int16)
    w13_Q = _param(torch.zeros(E, m_pad_w13, hidden_blk, dtype=torch.int16))
    w13_SU = _param(torch.zeros(E, m_pad_w13, dtype=torch.float16))
    w2_Q = _param(torch.zeros(E, m_pad_w2, inter_blk, dtype=torch.int16))
    for e in range(E):
        g, u = _split_gate_up({"Qidxs": fused_Q[e], "SU": fused_SU[e]}, inter)
        _glq_weight_loader(w13_Q, g["Qidxs"], "n", shard_id="w1", expert_id=e)
        _glq_weight_loader(w13_Q, u["Qidxs"], "n", shard_id="w3", expert_id=e)
        _glq_weight_loader(w13_SU, g["SU"], "n", shard_id="w1", expert_id=e)
        _glq_weight_loader(w13_SU, u["SU"], "n", shard_id="w3", expert_id=e)
        _glq_weight_loader(w2_Q, down_Q[e].clone(), "n", shard_id="w2", expert_id=e)
    assert torch.equal(w13_Q.data, fused_Q)
    assert torch.equal(w13_SU.data, fused_SU)
    assert torch.equal(w2_Q.data, down_Q)


# --------------------------------------------------------------------------- #
# Stage-3 (5/6 bpw) MoE serving regression guard.
#
# create_weights registers a (E,1,1) Qidxs3 PLACEHOLDER for <=4 bpw checkpoints
# (no Qidxs3 key to load). The C++ MoE entries gate stage 3 on the NUMEL check
# `Qidxs3.numel()>0 && inv_resid_scale2.numel()>0` (glq_cuda.cu ~3481/4332/4533),
# so a (E,1,1) placeholder (numel E>0) would WRONGLY enable stage 3 -> the kernel
# reads qidxs3[m_row*N_BLOCKS+j] out of bounds of the 1-elem-per-expert buffer
# (illegal memory access). process_weights_after_loading collapses any sentinel
# Qidxs3 (`dim==3 and shape[1]<=1`) to a numel-0 tensor; a real stage-3 buffer
# (shape[1]=m_pad>1) is left intact. These pin both halves of that contract.
# --------------------------------------------------------------------------- #

def _is_sentinel(q):
    """Mirror of the collapse predicate in fused_moe_method.process_weights."""
    return q.dim() == 3 and q.shape[1] <= 1


def _collapse_if_sentinel(q):
    return torch.empty(0, dtype=torch.int16) if _is_sentinel(q) else q


def _cpp_has_s3(qidxs3, inv_rs2):
    """Mirror of the C++ has_s3 gate: numel-based, both must be non-empty."""
    return qidxs3.numel() > 0 and inv_rs2.numel() > 0


def test_stage3_sentinel_collapse_disables_has_s3():
    """<=4 bpw: a (E,1,1) Qidxs3 placeholder must collapse to numel-0 so stage 3
    stays OFF (else the kernel reads OOB). inv_resid_scale2 stays (E,) zeros."""
    E = 4
    sentinel_q3 = torch.zeros(E, 1, 1, dtype=torch.int16)
    inv_rs2 = torch.zeros(E, dtype=torch.float32)  # (E,) zeros, kept for the fallback list
    assert _is_sentinel(sentinel_q3)
    collapsed = _collapse_if_sentinel(sentinel_q3)
    assert collapsed.numel() == 0
    assert not _cpp_has_s3(collapsed, inv_rs2), "stage 3 wrongly enabled for 2-stage model"


def test_stage3_fullshape_qidxs3_enables_has_s3():
    """5/6 bpw: a real (E, m_pad, n_blocks) Qidxs3 must NOT be collapsed and must
    enable stage 3 (with a loaded non-empty inv_resid_scale2)."""
    E, m_pad, nblk = 4, 1408, 512  # gemma-4-26B-A4B fused gate||up rows
    real_q3 = torch.zeros(E, m_pad, nblk, dtype=torch.int16)
    inv_rs2 = torch.full((E,), 31.0, dtype=torch.float32)
    assert not _is_sentinel(real_q3)
    kept = _collapse_if_sentinel(real_q3)
    assert kept.shape == (E, m_pad, nblk)
    assert _cpp_has_s3(kept, inv_rs2), "stage 3 should be enabled for a 3-stage model"


if __name__ == "__main__":
    test_gemma4_expert_gate_up_split_roundtrip()
    test_stage3_sentinel_collapse_disables_has_s3()
    test_stage3_fullshape_qidxs3_enables_has_s3()
    if not _HAS:
        print("glq_vllm not importable; loader tests run on the box")
    else:
        test_w13_gate_up_halves_reassemble()
        test_w2_down_is_full_width_copy()
        test_shared_sv_and_scalars()
        test_gemma4_expert_split_then_loader_reassemble()
        print("OK: gate+up loader reassembly lossless; w2 full copy; shared SV/scalars; gemma-4 experts")
    print("OK: stage-3 sentinel collapse / full-shape has_s3 gate")
