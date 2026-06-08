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


if __name__ == "__main__":
    if not _HAS:
        print("glq_vllm not importable; run on the box")
    else:
        test_w13_gate_up_halves_reassemble()
        test_w2_down_is_full_width_copy()
        test_shared_sv_and_scalars()
        print("OK: gate+up loader reassembly lossless; w2 full copy; shared SV/scalars")
