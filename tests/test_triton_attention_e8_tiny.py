"""Phase 1.5 acceptance gate: full fixture-matrix test (Phase 1 → 2).

Runs the v1.4 kernel — now the canonical Phase-1 attention kernel —
against ``reference_e8_attention`` over a grid of small shapes that
exercise:

- Tq=1 (single-query decode)
- Tq < Tk with context (decode with prior context)
- Tq == Tk (pure prefill)
- num_blocks ∈ {1, 2, 4}
- Single sequence, single KV head, head_size=128

This is the Phase 1 → Phase 2 acceptance gate. If any combination
fails, the kernel can't proceed to real Gemma-4 shapes (Phase 2)
because the structural correctness wouldn't transfer.
"""
from __future__ import annotations

import math
from itertools import product

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton kernels require CUDA",
)


def _quantize_to_paged(K, V, quantizer, *, num_blocks, block_size):
    Tk, H_kv, D = K.shape
    n_groups = D // 8

    qk = quantizer.quantize(K)
    qv = quantizer.quantize(V)

    def _pack_int16(qt_idx):
        flat = qt_idx.reshape(Tk, H_kv, n_groups).contiguous()
        paged = torch.zeros(
            num_blocks, block_size, H_kv, n_groups,
            dtype=torch.int16, device=K.device,
        )
        for t in range(Tk):
            paged[t // block_size, t % block_size] = flat[t]
        return paged

    def _pack_scale(scale_tensor):
        flat = scale_tensor.reshape(Tk, H_kv, n_groups).contiguous()
        paged = torch.zeros(
            num_blocks, block_size, H_kv, n_groups,
            dtype=flat.dtype, device=K.device,
        )
        for t in range(Tk):
            paged[t // block_size, t % block_size] = flat[t]
        return paged

    return (
        _pack_int16(qk["idx1"]),
        _pack_int16(qk["idx2"]),
        _pack_scale(qk["scale"]),
        _pack_int16(qv["idx1"]),
        _pack_int16(qv["idx2"]),
        _pack_scale(qv["scale"]),
        qk, qv,
    )


@pytest.fixture(scope="module")
def quantizer_4bpw():
    """E8 quantizer at 4 bpw (n_stages=2, n_secondary=0)."""
    from glq.codebook import E8ShellCodebook
    from glq.kv_e8 import E8KVQuantizer
    cb = E8ShellCodebook(verbose=False)
    for name in ("codebook", "codebook_norms", "codebook_half",
                 "codebook_half_t", "codebook_norms_half",
                 "codebook_packed"):
        setattr(cb, name, getattr(cb, name).to("cuda"))
    cb.device = "cuda"
    return E8KVQuantizer(cb, n_stages=2, secondary_stages=0)


@pytest.fixture(scope="module")
def hadamard_8():
    from glq_vllm.e8_paged_cache import _get_hadamard
    return _get_hadamard(torch.float32, "cuda")


def _build(Tq, Tk, head_size, seed):
    device = "cuda"
    dtype = torch.float16
    g = torch.Generator(device=device).manual_seed(seed)
    H = 1   # Phase 1: single KV head, no GQA
    return (
        torch.randn(Tq, H, head_size, device=device, dtype=dtype, generator=g),
        torch.randn(Tk, H, head_size, device=device, dtype=dtype, generator=g),
        torch.randn(Tk, H, head_size, device=device, dtype=dtype, generator=g),
    )


def _run_v1_4(q, k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
              codebook, H_mat, resid_scale, *, Tk, softmax_scale):
    from glq_vllm.triton_unified_attention_e8 import (
        unified_attention_e8_v1_4,
    )
    Tq = q.shape[0]
    num_blocks = k_idx1.shape[0]
    out = torch.zeros_like(q)
    cu_seqlens_q = torch.tensor([0, Tq], device=q.device, dtype=torch.int32)
    seqused_k = torch.tensor([Tk], device=q.device, dtype=torch.int32)
    block_table = torch.arange(
        num_blocks, device=q.device, dtype=torch.int32
    ).reshape(1, num_blocks)
    unified_attention_e8_v1_4(
        q=q,
        k_idx1=k_idx1, k_idx2=k_idx2, k_scale=k_scale,
        v_idx1=v_idx1, v_idx2=v_idx2, v_scale=v_scale,
        codebook=codebook, H_mat=H_mat,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        resid_scale=resid_scale,
        block_table=block_table,
    )
    return out


# Fixture matrix:
# - Tq ∈ {1, 4, 16, 32}
# - Tk ∈ {16, 32, 64} (block_size=16, num_blocks ∈ {1, 2, 4})
# - require Tq <= Tk (else causal mask + reference convention disagree)
_SHAPES = [
    (Tq, Tk) for Tq, Tk in product([1, 4, 16, 32], [16, 32, 64])
    if Tq <= Tk
]


@pytest.mark.parametrize("Tq,Tk", _SHAPES,
                         ids=[f"Tq{Tq}_Tk{Tk}" for Tq, Tk in _SHAPES])
def test_phase_1_5_full_matrix(quantizer_4bpw, hadamard_8, Tq, Tk):
    head_size = 128
    block_size = 16
    q, K, V = _build(Tq, Tk, head_size, seed=hash((Tq, Tk)) & 0xFFFF)
    num_blocks = (Tk + block_size - 1) // block_size

    (k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
     qk, qv) = _quantize_to_paged(
        K, V, quantizer_4bpw,
        num_blocks=num_blocks, block_size=block_size,
    )
    codebook = quantizer_4bpw.codebook.codebook_half.to(q.device)
    rs = float(quantizer_4bpw.codebook.resid_scale)
    softmax_scale = 1.0 / math.sqrt(head_size)

    from tests.test_e8_attention_reference import reference_e8_attention
    out_ref = reference_e8_attention(
        q, qk, qv, quantizer_4bpw,
        causal=True, softmax_scale=softmax_scale,
    )
    out_fk = _run_v1_4(
        q, k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
        codebook, hadamard_8, rs,
        Tk=Tk, softmax_scale=softmax_scale,
    )
    max_abs = (out_ref.float() - out_fk.float()).abs().max().item()
    rmse = (out_ref.float() - out_fk.float()).pow(2).mean().sqrt().item()
    print(f"  Tq={Tq:>2} Tk={Tk:>2} (blocks={num_blocks}): "
          f"max-abs={max_abs:.2e} RMSE={rmse:.2e}")
    assert max_abs < 5e-3, (
        f"Tq={Tq}, Tk={Tk}: max-abs={max_abs:.2e} >= 5e-3 "
        f"(RMSE={rmse:.2e})"
    )
