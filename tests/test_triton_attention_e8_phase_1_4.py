"""Phase 1.4 acceptance gate: full per-group dequant (Hadamard inverse).

Kernel ``kernel_unified_attention_2d_e8_v1_4`` completes the dequant
pipeline. Decode math:

    dec = (((codebook[idx1] + codebook[idx2] / RS) * scale[g]) @ H.T)

H is the 8×8 normalized Walsh-Hadamard matrix. Since H is symmetric
and orthogonal, ``H.T == H == H⁻¹``. The Hadamard is applied as a
broadcast + ``tl.sum`` instead of ``tl.dot`` because the 8-wide MMA
isn't available on most HW.

This is the first sub-phase whose dequant matches ``E8KVQuantizer.
dequantize`` exactly (modulo fp16/fp32 round-off), so we gate against
``reference_e8_attention`` directly.
"""
from __future__ import annotations

import math

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
def quantizer_stage2():
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


def _build_fixture(*, Tq, Tk, num_kv_heads, head_size, block_size, seed):
    device = "cuda"
    dtype = torch.float16
    g = torch.Generator(device=device).manual_seed(seed)
    num_q_heads = num_kv_heads
    q = torch.randn(Tq, num_q_heads, head_size,
                    device=device, dtype=dtype, generator=g)
    K = torch.randn(Tk, num_kv_heads, head_size,
                    device=device, dtype=dtype, generator=g)
    V = torch.randn(Tk, num_kv_heads, head_size,
                    device=device, dtype=dtype, generator=g)
    return q, K, V


def _run_forked_v1_4(q, k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
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


def _ref_using_reference_e8_attention(q, qk, qv, quantizer, *, softmax_scale):
    """Wrap ``reference_e8_attention`` from the Phase 0 test module so
    the Phase 1.4 kernel is gated against the same path we'll use to
    validate Phases 2-5."""
    from tests.test_e8_attention_reference import reference_e8_attention
    return reference_e8_attention(
        q, qk, qv, quantizer,
        causal=True, softmax_scale=softmax_scale,
    )


def _run_case(*, Tq, Tk, head_size, block_size, seed, quantizer, H_mat):
    q, K, V = _build_fixture(
        Tq=Tq, Tk=Tk, num_kv_heads=1, head_size=head_size,
        block_size=block_size, seed=seed,
    )
    num_blocks = (Tk + block_size - 1) // block_size

    (k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
     qk, qv) = _quantize_to_paged(
        K, V, quantizer, num_blocks=num_blocks, block_size=block_size,
    )
    codebook = quantizer.codebook.codebook_half.to(q.device)
    rs = float(quantizer.codebook.resid_scale)
    softmax_scale = 1.0 / math.sqrt(head_size)

    out_ref = _ref_using_reference_e8_attention(
        q, qk, qv, quantizer, softmax_scale=softmax_scale,
    )
    out_fk = _run_forked_v1_4(
        q, k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
        codebook, H_mat, rs,
        Tk=Tk, softmax_scale=softmax_scale,
    )
    max_abs = (out_ref.float() - out_fk.float()).abs().max().item()
    rmse = (out_ref.float() - out_fk.float()).pow(2).mean().sqrt().item()
    return out_ref, out_fk, max_abs, rmse


def test_phase_1_4_decode_tiny(quantizer_stage2, hadamard_8):
    _, _, max_abs, rmse = _run_case(
        Tq=1, Tk=16, head_size=128, block_size=16, seed=1,
        quantizer=quantizer_stage2, H_mat=hadamard_8,
    )
    print(f"  Tq=1 Tk=16: max-abs={max_abs:.2e}, RMSE={rmse:.2e}")
    assert max_abs < 5e-3, f"max-abs={max_abs:.2e}"


def test_phase_1_4_prefill(quantizer_stage2, hadamard_8):
    _, _, max_abs, rmse = _run_case(
        Tq=16, Tk=16, head_size=128, block_size=16, seed=42,
        quantizer=quantizer_stage2, H_mat=hadamard_8,
    )
    print(f"  Tq=Tk=16: max-abs={max_abs:.2e}, RMSE={rmse:.2e}")
    assert max_abs < 5e-3, f"max-abs={max_abs:.2e}"


def test_phase_1_4_two_blocks(quantizer_stage2, hadamard_8):
    _, _, max_abs, rmse = _run_case(
        Tq=1, Tk=32, head_size=128, block_size=16, seed=7,
        quantizer=quantizer_stage2, H_mat=hadamard_8,
    )
    print(f"  Tq=1 Tk=32 (2 blocks): max-abs={max_abs:.2e}, RMSE={rmse:.2e}")
    assert max_abs < 5e-3, f"max-abs={max_abs:.2e}"
