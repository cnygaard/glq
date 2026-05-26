"""Phase 5.2a acceptance — ``unified_attention_e8_v3_0`` correctness
on the Phase 2.0/2.1 fixture matrix with a 4 K-entry codebook.

Branch B of v0.5 Phase 5.0a: weights stay on 65 K strict codebook,
KV/attention side uses a smaller (4 K) codebook for L1/L2 cache
residency on Blackwell SM_120 (128 KB L1/Shared pool per SM).

Validates that v3.0 with 4 K codebook produces output within
``rtol=5e-3, atol=5e-3`` of the bf16 dequant-then-attention reference
(``reference_e8_attention`` from ``tests/test_e8_attention_reference``).
Same correctness gate as Phase 2.0/2.1 — only the codebook size
differs.

Phase 5.2a kernel body is byte-identical to v2.1; this test exists
to lock in the contract that the kernel works with a 4 K codebook,
not just the production 65 K one.
"""
from __future__ import annotations

import math
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton kernels require CUDA",
)


@pytest.fixture(scope="module")
def quantizer_4k_relaxed():
    """E8KVQuantizer backed by the **4 K relaxed** codebook — matches
    Phase 5.2 Branch B's KV side."""
    from glq.codebook_relaxed import E8RelaxedCodebook
    from glq.kv_e8 import E8KVQuantizer
    cb = E8RelaxedCodebook(verbose=False, target_size=4096)
    for name in ("codebook", "codebook_norms", "codebook_half",
                 "codebook_half_t", "codebook_norms_half"):
        setattr(cb, name, getattr(cb, name).to("cuda"))
    # codebook_packed exists on strict but not always relaxed; move if present
    if getattr(cb, "codebook_packed", None) is not None:
        cb.codebook_packed = cb.codebook_packed.to("cuda")
    cb.device = "cuda"
    return E8KVQuantizer(cb, n_stages=2, secondary_stages=0)


@pytest.fixture(scope="module")
def hadamard_8():
    from glq_vllm.e8_paged_cache import _get_hadamard
    return _get_hadamard(torch.float32, "cuda")


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
        _pack_int16(qk["idx1"]), _pack_int16(qk["idx2"]),
        _pack_scale(qk["scale"]),
        _pack_int16(qv["idx1"]), _pack_int16(qv["idx2"]),
        _pack_scale(qv["scale"]),
        qk, qv,
    )


def _build(Tq, Tk, head_size, num_q_heads, num_kv_heads, seed):
    device = "cuda"
    dtype = torch.float16
    g = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randn(Tq, num_q_heads, head_size, device=device, dtype=dtype, generator=g),
        torch.randn(Tk, num_kv_heads, head_size, device=device, dtype=dtype, generator=g),
        torch.randn(Tk, num_kv_heads, head_size, device=device, dtype=dtype, generator=g),
    )


def _run_v3_0(q, k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
              codebook, H_mat, resid_scale, *, Tk, softmax_scale,
              sliding_window):
    from glq_vllm.triton_unified_attention_e8 import unified_attention_e8_v3_0
    Tq = q.shape[0]
    num_blocks = k_idx1.shape[0]
    out = torch.zeros_like(q)
    cu_seqlens_q = torch.tensor([0, Tq], device=q.device, dtype=torch.int32)
    seqused_k = torch.tensor([Tk], device=q.device, dtype=torch.int32)
    block_table = torch.arange(
        num_blocks, device=q.device, dtype=torch.int32
    ).reshape(1, num_blocks)
    unified_attention_e8_v3_0(
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
        sliding_window=sliding_window,
    )
    return out


def test_v3_0_rejects_non_4k_codebook(quantizer_4k_relaxed, hadamard_8):
    """v3.0 launcher must reject a 65 K codebook with a clear error;
    callers should route to v2.1 instead."""
    from glq.codebook import E8ShellCodebook
    big_cb = E8ShellCodebook(device="cuda", verbose=False).codebook_half
    head_size = 128
    Tq, Tk = 16, 16
    q, K, V = _build(Tq, Tk, head_size, num_q_heads=1, num_kv_heads=1, seed=1)
    block_size = 16
    num_blocks = (Tk + block_size - 1) // block_size
    (k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
     _qk, _qv) = _quantize_to_paged(
        K, V, quantizer_4k_relaxed, num_blocks=num_blocks, block_size=block_size,
    )
    rs = float(quantizer_4k_relaxed.codebook.resid_scale)
    with pytest.raises(AssertionError, match="v3.0 launcher requires a 4 K codebook"):
        _run_v3_0(
            q, k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
            big_cb, hadamard_8, rs, Tk=Tk,
            softmax_scale=1.0 / math.sqrt(head_size), sliding_window=0,
        )


@pytest.mark.parametrize("head_size", [128, 256, 512])
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(1, 1), (4, 1), (8, 4)])
@pytest.mark.parametrize("Tk", [32, 64])
def test_v3_0_matches_reference_no_sliding(
        quantizer_4k_relaxed, hadamard_8,
        head_size, num_q_heads, num_kv_heads, Tk):
    """4 K-codebook attention via v3.0 must match reference within
    ``rtol=5e-3, atol=5e-3`` across the Phase 2.0 shape sweep."""
    block_size = 16
    Tq = 16
    seed = head_size * 100 + num_q_heads * 10 + Tk
    q, K, V = _build(Tq, Tk, head_size, num_q_heads, num_kv_heads, seed)
    num_blocks = (Tk + block_size - 1) // block_size

    (k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
     qk, qv) = _quantize_to_paged(
        K, V, quantizer_4k_relaxed, num_blocks=num_blocks, block_size=block_size,
    )
    codebook = quantizer_4k_relaxed.codebook.codebook_half
    rs = float(quantizer_4k_relaxed.codebook.resid_scale)
    softmax_scale = 1.0 / math.sqrt(head_size)

    out_v3 = _run_v3_0(
        q, k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
        codebook, hadamard_8, rs, Tk=Tk, softmax_scale=softmax_scale,
        sliding_window=0,
    )

    from tests.test_e8_attention_reference import reference_e8_attention
    out_ref = reference_e8_attention(
        q=q,
        k_qt=qk, v_qt=qv,
        quantizer=quantizer_4k_relaxed,
        softmax_scale=softmax_scale,
        sliding_window=0,
    )
    max_abs = (out_v3 - out_ref).abs().max().item()
    rmse = ((out_v3 - out_ref) ** 2).mean().sqrt().item()
    print(f"  head={head_size} (q,kv)=({num_q_heads},{num_kv_heads}) Tk={Tk}: "
          f"max-abs={max_abs:.2e} rmse={rmse:.2e}")
    torch.testing.assert_close(out_v3, out_ref, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("sliding_window", [128, 256, 512])
@pytest.mark.parametrize("Tk", [64, 128, 256])
def test_v3_0_matches_reference_sliding(
        quantizer_4k_relaxed, hadamard_8, sliding_window, Tk):
    """4 K-codebook attention with sliding window must match reference."""
    head_size = 256
    block_size = 16
    Tq = 16
    seed = sliding_window * 1000 + Tk
    q, K, V = _build(Tq, Tk, head_size, num_q_heads=1, num_kv_heads=1, seed=seed)
    num_blocks = (Tk + block_size - 1) // block_size

    (k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
     qk, qv) = _quantize_to_paged(
        K, V, quantizer_4k_relaxed, num_blocks=num_blocks, block_size=block_size,
    )
    codebook = quantizer_4k_relaxed.codebook.codebook_half
    rs = float(quantizer_4k_relaxed.codebook.resid_scale)
    softmax_scale = 1.0 / math.sqrt(head_size)

    out_v3 = _run_v3_0(
        q, k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
        codebook, hadamard_8, rs, Tk=Tk, softmax_scale=softmax_scale,
        sliding_window=sliding_window,
    )

    from tests.test_e8_attention_reference import reference_e8_attention
    out_ref = reference_e8_attention(
        q=q,
        k_qt=qk, v_qt=qv,
        quantizer=quantizer_4k_relaxed,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
    )
    max_abs = (out_v3 - out_ref).abs().max().item()
    print(f"  SW={sliding_window} Tk={Tk}: max-abs={max_abs:.2e}")
    torch.testing.assert_close(out_v3, out_ref, rtol=5e-3, atol=5e-3)
