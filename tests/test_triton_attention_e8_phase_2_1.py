"""Phase 2.1 acceptance gate: sliding-window mask.

Kernel ``kernel_unified_attention_2d_e8_v2_1`` extends v2.0 with a
``SLIDING_WINDOW: tl.constexpr`` path that mirrors upstream
``unified_attention``'s three sliding-window blocks (tile-pruning,
per-element seq_mask extension, V mask).

Gates:

1. ``sliding_window ∈ {128, 256, 512}`` × ``Tk ∈ {64, 128, 256}`` at
   head_size=256 (Gemma-4 sliding-layer shape): forked kernel matches
   ``reference_e8_attention(sliding_window=W)`` within
   ``rtol=5e-3, atol=5e-3``.

2. Regression: ``sliding_window=0`` invocation matches the v2.0
   kernel exactly — the SLIDING_WINDOW=0 branch must be no-op.
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


@pytest.fixture(scope="module")
def quantizer_4bpw():
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


def _run_v2_1(q, k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
              codebook, H_mat, resid_scale, *, Tk, softmax_scale,
              sliding_window):
    from glq_vllm.triton_unified_attention_e8 import (
        unified_attention_e8_v2_1,
    )
    Tq = q.shape[0]
    num_blocks = k_idx1.shape[0]
    out = torch.zeros_like(q)
    cu_seqlens_q = torch.tensor([0, Tq], device=q.device, dtype=torch.int32)
    seqused_k = torch.tensor([Tk], device=q.device, dtype=torch.int32)
    block_table = torch.arange(
        num_blocks, device=q.device, dtype=torch.int32
    ).reshape(1, num_blocks)
    unified_attention_e8_v2_1(
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


def _run_v2_0(q, k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
              codebook, H_mat, resid_scale, *, Tk, softmax_scale):
    from glq_vllm.triton_unified_attention_e8 import (
        unified_attention_e8_v2_0,
    )
    Tq = q.shape[0]
    num_blocks = k_idx1.shape[0]
    out = torch.zeros_like(q)
    cu_seqlens_q = torch.tensor([0, Tq], device=q.device, dtype=torch.int32)
    seqused_k = torch.tensor([Tk], device=q.device, dtype=torch.int32)
    block_table = torch.arange(
        num_blocks, device=q.device, dtype=torch.int32
    ).reshape(1, num_blocks)
    unified_attention_e8_v2_0(
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


# ── Test 1: sliding_window=0 regression — must match v2.0 exactly ──


def test_phase_2_1_sliding_zero_matches_v2_0(quantizer_4bpw, hadamard_8):
    """SLIDING_WINDOW=0 disables every sliding-window branch in the kernel.
    Output must match the v2.0 kernel bit-identically.
    """
    head_size = 256
    block_size = 16
    Tq, Tk = 16, 64
    q, K, V = _build(Tq, Tk, head_size, num_q_heads=1, num_kv_heads=1, seed=11)
    num_blocks = (Tk + block_size - 1) // block_size

    (k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
     _qk, _qv) = _quantize_to_paged(
        K, V, quantizer_4bpw, num_blocks=num_blocks, block_size=block_size,
    )
    codebook = quantizer_4bpw.codebook.codebook_half
    rs = float(quantizer_4bpw.codebook.resid_scale)
    softmax_scale = 1.0 / math.sqrt(head_size)

    out_v20 = _run_v2_0(
        q, k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
        codebook, hadamard_8, rs, Tk=Tk, softmax_scale=softmax_scale,
    )
    out_v21_zero = _run_v2_1(
        q, k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
        codebook, hadamard_8, rs, Tk=Tk, softmax_scale=softmax_scale,
        sliding_window=0,
    )
    max_abs = (out_v20 - out_v21_zero).abs().max().item()
    print(f"  v2.0 vs v2.1(SW=0): max-abs = {max_abs:.2e}")
    assert torch.equal(out_v20, out_v21_zero), (
        f"v2.1 with SW=0 must match v2.0 exactly; max-abs={max_abs}"
    )


# ── Test 2: sliding_window vs reference — correctness matrix ──

_SLIDING_PARAMS = [
    (sw, tk)
    for sw, tk in product([128, 256, 512], [64, 128, 256])
]


@pytest.mark.parametrize(
    "sliding_window,Tk",
    _SLIDING_PARAMS,
    ids=[f"SW{sw}_Tk{tk}" for sw, tk in _SLIDING_PARAMS],
)
def test_phase_2_1_sliding_matches_reference(
    quantizer_4bpw, hadamard_8, sliding_window, Tk,
):
    head_size = 256
    block_size = 16
    Tq = Tk   # prefill-shaped (matches reference causal convention)
    q, K, V = _build(
        Tq, Tk, head_size, num_q_heads=1, num_kv_heads=1,
        seed=hash((sliding_window, Tk)) & 0xFFFF,
    )
    num_blocks = (Tk + block_size - 1) // block_size

    (k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
     qk, qv) = _quantize_to_paged(
        K, V, quantizer_4bpw, num_blocks=num_blocks, block_size=block_size,
    )
    codebook = quantizer_4bpw.codebook.codebook_half
    rs = float(quantizer_4bpw.codebook.resid_scale)
    softmax_scale = 1.0 / math.sqrt(head_size)

    from tests.test_e8_attention_reference import reference_e8_attention
    out_ref = reference_e8_attention(
        q, qk, qv, quantizer_4bpw,
        causal=True, softmax_scale=softmax_scale,
        sliding_window=sliding_window,
    )
    out_fk = _run_v2_1(
        q, k_idx1, k_idx2, k_scale, v_idx1, v_idx2, v_scale,
        codebook, hadamard_8, rs, Tk=Tk, softmax_scale=softmax_scale,
        sliding_window=sliding_window,
    )
    max_abs = (out_ref.float() - out_fk.float()).abs().max().item()
    rmse = (out_ref.float() - out_fk.float()).pow(2).mean().sqrt().item()
    print(f"  SW={sliding_window:>3} Tk={Tk:>3}: max-abs={max_abs:.2e} RMSE={rmse:.2e}")
    assert max_abs < 5e-3, (
        f"sliding_window={sliding_window}, Tk={Tk}: "
        f"max-abs={max_abs:.2e} >= 5e-3 (RMSE={rmse:.2e})"
    )
