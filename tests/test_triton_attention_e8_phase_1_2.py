"""Phase 1.2 acceptance gate: stage-2 residual (idx1 + idx2 / RS).

Kernel ``kernel_unified_attention_2d_e8_v1_2`` extends Phase 1.1 with
the second-stage codebook lookup. Decode math:

    dec = codebook[idx1] + codebook[idx2] / RS

Where RS = ``codebook.resid_scale`` (the quantizer's residual-scale
constant). Per-group scale and Hadamard inverse are still NOT applied
(those land in Phase 1.3 and Phase 1.4).

The convention is **divide on decode**: the encoder multiplies the
residual by RS before quantising it, the decoder undoes by dividing.
If we accidentally multiplied instead of divided, the output would
have the right magnitude but wrong sign — caught by the rtol gate.
"""
from __future__ import annotations

import math

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton kernels require CUDA",
)


def _quantize_to_paged_2stage(K, V, quantizer, *, num_blocks, block_size):
    """Encode at n_stages=2 and arrange (idx1, idx2) into paged layout."""
    Tk, H_kv, D = K.shape
    n_groups = D // 8

    qk = quantizer.quantize(K)
    qv = quantizer.quantize(V)

    def _pack(qt_idx):
        flat = qt_idx.reshape(Tk, H_kv, n_groups).contiguous()
        paged = torch.zeros(
            num_blocks, block_size, H_kv, n_groups,
            dtype=torch.int16, device=K.device,
        )
        for t in range(Tk):
            blk = t // block_size
            slot = t % block_size
            paged[blk, slot] = flat[t]
        return paged

    return (_pack(qk["idx1"]), _pack(qk["idx2"]),
            _pack(qv["idx1"]), _pack(qv["idx2"]))


def _kernel_truth_decode_stage2(idx1_paged, idx2_paged, codebook, rs, *,
                                num_tokens, num_kv_heads, head_size):
    """Reproduce the Phase 1.2 dequant: ``codebook[idx1] + codebook[idx2]/RS``,
    reshape to ``[num_tokens, num_kv_heads, head_size]``.
    """
    nb, bs, H_kv, G = idx1_paged.shape

    def _flat(p):
        return (p.reshape(nb * bs, H_kv, G)[:num_tokens]
                .to(torch.int32) & 0xFFFF).long()
    f1 = _flat(idx1_paged)
    f2 = _flat(idx2_paged)
    dec = codebook[f1.reshape(-1)].to(torch.float32)
    dec = dec + codebook[f2.reshape(-1)].to(torch.float32) / rs
    dec = dec.reshape(num_tokens, H_kv, G, 8)
    return dec.reshape(num_tokens, H_kv, head_size)


def _sdpa_kernel_truth(q, K_truth, V_truth, *, softmax_scale):
    Tq, H_q, D = q.shape
    Tk, H_kv, _ = K_truth.shape
    if H_q != H_kv:
        K_truth = K_truth.repeat_interleave(H_q // H_kv, dim=1)
        V_truth = V_truth.repeat_interleave(H_q // H_kv, dim=1)
    Q_bh = q.transpose(0, 1).to(torch.float32)
    K_bh = K_truth.transpose(0, 1).to(torch.float32)
    V_bh = V_truth.transpose(0, 1).to(torch.float32)

    scores = torch.matmul(Q_bh, K_bh.transpose(-1, -2)) * softmax_scale
    causal_offset = Tk - Tq
    rows = torch.arange(Tq, device=q.device)[:, None]
    cols = torch.arange(Tk, device=q.device)[None, :]
    mask = cols > (rows + causal_offset)
    scores = scores.masked_fill(mask[None, :, :], float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    out_bh = torch.matmul(weights, V_bh)
    return out_bh.transpose(0, 1).contiguous().to(q.dtype)


@pytest.fixture(scope="module")
def quantizer_stage2():
    """E8 quantizer at n_stages=2 (Phase 1.2 surface — 4 bpw recipe)."""
    from glq.codebook import E8ShellCodebook
    from glq.kv_e8 import E8KVQuantizer
    cb = E8ShellCodebook(verbose=False)
    for name in ("codebook", "codebook_norms", "codebook_half",
                 "codebook_half_t", "codebook_norms_half",
                 "codebook_packed"):
        setattr(cb, name, getattr(cb, name).to("cuda"))
    cb.device = "cuda"
    return E8KVQuantizer(cb, n_stages=2, secondary_stages=0)


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


def _run_forked_v1_2(q, k_idx1, k_idx2, v_idx1, v_idx2, codebook,
                     resid_scale, *, Tk, softmax_scale):
    from glq_vllm.triton_unified_attention_e8 import (
        unified_attention_e8_v1_2,
    )
    Tq = q.shape[0]
    num_blocks = k_idx1.shape[0]
    out = torch.zeros_like(q)
    cu_seqlens_q = torch.tensor([0, Tq], device=q.device, dtype=torch.int32)
    seqused_k = torch.tensor([Tk], device=q.device, dtype=torch.int32)
    block_table = torch.arange(
        num_blocks, device=q.device, dtype=torch.int32
    ).reshape(1, num_blocks)
    unified_attention_e8_v1_2(
        q=q,
        k_idx1=k_idx1, k_idx2=k_idx2,
        v_idx1=v_idx1, v_idx2=v_idx2,
        codebook=codebook,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        resid_scale=resid_scale,
        block_table=block_table,
    )
    return out


def _run_case(*, Tq, Tk, head_size, block_size, seed, quantizer):
    q, K, V = _build_fixture(
        Tq=Tq, Tk=Tk, num_kv_heads=1, head_size=head_size,
        block_size=block_size, seed=seed,
    )
    num_blocks = (Tk + block_size - 1) // block_size

    k_idx1, k_idx2, v_idx1, v_idx2 = _quantize_to_paged_2stage(
        K, V, quantizer, num_blocks=num_blocks, block_size=block_size,
    )
    codebook = quantizer.codebook.codebook_half.to(q.device)
    rs = float(quantizer.codebook.resid_scale)

    K_truth = _kernel_truth_decode_stage2(
        k_idx1, k_idx2, codebook, rs,
        num_tokens=Tk, num_kv_heads=1, head_size=head_size,
    )
    V_truth = _kernel_truth_decode_stage2(
        v_idx1, v_idx2, codebook, rs,
        num_tokens=Tk, num_kv_heads=1, head_size=head_size,
    )
    softmax_scale = 1.0 / math.sqrt(head_size)
    out_ref = _sdpa_kernel_truth(
        q, K_truth.to(q.dtype), V_truth.to(q.dtype),
        softmax_scale=softmax_scale,
    )
    out_fk = _run_forked_v1_2(
        q, k_idx1, k_idx2, v_idx1, v_idx2, codebook, rs,
        Tk=Tk, softmax_scale=softmax_scale,
    )
    max_abs = (out_ref - out_fk).abs().max().item()
    rmse = (out_ref - out_fk).pow(2).mean().sqrt().item()
    return out_ref, out_fk, max_abs, rmse


def test_phase_1_2_decode_tiny(quantizer_stage2):
    _, out_fk, max_abs, rmse = _run_case(
        Tq=1, Tk=16, head_size=128, block_size=16, seed=1,
        quantizer=quantizer_stage2,
    )
    print(f"  Tq=1 Tk=16: max-abs={max_abs:.2e}, RMSE={rmse:.2e}")
    assert max_abs < 5e-3, f"max-abs={max_abs:.2e}"


def test_phase_1_2_prefill(quantizer_stage2):
    _, out_fk, max_abs, rmse = _run_case(
        Tq=16, Tk=16, head_size=128, block_size=16, seed=42,
        quantizer=quantizer_stage2,
    )
    print(f"  Tq=Tk=16: max-abs={max_abs:.2e}, RMSE={rmse:.2e}")
    assert max_abs < 5e-3, f"max-abs={max_abs:.2e}"


def test_phase_1_2_two_blocks(quantizer_stage2):
    _, out_fk, max_abs, rmse = _run_case(
        Tq=1, Tk=32, head_size=128, block_size=16, seed=7,
        quantizer=quantizer_stage2,
    )
    print(f"  Tq=1 Tk=32 (2 blocks): max-abs={max_abs:.2e}, RMSE={rmse:.2e}")
    assert max_abs < 5e-3, f"max-abs={max_abs:.2e}"
