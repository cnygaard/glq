"""Phase 1.1 acceptance gate: inline stage-1 dequant.

The kernel ``kernel_unified_attention_2d_e8_v1_1`` replaces the
upstream K/V loads with ``codebook[idx1]`` — no per-group scale, no
Hadamard. This is the simplest dequant arrangement that exercises
the idx-load → codebook-gather → reshape → ``tl.dot`` pipeline.

The Phase 1.1 reference is *not* ``reference_e8_attention``: that
function dequantises with scale + Hadamard, which the kernel won't
do until Phase 1.3-1.4. Instead, this test computes the reference
attention output using the **same dequant path the kernel runs**
(``codebook[idx1].reshape(...)``), then asserts the kernel matches
that path within ``rtol=5e-3, atol=5e-3``.

Once Phase 1.3 lands per-group scale, the test for *that* phase will
reuse this same kernel but compare against a reference with scale
applied. Each sub-phase's test is locked to what the kernel actually
computes at that sub-phase, so a regression at sub-phase N implicates
sub-phase N's diff.
"""
from __future__ import annotations

import math

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton kernels require CUDA",
)


# ── helpers ─────────────────────────────────────────────────────────


def _quantize_to_paged(K, V, quantizer, *, num_blocks, block_size):
    """Encode ``K, V`` with ``E8KVQuantizer`` and rearrange into the
    paged [num_blocks, block_size, num_kv_heads, n_groups] layout the
    kernel expects.

    ``K, V`` have shape ``[Tk, num_kv_heads, head_size]``. We zero-pad
    the trailing block when ``Tk`` doesn't fill it (mirrors what the
    production paged cache would store for unused slots).
    """
    Tk, H_kv, D = K.shape
    n_groups = D // 8
    head_size = D

    qk = quantizer.quantize(K)  # idx1: [Tk*H_kv, n_groups]
    qv = quantizer.quantize(V)

    # Reshape to [Tk, H_kv, n_groups]
    k_idx1_flat = qk["idx1"].reshape(Tk, H_kv, n_groups).contiguous()
    v_idx1_flat = qv["idx1"].reshape(Tk, H_kv, n_groups).contiguous()

    # Build paged buffer [num_blocks, block_size, H_kv, n_groups] int16.
    k_idx1 = torch.zeros(
        num_blocks, block_size, H_kv, n_groups,
        dtype=torch.int16, device=K.device,
    )
    v_idx1 = torch.zeros_like(k_idx1)

    for t in range(Tk):
        blk = t // block_size
        slot = t % block_size
        k_idx1[blk, slot] = k_idx1_flat[t]
        v_idx1[blk, slot] = v_idx1_flat[t]

    return k_idx1, v_idx1, qk, qv


def _kernel_truth_decode_stage1(idx1_paged, codebook, *, num_tokens,
                                 num_kv_heads, head_size, block_size):
    """Reproduce exactly what the kernel decodes to: ``codebook[idx1]``
    reshaped to ``[num_tokens, num_kv_heads, head_size]`` (no scale,
    no Hadamard).
    """
    # idx1_paged is [num_blocks, block_size, num_kv_heads, n_groups] int16.
    # First flatten to [num_blocks * block_size, num_kv_heads, n_groups],
    # then take only the first num_tokens rows.
    nb, bs, H_kv, G = idx1_paged.shape
    flat = idx1_paged.reshape(nb * bs, H_kv, G)[:num_tokens]
    # int16 → int32 unsigned (matches the kernel's ``& 0xFFFF`` mask).
    idx_u32 = (flat.to(torch.int32) & 0xFFFF).long()
    # codebook lookup
    dec = codebook[idx_u32.reshape(-1)]            # [num_tokens*H_kv*G, 8]
    dec = dec.reshape(num_tokens, H_kv, G, 8)
    return dec.reshape(num_tokens, H_kv, head_size)


def _sdpa_kernel_truth(q, K_truth, V_truth, *, softmax_scale):
    """Run plain causal SDPA on the kernel-truth K/V tensors. Returns
    the attention output the forked kernel must match.

    ``q`` has shape ``[Tq, H_q, D]``; ``K_truth, V_truth`` have shape
    ``[Tk, H_kv, D]``. GQA broadcasting handled by ``repeat_interleave``.
    """
    Tq, H_q, D = q.shape
    Tk, H_kv, _ = K_truth.shape
    if H_q != H_kv:
        assert H_q % H_kv == 0
        K_truth = K_truth.repeat_interleave(H_q // H_kv, dim=1)
        V_truth = V_truth.repeat_interleave(H_q // H_kv, dim=1)

    # Compute scores in fp32 to mirror the kernel's online softmax
    # accumulator dtype.
    Q_bh = q.transpose(0, 1).to(torch.float32)         # [H_q, Tq, D]
    K_bh = K_truth.transpose(0, 1).to(torch.float32)   # [H_q, Tk, D]
    V_bh = V_truth.transpose(0, 1).to(torch.float32)

    scores = torch.matmul(Q_bh, K_bh.transpose(-1, -2)) * softmax_scale
    # Causal: query i sees keys 0..Tk-Tq+i
    causal_offset = Tk - Tq
    rows = torch.arange(Tq, device=q.device)[:, None]
    cols = torch.arange(Tk, device=q.device)[None, :]
    mask = cols > (rows + causal_offset)
    scores = scores.masked_fill(mask[None, :, :], float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    out_bh = torch.matmul(weights, V_bh)
    return out_bh.transpose(0, 1).contiguous().to(q.dtype)


@pytest.fixture(scope="module")
def quantizer_stage1():
    """E8 quantizer with n_stages=1 (idx1 only — Phase 1.1 surface).

    Codebook is enumerated on CPU then moved to CUDA so the
    ``E8ShellCodebook.quantize`` Triton fast-path is used during the
    encode step (the fixture inputs are CUDA fp16).
    """
    from glq.codebook import E8ShellCodebook
    from glq.kv_e8 import E8KVQuantizer
    cb = E8ShellCodebook(verbose=False)
    # Move codebook tensors to CUDA (E8ShellCodebook keeps device locally).
    cb.codebook = cb.codebook.to("cuda")
    cb.codebook_norms = cb.codebook_norms.to("cuda")
    cb.codebook_half = cb.codebook_half.to("cuda")
    cb.codebook_half_t = cb.codebook_half_t.to("cuda")
    cb.codebook_norms_half = cb.codebook_norms_half.to("cuda")
    cb.codebook_packed = cb.codebook_packed.to("cuda")
    cb.device = "cuda"
    return E8KVQuantizer(cb, n_stages=1, secondary_stages=0)


def _build_fixture(*, Tq, Tk, num_kv_heads, head_size, block_size, seed):
    """Build the test inputs (queries + un-quantized K/V) on cuda fp16."""
    device = "cuda"
    dtype = torch.float16
    g = torch.Generator(device=device).manual_seed(seed)
    num_q_heads = num_kv_heads  # Phase 1: no GQA.

    q = torch.randn(Tq, num_q_heads, head_size,
                    device=device, dtype=dtype, generator=g)
    K = torch.randn(Tk, num_kv_heads, head_size,
                    device=device, dtype=dtype, generator=g)
    V = torch.randn(Tk, num_kv_heads, head_size,
                    device=device, dtype=dtype, generator=g)
    return q, K, V


def _run_forked_v1_1(q, k_idx1, v_idx1, codebook, *, Tk, softmax_scale,
                     block_size):
    from glq_vllm.triton_unified_attention_e8 import unified_attention_e8_v1_1
    Tq = q.shape[0]
    num_blocks = k_idx1.shape[0]
    out = torch.zeros_like(q)
    cu_seqlens_q = torch.tensor([0, Tq], device=q.device, dtype=torch.int32)
    seqused_k = torch.tensor([Tk], device=q.device, dtype=torch.int32)
    block_table = torch.arange(
        num_blocks, device=q.device, dtype=torch.int32
    ).reshape(1, num_blocks)

    unified_attention_e8_v1_1(
        q=q,
        k_idx1=k_idx1,
        v_idx1=v_idx1,
        codebook=codebook,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        block_table=block_table,
    )
    return out


# ── tests ───────────────────────────────────────────────────────────


def test_phase_1_1_decode_tiny(quantizer_stage1):
    """Tq=1, Tk=16 (one block), head_size=128, single KV head."""
    head_size = 128
    block_size = 16
    Tq, Tk = 1, 16
    q, K, V = _build_fixture(
        Tq=Tq, Tk=Tk, num_kv_heads=1, head_size=head_size,
        block_size=block_size, seed=1,
    )
    num_blocks = (Tk + block_size - 1) // block_size

    k_idx1, v_idx1, _, _ = _quantize_to_paged(
        K, V, quantizer_stage1,
        num_blocks=num_blocks, block_size=block_size,
    )
    codebook = quantizer_stage1.codebook.codebook_half.to(q.device)

    # Phase 1.1 reference: K/V are the codebook decodes (no scale, no Had).
    K_truth = _kernel_truth_decode_stage1(
        k_idx1, codebook,
        num_tokens=Tk, num_kv_heads=1, head_size=head_size,
        block_size=block_size,
    )
    V_truth = _kernel_truth_decode_stage1(
        v_idx1, codebook,
        num_tokens=Tk, num_kv_heads=1, head_size=head_size,
        block_size=block_size,
    )
    softmax_scale = 1.0 / math.sqrt(head_size)
    out_ref = _sdpa_kernel_truth(q, K_truth.to(q.dtype), V_truth.to(q.dtype),
                                  softmax_scale=softmax_scale)
    out_fk = _run_forked_v1_1(
        q, k_idx1, v_idx1, codebook,
        Tk=Tk, softmax_scale=softmax_scale, block_size=block_size,
    )

    max_abs = (out_ref - out_fk).abs().max().item()
    rmse = (out_ref - out_fk).pow(2).mean().sqrt().item()
    print(f"  Tq=1 Tk=16 H=128: max-abs={max_abs:.2e}, RMSE={rmse:.2e}")
    torch.testing.assert_close(
        out_fk, out_ref, rtol=5e-3, atol=5e-3,
        msg=f"max-abs={max_abs:.2e}, RMSE={rmse:.2e}",
    )


def test_phase_1_1_prefill_tiny(quantizer_stage1):
    """Tq=Tk=16, head_size=128."""
    head_size = 128
    block_size = 16
    Tq, Tk = 16, 16
    q, K, V = _build_fixture(
        Tq=Tq, Tk=Tk, num_kv_heads=1, head_size=head_size,
        block_size=block_size, seed=42,
    )
    num_blocks = (Tk + block_size - 1) // block_size

    k_idx1, v_idx1, _, _ = _quantize_to_paged(
        K, V, quantizer_stage1,
        num_blocks=num_blocks, block_size=block_size,
    )
    codebook = quantizer_stage1.codebook.codebook_half.to(q.device)

    K_truth = _kernel_truth_decode_stage1(
        k_idx1, codebook,
        num_tokens=Tk, num_kv_heads=1, head_size=head_size,
        block_size=block_size,
    )
    V_truth = _kernel_truth_decode_stage1(
        v_idx1, codebook,
        num_tokens=Tk, num_kv_heads=1, head_size=head_size,
        block_size=block_size,
    )
    softmax_scale = 1.0 / math.sqrt(head_size)
    out_ref = _sdpa_kernel_truth(q, K_truth.to(q.dtype), V_truth.to(q.dtype),
                                  softmax_scale=softmax_scale)
    out_fk = _run_forked_v1_1(
        q, k_idx1, v_idx1, codebook,
        Tk=Tk, softmax_scale=softmax_scale, block_size=block_size,
    )

    max_abs = (out_ref - out_fk).abs().max().item()
    rmse = (out_ref - out_fk).pow(2).mean().sqrt().item()
    print(f"  Tq=Tk=16 H=128: max-abs={max_abs:.2e}, RMSE={rmse:.2e}")
    torch.testing.assert_close(
        out_fk, out_ref, rtol=5e-3, atol=5e-3,
        msg=f"max-abs={max_abs:.2e}, RMSE={rmse:.2e}",
    )


def test_phase_1_1_two_blocks(quantizer_stage1):
    """Tq=1, Tk=32 (two blocks)."""
    head_size = 128
    block_size = 16
    Tq, Tk = 1, 32
    q, K, V = _build_fixture(
        Tq=Tq, Tk=Tk, num_kv_heads=1, head_size=head_size,
        block_size=block_size, seed=7,
    )
    num_blocks = (Tk + block_size - 1) // block_size

    k_idx1, v_idx1, _, _ = _quantize_to_paged(
        K, V, quantizer_stage1,
        num_blocks=num_blocks, block_size=block_size,
    )
    codebook = quantizer_stage1.codebook.codebook_half.to(q.device)

    K_truth = _kernel_truth_decode_stage1(
        k_idx1, codebook,
        num_tokens=Tk, num_kv_heads=1, head_size=head_size,
        block_size=block_size,
    )
    V_truth = _kernel_truth_decode_stage1(
        v_idx1, codebook,
        num_tokens=Tk, num_kv_heads=1, head_size=head_size,
        block_size=block_size,
    )
    softmax_scale = 1.0 / math.sqrt(head_size)
    out_ref = _sdpa_kernel_truth(q, K_truth.to(q.dtype), V_truth.to(q.dtype),
                                  softmax_scale=softmax_scale)
    out_fk = _run_forked_v1_1(
        q, k_idx1, v_idx1, codebook,
        Tk=Tk, softmax_scale=softmax_scale, block_size=block_size,
    )

    max_abs = (out_ref - out_fk).abs().max().item()
    rmse = (out_ref - out_fk).pow(2).mean().sqrt().item()
    print(f"  Tq=1 Tk=32 (2 blocks): max-abs={max_abs:.2e}, RMSE={rmse:.2e}")
    torch.testing.assert_close(
        out_fk, out_ref, rtol=5e-3, atol=5e-3,
        msg=f"max-abs={max_abs:.2e}, RMSE={rmse:.2e}",
    )
