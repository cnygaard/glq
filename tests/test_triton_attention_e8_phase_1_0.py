"""Phase 1.0 acceptance gate: forked skeleton kernel matches upstream
``unified_attention`` bit-identically when given the same pre-decompressed
K/V tensors.

This isolates "did I break the fork before adding any new math?" from
the subsequent sub-phases that will add the actual inline-dequant code.
A regression at any later sub-phase points back to that sub-phase's
diff, not to the fork itself.

Fixture choice — Tq=Tk=16, num_kv_heads=1, head_size=128 — matches the
Phase 1 scope (one tile = one block, single sequence, no GQA).

Test is GPU-only; the upstream Triton kernel is not designed to run on
CPU.
"""
from __future__ import annotations

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton kernels require CUDA",
)


def _make_fixture(
    *,
    num_seqs: int,
    Tq_per_seq: int,
    Tk_per_seq: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    seed: int = 0,
):
    """Build the inputs needed by upstream ``unified_attention`` *and*
    ``unified_attention_e8`` (Phase 1.0).

    All sequences in the batch share the same Tq / Tk for simplicity.
    Each sequence's K/V occupies ``ceil(Tk_per_seq / block_size)`` paged
    blocks; physical block indices are assigned contiguously per-seq.
    """
    g = torch.Generator(device=device).manual_seed(seed)

    blocks_per_seq = (Tk_per_seq + block_size - 1) // block_size
    total_blocks = num_seqs * blocks_per_seq

    total_q_tokens = num_seqs * Tq_per_seq

    q = torch.randn(
        total_q_tokens, num_q_heads, head_size,
        device=device, dtype=dtype, generator=g,
    )
    k_cache = torch.zeros(
        total_blocks, block_size, num_kv_heads, head_size,
        device=device, dtype=dtype,
    )
    v_cache = torch.zeros(
        total_blocks, block_size, num_kv_heads, head_size,
        device=device, dtype=dtype,
    )

    # Fill K/V with per-token random data only at valid slots.
    block_table = torch.empty(
        num_seqs, blocks_per_seq, device=device, dtype=torch.int32,
    )
    seqused_k = torch.full(
        (num_seqs,), Tk_per_seq, device=device, dtype=torch.int32,
    )
    for s in range(num_seqs):
        for b in range(blocks_per_seq):
            block_table[s, b] = s * blocks_per_seq + b

        for tok in range(Tk_per_seq):
            blk = tok // block_size
            slot = tok % block_size
            phys = int(block_table[s, blk].item())
            k_cache[phys, slot] = torch.randn(
                num_kv_heads, head_size,
                device=device, dtype=dtype, generator=g,
            )
            v_cache[phys, slot] = torch.randn(
                num_kv_heads, head_size,
                device=device, dtype=dtype, generator=g,
            )

    cu_seqlens_q = torch.tensor(
        [i * Tq_per_seq for i in range(num_seqs + 1)],
        device=device, dtype=torch.int32,
    )

    return dict(
        q=q,
        k=k_cache,
        v=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        block_table=block_table,
    )


def _run_upstream(fx, *, softmax_scale, max_seqlen_q, max_seqlen_k):
    """Call vLLM's unmodified ``unified_attention`` with all the
    Phase-1-out-of-scope features disabled."""
    from vllm.v1.attention.ops.triton_unified_attention import (
        unified_attention,
    )

    q = fx["q"]
    out = torch.zeros_like(q)
    unified_attention(
        q=q,
        k=fx["k"],
        v=fx["v"],
        out=out,
        cu_seqlens_q=fx["cu_seqlens_q"],
        max_seqlen_q=max_seqlen_q,
        seqused_k=fx["seqused_k"],
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(-1, -1),
        block_table=fx["block_table"],
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        alibi_slopes=None,
        output_scale=None,
        qq_bias=None,
        sinks=None,
        mm_prefix_range=None,
        use_alibi_sqrt=False,
    )
    return out


def _run_forked(fx, *, softmax_scale):
    """Call the Phase 1.0 forked kernel."""
    from glq_vllm.triton_unified_attention_e8 import unified_attention_e8

    q = fx["q"]
    out = torch.zeros_like(q)
    unified_attention_e8(
        q=q,
        k=fx["k"],
        v=fx["v"],
        out=out,
        cu_seqlens_q=fx["cu_seqlens_q"],
        seqused_k=fx["seqused_k"],
        softmax_scale=softmax_scale,
        block_table=fx["block_table"],
    )
    return out


# ── tests ──────────────────────────────────────────────────────


def test_phase_1_0_bit_identical_tiny_decode():
    """Tq=1, Tk=16, single sequence, head_size=128 — the smallest
    decode-shape that exercises the tile loop once."""
    import math
    head_size = 128
    fx = _make_fixture(
        num_seqs=1,
        Tq_per_seq=1,
        Tk_per_seq=16,
        num_q_heads=1,
        num_kv_heads=1,
        head_size=head_size,
        block_size=16,
        seed=42,
    )
    scale = 1.0 / math.sqrt(head_size)

    out_up = _run_upstream(
        fx, softmax_scale=scale, max_seqlen_q=1, max_seqlen_k=16)
    out_fk = _run_forked(fx, softmax_scale=scale)

    max_abs = (out_up - out_fk).abs().max().item()
    print(f"  Tq=1 Tk=16: max-abs diff = {max_abs:.2e}")
    assert torch.equal(out_up, out_fk), (
        f"Phase 1.0 forked kernel differs from upstream: max-abs={max_abs}"
    )


def test_phase_1_0_bit_identical_prefill():
    """Tq=Tk=16, single sequence, head_size=128 — the canonical
    pure-prefill shape used in the rest of the plan."""
    import math
    head_size = 128
    fx = _make_fixture(
        num_seqs=1,
        Tq_per_seq=16,
        Tk_per_seq=16,
        num_q_heads=1,
        num_kv_heads=1,
        head_size=head_size,
        block_size=16,
        seed=7,
    )
    scale = 1.0 / math.sqrt(head_size)

    out_up = _run_upstream(
        fx, softmax_scale=scale, max_seqlen_q=16, max_seqlen_k=16)
    out_fk = _run_forked(fx, softmax_scale=scale)

    max_abs = (out_up - out_fk).abs().max().item()
    print(f"  Tq=Tk=16: max-abs diff = {max_abs:.2e}")
    assert torch.equal(out_up, out_fk), (
        f"Phase 1.0 forked kernel differs from upstream: max-abs={max_abs}"
    )


def test_phase_1_0_bit_identical_multi_block():
    """Tq=1, Tk=32 (two blocks), head_size=128 — exercises the tile
    loop iterating over multiple blocks."""
    import math
    head_size = 128
    fx = _make_fixture(
        num_seqs=1,
        Tq_per_seq=1,
        Tk_per_seq=32,
        num_q_heads=1,
        num_kv_heads=1,
        head_size=head_size,
        block_size=16,
        seed=99,
    )
    scale = 1.0 / math.sqrt(head_size)

    out_up = _run_upstream(
        fx, softmax_scale=scale, max_seqlen_q=1, max_seqlen_k=32)
    out_fk = _run_forked(fx, softmax_scale=scale)

    max_abs = (out_up - out_fk).abs().max().item()
    print(f"  Tq=1 Tk=32 (2 blocks): max-abs diff = {max_abs:.2e}")
    assert torch.equal(out_up, out_fk), (
        f"Phase 1.0 forked kernel differs from upstream: max-abs={max_abs}"
    )
