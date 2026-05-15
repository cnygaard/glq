"""Phase 5.3 Stage 4a: tests for the fused Triton write-side kernels.

The slow ``write_kv`` calls ``quantizer.quantize`` (which fires 3
pre-NN + 1-3 NN + 3-9 post-NN ops) followed by ``_scatter_into_cache``
(which does 2-5 ``index_put_`` calls). Stage 4a replaces the scatter
with one Triton kernel; the math is unchanged so the output cache
state must be bit-exact to the slow path for every bpw rung.

Tests on CUDA only — the kernels here require Triton.
"""
from __future__ import annotations

import pytest
import torch

from glq.codebook_relaxed import E8RelaxedCodebook
from glq.kv_e8 import E8KVQuantizer
from glq_vllm.e8_paged_cache import (
    E8PagedKVCache,
    _BPW_RECIPE,
    gather_kv_to_paged_fp16,
    write_kv,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Fused write kernel requires CUDA",
)


@pytest.fixture(scope="module")
def relaxed_cuda():
    return E8RelaxedCodebook(device="cuda", verbose=False)


def _build_quantizer(cb, bpw: int):
    n_p, n_s = _BPW_RECIPE[bpw]
    small = cb.make_small() if n_s > 0 else None
    return E8KVQuantizer(cb, n_stages=n_p,
                        secondary_codebook=small, secondary_stages=n_s)


def _random_kv(seed: int, num_tokens: int, num_kv_heads: int,
               head_size: int, dtype=torch.bfloat16, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    key = torch.randn(num_tokens, num_kv_heads, head_size,
                      generator=g, dtype=torch.float32,
                      device=device).to(dtype)
    value = torch.randn(num_tokens, num_kv_heads, head_size,
                        generator=g, dtype=torch.float32,
                        device=device).to(dtype)
    return key, value


def _build_two_caches(quant, bpw, head_size, num_kv_heads=2,
                     block_size=16, num_blocks=4, dtype=torch.bfloat16):
    cache_a = E8PagedKVCache.alloc(
        num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size, bpw=bpw,
        device="cuda", dtype=dtype)
    cache_b = E8PagedKVCache.alloc(
        num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size, bpw=bpw,
        device="cuda", dtype=dtype)
    return cache_a, cache_b


def _assert_caches_equal(a: E8PagedKVCache, b: E8PagedKVCache, *, bpw: int):
    assert torch.equal(a.k_idx1, b.k_idx1), f"k_idx1 mismatch at bpw={bpw}"
    assert torch.equal(a.v_idx1, b.v_idx1), f"v_idx1 mismatch at bpw={bpw}"
    assert torch.equal(a.k_scale, b.k_scale), f"k_scale mismatch at bpw={bpw}"
    assert torch.equal(a.v_scale, b.v_scale), f"v_scale mismatch at bpw={bpw}"
    if a.n_primary >= 2:
        assert torch.equal(a.k_idx2, b.k_idx2), f"k_idx2 mismatch at bpw={bpw}"
        assert torch.equal(a.v_idx2, b.v_idx2), f"v_idx2 mismatch at bpw={bpw}"
    if a.n_primary >= 3:
        assert torch.equal(a.k_idx3, b.k_idx3), f"k_idx3 mismatch at bpw={bpw}"
        assert torch.equal(a.v_idx3, b.v_idx3), f"v_idx3 mismatch at bpw={bpw}"
    if a.n_secondary >= 1:
        assert torch.equal(a.k_idx_s1, b.k_idx_s1), f"k_idx_s1 mismatch at bpw={bpw}"
        assert torch.equal(a.v_idx_s1, b.v_idx_s1), f"v_idx_s1 mismatch at bpw={bpw}"


@pytest.mark.parametrize("bpw", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("head_size", [128, 256, 512])
def test_write_kv_fused_matches_slow_path(relaxed_cuda, bpw, head_size):
    """The fused write path must produce a cache state bit-identical
    to the slow ``write_kv`` for every bpw and head_size we support."""
    from glq_vllm.e8_paged_cache import write_kv_fused

    quant = _build_quantizer(relaxed_cuda, bpw)
    num_blocks, block_size, num_kv_heads = 4, 16, 2
    cache_slow, cache_fused = _build_two_caches(
        quant, bpw, head_size, num_kv_heads, block_size, num_blocks)

    n_tokens = num_blocks * block_size
    key, value = _random_kv(seed=bpw * 7 + head_size, num_tokens=n_tokens,
                            num_kv_heads=num_kv_heads, head_size=head_size,
                            dtype=torch.bfloat16, device="cuda")
    slot_mapping = torch.arange(n_tokens, dtype=torch.long, device="cuda")

    write_kv(quant, key, value, cache_slow, slot_mapping)
    write_kv_fused(quant, key, value, cache_fused, slot_mapping)

    _assert_caches_equal(cache_slow, cache_fused, bpw=bpw)


@pytest.mark.parametrize("bpw", [2, 4, 6])
def test_write_kv_fused_non_contiguous_slot_mapping(relaxed_cuda, bpw):
    """Slot mappings that span multiple blocks and aren't monotonic
    must still produce the same cache state."""
    from glq_vllm.e8_paged_cache import write_kv_fused

    quant = _build_quantizer(relaxed_cuda, bpw)
    num_blocks, block_size, num_kv_heads, head_size = 8, 16, 2, 256
    cache_slow, cache_fused = _build_two_caches(
        quant, bpw, head_size, num_kv_heads, block_size, num_blocks)

    n_tokens = 40
    key, value = _random_kv(seed=42, num_tokens=n_tokens,
                            num_kv_heads=num_kv_heads, head_size=head_size,
                            dtype=torch.bfloat16, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    slot_mapping = torch.randperm(
        num_blocks * block_size, generator=g,
        device="cuda")[:n_tokens].long()

    write_kv(quant, key, value, cache_slow, slot_mapping)
    write_kv_fused(quant, key, value, cache_fused, slot_mapping)

    _assert_caches_equal(cache_slow, cache_fused, bpw=bpw)


@pytest.mark.parametrize("bpw", [2, 4])
@pytest.mark.parametrize("head_size", [256, 512])
def test_write_kv_fused_all_padding_does_not_crash(
        relaxed_cuda, bpw, head_size):
    """vLLM's flashinfer_autotune dummy run passes slot_mapping where
    *every* entry is -1. The kernel must be safe in this case — either
    by skipping at the Python wrapper or with a robust in-kernel mask
    — and must not corrupt the cache. Repro for the head_size=512 boot
    crash during vLLM warmup.
    """
    from glq_vllm.e8_paged_cache import write_kv_fused

    quant = _build_quantizer(relaxed_cuda, bpw)
    num_blocks, block_size, num_kv_heads = 4, 16, 2
    cache_before, _ = _build_two_caches(
        quant, bpw, head_size, num_kv_heads, block_size, num_blocks)

    # Capture the cache state before the call.
    pre_idx1 = cache_before.k_idx1.clone()
    pre_scale = cache_before.k_scale.clone()

    n_tokens = 32
    key, value = _random_kv(seed=bpw * 71 + head_size,
                            num_tokens=n_tokens,
                            num_kv_heads=num_kv_heads,
                            head_size=head_size,
                            dtype=torch.bfloat16, device="cuda")
    # All slots are padding — mirrors vLLM's warmup.
    slot_mapping = torch.full((n_tokens,), -1,
                              dtype=torch.long, device="cuda")

    # Must not crash.
    write_kv_fused(quant, key, value, cache_before, slot_mapping)
    torch.cuda.synchronize()

    # Cache state must be unchanged (no slot was valid).
    assert torch.equal(cache_before.k_idx1, pre_idx1)
    assert torch.equal(cache_before.k_scale, pre_scale)


@pytest.mark.parametrize("bpw", [2, 4, 6])
def test_write_kv_fused_handles_negative_padding_slots(relaxed_cuda, bpw):
    """vLLM's flashinfer_autotune dummy run + chunked prefill padding
    pass slot_mapping with -1 entries for tokens that should be
    skipped. The slow ``buf[block, slot] = src`` path silently writes
    those to a wrap-around slot; our Triton kernel must mask them off
    so it doesn't crash with an out-of-bounds store. Regression test
    for the head_size=512 boot crash discovered in Stage 4a."""
    from glq_vllm.e8_paged_cache import write_kv_fused

    quant = _build_quantizer(relaxed_cuda, bpw)
    num_blocks, block_size, num_kv_heads, head_size = 4, 16, 2, 512
    cache_slow, cache_fused = _build_two_caches(
        quant, bpw, head_size, num_kv_heads, block_size, num_blocks)

    n_tokens = 20
    key, value = _random_kv(seed=bpw * 31, num_tokens=n_tokens,
                            num_kv_heads=num_kv_heads, head_size=head_size,
                            dtype=torch.bfloat16, device="cuda")
    # Half the tokens are real (slots 0..9); the rest carry the -1
    # sentinel that vLLM uses for padding.
    slot_mapping = torch.full((n_tokens,), -1, dtype=torch.long, device="cuda")
    slot_mapping[:10] = torch.arange(10, dtype=torch.long, device="cuda")

    # Should not crash on either path; ignoring the negative slots is
    # the expected behavior.
    write_kv_fused(quant, key, value, cache_fused, slot_mapping)
    # The slow path also "succeeds" (writes -1 entries to wrap-around
    # slots) — but the real slots 0..9 should match between paths.
    valid = slot_mapping >= 0
    write_kv(quant, key[valid], value[valid], cache_slow,
             slot_mapping[valid])

    # The valid slots must round-trip identically.
    from glq_vllm.e8_paged_cache import gather_kv
    k_slow, _ = gather_kv(quant, cache_slow, slot_mapping[valid])
    k_fused, _ = gather_kv(quant, cache_fused, slot_mapping[valid])
    assert torch.equal(k_slow, k_fused), (
        f"valid-slot K mismatch after padding-aware write at bpw={bpw}: "
        f"max delta={(k_slow-k_fused).abs().max().item()}")


@pytest.mark.parametrize("bpw", [2, 4])
@pytest.mark.parametrize("head_size", [256, 512])
def test_write_kv_fused_from_paged_storage_large_nb(
        relaxed_cuda, bpw, head_size):
    """Stage 2c-2c case: cache is a view into vLLM's allocated buffer
    with large num_blocks (>2^31 / stride). Exercises int64 offset math
    AND non-contiguous strided cache. Reproduces the head_size=512
    illegal-memory-access boot crash discovered in Stage 4a."""
    from glq_vllm.e8_paged_cache import write_kv_fused
    from glq_vllm.e8_kv_spec import compressed_kv_cache_shape

    quant = _build_quantizer(relaxed_cuda, bpw)
    # Big enough that block * cache_stride_nb overflows int32 for
    # head_size=512 (~668k blocks × 12k stride > 2.1B int16 elements).
    num_blocks, block_size, num_kv_heads = 700_000, 16, 2
    dtype = torch.bfloat16
    shape = compressed_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size,
        bpw=bpw, dtype_size=2)
    # We don't have memory for the full buffer — limit num_blocks just
    # for this test. 200k blocks × 2 × 16 × 2 × 192 × 2 bytes = ~2.5GB
    # for head_size=512 (still within a single GPU's free budget).
    # For head_size=256, 200k × 2 × 16 × 2 × 96 × 2 = ~1.3GB.
    if head_size == 512:
        num_blocks = 200_000
    else:
        num_blocks = 200_000  # also triggers stride * NB > 2^31
    shape = compressed_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size,
        bpw=bpw, dtype_size=2)
    raw = torch.zeros(shape, dtype=dtype, device="cuda")
    k_buf = raw[:, 0]
    v_buf = raw[:, 1]
    cache = E8PagedKVCache.from_paged_storage(
        k_buf, v_buf, head_size=head_size, bpw=bpw, dtype=dtype)

    # Write a small number of tokens at scattered slots near the
    # END of the cache so block ids approach num_blocks. This is
    # where int32 offset arithmetic overflows.
    n_tokens = 32
    key, value = _random_kv(seed=bpw * 5 + head_size,
                            num_tokens=n_tokens,
                            num_kv_heads=num_kv_heads,
                            head_size=head_size, dtype=dtype, device="cuda")
    # Slots near the high end of the cache.
    far_block_start = num_blocks - 4
    slot_mapping = torch.arange(
        far_block_start * block_size,
        far_block_start * block_size + n_tokens,
        dtype=torch.long, device="cuda")

    # Must not crash. Verify roundtrip matches the quantize-dequantize
    # reference (i.e. the writes ended up at the right slots).
    write_kv_fused(quant, key, value, cache, slot_mapping)
    torch.cuda.synchronize()  # force sync to surface async kernel errors

    from glq_vllm.e8_paged_cache import gather_kv
    k_out, v_out = gather_kv(quant, cache, slot_mapping)
    k_ref = quant.dequantize(quant.quantize(key))
    v_ref = quant.dequantize(quant.quantize(value))
    assert torch.equal(k_out, k_ref), \
        f"K roundtrip bpw={bpw} head={head_size}: max delta={(k_out-k_ref).abs().max().item()}"
    assert torch.equal(v_out, v_ref), \
        f"V roundtrip bpw={bpw} head={head_size}: max delta={(v_out-v_ref).abs().max().item()}"


@pytest.mark.parametrize("bpw", [2, 4, 6])
def test_write_kv_fused_then_gather_roundtrip(relaxed_cuda, bpw):
    """End-to-end: write through the fused path, then read back via the
    existing gather. Must match the quantize→dequantize reference."""
    from glq_vllm.e8_paged_cache import write_kv_fused, gather_kv

    quant = _build_quantizer(relaxed_cuda, bpw)
    num_blocks, block_size, num_kv_heads, head_size = 4, 16, 2, 256
    dtype = torch.bfloat16
    cache = E8PagedKVCache.alloc(
        num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size, bpw=bpw,
        device="cuda", dtype=dtype)

    n_tokens = num_blocks * block_size
    key, value = _random_kv(seed=bpw * 11, num_tokens=n_tokens,
                            num_kv_heads=num_kv_heads, head_size=head_size,
                            dtype=dtype, device="cuda")
    slot_mapping = torch.arange(n_tokens, dtype=torch.long, device="cuda")

    write_kv_fused(quant, key, value, cache, slot_mapping)
    k_out, v_out = gather_kv(quant, cache, slot_mapping)

    k_ref = quant.dequantize(quant.quantize(key))
    v_ref = quant.dequantize(quant.quantize(value))

    assert torch.equal(k_out, k_ref), \
        f"K roundtrip bpw={bpw}: max delta={(k_out-k_ref).abs().max().item()}"
    assert torch.equal(v_out, v_ref), \
        f"V roundtrip bpw={bpw}: max delta={(v_out-v_ref).abs().max().item()}"
