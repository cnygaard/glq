"""Phase 5.3 Stage 3b: tests for the fused Triton dequant-gather kernel.

The Python ``gather_kv_to_paged_fp16`` (slow path) is the reference;
the fused Triton kernel must produce the same fp16/bf16 output for
every bpw rung, both for full and scoped (block_indices subset)
decompression. The slow path runs on CPU; the fused kernel runs on
CUDA only.

Math: each (block, slot, head, group) tuple decodes 8 elements via
codebook lookup + (optional) residual + (optional) secondary +
per-group scale + 8x8 inverse Hadamard. Numerics across the two paths
should agree within ~ULP tolerance; we use a loose 1e-2 atol/rtol to
absorb fp16/bf16 reduction-order differences without missing real bugs.
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
    reason="Fused gather kernel requires CUDA",
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


@pytest.mark.parametrize("bpw", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("head_size", [128, 256, 512])
def test_fused_gather_full_matches_slow_path(relaxed_cuda, bpw, head_size):
    """Decompressing the entire cache via the fused kernel must match
    the slow Python path within tolerance for every bpw rung."""
    from glq_vllm.e8_paged_cache import gather_kv_to_paged_fp16_fused

    block_size, num_kv_heads, num_blocks = 16, 2, 4
    dtype = torch.bfloat16
    cache = E8PagedKVCache.alloc(
        num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size, bpw=bpw,
        device="cuda", dtype=dtype)
    quant = _build_quantizer(relaxed_cuda, bpw)

    n_tokens = num_blocks * block_size
    key, value = _random_kv(seed=bpw * 13 + head_size,
                            num_tokens=n_tokens,
                            num_kv_heads=num_kv_heads,
                            head_size=head_size, dtype=dtype, device="cuda")
    slot_mapping = torch.arange(n_tokens, dtype=torch.long, device="cuda")
    write_kv(quant, key, value, cache, slot_mapping)

    k_ref, v_ref = gather_kv_to_paged_fp16(quant, cache)
    k_fused, v_fused = gather_kv_to_paged_fp16_fused(quant, cache)

    assert k_fused.shape == k_ref.shape
    assert v_fused.shape == v_ref.shape
    assert k_fused.dtype == k_ref.dtype
    assert v_fused.dtype == v_ref.dtype
    # bf16 has ~3-decimal precision; the slow path also uses bf16 here so
    # the math should match closely. atol=1e-2 catches >0.01 drift.
    torch.testing.assert_close(k_fused, k_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(v_fused, v_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("bpw", [2, 3, 4, 5, 6, 7])
def test_fused_gather_scoped_matches_slow_path(relaxed_cuda, bpw):
    """Scoped decompression (only ``block_indices`` subset) must match
    the same subset of a full-cache decompression."""
    from glq_vllm.e8_paged_cache import gather_kv_to_paged_fp16_fused

    block_size, num_kv_heads, head_size, num_blocks = 16, 2, 256, 12
    dtype = torch.bfloat16
    cache = E8PagedKVCache.alloc(
        num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size, bpw=bpw,
        device="cuda", dtype=dtype)
    quant = _build_quantizer(relaxed_cuda, bpw)

    n_tokens = num_blocks * block_size
    key, value = _random_kv(seed=bpw * 19 + 5, num_tokens=n_tokens,
                            num_kv_heads=num_kv_heads,
                            head_size=head_size, dtype=dtype, device="cuda")
    slot_mapping = torch.arange(n_tokens, dtype=torch.long, device="cuda")
    write_kv(quant, key, value, cache, slot_mapping)

    selected = torch.tensor([3, 7, 9, 0], dtype=torch.long, device="cuda")
    k_fused, v_fused = gather_kv_to_paged_fp16_fused(
        quant, cache, block_indices=selected)
    k_full_ref, v_full_ref = gather_kv_to_paged_fp16(quant, cache)
    k_ref_sel = k_full_ref[selected]
    v_ref_sel = v_full_ref[selected]

    assert k_fused.shape == k_ref_sel.shape
    torch.testing.assert_close(k_fused, k_ref_sel, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(v_fused, v_ref_sel, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("bpw", [2, 4, 6])
def test_fused_gather_block_indices_arange_equals_full(relaxed_cuda, bpw):
    """Passing ``block_indices = arange(num_blocks)`` must return the
    same result as passing no block_indices."""
    from glq_vllm.e8_paged_cache import gather_kv_to_paged_fp16_fused

    block_size, num_kv_heads, head_size, num_blocks = 16, 2, 128, 6
    dtype = torch.bfloat16
    cache = E8PagedKVCache.alloc(
        num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size, bpw=bpw,
        device="cuda", dtype=dtype)
    quant = _build_quantizer(relaxed_cuda, bpw)
    n_tokens = num_blocks * block_size
    key, value = _random_kv(seed=11, num_tokens=n_tokens,
                            num_kv_heads=num_kv_heads,
                            head_size=head_size, dtype=dtype, device="cuda")
    slot_mapping = torch.arange(n_tokens, dtype=torch.long, device="cuda")
    write_kv(quant, key, value, cache, slot_mapping)

    arange_idx = torch.arange(num_blocks, dtype=torch.long, device="cuda")
    k_a, v_a = gather_kv_to_paged_fp16_fused(
        quant, cache, block_indices=arange_idx)
    k_n, v_n = gather_kv_to_paged_fp16_fused(quant, cache)
    assert torch.equal(k_a, k_n)
    assert torch.equal(v_a, v_n)


@pytest.mark.parametrize("bpw", [2, 4, 6])
def test_fused_gather_signature_matches_slow_path(relaxed_cuda, bpw):
    """Function signature + output shape contract must match the slow
    path so it can be dropped in as a replacement."""
    from glq_vllm.e8_paged_cache import gather_kv_to_paged_fp16_fused

    block_size, num_kv_heads, head_size, num_blocks = 16, 2, 256, 4
    dtype = torch.bfloat16
    cache = E8PagedKVCache.alloc(
        num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size, bpw=bpw,
        device="cuda", dtype=dtype)
    quant = _build_quantizer(relaxed_cuda, bpw)

    out = gather_kv_to_paged_fp16_fused(quant, cache)
    assert isinstance(out, tuple) and len(out) == 2
    k, v = out
    expected_shape = (num_blocks, block_size, num_kv_heads, head_size)
    assert k.shape == expected_shape
    assert v.shape == expected_shape
    assert k.dtype == dtype
    assert v.dtype == dtype
    assert k.device.type == "cuda"
