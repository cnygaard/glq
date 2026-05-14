"""Phase 5.3 Stage 1: validate the E8PagedKVCache layout.

The cache class lives in ``glq_vllm/e8_paged_cache.py`` and is the
Python prototype for the CUDA work in Stage 3. These tests pin the
correctness contract that the CUDA kernels will have to honour:

* Bit-exact equivalence with ``E8KVQuantizer.quantize`` /
  ``E8KVQuantizer.dequantize`` end-to-end.
* Storage budget per (token, head) matches the bpw recipe.
* Non-contiguous slot scattering (writes interleaved across blocks) is
  preserved by the gather path.
"""
from __future__ import annotations

import pytest
import torch

from glq.codebook_relaxed import E8RelaxedCodebook
from glq.kv_e8 import E8KVQuantizer
from glq_vllm.e8_paged_cache import (
    E8PagedKVCache,
    _BPW_RECIPE,
    gather_kv,
    write_kv,
)


@pytest.fixture(scope="module")
def relaxed():
    return E8RelaxedCodebook(device="cpu", verbose=False)


def _build_quantizer(cb, bpw: int):
    n_p, n_s = _BPW_RECIPE[bpw]
    small = cb.make_small() if n_s > 0 else None
    return E8KVQuantizer(cb, n_stages=n_p,
                        secondary_codebook=small, secondary_stages=n_s)


def _random_kv(seed: int, num_tokens: int, num_kv_heads: int, head_size: int,
               dtype=torch.float16):
    g = torch.Generator().manual_seed(seed)
    key = torch.randn(num_tokens, num_kv_heads, head_size,
                      generator=g, dtype=torch.float32).to(dtype)
    value = torch.randn(num_tokens, num_kv_heads, head_size,
                        generator=g, dtype=torch.float32).to(dtype)
    return key, value


@pytest.mark.parametrize("bpw", [2, 3, 4, 5, 6, 7])
def test_roundtrip_matches_quantizer(relaxed, bpw):
    """Storing K/V via the paged cache and gathering them back must
    reproduce ``quantizer.dequantize(quantizer.quantize(x))`` *exactly*
    — same math, just routed through the new storage layout."""
    block_size = 16
    num_kv_heads = 2
    head_size = 128
    num_blocks = 8

    cache = E8PagedKVCache.alloc(
        num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size, bpw=bpw,
        device="cpu", dtype=torch.float16)
    quant = _build_quantizer(relaxed, bpw)

    n_tokens = 40
    key, value = _random_kv(seed=bpw, num_tokens=n_tokens,
                            num_kv_heads=num_kv_heads, head_size=head_size)
    # Pack tokens into slots 0..39 (spans 3 blocks contiguously).
    slot_mapping = torch.arange(n_tokens, dtype=torch.long)
    write_kv(quant, key, value, cache, slot_mapping)
    k_out, v_out = gather_kv(quant, cache, slot_mapping)

    # Reference: straight quantize -> dequantize, no cache.
    k_ref = quant.dequantize(quant.quantize(key))
    v_ref = quant.dequantize(quant.quantize(value))

    assert torch.equal(k_out, k_ref), \
        f"K mismatch at bpw={bpw}: max delta={(k_out-k_ref).abs().max().item()}"
    assert torch.equal(v_out, v_ref), \
        f"V mismatch at bpw={bpw}: max delta={(v_out-v_ref).abs().max().item()}"


def test_non_contiguous_slot_scatter(relaxed):
    """Tokens written into scattered slots must come back individually
    correct — the cache must not assume a contiguous slot_mapping."""
    block_size = 16
    num_kv_heads = 2
    head_size = 64
    num_blocks = 4
    bpw = 3

    cache = E8PagedKVCache.alloc(
        num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size, bpw=bpw, device="cpu")
    quant = _build_quantizer(relaxed, bpw)

    n_tokens = 12
    key, value = _random_kv(seed=42, num_tokens=n_tokens,
                            num_kv_heads=num_kv_heads, head_size=head_size)
    # Interleave: pick slots that span multiple blocks in non-monotonic order.
    g = torch.Generator().manual_seed(0)
    slot_mapping = torch.randperm(
        num_blocks * block_size, generator=g)[:n_tokens].long()

    write_kv(quant, key, value, cache, slot_mapping)
    k_out, v_out = gather_kv(quant, cache, slot_mapping)
    k_ref = quant.dequantize(quant.quantize(key))
    v_ref = quant.dequantize(quant.quantize(value))
    assert torch.equal(k_out, k_ref)
    assert torch.equal(v_out, v_ref)


@pytest.mark.parametrize("bpw,expected_bytes_per_tok_per_head", [
    # head_size=128 → n_groups=16. Costs per group:
    #   idx1/idx2/idx3 = 2 bytes (int16);
    #   idx_s1 = 1 byte (uint8);
    #   scale  = 2 bytes (fp16).
    # Per (token, head) we store n_groups entries of each component.
    (2, (2 + 2) * 16),            # idx1 + scale         = 64 B
    (3, (2 + 1 + 2) * 16),        # idx1 + idx_s1 + scale = 80 B
    (4, (2 + 2 + 2) * 16),        # idx1 + idx2 + scale   = 96 B
    (5, (2 + 2 + 1 + 2) * 16),    # idx1 + idx2 + idx_s1 + scale = 112 B
    (6, (2 + 2 + 2 + 2) * 16),    # idx1 + idx2 + idx3 + scale   = 128 B
    (7, (2 + 2 + 2 + 1 + 2) * 16),# + idx_s1                      = 144 B
])
def test_bytes_per_token_per_head_matches_recipe(bpw,
                                                  expected_bytes_per_tok_per_head):
    """Storage cost per (token, head) must equal the theoretical sum of
    the recipe's components. Catches layout drift before kernel work."""
    cache = E8PagedKVCache.alloc(
        num_blocks=2, block_size=16, num_kv_heads=2, head_size=128,
        bpw=bpw, device="cpu", dtype=torch.float16)
    assert cache.bytes_per_token_per_head() == expected_bytes_per_tok_per_head


def test_total_bytes_strictly_smaller_than_fp16():
    """Allocated buffer must be smaller than the fp16 baseline at every
    bpw rung — otherwise the whole exercise is for nothing."""
    num_blocks, block_size, num_kv_heads, head_size = 16, 16, 4, 128
    fp16_total = (num_blocks * block_size * num_kv_heads * head_size
                  * 2          # bytes per fp16 element
                  * 2)         # K and V
    for bpw in sorted(_BPW_RECIPE):
        cache = E8PagedKVCache.alloc(
            num_blocks=num_blocks, block_size=block_size,
            num_kv_heads=num_kv_heads, head_size=head_size, bpw=bpw,
            device="cpu", dtype=torch.float16)
        ratio = cache.bytes_total() / fp16_total
        expected = {2: 0.25, 3: 0.3125, 4: 0.375,
                    5: 0.4375, 6: 0.5, 7: 0.5625}[bpw]
        assert abs(ratio - expected) < 1e-6, \
            f"bpw={bpw}: ratio {ratio:.4f} != expected {expected}"
        assert ratio < 1.0, f"bpw={bpw} not smaller than fp16"


def test_recipe_dtype_correctness():
    """Each per-group buffer must use the smallest sufficient dtype.
    Catches accidental int16 storage of 8-bit indices."""
    cache = E8PagedKVCache.alloc(
        num_blocks=2, block_size=16, num_kv_heads=2, head_size=128,
        bpw=3, device="cpu")
    assert cache.k_idx1.dtype == torch.int16
    assert cache.k_idx_s1.dtype == torch.uint8        # 8-bit secondary
    assert cache.k_scale.dtype == torch.float16
    assert cache.k_idx2 is None
    assert cache.k_idx3 is None
