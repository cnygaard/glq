"""Phase 5.3 Stage 2c-2b — E8-aware AttentionSpec subclasses.

vLLM's allocator uses ``kv_cache_spec.page_size_bytes`` (via the
``KVCacheTensor.size`` it derives at config time) to decide how many
bytes to ``torch.zeros`` per layer. Standard ``FullAttentionSpec`` /
``SlidingWindowSpec`` report the fp16 page size; if we override
``real_page_size_bytes`` to the compressed size, vLLM allocates a
smaller buffer.

Required co-changes:

1. Backend's ``get_kv_cache_shape`` must return a shape whose element
   count matches ``page_size_bytes / dtype_size`` (so the ``view()``
   call in ``_reshape_kv_cache`` succeeds with the smaller raw
   tensor). We provide ``compressed_kv_cache_shape()`` here for
   callers to wire into the backend monkey-patch.

2. The original ``triton_reshape_and_cache_flash`` writes fp16 K/V
   into the cache assuming the full shape. With our compressed shape,
   that write would either crash or silently corrupt. ``kv_compression``
   skips the call when this path is active — the sidecar holds the
   real K/V.

Bytes-per-group lookup keeps the math:
    bpw=2 -> 4 (int16 idx1 + bf16 scale)
    bpw=3 -> 5 (+ uint8 idx_s1)
    bpw=4 -> 6 (+ int16 idx2)
    bpw=5 -> 7
    bpw=6 -> 8
    bpw=7 -> 9
"""
from __future__ import annotations

from dataclasses import dataclass, replace


# Bytes per group of 8 elements at each bpw rung. Matches the on-card
# layout produced by ``E8PagedKVCache`` (Stage 1 fixed this).
E8_BYTES_PER_GROUP: dict[int, int] = {
    2: 4,   # int16 idx1 + bf16 scale
    3: 5,   # + uint8 idx_s1
    4: 6,   # + int16 idx2
    5: 7,   # + int16 idx2 + uint8 idx_s1
    6: 8,   # + int16 idx3
    7: 9,   # + int16 idx3 + uint8 idx_s1
}


def compressed_page_size_bytes(*, block_size: int, num_kv_heads: int,
                               head_size: int, bpw: int) -> int:
    """Bytes per page (=one block's K plus that block's V) at the given
    bpw. Mirrors ``E8PagedKVCache.bytes_total`` accounting."""
    if head_size % 8 != 0:
        raise ValueError(f"head_size {head_size} must be a multiple of 8")
    if bpw not in E8_BYTES_PER_GROUP:
        raise ValueError(
            f"bpw {bpw} not in E8 ladder; allowed {sorted(E8_BYTES_PER_GROUP)}")
    n_groups = head_size // 8
    return 2 * block_size * num_kv_heads * n_groups * E8_BYTES_PER_GROUP[bpw]


def compressed_kv_cache_shape(num_blocks: int, block_size: int,
                              num_kv_heads: int, head_size: int,
                              bpw: int, dtype_size: int = 2
                              ) -> tuple[int, int, int, int, int]:
    """Shape that ``get_kv_cache_shape`` should report when the spec
    declares the compressed page size.

    Returned as 5-D ``(num_blocks, 2, block_size, num_kv_heads,
    compressed_elems_per_tok_per_head)`` to match the standard
    Triton backend's stride-order shape (5 entries). Only the final
    dim shrinks vs the fp16 layout, so ``kv_cache.unbind(1)`` still
    yields tensors whose first three "real" dims (num_blocks,
    block_size, num_kv_heads) line up.

    Note: ``key_cache.shape[-1]`` now reports the *compressed* elem
    count, not the real ``head_size``. The patched read/write hooks
    must derive ``head_size`` from the input ``key`` / ``q`` tensors
    (which the model produces at full size) instead.
    """
    if head_size % 8 != 0:
        raise ValueError(f"head_size {head_size} must be a multiple of 8")
    n_groups = head_size // 8
    bytes_per_group = E8_BYTES_PER_GROUP[bpw]
    # Total bytes per (token, head): n_groups * bytes_per_group.
    elem_bytes_per_tok_per_head = n_groups * bytes_per_group
    if elem_bytes_per_tok_per_head % dtype_size != 0:
        raise ValueError(
            f"compressed bytes per (tok, head) {elem_bytes_per_tok_per_head} "
            f"not a multiple of dtype_size {dtype_size}")
    compressed_elems_per_tok_per_head = elem_bytes_per_tok_per_head // dtype_size
    return (
        num_blocks, 2, block_size, num_kv_heads,
        compressed_elems_per_tok_per_head,
    )


# --------------------------------------------------------------------------- #
# Spec subclasses
# --------------------------------------------------------------------------- #

def _make_spec_subclasses():
    """Build E8FullAttentionSpec / E8SlidingWindowSpec at import time
    (vllm is an optional import — keep the failure local)."""
    try:
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            SlidingWindowSpec,
        )
    except ImportError:
        return None, None

    @dataclass(frozen=True, kw_only=True)
    class E8FullAttentionSpec(FullAttentionSpec):
        """FullAttentionSpec that declares the compressed page size."""
        bpw: int = 2

        @property
        def real_page_size_bytes(self) -> int:
            return compressed_page_size_bytes(
                block_size=self.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                bpw=self.bpw,
            )

        @classmethod
        def merge(cls, specs):
            # ``FullAttentionSpec.merge`` reconstructs via ``cls(...)`` but
            # only forwards the base-class fields, so our ``bpw`` would
            # default back to 2 and the merged group's page size would
            # silently differ from the input specs at bpw != 2. Restore
            # ``bpw`` after the parent merge (all specs in a group must
            # share the same bpw).
            merged = super().merge(specs)
            bpws = {s.bpw for s in specs}
            if len(bpws) != 1:
                raise ValueError(
                    f"All E8FullAttentionSpec in a KV cache group must share "
                    f"the same bpw; got {sorted(bpws)}.")
            return replace(merged, bpw=bpws.pop())

    @dataclass(frozen=True, kw_only=True)
    class E8SlidingWindowSpec(SlidingWindowSpec):
        """SlidingWindowSpec that declares the compressed page size."""
        bpw: int = 2

        @property
        def real_page_size_bytes(self) -> int:
            return compressed_page_size_bytes(
                block_size=self.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                bpw=self.bpw,
            )

        # ``SlidingWindowSpec`` inherits ``KVCacheSpec.merge`` which uses
        # ``copy.deepcopy(specs[0])`` — that preserves our subclass and
        # all its fields, including ``bpw``. No override needed here.

    return E8FullAttentionSpec, E8SlidingWindowSpec


E8FullAttentionSpec, E8SlidingWindowSpec = _make_spec_subclasses()
