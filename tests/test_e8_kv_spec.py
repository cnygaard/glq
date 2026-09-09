"""Phase 5.3 Stage 2c-2b: regression tests for E8 KV cache specs.

The spec subclasses live in ``glq_vllm/e8_kv_spec.py`` and the merge
contract is load-bearing: vLLM's ``get_uniform_page_size`` checker
asserts that every KV cache group reports the same page size, and the
per-group spec is the result of calling ``layer_specs[0].merge(...)``.

Pre-merge specs come from our ``get_kv_cache_spec`` monkey-patch with
the correct bpw set. The merge path must NOT discard that bpw, or the
merged spec falls back to the dataclass default (bpw=2) and the page
size for full-attention groups differs from sliding-window groups at
any bpw != 2 (vLLM boot then crashes with AssertionError in
``get_uniform_page_size``).
"""
from __future__ import annotations

import pytest

pytest.importorskip("vllm")

from glq_vllm.e8_kv_spec import (  # noqa: E402
    E8FullAttentionSpec,
    E8SlidingWindowSpec,
    compressed_page_size_bytes,
)


pytestmark = pytest.mark.skipif(
    E8FullAttentionSpec is None or E8SlidingWindowSpec is None,
    reason="vLLM kv_cache_interface import failed",
)


@pytest.fixture
def torch_bf16():
    import torch
    return torch.bfloat16


def _full_spec(*, bpw, block_size=16, head_size=512, num_kv_heads=2):
    return E8FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=__import__("torch").bfloat16,
        sliding_window=None,
        attention_chunk_size=None,
        bpw=bpw,
    )


def _sliding_spec(*, bpw, block_size=32, head_size=256):
    return E8SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=head_size,
        dtype=__import__("torch").bfloat16,
        sliding_window=1024,
        bpw=bpw,
    )


@pytest.mark.parametrize("bpw", [2, 3, 4, 5, 6, 7])
def test_full_attention_spec_page_size(bpw):
    spec = _full_spec(bpw=bpw)
    expected = compressed_page_size_bytes(
        block_size=spec.block_size, num_kv_heads=spec.num_kv_heads,
        head_size=spec.head_size, bpw=bpw,
    )
    assert spec.real_page_size_bytes == expected
    assert spec.page_size_bytes == expected


@pytest.mark.parametrize("bpw", [2, 3, 4, 5, 6, 7])
def test_sliding_window_spec_page_size(bpw):
    spec = _sliding_spec(bpw=bpw)
    expected = compressed_page_size_bytes(
        block_size=spec.block_size, num_kv_heads=spec.num_kv_heads,
        head_size=spec.head_size, bpw=bpw,
    )
    assert spec.real_page_size_bytes == expected
    assert spec.page_size_bytes == expected


@pytest.mark.parametrize("bpw", [2, 3, 4, 5, 6, 7])
def test_full_attention_merge_preserves_bpw(bpw):
    """Regression for the bpw != 2 boot crash: ``FullAttentionSpec.merge``
    reconstructs via ``cls(...)`` and would lose ``bpw`` (default 2)
    without our override. After merge the page size must still reflect
    the configured bpw."""
    s1 = _full_spec(bpw=bpw)
    s2 = _full_spec(bpw=bpw)
    merged = type(s1).merge([s1, s2])
    assert isinstance(merged, E8FullAttentionSpec)
    assert merged.bpw == bpw, (
        f"E8FullAttentionSpec.merge lost bpw: expected {bpw}, "
        f"got {merged.bpw} (this is the 4 bpw vLLM boot crash)"
    )
    assert merged.page_size_bytes == s1.page_size_bytes


@pytest.mark.parametrize("bpw", [2, 3, 4, 5, 6, 7])
def test_sliding_window_merge_preserves_bpw(bpw):
    s1 = _sliding_spec(bpw=bpw)
    s2 = _sliding_spec(bpw=bpw)
    merged = type(s1).merge([s1, s2])
    assert isinstance(merged, E8SlidingWindowSpec)
    assert merged.bpw == bpw
    assert merged.page_size_bytes == s1.page_size_bytes


def test_full_attention_merge_mixed_bpw_rejected():
    """If layers in the same group have different bpw, merge must
    raise — otherwise we'd silently pick one and the per-layer
    dequantizer would corrupt the cache."""
    s1 = _full_spec(bpw=4)
    s2 = _full_spec(bpw=6)
    with pytest.raises(ValueError, match="same bpw"):
        type(s1).merge([s1, s2])


@pytest.mark.parametrize("bpw", [3, 4, 5, 6, 7])
def test_full_and_sliding_page_sizes_align_after_unification(bpw):
    """Mirrors the vLLM path: ``unify_kv_cache_spec_page_size`` doubles
    the sliding spec's block_size so its page_size matches the full
    spec's. After that, both group's merged specs must report the
    same page_size — otherwise ``get_uniform_page_size`` asserts."""
    full = _full_spec(bpw=bpw, block_size=16, head_size=512)
    # Sliding head_size is half the full head_size on Gemma-4, so
    # vLLM doubles its block_size to equalize the pre-merge page size.
    sliding = _sliding_spec(bpw=bpw, block_size=32, head_size=256)
    assert full.page_size_bytes == sliding.page_size_bytes, (
        f"pre-merge page sizes diverged at bpw={bpw}: "
        f"full={full.page_size_bytes} sliding={sliding.page_size_bytes}"
    )
    full_merged = type(full).merge([full])
    sliding_merged = type(sliding).merge([sliding])
    assert full_merged.page_size_bytes == sliding_merged.page_size_bytes, (
        f"post-merge page sizes diverged at bpw={bpw}: "
        f"full={full_merged.page_size_bytes} "
        f"sliding={sliding_merged.page_size_bytes}"
    )


# --------------------------------------------------------------------------- #
# The compressed cache SHAPE must have the rank vLLM allocates
# --------------------------------------------------------------------------- #
#
# How this broke: vLLM's "[6/N] Standardize KV cache layout" refactor packed K and V
# into the content dimension, taking the layer shape from 5-D `(num_blocks, 2,
# block_size, num_kv_heads, C)` to 4-D `(num_blocks, num_kv_heads, block_size,
# 2 * head_size)`, while `compressed_kv_cache_shape` still returned the 5-D form it had
# been written against. A rank mismatch is not a subtly wrong number: vLLM asserts on it
# and EngineCore goes down at startup.
#
# The same refactor moved WHERE the shape comes from. Before 0.28 the backend answered
# `get_kv_cache_stride_order()` and `_reshape_kv_cache` asserted its length against the
# shape; in 0.28 that classmethod is gone, backends expose no shape at all, and the spec
# owns the layout. So the invariant is asserted against
# `compute_layer_kv_cache_shape_bytes` — the function vLLM's own allocator calls — which
# is both version-correct and a stronger statement than the stride order was.

@pytest.mark.parametrize("bpw", [2, 3, 4, 5, 6, 7])
def test_compressed_shape_rank_matches_what_vllm_allocates(bpw, monkeypatch):
    """The rank of our compressed shape must equal the rank vLLM allocates for the layer.

    This is the invariant the 4-D change exists for: the pre-0.28 layout was 5-D
    ``(blocks, 2, block_size, heads, elems)`` and 0.28 packs K and V into the content
    dimension, giving 4-D. A rank mismatch is not a wrong number somewhere — vLLM
    asserts on it and EngineCore dies at startup.

    Checked against ``compute_layer_kv_cache_shape_bytes``, which is the function vLLM's
    own allocator calls. An earlier version of this test asked
    ``TritonAttentionBackend.get_kv_cache_stride_order()``; that classmethod was removed in
    0.28's layout standardization and the backend no longer exposes any shape at all — the
    spec owns it now. Asserting against the allocator is both version-correct and a
    stronger statement than the stride order was.
    """
    from glq_vllm.e8_kv_spec import compressed_kv_cache_shape

    try:
        from vllm.v1.kv_cache_interface import compute_layer_kv_cache_shape_bytes
    except ImportError:
        pytest.skip("compute_layer_kv_cache_shape_bytes not in this vllm")

    # The layout only picks the permutation; the RANK under test is the same either way.
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "NHD")

    spec = _full_spec(bpw=bpw, num_kv_heads=4, head_size=64, block_size=16)
    ours = compressed_kv_cache_shape(
        num_blocks=128, block_size=16, num_kv_heads=4, head_size=64,
        bpw=bpw, dtype_size=2)
    allocated = compute_layer_kv_cache_shape_bytes(spec, 128)
    assert len(ours) == len(allocated), (
        f"bpw={bpw}: compressed shape {ours} has rank {len(ours)}, but vLLM allocates "
        f"rank {len(allocated)} ({allocated}) for the same layer — vLLM asserts these "
        f"match and EngineCore dies at startup")


@pytest.mark.parametrize("bpw", [2, 3, 4, 5, 6, 7])
def test_compressed_shape_agrees_with_the_declared_page_size(bpw):
    """The shape and the page size are two views of one allocation; if they
    disagree, vLLM reserves a buffer that the E8 layout cannot address."""
    from glq_vllm.e8_kv_spec import compressed_kv_cache_shape

    num_blocks, block_size, num_kv_heads, head_size, dtype_size = 128, 16, 4, 64, 2
    shape = compressed_kv_cache_shape(
        num_blocks=num_blocks, block_size=block_size, num_kv_heads=num_kv_heads,
        head_size=head_size, bpw=bpw, dtype_size=dtype_size)

    elems = 1
    for d in shape:
        elems *= d
    page_bytes = compressed_page_size_bytes(
        block_size=block_size, num_kv_heads=num_kv_heads,
        head_size=head_size, bpw=bpw)
    assert elems * dtype_size == num_blocks * page_bytes, (
        f"bpw={bpw}: shape {shape} is {elems * dtype_size} bytes but the spec "
        f"declares {num_blocks * page_bytes}")


@pytest.mark.parametrize("bpw", [2, 4, 7])
def test_k_and_v_are_packed_in_the_content_dim(bpw):
    """vLLM splits the content dim to get K and V — `kv_cache.transpose(1, 2)
    .split(width, dim=-1)` — so the last dim must be exactly twice the per-side
    compressed width, and the middle dims must be (num_kv_heads, block_size)."""
    from glq_vllm.e8_kv_spec import compressed_kv_cache_shape

    num_blocks, block_size, num_kv_heads, head_size = 128, 16, 4, 64
    shape = compressed_kv_cache_shape(
        num_blocks=num_blocks, block_size=block_size, num_kv_heads=num_kv_heads,
        head_size=head_size, bpw=bpw, dtype_size=2)
    assert shape[0] == num_blocks
    assert shape[1] == num_kv_heads, "vLLM's layout is (B, H, N, 2*width)"
    assert shape[2] == block_size
    assert shape[3] % 2 == 0, "K and V share the content dim, so it must be even"


# --------------------------------------------------------------------------- #
# The K/V split: vLLM splits at self.head_size, which is the DENSE width
# --------------------------------------------------------------------------- #
#
# `TritonAttentionImpl.forward` gets its per-side views with
#
#     key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
#
# On a compressed content dim that returns ONE chunk (head_size 64 splitting a
# 48-wide dim), and the two-way unpack raises
# "ValueError: not enough values to unpack (expected 2, got 1)".
#
# GLQ hands the backend an impl subclass whose `head_size` is the compressed
# side width, because in this vLLM that attribute is used for nothing else in
# the class. These tests pin both halves of that bet: the arithmetic, and the
# claim that splitting at it actually yields two usable sides.

@pytest.mark.parametrize("bpw", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("head_size", [64, 128, 256])
def test_split_at_the_compressed_width_yields_two_sides(bpw, head_size):
    import torch

    from glq_vllm.e8_kv_spec import (compressed_kv_cache_shape,
                                     compressed_side_width)

    num_blocks, block_size, num_kv_heads = 4, 16, 2
    shape = compressed_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, bpw=bpw, dtype_size=2)
    kv_cache = torch.zeros(shape, dtype=torch.bfloat16)
    width = compressed_side_width(head_size, bpw, dtype_size=2)

    # Verbatim the expression from triton_attn.py, with the width GLQ supplies.
    parts = kv_cache.transpose(1, 2).split(width, dim=-1)
    assert len(parts) == 2, (
        f"bpw={bpw} hs={head_size}: split at {width} gave {len(parts)} chunk(s) "
        f"from a {shape[-1]}-wide content dim")
    key_cache, value_cache = parts
    assert key_cache.shape == value_cache.shape
    # (num_blocks, block_size, num_kv_heads, C) — what the sidecar expects.
    assert key_cache.shape == (num_blocks, block_size, num_kv_heads, width)


@pytest.mark.parametrize("bpw", [2, 3, 4, 5, 6, 7])
def test_splitting_at_the_dense_head_size_never_gives_two_correct_sides(bpw):
    """The bug, pinned — and it has two shapes depending on bpw.

    Below 2*C <= head_size (bpw <= 6 at head_size 64) the split returns a single
    chunk and the unpack raises "not enough values to unpack", which is the
    engine-start ValueError. At bpw 7 the compressed dim is 72, *wider* than the
    dense 64, so the same split returns two chunks — of 64 and 8. That one does
    not raise; it would hand the kernel mis-sized views and corrupt attention
    silently. Either way the dense width is wrong, which is the invariant here.
    """
    import torch

    from glq_vllm.e8_kv_spec import (compressed_kv_cache_shape,
                                     compressed_side_width)

    head_size = 64
    shape = compressed_kv_cache_shape(4, 16, 2, head_size, bpw=bpw, dtype_size=2)
    kv_cache = torch.zeros(shape, dtype=torch.bfloat16)
    width = compressed_side_width(head_size, bpw, dtype_size=2)

    parts = kv_cache.transpose(1, 2).split(head_size, dim=-1)
    two_correct_sides = (len(parts) == 2
                         and all(p.shape[-1] == width for p in parts))
    assert not two_correct_sides, (
        f"bpw={bpw}: splitting at the dense head_size={head_size} produced two "
        f"correct {width}-wide sides, so the head_size override is unnecessary "
        f"— check whether vLLM changed the split")


def test_the_impl_hook_reports_the_compressed_width():
    """The mechanism: the class vLLM gets from `get_impl_cls` must report the
    compressed side width as head_size, and keep the real one for the kernel."""
    import glq_vllm.kv_compression as kvc
    from glq_vllm.e8_kv_spec import compressed_side_width

    try:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
    except ImportError:
        pytest.skip("TritonAttentionImpl not importable in this vllm")

    head_size, bpw = 64, 4
    impl_cls = kvc._make_compressed_impl(TritonAttentionImpl, bpw=bpw)
    impl = impl_cls(num_heads=8, head_size=head_size, scale=1.0, num_kv_heads=2,
                    alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto")

    assert impl.head_size == compressed_side_width(head_size, bpw) == 24
    assert impl.glq_real_head_size == head_size, (
        "the true head size must survive — the E8 kernel needs it")
