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


def _full_spec(*, bpw, block_size=16, head_size=512):
    return E8FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=2,
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
# The compressed cache SHAPE must match the backend's stride order
# --------------------------------------------------------------------------- #
#
# vLLM asserts this itself, in `_reshape_kv_cache`:
#
#     kv_cache_stride_order = group.backend.get_kv_cache_stride_order()
#     assert len(kv_cache_stride_order) == len(kv_cache_shape)
#
# and the surrounding `try` catches only AttributeError/NotImplementedError, so
# a mismatch escapes as an AssertionError that takes EngineCore down at startup.
#
# It broke exactly that way: vLLM's "[6/N] Standardize KV cache layout" refactor
# packed K and V into the content dim, taking the Triton backend's shape from 5-D
# to 4-D `(num_blocks, num_kv_heads, block_size, 2 * head_size)`, while
# `compressed_kv_cache_shape` still returned the 5-D `(num_blocks, 2, block_size,
# num_kv_heads, C)` it had been written to match. Asserting the invariant here
# makes the next such refactor a fast unit-test failure instead of a boot crash.

@pytest.mark.parametrize("bpw", [2, 3, 4, 5, 6, 7])
def test_compressed_shape_rank_matches_backend_stride_order(bpw, monkeypatch):
    from glq_vllm.e8_kv_spec import compressed_kv_cache_shape

    try:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
    except ImportError:
        pytest.skip("TritonAttentionBackend not importable in this vllm")

    # get_kv_cache_stride_order -> get_kv_cache_layout, which otherwise needs a
    # live VllmConfig. The layout only picks the permutation; the RANK — the thing
    # under test — is the same for NHD and HND.
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "NHD")

    shape = compressed_kv_cache_shape(
        num_blocks=128, block_size=16, num_kv_heads=4, head_size=64,
        bpw=bpw, dtype_size=2)
    stride_order = TritonAttentionBackend.get_kv_cache_stride_order()
    assert len(shape) == len(stride_order), (
        f"bpw={bpw}: compressed shape {shape} has rank {len(shape)}, but the "
        f"Triton backend's stride order {stride_order} has {len(stride_order)} "
        f"entries — vLLM asserts these are equal and dies at startup")


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
