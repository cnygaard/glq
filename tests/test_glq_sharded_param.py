"""GLQShardedParameter lazy-alloc + zero-byte param.data unit tests.

Covers Phase A (sentinel=True empty(0) per-shard _shard_data) and
Phase B (param.data is a zero-byte placeholder instead of the full
concatenated dummy) of the 31B-GLQ load-footprint fix.
"""

import os

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import pytest
import torch

try:
    from glq_vllm.linear_method import GLQShardedParameter, _glq_pad
    _HAS_VLLM = True
except (ImportError, ModuleNotFoundError):
    _HAS_VLLM = False

requires_vllm = pytest.mark.skipif(not _HAS_VLLM, reason="vllm not installed")


@pytest.fixture(scope="module", autouse=True)
def _init_vllm_tp():
    """BasevLLMParameter touches the TP group + current VllmConfig at
    construction; bring up a single-rank dummy world and a default
    config context so we don't need a full engine."""
    if not _HAS_VLLM:
        yield
        return
    from vllm.distributed import (
        init_distributed_environment,
        ensure_model_parallel_initialized,
    )
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        if not torch.distributed.is_initialized():
            init_distributed_environment(
                world_size=1, rank=0,
                distributed_init_method="tcp://127.0.0.1:29512",
                local_rank=0, backend="gloo",
            )
        ensure_model_parallel_initialized(1, 1)
        yield


def _noop_loader(*args, **kwargs):
    pass


@requires_vllm
def test_param_data_is_zero_bytes():
    """Phase B: param.data must be empty(0) — no full concat dummy alloc."""
    shards = [4096, 1024, 1024]
    p = GLQShardedParameter(
        shards, 256, torch.int16, weight_loader=_noop_loader)
    assert p.data.numel() == 0, (
        f"param.data leaks {p.data.numel()} elems "
        f"({p.data.numel() * p.data.element_size()} bytes); "
        "Phase B fix means it should be empty(0).")
    assert p.data.dtype == torch.int16


@requires_vllm
def test_default_full_shard_alloc():
    """Default (sentinel=False) still allocates real per-shard storage."""
    shards = [4096, 1024, 1024]
    n_blocks = 32
    p = GLQShardedParameter(
        shards, n_blocks, torch.int16, weight_loader=_noop_loader)
    assert len(p._shard_data) == 3
    for sz, sd in zip(shards, p._shard_data):
        m_pad = _glq_pad(sz)
        assert sd.shape == (m_pad, n_blocks), (
            f"expected ({m_pad}, {n_blocks}); got {tuple(sd.shape)}")
        assert sd.dtype == torch.int16


@requires_vllm
def test_sentinel_shard_alloc():
    """Phase A: sentinel=True gives empty(0) per-shard buffers."""
    shards = [4096, 1024, 1024]
    n_blocks = 32
    p = GLQShardedParameter(
        shards, n_blocks, torch.int16,
        weight_loader=_noop_loader, sentinel=True)
    assert len(p._shard_data) == 3
    for sd in p._shard_data:
        assert sd.numel() == 0, (
            f"sentinel shard leaks {sd.numel()} elems")
        assert sd.dtype == torch.int16


@requires_vllm
def test_sentinel_loader_resizes_on_first_store():
    """Loader path resizes empty sentinel to match loaded_weight."""
    shards = [4096, 1024, 1024]
    n_blocks = 32
    p = GLQShardedParameter(
        shards, n_blocks, torch.int16,
        weight_loader=_noop_loader, sentinel=True)

    # Simulate a checkpoint Qidxs3 for shard 'q' with non-pow2 shape.
    loaded = torch.full((4096, n_blocks), 7, dtype=torch.int16)
    p.load_qkv_weight(loaded, shard_id="q")

    assert p.get_shard(0).shape == loaded.shape, (
        f"shard 0 should auto-resize to {loaded.shape}; "
        f"got {tuple(p.get_shard(0).shape)}")
    assert torch.equal(p.get_shard(0), loaded)
    # Other shards remain empty (didn't get a load call).
    assert p.get_shard(1).numel() == 0
    assert p.get_shard(2).numel() == 0


@requires_vllm
def test_sentinel_memory_savings():
    """Sentinel allocates zero bytes for per-shard data; default allocates real."""
    shards = [5376, 1280, 1280]  # gemma-4-31B QKV shapes
    n_blocks = 5376 // 8

    default = GLQShardedParameter(
        shards, n_blocks, torch.int16, weight_loader=_noop_loader)
    sentinel = GLQShardedParameter(
        shards, n_blocks, torch.int16,
        weight_loader=_noop_loader, sentinel=True)

    default_bytes = sum(sd.numel() * sd.element_size()
                        for sd in default._shard_data)
    sentinel_bytes = sum(sd.numel() * sd.element_size()
                         for sd in sentinel._shard_data)

    assert sentinel_bytes == 0
    # Default should be at least 8 MB for these shapes (4096*672*2 = 5.5 MB
    # for the q shard alone). Catches accidental sentinel-by-default.
    assert default_bytes > 4 * 1024 * 1024, (
        f"default alloc unexpectedly small ({default_bytes} bytes)")


@requires_vllm
def test_to_device_moves_shards():
    """Phase B safety: .to(device) propagates to _shard_data even with empty data."""
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    shards = [1024, 256, 256]
    p = GLQShardedParameter(
        shards, 32, torch.int16, weight_loader=_noop_loader)
    p = p.cuda()
    for sd in p._shard_data:
        assert sd.is_cuda, "_shard_data not moved to cuda after .cuda()"


@requires_vllm
def test_sentinel_to_device_empty_safe():
    """Empty(0) shards must survive .cuda() without crashing."""
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    shards = [1024, 256, 256]
    p = GLQShardedParameter(
        shards, 32, torch.int16,
        weight_loader=_noop_loader, sentinel=True)
    p = p.cuda()
    for sd in p._shard_data:
        assert sd.is_cuda
        assert sd.numel() == 0


@requires_vllm
def test_dtype_preserved_for_all_shapes():
    """Phase B empty param.data must use the requested dtype."""
    for dtype in [torch.int16, torch.float16, torch.float32]:
        p = GLQShardedParameter(
            [1024, 512], 64, dtype, weight_loader=_noop_loader)
        assert p.data.dtype == dtype, (
            f"param.data dtype {p.data.dtype} != requested {dtype}")
        assert p.data.numel() == 0


@requires_vllm
def test_scalar_shard_default_unchanged():
    """inv_resid_scale / Wscale (inner_dim=0) keeps scalar per-shard alloc."""
    shards = [1, 1, 1]
    p = GLQShardedParameter(
        shards, 0, torch.float32, weight_loader=_noop_loader)
    assert len(p._shard_data) == 3
    for sd in p._shard_data:
        assert sd.shape == (), f"expected scalar; got {tuple(sd.shape)}"
        assert sd.dtype == torch.float32


@requires_vllm
def test_vector_shard_default_unchanged():
    """SU (inner_dim=-1) keeps per-shard 1D vector at m_pad length."""
    shards = [4096, 1024, 1024]
    p = GLQShardedParameter(
        shards, -1, torch.float16, weight_loader=_noop_loader)
    for sz, sd in zip(shards, p._shard_data):
        m_pad = _glq_pad(sz)
        assert sd.shape == (m_pad,), (
            f"SU shard expected ({m_pad},); got {tuple(sd.shape)}")
