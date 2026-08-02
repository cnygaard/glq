"""Tests for GLQ vLLM plugin — registration, correctness, VRAM, throughput."""

import os
import time
import types

# vLLM v1 serializes model state between processes; GLQ params have
# function references (weight_loader) that aren't msgpack-serializable.
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import pytest
import torch

try:
    from vllm.model_executor.layers.quantization import get_quantization_config
    _HAS_VLLM = True
except (ImportError, ModuleNotFoundError):
    _HAS_VLLM = False

try:
    import transformers
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

requires_vllm = pytest.mark.skipif(not _HAS_VLLM, reason="vllm not installed")
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required"
)

MODEL_ID = "xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw"
TOKENIZER_ID = "HuggingFaceTB/SmolLM3-3B"


def _init_tp_once():
    """BasevLLMParameter asserts a TP group; a world-size-1 gloo group suffices. vLLM 0.25
    additionally requires a current-config context around both the parallel-state init and
    any CustomOp/parameter construction.

    Needed by ANY test that constructs a GLQShardedParameter — i.e. every fused (qkv /
    gate_up) layer, even a pure sizing check that never touches a GPU."""
    from vllm.config import VllmConfig, set_current_vllm_config
    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    import vllm.distributed.parallel_state as ps
    from vllm.distributed import (init_distributed_environment,
                                  initialize_model_parallel)
    if not ps.model_parallel_is_initialized():
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, backend="gloo",
            distributed_init_method="tcp://127.0.0.1:29511")
        initialize_model_parallel(1, 1)


# ── Test 1: Config registration (no GPU needed) ────────────────────────

@requires_vllm
def test_glq_config_registration():
    """GLQvLLMConfig should be discoverable after import glq_vllm."""
    import glq_vllm  # noqa: F401
    from glq_vllm.config import GLQvLLMConfig
    assert get_quantization_config("glq") == GLQvLLMConfig


# ── Test 2: Config parsing (no GPU needed) ──────────────────────────────

@requires_vllm
@requires_gpu
def test_fused_kernel_custom_ops_registered():
    """All 9 GLQ pybind entrypoints are registered as torch.ops.glq.*.

    The 6 dequant + RHT kernels were registered first; the 3 fused-linear /
    fused-MoE entrypoints were added later so vLLM's torch.compile (mode>=3)
    can trace through them as opaque ops. Without this registration,
    dynamo crashes with ``Attempted to call function marked as skipped`` on
    the pybind11 functions.
    """
    import glq_vllm.custom_ops
    glq_vllm.custom_ops._ensure_registered()
    expected = [
        "dequant_matvec", "dequant_matmul",
        "dequant_matvec_packed", "dequant_matmul_packed",
        "input_rht", "output_rht",
        "fused_linear", "fused_linear_block_diag", "fused_moe_block_diag",
        # v0.3.3: Triton-fallback RHT wrappers (n_pad > 16384).
        "input_rht_triton", "output_rht_triton",
        # v0.3.3: KV gather/scatter Triton wrappers (preparatory for
        # full-graph mode; the attention region is already a piecewise
        # graph-break boundary today).
        "gather_kv_paged_dequant", "scatter_kv_paged_quant",
    ]
    missing = [op for op in expected if not hasattr(torch.ops.glq, op)]
    assert not missing, f"unregistered: {missing}"


@requires_vllm
def test_fused_linear_fake_shape():
    """Fake implementation must match the live kernel's output shape so
    torch.compile shape inference doesn't disagree with runtime.

    Uses meta-device tensors so the fake impl runs without needing the
    real kernel — exercises only the shape inference path.
    """
    import glq_vllm.custom_ops
    glq_vllm.custom_ops._ensure_registered()
    if not hasattr(torch.ops.glq, "fused_linear"):
        pytest.skip("fused_linear not registered (no CUDA ext loaded)")

    B, in_features, out_features = 2, 1024, 768
    n_pad, m_pad = 1024, 1024
    meta = torch.device("meta")
    x = torch.empty(B, in_features, dtype=torch.float16, device=meta)
    sv = torch.empty(n_pad, dtype=torch.float16, device=meta)
    su = torch.empty(m_pad, dtype=torch.float16, device=meta)
    qidxs = torch.empty(m_pad, n_pad // 8, dtype=torch.int16, device=meta)
    cb = torch.empty(65536, 8, dtype=torch.float16, device=meta)
    empty_i16 = torch.empty(0, dtype=torch.int16, device=meta)
    empty_f16 = torch.empty(0, dtype=torch.float16, device=meta)

    fy = torch.ops.glq.fused_linear(
        x, sv, su, qidxs, cb, 1.0,
        in_features, out_features, n_pad, m_pad, 10, 10,
        empty_i16, empty_f16, 0.0,
        empty_i16, empty_f16, 0.0,
        empty_i16, empty_f16, 0.0,
    )
    assert fy.shape == (B, out_features), f"got {tuple(fy.shape)}"
    assert fy.dtype == torch.float16
    assert fy.device.type == "meta"


@requires_vllm
def test_fused_linear_e8p_fake_shape():
    """The e8p N-stage fused op carries 26 args (stage-0 + 3 residual stages,
    each E8P+E81B, plus 3 cumulative scales). This pins the schema⇆fake arg
    count/order: an off-by-one between the ``define`` string and
    ``_fused_linear_e8p_fake`` would raise here rather than at serve time.
    Meta-device, so it runs without the CUDA ext."""
    import glq_vllm.custom_ops
    glq_vllm.custom_ops._ensure_registered()
    if not hasattr(torch.ops.glq, "fused_linear_e8p"):
        pytest.skip("fused_linear_e8p not registered (no CUDA ext loaded)")

    B, in_features, out_features = 2, 1024, 768
    n_pad, m_pad = 1024, 1024
    meta = torch.device("meta")
    x = torch.empty(B, in_features, dtype=torch.float16, device=meta)
    sv = torch.empty(n_pad, dtype=torch.float16, device=meta)
    su = torch.empty(m_pad, dtype=torch.float16, device=meta)
    # stage-0 E8P weights: (m_pad//16, n_pad//64, 8, 4) int64
    qidxs = torch.empty(m_pad // 16, n_pad // 64, 8, 4, dtype=torch.int64, device=meta)
    cb_abs = torch.empty(256, dtype=torch.int32, device=meta)  # grid_packed_abs
    empty_i64 = torch.empty(0, dtype=torch.int64, device=meta)
    empty_f16 = torch.empty(0, dtype=torch.float16, device=meta)
    empty_i32 = torch.empty(0, dtype=torch.int32, device=meta)
    blk_n = torch.tensor([n_pad], dtype=torch.int64, device=meta)
    blk_m = torch.tensor([m_pad], dtype=torch.int64, device=meta)

    fy = torch.ops.glq.fused_linear_e8p(
        x, sv, su, qidxs,
        empty_i64, empty_i64,   # qidxs2_e8p, qidxs2_e81b
        empty_i64, empty_i64,   # qidxs3_e8p, qidxs3_e81b
        empty_i64, empty_i64,   # qidxs4_e8p, qidxs4_e81b
        cb_abs, empty_f16,      # codebook_abs, e81b_codebook
        blk_n, blk_m, empty_i32, empty_i32,
        1.0, 0.0, 0.0, 0.0,     # wscale, inv_resid_scale, inv_resid_scale2/3
        in_features, out_features, n_pad, m_pad, 10, 10,
    )
    assert fy.shape == (B, out_features), f"got {tuple(fy.shape)}"
    assert fy.dtype == torch.float16
    assert fy.device.type == "meta"


@requires_vllm
def test_fused_linear_trellis_fake_shape():
    """Pins the trellis schema⇆fake arg count/order (14 args). An off-by-one between the
    ``define`` string and ``_fused_linear_trellis_fake`` would raise here rather than at
    serve time. Meta-device, so it runs without the CUDA ext."""
    import glq_vllm.custom_ops
    glq_vllm.custom_ops._ensure_registered()
    if not hasattr(torch.ops.glq, "fused_linear_trellis"):
        pytest.skip("fused_linear_trellis not registered (no CUDA ext loaded)")

    B, in_features, out_features = 2, 1024, 768
    n_pad, m_pad = in_features, out_features        # trellis never pads
    meta = torch.device("meta")
    x = torch.empty(B, in_features, dtype=torch.float16, device=meta)
    sv = torch.empty(n_pad, dtype=torch.float16, device=meta)
    su = torch.empty(m_pad, dtype=torch.float16, device=meta)
    packed = torch.empty((m_pad // 16) * (n_pad // 16), 32, dtype=torch.int16, device=meta)
    tlut = torch.empty(512, 2, dtype=torch.float16, device=meta)
    empty_i32 = torch.empty(0, dtype=torch.int32, device=meta)
    blk_n = torch.tensor([n_pad], dtype=torch.int64, device=meta)
    blk_m = torch.tensor([m_pad], dtype=torch.int64, device=meta)

    fy = torch.ops.glq.fused_linear_trellis(
        x, sv, su, packed, tlut,
        blk_n, blk_m, empty_i32, empty_i32,
        1.0, in_features, out_features, n_pad, m_pad,
    )
    assert fy.shape == (B, out_features), f"got {tuple(fy.shape)}"
    assert fy.dtype == torch.float16
    assert fy.device.type == "meta"


@requires_vllm
def test_fused_linear_trellis_3inst_fake_shape():
    """Pins the 3INST (no-tlut) schema⇆fake arg count/order (13 args — the HYB schema minus
    tlut). An off-by-one between the ``define`` string and
    ``_fused_linear_trellis_3inst_fake`` would raise here rather than at serve time."""
    import glq_vllm.custom_ops
    glq_vllm.custom_ops._ensure_registered()
    if not hasattr(torch.ops.glq, "fused_linear_trellis_3inst"):
        pytest.skip("fused_linear_trellis_3inst not registered (no CUDA ext loaded)")

    B, in_features, out_features = 2, 1024, 768
    n_pad, m_pad = in_features, out_features        # trellis never pads
    meta = torch.device("meta")
    x = torch.empty(B, in_features, dtype=torch.float16, device=meta)
    sv = torch.empty(n_pad, dtype=torch.float16, device=meta)
    su = torch.empty(m_pad, dtype=torch.float16, device=meta)
    packed = torch.empty((m_pad // 16) * (n_pad // 16), 32, dtype=torch.int16, device=meta)
    empty_i32 = torch.empty(0, dtype=torch.int32, device=meta)
    blk_n = torch.tensor([n_pad], dtype=torch.int64, device=meta)
    blk_m = torch.tensor([m_pad], dtype=torch.int64, device=meta)

    fy = torch.ops.glq.fused_linear_trellis_3inst(
        x, sv, su, packed,
        blk_n, blk_m, empty_i32, empty_i32,
        1.0, in_features, out_features, n_pad, m_pad,
    )
    assert fy.shape == (B, out_features), f"got {tuple(fy.shape)}"
    assert fy.dtype == torch.float16
    assert fy.device.type == "meta"


def test_fused_linear_trellis_3inst_rvq2_fake_shape():
    """Pins the stacked-RVQ (5-8 bpw) schema⇆fake arg count/order — 15 args: the 13-arg
    1-stage list plus ``trellis_packed2`` after its stage-1 partner and ``inv_resid_scale2``
    after ``wscale`` (the e8p argument-order convention). Deliberately a SEPARATE op from
    ``fused_linear_trellis_3inst``, whose 13-arg test above is left untouched and therefore
    doubles as the back-compat gate for every shipped 2-4 bpw checkpoint."""
    import glq_vllm.custom_ops
    glq_vllm.custom_ops._ensure_registered()
    if not hasattr(torch.ops.glq, "fused_linear_trellis_3inst_rvq2"):
        pytest.skip("fused_linear_trellis_3inst_rvq2 not registered (no CUDA ext loaded)")

    B, in_features, out_features = 2, 1024, 768
    n_pad, m_pad = in_features, out_features        # trellis never pads
    meta = torch.device("meta")
    x = torch.empty(B, in_features, dtype=torch.float16, device=meta)
    sv = torch.empty(n_pad, dtype=torch.float16, device=meta)
    su = torch.empty(m_pad, dtype=torch.float16, device=meta)
    rows = (m_pad // 16) * (n_pad // 16)
    packed = torch.empty(rows, 16 * 4, dtype=torch.int16, device=meta)   # stage 1: K1 = 4
    packed2 = torch.empty(rows, 16 * 2, dtype=torch.int16, device=meta)  # stage 2: K2 = 2 (6 bpw)
    empty_i32 = torch.empty(0, dtype=torch.int32, device=meta)
    blk_n = torch.tensor([n_pad], dtype=torch.int64, device=meta)
    blk_m = torch.tensor([m_pad], dtype=torch.int64, device=meta)

    fy = torch.ops.glq.fused_linear_trellis_3inst_rvq2(
        x, sv, su, packed, packed2,
        blk_n, blk_m, empty_i32, empty_i32,
        1.0, 0.25, in_features, out_features, n_pad, m_pad,
    )
    assert fy.shape == (B, out_features), f"got {tuple(fy.shape)}"
    assert fy.dtype == torch.float16
    assert fy.device.type == "meta"


def test_s4b_rvq2_fake_shape():
    """Pins the stacked-RVQ S4b mutating-op schema⇆fake (13 args)."""
    import glq_vllm.custom_ops
    glq_vllm.custom_ops._ensure_registered()
    if not hasattr(torch.ops.glq, "fused_linear_trellis_3inst_yrht_rvq2"):
        pytest.skip("fused_linear_trellis_3inst_yrht_rvq2 not registered (no CUDA ext)")

    B, n, m, total = 2, 1024, 768, 1024
    meta = torch.device("meta")
    x = torch.empty(B, n, dtype=torch.float16, device=meta)
    sv = torch.empty(n, dtype=torch.float16, device=meta)
    rows = (m // 16) * (n // 16)
    packed = torch.empty(rows, 16 * 4, dtype=torch.int16, device=meta)
    packed2 = torch.empty(rows, 16 * 2, dtype=torch.int16, device=meta)
    empty_i32 = torch.empty(0, dtype=torch.int32, device=meta)
    blk_n = torch.tensor([n], dtype=torch.int64, device=meta)
    y_rht = torch.empty(B, total, dtype=torch.float32, device=meta)
    r = torch.ops.glq.fused_linear_trellis_3inst_yrht_rvq2(
        x, sv, packed, packed2, blk_n, empty_i32, 1.0, 0.25, n, n, m, y_rht, 0)
    assert r is None


def test_s4b_shard_batched_ops_fake_shape():
    """Pins the S4b mutating-op schemas⇆fakes (11 args yrht / 6 args output_rht_shards).
    Both write their output arg in place and return nothing; an arg-count drift between
    the ``define`` strings and the fakes would raise here rather than at serve time."""
    import glq_vllm.custom_ops
    glq_vllm.custom_ops._ensure_registered()
    if not hasattr(torch.ops.glq, "fused_linear_trellis_3inst_yrht"):
        pytest.skip("fused_linear_trellis_3inst_yrht not registered (no CUDA ext loaded)")

    B, n, m, total = 2, 1024, 768, 1024
    meta = torch.device("meta")
    x = torch.empty(B, n, dtype=torch.float16, device=meta)
    sv = torch.empty(n, dtype=torch.float16, device=meta)
    packed = torch.empty((m // 16) * (n // 16), 32, dtype=torch.int16, device=meta)
    empty_i32 = torch.empty(0, dtype=torch.int32, device=meta)
    blk_n = torch.tensor([n], dtype=torch.int64, device=meta)
    y_rht = torch.empty(B, total, dtype=torch.float32, device=meta)
    r = torch.ops.glq.fused_linear_trellis_3inst_yrht(
        x, sv, packed, blk_n, empty_i32, 1.0, n, n, m, y_rht, 0)
    assert r is None

    su = torch.empty(total, dtype=torch.float16, device=meta)
    y = torch.empty(B, total, dtype=torch.float16, device=meta)
    shard_meta = torch.empty(2, 4, dtype=torch.int32, device=meta)
    r = torch.ops.glq.output_rht_shards(y_rht, su, y, total, shard_meta, 1024)
    assert r is None


@requires_vllm
@pytest.mark.parametrize("bpw", [2, 3, 4, 5, 6, 7, 8])
def test_trellis_create_weights_sizing(bpw):
    """Trellis create_weights registers the compressed buffers at FULL checkpoint size, so
    vLLM's loader takes the in-place copy_ branch — a shape mismatch sends it down
    ``param.data = empty_like(loaded)``, stranding .data on CPU and aborting the kernel at
    cudagraph capture. Trellis never pads: m_pad == out, n_pad == in. CPU-only.

    Parameterized over every servable rate: the packed width is the ONLY record of K in a
    checkpoint (cols == 16*K), so sizing it right is what makes the checkpoint load in-place
    instead of being reallocated onto CPU. Above 4 bpw the layer is stacked RVQ, and the
    widths below are stated as CONCRETE numbers rather than re-derived from
    ``trellis_rvq_recipe`` — restating them independently is what catches the ``16 * bpw``
    sizing that would give a 6 bpw layer a 96-column stage 1 against a 64-column
    checkpoint."""
    from glq_vllm.linear_method import GLQLinearMethod

    in_sz, out_sz = 2048, 3072
    # Stacked RVQ has no HYB decode (the fused entries take no tlut), so 5-8 is 3inst-only.
    variant = "hyb" if bpw <= 4 else "3inst"
    m = GLQLinearMethod(None, bpw=bpw, codebook_type="trellis", variant=variant)
    layer = torch.nn.Module()
    m.create_weights(layer, in_sz, [out_sz], in_sz, out_sz, torch.float16)

    assert layer.glq_is_trellis is True
    assert layer.glq_n_pad == in_sz and layer.glq_m_pad == out_sz          # no padding
    rows = (out_sz // 16) * (in_sz // 16)
    K1, K2 = min(bpw, 4), max(bpw - 4, 0)          # 2-4 native; 5-8 == 4 + (bpw-4)
    assert tuple(layer.trellis_packed.shape) == (rows, 16 * K1)
    assert layer.trellis_packed.dtype == torch.int16
    if K2:
        assert tuple(layer.trellis_packed2.shape) == (rows, 16 * K2)
        assert layer.trellis_packed2.dtype == torch.int16
    else:
        # 0-size, never a small placeholder: numel()==0 IS the "no stage 2" test.
        assert layer.trellis_packed2.numel() == 0
    # The scale is registered unconditionally (mirrors e8p) — a scalar fp32.
    assert layer.inv_resid_scale2.dtype == torch.float32
    assert layer.inv_resid_scale2.numel() == 1
    # the tlut is rate-INdependent (kmeans on 2-D Gaussians; never sees K); 3inst is
    # lookup-free, and its ZERO-SIZE tlut is what routes apply() to the no-tlut op.
    assert layer.tlut.dtype == torch.float16
    assert tuple(layer.tlut.shape) == ((512, 2) if variant == "hyb" else (0,))
    assert tuple(layer.SU.shape) == (out_sz,) and tuple(layer.SV.shape) == (in_sz,)


@requires_vllm
def test_trellis_rejects_unservable_shape():
    """Trellis never pads, so the kernel's m%32 / k%64 requirement cannot be hidden — a TP
    split that violates it must fail loudly at load rather than serve garbage."""
    from glq_vllm.linear_method import GLQLinearMethod
    m = GLQLinearMethod(None, bpw=2, codebook_type="trellis")
    with pytest.raises(ValueError, match="never pads"):
        m.create_weights(torch.nn.Module(), 2048, [48], 2048, 48, torch.float16)  # 48 % 32 != 0


@requires_vllm
@pytest.mark.parametrize("bpw", [5, 6, 7, 8])
def test_trellis_stacked_rvq_hyb_refused_on_vllm(bpw):
    """Stacked RVQ has no 2-stage HYB kernel and never will — the fused entries take no
    tlut. ``_trellis_linear_apply`` does raise on HYB+stage2, but that fires mid-forward,
    layers into the run; refuse at load instead. 3inst at the same rate must still build."""
    from glq_vllm.linear_method import GLQLinearMethod
    m = GLQLinearMethod(None, bpw=bpw, codebook_type="trellis", variant="hyb")
    with pytest.raises(ValueError, match="needs variant 3inst"):
        m.create_weights(torch.nn.Module(), 2048, [3072], 2048, 3072, torch.float16)


@requires_vllm
@pytest.mark.parametrize("bpw", [2, 3, 4])
def test_trellis_single_stage_still_accepted_on_vllm(bpw):
    """The HYB 5-8 refusal must not catch 2-4 bpw — HYB serves those on vLLM today and must
    keep working. This is the negative half of the guard above."""
    from glq_vllm.linear_method import GLQLinearMethod
    m = GLQLinearMethod(None, bpw=bpw, codebook_type="trellis", variant="hyb")
    layer = torch.nn.Module()
    m.create_weights(layer, 2048, [3072], 2048, 3072, torch.float16)   # must not raise
    assert layer.glq_is_trellis is True


@requires_vllm
@pytest.mark.parametrize("bpw", [4, 6])
def test_trellis_fused_shard_rvq2_sizing(bpw):
    """Fused QKV/gate_up: the residual is a GLQShardedParameter with one sentinel slot per
    output partition, like stage 1. Sentinel (not full-size) is right here and ONLY here —
    GLQShardedParameter keeps its buffers in a private _shard_data list vLLM doesn't manage,
    so the loader's empty_like() replacement sticks; and ``get_shard(i).numel() == 0`` then
    doubles as the per-shard "no stage 2" test, covering a never-loaded KV-shared shard
    without a second flag. Both rates are asserted the same way because at build time the
    shards are shape-agnostic — bpw only decides what the loader later drops in."""
    from glq_vllm.linear_method import GLQLinearMethod, GLQShardedParameter
    _init_tp_once()          # GLQShardedParameter derives from BasevLLMParameter
    ops = [1024, 256, 256]
    m = GLQLinearMethod(None, bpw=bpw, codebook_type="trellis", variant="3inst")
    layer = torch.nn.Module()
    m.create_weights(layer, 2048, ops, 2048, sum(ops), torch.float16,
                     weight_loader=lambda *a, **k: None)

    assert layer.glq_is_fused is True and layer.glq_num_shards == len(ops)
    for p in (layer.trellis_packed, layer.trellis_packed2):
        assert isinstance(p, GLQShardedParameter)
        assert p.num_shards == len(ops)
        assert all(p.get_shard(i).numel() == 0 for i in range(len(ops)))
        assert p.get_shard(0).dtype == torch.int16
    assert isinstance(layer.inv_resid_scale2, GLQShardedParameter)
    assert layer.inv_resid_scale2.num_shards == len(ops)


def _loaded_trellis_layer(bpw, in_sz=128, out_sz=64, inv_rs2=0.75, drop_stage2=False):
    """A non-fused trellis layer as the loader leaves it: create_weights + an in-place
    copy_ into each registered buffer (which is what a shape-matching checkpoint does)."""
    from glq_vllm.linear_method import GLQLinearMethod
    m = GLQLinearMethod(None, bpw=bpw, codebook_type="trellis", variant="3inst")
    layer = torch.nn.Module()
    m.create_weights(layer, in_sz, [out_sz], in_sz, out_sz, torch.float16)
    if layer.trellis_packed2.numel() and not drop_stage2:
        layer.inv_resid_scale2.data.copy_(torch.tensor(inv_rs2))
    elif drop_stage2:                    # a checkpoint written before stacked RVQ
        layer.trellis_packed2.data = torch.zeros(0, dtype=torch.int16)
    return m, layer


@requires_vllm
@pytest.mark.parametrize("bpw", [4, 6])
def test_trellis_setup_hoists_stage2_scale(bpw):
    """``_setup_trellis_weights`` holds the ONLY .item() in the trellis path — apply() must
    stay pure tensor work or cudagraph capture aborts. Assert the hoist happened (a python
    float, not a Tensor) and that ``has_s2`` agrees with it: the e8p stage-3/4 silent drop
    came from a flag saying "stage present" beside a scale still at 0.0, resolved under a
    different condition, and the stage then vanished with no error."""
    _m, layer = _loaded_trellis_layer(bpw)
    _m._setup_trellis_weights(layer, torch.device('cpu'))

    meta = layer._glq_trellis_meta[0]
    assert meta['has_s2'] is (bpw >= 5)
    assert type(meta['inv_rs2']) is float          # not a 0-dim Tensor
    assert meta['inv_rs2'] == (0.75 if bpw >= 5 else 0.0)
    assert type(meta['wscale']) is float


@requires_vllm
def test_trellis_setup_refuses_missing_stage2():
    """vLLM has no dense fallback: a 6 bpw layer whose checkpoint carries no
    ``trellis_packed2`` would serve stage-1-only output — 4 bpw quality, silently. The HF
    path can warn and decode in torch; here the only safe answer is to fail at load."""
    _m, layer = _loaded_trellis_layer(6, drop_stage2=True)
    with pytest.raises(RuntimeError, match="residual stage would be dropped"):
        _m._setup_trellis_weights(layer, torch.device('cpu'))


@requires_vllm
def test_trellis_setup_refuses_zero_stage2_scale():
    """The same bug wearing a different mask: the buffer is present but its scale never
    loaded, so the residual would decode to exactly nothing. Flag and scale are checked
    together precisely because either alone can hide the drop."""
    _m, layer = _loaded_trellis_layer(6, inv_rs2=0.0)
    with pytest.raises(RuntimeError, match="inv_resid_scale2 is 0.0"):
        _m._setup_trellis_weights(layer, torch.device('cpu'))


@requires_vllm
def test_trellis_setup_refuses_unexpected_stage2():
    """The converse direction: a 4 bpw layer must not carry a residual. config.json's bpw
    is what sized every buffer, so a checkpoint that disagrees is corrupt — and leaving it
    unchecked would let has_s2 differ per shard, which the S4b gate assumes it cannot."""
    _m, layer = _loaded_trellis_layer(4)
    layer.trellis_packed2.data = torch.zeros(32, 16, dtype=torch.int16)   # not in the recipe
    with pytest.raises(RuntimeError, match="no stage for"):
        _m._setup_trellis_weights(layer, torch.device('cpu'))


@requires_vllm
def test_trellis_hyb_moe_refused():
    """HYB trellis MoE stays refused: the fused trellis entries take no tlut, so a HYB
    checkpoint would decode against a codebook the kernel never sees.

    The blanket trellis-MoE refusal this replaces is gone (3INST serves now). What it was
    really guarding — a trellis layer falling through the ``== e8p`` check into SHELL buffer
    registration, a different storage format entirely — is closed better, by
    ``_create_weights_trellis`` claiming the codebook before that branch is reachable."""
    from glq_vllm.fused_moe_method import GLQFusedMoEMethod
    # Built without __init__ on purpose: FusedMoEMethodBase wants a live FusedMoEConfig,
    # and the guard fires before create_weights reads anything but codebook_type.
    method = GLQFusedMoEMethod.__new__(GLQFusedMoEMethod)
    method.codebook_type = "trellis"
    method.quant_config = types.SimpleNamespace(variant="hyb", bpw=4)
    with pytest.raises(ValueError, match="only the 3inst variant"):
        method.create_weights(torch.nn.Module(), 8, 128, 256, torch.float16,
                              weight_loader=lambda *a, **k: None)


def test_fused_moe_trellis_3inst_schema_matches_fake():
    """The schema⇆fake arity check that does NOT need a built CUDA extension.

    Every other op's ``define`` string is inline behind ``hasattr(cuda, ...)``, so its arity
    is only checkable where the .so exists — and a drift there fails deep inside dispatch,
    possibly mid-capture. This one reads the schema constant and the fake's signature
    directly, so an added/removed/reordered argument is caught on plain CPU at edit time.
    Names are compared too, not just the count: a reorder keeps the count identical while
    binding weights to the wrong parameter, which decodes to plausible garbage."""
    import inspect
    from glq_vllm.custom_ops import (FUSED_MOE_TRELLIS_3INST_SCHEMA,
                                     _fused_moe_trellis_3inst_fake)

    args = FUSED_MOE_TRELLIS_3INST_SCHEMA.split("(", 1)[1].rsplit(")", 1)[0]
    schema_names = [a.strip().split()[-1] for a in args.split(",")]
    fake_names = list(inspect.signature(_fused_moe_trellis_3inst_fake).parameters)
    assert schema_names == fake_names, (
        f"schema/fake mismatch\n schema: {schema_names}\n fake:   {fake_names}")
    # 15 tensors + 7 ints + 8 block tensors + activation_type. Stated as a number so an
    # accidental deletion that keeps both sides in step still trips.
    assert len(schema_names) == 31


@requires_vllm
def test_fused_moe_trellis_3inst_fake_shape():
    """The fake returns the MoE output contract, (num_tokens, hidden) fp16 — same as every
    other GLQ MoE op, which is what lets apply() swap paths without the compiler noticing.
    Meta tensors, so it runs without a GPU (but needs the ext for the op to be registered)."""
    import glq_vllm.custom_ops
    glq_vllm.custom_ops._ensure_registered()
    if not hasattr(torch.ops.glq, "fused_moe_trellis_3inst"):
        pytest.skip("fused_moe_trellis_3inst not registered (no CUDA ext loaded)")

    E, T, top_k = 8, 3, 2
    hidden, inter, w13_out = 256, 128, 256
    meta = torch.device("meta")
    x = torch.empty(T, hidden, dtype=torch.float16, device=meta)
    topk_ids = torch.empty(T, top_k, dtype=torch.int64, device=meta)
    topk_w = torch.empty(T, top_k, dtype=torch.float32, device=meta)

    def _packed(m, n):
        return torch.empty(E, (m // 16) * (n // 16), 64, dtype=torch.int16, device=meta)
    sentinel = torch.empty(E, 1, 1, dtype=torch.int16, device=meta)
    empty_i32 = torch.empty(0, dtype=torch.int32, device=meta)
    y = torch.ops.glq.fused_moe_trellis_3inst(
        x, topk_ids, topk_w,
        _packed(w13_out, hidden), sentinel,
        torch.empty(E, w13_out, dtype=torch.float16, device=meta),
        torch.empty(hidden, dtype=torch.float16, device=meta),
        torch.empty(E, dtype=torch.float32, device=meta),
        torch.empty(E, dtype=torch.float32, device=meta),
        _packed(hidden, inter), sentinel,
        torch.empty(E, hidden, dtype=torch.float16, device=meta),
        torch.empty(inter, dtype=torch.float16, device=meta),
        torch.empty(E, dtype=torch.float32, device=meta),
        torch.empty(E, dtype=torch.float32, device=meta),
        hidden, inter, w13_out, hidden, w13_out, inter, hidden,
        torch.tensor([hidden], dtype=torch.int64, device=meta),
        torch.tensor([w13_out], dtype=torch.int64, device=meta), empty_i32, empty_i32,
        torch.tensor([inter], dtype=torch.int64, device=meta),
        torch.tensor([hidden], dtype=torch.int64, device=meta), empty_i32, empty_i32,
        1,
    )
    assert y.shape == (T, hidden) and y.dtype == torch.float16
    assert y.device.type == "meta"


def _trellis_moe_layer(bpw=4, num_experts=8, hidden=256, inter=128):
    """A trellis MoE layer as the loader leaves it: create_weights registers full-size
    buffers, then _process_trellis derives the block-diag meta and the fused gate."""
    from glq_vllm.fused_moe_method import GLQFusedMoEMethod
    method = GLQFusedMoEMethod.__new__(GLQFusedMoEMethod)
    method.codebook_type = "trellis"
    method.quant_config = types.SimpleNamespace(variant="3inst", bpw=bpw)
    method.moe = types.SimpleNamespace(is_act_and_mul=True)
    layer = torch.nn.Module()
    method.create_weights(layer, num_experts, hidden, inter, torch.float16,
                          weight_loader=lambda *a, **k: None)
    if bpw >= 5:                       # a real residual needs a nonzero scale to be valid
        layer.w13_inv_resid_scale2.data.fill_(0.75)
        layer.w2_inv_resid_scale2.data.fill_(0.75)
    return method, layer


@requires_vllm
@pytest.mark.parametrize("bpw", [4, 6])
def test_trellis_moe_fused_gate_accepts_servable_layer(bpw):
    """The positive half of the gate: a gemma-4-shaped layer at 4 and 6 bpw must route to the
    fused grouped op, because that is the ONLY cudagraph-capturable path — falling back to
    the per-expert loop is silently correct, so nothing else would notice."""
    _m, layer = _trellis_moe_layer(bpw=bpw)
    _m._process_trellis(layer)
    assert layer.glq_trellis_fused_ok is True
    assert layer._glq_trellis_moe_meta['w13']['has_s2'] is (bpw >= 5)


@requires_vllm
def test_trellis_moe_fused_gate_rejects_unservable_shape():
    """The kernel needs m % 32 and k % 64; trellis never pads, so a TP split that violates
    it cannot be hidden. The gate must decline rather than let the op abort mid-forward —
    the loop still serves the layer correctly, just eagerly."""
    _m, layer = _trellis_moe_layer(hidden=256, inter=160)     # w2 k = 160, 160 % 64 != 0
    _m._process_trellis(layer)
    assert layer.glq_trellis_fused_ok is False


@requires_vllm
def test_trellis_moe_fused_gate_ignores_missing_activation_attr():
    """``layer.activation`` is a runner attribute that need not exist at load time. Folding
    the gated-activation requirement into the load-time flag would read a missing one as the
    silu default and wrongly admit a non-gated layer, so the flag must not depend on it —
    apply() checks the activation at dispatch, where it is actually known."""
    _m, layer = _trellis_moe_layer()
    assert not hasattr(layer, 'activation')
    _m._process_trellis(layer)
    assert layer.glq_trellis_fused_ok is True


@requires_vllm
def test_glq_config_trellis_variant():
    """`variant` round-trips. Both shipped variants (hyb, 3inst-kernel-layout) are accepted;
    a pre-kernel NATURAL-layout 3inst checkpoint (no trellis_layout marker) and unknown
    variants are refused up front rather than served as garbage."""
    from glq_vllm.config import GLQvLLMConfig
    cfg = GLQvLLMConfig.from_config(
        {"bpw": 2, "codebook": "trellis", "variant": "hyb", "block_diagonal": True})
    assert cfg.codebook == "trellis" and cfg.variant == "hyb"
    assert GLQvLLMConfig.from_config({"bpw": 2, "codebook": "e8p"}).variant == "hyb"  # default
    # 3inst with the kernel-layout marker serves
    cfg3 = GLQvLLMConfig.from_config(
        {"bpw": 2, "codebook": "trellis", "variant": "3inst", "trellis_layout": "kernel"})
    assert cfg3.variant == "3inst" and cfg3.trellis_layout == "kernel"
    # legacy natural-layout 3inst (pre-kernel, no marker) → refused
    with pytest.raises(ValueError, match="NATURAL layout"):
        GLQvLLMConfig.from_config({"bpw": 2, "codebook": "trellis", "variant": "3inst"})
    # unknown variant → refused
    with pytest.raises(ValueError, match="no CUDA kernel"):
        GLQvLLMConfig.from_config(
            {"bpw": 2, "codebook": "trellis", "variant": "1mad", "trellis_layout": "kernel"})


@requires_vllm
def test_glq_config_from_config():
    """GLQvLLMConfig should parse bpw and layer_bpw from dict."""
    from glq_vllm.config import GLQvLLMConfig
    cfg = GLQvLLMConfig.from_config({"bpw": 4, "layer_bpw": {"layers.0.q_proj": 2}})
    assert cfg.bpw == 4
    assert cfg.layer_bpw["layers.0.q_proj"] == 2
    assert cfg.get_name() == "glq"
    assert cfg.get_config_filenames() == ["quantize_config.json"]
    assert torch.float16 in cfg.get_supported_act_dtypes()


@requires_vllm
def test_glq_config_block_diagonal_flag():
    """``block_diagonal`` is parsed from config and threaded to the per-layer
    method. It records the e8p RHT layout so the loader sizes the weight
    buffers to match the checkpoint: True = block-diagonal padding (the
    0.6.6+ default), False = legacy full pow2 Hadamard. Absent → True for
    back-compat with the block-diagonal default, NOT the older pow2 uploads
    (which must set it explicitly)."""
    from vllm.model_executor.layers.linear import LinearBase
    from glq_vllm.config import GLQvLLMConfig
    from glq_vllm.linear_method import GLQLinearMethod

    # Absent → defaults to True (block-diagonal e8p).
    cfg = GLQvLLMConfig.from_config({"bpw": 3, "codebook": "e8p"})
    assert cfg.block_diagonal is True

    # Explicit False (legacy pow2 e8p checkpoint).
    cfg = GLQvLLMConfig.from_config({
        "bpw": 3, "codebook": "e8p", "block_diagonal": False,
        "layer_bpw": {"model.layers.0.mlp.down_proj": 3},
    })
    assert cfg.block_diagonal is False

    # Threaded into the per-layer GLQLinearMethod.
    layer = LinearBase.__new__(LinearBase)
    method = cfg.get_quant_method(layer, "model.layers.0.mlp.down_proj")
    assert isinstance(method, GLQLinearMethod)
    assert method.block_diagonal is False
    assert method.codebook_type == "e8p"


@requires_vllm
def test_e8p_create_weights_block_diagonal_sizing():
    """e8p create_weights sizes the Qidxs buffers per the ``block_diagonal``
    flag — block-diag pads to a mult-of-64 (cols) / mult-of-16 (rows), pow2
    pads each dim to the next power of two. A non-pow2 shape distinguishes
    them: 11008 stays 11008 block-diag but inflates to 16384 pow2; 3072 stays
    3072 block-diag but inflates to 4096 pow2. CPU-only (no codebook load)."""
    from glq_vllm.linear_method import GLQLinearMethod, _glq_pad, _e8p_pad

    in_sz, out_sz = 11008, 3072
    assert _e8p_pad(in_sz, 64) == 11008 and _glq_pad(in_sz) == 16384
    assert _e8p_pad(out_sz, 16) == 3072 and _glq_pad(out_sz) == 4096

    def make(block_diagonal):
        m = GLQLinearMethod(None, bpw=3, codebook_type="e8p",
                            block_diagonal=block_diagonal)
        layer = torch.nn.Module()
        m.create_weights(layer, in_sz, [out_sz], in_sz, out_sz,
                         torch.float16)
        return layer

    bd = make(True)
    assert tuple(bd.Qidxs_e8p.shape) == (3072 // 16, 11008 // 64, 8, 4)
    assert bd.glq_n_pad == 11008 and bd.glq_m_pad == 3072

    pw = make(False)
    assert tuple(pw.Qidxs_e8p.shape) == (4096 // 16, 16384 // 64, 8, 4)
    assert pw.glq_n_pad == 16384 and pw.glq_m_pad == 4096


# ── GLQ embedding-method dispatch + dequant equivalence ───────────────

@requires_vllm
def test_glq_embedding_method_registered():
    """get_quant_method returns GLQEmbeddingMethod for VocabParallelEmbedding
    when the prefix is in layer_bpw, else UnquantizedEmbeddingMethod."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod, VocabParallelEmbedding,
    )
    from glq_vllm.config import GLQvLLMConfig
    from glq_vllm.embedding_method import GLQEmbeddingMethod

    cfg = GLQvLLMConfig.from_config({
        "bpw": 4,
        "layer_bpw": {
            # checkpoint-form key, like quantize_model.py emits
            "model.language_model.embed_tokens_per_layer": 4,
        },
    })
    # Avoid full __init__ (needs torch.distributed setup); just satisfy isinstance.
    layer = VocabParallelEmbedding.__new__(VocabParallelEmbedding)

    # Quantized embedding — runtime prefix is the multimodal mm-rewritten form.
    method = cfg.get_quant_method(
        layer, "language_model.model.embed_tokens_per_layer")
    assert isinstance(method, GLQEmbeddingMethod), type(method).__name__
    assert method.bpw == 4

    # Unrelated embedding (e.g. main embed_tokens) — not in layer_bpw → unquant.
    method = cfg.get_quant_method(layer, "language_model.model.embed_tokens")
    assert isinstance(method, UnquantizedEmbeddingMethod), type(method).__name__


@pytest.mark.skipif(
    not _HAS_TRANSFORMERS, reason="transformers not installed"
)
def test_glq_embedding_dequant_matches_hf():
    """``_dequant_embedding_rows`` must reproduce ``E8RHTEmbedding.forward``
    (with embed_scale=1.0, the convention vLLM uses) byte-for-byte.

    Uses synthetic GLQ buffers + a tiny codebook so the test is fast and
    needs no GPU.
    """
    import glq.hf_integration  # noqa: F401 (registers types)
    from glq.codebook import E8ShellCodebook
    from glq.quantized_linear import E8RHTEmbedding, _dequant_embedding_rows

    torch.manual_seed(0)
    vocab, dim = 64, 32
    n_pad = 32  # already pow2
    cb = E8ShellCodebook(device="cpu", verbose=False)

    # Stage-1 only (2bpw equivalent).
    emb = E8RHTEmbedding(num_embeddings=vocab, embedding_dim=dim,
                          embed_scale=1.0)
    emb.set_codebook(cb)
    emb.Qidxs.copy_(torch.randint(
        0, 65536, emb.Qidxs.shape, dtype=torch.int32).to(torch.int16))
    emb.SV.copy_(torch.randn_like(emb.SV))
    emb.Wscale.copy_(torch.rand_like(emb.Wscale) + 0.1)

    ids = torch.tensor([0, 5, 17, 63, 42], dtype=torch.long)
    hf_out = emb(ids)

    direct_out = _dequant_embedding_rows(
        ids,
        emb.Qidxs, emb.SV, emb.Wscale, cb.codebook,
        None, None, None,
        n_pad=n_pad, embedding_dim=dim,
        embed_scale=1.0, out_dtype=hf_out.dtype,
    )
    torch.testing.assert_close(hf_out, direct_out, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(
    not _HAS_TRANSFORMERS, reason="transformers not installed"
)
def test_shell_ple_quant_decode_roundtrip():
    """A PLE chunk quantized with the SHELL codebook via quantize_layer_e8_shell_rht
    (apply_left=False, bpw=4) — the path a Gemma-4 E2B/E4B now takes under
    --codebook e8p — loads into E8RHTEmbedding and reconstructs the original rows.
    Proves shell indices + a shell codebook decode correctly (no Qidxs_e8p, no
    codebook mismatch) end-to-end."""
    import glq.hf_integration  # noqa: F401 (registers types)
    from glq.codebook import E8ShellCodebook
    from glq.quantize_model import quantize_layer_e8_shell_rht
    from glq.quantized_linear import E8RHTEmbedding

    torch.manual_seed(0)
    vocab, dim = 64, 64
    cb = E8ShellCodebook.build(device="cpu", verbose=False)
    W = torch.randn(vocab, dim) * 0.1
    _, arts, _ = quantize_layer_e8_shell_rht(
        W, torch.eye(dim), cb, bpw=4, apply_left=False, block_diagonal=False)
    assert {'Qidxs', 'Qidxs2', 'SV', 'Wscale', 'inv_resid_scale'} <= set(arts)
    assert 'Qidxs_e8p' not in arts

    emb = E8RHTEmbedding(num_embeddings=vocab, embedding_dim=dim, embed_scale=1.0)
    assert emb.Qidxs.shape == arts['Qidxs'].shape  # n_pad matches the quant output
    emb.Qidxs.copy_(arts['Qidxs'])
    emb.Qidxs2.copy_(arts['Qidxs2'])
    emb.SV.copy_(arts['SV'])
    emb.Wscale.fill_(float(arts['Wscale']))          # one chunk -> one scalar, broadcast per-row
    emb.inv_resid_scale.fill_(float(arts['inv_resid_scale']))
    emb.set_codebook(cb, codebook2=cb)               # full-shell stage-2 (4bpw PLE)
    assert emb._n_stages == 2

    out = emb(torch.arange(vocab))
    assert out.shape == (vocab, dim) and torch.isfinite(out).all()
    cos = torch.nn.functional.cosine_similarity(
        out.flatten().float(), W.flatten().float(), dim=0).item()
    assert cos > 0.8, f"reconstruction cosine {cos:.3f} too low"


# ── Test 2b: _lookup_bpw across vLLM prefix transforms ─────────────────

@requires_vllm
def test_lookup_bpw_prefix_forms():
    """Whitelist must match the same logical layer across the prefix forms
    vLLM produces at runtime:

    - text-only ``Gemma4ForCausalLM`` strips ``model.language_model.`` to ``model.``
    - multimodal ``Gemma4ForConditionalGeneration`` rewrites
      ``model.language_model.X`` to ``language_model.model.X``
    - stacked-merge: ``qkv_proj`` matches when any of q/k/v are listed unmerged

    Storing keys in safetensors-checkpoint form (``model.language_model.X``)
    and asking the whitelist to map every runtime form back to it is the
    convention quantize_model writes.
    """
    from glq_vllm.config import GLQvLLMConfig
    cfg = GLQvLLMConfig.from_config({
        "bpw": 4,
        "layer_bpw": {
            # checkpoint-form keys, as quantize_model writes them
            "model.language_model.layers.0.mlp.down_proj": 4,
            "model.language_model.layers.0.self_attn.q_proj": 4,
            "model.language_model.layers.0.self_attn.k_proj": 4,
            "model.language_model.layers.0.self_attn.v_proj": 4,
            "model.language_model.layers.0.per_layer_input_gate": 4,
        },
    })

    # Direct match.
    assert cfg._lookup_bpw("model.language_model.layers.0.mlp.down_proj") == 4
    # Text-only Gemma-4 path: vLLM strips `language_model.` so prefix is
    # `model.layers.0...`. Whitelist re-adds the missing `language_model.`.
    assert cfg._lookup_bpw("model.layers.0.mlp.down_proj") == 4
    # Multimodal Gemma-4 path: vLLM mapper renames
    # `model.language_model.X` -> `language_model.model.X`.
    assert cfg._lookup_bpw("language_model.model.layers.0.mlp.down_proj") == 4
    assert cfg._lookup_bpw(
        "language_model.model.layers.0.per_layer_input_gate") == 4
    # Stacked-merge: vLLM packs q/k/v into qkv_proj at runtime; whitelist
    # finds at least one of the three subnames and returns the max bpw.
    assert cfg._lookup_bpw("model.layers.0.self_attn.qkv_proj") == 4
    assert cfg._lookup_bpw(
        "language_model.model.layers.0.self_attn.qkv_proj") == 4

    # Layer absent from whitelist returns None, so get_quant_method falls
    # through to UnquantizedLinearMethod (e.g. Gemma-4
    # `per_layer_model_projection`, which is bf16 in the checkpoint).
    assert cfg._lookup_bpw(
        "model.language_model.per_layer_model_projection") is None
    assert cfg._lookup_bpw(
        "language_model.model.per_layer_model_projection") is None
    assert cfg._lookup_bpw("model.per_layer_model_projection") is None


# ── Test 3: Dequant correctness (CPU, no vLLM server) ──────────────────

@pytest.mark.skipif(
    not _HAS_TRANSFORMERS, reason="transformers not installed"
)
def test_dequant_matches_hf():
    """dequantize_glq_weight must match E8RHTLinear.dequantize()."""
    import glq.hf_integration  # noqa: F401
    from transformers import AutoModelForCausalLM
    from glq_vllm.dequant import dequantize_glq_weight, get_codebook, get_codebook2

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cpu", torch_dtype=torch.float16
    )
    layer = model.model.layers[0].self_attn.q_proj
    hf_weight = layer.dequantize()

    cb = get_codebook()
    inv_rs = layer.inv_resid_scale.item()
    has_stage2 = inv_rs != 0.0
    cb2 = get_codebook2(4) if has_stage2 else None

    vllm_weight = dequantize_glq_weight(
        layer.Qidxs, layer.SU, layer.SV, layer.Wscale, cb,
        layer.Qidxs2 if has_stage2 else None, inv_rs, cb2,
        layer.out_features, layer.in_features,
    )

    cos = torch.nn.functional.cosine_similarity(
        hf_weight.flatten().float(), vllm_weight.flatten().float(), dim=0
    ).item()
    assert cos > 0.999, f"Dequant cosine similarity {cos:.4f} too low"


# ── Tests 4-7: GPU tests with shared LLM instance ──────────────────────

@requires_vllm
@requires_gpu
def test_glq_vllm_gpu():
    """All GPU tests in one: weights, generation, VRAM, throughput.

    Uses a single LLM instance to avoid repeated 5-min model loads.
    """
    import glq_vllm  # noqa: F401
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_ID,
        tokenizer=TOKENIZER_ID,
        quantization="glq",
        enforce_eager=True,
        dtype="half",
        gpu_memory_utilization=0.9,
        max_model_len=512,
    )

    # --- 4. Weights stay compressed ---
    # check_layers runs in the engine-core subprocess (vLLM >= 0.25): it must RETURN
    # its result — mutating a closed-over object only changes the subprocess copy.
    def check_layers(model):
        found = False
        for name, mod in model.named_modules():
            if hasattr(mod, "Qidxs"):
                assert mod.Qidxs.dtype == torch.int16, f"{name}.Qidxs is {mod.Qidxs.dtype}"
                # vLLM >= 0.25 registers a numel-1 fp16 stub `weight` on every linear
                # (PluggableLayer); the invariant is no DENSE weight alongside Qidxs.
                w = getattr(mod, "weight", None)
                assert w is None or w.numel() <= 1, \
                    f"{name} has dense weight {tuple(w.shape)}"
                found = True
        return found

    res = llm.apply_model(check_layers)
    flags = res if isinstance(res, list) else [res]
    assert any(flags), "No GLQ layers found with Qidxs"

    # --- 5. Generation correctness ---
    params = SamplingParams(max_tokens=30, temperature=0)
    outputs = llm.generate(["The capital of France is"], params)
    text = outputs[0].outputs[0].text
    assert len(text) > 10, f"Output too short: {text!r}"
    assert "Paris" in text, f"Expected 'Paris' in output: {text!r}"

    # --- 6. Throughput ---
    params = SamplingParams(max_tokens=128, temperature=0)
    t0 = time.time()
    outputs = llm.generate(["The theory of relativity states that"], params)
    t1 = time.time()
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tok_per_sec = total_tokens / (t1 - t0)
    assert tok_per_sec >= 10, f"Throughput {tok_per_sec:.1f} tok/s below 10 tok/s minimum"

    del llm


# ── Unloaded fused shard (gemma-4 E-series KV-shared layers) ────────────


@requires_vllm
@requires_gpu
@pytest.mark.parametrize("bpw", [5, 6, 8])
def test_trellis_fused_rvq2_paths_agree(bpw):
    """Stacked RVQ through a FUSED (qkv-shaped) vLLM layer: the S4b shard-batched path and
    the per-shard fallback must both equal the shared staticmethod the HF path runs.

    S4b is where a dropped residual would hide — its y_rht entry takes the same arguments
    with or without stage 2, so a 1-stage call on a 2-stage layer returns plausible output
    at 4 bpw quality. Hence the mechanism assertions: the layer must actually be on the
    batched path (``_glq_s4b`` built) and must know it carries a residual."""
    import glq_vllm.linear_method as lm
    from glq.quantized_linear import E8RHTLinear
    from glq.trellis import trellis_rvq_recipe

    if not hasattr(torch.ops.glq, "fused_linear_trellis_3inst_yrht_rvq2"):
        pytest.skip("fused_linear_trellis_3inst_yrht_rvq2 not in this build")
    _init_tp_once()

    torch.manual_seed(0)
    dev = torch.device("cuda")
    n, m0, m1 = 64, 32, 32
    K1, K2 = trellis_rvq_recipe(bpw)
    ldr = lambda *a, **k: None

    layer = torch.nn.Module()
    layer.glq_is_fused = True
    layer.glq_num_shards = 2
    layer.glq_shard_sizes = [m0, m1]
    layer.glq_in_features = n
    layer.glq_n_pad = n
    layer.glq_bpw = bpw
    layer.trellis_packed = lm.GLQShardedParameter(
        [m0, m1], 1, torch.int16, weight_loader=ldr, sentinel=True)
    layer.trellis_packed2 = lm.GLQShardedParameter(
        [m0, m1], 1, torch.int16, weight_loader=ldr, sentinel=True)
    layer.inv_resid_scale2 = lm.GLQShardedParameter(
        [1, 1], 0, torch.float32, weight_loader=ldr)
    layer.tlut = lm.GLQShardedParameter(
        [m0, m1], 1, torch.float16, weight_loader=ldr, sentinel=True)
    layer.SU = lm.GLQShardedParameter([m0, m1], -1, torch.float16, weight_loader=ldr)
    layer.SV = lm.GLQShardedParameter([n, n], -1, torch.float16, weight_loader=ldr)
    layer.Wscale = lm.GLQShardedParameter([1, 1], 0, torch.float32, weight_loader=ldr)

    # "Load" both shards. Random bits are valid trellis codes — the decode is arithmetic
    # (3INST), so every bit pattern maps to some weight; what is under test is the plumbing.
    sgn = lambda k: torch.where(torch.rand(k) < 0.5, -torch.ones(k), torch.ones(k)).half()
    for i, m in enumerate((m0, m1)):
        rows = (m // 16) * (n // 16)
        for buf, K in ((layer.trellis_packed, K1), (layer.trellis_packed2, K2)):
            buf._shard_data[i] = torch.randint(-2**15, 2**15 - 1, (rows, 16 * K),
                                               dtype=torch.int16)
        layer.SU._shard_data[i] = sgn(m)
        layer.SV._shard_data[i] = sgn(n)
        layer.Wscale._shard_data[i] = torch.tensor([1.5 + i])
        layer.inv_resid_scale2._shard_data[i] = torch.tensor(0.25 + 0.1 * i)

    method = lm.GLQLinearMethod(None, bpw=bpw, codebook_type="trellis", variant="3inst")
    method._setup_trellis_weights(layer, dev)
    assert layer._glq_s4b is not None, "stacked-RVQ layer fell off the S4b path"
    assert layer._glq_s4b['has_s2'] is True
    assert all(m['has_s2'] for m in layer._glq_trellis_meta)

    x = torch.randn(3, n, dtype=torch.float16, device=dev)
    y_s4b = lm._glq_apply_trellis(x, layer)
    prev = lm._GLQ_BATCH_OUT_RHT
    lm._GLQ_BATCH_OUT_RHT = False
    try:
        y_seq = lm._glq_apply_trellis(x, layer)
    finally:
        lm._GLQ_BATCH_OUT_RHT = prev

    refs = []
    for i, meta in enumerate(layer._glq_trellis_meta):
        refs.append(E8RHTLinear._trellis_linear_apply(
            x, layer.SV.get_shard(i), layer.SU.get_shard(i),
            layer.trellis_packed.get_shard(i), layer.tlut.get_shard(i),
            meta['_bn'], meta['_bm'], meta['_bnm'], meta['_bmm'], meta['wscale'],
            n, meta['out'], n, meta['m_pad'], bias=None, out_dtype=x.dtype,
            trellis_packed2=layer.trellis_packed2.get_shard(i),
            inv_resid_scale2=meta['inv_rs2']))
    ref = torch.cat(refs, dim=-1)
    for name, y in (("s4b", y_s4b), ("sequential", y_seq)):
        assert y.shape == (3, m0 + m1), name
        assert torch.equal(y, ref), f"{name} disagrees with the shared staticmethod"

    # And the residual must actually MOVE the output — a silently dropped stage 2 would
    # still pass every equality above if both sides dropped it.
    one_stage = torch.cat([E8RHTLinear._trellis_linear_apply(
        x, layer.SV.get_shard(i), layer.SU.get_shard(i),
        layer.trellis_packed.get_shard(i), layer.tlut.get_shard(i),
        meta['_bn'], meta['_bm'], meta['_bnm'], meta['_bmm'], meta['wscale'],
        n, meta['out'], n, meta['m_pad'], bias=None, out_dtype=x.dtype)
        for i, meta in enumerate(layer._glq_trellis_meta)], dim=-1)
    assert not torch.equal(y_s4b, one_stage), "stage 2 contributed nothing"


@requires_vllm
@requires_gpu
def test_trellis_fused_unloaded_shard_zero_columns():
    """vLLM builds q/k/v shards for EVERY gemma-4 E-series layer, but KV-shared
    layers carry no k/v weights in ANY checkpoint — those shards never load
    (sentinel empty packed, Wscale 0). The trellis apply must emit ZERO columns
    for them (the shell path's de-facto semantics: attention discards those
    columns), instead of aborting in the kernel's trellis_packed 2-D check.
    Covers BOTH the S4b shard-batched branch and the per-shard fallback."""
    import glq_vllm.linear_method as lm
    from glq.quantized_linear import E8RHTLinear
    _init_tp_once()

    torch.manual_seed(0)
    dev = torch.device("cuda")
    n, m0, m1, K = 64, 32, 32, 2
    ldr = lambda *a, **k: None

    layer = torch.nn.Module()
    layer.glq_is_fused = True
    layer.glq_num_shards = 2
    layer.glq_shard_sizes = [m0, m1]
    layer.glq_in_features = n
    layer.glq_n_pad = n
    layer.glq_bpw = K
    layer.trellis_packed = lm.GLQShardedParameter(
        [m0, m1], 1, torch.int16, weight_loader=ldr, sentinel=True)
    layer.trellis_packed2 = lm.GLQShardedParameter(
        [m0, m1], 1, torch.int16, weight_loader=ldr, sentinel=True)
    layer.inv_resid_scale2 = lm.GLQShardedParameter(
        [1, 1], 0, torch.float32, weight_loader=ldr)
    layer.tlut = lm.GLQShardedParameter(
        [m0, m1], 1, torch.float16, weight_loader=ldr, sentinel=True)
    layer.SU = lm.GLQShardedParameter([m0, m1], -1, torch.float16, weight_loader=ldr)
    layer.SV = lm.GLQShardedParameter([n, n], -1, torch.float16, weight_loader=ldr)
    layer.Wscale = lm.GLQShardedParameter([1, 1], 0, torch.float32, weight_loader=ldr)

    # "Load" shard 0 only: random-but-valid packed bits decode fine; shard 1 stays
    # exactly as an absent checkpoint key leaves it (numel-0 packed, Wscale 0).
    packed0 = torch.randint(-2**15, 2**15 - 1,
                            ((m0 // 16) * (n // 16), 16 * K), dtype=torch.int16)
    layer.trellis_packed._shard_data[0] = packed0.clone()
    layer.SU._shard_data[0] = torch.where(
        torch.rand(m0) < 0.5, -torch.ones(m0), torch.ones(m0)).half()
    layer.SV._shard_data[0] = torch.where(
        torch.rand(n) < 0.5, -torch.ones(n), torch.ones(n)).half()
    layer.Wscale._shard_data[0] = torch.tensor([2.0])

    method = lm.GLQLinearMethod(None, bpw=K, codebook_type="trellis",
                                variant="3inst")
    method._setup_trellis_weights(layer, dev)

    x = torch.randn(3, n, dtype=torch.float16, device=dev)
    y_s4b = lm._glq_apply_trellis(x, layer)          # default shard-batched path
    prev = lm._GLQ_BATCH_OUT_RHT
    lm._GLQ_BATCH_OUT_RHT = False
    try:
        y_seq = lm._glq_apply_trellis(x, layer)      # per-shard fallback path
    finally:
        lm._GLQ_BATCH_OUT_RHT = prev

    meta = layer._glq_trellis_meta[0]
    ref = E8RHTLinear._trellis_linear_apply(
        x, layer.SV.get_shard(0), layer.SU.get_shard(0),
        layer.trellis_packed.get_shard(0), layer.tlut.get_shard(0),
        meta['_bn'], meta['_bm'], meta['_bnm'], meta['_bmm'], meta['wscale'],
        n, m0, n, m0, bias=None, out_dtype=x.dtype)
    zeros = torch.zeros(3, m1, dtype=x.dtype, device=dev)
    for name, y in (("s4b", y_s4b), ("sequential", y_seq)):
        assert y.shape == (3, m0 + m1), name
        assert torch.equal(y[:, :m0], ref), f"{name}: loaded shard must be unaffected"
        assert torch.equal(y[:, m0:], zeros), \
            f"{name}: unloaded shard must contribute exact zeros"
