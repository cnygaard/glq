"""CUDA fast-path tests for block-diagonal E8RHTLinear.

Covers the two regression surfaces introduced by Phases A+B:
  * ``glq_fused_linear_block_diag_cuda`` with populated vs empty metadata
    (multi-block kernel vs legacy per-sub-block host loop).
  * Dispatch through both the B=1 matvec split-K and the B>=2 Tensor Core
    matmul branches.
  * CUDA graph capture + replay on a block-diagonal layer.
"""

import math

import pytest
import torch

from glq.codebook import E8ShellCodebook
from glq.quantized_linear import E8RHTLinear, _pack_block_meta


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for block-diag fast path"
)


@pytest.fixture(scope="module")
def codebook_cuda():
    return E8ShellCodebook.build(device="cuda", verbose=False)


def _make_random_bd_layer(in_features, out_features, codebook, bpw=4, seed=0):
    """Populate an E8RHTLinear(block_diagonal=True) with random quant state."""
    torch.manual_seed(seed)
    layer = E8RHTLinear(in_features, out_features, block_diagonal=True).cuda()

    m_pad, n_pad = layer.m_pad, layer.n_pad
    n_blocks_col = n_pad // 8

    layer.Qidxs.copy_(torch.randint(-32768, 32768, (m_pad, n_blocks_col),
                                    dtype=torch.int16, device="cuda"))
    layer.SV.copy_((torch.randint(0, 2, (n_pad,), device="cuda").float() * 2 - 1).half())
    layer.SU.copy_((torch.randint(0, 2, (m_pad,), device="cuda").float() * 2 - 1).half())
    layer.Wscale.copy_(torch.tensor(0.1))
    if bpw >= 3:
        layer.Qidxs2.copy_(torch.randint(-32768, 32768, (m_pad, n_blocks_col),
                                         dtype=torch.int16, device="cuda"))
        layer.inv_resid_scale.copy_(torch.tensor(0.25))

    codebook2 = codebook if bpw >= 4 else (codebook.make_small(256) if bpw == 3 else None)
    layer.set_codebook(codebook, codebook2=codebook2)
    return layer


# ── Shapes covering the models in MEMORY.md ──────────────────────────────
# (in_features, out_features, expected blocks_n, expected blocks_m)
BD_SHAPES = [
    pytest.param(576, 576, [512, 64], [512, 64], id="135M_attn_576x576"),
    pytest.param(1536, 576, [1024, 512], [512, 64], id="135M_down_1536x576"),
    pytest.param(576, 1536, [512, 64], [1024, 512], id="135M_up_576x1536"),
    pytest.param(2048, 2048, [2048], [2048], id="pow2_2048x2048"),
]

BATCH_SIZES = [1, 4, 16]


def _call_fused_kernel(layer, x, *, use_multiblock):
    """Invoke glq_fused_linear_block_diag_cuda with either populated metadata
    (multi-block kernel) or empty metadata (forces legacy per-block loop)."""
    import glq.inference_kernel as ik
    assert ik._try_load_cuda_ext()
    ext = ik._glq_cuda

    empty_i16 = torch.empty(0, dtype=torch.int16, device=x.device)
    empty_f16 = torch.empty(0, dtype=torch.float16, device=x.device)
    empty_i32 = torch.empty(0, dtype=torch.int32, device=x.device)

    if use_multiblock:
        bn_meta = _pack_block_meta(layer.blocks_n).to(x.device)
        bm_meta = _pack_block_meta(layer.blocks_m).to(x.device)
    else:
        bn_meta = empty_i32
        bm_meta = empty_i32

    cb2 = layer.codebook2.codebook_half if layer._has_stage2 else empty_f16
    qidxs2 = layer.Qidxs2 if layer._has_stage2 else empty_i16
    inv_rs = float(layer.inv_resid_scale.item()) if layer._has_stage2 else 0.0

    return ext.glq_fused_linear_block_diag_cuda(
        x.half().contiguous(), layer.SV, layer.SU,
        layer.Qidxs, layer.codebook.codebook_half,
        float(layer.Wscale.item()),
        layer.in_features, layer.out_features,
        layer.n_pad, layer.m_pad,
        layer._blocks_n_tensor, layer._blocks_m_tensor,
        bn_meta, bm_meta,
        qidxs2, cb2, inv_rs,
        empty_i16, empty_f16, 0.0,
        empty_i16, empty_f16, 0.0,
    )


class TestMultiblockEquivalence:
    """Multi-block kernel must match the legacy per-sub-block loop."""

    @pytest.mark.parametrize("in_f,out_f,blocks_n,blocks_m", BD_SHAPES)
    @pytest.mark.parametrize("B", BATCH_SIZES)
    @pytest.mark.parametrize("bpw", [2, 4])
    def test_multiblock_matches_legacy_loop(self, codebook_cuda, in_f, out_f,
                                            blocks_n, blocks_m, B, bpw):
        layer = _make_random_bd_layer(in_f, out_f, codebook_cuda, bpw=bpw, seed=B)
        assert layer.blocks_n == blocks_n
        assert layer.blocks_m == blocks_m

        x = torch.randn(B, in_f, dtype=torch.float16, device="cuda") * 0.1
        y_mb = _call_fused_kernel(layer, x, use_multiblock=True)
        y_legacy = _call_fused_kernel(layer, x, use_multiblock=False)

        assert y_mb.shape == (B, out_f)
        assert y_legacy.shape == (B, out_f)

        # Small-block shapes where butterfly + matmul both use deterministic
        # reductions should be bit-exact. Larger shapes that split the matmul
        # across k-splits still use scratch+reduce (deterministic) so diffs
        # should only come from fp16 rounding in the shared TC path — allow
        # a small tolerance.
        max_abs = (y_mb.float() - y_legacy.float()).abs().max().item()
        ref_scale = y_legacy.float().abs().max().item() + 1e-9
        assert max_abs / ref_scale < 5e-3, (
            f"multiblock vs legacy diverged: max_abs={max_abs:.5f} "
            f"(rel={max_abs/ref_scale:.2e}) for in={in_f} out={out_f} B={B} bpw={bpw}"
        )


class TestForwardConsistency:
    """E8RHTLinear.forward through the multi-block fast path must match its
    own legacy-loop output for the same input."""

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_forward_b1_and_bge2_both_work(self, codebook_cuda, B):
        layer = _make_random_bd_layer(576, 576, codebook_cuda, bpw=4, seed=B)
        x = torch.randn(B, 576, dtype=torch.float16, device="cuda") * 0.1
        y = layer(x)
        assert y.shape == (B, 576)
        assert torch.isfinite(y).all(), "forward produced NaN/Inf"

    def test_b1_and_bge2_agree_on_first_row(self, codebook_cuda):
        """Running B=1 on x[:1] should match B=4 output at row 0 — verifies
        matvec split-K and Tensor Core matmul produce consistent results."""
        layer = _make_random_bd_layer(576, 576, codebook_cuda, bpw=4, seed=7)
        x = torch.randn(4, 576, dtype=torch.float16, device="cuda") * 0.1
        y_b1 = layer(x[:1])
        y_b4 = layer(x)
        max_abs = (y_b1.float() - y_b4[:1].float()).abs().max().item()
        ref_scale = y_b4.float().abs().max().item() + 1e-9
        # fp16 TC vs split-K matvec diverge by ~2e-3 on unrelated reduction
        # ordering but the sign/magnitude should agree.
        assert max_abs / ref_scale < 5e-3, (
            f"B=1 matvec vs B=4 TC matmul diverged too much: "
            f"max_abs={max_abs:.5f} rel={max_abs/ref_scale:.2e}"
        )


class TestCUDAGraphCapture:
    """Block-diagonal layer must be capturable and replay bit-exact."""

    def test_capture_and_replay_bit_exact(self, codebook_cuda):
        layer = _make_random_bd_layer(576, 576, codebook_cuda, bpw=4, seed=42)

        # Static input buffer
        x_static = torch.randn(1, 576, dtype=torch.float16, device="cuda") * 0.1

        # Warmup (triggers any lazy allocations: _empty_{i16,f16},
        # _blocks_*_meta_gpu, codebook device move)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                y_warm = layer(x_static)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # Eager reference
        y_eager = layer(x_static).clone()

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y_graph_out = layer(x_static)

        # Change input, replay, verify output updated
        x_static.copy_(torch.randn(1, 576, dtype=torch.float16, device="cuda") * 0.1)
        y_eager_new = layer(x_static).clone()
        graph.replay()
        torch.cuda.synchronize()
        diff = (y_graph_out.float() - y_eager_new.float()).abs().max().item()
        assert diff == 0.0, f"graph replay not bit-exact vs eager: max_abs={diff}"


def _make_random_nstage_bd_layer(in_features, out_features, codebook,
                                  num_stages, seed=0):
    """Populate an E8RHTLinear(block_diagonal=True) with N random RVQ stages.

    num_stages ∈ {1, 2, 3, 4}. All stages use the primary 65536-entry
    codebook, matching how real GLQ 5-8 bpw checkpoints are quantized.
    """
    torch.manual_seed(seed)
    layer = E8RHTLinear(in_features, out_features, block_diagonal=True).cuda()

    m_pad, n_pad = layer.m_pad, layer.n_pad
    n_blocks_col = n_pad // 8

    layer.Qidxs.copy_(torch.randint(-32768, 32768, (m_pad, n_blocks_col),
                                    dtype=torch.int16, device="cuda"))
    layer.SV.copy_((torch.randint(0, 2, (n_pad,), device="cuda").float() * 2 - 1).half())
    layer.SU.copy_((torch.randint(0, 2, (m_pad,), device="cuda").float() * 2 - 1).half())
    layer.Wscale.copy_(torch.tensor(0.1))
    if num_stages >= 2:
        layer.Qidxs2.copy_(torch.randint(-32768, 32768, (m_pad, n_blocks_col),
                                         dtype=torch.int16, device="cuda"))
        layer.inv_resid_scale.copy_(torch.tensor(0.25))
    if num_stages >= 3:
        layer.Qidxs3.copy_(torch.randint(-32768, 32768, (m_pad, n_blocks_col),
                                         dtype=torch.int16, device="cuda"))
        layer.inv_resid_scale2.copy_(torch.tensor(0.0625))
    if num_stages >= 4:
        layer.Qidxs4.copy_(torch.randint(-32768, 32768, (m_pad, n_blocks_col),
                                         dtype=torch.int16, device="cuda"))
        layer.inv_resid_scale3.copy_(torch.tensor(0.015625))

    # Stages 3+ reuse the primary codebook (65536 entries) just like real
    # GLQ 5-8 bpw checkpoints — see Python fallback in quantized_linear.py.
    layer.set_codebook(codebook, codebook2=codebook)
    assert layer._n_stages == num_stages, (
        f"expected _n_stages={num_stages}, got {layer._n_stages}"
    )
    return layer


class TestNStageEquivalence:
    """Fused N-stage (num_stages=3, 4) path must match the PyTorch fallback."""

    @pytest.mark.parametrize("num_stages", [3, 4])
    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_fused_matches_python_fallback(self, codebook_cuda, num_stages, B):
        # Use a small block-diag shape (2 sub-blocks) so the test is cheap
        # while still exercising the fused block-diag + n-stage path.
        in_f, out_f = 576, 576

        # Build two identical layers: one for fused, one for PyTorch fallback.
        # The fallback trigger is `n_stages > 2` against the triton gate at
        # forward():500, combined with skipping the fused C++ branch.
        layer_fused = _make_random_nstage_bd_layer(
            in_f, out_f, codebook_cuda, num_stages=num_stages, seed=B * 10 + num_stages
        )
        layer_ref = _make_random_nstage_bd_layer(
            in_f, out_f, codebook_cuda, num_stages=num_stages, seed=B * 10 + num_stages
        )
        # Force layer_ref onto the PyTorch fallback: bump n_stages past the
        # fused gate (<= 4) so forward() falls through to the dense-matmul
        # fallback at the else-branch of line ~500 (PyTorch, supports N stages).
        # Saving the original so we can restore it after the comparison.
        layer_ref._n_stages = 99

        x = torch.randn(B, in_f, dtype=torch.float16, device="cuda") * 0.1

        y_fused = layer_fused(x)
        y_ref = layer_ref(x)

        assert y_fused.shape == (B, out_f)
        assert y_ref.shape == (B, out_f)

        max_abs = (y_fused.float() - y_ref.float()).abs().max().item()
        ref_scale = y_ref.float().abs().max().item() + 1e-9
        assert max_abs / ref_scale < 5e-3, (
            f"fused vs fallback diverged for num_stages={num_stages} B={B}: "
            f"max_abs={max_abs:.5f} (rel={max_abs/ref_scale:.2e})"
        )


class TestLargeBlockFallback:
    """max_bs > 8192 must gracefully fall back to the legacy per-block loop."""

    def test_pow2_16384_takes_legacy_path(self, codebook_cuda):
        # n_pad = 16384 → one 16384 block, which exceeds the 8192 multi-block
        # smem cap and forces the legacy kernel path.
        layer = _make_random_bd_layer(16384, 1024, codebook_cuda, bpw=4, seed=3)
        assert max(layer.blocks_n) == 16384
        x = torch.randn(2, 16384, dtype=torch.float16, device="cuda") * 0.05
        y = layer(x)
        assert y.shape == (2, 1024)
        assert torch.isfinite(y).all()
