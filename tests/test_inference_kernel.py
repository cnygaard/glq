"""Tests for GLQ Triton fused dequant+matmul inference kernel."""

import pytest
import torch

from glq.inference_kernel import glq_dequant_matmul, _fallback_dequant_matmul

try:
    import triton
    _has_triton = True
except ImportError:
    _has_triton = False

_has_cuda = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _has_cuda, reason="CUDA not available")
requires_triton = pytest.mark.skipif(
    not (_has_cuda and _has_triton), reason="CUDA + Triton required"
)


def _make_random_codebook(K=65536, D=8, device="cpu"):
    """Create a random codebook for testing."""
    return torch.randn(K, D, device=device, dtype=torch.float16)


def _make_random_inputs(B, M, N, K=65536, device="cpu"):
    """Create random test inputs."""
    x = torch.randn(B, N, device=device, dtype=torch.float16)
    # Generate as int32 then view as int16 (handles unsigned 0-65535 range)
    Qidxs = torch.randint(0, K, (M, N // 8), device=device, dtype=torch.int32).to(torch.int16)
    codebook = _make_random_codebook(K, device=device)
    Wscale = 0.5 + torch.rand(1).item()
    return x, Qidxs, codebook, Wscale


def _naive_dequant_matmul(x, Qidxs, codebook, Wscale, Qidxs2=None, codebook2=None, inv_resid_scale=0.0):
    """Reference implementation: materialize W then matmul."""
    M, n_blocks = Qidxs.shape
    N = n_blocks * 8
    # Convert int16 to unsigned indices (0-65535)
    idx = (Qidxs.long() & 0xFFFF).reshape(-1)
    W = codebook[idx].reshape(M, N)
    if Qidxs2 is not None and codebook2 is not None:
        idx2 = (Qidxs2.long() & 0xFFFF).reshape(-1)
        W2 = codebook2[idx2].reshape(M, N)
        W = W + W2 * inv_resid_scale
    return x.float() @ W.float().T * Wscale


# ---- Fallback tests (CPU, no Triton) ----

class TestFallback:
    def test_basic(self):
        B, M, N = 4, 64, 128
        x, Qidxs, cb, Wscale = _make_random_inputs(B, M, N)
        y = _fallback_dequant_matmul(x, Qidxs, cb, Wscale)
        y_ref = _naive_dequant_matmul(x, Qidxs, cb, Wscale)
        assert torch.allclose(y, y_ref, atol=1e-3, rtol=1e-3)

    def test_two_stage(self):
        B, M, N = 4, 64, 128
        x, Qidxs, cb, Wscale = _make_random_inputs(B, M, N)
        Qidxs2 = torch.randint(0, 256, (M, N // 8), dtype=torch.int32).to(torch.int16)
        cb2 = _make_random_codebook(256)
        inv_rs = 0.05

        y = _fallback_dequant_matmul(x, Qidxs, cb, Wscale, Qidxs2, cb2, inv_rs)
        y_ref = _naive_dequant_matmul(x, Qidxs, cb, Wscale, Qidxs2, cb2, inv_rs)
        assert torch.allclose(y, y_ref, atol=1e-3, rtol=1e-3)


# ---- Triton kernel tests (require CUDA + Triton) ----

@requires_triton
class TestTritonKernel:
    def test_matvec_basic(self):
        """B=1 triggers matvec kernel."""
        B, M, N = 1, 128, 256
        x, Qidxs, cb, Wscale = _make_random_inputs(B, M, N, device="cuda")
        y = glq_dequant_matmul(x, Qidxs, cb, Wscale)
        y_ref = _naive_dequant_matmul(x, Qidxs, cb, Wscale)
        assert y.shape == (B, M)
        assert torch.allclose(y.cpu(), y_ref.cpu(), atol=5e-2, rtol=5e-2), \
            f"max diff: {(y.cpu() - y_ref.cpu()).abs().max().item()}"

    def test_matmul_basic(self):
        """B>4 triggers matmul kernel."""
        B, M, N = 32, 128, 256
        x, Qidxs, cb, Wscale = _make_random_inputs(B, M, N, device="cuda")
        y = glq_dequant_matmul(x, Qidxs, cb, Wscale)
        y_ref = _naive_dequant_matmul(x, Qidxs, cb, Wscale)
        assert y.shape == (B, M)
        assert torch.allclose(y.cpu(), y_ref.cpu(), atol=5e-2, rtol=5e-2), \
            f"max diff: {(y.cpu() - y_ref.cpu()).abs().max().item()}"

    @pytest.mark.parametrize("B", [1, 8, 32, 128])
    @pytest.mark.parametrize("M,N", [(128, 256), (256, 512), (512, 1024)])
    def test_shapes(self, B, M, N):
        """Test various shapes."""
        x, Qidxs, cb, Wscale = _make_random_inputs(B, M, N, device="cuda")
        y = glq_dequant_matmul(x, Qidxs, cb, Wscale)
        y_ref = _naive_dequant_matmul(x, Qidxs, cb, Wscale)
        assert y.shape == (B, M)
        assert torch.allclose(y.cpu(), y_ref.cpu(), atol=0.1, rtol=0.1), \
            f"Shape ({B},{M},{N}): max diff {(y.cpu()-y_ref.cpu()).abs().max():.4f}"

    def test_two_stage_matvec(self):
        """Two-stage (3/4bpw) with B=1."""
        B, M, N = 1, 128, 256
        x, Qidxs, cb, Wscale = _make_random_inputs(B, M, N, device="cuda")
        Qidxs2 = torch.randint(0, 256, (M, N // 8), device="cuda", dtype=torch.int16)
        cb2 = _make_random_codebook(256, device="cuda")
        inv_rs = 0.05

        y = glq_dequant_matmul(x, Qidxs, cb, Wscale, Qidxs2, cb2, inv_rs)
        y_ref = _naive_dequant_matmul(x, Qidxs, cb, Wscale, Qidxs2, cb2, inv_rs)
        assert torch.allclose(y.cpu(), y_ref.cpu(), atol=5e-2, rtol=5e-2), \
            f"max diff: {(y.cpu() - y_ref.cpu()).abs().max().item()}"

    def test_two_stage_matmul(self):
        """Two-stage (3/4bpw) with B>4."""
        B, M, N = 32, 128, 256
        x, Qidxs, cb, Wscale = _make_random_inputs(B, M, N, device="cuda")
        Qidxs2 = torch.randint(0, 256, (M, N // 8), device="cuda", dtype=torch.int16)
        cb2 = _make_random_codebook(256, device="cuda")
        inv_rs = 0.05

        y = glq_dequant_matmul(x, Qidxs, cb, Wscale, Qidxs2, cb2, inv_rs)
        y_ref = _naive_dequant_matmul(x, Qidxs, cb, Wscale, Qidxs2, cb2, inv_rs)
        assert torch.allclose(y.cpu(), y_ref.cpu(), atol=0.1, rtol=0.1), \
            f"max diff: {(y.cpu() - y_ref.cpu()).abs().max().item()}"

    def test_max_indices(self):
        """Test with maximum index values (0 and 65535)."""
        B, M, N = 4, 64, 128
        K = 65536
        x = torch.randn(B, N, device="cuda", dtype=torch.float16)
        cb = _make_random_codebook(K, device="cuda")
        Wscale = 1.0

        # Indices at boundaries
        Qidxs = torch.zeros(M, N // 8, device="cuda", dtype=torch.int16)
        Qidxs[::2] = -1  # int16(-1) = uint16(65535)

        y = glq_dequant_matmul(x, Qidxs, cb, Wscale)
        y_ref = _naive_dequant_matmul(x, Qidxs, cb, Wscale)
        assert torch.allclose(y.cpu(), y_ref.cpu(), atol=5e-2, rtol=5e-2)

    def test_cpu_fallback(self):
        """CPU inputs should use fallback path."""
        B, M, N = 4, 64, 128
        x, Qidxs, cb, Wscale = _make_random_inputs(B, M, N, device="cpu")
        y = glq_dequant_matmul(x, Qidxs, cb, Wscale)
        y_ref = _naive_dequant_matmul(x, Qidxs, cb, Wscale)
        assert torch.allclose(y, y_ref, atol=1e-3, rtol=1e-3)


# ---- Fallback edge cases (CPU, broader coverage) ----

class TestFallbackEdgeCases:
    def test_wscale_large(self):
        """Large Wscale should scale output proportionally."""
        B, M, N = 2, 32, 64
        x, Qidxs, cb, _ = _make_random_inputs(B, M, N)
        y1 = _fallback_dequant_matmul(x, Qidxs, cb, 1.0)
        y10 = _fallback_dequant_matmul(x, Qidxs, cb, 10.0)
        torch.testing.assert_close(y10, y1 * 10.0, atol=1e-3, rtol=1e-3)

    def test_wscale_negative(self):
        """Negative Wscale should negate output."""
        B, M, N = 2, 32, 64
        x, Qidxs, cb, _ = _make_random_inputs(B, M, N)
        y_pos = _fallback_dequant_matmul(x, Qidxs, cb, 1.0)
        y_neg = _fallback_dequant_matmul(x, Qidxs, cb, -1.0)
        torch.testing.assert_close(y_neg, -y_pos, atol=1e-3, rtol=1e-3)

    def test_zero_input(self):
        """Zero input should produce zero output."""
        B, M, N = 2, 32, 64
        x = torch.zeros(B, N, dtype=torch.float16)
        Qidxs = torch.randint(0, 65536, (M, N // 8), dtype=torch.int32).to(torch.int16)
        cb = _make_random_codebook()
        y = _fallback_dequant_matmul(x, Qidxs, cb, 1.0)
        assert y.abs().max().item() < 1e-6

    def test_single_row(self):
        """M=1: single output neuron."""
        B, M, N = 4, 1, 64
        x, Qidxs, cb, Wscale = _make_random_inputs(B, M, N)
        y = _fallback_dequant_matmul(x, Qidxs, cb, Wscale)
        y_ref = _naive_dequant_matmul(x, Qidxs, cb, Wscale)
        assert y.shape == (B, 1)
        assert torch.allclose(y, y_ref, atol=1e-3, rtol=1e-3)

    def test_single_batch(self):
        """B=1."""
        B, M, N = 1, 64, 128
        x, Qidxs, cb, Wscale = _make_random_inputs(B, M, N)
        y = _fallback_dequant_matmul(x, Qidxs, cb, Wscale)
        y_ref = _naive_dequant_matmul(x, Qidxs, cb, Wscale)
        assert y.shape == (1, M)
        assert torch.allclose(y, y_ref, atol=1e-3, rtol=1e-3)

    def test_large_batch(self):
        """B=256 large batch."""
        B, M, N = 256, 32, 64
        x, Qidxs, cb, Wscale = _make_random_inputs(B, M, N)
        y = _fallback_dequant_matmul(x, Qidxs, cb, Wscale)
        y_ref = _naive_dequant_matmul(x, Qidxs, cb, Wscale)
        assert torch.allclose(y, y_ref, atol=1e-3, rtol=1e-3)

    def test_all_zero_indices(self):
        """All indices are 0 — uses only codebook[0]."""
        B, M, N = 4, 32, 64
        x = torch.randn(B, N, dtype=torch.float16)
        Qidxs = torch.zeros(M, N // 8, dtype=torch.int16)
        cb = _make_random_codebook()
        y = _fallback_dequant_matmul(x, Qidxs, cb, 1.0)
        y_ref = _naive_dequant_matmul(x, Qidxs, cb, 1.0)
        assert torch.allclose(y, y_ref, atol=1e-3, rtol=1e-3)

    def test_all_max_indices(self):
        """All indices are 65535 (int16 = -1) — edge of codebook."""
        B, M, N = 4, 32, 64
        x = torch.randn(B, N, dtype=torch.float16)
        Qidxs = torch.full((M, N // 8), -1, dtype=torch.int16)  # 65535 unsigned
        cb = _make_random_codebook()
        y = _fallback_dequant_matmul(x, Qidxs, cb, 1.0)
        y_ref = _naive_dequant_matmul(x, Qidxs, cb, 1.0)
        assert torch.allclose(y, y_ref, atol=1e-3, rtol=1e-3)

    def test_small_codebook_two_stage(self):
        """Two-stage with 256-entry secondary codebook."""
        B, M, N = 4, 32, 64
        x, Qidxs, cb, Wscale = _make_random_inputs(B, M, N)
        K2 = 256
        Qidxs2 = torch.randint(0, K2, (M, N // 8), dtype=torch.int32).to(torch.int16)
        cb2 = _make_random_codebook(K2)
        inv_rs = 0.1

        y = _fallback_dequant_matmul(x, Qidxs, cb, Wscale, Qidxs2, cb2, inv_rs)
        y_ref = _naive_dequant_matmul(x, Qidxs, cb, Wscale, Qidxs2, cb2, inv_rs)
        assert torch.allclose(y, y_ref, atol=1e-3, rtol=1e-3)

    def test_inv_resid_scale_zero(self):
        """inv_resid_scale=0 should effectively ignore stage-2."""
        B, M, N = 4, 32, 64
        x, Qidxs, cb, Wscale = _make_random_inputs(B, M, N)
        Qidxs2 = torch.randint(0, 256, (M, N // 8), dtype=torch.int32).to(torch.int16)
        cb2 = _make_random_codebook(256)

        y_no_stage2 = _fallback_dequant_matmul(x, Qidxs, cb, Wscale)
        y_with_zero = _fallback_dequant_matmul(x, Qidxs, cb, Wscale, Qidxs2, cb2, 0.0)
        torch.testing.assert_close(y_no_stage2, y_with_zero, atol=1e-3, rtol=1e-3)

    def test_output_dtype_is_float32(self):
        """Output should always be float32."""
        B, M, N = 2, 32, 64
        x, Qidxs, cb, Wscale = _make_random_inputs(B, M, N)
        y = _fallback_dequant_matmul(x, Qidxs, cb, Wscale)
        assert y.dtype == torch.float32

    def test_glq_dequant_matmul_dispatches_to_fallback_on_cpu(self):
        """glq_dequant_matmul should use fallback on CPU."""
        B, M, N = 4, 32, 64
        x, Qidxs, cb, Wscale = _make_random_inputs(B, M, N)
        y = glq_dequant_matmul(x, Qidxs, cb, Wscale)
        y_ref = _naive_dequant_matmul(x, Qidxs, cb, Wscale)
        assert torch.allclose(y, y_ref, atol=1e-3, rtol=1e-3)
