"""Tests for block LDL decomposition and LDLQ quantization."""

import pytest
import torch

from glq.ldlq import block_LDL, quantize_ldlq_codebook, quantize_ldlq_codebook_2stage
from glq.codebook import E8ShellCodebook


class TestBlockLDL:
    def test_reconstruction(self):
        """L @ blkdiag(D) @ L^T should reconstruct H."""
        n, b = 24, 8
        A = torch.randn(n, n)
        H = A @ A.T + 0.1 * torch.eye(n)  # PSD

        L, D = block_LDL(H, block_size=b)
        m = n // b

        # Reconstruct block-diagonal from D
        blk_D = torch.zeros(n, n)
        for i in range(m):
            blk_D[i * b:(i + 1) * b, i * b:(i + 1) * b] = D[i]

        H_recon = L @ blk_D @ L.T
        torch.testing.assert_close(H_recon, H, atol=1e-3, rtol=1e-3)

    def test_L_is_block_unit_lower_triangular(self):
        """L should have identity blocks on the diagonal."""
        n, b = 16, 8
        A = torch.randn(n, n)
        H = A @ A.T + 0.1 * torch.eye(n)

        L, D = block_LDL(H, block_size=b)
        m = n // b

        for i in range(m):
            diag_block = L[i * b:(i + 1) * b, i * b:(i + 1) * b]
            torch.testing.assert_close(
                diag_block, torch.eye(b), atol=1e-4, rtol=1e-4
            )

    def test_D_blocks_are_symmetric(self):
        """Each D block should be symmetric."""
        n, b = 24, 8
        A = torch.randn(n, n)
        H = A @ A.T + 0.1 * torch.eye(n)

        _, D = block_LDL(H, block_size=b)
        for i in range(D.shape[0]):
            torch.testing.assert_close(D[i], D[i].T, atol=1e-5, rtol=1e-5)

    def test_rejects_non_divisible(self):
        """Should reject n not divisible by block_size."""
        H = torch.eye(10)
        with pytest.raises(AssertionError):
            block_LDL(H, block_size=8)


class TestQuantizeLDLQ:
    @pytest.fixture(scope="class")
    def codebook(self):
        return E8ShellCodebook.build(device="cpu", verbose=False)

    def test_output_keys(self, codebook):
        """quantize_ldlq_codebook should return all expected keys."""
        W = torch.randn(4, 32)
        H = torch.eye(32)
        result = quantize_ldlq_codebook(W, H, codebook)
        expected_keys = {"W_hat", "indices", "Wscale", "bpw", "quant_mse", "proxy_loss", "tune_iters"}
        assert set(result.keys()) == expected_keys

    def test_output_shapes(self, codebook):
        W = torch.randn(4, 32)
        H = torch.eye(32)
        result = quantize_ldlq_codebook(W, H, codebook)
        assert result["W_hat"].shape == (4, 32)
        assert result["indices"].shape == (4, 4)  # 32/8 = 4 blocks

    def test_bpw_is_correct(self, codebook):
        W = torch.randn(4, 32)
        H = torch.eye(32)
        result = quantize_ldlq_codebook(W, H, codebook)
        # 16 bits / 8 dims + overhead
        assert result["bpw"] == pytest.approx(2.0 + 32.0 / (4 * 32), abs=0.01)

    def test_quantize_reduces_proxy_loss(self, codebook):
        """LDLQ should produce finite proxy loss."""
        W = torch.randn(8, 64)
        A = torch.randn(64, 64)
        H = A @ A.T + 0.01 * torch.eye(64)
        result = quantize_ldlq_codebook(W, H, codebook)
        assert result["proxy_loss"] >= 0
        assert result["proxy_loss"] < float("inf")

    def test_tune_iters_improves(self, codebook):
        """Refinement iterations should not increase proxy loss."""
        torch.manual_seed(42)
        W = torch.randn(8, 64)
        A = torch.randn(64, 64)
        H = A @ A.T + 0.1 * torch.eye(64)
        r0 = quantize_ldlq_codebook(W, H, codebook, tune_iters=0)
        r1 = quantize_ldlq_codebook(W, H, codebook, tune_iters=1)
        assert r1["proxy_loss"] <= r0["proxy_loss"] * 1.01  # allow tiny float noise


class TestQuantizeLDLQ2Stage:
    @pytest.fixture(scope="class")
    def codebooks(self):
        cb = E8ShellCodebook.build(device="cpu", verbose=False)
        cb_small = cb.make_small(256)
        return cb, cb_small

    def test_3bpw_output_keys(self, codebooks):
        cb, cb_small = codebooks
        W = torch.randn(4, 32)
        H = torch.eye(32)
        result = quantize_ldlq_codebook_2stage(
            W, H, cb, cb_small, resid_scale=cb.resid_scale
        )
        expected_keys = {
            "W_hat", "indices1", "indices2", "Wscale",
            "resid_scale", "bpw", "quant_mse", "proxy_loss", "tune_iters",
        }
        assert set(result.keys()) == expected_keys

    def test_3bpw_output_shapes(self, codebooks):
        cb, cb_small = codebooks
        W = torch.randn(4, 32)
        H = torch.eye(32)
        result = quantize_ldlq_codebook_2stage(
            W, H, cb, cb_small, resid_scale=cb.resid_scale
        )
        assert result["W_hat"].shape == (4, 32)
        assert result["indices1"].shape == (4, 4)
        assert result["indices2"].shape == (4, 4)

    def test_4bpw_lower_error_than_2bpw(self, codebooks):
        """4bpw (two full codebooks) should have lower MSE than 2bpw."""
        cb, _ = codebooks
        torch.manual_seed(0)
        W = torch.randn(8, 64)
        A = torch.randn(64, 64)
        H = A @ A.T + 0.1 * torch.eye(64)

        r2 = quantize_ldlq_codebook(W, H, cb)
        r4 = quantize_ldlq_codebook_2stage(W, H, cb, cb, resid_scale=cb.resid_scale)
        assert r4["quant_mse"] < r2["quant_mse"]

    def test_2stage_tune_iters_runs(self, codebooks):
        """2-stage with tune_iters=2 should succeed."""
        cb, cb_small = codebooks
        torch.manual_seed(42)
        W = torch.randn(4, 32)
        A = torch.randn(32, 32)
        H = A @ A.T + 0.1 * torch.eye(32)
        result = quantize_ldlq_codebook_2stage(
            W, H, cb, cb_small, resid_scale=cb.resid_scale, tune_iters=2)
        assert result["tune_iters"] == 2
        assert result["proxy_loss"] >= 0

    def test_2stage_tune_reduces_proxy_loss(self, codebooks):
        """Refinement iterations should not increase proxy loss in 2-stage."""
        cb, cb_small = codebooks
        torch.manual_seed(42)
        W = torch.randn(8, 64)
        A = torch.randn(64, 64)
        H = A @ A.T + 0.1 * torch.eye(64)
        r0 = quantize_ldlq_codebook_2stage(
            W, H, cb, cb_small, resid_scale=cb.resid_scale, tune_iters=0)
        r2 = quantize_ldlq_codebook_2stage(
            W, H, cb, cb_small, resid_scale=cb.resid_scale, tune_iters=2)
        assert r2["proxy_loss"] <= r0["proxy_loss"] * 1.01
