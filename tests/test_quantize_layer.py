"""Tests for quantize_layer_e8_shell_rht — per-layer quantization pipeline."""

import pytest
import torch

from glq.codebook import E8ShellCodebook
from glq.quantize_model import quantize_layer_e8_shell_rht, pad_to_multiple, pad_hessian


@pytest.fixture(scope="module")
def codebook():
    return E8ShellCodebook.build(device="cpu", verbose=False)


def _random_psd(n):
    """Create a random positive semi-definite matrix."""
    A = torch.randn(n, n)
    return A @ A.T + 0.1 * torch.eye(n)


class TestPadHelpers:
    def test_pad_to_multiple_exact(self):
        W = torch.randn(4, 32)
        W_pad, n_orig, pad = pad_to_multiple(W, 8)
        assert W_pad.shape == (4, 32)
        assert n_orig == 32
        assert pad == 0

    def test_pad_to_multiple_needs_padding(self):
        W = torch.randn(4, 30)
        W_pad, n_orig, pad = pad_to_multiple(W, 8)
        assert W_pad.shape == (4, 32)
        assert n_orig == 30
        assert pad == 2
        # Padded columns should be zero
        assert W_pad[:, 30:].abs().max() == 0

    def test_pad_hessian_exact(self):
        H = torch.eye(32)
        H_pad = pad_hessian(H, 8)
        assert H_pad.shape == (32, 32)

    def test_pad_hessian_needs_padding(self):
        H = torch.eye(30)
        H_pad = pad_hessian(H, 8)
        assert H_pad.shape == (32, 32)
        # Original block preserved
        torch.testing.assert_close(H_pad[:30, :30], H)
        # Padded diagonal has tiny values for regularization
        for i in range(30, 32):
            assert H_pad[i, i].item() > 0


class TestQuantizeLayer2bpw:
    def test_output_keys(self, codebook):
        W = torch.randn(8, 64)
        H = _random_psd(64)
        W_hat, artifacts, metrics = quantize_layer_e8_shell_rht(W, H, codebook, bpw=2)
        assert set(artifacts.keys()) == {'Qidxs', 'SU', 'SV', 'Wscale'}
        assert set(metrics.keys()) == {'sqnr', 'bpw', 'Wscale'}
        # 2bpw: no Qidxs2 or inv_resid_scale (omitted to save disk space)
        assert 'Qidxs2' not in artifacts
        assert 'inv_resid_scale' not in artifacts

    def test_artifact_dtypes(self, codebook):
        W = torch.randn(8, 64)
        H = _random_psd(64)
        _, artifacts, _ = quantize_layer_e8_shell_rht(W, H, codebook, bpw=2)
        assert artifacts['Qidxs'].dtype == torch.int16
        assert artifacts['SU'].dtype == torch.float16
        assert artifacts['SV'].dtype == torch.float16
        assert artifacts['Wscale'].dtype == torch.float32

    def test_artifact_shapes(self, codebook):
        W = torch.randn(8, 64)
        H = _random_psd(64)
        _, artifacts, _ = quantize_layer_e8_shell_rht(W, H, codebook, bpw=2)
        # m_pad = 8, n_pad = 64
        assert artifacts['Qidxs'].shape == (8, 8)  # (8, 64//8)
        assert artifacts['SU'].shape == (8,)
        assert artifacts['SV'].shape == (64,)

    def test_what_shape(self, codebook):
        W = torch.randn(8, 64)
        H = _random_psd(64)
        W_hat, _, _ = quantize_layer_e8_shell_rht(W, H, codebook, bpw=2)
        assert W_hat.shape == W.shape

    def test_sqnr_positive(self, codebook):
        W = torch.randn(8, 64)
        H = _random_psd(64)
        _, _, metrics = quantize_layer_e8_shell_rht(W, H, codebook, bpw=2)
        assert metrics['sqnr'] > 0

    def test_bpw_in_metrics(self, codebook):
        W = torch.randn(8, 64)
        H = _random_psd(64)
        _, _, metrics = quantize_layer_e8_shell_rht(W, H, codebook, bpw=2)
        # 2bpw = 16/8 + tiny overhead
        assert 2.0 <= metrics['bpw'] <= 2.1

    def test_non_power_of_2_dims(self, codebook):
        """Non-power-of-2 dimensions should pad internally."""
        W = torch.randn(10, 48)
        H = _random_psd(48)
        W_hat, artifacts, metrics = quantize_layer_e8_shell_rht(W, H, codebook, bpw=2)
        assert W_hat.shape == (10, 48)
        assert metrics['sqnr'] > 0
        # Padded dims: m_pad=16, n_pad=64
        assert artifacts['SU'].shape[0] == 16
        assert artifacts['SV'].shape[0] == 64


class TestQuantizeLayer3bpw:
    def test_output_keys(self, codebook):
        W = torch.randn(8, 64)
        H = _random_psd(64)
        _, artifacts, _ = quantize_layer_e8_shell_rht(W, H, codebook, bpw=3)
        assert 'Qidxs' in artifacts
        assert 'Qidxs2' in artifacts
        assert 'inv_resid_scale' in artifacts
        assert 'SU' in artifacts
        assert 'SV' in artifacts
        assert 'Wscale' in artifacts

    def test_artifact_shapes(self, codebook):
        W = torch.randn(8, 64)
        H = _random_psd(64)
        _, artifacts, _ = quantize_layer_e8_shell_rht(W, H, codebook, bpw=3)
        assert artifacts['Qidxs'].shape == artifacts['Qidxs2'].shape
        assert artifacts['Qidxs'].dtype == torch.int16
        assert artifacts['Qidxs2'].dtype == torch.int16
        assert artifacts['inv_resid_scale'].dtype == torch.float32

    def test_bpw(self, codebook):
        W = torch.randn(8, 64)
        H = _random_psd(64)
        _, _, metrics = quantize_layer_e8_shell_rht(W, H, codebook, bpw=3)
        # 3bpw = (16+8)/8 + tiny overhead
        assert 3.0 <= metrics['bpw'] <= 3.1

    def test_sqnr_better_than_2bpw(self, codebook):
        """3bpw should have higher SQNR than 2bpw."""
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        H = _random_psd(64)
        _, _, m2 = quantize_layer_e8_shell_rht(W, H, codebook, bpw=2)
        _, _, m3 = quantize_layer_e8_shell_rht(W, H, codebook, bpw=3)
        assert m3['sqnr'] > m2['sqnr']


class TestQuantizeLayer4bpw:
    def test_output_keys(self, codebook):
        W = torch.randn(8, 64)
        H = _random_psd(64)
        _, artifacts, _ = quantize_layer_e8_shell_rht(W, H, codebook, bpw=4)
        assert 'Qidxs' in artifacts
        assert 'Qidxs2' in artifacts
        assert 'inv_resid_scale' in artifacts

    def test_bpw(self, codebook):
        W = torch.randn(8, 64)
        H = _random_psd(64)
        _, _, metrics = quantize_layer_e8_shell_rht(W, H, codebook, bpw=4)
        # 4bpw = (16+16)/8 + tiny overhead
        assert 4.0 <= metrics['bpw'] <= 4.1

    def test_sqnr_better_than_3bpw(self, codebook):
        """4bpw should have higher SQNR than 3bpw."""
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        H = _random_psd(64)
        _, _, m3 = quantize_layer_e8_shell_rht(W, H, codebook, bpw=3)
        _, _, m4 = quantize_layer_e8_shell_rht(W, H, codebook, bpw=4)
        assert m4['sqnr'] > m3['sqnr']


class TestTuneIters:
    def test_tune_does_not_increase_proxy_loss(self, codebook):
        """Refinement iterations should not worsen quality."""
        torch.manual_seed(42)
        W = torch.randn(8, 64)
        A = torch.randn(64, 64)
        H = A @ A.T + 0.1 * torch.eye(64)

        _, _, m0 = quantize_layer_e8_shell_rht(W, H, codebook, bpw=2, tune_iters=0)
        _, _, m1 = quantize_layer_e8_shell_rht(W, H, codebook, bpw=2, tune_iters=1)
        # Allow small float noise
        assert m1['sqnr'] >= m0['sqnr'] * 0.99
