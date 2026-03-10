"""End-to-end tests: quantize weights → load into E8RHTLinear → verify forward."""

import pytest
import torch

from glq.codebook import E8ShellCodebook
from glq.quantize_model import quantize_layer_e8_shell_rht
from glq.quantized_linear import E8RHTLinear


@pytest.fixture(scope="module")
def codebook():
    return E8ShellCodebook.build(device="cpu", verbose=False)


def _random_psd(n):
    A = torch.randn(n, n)
    return A @ A.T + 0.1 * torch.eye(n)


def _load_artifacts_into_layer(layer, artifacts, codebook, codebook2=None):
    """Load quantization artifacts into an E8RHTLinear layer."""
    layer.Qidxs.copy_(artifacts['Qidxs'])
    layer.SU.copy_(artifacts['SU'])
    layer.SV.copy_(artifacts['SV'])
    layer.Wscale.copy_(artifacts['Wscale'])
    if 'Qidxs2' in artifacts:
        layer.Qidxs2.copy_(artifacts['Qidxs2'])
        layer.inv_resid_scale.copy_(artifacts['inv_resid_scale'])
    layer.set_codebook(codebook, codebook2=codebook2)


class TestQuantizeAndForward2bpw:
    def test_forward_matches_what(self, codebook):
        """Quantize → load → forward(x) should match x @ W_hat.T."""
        torch.manual_seed(42)
        m, n = 32, 64
        W = torch.randn(m, n) * 0.1
        H = _random_psd(n)

        W_hat, artifacts, metrics = quantize_layer_e8_shell_rht(W, H, codebook, bpw=2)

        layer = E8RHTLinear(n, m)
        _load_artifacts_into_layer(layer, artifacts, codebook)

        x = torch.randn(8, n)
        y = layer(x).float()
        y_ref = x.float() @ W_hat.float().T

        # Should match to reasonable precision
        torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)

    def test_dequantize_matches_what(self, codebook):
        """dequantize() should recover W_hat from quantize_layer."""
        torch.manual_seed(42)
        m, n = 32, 64
        W = torch.randn(m, n) * 0.1
        H = _random_psd(n)

        W_hat, artifacts, _ = quantize_layer_e8_shell_rht(W, H, codebook, bpw=2)

        layer = E8RHTLinear(n, m)
        _load_artifacts_into_layer(layer, artifacts, codebook)
        W_deq = layer.dequantize()

        torch.testing.assert_close(W_deq, W_hat, atol=1e-2, rtol=1e-2)

    def test_quantization_quality(self, codebook):
        """Quantized forward should produce output close to original W."""
        torch.manual_seed(42)
        m, n = 32, 64
        W = torch.randn(m, n) * 0.1
        H = _random_psd(n)

        W_hat, artifacts, metrics = quantize_layer_e8_shell_rht(W, H, codebook, bpw=2)

        layer = E8RHTLinear(n, m)
        _load_artifacts_into_layer(layer, artifacts, codebook)

        x = torch.randn(16, n)
        y_quant = layer(x).float()
        y_orig = x.float() @ W.float().T

        # SQNR should be positive (quantized output resembles original)
        signal = y_orig.pow(2).mean()
        noise = (y_quant - y_orig).pow(2).mean()
        sqnr = 10 * torch.log10(signal / noise.clamp(min=1e-20))
        assert sqnr.item() > 5, f"Output SQNR too low: {sqnr.item():.1f} dB"


class TestQuantizeAndForward3bpw:
    def test_forward_matches_what(self, codebook):
        torch.manual_seed(42)
        m, n = 32, 64
        W = torch.randn(m, n) * 0.1
        H = _random_psd(n)

        W_hat, artifacts, _ = quantize_layer_e8_shell_rht(W, H, codebook, bpw=3)
        codebook_small = codebook.make_small(256)

        layer = E8RHTLinear(n, m)
        _load_artifacts_into_layer(layer, artifacts, codebook, codebook2=codebook_small)

        x = torch.randn(8, n)
        y = layer(x).float()
        y_ref = x.float() @ W_hat.float().T

        torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)

    def test_better_quality_than_2bpw(self, codebook):
        """3bpw should have better output quality than 2bpw."""
        torch.manual_seed(42)
        m, n = 32, 64
        W = torch.randn(m, n) * 0.1
        H = _random_psd(n)
        x = torch.randn(16, n)
        y_orig = x.float() @ W.float().T

        # 2bpw
        W_hat2, art2, _ = quantize_layer_e8_shell_rht(W, H, codebook, bpw=2)
        layer2 = E8RHTLinear(n, m)
        _load_artifacts_into_layer(layer2, art2, codebook)
        y2 = layer2(x).float()
        mse2 = (y2 - y_orig).pow(2).mean().item()

        # 3bpw
        W_hat3, art3, _ = quantize_layer_e8_shell_rht(W, H, codebook, bpw=3)
        codebook_small = codebook.make_small(256)
        layer3 = E8RHTLinear(n, m)
        _load_artifacts_into_layer(layer3, art3, codebook, codebook2=codebook_small)
        y3 = layer3(x).float()
        mse3 = (y3 - y_orig).pow(2).mean().item()

        assert mse3 < mse2, f"3bpw MSE ({mse3:.6f}) should be lower than 2bpw MSE ({mse2:.6f})"


class TestQuantizeAndForward4bpw:
    def test_forward_matches_what(self, codebook):
        torch.manual_seed(42)
        m, n = 32, 64
        W = torch.randn(m, n) * 0.1
        H = _random_psd(n)

        W_hat, artifacts, _ = quantize_layer_e8_shell_rht(W, H, codebook, bpw=4)

        layer = E8RHTLinear(n, m)
        _load_artifacts_into_layer(layer, artifacts, codebook, codebook2=codebook)

        x = torch.randn(8, n)
        y = layer(x).float()
        y_ref = x.float() @ W_hat.float().T

        torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)

    def test_best_quality(self, codebook):
        """4bpw should have the best output quality."""
        torch.manual_seed(42)
        m, n = 32, 64
        W = torch.randn(m, n) * 0.1
        H = _random_psd(n)
        x = torch.randn(16, n)
        y_orig = x.float() @ W.float().T

        # 2bpw
        _, art2, _ = quantize_layer_e8_shell_rht(W, H, codebook, bpw=2)
        layer2 = E8RHTLinear(n, m)
        _load_artifacts_into_layer(layer2, art2, codebook)
        mse2 = (layer2(x).float() - y_orig).pow(2).mean().item()

        # 4bpw
        _, art4, _ = quantize_layer_e8_shell_rht(W, H, codebook, bpw=4)
        layer4 = E8RHTLinear(n, m)
        _load_artifacts_into_layer(layer4, art4, codebook, codebook2=codebook)
        mse4 = (layer4(x).float() - y_orig).pow(2).mean().item()

        assert mse4 < mse2, f"4bpw MSE ({mse4:.6f}) should be lower than 2bpw MSE ({mse2:.6f})"


class TestNonPowerOf2:
    def test_odd_dimensions(self, codebook):
        """Non-power-of-2 dimensions should still produce correct output."""
        torch.manual_seed(42)
        m, n = 10, 48
        W = torch.randn(m, n) * 0.1
        H = _random_psd(n)

        W_hat, artifacts, _ = quantize_layer_e8_shell_rht(W, H, codebook, bpw=2)

        layer = E8RHTLinear(n, m)
        _load_artifacts_into_layer(layer, artifacts, codebook)

        x = torch.randn(4, n)
        y = layer(x).float()
        y_ref = x.float() @ W_hat.float().T

        torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)
