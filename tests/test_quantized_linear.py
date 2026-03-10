"""Tests for E8RHTLinear — quantized linear layer with E8 codebook + RHT."""

import pytest
import torch

from glq.codebook import E8ShellCodebook
from glq.quantized_linear import E8RHTLinear


@pytest.fixture(scope="module")
def codebook():
    return E8ShellCodebook.build(device="cpu", verbose=False)


@pytest.fixture(scope="module")
def codebook_small(codebook):
    return codebook.make_small(256)


class TestConstruction:
    def test_buffer_shapes(self):
        layer = E8RHTLinear(48, 32)
        assert layer.Qidxs.shape == (32, 64 // 8)  # m_pad=32, n_pad=64
        assert layer.SU.shape == (32,)
        assert layer.SV.shape == (64,)
        assert layer.Wscale.shape == ()

    def test_padding(self):
        layer = E8RHTLinear(50, 33)
        assert layer.m_pad == 64  # next power of 2 above 33
        assert layer.n_pad == 64  # next power of 2 above 50

    def test_power_of_2_no_padding(self):
        layer = E8RHTLinear(128, 64)
        assert layer.m_pad == 64
        assert layer.n_pad == 128

    def test_bias(self):
        layer = E8RHTLinear(32, 16, bias=True)
        assert layer.bias is not None
        assert layer.bias.shape == (16,)

    def test_no_bias(self):
        layer = E8RHTLinear(32, 16, bias=False)
        assert layer.bias is None

    def test_stage2_buffers_zero_by_default(self):
        layer = E8RHTLinear(32, 16)
        assert layer.Qidxs2.shape == layer.Qidxs.shape
        assert layer.Qidxs2.abs().max() == 0
        assert layer.inv_resid_scale.item() == 0.0
        assert layer._has_stage2 is False

    def test_extra_repr(self):
        layer = E8RHTLinear(48, 32)
        s = layer.extra_repr()
        assert "in_features=48" in s
        assert "out_features=32" in s


class TestSetCodebook:
    def test_single_codebook(self, codebook):
        layer = E8RHTLinear(32, 16)
        layer.set_codebook(codebook)
        assert layer.codebook is codebook
        assert layer.codebook2 is None

    def test_dual_codebook(self, codebook, codebook_small):
        layer = E8RHTLinear(32, 16)
        layer.set_codebook(codebook, codebook2=codebook_small)
        assert layer.codebook is codebook
        assert layer.codebook2 is codebook_small


class TestDequantize:
    def test_output_shape(self, codebook):
        layer = E8RHTLinear(48, 32)
        layer.set_codebook(codebook)
        W = layer.dequantize()
        assert W.shape == (32, 48)

    def test_output_shape_power_of_2(self, codebook):
        layer = E8RHTLinear(64, 32)
        layer.set_codebook(codebook)
        W = layer.dequantize()
        assert W.shape == (32, 64)

    def test_deterministic(self, codebook):
        layer = E8RHTLinear(64, 32)
        layer.set_codebook(codebook)
        W1 = layer.dequantize()
        W2 = layer.dequantize()
        torch.testing.assert_close(W1, W2)


class TestForward:
    def test_output_shape(self, codebook):
        layer = E8RHTLinear(48, 32)
        layer.set_codebook(codebook)
        x = torch.randn(4, 48)
        y = layer(x)
        assert y.shape == (4, 32)

    def test_3d_input(self, codebook):
        """Forward should handle (batch, seq, features) input."""
        layer = E8RHTLinear(64, 32)
        layer.set_codebook(codebook)
        x = torch.randn(2, 8, 64)
        y = layer(x)
        assert y.shape == (2, 8, 32)

    def test_single_token(self, codebook):
        """B=1, seq=1 input."""
        layer = E8RHTLinear(64, 32)
        layer.set_codebook(codebook)
        x = torch.randn(1, 64)
        y = layer(x)
        assert y.shape == (1, 32)

    def test_forward_matches_dequantize(self, codebook):
        """forward(x) should equal x @ dequantize().T (approximately)."""
        layer = E8RHTLinear(64, 32)
        layer.set_codebook(codebook)

        x = torch.randn(8, 64)
        y_forward = layer(x).float()
        W_deq = layer.dequantize().float()
        y_ref = x.float() @ W_deq.T

        # Should be exact (both use same decode path on CPU)
        torch.testing.assert_close(y_forward, y_ref, atol=1e-3, rtol=1e-3)

    def test_with_bias(self, codebook):
        layer = E8RHTLinear(64, 32, bias=True)
        layer.set_codebook(codebook)
        layer.bias.fill_(1.0)

        x = torch.randn(4, 64)
        y = layer(x)
        assert y.shape == (4, 32)

        # Verify bias is actually applied
        layer_no_bias = E8RHTLinear(64, 32, bias=False)
        # Copy all buffers
        layer_no_bias.Qidxs.copy_(layer.Qidxs)
        layer_no_bias.SU.copy_(layer.SU)
        layer_no_bias.SV.copy_(layer.SV)
        layer_no_bias.Wscale.copy_(layer.Wscale)
        layer_no_bias.set_codebook(codebook)
        y_no_bias = layer_no_bias(x)

        diff = (y - y_no_bias).float()
        # Bias of 1.0 should shift output by 1.0
        torch.testing.assert_close(diff, torch.ones_like(diff), atol=1e-3, rtol=1e-3)

    def test_zero_input(self, codebook):
        layer = E8RHTLinear(64, 32)
        layer.set_codebook(codebook)
        x = torch.zeros(2, 64)
        y = layer(x)
        # Zero input through linear should give zero output
        assert y.abs().max().item() < 1e-3

    def test_deterministic(self, codebook):
        layer = E8RHTLinear(64, 32)
        layer.set_codebook(codebook)
        x = torch.randn(4, 64)
        y1 = layer(x)
        y2 = layer(x)
        torch.testing.assert_close(y1, y2)


def _setup_two_stage(layer, codebook2, K2=256):
    """Helper: fill Qidxs2/inv_resid_scale then call set_codebook to detect stage2."""
    m_pad, n_blocks = layer.Qidxs.shape
    layer.Qidxs2.copy_(torch.randint(0, K2, (m_pad, n_blocks), dtype=torch.int32).to(torch.int16))
    layer.inv_resid_scale.fill_(1.0 / layer.codebook.resid_scale if layer.codebook else 1.0)


class TestTwoStage:
    def test_3bpw_forward(self, codebook, codebook_small):
        """3bpw forward with secondary codebook."""
        layer = E8RHTLinear(64, 32)
        layer.set_codebook(codebook)
        layer.inv_resid_scale.fill_(1.0 / codebook.resid_scale)
        _setup_two_stage(layer, codebook_small, K2=256)
        layer.set_codebook(codebook, codebook2=codebook_small)

        x = torch.randn(4, 64)
        y = layer(x)
        assert y.shape == (4, 32)

    def test_3bpw_dequantize(self, codebook, codebook_small):
        """3bpw dequantize returns correct shape."""
        layer = E8RHTLinear(64, 32)
        layer.set_codebook(codebook)
        _setup_two_stage(layer, codebook_small, K2=256)
        layer.set_codebook(codebook, codebook2=codebook_small)

        W = layer.dequantize()
        assert W.shape == (32, 64)

    def test_4bpw_forward(self, codebook):
        """4bpw uses full codebook for both stages."""
        layer = E8RHTLinear(64, 32)
        layer.set_codebook(codebook)
        _setup_two_stage(layer, codebook, K2=65536)
        layer.set_codebook(codebook, codebook2=codebook)

        x = torch.randn(4, 64)
        y = layer(x)
        assert y.shape == (4, 32)

    def test_two_stage_forward_matches_dequantize(self, codebook, codebook_small):
        """Two-stage forward(x) should match x @ dequantize().T."""
        layer = E8RHTLinear(64, 32)
        layer.set_codebook(codebook)
        _setup_two_stage(layer, codebook_small, K2=256)
        layer.set_codebook(codebook, codebook2=codebook_small)

        x = torch.randn(8, 64)
        y_forward = layer(x).float()
        W_deq = layer.dequantize().float()
        y_ref = x.float() @ W_deq.T

        torch.testing.assert_close(y_forward, y_ref, atol=1e-3, rtol=1e-3)
