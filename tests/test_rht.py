"""Tests for the Randomized Hadamard Transform."""

import pytest
import torch

from glq.rht import RHT


class TestRHT:
    def test_weight_roundtrip(self):
        """transform_weights -> inverse_transform_weights recovers original."""
        m, n = 32, 64
        rht = RHT(m, n, device="cpu")
        W = torch.randn(m, n)
        W_t = rht.transform_weights(W)
        W_back = rht.inverse_transform_weights(W_t)
        torch.testing.assert_close(W_back, W, atol=1e-4, rtol=1e-4)

    def test_transform_preserves_frobenius_norm(self):
        """RHT (orthogonal transform) should preserve Frobenius norm."""
        m, n = 16, 32
        rht = RHT(m, n, device="cpu")
        W = torch.randn(m, n)
        W_t = rht.transform_weights(W)
        # Frobenius norm of padded W should be preserved
        W_padded = torch.zeros(rht.m_pad, rht.n_pad)
        W_padded[:m, :n] = W
        torch.testing.assert_close(
            W_padded.norm(), W_t.norm(), atol=1e-4, rtol=1e-4
        )

    def test_hessian_transform_shape(self):
        """Hessian transform should produce (n_pad, n_pad)."""
        m, n = 16, 48
        rht = RHT(m, n, device="cpu")
        H = torch.eye(n) + 0.1 * torch.randn(n, n)
        H = (H + H.T) / 2  # symmetrize
        H_t = rht.transform_hessian(H)
        assert H_t.shape == (rht.n_pad, rht.n_pad)

    def test_hessian_transform_preserves_symmetry(self):
        """Transformed Hessian should remain symmetric."""
        m, n = 16, 32
        rht = RHT(m, n, device="cpu")
        H = torch.randn(n, n)
        H = H @ H.T  # make PSD
        H_t = rht.transform_hessian(H)
        torch.testing.assert_close(H_t, H_t.T, atol=1e-4, rtol=1e-4)

    def test_padding(self):
        """Non-power-of-2 dimensions should be padded correctly."""
        rht = RHT(10, 48, device="cpu")
        assert rht.m_pad == 16
        assert rht.n_pad == 64

    def test_power_of_2_no_padding(self):
        """Power-of-2 dimensions should not change."""
        rht = RHT(32, 128, device="cpu")
        assert rht.m_pad == 32
        assert rht.n_pad == 128

    def test_deterministic_signs(self):
        """Same seed should produce same random signs."""
        rht1 = RHT(16, 32, device="cpu", seed=42)
        rht2 = RHT(16, 32, device="cpu", seed=42)
        torch.testing.assert_close(rht1.su, rht2.su)
        torch.testing.assert_close(rht1.sv, rht2.sv)

    def test_different_seeds(self):
        """Different seeds should produce different signs."""
        rht1 = RHT(16, 32, device="cpu", seed=42)
        rht2 = RHT(16, 32, device="cpu", seed=99)
        assert not torch.equal(rht1.su, rht2.su) or not torch.equal(rht1.sv, rht2.sv)

    def test_input_transform_shape(self):
        """transform_input should produce (batch, n_pad)."""
        rht = RHT(16, 48, device="cpu")
        x = torch.randn(5, 48)
        x_t = rht.transform_input(x)
        assert x_t.shape == (5, 64)
