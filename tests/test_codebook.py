"""Tests for the E8 lattice shell codebook."""

import os
import tempfile

import pytest
import torch

from glq.codebook import E8ShellCodebook, e8_basis, enumerate_short_vectors


class TestE8Basis:
    def test_basis_shapes(self):
        G, G_inv = e8_basis()
        assert G.shape == (8, 8)
        assert G_inv.shape == (8, 8)

    def test_basis_inverse(self):
        G, G_inv = e8_basis()
        product = G @ G_inv
        torch.testing.assert_close(
            product, torch.eye(8), atol=1e-5, rtol=1e-5
        )

    def test_basis_determinant(self):
        G, _ = e8_basis()
        det = torch.linalg.det(G.double())
        assert abs(abs(det.item()) - 1.0) < 1e-6, f"det(G) = {det.item()}, expected ±1"


class TestEnumerateShortVectors:
    def test_shell_0(self):
        """Shell 0 is just the zero vector."""
        G, _ = e8_basis()
        coords, norms = enumerate_short_vectors(G, max_norm_sq=0.01)
        assert coords.shape[0] == 1
        assert norms[0].item() < 0.01

    def test_shell_1_kissing_number(self):
        """E8 has kissing number 240 (shell 1, norm^2 = 2)."""
        G, _ = e8_basis()
        coords, norms = enumerate_short_vectors(G, max_norm_sq=2.01)
        # Shell 0 (1) + shell 1 (240)
        assert coords.shape[0] == 241

    def test_through_shell_5(self):
        """Shells 0-5 should contain exactly 56881 vectors."""
        G, _ = e8_basis()
        # Shell 5 has norm^2 = 10
        coords, norms = enumerate_short_vectors(G, max_norm_sq=10.01)
        assert coords.shape[0] == 56881


class TestE8ShellCodebook:
    @pytest.fixture(scope="class")
    def codebook(self):
        return E8ShellCodebook(device="cpu", verbose=False)

    def test_codebook_size(self, codebook):
        assert codebook.codebook.shape == (65536, 8)

    def test_codebook_contains_zero(self, codebook):
        """The zero vector should be in the codebook."""
        norms = codebook.codebook.norm(dim=1)
        assert (norms < 1e-6).any()

    def test_quantize_zero(self, codebook):
        """Quantizing zeros should return the zero vector."""
        x = torch.zeros(1, 8)
        decoded, indices = codebook.quantize(x)
        assert decoded.norm().item() < 1e-6

    def test_quantize_roundtrip(self, codebook):
        """Quantizing a codebook vector should return itself."""
        # Pick some arbitrary codebook entry
        v = codebook.codebook[100].unsqueeze(0)
        decoded, idx = codebook.quantize(v)
        torch.testing.assert_close(decoded, v, atol=1e-5, rtol=1e-5)

    def test_quantize_batched(self, codebook):
        """Quantize works on batches."""
        x = torch.randn(1000, 8)
        decoded, indices = codebook.quantize(x)
        assert decoded.shape == (1000, 8)
        assert indices.shape == (1000,)

    def test_decode(self, codebook):
        """Decode returns the correct codebook vectors."""
        indices = torch.tensor([0, 1, 100, 65535])
        decoded = codebook.decode(indices)
        expected = codebook.codebook[indices]
        torch.testing.assert_close(decoded, expected)

    def test_quantize_reduces_error(self, codebook):
        """Quantized vectors should be closer than random guesses."""
        x = torch.randn(500, 8) * codebook.opt_scale
        decoded, _ = codebook.quantize(x)
        quant_mse = ((x - decoded) ** 2).mean().item()
        random_mse = ((x - torch.randn_like(x)) ** 2).mean().item()
        assert quant_mse < random_mse

    def test_opt_scale_positive(self, codebook):
        assert codebook.opt_scale > 0

    def test_resid_scale_positive(self, codebook):
        assert codebook.resid_scale > 0

    def test_save_load_roundtrip(self, codebook):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            codebook.save(path)
            loaded = E8ShellCodebook.load(path, device="cpu")
            torch.testing.assert_close(loaded.codebook, codebook.codebook)
            assert abs(loaded.opt_scale - codebook.opt_scale) < 1e-6
            assert abs(loaded.resid_scale - codebook.resid_scale) < 1e-6
        finally:
            os.unlink(path)

    def test_make_small(self, codebook):
        small = codebook.make_small(256)
        assert small.codebook.shape == (256, 8)
        assert small.CODEBOOK_SIZE == 256
        # Small codebook vectors should be a subset of full codebook
        torch.testing.assert_close(small.codebook, codebook.codebook[:256])

    def test_to_device(self, codebook):
        cb2 = codebook.to("cpu")
        assert cb2.device == "cpu"
        torch.testing.assert_close(cb2.codebook, codebook.codebook)

    def test_build_from_bundled(self):
        """build() should load from bundled file if it exists."""
        cb = E8ShellCodebook.build(device="cpu", verbose=False)
        assert cb.codebook.shape == (65536, 8)

    def test_rvq_reduces_error(self, codebook):
        """Two-stage RVQ should have lower error than single-stage."""
        x = torch.randn(200, 8) * codebook.opt_scale
        dec1, _ = codebook.quantize(x)
        dec_rvq, _ = codebook.quantize_rvq(x)
        mse_1stage = ((x - dec1) ** 2).mean().item()
        mse_rvq = ((x - dec_rvq) ** 2).mean().item()
        assert mse_rvq < mse_1stage
