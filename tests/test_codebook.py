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
    @classmethod
    def codebook(cls):
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

    def test_quantize_large_batch(self, codebook):
        """Large batch (> _batch_size) should use batched loop.

        `_batch_size` is the chunk width, not the batch: 20000 rows still go through the
        chunked branch, in 40 chunks rather than 2. Deliberately NOT the production default of
        16384, which on CPU makes each distance matrix [16384, 65536] fp32, 4 GiB. Before
        `_sq_dists` reused `dots`, three of them were live at once — measured to take the
        process from 6.14 to 12.27 GiB peak and run this suite out of memory on a 40 GiB
        machine.
        512 is 128 MiB per matrix.
        """
        x = torch.randn(20000, 8) * codebook.opt_scale
        decoded, indices = codebook._quantize_pytorch(x, _batch_size=512)
        assert decoded.shape == (20000, 8)
        assert indices.shape == (20000,)
        # All indices should be valid
        assert indices.min() >= 0
        assert indices.max() < codebook.CODEBOOK_SIZE

    @pytest.mark.parametrize("chunks", [1, 3])
    def test_nn_search_holds_one_distance_matrix_at_a_time(self, codebook, chunks):
        """The CPU search's peak is one chunk's [rows, 65536] distance matrix, not three.

        `-2.0 * dots + cb_norms` kept `dots`, `-2.0 * dots` and `dists` alive together, and
        the previous chunk's pair overlapped the next chunk's matmul. At the production chunk
        of 16384 rows that is 4 GiB per matrix: tests/test_kv_e8.py's real-shape test peaked
        at 16.5 GiB RSS, more than a 16 GB GitHub runner has, and the runner was shut down
        mid-test (main, run 40793950584). Measured with Linux's resettable peak-RSS mark, so
        the bound is on the search itself, not on whatever the process held before it.
        """
        clear_refs = "/proc/self/clear_refs"
        if not os.access(clear_refs, os.W_OK):
            pytest.skip("needs Linux /proc/self/clear_refs to reset the peak-RSS mark")

        def kib(field):
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith(field + ":"):
                        return int(line.split()[1])

        rows = 1024
        matrix = rows * codebook.CODEBOOK_SIZE * 4          # one fp32 chunk: 256 MiB
        x = torch.randn(chunks * rows, 8) * codebook.opt_scale
        with open(clear_refs, "w") as f:
            f.write("5")                                    # VmHWM := VmRSS
        before = kib("VmRSS")
        codebook._quantize_pytorch(x, _batch_size=rows)
        peak = (kib("VmHWM") - before) * 1024
        assert peak < 1.5 * matrix, (
            f"search peaked {peak / 2**20:.0f} MiB above its start, "
            f"{peak / matrix:.2f}x one {matrix / 2**20:.0f} MiB distance matrix")

    def test_nn_search_matches_the_out_of_place_reference(self, codebook):
        """Encode-path exactness: chunked or not, the indices and decoded vectors equal
        argmin(-2·x·Cᵀ + ‖C‖²) computed the straightforward way."""
        x = torch.randn(600, 8) * codebook.opt_scale
        ref = (-2.0 * (x @ codebook.codebook.T) + codebook.codebook_norms).argmin(dim=-1)
        for rows in (128, 1024):                            # chunked, then one shot
            decoded, idx = codebook._quantize_pytorch(x, _batch_size=rows)
            assert torch.equal(idx, ref)
            assert torch.equal(decoded, codebook.codebook[ref])

    def test_nn_search_keeps_the_promoted_dtype_for_low_precision_input(self, codebook):
        """A bf16 input gets bf16 dot products but fp32 distances, because the fp32 norms
        promote the sum. Reusing the bf16 `dots` buffer for the sum would round every
        distance to bf16 and pick different neighbours."""
        x = (torch.randn(600, 8) * codebook.opt_scale).bfloat16()
        dots = x @ codebook.codebook.T.to(torch.bfloat16)
        ref = (-2.0 * dots + codebook.codebook_norms).argmin(dim=-1)
        _, idx = codebook._quantize_pytorch(x, _batch_size=1024)
        assert torch.equal(idx, ref)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_nn_search_on_cuda_without_triton_matches_the_reference(self, codebook):
        """The fp16 path a CUDA box takes when Triton is missing. The reference is the
        out-of-place expression over the same chunks, so the comparison isolates the
        in-place rewrite from cuBLAS picking a different GEMM for a different row count."""
        cb = codebook.to("cuda")
        x = (torch.randn(3000, 8) * codebook.opt_scale).cuda()
        for rows in (512, 4096):                            # chunked, then one shot
            ref = torch.cat([(-2.0 * (c.half() @ cb.codebook_half_t)
                              + cb.codebook_norms_half).argmin(dim=-1)
                             for c in x.split(rows)])
            decoded, idx = cb._quantize_pytorch(x, _batch_size=rows)
            assert torch.equal(idx, ref)
            assert torch.equal(decoded, cb.codebook[ref])

    def test_decode_rvq(self, codebook):
        """decode_rvq should match manual two-stage decode."""
        idx1 = torch.tensor([0, 100, 65535])
        idx2 = torch.tensor([50, 200, 1000])
        result = codebook.decode_rvq(idx1, idx2)
        expected = codebook.codebook[idx1] + codebook.codebook[idx2] / codebook.resid_scale
        torch.testing.assert_close(result, expected)


class TestEnumerateEmpty:
    def test_negative_norm_returns_empty(self):
        """max_norm_sq < 0 should return empty tensors."""
        G, _ = e8_basis()
        coords, norms = enumerate_short_vectors(G, max_norm_sq=-0.01)
        assert coords.shape[0] == 0
        assert norms.shape[0] == 0


class TestE8RelaxedCodebook:
    """Sanity tests for the relaxed (D̃8) codebook + parametric target_size.

    Phase 5.0a: target_size lets us probe whether the denser inner shells
    of D̃8 (256 shell-1 points vs strict's 240, ~4320 shell-2 vs 2160)
    transfer to weight quantization at small codebook sizes.
    """

    def test_default_size(self):
        from glq.codebook_relaxed import E8RelaxedCodebook
        cb = E8RelaxedCodebook(device="cpu", verbose=False)
        assert cb.codebook.shape == (65536, 8)
        assert cb.codebook_size == 65536

    def test_target_size_4k(self):
        from glq.codebook_relaxed import E8RelaxedCodebook
        cb = E8RelaxedCodebook(device="cpu", verbose=False, target_size=4096)
        assert cb.codebook.shape == (4096, 8)
        assert cb.codebook_size == 4096
        # opt_scale + resid_scale must be recomputed for the truncated codebook
        assert cb.opt_scale > 0
        assert cb.resid_scale > 0

    def test_4k_quantize_reduces_error(self):
        from glq.codebook_relaxed import E8RelaxedCodebook
        cb = E8RelaxedCodebook(device="cpu", verbose=False, target_size=4096)
        x = torch.randn(500, 8) * cb.opt_scale
        decoded, _ = cb.quantize(x)
        quant_mse = ((x - decoded) ** 2).mean().item()
        random_mse = ((x - torch.randn_like(x)) ** 2).mean().item()
        assert quant_mse < random_mse


class TestVerboseOutput:
    def test_build_verbose(self, capsys):
        """build(verbose=True) prints loading message."""
        E8ShellCodebook.build(device="cpu", verbose=True)
        captured = capsys.readouterr()
        assert "E8ShellCodebook" in captured.out

    def test_init_verbose(self, capsys):
        """__init__(verbose=True) prints enumeration progress."""
        E8ShellCodebook(device="cpu", verbose=True)
        captured = capsys.readouterr()
        assert "enumerating" in captured.out
        assert "entries" in captured.out
