"""Relaxed E8 (D̃8) codebook for KV-cache quantization.

The strict E8 lattice imposes an even-sum parity constraint on the
half-integer coset, halving each "half-shell" to 128 points. The relaxed
lattice D̃8 = Z^8 ∪ (Z^8 + (½)^8) drops that constraint, giving 256
points per half-shell. For sub-Gaussian Hadamard-rotated KV vectors,
which concentrate mass near the origin, the doubled density near zero
gives a measurable MSE improvement over strict E8 — see
``infra/kv-cache.md`` for the mathematical argument.

Construction mirrors ``E8ShellCodebook``: enumerate up to 65,536
lattice points sorted by norm² (= 16-bit indexing = 2 bpw per coord).
"""

import math
import time
import torch

from .codebook import E8ShellCodebook, enumerate_short_vectors, _pack_codebook


def enumerate_dtilde8(max_norm_sq: float):
    """Enumerate D̃8 = Z^8 ∪ (Z^8 + (½)^8) lattice points with ||v||² ≤ max_norm_sq.

    Returns (coords, norms_sq) on the double-precision CPU side, same shape
    convention as ``enumerate_short_vectors``: ``coords`` are the lattice
    points themselves (not basis coefficients), and ``norms_sq`` is their
    squared Euclidean norms.

    Half-integer points have ||z + s||² minimum at z=0 (= ||s||²=2), so the
    integer search for the half-integer branch needs an enlarged radius
    R' = (sqrt(max_norm_sq) + sqrt(2))² to be sound.
    """
    I = torch.eye(8, dtype=torch.float64)

    # Integer branch: enumerate Z^8 with ||v||² ≤ max_norm_sq.
    int_coords, int_norms = enumerate_short_vectors(I, max_norm_sq)

    # Half-integer branch: enumerate integer z, shift by s=(½)^8, filter.
    s = torch.full((8,), 0.5, dtype=torch.float64)
    s_norm_sq = 2.0
    # ||z + s||² ≤ R² ⇒ ||z|| ≤ sqrt(R²) + sqrt(2)
    bounding_R2 = (math.sqrt(max_norm_sq) + math.sqrt(s_norm_sq)) ** 2
    z_candidates, _ = enumerate_short_vectors(I, bounding_R2)
    half_vecs = z_candidates + s
    half_norms = (half_vecs * half_vecs).sum(-1)
    keep = half_norms <= max_norm_sq + 1e-9
    half_vecs = half_vecs[keep]
    half_norms = half_norms[keep]

    coords = torch.cat([int_coords, half_vecs])
    norms_sq = torch.cat([int_norms, half_norms])
    return coords, norms_sq


class E8RelaxedCodebook(E8ShellCodebook):
    """Relaxed E8 (D̃8) lattice codebook with 65,536 entries.

    Same 2 bpw per coordinate as ``E8ShellCodebook`` (16 bits / 8 dims),
    same tensor layout (so the Triton NN kernel and the LDLQ / RVQ paths
    can consume it interchangeably), but enumerated from D̃8 instead of
    strict E8. The first shell (norm²=2) has 256 half-integer points
    instead of 128.

    For weight quantization the strict codebook is preferred; for KV
    cache (sub-Gaussian Hadamard-rotated vectors with mass near origin)
    the relaxed codebook is expected to yield ~22% lower MSE — see the
    microbench in ``tests/test_kv_e8.py``.
    """

    CODEBOOK_SIZE = 65536
    DIM = 8

    def __init__(self, device='cpu', verbose=True, max_norm_sq=12.5,
                 target_size: int = None):
        """Enumerate D̃8 lattice points and build the codebook.

        Args:
            device: torch device for the codebook tensors.
            verbose: print enumeration progress.
            max_norm_sq: ceiling for the D̃8 enumeration search.
                Defaults to 12.5 to match the strict E8 ceiling; D̃8 has
                roughly double the points per shell so the search yields
                more candidates than strict.
            target_size: codebook size to truncate to. Defaults to
                ``CODEBOOK_SIZE`` (= 65 536). For Phase 5.0a relaxed-
                codebook smem probe use ``target_size=4096`` (= 64 KB
                at fp16). Truncation is shell-sorted: smaller values
                drop high-norm vectors first. ``opt_scale`` and
                ``resid_scale`` are recomputed for the truncated
                codebook.
        """
        if target_size is None:
            target_size = self.CODEBOOK_SIZE
        if verbose:
            print("E8RelaxedCodebook: enumerating D̃8 lattice ...")
        t0 = time.perf_counter()
        coords, norms_sq = enumerate_dtilde8(max_norm_sq=max_norm_sq)
        t_enum = time.perf_counter() - t0

        assert coords.shape[0] >= target_size, (
            f"D̃8 enumeration yielded only {coords.shape[0]} points within "
            f"||v||²≤{max_norm_sq}; need at least {target_size}. "
            f"Increase max_norm_sq.")

        order = norms_sq.argsort()
        coords = coords[order][:target_size]

        self.codebook = coords.float().to(device)
        self.codebook_norms = (self.codebook ** 2).sum(-1)
        self.codebook_half = self.codebook.half()
        self.codebook_half_t = self.codebook_half.T.contiguous()
        self.codebook_norms_half = self.codebook_norms.half()
        self.codebook_packed = _pack_codebook(self.codebook).to(device)
        self.codesz = self.DIM
        self.device = device
        self.codebook_size = target_size

        self.opt_scale = self._compute_opt_scale()
        self.resid_scale = self._compute_resid_scale()
        if verbose:
            print(f"  {target_size} entries in {t_enum:.2f}s, "
                  f"opt_scale={self.opt_scale:.4f}, "
                  f"resid_scale={self.resid_scale:.2f}")
