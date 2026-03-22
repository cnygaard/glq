"""E8 lattice shell codebook for 2bpw quantization."""

import math
import os
import time
import torch
from typing import Tuple

_BUNDLED_CODEBOOK = os.path.join(os.path.dirname(__file__), 'e8_codebook.pt')

try:
    import triton  # noqa: F401
    _triton_available = True
except ImportError:
    _triton_available = False


def e8_basis() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generator matrix G for the E8 lattice (8 dimensions).
    Uses the simple-roots basis from Conway & Sloane.
    Properties: det(G) = 1, min norm^2 = 2, kissing number = 240.

    Returns:
        (G, G_inv): generator and inverse, float32, shape (8, 8)
    """
    M = torch.tensor([
        [ 2,  0,  0,  0,  0,  0,  0,  0],
        [-1,  1,  0,  0,  0,  0,  0,  0],
        [ 0, -1,  1,  0,  0,  0,  0,  0],
        [ 0,  0, -1,  1,  0,  0,  0,  0],
        [ 0,  0,  0, -1,  1,  0,  0,  0],
        [ 0,  0,  0,  0, -1,  1,  0,  0],
        [ 0,  0,  0,  0,  0, -1,  1,  0],
        [.5, .5, .5, .5, .5, .5, .5, .5],
    ], dtype=torch.float64)

    G = M.clone()
    G_inv = torch.linalg.inv(G)
    return G.T.float(), G_inv.T.float()


def enumerate_short_vectors(G, max_norm_sq):
    """
    Enumerate all lattice vectors v = z @ G^T with ||v||^2 <= max_norm_sq.
    Uses Fincke-Pohst / Schnorr-Euchner tree search.
    """
    GT = G.double().T
    n = GT.shape[0]

    gso = GT.clone()
    mu = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(i):
            mu[i, j] = (GT[i] @ gso[j]) / (gso[j] @ gso[j])
            gso[i] = gso[i] - mu[i, j] * gso[j]
    gso_sq = (gso * gso).sum(dim=1)

    mu_np = mu.numpy()
    gso_sq_np = gso_sq.numpy()

    results = []

    def _search(level, z, budget):
        center = 0.0
        for i in range(level + 1, n):
            center -= z[i] * mu_np[i, level]

        if budget < -1e-12:
            return
        r = math.sqrt(max(budget / gso_sq_np[level], 0.0))
        z_lo = math.ceil(center - r)
        z_hi = math.floor(center + r)

        if level == 0:
            for zi in range(z_lo, z_hi + 1):
                sigma = zi - center
                if budget - sigma * sigma * gso_sq_np[0] >= -1e-10:
                    z[0] = zi
                    results.append(tuple(z))
        else:
            for zi in range(z_lo, z_hi + 1):
                sigma = zi - center
                nb = budget - sigma * sigma * gso_sq_np[level]
                if nb >= -1e-10:
                    z[level] = zi
                    _search(level - 1, z, nb)

    z = [0] * n
    _search(n - 1, z, float(max_norm_sq))

    if not results:
        return (torch.zeros(0, n, dtype=torch.float64),
                torch.zeros(0, dtype=torch.float64))

    coords = torch.tensor(results, dtype=torch.float64)
    vecs = coords @ GT
    norms_sq = (vecs * vecs).sum(dim=1)
    return coords, norms_sq


def _pack_codebook(codebook: torch.Tensor) -> torch.Tensor:
    """Pack E8 codebook (K, 8) float into (K,) uint32.

    Each coordinate is one of 13 values: {-3, -2.5, ..., 2.5, 3} = (code-6)*0.5
    where code ∈ [0, 12] fits in 4 bits. Pack 8 nibbles into one uint32.
    Decode: val_i = ((packed >> (4*i)) & 0xF) * 0.5 - 3.0
    """
    codes = (codebook * 2.0).round().long() + 6  # map [-3,3] → [0,12]
    codes = codes.clamp(0, 15)  # safety clamp to 4-bit range
    packed = torch.zeros(codebook.shape[0], dtype=torch.int32,
                         device=codebook.device)
    for i in range(8):
        packed = packed | (codes[:, i].int() << (4 * i))
    return packed


class E8ShellCodebook:
    """
    Finite codebook from E8 lattice shells 0-6.

    65536 = 2^16 entries -> 16 bits / 8 dims = 2 bpw.
    With RVQ (two-stage): 32 bits / 8 dims = 4 bpw.
    """

    CODEBOOK_SIZE = 65536
    DIM = 8

    def __init__(self, device='cpu', verbose=True):
        """Enumerate E8 shells and build codebook (~1.3s CPU)."""
        G, _ = e8_basis()

        if verbose:
            print("E8ShellCodebook: enumerating lattice shells ...")
        t0 = time.perf_counter()
        coords, norms_sq = enumerate_short_vectors(G, max_norm_sq=12.5)
        t_enum = time.perf_counter() - t0

        order = norms_sq.argsort()
        coords = coords[order]
        norms_sq = norms_sq[order]

        shell_boundaries = [0]
        for i in range(1, len(norms_sq)):
            if norms_sq[i] - norms_sq[i - 1] > 0.01:
                shell_boundaries.append(i)
        shell_boundaries.append(len(norms_sq))

        shell_counts = [shell_boundaries[i + 1] - shell_boundaries[i]
                        for i in range(len(shell_boundaries) - 1)]

        expected_counts = [1, 240, 2160, 6720, 17520, 30240, 60480]
        for i, (exp, got) in enumerate(zip(expected_counts, shell_counts)):
            assert got == exp, f"Shell {i} count mismatch: expected {exp}, got {got}"

        through_shell5 = sum(shell_counts[:6])  # 56881
        n_from_shell6 = self.CODEBOOK_SIZE - through_shell5  # 8655
        keep = shell_boundaries[6] + n_from_shell6
        coords_sel = coords[:keep]
        assert coords_sel.shape[0] == self.CODEBOOK_SIZE

        GT = G.T.double()
        vecs = coords_sel @ GT

        self.codebook = vecs.float().to(device)
        self.codebook_norms = (self.codebook ** 2).sum(-1)
        self.codebook_half = self.codebook.half()
        self.codebook_half_t = self.codebook_half.T.contiguous()
        self.codebook_norms_half = self.codebook_norms.half()
        self.codebook_packed = _pack_codebook(self.codebook).to(device)
        self.codesz = self.DIM
        self.device = device

        self.opt_scale = self._compute_opt_scale()
        self.resid_scale = self._compute_resid_scale()
        if verbose:
            print(f"  {self.CODEBOOK_SIZE} entries in {t_enum:.2f}s, "
                  f"opt_scale={self.opt_scale:.4f}, resid_scale={self.resid_scale:.2f}")

    @classmethod
    def from_precomputed(cls, codebook_tensor: torch.Tensor,
                         opt_scale: float = None, resid_scale: float = None,
                         device='cpu'):
        """Load from a precomputed codebook tensor (skips shell enumeration)."""
        obj = object.__new__(cls)
        obj.codebook = codebook_tensor.float().to(device)
        obj.codebook_norms = (obj.codebook ** 2).sum(-1)
        obj.codebook_half = obj.codebook.half()
        obj.codebook_half_t = obj.codebook_half.T.contiguous()
        obj.codebook_norms_half = obj.codebook_norms.half()
        obj.codebook_packed = _pack_codebook(obj.codebook).to(device)
        obj.codesz = cls.DIM
        obj.device = device
        obj.opt_scale = opt_scale if opt_scale is not None else obj._compute_opt_scale()
        obj.resid_scale = resid_scale if resid_scale is not None else obj._compute_resid_scale()
        return obj

    def quantize(self, x, _batch_size=16384):
        """Nearest-neighbour lookup. Uses Triton kernel on CUDA, fp16 matmul fallback."""
        if x.is_cuda and _triton_available:
            return self._quantize_triton(x)
        return self._quantize_pytorch(x, _batch_size)

    def _quantize_triton(self, x):
        """Fused codebook NN via Triton kernel (no intermediate distance matrix)."""
        from .codebook_kernel import triton_codebook_nn
        # Pass pre-converted fp16 codebook and fp32 norms to avoid per-call copies
        indices = triton_codebook_nn(
            x, self.codebook_half, self.codebook_norms)
        return self.codebook[indices], indices

    def quantize_fast(self, x_half, decoded_out=None, idx_out=None):
        """Fast quantize with fused NN+decode: x must be fp16 CUDA.

        Returns (decoded_fp16, indices_i64).
        Used by the LDLQ inner loop where we control dtypes.
        Pre-allocated buffers can be passed to avoid per-call allocation.
        """
        from .codebook_kernel import triton_codebook_nn_decode
        return triton_codebook_nn_decode(
            x_half, self.codebook_half, self.codebook_norms,
            decoded_out=decoded_out, idx_out=idx_out)

    def _quantize_pytorch(self, x, _batch_size=16384):
        """Batched fp16 matmul + argmin fallback."""
        use_half = x.is_cuda
        if use_half:
            cb_t = self.codebook_half_t
            cb_norms = self.codebook_norms_half
        else:
            cb_t = self.codebook.T
            cb_norms = self.codebook_norms

        if x.shape[0] <= _batch_size:
            xq = x.half() if use_half else x
            dots = xq @ cb_t
            dists = -2.0 * dots + cb_norms
            indices = dists.argmin(dim=-1)
            return self.codebook[indices], indices

        all_indices = []
        for start in range(0, x.shape[0], _batch_size):
            chunk = x[start:start + _batch_size]
            xq = chunk.half() if use_half else chunk
            dots = xq @ cb_t
            dists = -2.0 * dots + cb_norms
            all_indices.append(dists.argmin(dim=-1))
        indices = torch.cat(all_indices)
        return self.codebook[indices], indices

    def decode(self, indices):
        """Table lookup: index -> 8-d vector."""
        return self.codebook[indices]

    def quantize_rvq(self, x):
        """Two-stage RVQ: 4 bpw (32 bits / 8 dims)."""
        dec1, idx1 = self.quantize(x)
        residual = x - dec1
        dec2, idx2 = self.quantize(residual * self.resid_scale)
        decoded = dec1 + dec2 / self.resid_scale
        return decoded, (idx1, idx2)

    def decode_rvq(self, idx1, idx2):
        """Decode two-stage RVQ indices."""
        return self.codebook[idx1] + self.codebook[idx2] / self.resid_scale

    def _compute_opt_scale(self, n_samples=8_000):
        x = torch.randn(n_samples, self.DIM, device=self.device)
        best_nmse, best_s = float('inf'), 1.0
        for s in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
            _, idx = self.quantize(x / s)
            dec = self.codebook[idx]
            mse = ((x / s - dec) ** 2).sum(-1).mean().item()
            nmse = mse * (s ** 2)
            if nmse < best_nmse:
                best_nmse, best_s = nmse, s
        return best_s

    def _compute_resid_scale(self, n_samples=8_000):
        x = torch.randn(n_samples, self.DIM, device=self.device)
        x_s = x / self.opt_scale
        dec1, _ = self.quantize(x_s)
        residual = x_s - dec1

        best_nmse, best_rs = float('inf'), 1.0
        for rs in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]:
            dec2, _ = self.quantize(residual * rs)
            total = dec1 + dec2 / rs
            mse = ((x_s - total) ** 2).sum(-1).mean().item()
            nmse = mse * (self.opt_scale ** 2)
            if nmse < best_nmse:
                best_nmse, best_rs = nmse, rs
        return best_rs

    def make_small(self, n_entries=256):
        """Create a small codebook from the first n_entries vectors (shells 0-1).

        Used as the secondary codebook for 3bpw two-stage quantization.
        256 entries = 8 bits -> 24 bits / 8 dims = 3bpw.
        """
        assert n_entries <= self.CODEBOOK_SIZE
        small_cb = self.codebook[:n_entries].clone()
        obj = object.__new__(type(self))
        obj.codebook = small_cb.to(self.device)
        obj.codebook_norms = (obj.codebook ** 2).sum(-1)
        obj.codebook_half = obj.codebook.half()
        obj.codebook_half_t = obj.codebook_half.T.contiguous()
        obj.codebook_norms_half = obj.codebook_norms.half()
        obj.codebook_packed = _pack_codebook(obj.codebook).to(self.device)
        obj.codesz = self.DIM
        obj.device = self.device
        obj.opt_scale = obj._compute_opt_scale()
        obj.resid_scale = 1.0  # not used for small codebook
        obj.CODEBOOK_SIZE = n_entries
        return obj

    @classmethod
    def build(cls, device='cpu', verbose=True):
        """Fast constructor: load from bundled file if available, else enumerate."""
        if os.path.exists(_BUNDLED_CODEBOOK):
            if verbose:
                print(f"E8ShellCodebook: loading from {_BUNDLED_CODEBOOK}")
            return cls.load(_BUNDLED_CODEBOOK, device=device)
        return cls(device=device, verbose=verbose)

    def _move_to_device(self, device):
        """Move all tensors in-place, preserving object identity for sharing."""
        device = torch.device(device)
        if self.codebook.device == device:
            return
        self.codebook = self.codebook.to(device)
        self.codebook_half = self.codebook.half()
        self.codebook_half_t = self.codebook_half.T.contiguous()
        if hasattr(self, 'codebook_packed') and self.codebook_packed is not None:
            self.codebook_packed = self.codebook_packed.to(device)

    def to(self, device):
        """Move codebook tensors to a device. Returns self if already there."""
        if self.codebook.device == torch.device(device):
            return self
        return self.from_precomputed(
            self.codebook, self.opt_scale, self.resid_scale, device=device)

    def save(self, path: str):
        """Save codebook + parameters for fast loading."""
        torch.save({
            'codebook': self.codebook.cpu(),
            'opt_scale': self.opt_scale,
            'resid_scale': self.resid_scale,
        }, path)

    @classmethod
    def load(cls, path: str, device='cpu'):
        """Load from saved file."""
        data = torch.load(path, map_location='cpu', weights_only=True)
        return cls.from_precomputed(
            data['codebook'], data['opt_scale'], data['resid_scale'], device=device)
