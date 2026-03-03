"""E8 lattice shell codebook for 2bpw quantization."""

import math
import time
import torch
from typing import Tuple


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
        obj.codesz = cls.DIM
        obj.device = device
        obj.opt_scale = opt_scale if opt_scale is not None else obj._compute_opt_scale()
        obj.resid_scale = resid_scale if resid_scale is not None else obj._compute_resid_scale()
        return obj

    def quantize(self, x, _batch_size=4096):
        """Nearest-neighbour lookup via batch matmul."""
        if x.shape[0] <= _batch_size:
            dots = x @ self.codebook.T
            dists = -2.0 * dots + self.codebook_norms
            indices = dists.argmin(dim=-1)
            return self.codebook[indices], indices

        all_indices = []
        for start in range(0, x.shape[0], _batch_size):
            chunk = x[start:start + _batch_size]
            dots = chunk @ self.codebook.T
            dists = -2.0 * dots + self.codebook_norms
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

    def to(self, device):
        """Move codebook tensors to a device. Returns a new instance."""
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
