"""
lattice_quant.py — E₈ / BW₁₆ / Λ₂₄ lattice quantization with Babai rounding + RHT

Proof-of-concept for lattice quantization of LLM weights at multiple dimensions.
Compares E₈ (8-dim), Barnes-Wall BW₁₆ (16-dim), and Leech Λ₂₄ (24-dim) lattices
with Babai nearest-plane rounding, LDLQ (Hessian-aware), and scalar baselines.

Usage:
    python golay-leech-quant-prototype-v2.py                  # run all benchmarks
    python golay-leech-quant-prototype-v2.py --skip-throughput # quality only
    python golay-leech-quant-prototype-v2.py --skip-ldlq      # skip LDLQ

Requires: torch, scipy (for Hadamard matrix)
"""

import torch
import math
import time
import argparse
from typing import Tuple, Dict, Optional

# ============================================================
# Lattice Basis Construction (E₈, BW₁₆, Leech Λ₂₄)
# ============================================================

def leech_basis(lll_reduce: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generator matrix G for the Leech lattice Λ₂₄.

    Uses the SPLAG (Conway & Sloane) integer matrix M divided by √8.
    If fpylll is available and lll_reduce=True, LLL-reduces M first so
    that all basis vectors have minimal norm² = 4.

    Properties: det(G) = 1 (unimodular), min norm² = 4, kissing number = 196,560.

    The matrix encodes three E₈-like blocks in coordinates {1–8}, {9–16},
    {17–24}, with Golay-code cross-glue vectors binding them together,
    and the critical row 24 vector (−3, 1²³) connecting even/odd cosets.

    Returns:
        (G, G_inv): generator and inverse, float32, shape (24, 24)
    """
    # SPLAG integer matrix M — verified across Nebe/Sloane Catalogue,
    # Wikipedia, and LMFDB. The Leech lattice basis is M / √8.
    M = torch.tensor([
        [ 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
        [ 2, 0, 2, 0, 2, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        [ 2, 0, 0, 2, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
        [ 2, 2, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0],
        [ 0, 2, 2, 2, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0],
        [-3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=torch.float64)

    # LLL-reduce for optimal Babai nearest-plane (all basis vectors → norm²=4)
    if lll_reduce:
        try:
            from fpylll import IntegerMatrix, LLL
            B = IntegerMatrix.from_matrix(M.to(torch.int64).tolist())
            LLL.reduction(B)
            M = torch.tensor([[B[i][j] for j in range(24)] for i in range(24)],
                             dtype=torch.float64)
        except ImportError:
            pass  # Use unreduced SPLAG (still det=1, just higher norm range)

    G = M / math.sqrt(8.0)
    G_inv = torch.linalg.inv(G)

    # Convention: the code decodes via `coords @ G.T`, so the effective
    # lattice basis vectors are rows of G.T. We transpose so that G.T
    # contains the LLL-reduced rows (= true Leech lattice basis vectors).
    return G.T.float(), G_inv.T.float()


def e8_basis() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generator matrix G for the E₈ lattice (8 dimensions).

    Uses the simple-roots basis from Conway & Sloane. Already well-conditioned
    for Babai nearest-plane (practical approximation factor ~1).

    Properties: det(G) = 1, min norm² = 2, kissing number = 240.

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


def bw16_basis() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generator matrix G for the Barnes-Wall lattice BW₁₆ (16 dimensions).

    Constructed via the Kronecker product over Gaussian integers ℤ[i]:
      B = [[1, 1], [0, 1+i]],  BW₁₆ generator = B ⊗ B ⊗ B
    The 8×8 complex result is embedded into 16×16 real via
      (a+bi) → [[a, -b], [b, a]].

    This basis is naturally LLL-reduced with condition number ~14.1
    and max consecutive GSO ratio √2. No reduction needed.

    BW₁₆ is the next lattice in the Barnes-Wall family after E₈:
      Z → D₄ → E₈ → BW₁₆ → BW₃₂

    Properties: kissing = 4,320. Normalized to det=1.

    Returns:
        (G, G_inv): generator and inverse, float32, shape (16, 16)
    """
    # B over ℤ[i]: [[1, 1], [0, 1+i]]
    B = torch.tensor([[1+0j, 1+0j],
                      [0+0j, 1+1j]], dtype=torch.complex128)

    # 3-fold Kronecker product: 2×2 → 4×4 → 8×8 complex
    C = torch.kron(torch.kron(B, B), B)

    # Embed 8×8 complex → 16×16 real: (a+bi) → [[a, -b], [b, a]]
    R = torch.zeros(16, 16, dtype=torch.float64)
    for i in range(8):
        for j in range(8):
            a = C[i, j].real.item()
            b = C[i, j].imag.item()
            R[2*i,   2*j]   =  a
            R[2*i,   2*j+1] = -b
            R[2*i+1, 2*j]   =  b
            R[2*i+1, 2*j+1] =  a

    # Normalize to det=1
    det_raw = torch.linalg.det(R).abs().item()
    R = R / (det_raw ** (1.0 / 16))

    G = R.clone()
    G_inv = torch.linalg.inv(G)
    return G.T.float(), G_inv.T.float()


def verify_lattice(basis_fn=None, name="Leech", verbose=True) -> dict:
    """Check lattice properties and print diagnostics. Returns metrics dict."""
    if basis_fn is None:
        basis_fn = leech_basis
    G, G_inv = basis_fn()
    dim = G.shape[0]
    G64 = G.double()
    G_inv64 = G_inv.double()

    # Effective lattice basis vectors are rows of G.T; Gram = G.T @ G
    gram = G64.T @ G64
    det_val = torch.linalg.det(G64).abs().item()
    cond = torch.linalg.cond(G64.T).item()
    norms_sq = gram.diag()

    # Roundtrip error: encode then decode random Gaussian vectors
    # Simple rounding: coords = round(x @ G_inv.T), decoded = coords @ G.T
    torch.manual_seed(0)
    X = torch.randn(5000, dim, dtype=torch.float64) * 0.5
    C = torch.round(X @ G_inv64.T)
    X_hat = C @ G64.T
    errs_simple = (X - X_hat).norm(dim=1)

    # Nearest-plane (Babai's algorithm on GSO of effective basis G.T rows)
    GT = G64.T
    gso = GT.clone()
    for i in range(dim):
        for j in range(i):
            mu_ij = (GT[i] @ gso[j]) / (gso[j] @ gso[j])
            gso[i] = gso[i] - mu_ij * gso[j]
    gsq = (gso * gso).sum(dim=1)
    b = X.clone()
    C_np = torch.zeros_like(X)
    for i in reversed(range(dim)):
        proj = (b * gso[i]).sum(dim=1) / gsq[i]
        ci = torch.round(proj)
        C_np[:, i] = ci
        b = b - ci.unsqueeze(1) * GT[i].unsqueeze(0)
    X_hat_np = C_np @ GT
    errs_np = (X - X_hat_np).norm(dim=1)

    errs = errs_np.tolist()  # report nearest-plane as primary

    # Orthogonality defect of the effective basis (rows of G.T)
    offdiag = gram - torch.diag(gram.diag())
    orth_defect = offdiag.abs().max().item()

    gso_ratio = gsq.max().item() / gsq.min().item()

    metrics = {
        'det': det_val, 'cond': cond,
        'min_norm_sq': norms_sq.min().item(),
        'max_norm_sq': norms_sq.max().item(),
        'mean_err': sum(errs) / len(errs),
        'max_err': max(errs),
        'orth_defect': orth_defect,
        'gso_ratio': gso_ratio,
    }

    if verbose:
        print(f"{name} Lattice Diagnostics (dim={dim})")
        print("-" * 55)
        print(f"  |det(G)|:           {det_val:.6f}  (target: 1.0)")
        print(f"  Condition number:    {cond:.2f}  (lower → better Babai)")
        print(f"  Basis norm² range:   [{norms_sq.min():.4f}, {norms_sq.max():.4f}]")
        print(f"  Orthogonality defect:{orth_defect:.4f}  (0 = orthogonal)")
        print(f"  Simple rounding (5k): mean={errs_simple.mean().item():.4f}  "
              f"max={errs_simple.max().item():.4f}")
        print(f"  Nearest-plane (5k):   mean={metrics['mean_err']:.4f}  "
              f"max={metrics['max_err']:.4f}")
        print(f"  GSO ‖b*‖² range:      [{gsq.min():.4f}, {gsq.max():.4f}] "
              f"(ratio {gso_ratio:.1f})")
        print()

    return metrics


# ============================================================
# Short Vector Enumeration (Fincke-Pohst)
# ============================================================

def enumerate_short_vectors(G, max_norm_sq):
    """
    Enumerate all lattice vectors v = z @ G^T with ||v||² ≤ max_norm_sq.

    Uses Fincke-Pohst / Schnorr-Euchner tree search on the Gram-Schmidt
    orthogonalization.  Processes dimensions from n-1 down to 0, pruning
    the integer-coordinate search tree whenever the remaining norm budget
    is exhausted.

    Args:
        G: (n, n) generator matrix (lattice points = z @ G^T for integer z)
        max_norm_sq: upper bound on squared norm

    Returns:
        coords: (N, n) integer coordinates (float64)
        norms_sq: (N,) squared norms (float64)
    """
    GT = G.double().T          # rows = basis vectors
    n = GT.shape[0]

    # Gram-Schmidt orthogonalization (same as LatticeQuantizer.__init__)
    gso = GT.clone()
    mu = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(i):
            mu[i, j] = (GT[i] @ gso[j]) / (gso[j] @ gso[j])
            gso[i] = gso[i] - mu[i, j] * gso[j]
    gso_sq = (gso * gso).sum(dim=1)      # ||b*_i||²

    # Move to plain Python/numpy for tight DFS loop
    mu_np = mu.numpy()
    gso_sq_np = gso_sq.numpy()

    results = []      # accumulates integer-coordinate tuples

    def _search(level, z, budget):
        """DFS: decide z[level], recurse to level-1."""
        # Projection center: center_j = -Σ_{i>j} z_i · μ_{i,j}
        center = 0.0
        for i in range(level + 1, n):
            center -= z[i] * mu_np[i, level]

        # Bounding: (z[level] - center)² · ||b*_level||² ≤ budget
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


# ============================================================
# Randomized Hadamard Transform (RHT)
# ============================================================

def hadamard_matrix(n: int, device='cpu') -> torch.Tensor:
    """
    Construct a normalized Hadamard matrix of size n (must be power of 2).
    Uses the Sylvester construction: H₂ₖ = [[Hₖ, Hₖ], [Hₖ, -Hₖ]] / √2.
    """
    assert n > 0 and (n & (n - 1)) == 0, f"n={n} must be a power of 2"
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0) / math.sqrt(2.0)
    return H


def fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    In-place Fast Walsh-Hadamard transform along the last dimension.
    O(n log n) instead of O(n²) matrix multiply.
    Input last dim must be a power of 2.
    """
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, f"Last dim {n} must be power of 2"

    h = 1
    while h < n:
        x_reshaped = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a = x_reshaped[..., 0, :].clone()
        b = x_reshaped[..., 1, :].clone()
        x_reshaped[..., 0, :] = a + b
        x_reshaped[..., 1, :] = a - b
        x = x_reshaped.reshape(*x.shape)
        h *= 2

    return x / math.sqrt(n)


class RHT:
    """
    Randomized Hadamard Transform for incoherence processing.

    Transforms weight matrix W and its proxy Hessian H to suppress outliers.
    This is identical to QuIP#'s IP-RHT (Algorithm 3 in the paper).

    For dimensions not a power of 2, pads to next power of 2.

    Usage during quantization:
        rht = RHT(m, n, device)
        W_transformed = rht.transform_weights(W)
        # ... quantize W_transformed ...

    During inference:
        y = rht.transform_input(x)        # V @ diag(sv) @ x
        y = dequant_matmul(W_quant, y)     # fused dequant + GEMV
        y = rht.transform_output(y)        # U @ diag(su) @ y
    """

    def __init__(self, m: int, n: int, device='cpu', seed=42):
        """
        Args:
            m: output dimension (rows of weight matrix)
            n: input dimension (columns of weight matrix)
            device: torch device
            seed: random seed for sign vectors (must be stored for inference)
        """
        self.m_orig, self.n_orig = m, n
        self.device = device

        # Pad to next power of 2
        self.m_pad = 1 << (m - 1).bit_length() if m > 0 else 1
        self.n_pad = 1 << (n - 1).bit_length() if n > 0 else 1

        # Random sign vectors (±1): these are the only stored state
        gen = torch.Generator(device='cpu').manual_seed(seed)
        self.su = (torch.randint(0, 2, (self.m_pad,), generator=gen).float() * 2 - 1).to(device)
        self.sv = (torch.randint(0, 2, (self.n_pad,), generator=gen).float() * 2 - 1).to(device)

    def transform_weights(self, W: torch.Tensor) -> torch.Tensor:
        """
        Apply RHT to weight matrix: W_hat = Had(S_U · Had(S_V · W^T)^T)
        Pads if necessary, returns (m_pad, n_pad) shaped result.
        """
        m, n = W.shape
        dtype = W.dtype

        # Pad W to power-of-2 dims
        W_padded = torch.zeros(self.m_pad, self.n_pad, device=self.device, dtype=dtype)
        W_padded[:m, :n] = W.to(self.device)

        # Step 1: multiply columns by sv, then Hadamard along rows (n-dim)
        W_t = W_padded * self.sv.unsqueeze(0).to(dtype)  # broadcast sv across rows
        W_t = fast_hadamard_transform(W_t)  # FHT along last dim (n)

        # Step 2: multiply rows by su, then Hadamard along columns (m-dim)
        W_t = W_t.T  # (n_pad, m_pad)
        W_t = W_t * self.su.unsqueeze(0).to(dtype)
        W_t = fast_hadamard_transform(W_t)
        W_t = W_t.T  # back to (m_pad, n_pad)

        return W_t

    def transform_hessian(self, H: torch.Tensor) -> torch.Tensor:
        """
        Apply RHT to the Hessian: H_r = Had @ diag(sv) @ H @ diag(sv) @ Had.

        The Hessian H = X^T X lives in column (input) space, so only the column
        transform (sv + Hadamard) applies — not su. This matches QuIP#'s
        RHT_H(H, SU) in quip.py:11-12.
        """
        n = H.shape[0]
        dtype = H.dtype

        # Pad to (n_pad, n_pad)
        H_pad = torch.zeros(self.n_pad, self.n_pad, device=self.device, dtype=dtype)
        H_pad[:n, :n] = H.to(self.device)

        # Step 1: diag(sv) @ H @ diag(sv) — element-wise: H[i,j] *= sv[i] * sv[j]
        sv = self.sv.to(dtype)
        H_pad = H_pad * sv.unsqueeze(0) * sv.unsqueeze(1)

        # Step 2: Had @ H_scaled @ Had (FHT along both axes)
        H_pad = fast_hadamard_transform(H_pad)           # FHT along columns (last dim)
        H_pad = fast_hadamard_transform(H_pad.T).T        # FHT along rows

        return H_pad

    def inverse_transform_weights(self, W_tilde: torch.Tensor) -> torch.Tensor:
        """
        Inverse RHT: W = diag(su) @ Had_m @ W_tilde @ Had_n @ diag(sv).
        Returns (m_orig, n_orig) shaped result (unpadded).
        """
        dtype = W_tilde.dtype

        # Step 1: FHT along last dim (n) → W_tilde @ Had_n
        W_t = fast_hadamard_transform(W_tilde.clone())

        # Step 2: multiply columns by sv → ... @ diag(sv)
        W_t = W_t * self.sv.unsqueeze(0).to(dtype)

        # Step 3: FHT along first dim (m) → Had_m @ ...
        W_t = W_t.T
        W_t = fast_hadamard_transform(W_t.clone())
        W_t = W_t.T

        # Step 4: multiply rows by su → diag(su) @ ...
        W_t = W_t * self.su.unsqueeze(1).to(dtype)

        return W_t[:self.m_orig, :self.n_orig]

    def inverse_transform_output(self, y: torch.Tensor) -> torch.Tensor:
        """Undo the row transform: y_out = Had(S_U · y) [for inference]."""
        y_s = y * self.su[:y.shape[-1]].to(y.dtype)
        return fast_hadamard_transform(y_s)

    def transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input activation: x_hat = Had(S_V · x) [for inference]."""
        x_padded = torch.zeros(self.n_pad, device=self.device, dtype=x.dtype)
        x_padded[:x.shape[-1]] = x.to(self.device)
        x_s = x_padded * self.sv.to(x.dtype)
        return fast_hadamard_transform(x_s)


# ============================================================
# Block LDL Decomposition
# ============================================================

def block_LDL(H: torch.Tensor, block_size: int = 24) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Block LDL decomposition of a symmetric positive-definite matrix.

    Decomposes H = L @ blkdiag(D) @ L^T where:
      - L is block unit-lower-triangular (identity on b×b diagonal blocks)
      - D is block-diagonal with b×b blocks

    This is the decomposition used by LDLQ to propagate quantization error
    through the Hessian structure. Matches quip-sharp/lib/utils/math_utils.py.

    Args:
        H: (n, n) symmetric positive-definite matrix
        block_size: b, must divide n (24 for Leech lattice)

    Returns:
        (L, D): L is (n, n), D is (n//b, b, b)
    """
    n = H.shape[0]
    b = block_size
    assert n % b == 0, f"n={n} must be divisible by block_size={b}"
    m = n // b

    # Standard Cholesky: H = L_chol @ L_chol^T
    L_chol = torch.linalg.cholesky(H)

    # Extract b×b diagonal blocks of L_chol
    # Reshape to (m, b, m, b) and take the diagonal over block indices
    DL = torch.diagonal(L_chol.reshape(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)  # (m, b, b)

    # D = DL @ DL^T (block-diagonal part of the LDL decomposition)
    D = DL @ DL.permute(0, 2, 1)  # (m, b, b)

    # Normalize L by inverting each diagonal block: L[:, i] @= inv(DL[i])
    DL_inv = torch.linalg.inv(DL)  # (m, b, b)
    L = L_chol.reshape(n, m, b).clone()
    for i in range(m):
        L[:, i, :] = L[:, i, :] @ DL_inv[i]
    L = L.reshape(n, n)

    return L, D


# ============================================================
# Synthetic Hessian
# ============================================================

def simulate_hessian(
    n: int,
    n_samples: int = 256,
    sigma_reg: float = 1e-4,
    device: str = 'cpu',
    seed: int = 123,
) -> torch.Tensor:
    """
    Generate a synthetic proxy Hessian H = X^T X / n_samples + sigma_reg * I.

    This approximates the Gauss-Newton Hessian used in GPTQ/QuIP# (the empirical
    covariance of input activations to a linear layer). For benchmarking without
    real calibration data.

    Args:
        n: dimension (number of columns in the weight matrix)
        n_samples: number of simulated calibration samples
        sigma_reg: Tikhonov regularization for positive-definiteness
        device: torch device
        seed: random seed

    Returns:
        H: (n, n) symmetric positive-definite matrix
    """
    gen = torch.Generator(device='cpu').manual_seed(seed)
    X = torch.randn(n_samples, n, generator=gen, device=device)
    H = (X.T @ X) / n_samples
    H = H + sigma_reg * torch.eye(n, device=device)
    return H


# ============================================================
# Quantizer
# ============================================================

class LatticeQuantizer:
    """
    Quantize weight tensors using Babai rounding on a fixed lattice.

    Generic base class parameterized by generator matrix G and dimension.
    All quantization methods (Babai, LDLQ, etc.) are lattice-agnostic.

    Each group of `dim` weights is:
      1. Affine-normalized:  w_norm = (w - offset) / scale
      2. Encoded:            coords = round(G⁻¹ · w_norm)
      3. Clamped:            coords = clamp(coords, -2^(b-1), 2^(b-1)-1)

    Decoding reverses: w_hat = G · coords * scale + offset
    """

    def __init__(self, G: torch.Tensor, G_inv: torch.Tensor, dim: int,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.G = G.to(device)
        self.G_inv = G_inv.to(device)
        self.dim = dim
        self.device = device

        # Precompute Gram-Schmidt orthogonalization of the effective basis
        # (rows of G.T) for Babai nearest-plane decoding.
        GT = self.G.T.double()
        n = GT.shape[0]
        gso = GT.clone()
        for i in range(n):
            for j in range(i):
                mu_ij = (GT[i] @ gso[j]) / (gso[j] @ gso[j])
                gso[i] = gso[i] - mu_ij * gso[j]
        self._gso = gso.float().to(device)
        self._gso_sq = (gso * gso).sum(dim=1).float().to(device)

    def _babai_nearest_plane(self, targets: torch.Tensor,
                              lo: int = None, hi: int = None) -> torch.Tensor:
        """
        Babai nearest-plane algorithm for batch CVP on the Leech lattice.

        Processes GSO dimensions in reverse order, rounding each coordinate
        using the Gram-Schmidt projection. When lo/hi bounds are given, clamps
        each coordinate INSIDE the loop so that excess error flows to subsequent
        (lower-index, larger-GSO) dimensions — bounded nearest-plane.

        Args:
            targets: (m, 24) batch of target vectors
            lo, hi: optional coordinate bounds (e.g. -4, 3 for 3-bit)

        Returns:
            coords: (m, 24) integer coordinates such that coords @ G.T ≈ targets
        """
        GT = self.G.T  # rows = lattice basis vectors
        m = targets.shape[0]
        b = targets.clone()
        coords = torch.zeros(m, self.dim, device=targets.device, dtype=targets.dtype)
        for i in reversed(range(self.dim)):
            proj = (b * self._gso[i]).sum(dim=1) / self._gso_sq[i]
            ci = torch.round(proj)
            if lo is not None:
                ci = ci.clamp(lo, hi)
            coords[:, i] = ci
            b = b - ci.unsqueeze(1) * GT[i].unsqueeze(0)
        return coords

    def quantize(self, w: torch.Tensor, coord_bits: int = 4) -> Dict:
        """
        Quantize weights into lattice coordinates.

        Args:
            w:           weight tensor (any shape, padded to multiple of 24)
            coord_bits:  bits per integer coordinate

        Returns:
            dict with codes, scales, offsets, metadata
        """
        orig_shape = w.shape
        w_flat = w.detach().reshape(-1).float().to(self.device)

        # Pad to multiple of 24
        n = w_flat.shape[0]
        pad = (self.dim - n % self.dim) % self.dim
        if pad > 0:
            w_flat = torch.cat([w_flat, torch.zeros(pad, device=self.device)])

        n_groups = w_flat.shape[0] // self.dim
        w_groups = w_flat.reshape(n_groups, self.dim)

        # Per-group affine normalization
        offsets = w_groups.mean(dim=1, keepdim=True)
        centered = w_groups - offsets
        scales = centered.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
        normalized = centered / scales

        # Bounded Babai nearest-plane: clamp inside the GSO loop
        lo = -(2 ** (coord_bits - 1))
        hi = 2 ** (coord_bits - 1) - 1
        codes_clamped = self._babai_nearest_plane(normalized, lo=lo, hi=hi)

        # Compute actual quantization error (for diagnostics)
        decoded = codes_clamped @ self.G.T
        decoded_denorm = decoded * scales + offsets
        w_hat_flat = decoded_denorm.reshape(-1)
        if pad > 0:
            w_hat_flat = w_hat_flat[:-pad]
        quant_mse = ((w.reshape(-1).float().to(self.device) - w_hat_flat) ** 2).mean().item()

        return {
            'codes': codes_clamped.to(torch.int8 if coord_bits <= 7 else torch.int16),
            'scales': scales.squeeze(1).half(),
            'offsets': offsets.squeeze(1).half(),
            'orig_shape': orig_shape,
            'pad': pad,
            'coord_bits': coord_bits,
            'dim': self.dim,
            'quant_mse': quant_mse,
        }

    def dequantize(self, qd: Dict) -> torch.Tensor:
        """Reconstruct weights from quantized representation."""
        codes = qd['codes'].float().to(self.device)
        scales = qd['scales'].float().to(self.device)
        offsets = qd['offsets'].float().to(self.device)

        decoded = codes @ self.G.T
        decoded = decoded * scales.unsqueeze(1) + offsets.unsqueeze(1)

        w_flat = decoded.reshape(-1)
        if qd['pad'] > 0:
            w_flat = w_flat[:-qd['pad']]
        return w_flat.reshape(qd['orig_shape'])

    @staticmethod
    def bits_per_weight(qd: Dict) -> float:
        """Actual bits per weight including scale/offset overhead."""
        cb = qd['coord_bits']
        n_groups = qd['codes'].shape[0]
        total_weights = math.prod(qd['orig_shape'])
        dim = qd.get('dim', 24)  # backwards compat
        total_bits = n_groups * (dim * cb + 16 + 16)  # coords + FP16 scale + FP16 offset
        return total_bits / total_weights

    def babai_round_global(self, W_block: torch.Tensor, coord_bits: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bounded Babai nearest-plane with NO per-group normalization.

        Used as the inner quantizer within LDLQ. Coordinates are clamped
        inside the nearest-plane loop so excess error flows to subsequent
        dimensions rather than being silently truncated.

        Args:
            W_block: (m, 24) block of weights, already globally scaled
            coord_bits: bits per coordinate for clamping

        Returns:
            (decoded, coords): decoded lattice values (m, 24) and integer coords (m, 24)
        """
        lo = -(2 ** (coord_bits - 1))
        hi = 2 ** (coord_bits - 1) - 1
        coords = self._babai_nearest_plane(W_block, lo=lo, hi=hi)
        decoded = coords @ self.G.T  # (m, 24)
        return decoded, coords

    @staticmethod
    def _babai_on_basis(targets: torch.Tensor, R: torch.Tensor,
                        lo: int = None, hi: int = None) -> torch.Tensor:
        """
        Generic Babai nearest-plane on an arbitrary basis R.

        Computes GSO on the fly from R (rows = basis vectors), then
        runs the standard Babai nearest-plane algorithm. Used for
        Hessian-aware decoding where the whitened basis changes per block.

        Row-vector convention: lattice point = coords @ R.

        Args:
            targets: (m, dim) batch of target vectors
            R: (dim, dim) basis matrix (rows = basis vectors)
            lo, hi: optional coordinate bounds for clamping

        Returns:
            coords: (m, dim) integer coordinates such that coords @ R ~ targets
        """
        dim = R.shape[0]
        R_d = R.double()

        # Compute GSO of R
        gso = R_d.clone()
        for i in range(dim):
            for j in range(i):
                mu_ij = (R_d[i] @ gso[j]) / (gso[j] @ gso[j])
                gso[i] = gso[i] - mu_ij * gso[j]
        gso_f = gso.float().to(targets.device)
        gso_sq = (gso * gso).sum(dim=1).float().to(targets.device)

        # Babai nearest-plane
        m = targets.shape[0]
        b = targets.clone()
        coords = torch.zeros(m, dim, device=targets.device, dtype=targets.dtype)
        for i in reversed(range(dim)):
            proj = (b * gso_f[i]).sum(dim=1) / gso_sq[i]
            ci = torch.round(proj)
            if lo is not None:
                ci = ci.clamp(lo, hi)
            coords[:, i] = ci
            b = b - ci.unsqueeze(1) * R.to(targets.device)[i].unsqueeze(0)
        return coords

    def babai_round_global_weighted(self, W_block: torch.Tensor,
                                     D_k: torch.Tensor,
                                     coord_bits: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hessian-aware Babai nearest-plane: whitens the problem by D_k
        so that Euclidean Babai in the whitened space minimizes the
        Hessian-weighted distance (x - target) @ D_k @ (x - target)^T.

        Derivation (row-vector convention):
            We want to minimize: e @ D_k @ e^T  where e = x - target.
            Factor D_k = U Lambda U^T (eigendecomposition).
            Let S^T = U @ Lambda^{1/2}, so S^T @ S = D_k.
            Then e @ D_k @ e^T = ||e @ S^T||^2.
            With x = coords @ R (original basis):
                ||coords @ R @ S^T - target @ S^T||^2
            So define:
                R_w = R @ S^T = R @ U @ Lambda^{1/2}  (whitened basis)
                target_w = target @ S^T                (whitened target)
            Babai on (target_w, R_w) minimizes the Hessian-weighted distance.
            The decoded point in ORIGINAL space is: coords @ R (NOT coords @ R_w).

        Args:
            W_block: (m, dim) block of weights (row vectors)
            D_k: (dim, dim) block-diagonal Hessian for this block (SPD)
            coord_bits: bits per coordinate for clamping

        Returns:
            (decoded, coords): decoded lattice values (m, dim) in original space,
                               and integer coords (m, dim)
        """
        lo = -(2 ** (coord_bits - 1))
        hi = 2 ** (coord_bits - 1) - 1

        R = self.G.T  # (dim, dim) rows = basis vectors

        # Eigendecompose D_k (symmetric positive semi-definite)
        # Clamp eigenvalues for numerical stability
        eigvals, U = torch.linalg.eigh(D_k.double())
        eigvals = eigvals.clamp(min=1e-10)
        sqrt_eigvals = eigvals.sqrt()

        # S^T = U @ diag(sqrt(lambda))
        ST = U * sqrt_eigvals.unsqueeze(0)  # (dim, dim)
        ST = ST.float()

        # Whitened basis and target
        R_w = R @ ST                            # (dim, dim)
        target_w = W_block @ ST                  # (m, dim)

        # Run generic Babai on the whitened basis
        coords = LatticeQuantizer._babai_on_basis(target_w, R_w, lo=lo, hi=hi)

        # Decode in ORIGINAL space (not whitened)
        decoded = coords @ R  # (m, dim)

        return decoded, coords

    def _ldlq_sweep(self, Wr: torch.Tensor, L: torch.Tensor,
                     num_blocks: int, b: int, coord_bits: int,
                     D: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single backward LDLQ sweep: process blocks right-to-left with
        L-matrix error feedback. Returns (hatWr, codes).

        Args:
            Wr: (m, n) scaled weight matrix
            L: (n, n) lower-triangular factor from block LDL
            num_blocks: number of blocks
            b: block size (dim)
            coord_bits: bits per coordinate
            D: (num_blocks, b, b) optional block-diagonal Hessian from LDL.
               If provided, uses Hessian-aware Babai decoder per block.
        """
        m, n = Wr.shape
        hatWr = torch.zeros_like(Wr)
        codes = torch.zeros_like(Wr)

        for k in reversed(range(num_blocks)):
            kb = k * b
            ke = kb + b

            if ke < n:
                residual_later = Wr[:, ke:] - hatWr[:, ke:]
                L_block = L[ke:, kb:ke]
                feedback = residual_later @ L_block
            else:
                feedback = 0.0

            WXWX = Wr[:, kb:ke] + feedback

            if D is not None:
                decoded, block_codes = self.babai_round_global_weighted(
                    WXWX, D[k], coord_bits=coord_bits)
            else:
                decoded, block_codes = self.babai_round_global(WXWX, coord_bits=coord_bits)

            hatWr[:, kb:ke] = decoded
            codes[:, kb:ke] = block_codes

        return hatWr, codes

    def quantize_ldlq(
        self,
        W: torch.Tensor,
        H: torch.Tensor,
        coord_bits: int = 4,
        tune_iters: int = 0,
        Wscale: Optional[float] = None,
        hessian_aware: bool = False,
    ) -> Dict:
        """
        LDLQ: Hessian-aware sequential vector quantization.

        Processes columns of W in blocks of dim (right-to-left), feeding
        quantization error from later blocks into earlier blocks via the
        LDL decomposition of the Hessian. This is the vector-quantization
        generalization of GPTQ. Matches quip-sharp/lib/algo/quip.py.

        Args:
            W:              (m, n) weight matrix. n must be divisible by dim.
            H:              (n, n) Hessian (proxy or actual), symmetric positive-definite.
            coord_bits:     bits per lattice coordinate
            tune_iters:     number of Hessian refinement passes (0 = L-sweep only)
            Wscale:         global scale. If None, computed as rms(W) / opt_scale.
            hessian_aware:  if True, use Hessian-aware Babai decoder that whitens
                            each block by its D_k from the LDL decomposition.
                            This minimizes (x-target)^T D_k (x-target) instead
                            of ||x-target||^2, closing the metric mismatch gap
                            between lattice and scalar quantizers.

        Returns:
            dict with W_hat, codes, Wscale, quant_mse, proxy_loss, etc.
        """
        m, n = W.shape
        b = self.dim  # block size
        assert n % b == 0, f"n={n} must be divisible by {b}"
        num_blocks = n // b

        W = W.float().to(self.device)
        H = H.float().to(self.device)

        # --- Step 1: Global scaling ---
        # Compute block LDL once (independent of scale)
        damp = 0.01 * torch.diag(H).mean()
        H_reg = H + damp * torch.eye(n, device=self.device)
        L, D = block_LDL(H_reg, block_size=b)  # L: (n,n), D: (num_blocks, b, b)

        # D_blocks is passed to _ldlq_sweep only when hessian_aware is enabled
        D_blocks = D if hessian_aware else None

        if Wscale is None:
            # Auto-tune: grid search over scale factors to minimize proxy loss.
            # QuIP# tunes opt_scale per codebook; we do it per quantization call.
            W_rms = W.pow(2).mean().sqrt().item()
            best_scale, best_loss = None, float('inf')
            for opt_s in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]:
                trial_scale = W_rms / opt_s if W_rms > 1e-10 else 1.0
                trial_hat, _ = self._ldlq_sweep(W / trial_scale, L, num_blocks, b,
                                                coord_bits, D=D_blocks)
                trial_hat = trial_hat * trial_scale
                diff = W - trial_hat
                loss = (diff @ H @ diff.T).diagonal().mean().item()
                if loss < best_loss:
                    best_loss = loss
                    best_scale = trial_scale
            Wscale = best_scale

        Wr = W / Wscale  # (m, n)
        hatWr, codes = self._ldlq_sweep(Wr, L, num_blocks, b, coord_bits, D=D_blocks)

        # --- Step 3: Pass 2+ — Hessian refinement ---
        for ie in range(tune_iters):
            for k in reversed(range(num_blocks)):
                kb = k * b
                ke = kb + b

                residual = Wr - hatWr                                  # (m, n)
                H_col = H_reg[:, kb:ke]                                # (n, b)
                H_diag_inv = torch.linalg.inv(H_reg[kb:ke, kb:ke])    # (b, b)
                feedback = residual @ H_col @ H_diag_inv               # (m, b)

                WXWX = hatWr[:, kb:ke] + feedback
                if hessian_aware:
                    decoded, block_codes = self.babai_round_global_weighted(
                        WXWX, D[k], coord_bits=coord_bits)
                else:
                    decoded, block_codes = self.babai_round_global(WXWX, coord_bits=coord_bits)
                hatWr[:, kb:ke] = decoded
                codes[:, kb:ke] = block_codes

        # --- Step 5: Rescale ---
        W_hat = hatWr * Wscale  # (m, n)

        # --- Step 6: Metrics ---
        quant_mse = ((W - W_hat) ** 2).mean().item()
        diff = W - W_hat
        proxy_loss = (diff @ H @ diff.T).diagonal().mean().item()

        # Effective bpw: coord_bits per weight + negligible global scale overhead
        bpw = coord_bits + 32.0 / (m * n)

        return {
            'W_hat': W_hat,
            'codes': codes.to(torch.int8 if coord_bits <= 7 else torch.int16),
            'Wscale': Wscale,
            'coord_bits': coord_bits,
            'bpw': bpw,
            'quant_mse': quant_mse,
            'proxy_loss': proxy_loss,
            'tune_iters': tune_iters,
            'hessian_aware': hessian_aware,
        }

    def dequantize_global(self, qd: Dict) -> torch.Tensor:
        """Reconstruct weight matrix from global-scale LDLQ codes."""
        codes = qd['codes'].float().to(self.device)
        m, n = codes.shape
        codes_blocks = codes.reshape(-1, self.dim)
        decoded = codes_blocks @ self.G.T
        W_hat = decoded.reshape(m, n) * qd['Wscale']
        return W_hat

    @staticmethod
    def bits_per_weight_global(qd: Dict) -> float:
        """Actual bits per weight for global-scale LDLQ (no per-group overhead)."""
        cb = qd['coord_bits']
        m, n = qd['codes'].shape
        total_bits = m * n * cb + 32  # coords + one float32 global scale
        return total_bits / (m * n)


class LeechQuantizer(LatticeQuantizer):
    """Leech lattice (Λ₂₄) quantizer — 24 dimensions."""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        G, G_inv = leech_basis()
        super().__init__(G, G_inv, 24, device)


class E8Quantizer(LatticeQuantizer):
    """E₈ lattice quantizer — 8 dimensions."""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        G, G_inv = e8_basis()
        super().__init__(G, G_inv, 8, device)


class BW16Quantizer(LatticeQuantizer):
    """Barnes-Wall lattice (BW₁₆) quantizer — 16 dimensions.

    Uses a recursive Kronecker decoder instead of Babai nearest-plane.
    The Kronecker structure B⊗B⊗B splits BW₁₆ → 2×BW₈ → 4×BW₄ → 8×ℤ[i],
    requiring only 3 recursion levels vs Babai's 16 sequential steps.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        G, G_inv = bw16_basis()
        super().__init__(G, G_inv, 16, device)
        # Precompute normalization scale α: G = R_raw / α where R_raw is
        # the real embedding of B⊗B⊗B. We need α to scale targets into
        # the unnormalized lattice for the recursive decoder.
        B = torch.tensor([[1+0j, 1+0j], [0+0j, 1+1j]], dtype=torch.complex128)
        C = torch.kron(torch.kron(B, B), B)
        R = torch.zeros(16, 16, dtype=torch.float64)
        for i in range(8):
            for j in range(8):
                a, b = C[i, j].real.item(), C[i, j].imag.item()
                R[2*i, 2*j] = a;     R[2*i, 2*j+1] = -b
                R[2*i+1, 2*j] = b;   R[2*i+1, 2*j+1] = a
        self._norm_alpha = torch.linalg.det(R).abs().item() ** (1.0 / 16)

    @staticmethod
    def _recursive_decode_complex(target, depth):
        """Recursive BDD for BW_{2^(depth+1)} via Kronecker structure.

        Uses z@C convention (row-vector coords × generator). The generator
        C = B⊗...⊗B has block structure [[D, D], [0, φD]] where φ = 1+i.
        For z=(a,b): t_top = a@D, t_bot = a@D + φ·(b@D). We decode b
        from the difference (t_bot - t_top)/φ, then a from t_top.

        Args:
            target: (..., 2^depth) complex tensor
            depth: recursion depth (3→BW₁₆, 2→BW₈, 1→BW₄, 0→ℤ[i])

        Returns:
            (point, coords): decoded lattice point and ℤ[i] coordinates
        """
        if depth == 0:
            # Base case: ℤ[i] — round real and imag independently
            z = torch.round(target.real) + 1j * torch.round(target.imag)
            return z, z

        half = target.shape[-1] // 2
        t_top = target[..., :half]
        t_bot = target[..., half:]

        phi = 1 + 1j
        phi_inv = (1 - 1j) / 2  # = 1/(1+i)

        # z@C convention: for z=(a,b) and C=[[D,D],[0,φD]],
        #   t_top = a@D,  t_bot = a@D + φ·(b@D)
        # So φ·(b@D) = t_bot - t_top, and a@D = t_top.

        # Decode b from difference: φ·(b@D) = t_bot - t_top
        b_target = (t_bot - t_top) * phi_inv
        b_point, z_bot = BW16Quantizer._recursive_decode_complex(b_target, depth - 1)

        # Decode a from top half directly: a@D = t_top
        a_target = t_top
        a_point, z_top = BW16Quantizer._recursive_decode_complex(a_target, depth - 1)

        point = torch.cat([a_point, a_point + phi * b_point], dim=-1)
        coords = torch.cat([z_top, z_bot], dim=-1)
        return point, coords

    def _babai_nearest_plane(self, targets, lo=None, hi=None):
        """Override: recursive Kronecker decoder instead of Babai.

        Scales targets into the unnormalized lattice, decodes via the
        recursive Kronecker structure in ℂ⁸, then converts ℤ[i]⁸ → ℤ¹⁶.

        The real embedding uses column convention [[a,-b],[b,a]], so
        z_real @ R implements z_complex @ conj(C) in complex. The decoder
        inverts z @ C, so we conjugate the target before and coords after.
        """
        alpha = self._norm_alpha

        # Scale to unnormalized lattice: G = R_raw/α, so targets*α are
        # in the space where the recursive decoder operates.
        t_scaled = targets.double() * alpha

        # ℝ¹⁶ → ℂ⁸ (our real embedding: complex[k] = real[2k] + i·real[2k+1])
        t_complex = torch.complex(t_scaled[..., 0::2], t_scaled[..., 1::2])

        # Column-convention embedding: z_real@R = real_embed(z@C̄).
        # Decoder inverts z@C, so pass conj(target), get conj(coords).
        _, coords_conj = self._recursive_decode_complex(t_complex.conj(), depth=3)
        coords_complex = coords_conj.conj()

        # ℤ[i]⁸ → ℤ¹⁶ (interleave real/imag)
        coords = torch.zeros_like(targets)
        coords[..., 0::2] = coords_complex.real.float()
        coords[..., 1::2] = coords_complex.imag.float()

        if lo is not None:
            coords = coords.clamp(lo, hi)

        return coords


# ============================================================
# BW₁₆ Shell-Based Finite Codebook
# ============================================================

class BW16ShellCodebook:
    """
    Finite codebook from BW₁₆ lattice shells.

    Enumerates lattice vectors in shells 0 (origin), 1 (kissing, 4320),
    and 2 (61440) for a total of 65761 ≈ 2^16 vectors.  Selects exactly
    2^16 = 65536 entries.  Encoding is nearest-neighbour via batch matmul;
    decoding is a simple table lookup.

    Bit rate: 16 bits / 16 dims = 1 bpw (+ negligible global scale).
    """

    CODEBOOK_SIZE = 65536   # 2^16
    DIM = 16

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu',
                 verbose=True):
        G, _ = bw16_basis()

        # ---- shell enumeration ----
        # Shell 1 at norm²=2√2, shell 2 at norm²=3√2 ≈ 4.243.
        # Use 4.5 to include shells 0+1+2 but exclude shell 3 (≈5.657).
        if verbose:
            print("BW16ShellCodebook: enumerating lattice shells …")
        t0 = time.perf_counter()
        coords, norms_sq = enumerate_short_vectors(G, max_norm_sq=4.5)
        t_enum = time.perf_counter() - t0

        # Group into shells and verify theta-series counts
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
        shell_norms = [norms_sq[shell_boundaries[i]].item()
                       for i in range(len(shell_boundaries) - 1)]

        if verbose:
            print(f"  Enumerated {len(coords)} vectors in {t_enum:.2f}s")
            for sn, sc in zip(shell_norms, shell_counts):
                print(f"    norm²={sn:.6f}: {sc:,} vectors")

        assert shell_counts[0] == 1, f"Shell 0 should be origin, got {shell_counts[0]}"
        assert shell_counts[1] == 4320, f"Shell 1 kissing number should be 4320, got {shell_counts[1]}"
        assert shell_counts[2] == 61440, f"Shell 2 should be 61440, got {shell_counts[2]}"

        total = sum(shell_counts[:3])
        assert total == 65761, f"Expected 65761 total, got {total}"

        # ---- select 2^16 entries ----
        # Keep all of shells 0+1 (4321), plus first 61215 of shell 2
        # (sorted by ascending norm² — all shell-2 vecs have same norm,
        # so just take the first 61215 in enumeration order).
        n_from_shell2 = self.CODEBOOK_SIZE - (shell_counts[0] + shell_counts[1])
        keep = shell_boundaries[2] + n_from_shell2
        coords_sel = coords[:keep]

        # Compute lattice vectors from integer coordinates
        GT = G.T.double()
        vecs = coords_sel @ GT    # (65536, 16)

        self.codebook = vecs.float().to(device)              # (65536, 16)
        self.codebook_norms = (self.codebook ** 2).sum(-1)   # (65536,)
        self.codesz = self.DIM
        self.device = device

        # Compute opt_scale (Gaussian NMSE minimisation) and resid_scale (for RVQ)
        self.opt_scale = self._compute_opt_scale()
        self.resid_scale = self._compute_resid_scale()
        if verbose:
            print(f"  Codebook: {self.CODEBOOK_SIZE} entries, "
                  f"opt_scale={self.opt_scale:.4f}, resid_scale={self.resid_scale:.2f}")
            print()

    # ---- core quantise / decode -----------------------------------

    def quantize(self, x, _batch_size=4096):
        """
        Nearest-neighbour lookup via batch matmul.

        Args:
            x: (batch, 16) float tensor

        Returns:
            (decoded, indices): decoded codebook vectors (batch, 16)
                                and uint16-range indices (batch,)
        """
        # Process in chunks to avoid OOM on the (batch, 65536) distance matrix
        if x.shape[0] <= _batch_size:
            dots = x @ self.codebook.T                      # (batch, 65536)
            dists = -2.0 * dots + self.codebook_norms       # (batch, 65536)
            indices = dists.argmin(dim=-1)                   # (batch,)
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
        """Table lookup: index → 16-d vector."""
        return self.codebook[indices]

    # ---- RVQ (residual vector quantisation) for 2+ bpw ------------

    def quantize_rvq(self, x):
        """
        Two-stage RVQ: 2 bpw (32 bits / 16 dims).

        Stage 1: nearest-neighbour → 16 bits
        Stage 2: nearest-neighbour on scaled residual → 16 bits

        Args:
            x: (batch, 16) float tensor (already in codebook scale)

        Returns:
            (decoded, (idx1, idx2)): decoded vectors and index pairs
        """
        dec1, idx1 = self.quantize(x)
        residual = x - dec1
        dec2, idx2 = self.quantize(residual * self.resid_scale)
        decoded = dec1 + dec2 / self.resid_scale
        return decoded, (idx1, idx2)

    def decode_rvq(self, idx1, idx2):
        """Decode two-stage RVQ indices."""
        return self.codebook[idx1] + self.codebook[idx2] / self.resid_scale

    # ---- opt_scale / resid_scale -----------------------------------

    def _compute_opt_scale(self, n_samples=8_000):
        """Find scaling s minimising normalised MSE for Gaussian input."""
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
        """Find residual scaling for two-stage RVQ."""
        x = torch.randn(n_samples, self.DIM, device=self.device)
        # Stage 1 at opt_scale
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


# ---- standalone helpers for codebook-based quantisation --------

def quantize_pergroup_codebook(w: torch.Tensor, codebook: BW16ShellCodebook):
    """
    Per-group quantisation with finite codebook.

    Each 16-weight group is affine-normalised (mean/scale), quantised
    via nearest-neighbour codebook lookup, then denormalised.

    Args:
        w: 1-d weight tensor (length must be multiple of 16)
        codebook: BW16ShellCodebook instance

    Returns:
        (w_hat, indices): reconstructed weights and codebook indices
    """
    D = codebook.codesz
    w_flat = w.detach().float().to(codebook.device).reshape(-1)
    n = w_flat.shape[0]
    pad = (D - n % D) % D
    if pad > 0:
        w_flat = torch.cat([w_flat, torch.zeros(pad, device=codebook.device)])
    groups = w_flat.reshape(-1, D)

    offsets = groups.mean(dim=1, keepdim=True)
    centered = groups - offsets
    scales = centered.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    normalized = centered / (scales / codebook.opt_scale)

    decoded, indices = codebook.quantize(normalized)
    w_hat = decoded * (scales / codebook.opt_scale) + offsets
    w_hat = w_hat.reshape(-1)
    if pad > 0:
        w_hat = w_hat[:-pad]
    return w_hat, indices


def quantize_ldlq_codebook(
    W: torch.Tensor,
    H: torch.Tensor,
    codebook: BW16ShellCodebook,
    tune_iters: int = 0,
    Wscale: Optional[float] = None,
):
    """
    LDLQ (Hessian-aware sequential VQ) using a finite codebook.

    Mirrors LatticeQuantizer.quantize_ldlq() but replaces bounded
    Babai rounding with codebook nearest-neighbour lookup.

    Args:
        W: (m, n) weight matrix. n must be divisible by 16.
        H: (n, n) Hessian, symmetric positive-definite.
        codebook: BW16ShellCodebook instance.
        tune_iters: extra refinement passes (0 = single L-sweep).
        Wscale: global scale. If None, computed from rms(W)/opt_scale.

    Returns:
        dict with W_hat, indices, Wscale, bpw, quant_mse, proxy_loss.
    """
    device = codebook.device
    m, n = W.shape
    b = codebook.codesz  # 16
    assert n % b == 0, f"n={n} must be divisible by {b}"
    num_blocks = n // b

    W = W.float().to(device)
    H = H.float().to(device)

    # Block LDL decomposition — damp like GPTQ to ensure PD
    damp = 0.01 * torch.diag(H).mean()
    H_reg = H + damp * torch.eye(n, device=device)
    L, D = block_LDL(H_reg, block_size=b)

    # Global scaling — use precomputed opt_scale (like QuIP#) instead of
    # expensive grid search (each trial requires a full NN sweep).
    if Wscale is None:
        W_rms = W.pow(2).mean().sqrt().item()
        Wscale = W_rms / codebook.opt_scale if W_rms > 1e-10 else 1.0

    # Final sweep with best scale
    Wr = W / Wscale
    hatWr = torch.zeros_like(Wr)
    all_indices = torch.zeros(m, num_blocks, dtype=torch.long, device=device)

    for k in reversed(range(num_blocks)):
        kb, ke = k * b, (k + 1) * b
        if ke < n:
            feedback = (Wr[:, ke:] - hatWr[:, ke:]) @ L[ke:, kb:ke]
        else:
            feedback = 0.0
        WXWX = Wr[:, kb:ke] + feedback
        hatWr[:, kb:ke], all_indices[:, k] = codebook.quantize(WXWX)

    # Refinement passes
    for ie in range(tune_iters):
        for k in reversed(range(num_blocks)):
            kb, ke = k * b, (k + 1) * b
            residual = Wr - hatWr
            H_col = H_reg[:, kb:ke]
            H_diag_inv = torch.linalg.inv(H_reg[kb:ke, kb:ke])
            feedback = residual @ H_col @ H_diag_inv
            WXWX = hatWr[:, kb:ke] + feedback
            hatWr[:, kb:ke], all_indices[:, k] = codebook.quantize(WXWX)

    W_hat = hatWr * Wscale

    # Metrics
    quant_mse = ((W - W_hat) ** 2).mean().item()
    diff = W - W_hat
    proxy_loss = (diff @ H @ diff.T).diagonal().mean().item()
    bpw = 16.0 / b + 32.0 / (m * n)   # 1.0 + negligible global scale

    return {
        'W_hat': W_hat,
        'indices': all_indices,
        'Wscale': Wscale,
        'bpw': bpw,
        'quant_mse': quant_mse,
        'proxy_loss': proxy_loss,
        'tune_iters': tune_iters,
    }


def quantize_ldlq_codebook_rvq(
    W: torch.Tensor,
    H: torch.Tensor,
    codebook: BW16ShellCodebook,
    tune_iters: int = 0,
    Wscale: Optional[float] = None,
):
    """
    LDLQ with two-stage RVQ codebook — 2 bpw (32 bits / 16 dims).

    Same as quantize_ldlq_codebook but uses codebook.quantize_rvq()
    for each block, giving twice the bit budget.
    """
    device = codebook.device
    m, n = W.shape
    b = codebook.codesz
    assert n % b == 0
    num_blocks = n // b

    W = W.float().to(device)
    H = H.float().to(device)

    damp = 0.01 * torch.diag(H).mean()
    H_reg = H + damp * torch.eye(n, device=device)
    L, D = block_LDL(H_reg, block_size=b)

    if Wscale is None:
        W_rms = W.pow(2).mean().sqrt().item()
        Wscale = W_rms / codebook.opt_scale if W_rms > 1e-10 else 1.0

    Wr = W / Wscale
    hatWr = torch.zeros_like(Wr)

    for k in reversed(range(num_blocks)):
        kb, ke = k * b, (k + 1) * b
        if ke < n:
            feedback = (Wr[:, ke:] - hatWr[:, ke:]) @ L[ke:, kb:ke]
        else:
            feedback = 0.0
        WXWX = Wr[:, kb:ke] + feedback
        hatWr[:, kb:ke], _ = codebook.quantize_rvq(WXWX)

    for ie in range(tune_iters):
        for k in reversed(range(num_blocks)):
            kb, ke = k * b, (k + 1) * b
            residual = Wr - hatWr
            H_col = H_reg[:, kb:ke]
            H_diag_inv = torch.linalg.inv(H_reg[kb:ke, kb:ke])
            feedback = residual @ H_col @ H_diag_inv
            WXWX = hatWr[:, kb:ke] + feedback
            hatWr[:, kb:ke], _ = codebook.quantize_rvq(WXWX)

    W_hat = hatWr * Wscale

    quant_mse = ((W - W_hat) ** 2).mean().item()
    diff = W - W_hat
    proxy_loss = (diff @ H @ diff.T).diagonal().mean().item()
    bpw = 32.0 / b + 32.0 / (m * n)   # 2.0 + negligible

    return {
        'W_hat': W_hat,
        'Wscale': Wscale,
        'bpw': bpw,
        'quant_mse': quant_mse,
        'proxy_loss': proxy_loss,
        'tune_iters': tune_iters,
    }


# ============================================================
# E₈ Shell-Based Finite Codebook
# ============================================================

class E8ShellCodebook:
    """
    Finite codebook from E₈ lattice shells.

    Enumerates lattice vectors in shells 0 through 6 to collect >= 65536 vectors.
    E₈ theta series (OEIS A004009):
      Shell 0: norm²=0,  1 vector (origin)
      Shell 1: norm²=2,  240 vectors (kissing number)
      Shell 2: norm²=4,  2160 vectors
      Shell 3: norm²=6,  6720 vectors
      Shell 4: norm²=8,  17520 vectors
      Shell 5: norm²=10, 30240 vectors
      Shell 6: norm²=12, 60480 vectors
      Cumulative through shell 5: 56881
      Need 8655 from shell 6 to reach 65536 = 2^16

    Encoding is nearest-neighbour via batch matmul; decoding is table lookup.

    Bit rate: 16 bits / 8 dims = 2 bpw (+ negligible global scale).
    With RVQ (two-stage): 32 bits / 8 dims = 4 bpw.

    This is the core codebook used by QuIP# (E8P lattice), which achieves
    state-of-the-art 2 bpw quantization by combining E8 shells with RHT
    incoherence processing and LDLQ Hessian-aware error propagation.
    """

    CODEBOOK_SIZE = 65536   # 2^16
    DIM = 8

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu',
                 verbose=True):
        G, _ = e8_basis()

        # ---- shell enumeration ----
        # E₈ has norm² = 2k for shell k. We need through shell 6 (norm²=12).
        # Use 12.5 to include shell 6 but exclude shell 7 (norm²=14).
        if verbose:
            print("E8ShellCodebook: enumerating lattice shells ...")
        t0 = time.perf_counter()
        coords, norms_sq = enumerate_short_vectors(G, max_norm_sq=12.5)
        t_enum = time.perf_counter() - t0

        # Group into shells and verify theta-series counts
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
        shell_norms = [norms_sq[shell_boundaries[i]].item()
                       for i in range(len(shell_boundaries) - 1)]

        if verbose:
            print(f"  Enumerated {len(coords)} vectors in {t_enum:.2f}s")
            cumul = 0
            for sn, sc in zip(shell_norms, shell_counts):
                cumul += sc
                print(f"    norm^2={sn:.4f}: {sc:>6,} vectors (cumul: {cumul:>7,})")

        # Verify E₈ theta series
        expected_counts = [1, 240, 2160, 6720, 17520, 30240, 60480]
        for i, (exp, got) in enumerate(zip(expected_counts, shell_counts)):
            assert got == exp, (
                f"Shell {i} count mismatch: expected {exp}, got {got}")

        total = sum(shell_counts)
        assert total >= self.CODEBOOK_SIZE, (
            f"Need {self.CODEBOOK_SIZE} vectors but only enumerated {total}")

        # ---- select 2^16 entries ----
        # Keep all of shells 0-5 (56881), plus first 8655 of shell 6.
        # All shell-6 vectors have the same norm², so selection order
        # within that shell is arbitrary (we use enumeration order).
        through_shell5 = sum(shell_counts[:6])   # 56881
        n_from_shell6 = self.CODEBOOK_SIZE - through_shell5  # 8655
        keep = shell_boundaries[6] + n_from_shell6
        coords_sel = coords[:keep]

        assert coords_sel.shape[0] == self.CODEBOOK_SIZE, (
            f"Expected {self.CODEBOOK_SIZE} entries, got {coords_sel.shape[0]}")

        # Compute lattice vectors from integer coordinates
        GT = G.T.double()
        vecs = coords_sel @ GT    # (65536, 8)

        self.codebook = vecs.float().to(device)              # (65536, 8)
        self.codebook_norms = (self.codebook ** 2).sum(-1)   # (65536,)
        self.codesz = self.DIM
        self.device = device

        # Compute opt_scale (Gaussian NMSE minimisation) and resid_scale (for RVQ)
        self.opt_scale = self._compute_opt_scale()
        self.resid_scale = self._compute_resid_scale()
        if verbose:
            print(f"  Codebook: {self.CODEBOOK_SIZE} entries, "
                  f"opt_scale={self.opt_scale:.4f}, resid_scale={self.resid_scale:.2f}")
            print()

    # ---- core quantise / decode -----------------------------------

    def quantize(self, x, _batch_size=4096):
        """
        Nearest-neighbour lookup via batch matmul.

        Args:
            x: (batch, 8) float tensor

        Returns:
            (decoded, indices): decoded codebook vectors (batch, 8)
                                and uint16-range indices (batch,)
        """
        # Process in chunks to avoid OOM on the (batch, 65536) distance matrix
        if x.shape[0] <= _batch_size:
            dots = x @ self.codebook.T                      # (batch, 65536)
            dists = -2.0 * dots + self.codebook_norms       # (batch, 65536)
            indices = dists.argmin(dim=-1)                   # (batch,)
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

    # ---- RVQ (residual vector quantisation) for 4 bpw ---------------

    def quantize_rvq(self, x):
        """
        Two-stage RVQ: 4 bpw (32 bits / 8 dims).

        Stage 1: nearest-neighbour -> 16 bits
        Stage 2: nearest-neighbour on scaled residual -> 16 bits

        Args:
            x: (batch, 8) float tensor (already in codebook scale)

        Returns:
            (decoded, (idx1, idx2)): decoded vectors and index pairs
        """
        dec1, idx1 = self.quantize(x)
        residual = x - dec1
        dec2, idx2 = self.quantize(residual * self.resid_scale)
        decoded = dec1 + dec2 / self.resid_scale
        return decoded, (idx1, idx2)

    def decode_rvq(self, idx1, idx2):
        """Decode two-stage RVQ indices."""
        return self.codebook[idx1] + self.codebook[idx2] / self.resid_scale

    # ---- opt_scale / resid_scale -----------------------------------

    def _compute_opt_scale(self, n_samples=8_000):
        """Find scaling s minimising normalised MSE for Gaussian input."""
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
        """Find residual scaling for two-stage RVQ."""
        x = torch.randn(n_samples, self.DIM, device=self.device)
        # Stage 1 at opt_scale
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


# ============================================================
# Scalar Quantization Baselines
# ============================================================

def scalar_quantize(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-tensor uniform scalar quantization."""
    w_min, w_max = w.min(), w.max()
    levels = 2 ** bits
    step = (w_max - w_min) / (levels - 1)
    return torch.round((w - w_min) / step) * step + w_min


def scalar_quantize_grouped(w: torch.Tensor, bits: int, group_size: int = 32) -> torch.Tensor:
    """
    Per-group scalar quantization (similar to Q_K quants).
    Each group of `group_size` weights gets its own min/max.
    """
    orig_shape = w.shape
    w_flat = w.reshape(-1)
    n = w_flat.shape[0]
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        w_flat = torch.cat([w_flat, torch.zeros(pad, device=w.device)])

    groups = w_flat.reshape(-1, group_size)
    g_min = groups.min(dim=1, keepdim=True).values
    g_max = groups.max(dim=1, keepdim=True).values
    levels = 2 ** bits
    step = (g_max - g_min) / (levels - 1)
    step = step.clamp(min=1e-10)
    quantized = torch.round((groups - g_min) / step) * step + g_min

    w_out = quantized.reshape(-1)
    if pad > 0:
        w_out = w_out[:-pad]
    return w_out.reshape(orig_shape)


# ============================================================
# Benchmarks
# ============================================================

def benchmark_quality(use_rht: bool = True, n_weights: int = 48000, seed: int = 42):
    """
    Compare quantization quality: E₈/BW₁₆/Leech lattices vs scalar baselines.
    Tests on Gaussian weights (simulating RHT-processed LLM weights).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)

    w_raw = torch.randn(n_weights, device=device) * 0.02

    if use_rht:
        label = "Gaussian (post-RHT)"
        w = w_raw
    else:
        w = w_raw.clone()
        n_outliers = n_weights // 100
        outlier_idx = torch.randperm(n_weights)[:n_outliers]
        w[outlier_idx] *= 10
        label = "Gaussian + 1% outliers (no RHT)"

    signal_power = (w ** 2).mean().item()

    lattices = [
        ('E8', E8Quantizer(device)),
        ('BW16', BW16Quantizer(device)),
        ('Leech', LeechQuantizer(device)),
    ]

    print(f"Quality Benchmark — {label}")
    print(f"  {n_weights:,} weights on {device}, σ_signal = {math.sqrt(signal_power):.4f}")
    print()

    header = f"  {'Method':<40} {'bpw':>6} {'MSE':>11} {'SQNR dB':>9}"
    print(header)
    print("  " + "=" * (len(header) - 2))

    for lname, quant in lattices:
        for cb in [1, 2, 3, 4, 5, 6, 7]:
            qd = quant.quantize(w, coord_bits=cb)
            w_hat = quant.dequantize(qd)
            mse = ((w - w_hat) ** 2).mean().item()
            sqnr = 10 * math.log10(signal_power / max(mse, 1e-30))
            bpw = quant.bits_per_weight(qd)
            print(f"  {lname} Babai per-grp ({cb}b){' '*(20-len(lname))} "
                  f"{bpw:>6.2f} {mse:>11.2e} {sqnr:>9.2f}")

    # BW16 Shell Codebook — per-group
    shell_cb = BW16ShellCodebook(device=device, verbose=False)
    w_hat_shell, _ = quantize_pergroup_codebook(w, shell_cb)
    mse_shell = ((w - w_hat_shell) ** 2).mean().item()
    sqnr_shell = 10 * math.log10(signal_power / max(mse_shell, 1e-30))
    bpw_shell = 1.0 + 32.0 / 16  # 16-bit index + FP16 scale+offset per 16-group
    print(f"  {'BW16 Shell per-grp (1b+overhead)':<40} "
          f"{bpw_shell:>6.2f} {mse_shell:>11.2e} {sqnr_shell:>9.2f}")

    print()

    # Scalar baselines: per-tensor
    for bits in [1, 2, 3, 4, 5, 6, 7, 8]:
        w_hat = scalar_quantize(w, bits)
        mse = ((w - w_hat) ** 2).mean().item()
        sqnr = 10 * math.log10(signal_power / max(mse, 1e-30))
        print(f"  {'Scalar per-tensor (' + str(bits) + 'b)':<40} "
              f"{bits:>6.2f} {mse:>11.2e} {sqnr:>9.2f}")

    print()

    # Scalar baselines: per-group
    for bits, gs in [(1, 32), (2, 32), (3, 32), (4, 32), (5, 32), (6, 32), (7, 32), (4, 128)]:
        w_hat = scalar_quantize_grouped(w, bits, group_size=gs)
        mse = ((w - w_hat) ** 2).mean().item()
        sqnr = 10 * math.log10(signal_power / max(mse, 1e-30))
        oh = 32 / gs
        eff_bpw = bits + oh
        print(f"  {'Scalar grp=' + str(gs) + ' (' + str(bits) + 'b)':<40} "
              f"{eff_bpw:>6.1f} {mse:>11.2e} {sqnr:>9.2f}")

    print()

    # BW₁₆ shell codebook (per-group)
    cb = BW16ShellCodebook(device=device, verbose=False)
    w_hat_cb, _ = quantize_pergroup_codebook(w, cb)
    mse_cb = ((w - w_hat_cb) ** 2).mean().item()
    sqnr_cb = 10 * math.log10(signal_power / max(mse_cb, 1e-30))
    # bpw = 16/16 + (16+16)/16 = 3.0 (1bpw base + 2bpw scale/offset overhead)
    bpw_cb = 1.0 + 32.0 / 16
    print(f"  {'BW16 Shell codebook (per-grp)':<40} {bpw_cb:>6.2f} "
          f"{mse_cb:>11.2e} {sqnr_cb:>9.2f}")
    print()


def benchmark_rht_effect(n_weights: int = 48000, seed: int = 42):
    """Show the effect of RHT on quantization quality."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)

    # Raw weights with outliers
    w = torch.randn(n_weights, device=device) * 0.02
    n_outliers = n_weights // 100
    outlier_idx = torch.randperm(n_weights, device=device)[:n_outliers]
    w[outlier_idx] *= 10.0

    # Reshape to a "weight matrix" for RHT
    # RHT needs power-of-2 dims, so pick something reasonable
    m = 192  # 192 = 48 * 4, multiple of lcm(8,16,24)=48
    n = n_weights // m
    W_mat = w[:m * n].reshape(m, n)

    # Apply RHT
    rht = RHT(m, n, device=device)
    W_rht = rht.transform_weights(W_mat.float())
    # Take the valid region
    w_rht = W_rht[:m, :n].reshape(-1)

    lattices = [('E8', E8Quantizer), ('BW16', BW16Quantizer), ('Leech', LeechQuantizer)]

    for lname, QuantCls in lattices:
        quant = QuantCls(device)

        print(f"RHT Effect on {lname} Quantization (coord_bits=3)")
        print("-" * 55)

        for label, ww in [("Raw (with outliers)", w[:m*n]), ("After RHT", w_rht)]:
            sp = (ww ** 2).mean().item()
            qd = quant.quantize(ww, coord_bits=3)
            wh = quant.dequantize(qd)
            mse = ((ww - wh) ** 2).mean().item()
            sqnr = 10 * math.log10(sp / max(mse, 1e-30))
            bpw = quant.bits_per_weight(qd)

            max_val = ww.abs().max().item()
            kurt = ((ww ** 4).mean() / (ww ** 2).mean() ** 2).item()

            print(f"  {label:<25} bpw={bpw:.2f}  SQNR={sqnr:.1f}dB  "
                  f"max|w|={max_val:.4f}  kurtosis={kurt:.1f}")

        print()


def benchmark_throughput(n_groups: int = 100_000, n_iters: int = 100):
    """Measure encode/decode speed for each lattice."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    lattices = [('E8', E8Quantizer), ('BW16', BW16Quantizer), ('Leech', LeechQuantizer)]

    print("Throughput Benchmark (PyTorch, not fused CUDA)")
    print("-" * 65)
    print(f"  Device: {device}")

    for lname, QuantCls in lattices:
        quant = QuantCls(device)
        n_weights = n_groups * quant.dim
        w = torch.randn(n_weights, device=device) * 0.02

        # Warmup
        for _ in range(5):
            qd = quant.quantize(w, coord_bits=3)
            _ = quant.dequantize(qd)
        if device == 'cuda':
            torch.cuda.synchronize()

        # Encode
        t0 = time.perf_counter()
        for _ in range(n_iters):
            qd = quant.quantize(w, coord_bits=3)
        if device == 'cuda':
            torch.cuda.synchronize()
        t_enc = (time.perf_counter() - t0) / n_iters

        # Decode
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = quant.dequantize(qd)
        if device == 'cuda':
            torch.cuda.synchronize()
        t_dec = (time.perf_counter() - t0) / n_iters

        print(f"  [{lname:>5}] {n_weights:,}w  "
              f"enc={t_enc*1000:.2f}ms ({n_weights/t_enc/1e9:.2f} Gw/s)  "
              f"dec={t_dec*1000:.2f}ms ({n_weights/t_dec/1e9:.2f} Gw/s)")

    print(f"  Note: a fused CUDA kernel would be ~10-50× faster")
    print()


def benchmark_ldlq(m: int = 256, n: int = 240, tune_iters: int = 1, seed: int = 42):
    """
    Compare LDLQ (Hessian-aware) vs independent Babai vs scalar baselines.
    Tests E₈, BW₁₆, and Leech at multiple coord_bits.
    Reports bpw, MSE/SQNR, Hessian-weighted proxy loss, and clamp fraction.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)

    # n must be divisible by lcm(8, 16, 24) = 48
    assert n % 48 == 0, f"n={n} must be divisible by 48 (lcm of 8,16,24)"

    W = torch.randn(m, n, device=device) * 0.02
    H = simulate_hessian(n, n_samples=max(n, 256), device=device, seed=seed + 1)

    signal_power = (W ** 2).mean().item()

    lattices = [
        ('E8', E8Quantizer(device)),
        ('BW16', BW16Quantizer(device)),
        ('Leech', LeechQuantizer(device)),
    ]

    def compute_proxy_loss(W_orig, W_hat, H):
        diff = (W_orig - W_hat).float()
        return (diff @ H.float() @ diff.T).diagonal().mean().item()

    print(f"LDLQ Benchmark — W: ({m}, {n})")
    print(f"  signal rms = {math.sqrt(signal_power):.4f}")
    print()

    header = f"  {'Method':<40} {'bpw':>6} {'MSE':>11} {'SQNR dB':>9} {'Proxy Loss':>12} {'Clamp%':>7}"
    print(header)
    print("  " + "=" * (len(header) - 2))

    for lname, quant in lattices:
        for coord_bits in [1, 2, 3, 4, 5, 6, 7]:
            lo = -(2 ** (coord_bits - 1))
            hi = 2 ** (coord_bits - 1) - 1

            # (a) Independent Babai (per-group)
            w_flat = W.reshape(-1)
            qd = quant.quantize(w_flat, coord_bits=coord_bits)
            w_hat_flat = quant.dequantize(qd)
            W_hat_indep = w_hat_flat.reshape(m, n)
            mse_indep = ((W - W_hat_indep) ** 2).mean().item()
            sqnr_indep = 10 * math.log10(signal_power / max(mse_indep, 1e-30))
            proxy_indep = compute_proxy_loss(W, W_hat_indep, H)
            bpw_indep = quant.bits_per_weight(qd)
            print(f"  {f'{lname} Babai per-grp ({coord_bits}b)':<40} {bpw_indep:>6.2f} "
                  f"{mse_indep:>11.2e} {sqnr_indep:>9.2f} {proxy_indep:>12.4e} {'':>7}")

            # (b) LDLQ global (tune=0)
            result0 = quant.quantize_ldlq(W, H, coord_bits=coord_bits, tune_iters=0)
            sqnr0 = 10 * math.log10(signal_power / max(result0['quant_mse'], 1e-30))
            codes = result0['codes'].float()
            clamp_frac = ((codes == lo) | (codes == hi)).float().mean().item() * 100
            print(f"  {f'{lname} LDLQ global ({coord_bits}b, t=0)':<40} {result0['bpw']:>6.2f} "
                  f"{result0['quant_mse']:>11.2e} {sqnr0:>9.2f} {result0['proxy_loss']:>12.4e} {clamp_frac:>6.1f}%")

            # Verify dequantize_global roundtrip (once per lattice)
            if coord_bits == 3:
                W_hat_rt = quant.dequantize_global(result0)
                rt_err = ((result0['W_hat'] - W_hat_rt) ** 2).mean().item()
                if rt_err > 1e-10:
                    print(f"  WARNING: {lname} dequantize_global roundtrip error = {rt_err:.2e}")

            # (c) LDLQ with refinement
            if tune_iters > 0:
                result_t = quant.quantize_ldlq(W, H, coord_bits=coord_bits, tune_iters=tune_iters)
                sqnr_t = 10 * math.log10(signal_power / max(result_t['quant_mse'], 1e-30))
                codes_t = result_t['codes'].float()
                clamp_frac_t = ((codes_t == lo) | (codes_t == hi)).float().mean().item() * 100
                print(f"  {f'{lname} LDLQ global ({coord_bits}b, t={tune_iters})':<40} {result_t['bpw']:>6.2f} "
                      f"{result_t['quant_mse']:>11.2e} {sqnr_t:>9.2f} {result_t['proxy_loss']:>12.4e} {clamp_frac_t:>6.1f}%")

        print()  # blank line between lattices

    # Scalar baselines
    for bits in [1, 2, 3, 4, 5, 6, 7]:
        w_hat_s = scalar_quantize(W.reshape(-1), bits).reshape(m, n)
        mse_s = ((W - w_hat_s) ** 2).mean().item()
        sqnr_s = 10 * math.log10(signal_power / max(mse_s, 1e-30))
        proxy_s = compute_proxy_loss(W, w_hat_s, H)
        print(f"  {f'Scalar {bits}b (per-tensor)':<40} {bits:>6.2f} "
              f"{mse_s:>11.2e} {sqnr_s:>9.2f} {proxy_s:>12.4e}")

    for bits, gs in [(1, 32), (2, 32), (3, 32), (4, 32), (5, 32), (6, 32), (7, 32)]:
        w_hat_sg = scalar_quantize_grouped(W.reshape(-1), bits, group_size=gs).reshape(m, n)
        mse_sg = ((W - w_hat_sg) ** 2).mean().item()
        sqnr_sg = 10 * math.log10(signal_power / max(mse_sg, 1e-30))
        proxy_sg = compute_proxy_loss(W, w_hat_sg, H)
        eff_bpw = bits + 32 / gs
        print(f"  {f'Scalar grp={gs} ({bits}b)':<40} {eff_bpw:>6.2f} "
              f"{mse_sg:>11.2e} {sqnr_sg:>9.2f} {proxy_sg:>12.4e}")

    # BW16 Shell Codebook
    print()
    shell_cb = BW16ShellCodebook(device=device, verbose=False)
    for ti in [0, tune_iters] if tune_iters > 0 else [0]:
        result_cb = quantize_ldlq_codebook(W, H, shell_cb, tune_iters=ti)
        sqnr_cb = 10 * math.log10(signal_power / max(result_cb['quant_mse'], 1e-30))
        print(f"  {f'BW16 Shell LDLQ (t={ti})':<40} {result_cb['bpw']:>6.2f} "
              f"{result_cb['quant_mse']:>11.2e} {sqnr_cb:>9.2f} "
              f"{result_cb['proxy_loss']:>12.4e}")

    w_hat_cb, _ = quantize_pergroup_codebook(W.reshape(-1), shell_cb)
    W_hat_cb = w_hat_cb.reshape(m, n)
    mse_pcb = ((W - W_hat_cb) ** 2).mean().item()
    sqnr_pcb = 10 * math.log10(signal_power / max(mse_pcb, 1e-30))
    proxy_pcb = compute_proxy_loss(W, W_hat_cb, H)
    bpw_pcb = 1.0 + 32.0 / 16
    print(f"  {'BW16 Shell per-grp':<40} {bpw_pcb:>6.2f} "
          f"{mse_pcb:>11.2e} {sqnr_pcb:>9.2f} {proxy_pcb:>12.4e}")

    print()


def benchmark_ldlq_with_rht(m: int = 192, n: int = 192,
                             tune_iters: int = 1, seed: int = 42):
    """
    Full pipeline: RHT (incoherence) + LDLQ (Hessian-aware quantization).
    Tests E₈, BW₁₆, and Leech with outlier weights to demonstrate RHT's effect.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)

    assert n % 48 == 0, f"n={n} must be divisible by 48"

    # Raw weights with outliers
    W_raw = torch.randn(m, n, device=device) * 0.02
    n_outliers = (m * n) // 100
    outlier_idx = torch.randperm(m * n, device=device)[:n_outliers]
    W_raw.reshape(-1)[outlier_idx] *= 10.0

    H_raw = simulate_hessian(n, n_samples=max(n, 256), device=device, seed=seed + 1)

    # Apply RHT to both W and H
    rht = RHT(m, n, device=device, seed=seed + 2)
    W_rht = rht.transform_weights(W_raw.float())
    H_rht = rht.transform_hessian(H_raw.float())

    # Pad RHT output to next multiple of 48 (lcm of 8,16,24)
    m_pad, n_pad = W_rht.shape
    n_ldlq = ((n_pad + 47) // 48) * 48
    if n_ldlq != n_pad:
        W_ldlq = torch.zeros(m_pad, n_ldlq, device=device)
        W_ldlq[:, :n_pad] = W_rht
        H_ldlq = torch.eye(n_ldlq, device=device) * 1e-4
        H_ldlq[:n_pad, :n_pad] = H_rht
    else:
        W_ldlq = W_rht
        H_ldlq = H_rht

    signal_power_rht = (W_ldlq ** 2).mean().item()

    lattices = [
        ('E8', E8Quantizer(device)),
        ('BW16', BW16Quantizer(device)),
        ('Leech', LeechQuantizer(device)),
    ]

    def compute_proxy_loss(W_orig, W_hat, H):
        diff = (W_orig - W_hat).float()
        return (diff @ H.float() @ diff.T).diagonal().mean().item()

    print(f"RHT + LDLQ Benchmark — W_raw: ({m}, {n}), padded: ({m_pad}, {n_ldlq})")
    print(f"  raw kurtosis = {((W_raw**4).mean() / (W_raw**2).mean()**2).item():.1f}")
    print(f"  post-RHT kurtosis = {((W_ldlq**4).mean() / max((W_ldlq**2).mean()**2, 1e-30)).item():.1f}")
    print()

    header = f"  {'Method':<40} {'bpw':>6} {'MSE':>11} {'SQNR dB':>9} {'Proxy Loss':>12} {'Clamp%':>7}"
    print(header)
    print("  " + "=" * (len(header) - 2))

    for lname, quant in lattices:
        for coord_bits in [1, 2, 3, 4, 5, 6, 7]:
            lo = -(2 ** (coord_bits - 1))
            hi = 2 ** (coord_bits - 1) - 1

            # (a) RHT + Independent Babai (per-group scale)
            w_flat = W_ldlq.reshape(-1)
            qd = quant.quantize(w_flat, coord_bits=coord_bits)
            w_hat_flat = quant.dequantize(qd)
            W_hat_indep = w_hat_flat.reshape(m_pad, n_ldlq)
            mse_indep = ((W_ldlq - W_hat_indep) ** 2).mean().item()
            sqnr_indep = 10 * math.log10(signal_power_rht / max(mse_indep, 1e-30))
            proxy_indep = compute_proxy_loss(W_ldlq, W_hat_indep, H_ldlq)
            bpw_indep = quant.bits_per_weight(qd)
            print(f"  {f'RHT+{lname} Babai ({coord_bits}b)':<40} {bpw_indep:>6.2f} "
                  f"{mse_indep:>11.2e} {sqnr_indep:>9.2f} {proxy_indep:>12.4e} {'':>7}")

            # (b) RHT + LDLQ global
            result0 = quant.quantize_ldlq(W_ldlq, H_ldlq, coord_bits=coord_bits, tune_iters=0)
            sqnr0 = 10 * math.log10(signal_power_rht / max(result0['quant_mse'], 1e-30))
            codes0 = result0['codes'].float()
            clamp0 = ((codes0 == lo) | (codes0 == hi)).float().mean().item() * 100
            print(f"  {f'RHT+{lname} LDLQ ({coord_bits}b, t=0)':<40} {result0['bpw']:>6.2f} "
                  f"{result0['quant_mse']:>11.2e} {sqnr0:>9.2f} {result0['proxy_loss']:>12.4e} {clamp0:>6.1f}%")

            if tune_iters > 0:
                result_t = quant.quantize_ldlq(W_ldlq, H_ldlq, coord_bits=coord_bits, tune_iters=tune_iters)
                sqnr_t = 10 * math.log10(signal_power_rht / max(result_t['quant_mse'], 1e-30))
                codes_t = result_t['codes'].float()
                clamp_t = ((codes_t == lo) | (codes_t == hi)).float().mean().item() * 100
                print(f"  {f'RHT+{lname} LDLQ ({coord_bits}b, t={tune_iters})':<40} {result_t['bpw']:>6.2f} "
                      f"{result_t['quant_mse']:>11.2e} {sqnr_t:>9.2f} {result_t['proxy_loss']:>12.4e} {clamp_t:>6.1f}%")

        print()

    # Scalar baselines
    for bits, gs in [(1, 32), (2, 32), (3, 32), (4, 32), (5, 32), (6, 32), (7, 32)]:
        w_hat_sg = scalar_quantize_grouped(W_ldlq.reshape(-1), bits, group_size=gs).reshape(m_pad, n_ldlq)
        mse_sg = ((W_ldlq - w_hat_sg) ** 2).mean().item()
        sqnr_sg = 10 * math.log10(signal_power_rht / max(mse_sg, 1e-30))
        proxy_sg = compute_proxy_loss(W_ldlq, w_hat_sg, H_ldlq)
        eff_bpw = bits + 32 / gs
        print(f"  {f'Scalar grp={gs} ({bits}b)':<40} {eff_bpw:>6.2f} "
              f"{mse_sg:>11.2e} {sqnr_sg:>9.2f} {proxy_sg:>12.4e}")

    print()


def benchmark_fused_gemv(m: int = 4096, k: int = 4096, n_iters: int = 200):
    """
    Simulate fused dequant+GEMV to measure realistic inference throughput.
    Only runs on CUDA.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print("Fused GEMV benchmark requires CUDA, skipping.\n")
        return

    # k must be divisible by lcm(8,16,24) = 48
    assert k % 48 == 0, "K must be divisible by 48"

    lattices = [
        ('E8', e8_basis, 8),
        ('BW16', bw16_basis, 16),
        ('Leech', leech_basis, 24),
    ]

    x = torch.randn(k, device=device).half()

    # FP16 baseline (shared across lattices)
    w_fp16 = torch.randn(m, k, device=device).half()
    for _ in range(20):
        _ = w_fp16 @ x
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = w_fp16 @ x
    torch.cuda.synchronize()
    t_fp16 = (time.perf_counter() - t0) / n_iters

    print("Fused Dequant+GEMV Benchmark (PyTorch, simulated)")
    print("-" * 65)
    print(f"  Matrix: {m} × {k}  ({m*k:,} weights)")
    print(f"  FP16 GEMV: {t_fp16 * 1000:.3f} ms")

    for lname, basis_fn, dim in lattices:
        groups_per_row = k // dim
        codes = torch.randint(-4, 4, (m, k), device=device, dtype=torch.int8)
        scales = torch.randn(m, groups_per_row, device=device).half().abs() * 0.01
        offsets = torch.randn(m, groups_per_row, device=device).half() * 0.001
        G, _ = basis_fn()
        G = G.half().to(device)

        def fused_dequant_gemv(codes=codes, scales=scales, offsets=offsets, G=G,
                                groups_per_row=groups_per_row, dim=dim):
            c = codes.reshape(m, groups_per_row, dim).float()
            w = c @ G.float().T
            w = w * scales.float().unsqueeze(2) + offsets.float().unsqueeze(2)
            return (w.reshape(m, k).half() @ x)

        # Warmup
        for _ in range(20):
            _ = fused_dequant_gemv()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = fused_dequant_gemv()
        torch.cuda.synchronize()
        t_avg = (time.perf_counter() - t0) / n_iters

        overhead = t_avg / t_fp16
        print(f"  [{lname:>5}] {t_avg*1000:.3f}ms  "
              f"({m*k/t_avg/1e9:.2f} Gw/s)  overhead={overhead:.1f}×")

    print(f"  Note: fused CUDA target <2×")
    print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Lattice quantization prototype (E8/BW16/Leech)")
    parser.add_argument('--no-rht', action='store_true',
                        help='Skip RHT benchmarks')
    parser.add_argument('--skip-throughput', action='store_true',
                        help='Skip throughput benchmarks')
    parser.add_argument('--skip-ldlq', action='store_true',
                        help='Skip LDLQ benchmarks')
    parser.add_argument('--layer', type=str, default=None,
                        help='HuggingFace model to test (e.g. meta-llama/Llama-2-7b-hf)')
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Lattice Quantization Prototype v2                      ║")
    print("║   E₈ (8d) · BW₁₆ (16d) · Λ₂₄ (24d)                     ║")
    print("║   Babai nearest-plane · LDLQ · RHT                       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # 1. Verify lattice constructions
    for name, basis_fn in [('E8', e8_basis), ('BW16', bw16_basis), ('Leech', leech_basis)]:
        verify_lattice(basis_fn=basis_fn, name=name)

    # 2. Quality benchmarks
    benchmark_quality(use_rht=True)
    benchmark_quality(use_rht=False)

    # 3. RHT effect comparison
    if not args.no_rht:
        benchmark_rht_effect()

    # 4. LDLQ benchmarks (Hessian-aware quantization)
    if not args.skip_ldlq:
        benchmark_ldlq()
        benchmark_ldlq_with_rht()

    # 5. Throughput
    if not args.skip_throughput:
        benchmark_throughput()
        benchmark_fused_gemv()


if __name__ == '__main__':
    main()
