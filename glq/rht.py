"""Randomized Hadamard Transform (RHT) for incoherence processing."""

import torch
from .hadamard import fast_hadamard_transform, block_diagonal_fht, _block_decompose


def _next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 0 else 1


class RHT:
    """
    Randomized Hadamard Transform for incoherence processing.

    Transforms weight matrix W and its proxy Hessian H to suppress outliers.
    Identical to QuIP#'s IP-RHT (Algorithm 3).

    When ``block_diagonal=True`` (the default for new quantizations),
    decomposes non-power-of-2 dimensions into a sum of powers of 2 and
    applies independent FHTs to each block.  This avoids padding to the
    next power of 2, saving 30-50% storage on models with non-power-of-2
    hidden sizes (e.g. 2688 → [2048, 512, 128] instead of 4096).

    When ``block_diagonal=False``, pads to the next power of 2 (legacy).

    Usage during quantization:
        rht = RHT(m, n, device)
        W_tilde = rht.transform_weights(W)
        H_tilde = rht.transform_hessian(H)
        # ... quantize W_tilde with H_tilde ...
        W_hat = rht.inverse_transform_weights(W_hat_tilde)

    During inference (E8RHTLinear.forward):
        x_hat = rht.transform_input(x)
        y_tilde = W_quant @ x_hat
        y = rht.inverse_transform_output(y_tilde)
    """

    def __init__(self, m: int, n: int, device='cpu', seed=42,
                 block_diagonal: bool = True):
        self.m_orig, self.n_orig = m, n
        self.device = device
        self.block_diagonal = block_diagonal

        if block_diagonal:
            self.blocks_m = _block_decompose(m)
            self.blocks_n = _block_decompose(n)
            self.m_pad = sum(self.blocks_m)  # = m (no padding)
            self.n_pad = sum(self.blocks_n)  # = n (no padding)
        else:
            self.blocks_m = [_next_pow2(m)]
            self.blocks_n = [_next_pow2(n)]
            self.m_pad = self.blocks_m[0]
            self.n_pad = self.blocks_n[0]

        gen = torch.Generator(device='cpu').manual_seed(seed)
        self.su = (torch.randint(0, 2, (self.m_pad,), generator=gen).float() * 2 - 1).to(device)
        self.sv = (torch.randint(0, 2, (self.n_pad,), generator=gen).float() * 2 - 1).to(device)

    def _fht_cols(self, X: torch.Tensor) -> torch.Tensor:
        """FHT along columns (last dim) using block structure."""
        if len(self.blocks_n) == 1 and self.blocks_n[0] == X.shape[-1]:
            return fast_hadamard_transform(X)
        return block_diagonal_fht(X, self.blocks_n)

    def _fht_rows(self, X: torch.Tensor) -> torch.Tensor:
        """FHT along rows: transpose, apply column FHT with blocks_m, transpose back."""
        if len(self.blocks_m) == 1 and self.blocks_m[0] == X.shape[0]:
            return fast_hadamard_transform(X.T.contiguous()).T
        return block_diagonal_fht(X.T.contiguous(), self.blocks_m).T

    def transform_weights(self, W: torch.Tensor) -> torch.Tensor:
        """Apply RHT to weight matrix. Returns (m_pad, n_pad)."""
        m, n = W.shape
        dtype = W.dtype

        W_padded = torch.zeros(self.m_pad, self.n_pad, device=self.device, dtype=dtype)
        W_padded[:m, :n] = W.to(self.device)

        W_t = W_padded * self.sv.unsqueeze(0).to(dtype)
        W_t = self._fht_cols(W_t)

        W_t = W_t.T.contiguous()
        W_t = W_t * self.su.unsqueeze(0).to(dtype)
        W_t = self._fht_rows(W_t.T.contiguous())

        return W_t

    def transform_hessian(self, H: torch.Tensor) -> torch.Tensor:
        """Apply RHT to Hessian: H_r = Had @ diag(sv) @ H @ diag(sv) @ Had."""
        n = H.shape[0]
        dtype = H.dtype

        H_pad = torch.zeros(self.n_pad, self.n_pad, device=self.device, dtype=dtype)
        H_pad[:n, :n] = H.to(self.device)

        sv = self.sv.to(dtype)
        H_pad = H_pad * sv.unsqueeze(0) * sv.unsqueeze(1)

        H_pad = self._fht_cols(H_pad)
        H_pad = self._fht_cols(H_pad.T.contiguous()).T

        return H_pad

    def inverse_transform_weights(self, W_tilde: torch.Tensor) -> torch.Tensor:
        """Inverse RHT. Returns (m_orig, n_orig)."""
        dtype = W_tilde.dtype

        W_t = self._fht_cols(W_tilde.clone())
        W_t = W_t * self.sv.unsqueeze(0).to(dtype)

        W_t = self._fht_rows(W_t)

        W_t = W_t * self.su.unsqueeze(1).to(dtype)

        return W_t[:self.m_orig, :self.n_orig]

    def transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input activation: x_hat = Had(SV * pad(x))."""
        x_padded = torch.zeros(*x.shape[:-1], self.n_pad, device=self.device, dtype=x.dtype)
        x_padded[..., :x.shape[-1]] = x.to(self.device)
        x_s = x_padded * self.sv.to(x.dtype)
        return self._fht_cols(x_s)

    def inverse_transform_output(self, y: torch.Tensor) -> torch.Tensor:
        """Undo row transform: y_out = Had(y) * SU, truncated to m_orig."""
        out = block_diagonal_fht(y.clone(), self.blocks_m) * self.su.to(y.dtype)
        return out[..., :self.m_orig]
