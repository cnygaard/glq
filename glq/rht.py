"""Randomized Hadamard Transform (RHT) for incoherence processing."""

import torch
from .hadamard import fast_hadamard_transform


class RHT:
    """
    Randomized Hadamard Transform for incoherence processing.

    Transforms weight matrix W and its proxy Hessian H to suppress outliers.
    Identical to QuIP#'s IP-RHT (Algorithm 3).

    For dimensions not a power of 2, pads to next power of 2.

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

    def __init__(self, m: int, n: int, device='cpu', seed=42):
        self.m_orig, self.n_orig = m, n
        self.device = device

        self.m_pad = 1 << (m - 1).bit_length() if m > 0 else 1
        self.n_pad = 1 << (n - 1).bit_length() if n > 0 else 1

        gen = torch.Generator(device='cpu').manual_seed(seed)
        self.su = (torch.randint(0, 2, (self.m_pad,), generator=gen).float() * 2 - 1).to(device)
        self.sv = (torch.randint(0, 2, (self.n_pad,), generator=gen).float() * 2 - 1).to(device)

    def transform_weights(self, W: torch.Tensor) -> torch.Tensor:
        """Apply RHT to weight matrix. Returns (m_pad, n_pad)."""
        m, n = W.shape
        dtype = W.dtype

        W_padded = torch.zeros(self.m_pad, self.n_pad, device=self.device, dtype=dtype)
        W_padded[:m, :n] = W.to(self.device)

        W_t = W_padded * self.sv.unsqueeze(0).to(dtype)
        W_t = fast_hadamard_transform(W_t)

        W_t = W_t.T
        W_t = W_t * self.su.unsqueeze(0).to(dtype)
        W_t = fast_hadamard_transform(W_t)
        W_t = W_t.T

        return W_t

    def transform_hessian(self, H: torch.Tensor) -> torch.Tensor:
        """Apply RHT to Hessian: H_r = Had @ diag(sv) @ H @ diag(sv) @ Had."""
        n = H.shape[0]
        dtype = H.dtype

        H_pad = torch.zeros(self.n_pad, self.n_pad, device=self.device, dtype=dtype)
        H_pad[:n, :n] = H.to(self.device)

        sv = self.sv.to(dtype)
        H_pad = H_pad * sv.unsqueeze(0) * sv.unsqueeze(1)

        H_pad = fast_hadamard_transform(H_pad)
        H_pad = fast_hadamard_transform(H_pad.T).T

        return H_pad

    def inverse_transform_weights(self, W_tilde: torch.Tensor) -> torch.Tensor:
        """Inverse RHT. Returns (m_orig, n_orig)."""
        dtype = W_tilde.dtype

        W_t = fast_hadamard_transform(W_tilde.clone())
        W_t = W_t * self.sv.unsqueeze(0).to(dtype)

        W_t = W_t.T
        W_t = fast_hadamard_transform(W_t.clone())
        W_t = W_t.T

        W_t = W_t * self.su.unsqueeze(1).to(dtype)

        return W_t[:self.m_orig, :self.n_orig]

    def transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input activation: x_hat = Had(SV * pad(x))."""
        x_padded = torch.zeros(*x.shape[:-1], self.n_pad, device=self.device, dtype=x.dtype)
        x_padded[..., :x.shape[-1]] = x.to(self.device)
        x_s = x_padded * self.sv.to(x.dtype)
        return fast_hadamard_transform(x_s)

    def inverse_transform_output(self, y: torch.Tensor) -> torch.Tensor:
        """Undo row transform: y_out = Had(y) * SU, truncated to m_orig."""
        out = fast_hadamard_transform(y) * self.su.to(y.dtype)
        return out[..., :self.m_orig]
