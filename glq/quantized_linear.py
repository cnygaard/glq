"""E8RHTLinear — quantized linear layer with E8 shell codebook + RHT."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .hadamard import fast_hadamard_transform


class E8RHTLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with E8+RHT quantized weights.

    Stores quantized weights as codebook indices + RHT sign vectors.
    On forward, dequantizes in the RHT domain and uses Hadamard transforms
    on input/output to avoid full weight materialization.

    Buffers (loaded from safetensors):
        Qidxs: (m_pad, n_pad // 8) int16 — codebook indices
        SU: (m_pad,) float16 — row sign vector (±1)
        SV: (n_pad,) float16 — column sign vector (±1)
        Wscale: () float32 — global scale factor

    Shared state (set after loading):
        codebook: E8ShellCodebook instance
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Padded dimensions (power of 2)
        self.m_pad = 1 << (out_features - 1).bit_length() if out_features > 0 else 1
        self.n_pad = 1 << (in_features - 1).bit_length() if in_features > 0 else 1

        # Quantized weight storage
        self.register_buffer('Qidxs', torch.zeros(self.m_pad, self.n_pad // 8, dtype=torch.int16))
        self.register_buffer('SU', torch.ones(self.m_pad, dtype=torch.float16))
        self.register_buffer('SV', torch.ones(self.n_pad, dtype=torch.float16))
        self.register_buffer('Wscale', torch.ones((), dtype=torch.float32))

        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

        # Codebook reference — set after loading via set_codebook()
        self.codebook = None

    def set_codebook(self, codebook):
        """Attach the shared E8ShellCodebook (called after weight loading)."""
        self.codebook = codebook

    def _ensure_codebook_device(self):
        """Move codebook to match Qidxs device (lazy, once)."""
        if self.codebook is not None and self.codebook.codebook.device != self.Qidxs.device:
            self.codebook = self.codebook.to(self.Qidxs.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with RHT-domain matmul.

        1. Pad input, apply SV signs, FHT → transform to RHT domain
        2. Decode Qidxs → W_rht (m_pad, n_pad) in codebook scale
        3. y_rht = x_rht @ W_rht.T * Wscale
        4. FHT on y_rht, apply SU signs, unpad → inverse RHT on output
        """
        self._ensure_codebook_device()

        shape = x.shape
        x = x.reshape(-1, self.in_features)
        batch = x.shape[0]
        dtype = x.dtype

        # Transform input to RHT domain
        x_f = x.float()
        x_pad = F.pad(x_f, (0, self.n_pad - self.in_features))
        sv = self.SV.float()
        x_rht = fast_hadamard_transform(x_pad * sv.unsqueeze(0))

        # Decode weights in RHT domain
        W_rht = self.codebook.decode(self.Qidxs.long().reshape(-1))
        W_rht = W_rht.reshape(self.m_pad, self.n_pad)

        # Matmul in RHT domain
        y_rht = x_rht @ W_rht.T * self.Wscale.float()

        # Inverse RHT on output: y = y_rht @ Had_m @ diag(SU)
        su = self.SU.float()
        y = fast_hadamard_transform(y_rht) * su.unsqueeze(0)
        y = y[:, :self.out_features]

        if self.bias is not None:
            y = y + self.bias.float().unsqueeze(0)

        y = y.to(dtype)
        return y.reshape(*shape[:-1], self.out_features)

    def dequantize(self) -> torch.Tensor:
        """
        Full weight dequantization for debugging/validation.
        Returns (out_features, in_features) dense weight matrix.
        """
        self._ensure_codebook_device()
        W_rht = self.codebook.decode(self.Qidxs.long().reshape(-1))
        W_rht = W_rht.reshape(self.m_pad, self.n_pad) * self.Wscale.float()

        # Inverse RHT
        sv = self.SV.float()
        su = self.SU.float()

        W_t = fast_hadamard_transform(W_rht.clone())
        W_t = W_t * sv.unsqueeze(0)

        W_t = W_t.T
        W_t = fast_hadamard_transform(W_t.clone())
        W_t = W_t.T

        W_t = W_t * su.unsqueeze(1)

        return W_t[:self.out_features, :self.in_features]

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, m_pad={self.m_pad}, n_pad={self.n_pad}')
