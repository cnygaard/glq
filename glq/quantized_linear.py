"""E8RHTLinear — quantized linear layer with E8 shell codebook + RHT."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .hadamard import fast_hadamard_transform

try:
    import triton
    import triton.language as tl
    _triton_available = True
except ImportError:
    _triton_available = False


# ────────────────────────────────────────────────────────────────
# Fused RHT kernels: pad+SV+FHT and FHT+SU+unpad in one launch
# ────────────────────────────────────────────────────────────────

if _triton_available:

    @triton.jit
    def _input_rht_kernel(
        x_ptr,          # (B, in_features) fp16/fp32 input
        sv_ptr,         # (n_pad,) fp16 sign vector
        out_ptr,        # (B, n_pad) fp32 output — x_rht
        in_features,
        stride_x,       # x.stride(0)
        rsqrt_n,        # 1 / sqrt(n_pad)
        N: tl.constexpr,
        LOG_N: tl.constexpr,
    ):
        """Fused pad + SV multiply + FHT in one kernel launch.

        Replaces: x.float() -> F.pad -> * SV -> FHT -> x_rht
        """
        row = tl.program_id(0)
        offs = tl.arange(0, N)
        base = out_ptr + row * N

        # Load x with zero-padding beyond in_features, cast to fp32
        x = tl.load(x_ptr + row * stride_x + offs,
                     mask=offs < in_features, other=0.0).to(tl.float32)

        # Apply SV signs
        sv = tl.load(sv_ptr + offs).to(tl.float32)
        x = x * sv

        # FHT butterfly stages (in-place via global memory + barrier)
        for k in tl.static_range(LOG_N):
            tl.store(base + offs, x)
            tl.debug_barrier()
            x_partner = tl.load(base + (offs ^ (1 << k)))
            lo = (offs & (1 << k)) == 0
            x = tl.where(lo, x + x_partner, x_partner - x)

        # Store normalized result
        tl.store(base + offs, x * rsqrt_n)

    @triton.jit
    def _output_rht_kernel(
        y_rht_ptr,      # (B, m_pad) fp32 input — y_rht from dequant
        su_ptr,         # (m_pad,) fp16 sign vector
        out_ptr,        # (B, out_features) output in target dtype
        out_features,
        stride_y,       # y_rht.stride(0)
        stride_out,     # out.stride(0)
        rsqrt_m,        # 1 / sqrt(m_pad)
        OUTPUT_FP16: tl.constexpr,
        M: tl.constexpr,
        LOG_M: tl.constexpr,
    ):
        """Fused FHT + SU multiply + unpad + cast in one kernel launch.

        Replaces: FHT(y_rht) -> * SU -> [:, :out_features] -> .to(dtype)
        """
        row = tl.program_id(0)
        offs = tl.arange(0, M)
        base = y_rht_ptr + row * stride_y

        # Load y_rht
        x = tl.load(base + offs)

        # FHT butterfly stages
        for k in tl.static_range(LOG_M):
            tl.store(base + offs, x)
            tl.debug_barrier()
            x_partner = tl.load(base + (offs ^ (1 << k)))
            lo = (offs & (1 << k)) == 0
            x = tl.where(lo, x + x_partner, x_partner - x)

        # Normalize, apply SU, and store with unpadding + dtype cast
        su = tl.load(su_ptr + offs).to(tl.float32)
        x = x * rsqrt_m * su
        mask = offs < out_features
        if OUTPUT_FP16:
            tl.store(out_ptr + row * stride_out + offs, x.to(tl.float16), mask=mask)
        else:
            tl.store(out_ptr + row * stride_out + offs, x, mask=mask)


class E8RHTLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with E8+RHT quantized weights.

    Stores quantized weights as codebook indices + RHT sign vectors.
    On forward, dequantizes in the RHT domain and uses Hadamard transforms
    on input/output to avoid full weight materialization.

    Buffers (loaded from safetensors):
        Qidxs: (m_pad, n_pad // 8) int16 — primary codebook indices
        SU: (m_pad,) float16 — row sign vector (±1)
        SV: (n_pad,) float16 — column sign vector (±1)
        Wscale: () float32 — global scale factor
        Qidxs2: (m_pad, n_pad // 8) int16 — secondary indices (3/4bpw), optional
        inv_resid_scale: () float32 — 1/resid_scale (3/4bpw), optional

    Shared state (set after loading):
        codebook: E8ShellCodebook instance (primary)
        codebook2: E8ShellCodebook instance (secondary, for 3/4bpw)
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

        # Two-stage buffers for 3/4bpw (always allocated so state_dict loading works)
        self.register_buffer('Qidxs2', torch.zeros(self.m_pad, self.n_pad // 8, dtype=torch.int16))
        self.register_buffer('inv_resid_scale', torch.zeros((), dtype=torch.float32))
        self._has_stage2 = False  # set True when loaded with actual two-stage data

        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

        # Codebook references — set after loading via set_codebook()
        self.codebook = None
        self.codebook2 = None

        # Cached scalar values (set in set_codebook, avoids GPU→CPU sync per forward)
        self._wscale_float = 1.0
        self._inv_rs_float = 0.0

    @property
    def weight(self):
        """Proxy so code checking weight.device works (e.g. Mamba).

        Returns a zero-element tensor on the same device as the quantized
        weights, without allocating any real memory.
        """
        return torch.empty(0, dtype=torch.float16, device=self.Qidxs.device)

    def set_codebook(self, codebook, codebook2=None):
        """Attach the shared E8ShellCodebook(s) (called after weight loading)."""
        self.codebook = codebook
        self.codebook2 = codebook2
        # Skip scalar caching for meta tensors (offloaded layers);
        # values will be resolved lazily in forward() if needed.
        if self.Wscale.device.type == "meta":
            self._has_stage2 = codebook2 is not None
            self._wscale_float = None
            self._inv_rs_float = None
            return
        # Detect two-stage from inv_resid_scale (non-zero means 3/4bpw data was loaded)
        self._has_stage2 = (
            codebook2 is not None
            and self.inv_resid_scale is not None
            and self.inv_resid_scale.abs().item() > 0
        )
        # Cache scalar values to avoid GPU→CPU sync on every forward pass
        self._wscale_float = self.Wscale.item()
        self._inv_rs_float = self.inv_resid_scale.item() if self._has_stage2 else 0.0

    def _ensure_codebook_device(self):
        """Move codebook(s) to match Qidxs device (lazy, once)."""
        if self.codebook is not None and self.codebook.codebook.device != self.Qidxs.device:
            self.codebook = self.codebook.to(self.Qidxs.device)
        if self.codebook2 is not None and self.codebook2.codebook.device != self.Qidxs.device:
            self.codebook2 = self.codebook2.to(self.Qidxs.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with RHT-domain matmul.

        1. Pad input, apply SV signs, FHT → transform to RHT domain
        2. Fused dequant+matmul in RHT domain (Triton) or decode+matmul (fallback)
        3. FHT on y_rht, apply SU signs, unpad → inverse RHT on output
        """
        self._ensure_codebook_device()
        # Lazy-resolve cached scalars (for layers that were on meta at set_codebook time)
        if self._wscale_float is None:
            self._wscale_float = self.Wscale.item()
            self._has_stage2 = (
                self.codebook2 is not None
                and self.inv_resid_scale is not None
                and self.inv_resid_scale.abs().item() > 0
            )
            self._inv_rs_float = self.inv_resid_scale.item() if self._has_stage2 else 0.0

        shape = x.shape
        x = x.reshape(-1, self.in_features)
        B = x.shape[0]
        dtype = x.dtype
        use_fused = x.is_cuda and _triton_available and B <= 64

        # Transform input to RHT domain
        if use_fused:
            n_pad = self.n_pad
            log_n = int(math.log2(n_pad))
            x_rht = torch.empty(B, n_pad, dtype=torch.float32, device=x.device)
            _input_rht_kernel[(B,)](
                x, self.SV, x_rht,
                self.in_features, x.stride(0),
                1.0 / math.sqrt(n_pad),
                N=n_pad, LOG_N=log_n,
                num_warps=8,
            )
        else:
            x_f = x.float()
            x_pad = F.pad(x_f, (0, self.n_pad - self.in_features))
            sv = self.SV.float()
            x_rht = fast_hadamard_transform(x_pad * sv.unsqueeze(0))

        # Dequant + matmul in RHT domain
        has_stage2 = self._has_stage2
        if x_rht.is_cuda and _triton_available:
            from .inference_kernel import glq_dequant_matmul
            cb2_tensor = self.codebook2.codebook_half if has_stage2 else None
            y_rht = glq_dequant_matmul(
                x_rht, self.Qidxs,
                self.codebook.codebook_half,
                self._wscale_float,
                Qidxs2=self.Qidxs2 if has_stage2 else None,
                codebook2=cb2_tensor,
                inv_resid_scale=self._inv_rs_float,
            )
        else:
            # Fallback: materialize W then matmul
            W_rht = self.codebook.decode(self.Qidxs.long().reshape(-1))
            W_rht = W_rht.reshape(self.m_pad, self.n_pad)
            if has_stage2:
                W_rht2 = self.codebook2.decode(self.Qidxs2.long().reshape(-1))
                W_rht2 = W_rht2.reshape(self.m_pad, self.n_pad)
                W_rht = W_rht + W_rht2 * self.inv_resid_scale.item()
            y_rht = x_rht @ W_rht.T * self.Wscale.float()

        # Inverse RHT on output: y = y_rht @ Had_m @ diag(SU)
        if use_fused:
            m_pad = self.m_pad
            log_m = int(math.log2(m_pad))
            output_fp16 = (dtype == torch.float16)
            y = torch.empty(B, self.out_features, dtype=dtype, device=x.device)
            _output_rht_kernel[(B,)](
                y_rht, self.SU, y,
                self.out_features, y_rht.stride(0), y.stride(0),
                1.0 / math.sqrt(m_pad),
                OUTPUT_FP16=output_fp16,
                M=m_pad, LOG_M=log_m,
                num_warps=8,
            )
            if self.bias is not None:
                y = y + self.bias.unsqueeze(0).to(dtype)
        else:
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
        W_rht = W_rht.reshape(self.m_pad, self.n_pad)
        if self._has_stage2:
            W_rht2 = self.codebook2.decode(self.Qidxs2.long().reshape(-1))
            W_rht2 = W_rht2.reshape(self.m_pad, self.n_pad)
            W_rht = W_rht + W_rht2 * self.inv_resid_scale.item()
        W_rht = W_rht * self.Wscale.float()

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
