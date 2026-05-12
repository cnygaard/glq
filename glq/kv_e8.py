"""E8-lattice KV-cache quantizer (strict or relaxed parity).

Per-group pipeline (matches the math in ``infra/kv-cache.md``):
  1. Reshape last dim into groups of 8 (the E8 codebook dimension)
  2. Apply normalized Walsh-Hadamard inside each group (sub-Gaussian-ify)
  3. Per-group NSN scale: divide by max|x| so each group lies in unit ball
  4. Nearest E8 codepoint lookup (16-bit index per group = 2 bpw/coord)
  5. Optional RVQ: second-stage codebook on residual for 4 bpw/coord

Tensor shapes throughout (HF-cache convention):
    [batch, n_heads, seq, head_dim]
where head_dim is divisible by 8.

The encoded representation is a dict so it survives ``QuantizedLayer``'s
opaque-handle convention. Decode reverses the pipeline using ``H.T``.
"""

import math
from typing import Any

import torch


def _hadamard_8(dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Normalized 8×8 Walsh-Hadamard matrix (H @ H.T = I)."""
    H1 = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    H2 = torch.kron(H1, H1)        # 4x4
    H3 = torch.kron(H2, H1)        # 8x8
    return (H3 / math.sqrt(8.0)).to(dtype=dtype, device=device)


class E8KVQuantizer:
    """Per-group E8 KV quantizer for one cache layer.

    Operates on the last dimension of the input tensor (head_dim). Stores
    encoded results as dicts of int16 indices + fp16 scale tensors.

    Args:
        codebook: An ``E8ShellCodebook`` (strict) or ``E8RelaxedCodebook``
            instance, used for stage-1 nearest-neighbor lookup.
        n_stages: 1 = 2 bpw (single codebook lookup), 2 = 4 bpw (RVQ).
        group_size: Must be 8 (the E8 dimension).
    """

    GROUP_SIZE = 8

    def __init__(self, codebook, n_stages: int = 1, dtype=torch.float16):
        if codebook.codesz != self.GROUP_SIZE:
            raise ValueError(
                f"E8KVQuantizer requires an 8-dimensional codebook; got "
                f"codesz={codebook.codesz}")
        if n_stages not in (1, 2, 3):
            raise ValueError(f"n_stages must be 1, 2, or 3, got {n_stages}")
        self.codebook = codebook
        self.n_stages = n_stages
        self.dtype = dtype
        # Hadamard built lazily once we see the runtime device.
        self._H = None

    def _hadamard_on(self, device, dtype):
        if (self._H is None
                or self._H.device != device
                or self._H.dtype != dtype):
            self._H = _hadamard_8(dtype=dtype, device=device)
        return self._H

    # ------------------------------------------------------------------ #
    # Encode
    # ------------------------------------------------------------------ #

    def quantize(self, x: torch.Tensor) -> dict[str, Any]:
        """Encode a KV tensor.

        Args:
            x: [..., head_dim] tensor with head_dim % 8 == 0.

        Returns:
            dict with keys:
              - 'idx1'   : int16 [N, G]
              - 'idx2'   : int16 [N, G]      (only if n_stages == 2)
              - 'scale'  : fp16  [..., G]
              - 'shape'  : original tensor shape (for decode)
              - 'dtype'  : original dtype (decode returns same dtype)
        """
        orig_shape = tuple(x.shape)
        orig_dtype = x.dtype
        head_dim = orig_shape[-1]
        if head_dim % self.GROUP_SIZE != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be a multiple of "
                f"{self.GROUP_SIZE}")
        n_groups = head_dim // self.GROUP_SIZE

        # Flatten everything except the last dim, then split into groups of 8.
        x = x.reshape(-1, n_groups, self.GROUP_SIZE).contiguous()
        N = x.shape[0]

        # Hadamard rotate inside each group of 8.
        H = self._hadamard_on(x.device, x.dtype)
        x_rot = x @ H   # [N, G, 8]

        # Per-group NSN scale (so max|coord| = 1 within each group).
        scale = x_rot.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        x_norm = x_rot / scale  # [N, G, 8]

        # Codebook NN over (N*G) 8-vectors.
        flat = x_norm.reshape(-1, self.GROUP_SIZE)
        if self.n_stages == 1:
            _, idx1 = self.codebook.quantize(flat)
            out = {
                "idx1": idx1.reshape(N, n_groups).to(torch.int16),
            }
        elif self.n_stages == 2:
            _, (idx1, idx2) = self.codebook.quantize_rvq(flat)
            out = {
                "idx1": idx1.reshape(N, n_groups).to(torch.int16),
                "idx2": idx2.reshape(N, n_groups).to(torch.int16),
            }
        else:  # n_stages == 3 → 6 bpw
            rs = self.codebook.resid_scale
            dec1, idx1 = self.codebook.quantize(flat)
            r1 = (flat - dec1) * rs
            dec2, idx2 = self.codebook.quantize(r1)
            r2 = (r1 - dec2) * rs
            _, idx3 = self.codebook.quantize(r2)
            out = {
                "idx1": idx1.reshape(N, n_groups).to(torch.int16),
                "idx2": idx2.reshape(N, n_groups).to(torch.int16),
                "idx3": idx3.reshape(N, n_groups).to(torch.int16),
            }
        # Reshape scale back to original leading shape (drop the last
        # singleton dim from amax keepdim=True).
        scale_shape = orig_shape[:-1] + (n_groups,)
        out["scale"] = scale.reshape(scale_shape).to(self.dtype)
        out["shape"] = orig_shape
        out["dtype"] = orig_dtype
        return out

    # ------------------------------------------------------------------ #
    # Decode
    # ------------------------------------------------------------------ #

    def dequantize(self, qt: dict[str, Any]) -> torch.Tensor:
        """Decode an encoded KV tensor back to its original shape and dtype."""
        idx1 = qt["idx1"].long()             # [N, G]
        scale = qt["scale"]                  # [..., G] fp16
        orig_shape = qt["shape"]
        orig_dtype = qt["dtype"]
        N, n_groups = idx1.shape

        # Stage 1 lookup (and optional stage 2 / stage 3).
        flat1 = idx1.flatten()
        if "idx3" in qt:
            flat2 = qt["idx2"].long().flatten()
            flat3 = qt["idx3"].long().flatten()
            rs = self.codebook.resid_scale
            dec = (
                self.codebook.decode(flat1)
                + self.codebook.decode(flat2) / rs
                + self.codebook.decode(flat3) / (rs * rs))
        elif "idx2" in qt:
            flat2 = qt["idx2"].long().flatten()
            dec = self.codebook.decode_rvq(flat1, flat2)
        else:
            dec = self.codebook.decode(flat1)  # [N*G, 8]

        # Restore per-group shape + scale + inverse Hadamard.
        dec = dec.reshape(N, n_groups, self.GROUP_SIZE)
        H = self._hadamard_on(dec.device, dec.dtype)
        scale_b = scale.reshape(N, n_groups, 1).to(dec.dtype)
        dec = (dec * scale_b) @ H.T   # H is orthogonal → H.T = H⁻¹
        return dec.reshape(orig_shape).to(orig_dtype)
