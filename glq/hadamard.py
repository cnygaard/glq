"""Fast Walsh-Hadamard Transform.

Priority: CUDA extension > Triton fused kernel (small batch) > PyTorch fallback.
"""

import math
import torch


def _pytorch_fht(x: torch.Tensor) -> torch.Tensor:
    """
    Pure-PyTorch Fast Walsh-Hadamard transform (fallback).
    O(n log n) but launches many small CUDA kernels — slow for B=1.
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


# ---------- Triton FHT kernel ----------

_triton_fht_available = False

# Max rows for fused kernel (debug_barrier correctness limit).
# Above this, fall back to PyTorch.
_FUSED_MAX_ROWS = 64

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _fht_fused_kernel(
        IO_ptr,
        stride_row,
        N: tl.constexpr,
        LOG_N: tl.constexpr,
    ):
        """All butterfly stages of the FHT in a single kernel.

        Correct for small grids (nrows <= 64). Uses debug_barrier for
        intra-program synchronization between butterfly stages.
        """
        row = tl.program_id(0)
        offs = tl.arange(0, N)
        base = IO_ptr + row * stride_row

        x = tl.load(base + offs)

        for k in tl.static_range(LOG_N):
            # Write current state so all threads can read partners
            tl.store(base + offs, x)
            tl.debug_barrier()
            # Read partner value (partner index = offs XOR 2^k)
            x_partner = tl.load(base + (offs ^ (1 << k)))
            # Butterfly: low half gets sum, high half gets difference
            lo = (offs & (1 << k)) == 0
            x = tl.where(lo, x + x_partner, x_partner - x)

        tl.store(base + offs, x)

    def _triton_fht(x: torch.Tensor) -> torch.Tensor:
        """Triton-accelerated FHT for small batch sizes (inference)."""
        n = x.shape[-1]
        assert n > 0 and (n & (n - 1)) == 0, f"Last dim {n} must be power of 2"

        needs_batch = x.dim() == 1
        if needs_batch:
            x = x.unsqueeze(0)

        flat = x.reshape(-1, n).contiguous()
        nrows = flat.shape[0]
        log_n = int(math.log2(n))

        _fht_fused_kernel[(nrows,)](flat, n, n, log_n)

        result = flat.reshape(x.shape) / math.sqrt(n)
        if needs_batch:
            result = result.squeeze(0)
        return result

    _triton_fht_available = True

except ImportError:
    pass


# ---------- Pick best available implementation ----------

try:
    from fast_hadamard_transform import hadamard_transform as _fht_cuda

    def fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
        """CUDA-accelerated Fast Walsh-Hadamard transform (normalized)."""
        n = x.shape[-1]
        assert n > 0 and (n & (n - 1)) == 0, f"Last dim {n} must be power of 2"
        needs_batch = x.dim() == 1
        if needs_batch:
            x = x.unsqueeze(0)
        out = _fht_cuda(x) / math.sqrt(n)
        if needs_batch:
            out = out.squeeze(0)
        return out

except ImportError:
    if _triton_fht_available:
        def fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
            """FHT with Triton for small batch, PyTorch for large batch."""
            if x.is_cuda:
                nrows = x.reshape(-1, x.shape[-1]).shape[0]
                if nrows <= _FUSED_MAX_ROWS:
                    return _triton_fht(x)
            return _pytorch_fht(x)
    else:
        fast_hadamard_transform = _pytorch_fht
