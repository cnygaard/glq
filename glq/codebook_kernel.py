"""Triton fused codebook nearest-neighbour kernel.

Fuses matmul + argmin to avoid materializing the (batch, K) distance matrix.
For K=65536, D=8: reduces memory traffic by ~60000x vs naive matmul+argmin.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _codebook_nn_kernel(
    x_ptr,          # (N, D) input vectors, fp16
    cb_ptr,         # (K, D) codebook vectors, fp16
    cb_norms_ptr,   # (K,) precomputed ||c||^2, fp16
    idx_ptr,        # (N,) output indices, int32
    N,              # number of input vectors
    D: tl.constexpr,              # dimension (8)
    K,              # codebook size (65536)
    stride_x_n,     # x stride along N dim
    stride_cb_k,    # codebook stride along K dim
    BLOCK_K: tl.constexpr = 256,  # tile size over codebook
):
    """Find nearest codebook vector for each input row.

    dist(x, c) = ||x||^2 - 2*x.c + ||c||^2
    Since ||x||^2 is constant per row, argmin(dist) = argmin(-2*x.c + ||c||^2).
    """
    pid = tl.program_id(0)
    row = pid
    if row >= N:
        return

    # Load x[row, :] — D=8 fits entirely in registers
    x_offsets = tl.arange(0, D)
    x_vec = tl.load(x_ptr + row * stride_x_n + x_offsets).to(tl.float32)

    best_dist = float('inf')
    best_idx = 0

    # Tile over codebook
    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        mask = k_range < K

        # Load codebook block: (BLOCK_K, D)
        # cb_ptr[k, d] = cb_ptr + k * stride_cb_k + d
        cb_block = tl.load(
            cb_ptr + k_range[:, None] * stride_cb_k + x_offsets[None, :],
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)

        # Load precomputed norms
        cb_norms = tl.load(cb_norms_ptr + k_range, mask=mask, other=float('inf')).to(tl.float32)

        # Dot products: (BLOCK_K,)
        dots = tl.sum(cb_block * x_vec[None, :], axis=1)

        # Distance: -2*dot + ||c||^2 (skip ||x||^2, constant)
        dists = -2.0 * dots + cb_norms

        # Running argmin
        local_min_val = tl.min(dists)
        if local_min_val < best_dist:
            local_min_idx = tl.argmin(dists, axis=0)
            best_dist = local_min_val
            best_idx = k_start + local_min_idx

    tl.store(idx_ptr + row, best_idx)


def triton_codebook_nn(x: torch.Tensor, codebook: torch.Tensor,
                       codebook_norms: torch.Tensor) -> torch.Tensor:
    """Find nearest codebook vector for each row of x.

    Args:
        x: (N, D) input vectors (any dtype, converted to fp16)
        codebook: (K, D) codebook vectors (any dtype, converted to fp16)
        codebook_norms: (K,) precomputed ||c||^2 (any dtype, converted to fp16)

    Returns:
        indices: (N,) int32 indices into codebook
    """
    N, D = x.shape
    K = codebook.shape[0]
    assert codebook.shape[1] == D
    assert codebook_norms.shape[0] == K

    x_half = x.half().contiguous()
    cb_half = codebook.half().contiguous()
    cb_norms_half = codebook_norms.half().contiguous()

    indices = torch.empty(N, dtype=torch.int32, device=x.device)

    grid = (N,)
    _codebook_nn_kernel[grid](
        x_half, cb_half, cb_norms_half, indices,
        N, D, K,
        x_half.stride(0), cb_half.stride(0),
    )

    return indices.long()
