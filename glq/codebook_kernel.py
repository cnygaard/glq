"""Triton fused codebook nearest-neighbour kernel.

Fuses matmul + argmin to avoid materializing the (batch, K) distance matrix.
Uses row-tiling (BLOCK_N rows per program) + D→16 zero-padding for Tensor Core.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _codebook_nn_kernel(
    x_ptr,          # (N, D) input vectors, fp16
    cb_ptr,         # (K, D) codebook vectors, fp16
    cb_norms_ptr,   # (K,) precomputed ||c||^2, fp32
    idx_ptr,        # (N,) output indices, int32
    N,              # number of input vectors
    D: tl.constexpr,              # actual dimension (8)
    K,              # codebook size (65536)
    stride_x_n,     # x stride along N dim
    stride_cb_k,    # codebook stride along K dim
    BLOCK_N: tl.constexpr,   # tile size over input rows
    BLOCK_K: tl.constexpr,   # tile size over codebook
    D_PAD: tl.constexpr,     # padded dimension for Tensor Core (16)
):
    """Find nearest codebook vector for a tile of BLOCK_N input rows.

    Tiles BLOCK_N rows per program so the codebook is read once per tile
    instead of once per row, reducing L2 traffic by BLOCK_N×.
    D is zero-padded to D_PAD=16 to enable fp16 Tensor Core mma.m16n8k16.
    """
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    d_offsets = tl.arange(0, D_PAD)
    d_mask = d_offsets < D

    # Load x tile: (BLOCK_N, D_PAD), zero-padded beyond D
    x_tile = tl.load(
        x_ptr + n_offsets[:, None] * stride_x_n + d_offsets[None, :],
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float16)

    best_dist = tl.full((BLOCK_N,), float('inf'), dtype=tl.float32)
    best_idx = tl.zeros((BLOCK_N,), dtype=tl.int32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load codebook tile: (BLOCK_K, D_PAD), zero-padded
        cb_tile = tl.load(
            cb_ptr + k_offsets[:, None] * stride_cb_k + d_offsets[None, :],
            mask=k_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float16)

        # Load precomputed norms: (BLOCK_K,)
        cb_norms = tl.load(
            cb_norms_ptr + k_offsets, mask=k_mask, other=float('inf'),
        ).to(tl.float32)

        # Tensor Core dot: (BLOCK_N, D_PAD) @ (D_PAD, BLOCK_K) -> (BLOCK_N, BLOCK_K)
        dots = tl.dot(x_tile, tl.trans(cb_tile))

        # Distance: -2*dot + ||c||^2
        dists = -2.0 * dots + cb_norms[None, :]

        # Per-row argmin over this tile
        local_min_val = tl.min(dists, axis=1)
        local_min_idx = tl.argmin(dists, axis=1)

        # Update running best
        update = local_min_val < best_dist
        best_dist = tl.where(update, local_min_val, best_dist)
        best_idx = tl.where(update, (local_min_idx + k_start).to(tl.int32), best_idx)

    tl.store(idx_ptr + n_offsets, best_idx, mask=n_mask)


def triton_codebook_nn(x: torch.Tensor, codebook: torch.Tensor,
                       codebook_norms: torch.Tensor,
                       _block_n: int = 32, _block_k: int = 256) -> torch.Tensor:
    """Find nearest codebook vector for each row of x.

    Args:
        x: (N, D) input vectors (any dtype, converted to fp16)
        codebook: (K, D) codebook vectors (any dtype, converted to fp16)
        codebook_norms: (K,) precomputed ||c||^2

    Returns:
        indices: (N,) int32 indices into codebook
    """
    N, D = x.shape
    K = codebook.shape[0]
    D_PAD = 16  # pad D=8 to 16 for Tensor Core mma.m16n8k16

    x_half = x.half().contiguous()
    cb_half = codebook.half().contiguous()
    cb_norms_f32 = codebook_norms.float().contiguous()

    indices = torch.empty(N, dtype=torch.int32, device=x.device)

    grid = ((N + _block_n - 1) // _block_n,)
    _codebook_nn_kernel[grid](
        x_half, cb_half, cb_norms_f32, indices,
        N, D, K,
        x_half.stride(0), cb_half.stride(0),
        BLOCK_N=_block_n, BLOCK_K=_block_k, D_PAD=D_PAD,
    )

    return indices.long()
