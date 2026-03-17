"""Triton fused codebook nearest-neighbour kernels.

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


@triton.jit
def _codebook_nn_decode_kernel(
    x_ptr,          # (N, D) input target vectors, fp16
    cb_ptr,         # (K, D) codebook vectors, fp16
    cb_norms_ptr,   # (K,) precomputed ||c||^2, fp32
    idx_ptr,        # (N,) output indices, int64
    dec_ptr,        # (N, D) output decoded vectors, fp16
    N,
    D: tl.constexpr,
    K,
    stride_x_n,
    stride_cb_k,
    stride_dec_n,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    D_PAD: tl.constexpr,
):
    """Fused NN search + decode: finds nearest codebook vector AND writes
    the decoded values, eliminating a separate gather kernel launch."""
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    d_offsets = tl.arange(0, D_PAD)
    d_mask = d_offsets < D

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

        cb_tile = tl.load(
            cb_ptr + k_offsets[:, None] * stride_cb_k + d_offsets[None, :],
            mask=k_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float16)

        cb_norms = tl.load(
            cb_norms_ptr + k_offsets, mask=k_mask, other=float('inf'),
        ).to(tl.float32)

        dots = tl.dot(x_tile, tl.trans(cb_tile))
        dists = -2.0 * dots + cb_norms[None, :]

        local_min_val = tl.min(dists, axis=1)
        local_min_idx = tl.argmin(dists, axis=1)

        update = local_min_val < best_dist
        best_dist = tl.where(update, local_min_val, best_dist)
        best_idx = tl.where(update, (local_min_idx + k_start).to(tl.int32), best_idx)

    # Store indices
    tl.store(idx_ptr + n_offsets, best_idx.to(tl.int64), mask=n_mask)

    # Gather decoded vectors from codebook using best_idx
    dec_tile = tl.load(
        cb_ptr + best_idx[:, None].to(tl.int64) * stride_cb_k + d_offsets[None, :],
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    tl.store(
        dec_ptr + n_offsets[:, None] * stride_dec_n + d_offsets[None, :],
        dec_tile,
        mask=n_mask[:, None] & d_mask[None, :],
    )


def _select_block_params(N, K):
    """Select BLOCK_N and BLOCK_K based on problem size."""
    if N <= 1024:
        return 16, 1024
    elif N <= 4096:
        return 16, 512
    else:
        return 32, 512


def triton_codebook_nn(x: torch.Tensor, codebook: torch.Tensor,
                       codebook_norms: torch.Tensor,
                       _block_n: int = 0, _block_k: int = 0,
                       out: torch.Tensor = None) -> torch.Tensor:
    """Find nearest codebook vector for each row of x.

    Args:
        x: (N, D) input vectors (any dtype, converted to fp16)
        codebook: (K, D) codebook vectors, fp16 preferred (avoids copy)
        codebook_norms: (K,) precomputed ||c||^2, fp32 preferred
        out: optional pre-allocated (N,) int64 output buffer

    Returns:
        indices: (N,) int64 indices into codebook
    """
    N, D = x.shape
    K = codebook.shape[0]
    D_PAD = 16  # pad D=8 to 16 for Tensor Core mma.m16n8k16

    # Auto-select tile sizes if not specified
    if _block_n == 0 or _block_k == 0:
        _block_n, _block_k = _select_block_params(N, K)

    # Avoid redundant conversions — codebook is typically already fp16
    x_half = x if x.dtype == torch.float16 else x.half()
    if not x_half.is_contiguous():
        x_half = x_half.contiguous()
    cb_half = codebook if codebook.dtype == torch.float16 else codebook.half()
    if not cb_half.is_contiguous():
        cb_half = cb_half.contiguous()
    cb_norms_f32 = codebook_norms if codebook_norms.dtype == torch.float32 else codebook_norms.float()
    if not cb_norms_f32.is_contiguous():
        cb_norms_f32 = cb_norms_f32.contiguous()

    indices_i32 = torch.empty(N, dtype=torch.int32, device=x.device)

    grid = ((N + _block_n - 1) // _block_n,)
    _codebook_nn_kernel[grid](
        x_half, cb_half, cb_norms_f32, indices_i32,
        N, D, K,
        x_half.stride(0), cb_half.stride(0),
        BLOCK_N=_block_n, BLOCK_K=_block_k, D_PAD=D_PAD,
    )

    if out is not None:
        out.copy_(indices_i32)
        return out
    return indices_i32.long()


def triton_codebook_nn_decode(x: torch.Tensor, codebook: torch.Tensor,
                              codebook_norms: torch.Tensor,
                              decoded_out: torch.Tensor = None,
                              idx_out: torch.Tensor = None,
                              ):
    """Fused NN search + decode: returns (decoded_vectors, indices).

    Eliminates the separate gather kernel that codebook[indices] requires.
    """
    N, D = x.shape
    K = codebook.shape[0]
    D_PAD = 16
    _block_n, _block_k = _select_block_params(N, K)

    x_half = x if x.dtype == torch.float16 else x.half()
    if not x_half.is_contiguous():
        x_half = x_half.contiguous()
    cb_half = codebook if codebook.dtype == torch.float16 else codebook.half()
    if not cb_half.is_contiguous():
        cb_half = cb_half.contiguous()
    cb_norms_f32 = codebook_norms if codebook_norms.dtype == torch.float32 else codebook_norms.float()
    if not cb_norms_f32.is_contiguous():
        cb_norms_f32 = cb_norms_f32.contiguous()

    if idx_out is None:
        idx_out = torch.empty(N, dtype=torch.int64, device=x.device)
    if decoded_out is None:
        decoded_out = torch.empty(N, D, dtype=torch.float16, device=x.device)

    grid = ((N + _block_n - 1) // _block_n,)
    _codebook_nn_decode_kernel[grid](
        x_half, cb_half, cb_norms_f32, idx_out, decoded_out,
        N, D, K,
        x_half.stride(0), cb_half.stride(0), decoded_out.stride(0),
        BLOCK_N=_block_n, BLOCK_K=_block_k, D_PAD=D_PAD,
    )

    return decoded_out, idx_out
