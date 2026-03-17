"""Triton fused dequant+matmul kernels for GLQ inference.

Computes Y = X @ W^T * Wscale where W is stored as codebook indices,
without materializing the full weight matrix in GPU memory.

For 2bpw: W[i, j:j+8] = codebook[Qidxs[i, j//8]]
For 3/4bpw: W[i, j:j+8] = cb1[idx1[i, j//8]] + cb2[idx2[i, j//8]] * inv_resid_scale

Kernels:
  _glq_dequant_matvec_kernel: B=1 decode (autotuned BLOCK_M)
  _glq_dequant_matmul_tc_kernel: B>=2 prefill (Tensor Core tl.dot with K=16)
  _splitk_matvec_kernel: B=1 split-K for small grids (more CTAs → better SM util)
  _splitk_matmul_tc_kernel: B>=2 split-K Tensor Core variant
"""

import torch

try:
    import triton
    import triton.language as tl

    _triton_available = True
except ImportError:
    _triton_available = False


if _triton_available:

    # ────────────────────────────────────────────────────────────────
    # Matvec kernel (B=1 decode) — autotuned BLOCK_M
    # ────────────────────────────────────────────────────────────────

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=4),
            triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=4),
        ],
        key=['M', 'N_BLOCKS'],
    )
    @triton.jit
    def _glq_dequant_matvec_kernel(
        # Pointers
        x_ptr,
        qidxs_ptr,
        codebook_ptr,
        y_ptr,
        # 2-stage
        qidxs2_ptr,
        codebook2_ptr,
        inv_resid_scale,
        # Dimensions
        M,
        N_BLOCKS,  # N // 8
        Wscale,
        # Strides
        stride_q_m,
        stride_q_k,
        stride_cb_k,
        stride_q2_m,
        stride_q2_k,
        stride_cb2_k,
        # Config
        HAS_STAGE2: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Fused dequant+matvec for B=1: y = dequant(Qidxs) @ x * Wscale."""
        pid = tl.program_id(0)
        m_start = pid * BLOCK_M
        m_range = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_range < M

        d_range = tl.arange(0, 8)
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for j in range(N_BLOCKS):
            # Load x[j*8 : j*8+8]
            x_vec = tl.load(x_ptr + j * 8 + d_range).to(tl.float32)

            # Load indices: (BLOCK_M,)
            indices = tl.load(
                qidxs_ptr + m_range * stride_q_m + j * stride_q_k,
                mask=m_mask,
                other=0,
            )
            indices = (indices.to(tl.int32) & 0xFFFF)

            # Gather codebook: (BLOCK_M, 8)
            cb_vecs = tl.load(
                codebook_ptr + indices[:, None] * stride_cb_k + d_range[None, :],
                mask=m_mask[:, None],
                other=0.0,
            ).to(tl.float32)

            if HAS_STAGE2:
                indices2 = tl.load(
                    qidxs2_ptr + m_range * stride_q2_m + j * stride_q2_k,
                    mask=m_mask,
                    other=0,
                )
                indices2 = (indices2.to(tl.int32) & 0xFFFF)
                cb_vecs2 = tl.load(
                    codebook2_ptr + indices2[:, None] * stride_cb2_k + d_range[None, :],
                    mask=m_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                cb_vecs = cb_vecs + cb_vecs2 * inv_resid_scale

            # Dot: (BLOCK_M, 8) . (8,) -> (BLOCK_M,)
            acc += tl.sum(cb_vecs * x_vec[None, :], axis=1)

        acc *= Wscale
        tl.store(y_ptr + m_range, acc, mask=m_mask)

    # ────────────────────────────────────────────────────────────────
    # Tensor Core matmul kernel (B>=2) — tl.dot with K=16
    # ────────────────────────────────────────────────────────────────

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_B': 16, 'BLOCK_M': 16}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 16, 'BLOCK_M': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 16, 'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 64}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 64}, num_warps=8, num_stages=4),
        ],
        key=['B', 'M', 'N_BLOCKS'],
    )
    @triton.jit
    def _glq_dequant_matmul_tc_kernel(
        # Pointers
        x_ptr,
        qidxs_ptr,
        codebook_ptr,
        y_ptr,
        # 2-stage pointers (optional)
        qidxs2_ptr,
        codebook2_ptr,
        inv_resid_scale,
        # Dimensions
        B,
        M,
        N,
        N_BLOCKS,  # N // 8
        Wscale,
        # Strides
        stride_x_b,
        stride_q_m,
        stride_q_k,
        stride_cb_k,
        stride_q2_m,
        stride_q2_k,
        stride_cb2_k,
        stride_y_b,
        # Config
        HAS_STAGE2: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Fused dequant+matmul with Tensor Core tl.dot (K=16).

        Processes 2 consecutive codebook blocks per iteration to form K=16
        tiles suitable for mma.m16n8k16 Tensor Core instructions.

        For each pair of blocks (j, j+1):
          x_tile: (BLOCK_B, 16) — 16 contiguous input values
          w_tile: (BLOCK_M, 16) — concat of 2 codebook gathers
          acc += tl.dot(x_tile, w_tile^T)
        """
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)

        b_start = pid_b * BLOCK_B
        m_start = pid_m * BLOCK_M

        b_range = b_start + tl.arange(0, BLOCK_B)
        m_range = m_start + tl.arange(0, BLOCK_M)
        b_mask = b_range < B
        m_mask = m_range < M

        acc = tl.zeros((BLOCK_B, BLOCK_M), dtype=tl.float32)

        # Precompute dimension helpers for K=16 tile construction
        d16 = tl.arange(0, 16)
        is_hi = d16 >= 8                           # which dims come from block j+1
        d_local = tl.where(is_hi, d16 - 8, d16)   # local dim within 8-d codebook entry

        # Process pairs of codebook blocks for K=16
        for j in range(0, N_BLOCKS, 2):
            # --- x tile: (BLOCK_B, 16) from two consecutive 8-blocks ---
            x_tile = tl.load(
                x_ptr + b_range[:, None] * stride_x_b + (j * 8 + d16[None, :]),
                mask=b_mask[:, None] & ((j * 8 + d16[None, :]) < N),
                other=0.0,
            ).to(tl.float16)

            # --- Primary indices for blocks j and j+1 ---
            idx0 = (tl.load(
                qidxs_ptr + m_range * stride_q_m + j * stride_q_k,
                mask=m_mask, other=0,
            ).to(tl.int32) & 0xFFFF)

            j1_valid = (j + 1) < N_BLOCKS
            idx1 = (tl.load(
                qidxs_ptr + m_range * stride_q_m + (j + 1) * stride_q_k,
                mask=m_mask & j1_valid, other=0,
            ).to(tl.int32) & 0xFFFF)

            # --- w tile: (BLOCK_M, 16) by concatenating two 8-d codebook gathers ---
            # For d in 0..7:  codebook[idx0[m], d]
            # For d in 8..15: codebook[idx1[m], d-8]
            idx_sel = tl.where(is_hi[None, :], idx1[:, None], idx0[:, None])
            w_tile = tl.load(
                codebook_ptr + idx_sel * stride_cb_k + d_local[None, :],
                mask=m_mask[:, None],
                other=0.0,
            ).to(tl.float16)

            if HAS_STAGE2:
                # Secondary codebook indices
                idx2_0 = (tl.load(
                    qidxs2_ptr + m_range * stride_q2_m + j * stride_q2_k,
                    mask=m_mask, other=0,
                ).to(tl.int32) & 0xFFFF)

                idx2_1 = (tl.load(
                    qidxs2_ptr + m_range * stride_q2_m + (j + 1) * stride_q2_k,
                    mask=m_mask & j1_valid, other=0,
                ).to(tl.int32) & 0xFFFF)

                idx2_sel = tl.where(is_hi[None, :], idx2_1[:, None], idx2_0[:, None])
                w2_tile = tl.load(
                    codebook2_ptr + idx2_sel * stride_cb2_k + d_local[None, :],
                    mask=m_mask[:, None],
                    other=0.0,
                ).to(tl.float32)

                # Combine in fp32 then cast back to fp16
                w_tile = (w_tile.to(tl.float32) + w2_tile * inv_resid_scale).to(tl.float16)

            # --- Tensor Core matmul: (BLOCK_B, 16) @ (16, BLOCK_M) ---
            acc += tl.dot(x_tile, tl.trans(w_tile))

        acc *= Wscale
        tl.store(
            y_ptr + b_range[:, None] * stride_y_b + m_range[None, :],
            acc,
            mask=b_mask[:, None] & m_mask[None, :],
        )

    # ────────────────────────────────────────────────────────────────
    # Split-K matvec (B=1) — distributes N reduction across CTAs
    # ────────────────────────────────────────────────────────────────

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=4),
            triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=2),
        ],
        key=['M', 'BLOCKS_PER_SPLIT'],
        reset_to_zero=['y_ptr'],
    )
    @triton.jit
    def _splitk_matvec_kernel(
        x_ptr,
        qidxs_ptr,
        codebook_ptr,
        y_ptr,
        qidxs2_ptr,
        codebook2_ptr,
        inv_resid_scale,
        M,
        N_BLOCKS,
        Wscale,
        stride_q_m,
        stride_q_k,
        stride_cb_k,
        stride_q2_m,
        stride_q2_k,
        stride_cb2_k,
        HAS_STAGE2: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
    ):
        """Split-K dequant+matvec: each CTA processes BLOCKS_PER_SPLIT blocks,
        atomicAdds partial sums. Grid: (M/BLOCK_M, ceil(N_BLOCKS/BPS))."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        m_start = pid_m * BLOCK_M
        m_range = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_range < M

        j_start = pid_k * BLOCKS_PER_SPLIT
        d_range = tl.arange(0, 8)
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for j_off in tl.static_range(BLOCKS_PER_SPLIT):
            j = j_start + j_off
            j_mask = j < N_BLOCKS

            x_vec = tl.load(x_ptr + j * 8 + d_range,
                            mask=j_mask, other=0.0).to(tl.float32)

            indices = tl.load(
                qidxs_ptr + m_range * stride_q_m + j * stride_q_k,
                mask=m_mask & j_mask, other=0,
            )
            indices = (indices.to(tl.int32) & 0xFFFF)

            cb_vecs = tl.load(
                codebook_ptr + indices[:, None] * stride_cb_k + d_range[None, :],
                mask=m_mask[:, None] & j_mask, other=0.0,
            ).to(tl.float32)

            if HAS_STAGE2:
                indices2 = tl.load(
                    qidxs2_ptr + m_range * stride_q2_m + j * stride_q2_k,
                    mask=m_mask & j_mask, other=0,
                )
                indices2 = (indices2.to(tl.int32) & 0xFFFF)
                cb_vecs2 = tl.load(
                    codebook2_ptr + indices2[:, None] * stride_cb2_k + d_range[None, :],
                    mask=m_mask[:, None] & j_mask, other=0.0,
                ).to(tl.float32)
                cb_vecs = cb_vecs + cb_vecs2 * inv_resid_scale

            acc += tl.sum(cb_vecs * x_vec[None, :], axis=1)

        tl.atomic_add(y_ptr + m_range, acc * Wscale, mask=m_mask)

    # ────────────────────────────────────────────────────────────────
    # Split-K Tensor Core matmul (B>=2) — 3D grid
    # ────────────────────────────────────────────────────────────────

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_B': 16, 'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 16, 'BLOCK_M': 64}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 32}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 64}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 64}, num_warps=8, num_stages=2),
        ],
        key=['B', 'M', 'BLOCKS_PER_SPLIT'],
        reset_to_zero=['y_ptr'],
    )
    @triton.jit
    def _splitk_matmul_tc_kernel(
        x_ptr,
        qidxs_ptr,
        codebook_ptr,
        y_ptr,
        qidxs2_ptr,
        codebook2_ptr,
        inv_resid_scale,
        B,
        M,
        N,
        N_BLOCKS,
        Wscale,
        stride_x_b,
        stride_q_m,
        stride_q_k,
        stride_cb_k,
        stride_q2_m,
        stride_q2_k,
        stride_cb2_k,
        stride_y_b,
        HAS_STAGE2: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
    ):
        """Split-K dequant+matmul with Tensor Cores.
        Grid: (B/BLOCK_B, M/BLOCK_M, ceil(N_BLOCKS/BPS))."""
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_k = tl.program_id(2)

        b_start = pid_b * BLOCK_B
        m_start = pid_m * BLOCK_M

        b_range = b_start + tl.arange(0, BLOCK_B)
        m_range = m_start + tl.arange(0, BLOCK_M)
        b_mask = b_range < B
        m_mask = m_range < M

        j_start = pid_k * BLOCKS_PER_SPLIT
        acc = tl.zeros((BLOCK_B, BLOCK_M), dtype=tl.float32)

        d16 = tl.arange(0, 16)
        is_hi = d16 >= 8
        d_local = tl.where(is_hi, d16 - 8, d16)

        # Process pairs of blocks for K=16 TC tiles
        for pair_off in tl.static_range(BLOCKS_PER_SPLIT // 2):
            j = j_start + pair_off * 2
            j_valid = j < N_BLOCKS

            x_tile = tl.load(
                x_ptr + b_range[:, None] * stride_x_b + (j * 8 + d16[None, :]),
                mask=b_mask[:, None] & ((j * 8 + d16[None, :]) < N) & j_valid,
                other=0.0,
            ).to(tl.float16)

            idx0 = (tl.load(
                qidxs_ptr + m_range * stride_q_m + j * stride_q_k,
                mask=m_mask & j_valid, other=0,
            ).to(tl.int32) & 0xFFFF)

            j1_valid = (j + 1) < N_BLOCKS
            idx1 = (tl.load(
                qidxs_ptr + m_range * stride_q_m + (j + 1) * stride_q_k,
                mask=m_mask & j1_valid, other=0,
            ).to(tl.int32) & 0xFFFF)

            idx_sel = tl.where(is_hi[None, :], idx1[:, None], idx0[:, None])
            w_tile = tl.load(
                codebook_ptr + idx_sel * stride_cb_k + d_local[None, :],
                mask=m_mask[:, None],
                other=0.0,
            ).to(tl.float16)

            if HAS_STAGE2:
                idx2_0 = (tl.load(
                    qidxs2_ptr + m_range * stride_q2_m + j * stride_q2_k,
                    mask=m_mask & j_valid, other=0,
                ).to(tl.int32) & 0xFFFF)

                idx2_1 = (tl.load(
                    qidxs2_ptr + m_range * stride_q2_m + (j + 1) * stride_q2_k,
                    mask=m_mask & j1_valid, other=0,
                ).to(tl.int32) & 0xFFFF)

                idx2_sel = tl.where(is_hi[None, :], idx2_1[:, None], idx2_0[:, None])
                w2_tile = tl.load(
                    codebook2_ptr + idx2_sel * stride_cb2_k + d_local[None, :],
                    mask=m_mask[:, None],
                    other=0.0,
                ).to(tl.float32)

                w_tile = (w_tile.to(tl.float32) + w2_tile * inv_resid_scale).to(tl.float16)

            acc += tl.dot(x_tile, tl.trans(w_tile))

        # 2D atomic add
        scaled = acc * Wscale
        addrs = b_range[:, None] * stride_y_b + m_range[None, :]
        tl.atomic_add(y_ptr + addrs, scaled, mask=b_mask[:, None] & m_mask[None, :])

    # ────────────────────────────────────────────────────────────────
    # Packed uint32 split-K matvec (B=1) — 4x less L2 traffic
    # E8 codebook coords are {-3,-2.5,...,2.5,3} = (nibble-6)*0.5
    # Pack 8 nibbles into 1 uint32: 16B gather → 4B gather
    # ────────────────────────────────────────────────────────────────

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=4),
            triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=2),
        ],
        key=['M', 'BLOCKS_PER_SPLIT'],
        reset_to_zero=['y_ptr'],
    )
    @triton.jit
    def _packed_splitk_matvec_kernel(
        x_ptr,
        qidxs_ptr,
        packed_cb_ptr,
        y_ptr,
        M,
        N_BLOCKS,
        Wscale,
        stride_q_m,
        stride_q_k,
        BLOCK_M: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
    ):
        """Split-K matvec with packed uint32 codebook (4B per entry vs 16B)."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        m_start = pid_m * BLOCK_M
        m_range = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_range < M

        j_start = pid_k * BLOCKS_PER_SPLIT
        d_range = tl.arange(0, 8)
        shifts = d_range * 4
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for j_off in tl.static_range(BLOCKS_PER_SPLIT):
            j = j_start + j_off
            j_mask = j < N_BLOCKS

            x_vec = tl.load(x_ptr + j * 8 + d_range,
                            mask=j_mask, other=0.0).to(tl.float32)

            indices = tl.load(
                qidxs_ptr + m_range * stride_q_m + j * stride_q_k,
                mask=m_mask & j_mask, other=0,
            )
            indices = (indices.to(tl.int32) & 0xFFFF)

            packed = tl.load(packed_cb_ptr + indices,
                             mask=m_mask & j_mask, other=0)
            nibbles = (packed[:, None] >> shifts[None, :]) & 0xF
            cb_vecs = nibbles.to(tl.float32) * 0.5 - 3.0

            acc += tl.sum(cb_vecs * x_vec[None, :], axis=1)

        tl.atomic_add(y_ptr + m_range, acc * Wscale, mask=m_mask)

    # ────────────────────────────────────────────────────────────────
    # Packed uint32 split-K TC matmul (B>=2)
    # ────────────────────────────────────────────────────────────────

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_B': 16, 'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 16, 'BLOCK_M': 64}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 32}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 64}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_B': 32, 'BLOCK_M': 64}, num_warps=8, num_stages=2),
        ],
        key=['B', 'M', 'BLOCKS_PER_SPLIT'],
        reset_to_zero=['y_ptr'],
    )
    @triton.jit
    def _packed_splitk_matmul_tc_kernel(
        x_ptr,
        qidxs_ptr,
        packed_cb_ptr,
        y_ptr,
        B, M, N, N_BLOCKS, Wscale,
        stride_x_b, stride_q_m, stride_q_k, stride_y_b,
        BLOCK_B: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
    ):
        """Split-K TC matmul with packed uint32 codebook."""
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_k = tl.program_id(2)

        b_start = pid_b * BLOCK_B
        m_start = pid_m * BLOCK_M

        b_range = b_start + tl.arange(0, BLOCK_B)
        m_range = m_start + tl.arange(0, BLOCK_M)
        b_mask = b_range < B
        m_mask = m_range < M

        j_start = pid_k * BLOCKS_PER_SPLIT
        acc = tl.zeros((BLOCK_B, BLOCK_M), dtype=tl.float32)

        d16 = tl.arange(0, 16)
        is_hi = d16 >= 8
        d_local = tl.where(is_hi, d16 - 8, d16)
        shifts = d_local * 4

        for pair_off in tl.static_range(BLOCKS_PER_SPLIT // 2):
            j = j_start + pair_off * 2
            j_valid = j < N_BLOCKS

            x_tile = tl.load(
                x_ptr + b_range[:, None] * stride_x_b + (j * 8 + d16[None, :]),
                mask=b_mask[:, None] & ((j * 8 + d16[None, :]) < N) & j_valid,
                other=0.0,
            ).to(tl.float16)

            idx0 = (tl.load(
                qidxs_ptr + m_range * stride_q_m + j * stride_q_k,
                mask=m_mask & j_valid, other=0,
            ).to(tl.int32) & 0xFFFF)

            j1_valid = (j + 1) < N_BLOCKS
            idx1 = (tl.load(
                qidxs_ptr + m_range * stride_q_m + (j + 1) * stride_q_k,
                mask=m_mask & j1_valid, other=0,
            ).to(tl.int32) & 0xFFFF)

            packed0 = tl.load(packed_cb_ptr + idx0, mask=m_mask & j_valid, other=0)
            packed1 = tl.load(packed_cb_ptr + idx1, mask=m_mask & j1_valid, other=0)
            packed_sel = tl.where(is_hi[None, :], packed1[:, None], packed0[:, None])
            nibbles = (packed_sel >> shifts[None, :]) & 0xF
            w_tile = (nibbles.to(tl.float32) * 0.5 - 3.0).to(tl.float16)

            acc += tl.dot(x_tile, tl.trans(w_tile))

        scaled = acc * Wscale
        addrs = b_range[:, None] * stride_y_b + m_range[None, :]
        tl.atomic_add(y_ptr + addrs, scaled, mask=b_mask[:, None] & m_mask[None, :])


# Cached SM count to avoid repeated GPU queries
_num_sms: int = 0
_BLOCKS_PER_SPLIT = 64


def _get_num_sms(device) -> int:
    global _num_sms
    if _num_sms == 0:
        _num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    return _num_sms


def glq_dequant_matmul(
    x: torch.Tensor,
    Qidxs: torch.Tensor,
    codebook: torch.Tensor,
    Wscale: float,
    Qidxs2: torch.Tensor = None,
    codebook2: torch.Tensor = None,
    inv_resid_scale: float = 0.0,
    codebook_packed: torch.Tensor = None,
) -> torch.Tensor:
    """Fused dequant+matmul: Y = X @ dequant(Qidxs)^T * Wscale.

    Auto-selects the fastest kernel variant based on shape and GPU:
    - Split-K when grid undersaturates SMs (1.5-2.4x speedup)
    - Packed uint32 codebook for small matrices (additional 1.3x speedup)

    Args:
        x: (B, N) input activations
        Qidxs: (M, N//8) primary codebook indices, int16
        codebook: (K, 8) primary codebook vectors, fp16/fp32
        Wscale: global scale factor
        Qidxs2: (M, N//8) secondary indices for 3/4bpw, or None
        codebook2: (K2, 8) secondary codebook for 3/4bpw, or None
        inv_resid_scale: 1.0 / resid_scale for 3/4bpw
        codebook_packed: (K,) uint32 packed codebook, or None

    Returns:
        Y: (B, M) output in fp32
    """
    if not _triton_available or not x.is_cuda:
        return _fallback_dequant_matmul(
            x, Qidxs, codebook, Wscale, Qidxs2, codebook2, inv_resid_scale
        )

    B, N = x.shape
    M, N_BLOCKS = Qidxs.shape
    assert N_BLOCKS == N // 8, f"Qidxs shape {Qidxs.shape} incompatible with N={N}"
    assert codebook.shape[1] == 8

    x_fp16 = x.half().contiguous()
    cb = codebook.half().contiguous()

    has_stage2 = Qidxs2 is not None and codebook2 is not None

    if has_stage2:
        q2 = Qidxs2.contiguous()
        cb2 = codebook2.half().contiguous()
    else:
        # Dummy pointers (won't be accessed when HAS_STAGE2=False)
        q2 = Qidxs
        cb2 = cb

    # Decide whether to use split-K based on estimated grid saturation.
    # BLOCK_M=64 is typical for both matvec and TC paths.
    num_sms = _get_num_sms(x.device)
    block_m_est = 64
    if B == 1:
        est_grid = triton.cdiv(M, block_m_est)
    else:
        block_b_est = 32
        est_grid = triton.cdiv(B, block_b_est) * triton.cdiv(M, block_m_est)

    use_splitk = (est_grid < num_sms) and (N_BLOCKS >= _BLOCKS_PER_SPLIT)

    # Packed uint32 codebook: 4B gather instead of 16B.
    # Benchmarks show packed is 2-6% faster across all shapes (ALU decode
    # cost is negligible vs L2 bandwidth savings on L40S).
    # Only for 2bpw (no 2-stage support in packed kernels).
    use_packed = (
        use_splitk
        and codebook_packed is not None
        and not has_stage2
    )

    if use_packed:
        # Packed split-K path: 4B gather instead of 16B
        y = torch.zeros(B, M, dtype=torch.float32, device=x.device)
        bps = _BLOCKS_PER_SPLIT
        cb_packed = codebook_packed.contiguous()

        if B == 1:
            n_splits = triton.cdiv(N_BLOCKS, bps)
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), n_splits)
            _packed_splitk_matvec_kernel[grid](
                x_fp16[0], Qidxs, cb_packed, y[0],
                M, N_BLOCKS, Wscale,
                Qidxs.stride(0), Qidxs.stride(1),
                BLOCKS_PER_SPLIT=bps,
            )
        else:
            bps_even = bps if bps % 2 == 0 else bps - 1
            n_splits = triton.cdiv(N_BLOCKS, bps_even)
            grid = lambda meta: (
                triton.cdiv(B, meta['BLOCK_B']),
                triton.cdiv(M, meta['BLOCK_M']),
                n_splits,
            )
            _packed_splitk_matmul_tc_kernel[grid](
                x_fp16, Qidxs, cb_packed, y,
                B, M, N, N_BLOCKS, Wscale,
                x_fp16.stride(0),
                Qidxs.stride(0), Qidxs.stride(1),
                y.stride(0),
                BLOCKS_PER_SPLIT=bps_even,
            )
    elif use_splitk:
        # Split-K: zero-init output for atomic accumulation
        y = torch.zeros(B, M, dtype=torch.float32, device=x.device)
        bps = _BLOCKS_PER_SPLIT

        if B == 1:
            n_splits = triton.cdiv(N_BLOCKS, bps)
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), n_splits)
            _splitk_matvec_kernel[grid](
                x_fp16[0],
                Qidxs, cb, y[0],
                q2, cb2, inv_resid_scale,
                M, N_BLOCKS, Wscale,
                Qidxs.stride(0), Qidxs.stride(1), cb.stride(0),
                q2.stride(0), q2.stride(1), cb2.stride(0),
                HAS_STAGE2=has_stage2,
                BLOCKS_PER_SPLIT=bps,
            )
        else:
            # Ensure bps is even for K=16 TC pairing
            bps_even = bps if bps % 2 == 0 else bps - 1
            n_splits = triton.cdiv(N_BLOCKS, bps_even)
            grid = lambda meta: (
                triton.cdiv(B, meta['BLOCK_B']),
                triton.cdiv(M, meta['BLOCK_M']),
                n_splits,
            )
            _splitk_matmul_tc_kernel[grid](
                x_fp16, Qidxs, cb, y,
                q2, cb2, inv_resid_scale,
                B, M, N, N_BLOCKS, Wscale,
                x_fp16.stride(0),
                Qidxs.stride(0), Qidxs.stride(1), cb.stride(0),
                q2.stride(0), q2.stride(1), cb2.stride(0),
                y.stride(0),
                HAS_STAGE2=has_stage2,
                BLOCKS_PER_SPLIT=bps_even,
            )
    elif B == 1:
        # Original matvec path
        y = torch.empty(B, M, dtype=torch.float32, device=x.device)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
        _glq_dequant_matvec_kernel[grid](
            x_fp16[0],
            Qidxs, cb, y[0],
            q2, cb2, inv_resid_scale,
            M, N_BLOCKS, Wscale,
            Qidxs.stride(0), Qidxs.stride(1), cb.stride(0),
            q2.stride(0), q2.stride(1), cb2.stride(0),
            HAS_STAGE2=has_stage2,
        )
    else:
        # Original TC matmul path
        y = torch.empty(B, M, dtype=torch.float32, device=x.device)
        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_B']), triton.cdiv(M, meta['BLOCK_M']))
        _glq_dequant_matmul_tc_kernel[grid](
            x_fp16, Qidxs, cb, y,
            q2, cb2, inv_resid_scale,
            B, M, N, N_BLOCKS, Wscale,
            x_fp16.stride(0),
            Qidxs.stride(0), Qidxs.stride(1), cb.stride(0),
            q2.stride(0), q2.stride(1), cb2.stride(0),
            y.stride(0),
            HAS_STAGE2=has_stage2,
        )

    return y


def _fallback_dequant_matmul(
    x: torch.Tensor,
    Qidxs: torch.Tensor,
    codebook: torch.Tensor,
    Wscale: float,
    Qidxs2: torch.Tensor = None,
    codebook2: torch.Tensor = None,
    inv_resid_scale: float = 0.0,
) -> torch.Tensor:
    """Naive fallback: materialize W then matmul."""
    M, n_blocks = Qidxs.shape
    N = n_blocks * 8

    # Convert int16 to unsigned indices (0-65535)
    idx = (Qidxs.long() & 0xFFFF).reshape(-1)
    W = codebook[idx].reshape(M, N)

    if Qidxs2 is not None and codebook2 is not None:
        idx2 = (Qidxs2.long() & 0xFFFF).reshape(-1)
        W2 = codebook2[idx2].reshape(M, N)
        W = W + W2 * inv_resid_scale

    return x.float() @ W.float().T * Wscale
