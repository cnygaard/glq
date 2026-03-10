"""Triton fused dequant+matmul kernels for GLQ inference.

Computes Y = X @ W^T * Wscale where W is stored as codebook indices,
without materializing the full weight matrix in GPU memory.

For 2bpw: W[i, j:j+8] = codebook[Qidxs[i, j//8]]
For 3/4bpw: W[i, j:j+8] = cb1[idx1[i, j//8]] + cb2[idx2[i, j//8]] * inv_resid_scale
"""

import torch

try:
    import triton
    import triton.language as tl

    _triton_available = True
except ImportError:
    _triton_available = False


if _triton_available:

    @triton.jit
    def _glq_dequant_matmul_kernel(
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
        """Fused dequant+matmul via rank-8 outer-product accumulation.

        Each program computes a (BLOCK_B, BLOCK_M) tile of Y by iterating
        over all N/8 codebook blocks. For each block, loads 8 x-columns and
        8 w-columns individually and accumulates outer products.
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

        for j in range(N_BLOCKS):
            k_off = j * 8

            # Load primary indices: (BLOCK_M,)
            indices = tl.load(
                qidxs_ptr + m_range * stride_q_m + j * stride_q_k,
                mask=m_mask,
                other=0,
            )
            indices = (indices.to(tl.int32) & 0xFFFF)

            if HAS_STAGE2:
                indices2 = tl.load(
                    qidxs2_ptr + m_range * stride_q2_m + j * stride_q2_k,
                    mask=m_mask,
                    other=0,
                )
                indices2 = (indices2.to(tl.int32) & 0xFFFF)

            # Rank-8 outer-product accumulation using 1D column loads
            # (avoids 2D integer indexing unsupported in Triton 3.6+)
            for d in tl.static_range(8):
                # Load x[b, k_off+d] for each b: (BLOCK_B,)
                x_col = tl.load(
                    x_ptr + b_range * stride_x_b + (k_off + d),
                    mask=b_mask,
                    other=0.0,
                ).to(tl.float32)

                # Gather codebook[indices, d] for each m: (BLOCK_M,)
                w_col = tl.load(
                    codebook_ptr + indices * stride_cb_k + d,
                    mask=m_mask,
                    other=0.0,
                ).to(tl.float32)

                if HAS_STAGE2:
                    w_col2 = tl.load(
                        codebook2_ptr + indices2 * stride_cb2_k + d,
                        mask=m_mask,
                        other=0.0,
                    ).to(tl.float32)
                    w_col = w_col + w_col2 * inv_resid_scale

                # Outer product: (BLOCK_B, 1) * (1, BLOCK_M) -> (BLOCK_B, BLOCK_M)
                acc += x_col[:, None] * w_col[None, :]

        acc *= Wscale
        tl.store(
            y_ptr + b_range[:, None] * stride_y_b + m_range[None, :],
            acc,
            mask=b_mask[:, None] & m_mask[None, :],
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


def glq_dequant_matmul(
    x: torch.Tensor,
    Qidxs: torch.Tensor,
    codebook: torch.Tensor,
    Wscale: float,
    Qidxs2: torch.Tensor = None,
    codebook2: torch.Tensor = None,
    inv_resid_scale: float = 0.0,
) -> torch.Tensor:
    """Fused dequant+matmul: Y = X @ dequant(Qidxs)^T * Wscale.

    Args:
        x: (B, N) input activations
        Qidxs: (M, N//8) primary codebook indices, int16
        codebook: (K, 8) primary codebook vectors, fp16/fp32
        Wscale: global scale factor
        Qidxs2: (M, N//8) secondary indices for 3/4bpw, or None
        codebook2: (K2, 8) secondary codebook for 3/4bpw, or None
        inv_resid_scale: 1.0 / resid_scale for 3/4bpw

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

    y = torch.empty(B, M, dtype=torch.float32, device=x.device)

    if B <= 4:
        # Matvec path — process one token at a time
        BLOCK_M = 64
        grid = (triton.cdiv(M, BLOCK_M),)
        for b in range(B):
            _glq_dequant_matvec_kernel[grid](
                x_fp16[b],
                Qidxs,
                cb,
                y[b],
                q2,
                cb2,
                inv_resid_scale,
                M,
                N_BLOCKS,
                Wscale,
                Qidxs.stride(0),
                Qidxs.stride(1),
                cb.stride(0),
                q2.stride(0),
                q2.stride(1),
                cb2.stride(0),
                HAS_STAGE2=has_stage2,
                BLOCK_M=BLOCK_M,
            )
    else:
        # Batched matmul path
        BLOCK_B = 32
        BLOCK_M = 64
        grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(M, BLOCK_M))
        _glq_dequant_matmul_kernel[grid](
            x_fp16,
            Qidxs,
            cb,
            y,
            q2,
            cb2,
            inv_resid_scale,
            B,
            M,
            N,
            N_BLOCKS,
            Wscale,
            x_fp16.stride(0),
            Qidxs.stride(0),
            Qidxs.stride(1),
            cb.stride(0),
            q2.stride(0),
            q2.stride(1),
            cb2.stride(0),
            y.stride(0),
            HAS_STAGE2=has_stage2,
            BLOCK_B=BLOCK_B,
            BLOCK_M=BLOCK_M,
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
