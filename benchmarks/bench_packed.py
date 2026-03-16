"""Benchmark packed uint32 codebook kernel vs fp16 codebook kernel.

The E8 codebook uses only 13 coordinate values that fit in 4-bit nibbles.
Packing 8 nibbles into a uint32 reduces each gather from 16B to 4B (4x less
L2 bandwidth). The ALU decode cost (shift+mask+scale) should be negligible
compared to the memory savings.

Usage:
    python benchmarks/bench_packed.py
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    raise RuntimeError("Triton required")


def cuda_median_us(fn, n_warmup=20, n_iter=200):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000)
    times.sort()
    return times[len(times) // 2]


# ────────────────────────────────────────────────────────────────
# Packed split-K matvec (B=1)
# ────────────────────────────────────────────────────────────────

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

        # Load packed uint32 (4 bytes instead of 16)
        packed = tl.load(
            packed_cb_ptr + indices,
            mask=m_mask & j_mask, other=0,
        )

        # Decode 8 nibbles → 8 fp32 values
        # val_i = ((packed >> (4*i)) & 0xF) * 0.5 - 3.0
        for d in tl.static_range(8):
            nibble = (packed >> (d * 4)) & 0xF
            val = nibble.to(tl.float32) * 0.5 - 3.0
            acc += val * tl.load(x_ptr + j * 8 + d, mask=j_mask, other=0.0).to(tl.float32)

    tl.atomic_add(y_ptr + m_range, acc * Wscale, mask=m_mask)


# Variant using vectorized decode (decode all 8 at once)
@triton.jit
def _packed_splitk_matvec_v2_kernel(
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
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    m_range = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_range < M

    j_start = pid_k * BLOCKS_PER_SPLIT
    d_range = tl.arange(0, 8)
    shifts = d_range * 4  # [0, 4, 8, 12, 16, 20, 24, 28]
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

        # Load packed uint32 (4 bytes): (BLOCK_M,)
        packed = tl.load(
            packed_cb_ptr + indices,
            mask=m_mask & j_mask, other=0,
        )

        # Vectorized decode: (BLOCK_M, 8)
        # Broadcast packed to (BLOCK_M, 8) and shift
        nibbles = (packed[:, None] >> shifts[None, :]) & 0xF
        cb_vecs = nibbles.to(tl.float32) * 0.5 - 3.0

        # Dot: (BLOCK_M, 8) . (8,) -> (BLOCK_M,)
        acc += tl.sum(cb_vecs * x_vec[None, :], axis=1)

    tl.atomic_add(y_ptr + m_range, acc * Wscale, mask=m_mask)


# ────────────────────────────────────────────────────────────────
# Packed split-K TC matmul (B>=2)
# ────────────────────────────────────────────────────────────────

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

        # Load packed uint32 for both blocks: (BLOCK_M,) each
        packed0 = tl.load(packed_cb_ptr + idx0, mask=m_mask & j_valid, other=0)
        packed1 = tl.load(packed_cb_ptr + idx1, mask=m_mask & j1_valid, other=0)

        # Select packed value based on hi/lo
        packed_sel = tl.where(is_hi[None, :], packed1[:, None], packed0[:, None])

        # Decode: (BLOCK_M, 16)
        nibbles = (packed_sel >> shifts[None, :]) & 0xF
        w_tile = (nibbles.to(tl.float32) * 0.5 - 3.0).to(tl.float16)

        acc += tl.dot(x_tile, tl.trans(w_tile))

    scaled = acc * Wscale
    addrs = b_range[:, None] * stride_y_b + m_range[None, :]
    tl.atomic_add(y_ptr + addrs, scaled, mask=b_mask[:, None] & m_mask[None, :])


# ────────────────────────────────────────────────────────────────
# Benchmark
# ────────────────────────────────────────────────────────────────

def main():
    from glq.codebook import E8ShellCodebook
    from glq.inference_kernel import glq_dequant_matmul

    print(f"GPU: {torch.cuda.get_device_name()}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_memory / 1e9:.1f} GB, SMs: {props.multi_processor_count}")
    print()

    codebook = E8ShellCodebook.build(device="cuda", verbose=False)
    K = codebook.codebook.shape[0]
    cb_half = codebook.codebook_half.to("cuda")
    cb_packed = codebook.codebook_packed.to("cuda")
    Wscale = 0.01

    shapes = [
        (3072, 3072),
        (9216, 3072),
        (3072, 9216),
    ]

    BPS = 64

    # ================================================================
    # Part 1: Packed B=1 matvec
    # ================================================================
    print("=" * 80)
    print("Part 1: Packed vs fp16 split-K matvec (B=1)")
    print("=" * 80)

    for M, N in shapes:
        m_pad = 1 << (M - 1).bit_length()
        n_pad = 1 << (N - 1).bit_length()
        N_BLOCKS = n_pad // 8
        Qidxs = torch.randint(0, K, (m_pad, N_BLOCKS),
                               device="cuda", dtype=torch.int32).to(torch.int16)
        x = torch.randn(1, n_pad, device="cuda", dtype=torch.float16)

        # Current production kernel (auto-dispatched, includes split-K)
        current_us = cuda_median_us(
            lambda: glq_dequant_matmul(x, Qidxs, cb_half, Wscale))

        # Dense cuBLAS baseline
        W_dense = torch.randn(M, N, device="cuda", dtype=torch.float16)
        x_raw = torch.randn(1, N, device="cuda", dtype=torch.float16)
        dense_us = cuda_median_us(lambda: x_raw @ W_dense.T)

        # Packed v2 (vectorized decode)
        n_splits = triton.cdiv(N_BLOCKS, BPS)
        block_m = 64

        def run_packed_v2():
            y = torch.zeros(m_pad, dtype=torch.float32, device="cuda")
            grid = (triton.cdiv(m_pad, block_m), n_splits)
            _packed_splitk_matvec_v2_kernel[grid](
                x[0], Qidxs, cb_packed, y,
                m_pad, N_BLOCKS, Wscale,
                Qidxs.stride(0), Qidxs.stride(1),
                BLOCK_M=block_m, BLOCKS_PER_SPLIT=BPS,
            )
            return y

        packed_v2_us = cuda_median_us(run_packed_v2)

        # Correctness check
        y_ref = glq_dequant_matmul(x, Qidxs, cb_half, Wscale)
        y_packed = run_packed_v2()
        err = (y_ref[0, :m_pad] - y_packed).abs().max().item()
        rel = err / y_ref.abs().mean().item()

        print(f"  {M}x{N}  current={current_us:7.1f}us  packed={packed_v2_us:7.1f}us  "
              f"dense={dense_us:7.1f}us  "
              f"speedup={current_us/packed_v2_us:.2f}x  "
              f"gap={packed_v2_us/dense_us:.2f}x  "
              f"err={rel:.6f}")

    # ================================================================
    # Part 2: Packed B=8 TC matmul
    # ================================================================
    print()
    print("=" * 80)
    print("Part 2: Packed vs fp16 split-K TC matmul (B=8, B=32)")
    print("=" * 80)

    for M, N in shapes:
        m_pad = 1 << (M - 1).bit_length()
        n_pad = 1 << (N - 1).bit_length()
        N_BLOCKS = n_pad // 8
        Qidxs = torch.randint(0, K, (m_pad, N_BLOCKS),
                               device="cuda", dtype=torch.int32).to(torch.int16)

        W_dense = torch.randn(M, N, device="cuda", dtype=torch.float16)

        bps_even = BPS if BPS % 2 == 0 else BPS - 1
        n_splits = triton.cdiv(N_BLOCKS, bps_even)

        for B in [8, 32]:
            x = torch.randn(B, n_pad, device="cuda", dtype=torch.float16)
            x_raw = torch.randn(B, N, device="cuda", dtype=torch.float16)

            current_us = cuda_median_us(
                lambda: glq_dequant_matmul(x, Qidxs, cb_half, Wscale))
            dense_us = cuda_median_us(lambda: x_raw @ W_dense.T)

            block_b, block_m = 32, 64

            def run_packed_tc():
                y = torch.zeros(B, m_pad, dtype=torch.float32, device="cuda")
                grid = (triton.cdiv(B, block_b),
                        triton.cdiv(m_pad, block_m), n_splits)
                _packed_splitk_matmul_tc_kernel[grid](
                    x, Qidxs, cb_packed, y,
                    B, m_pad, n_pad, N_BLOCKS, Wscale,
                    x.stride(0), Qidxs.stride(0), Qidxs.stride(1), y.stride(0),
                    BLOCK_B=block_b, BLOCK_M=block_m,
                    BLOCKS_PER_SPLIT=bps_even,
                )
                return y

            packed_tc_us = cuda_median_us(run_packed_tc)

            # Correctness
            y_ref = glq_dequant_matmul(x, Qidxs, cb_half, Wscale)
            y_packed = run_packed_tc()
            err = (y_ref[:, :m_pad] - y_packed).abs().max().item()
            rel = err / y_ref.abs().mean().item()

            print(f"  {M}x{N} B={B:2d}  current={current_us:7.1f}us  "
                  f"packed={packed_tc_us:7.1f}us  dense={dense_us:7.1f}us  "
                  f"speedup={current_us/packed_tc_us:.2f}x  "
                  f"gap={packed_tc_us/dense_us:.2f}x  "
                  f"err={rel:.6f}")


if __name__ == "__main__":
    main()
