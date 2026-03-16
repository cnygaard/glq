"""Benchmark split-K and 2D-tiled kernel variants.

The current kernels under-utilize the L40S's 142 SMs because the grid is too
small (32-128 CTAs for typical layer shapes). Split-K distributes the N
reduction across more CTAs, saturating the GPU.

Usage:
    python benchmarks/bench_splitk.py
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
# Split-K fused matvec (B=1)
# ────────────────────────────────────────────────────────────────

@triton.jit
def _splitk_matvec_kernel(
    x_ptr, qidxs_ptr, codebook_ptr, y_ptr,
    M, N_BLOCKS, Wscale,
    stride_q_m, stride_q_k, stride_cb_k,
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

        x_vec = tl.load(x_ptr + j * 8 + d_range, mask=j_mask, other=0.0).to(tl.float32)

        indices = tl.load(
            qidxs_ptr + m_range * stride_q_m + j * stride_q_k,
            mask=m_mask & j_mask, other=0,
        )
        indices = (indices.to(tl.int32) & 0xFFFF)

        cb_vecs = tl.load(
            codebook_ptr + indices[:, None] * stride_cb_k + d_range[None, :],
            mask=m_mask[:, None] & j_mask, other=0.0,
        ).to(tl.float32)

        acc += tl.sum(cb_vecs * x_vec[None, :], axis=1)

    tl.atomic_add(y_ptr + m_range, acc * Wscale, mask=m_mask)


def splitk_matvec(x, Qidxs, codebook, Wscale, block_m=64, blocks_per_split=64):
    """Split-K fused dequant+matvec."""
    B, N = x.shape
    assert B == 1
    M, N_BLOCKS = Qidxs.shape

    x_fp32 = x[0].float().contiguous()
    cb = codebook.half().contiguous()
    y = torch.zeros(M, dtype=torch.float32, device=x.device)

    n_splits = (N_BLOCKS + blocks_per_split - 1) // blocks_per_split
    grid = (triton.cdiv(M, block_m), n_splits)

    _splitk_matvec_kernel[grid](
        x_fp32, Qidxs, cb, y,
        M, N_BLOCKS, Wscale,
        Qidxs.stride(0), Qidxs.stride(1), cb.stride(0),
        BLOCK_M=block_m,
        BLOCKS_PER_SPLIT=blocks_per_split,
    )
    return y.unsqueeze(0)


# ────────────────────────────────────────────────────────────────
# Split-K fused matmul with Tensor Cores (B>=2)
# ────────────────────────────────────────────────────────────────

@triton.jit
def _splitk_matmul_tc_kernel(
    x_ptr, qidxs_ptr, codebook_ptr, y_ptr,
    B, M, N, N_BLOCKS, Wscale,
    stride_x_b, stride_q_m, stride_q_k, stride_cb_k, stride_y_b,
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

    # Process pairs of blocks for K=16 Tensor Core tiles
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

        acc += tl.dot(x_tile, tl.trans(w_tile))

    # Atomic add partial results (2D)
    scaled = acc * Wscale
    addrs = b_range[:, None] * stride_y_b + m_range[None, :]
    tl.atomic_add(y_ptr + addrs, scaled, mask=b_mask[:, None] & m_mask[None, :])


def splitk_matmul(x, Qidxs, codebook, Wscale,
                  block_b=32, block_m=64, blocks_per_split=64):
    """Split-K fused dequant+matmul with Tensor Cores."""
    B, N = x.shape
    M, N_BLOCKS = Qidxs.shape

    x_fp16 = x.half().contiguous()
    cb = codebook.half().contiguous()
    y = torch.zeros(B, M, dtype=torch.float32, device=x.device)

    # Ensure blocks_per_split is even (for K=16 TC pairing)
    blocks_per_split = max(2, blocks_per_split - (blocks_per_split % 2))
    n_splits = (N_BLOCKS + blocks_per_split - 1) // blocks_per_split

    grid = (triton.cdiv(B, block_b), triton.cdiv(M, block_m), n_splits)

    _splitk_matmul_tc_kernel[grid](
        x_fp16, Qidxs, cb, y,
        B, M, N, N_BLOCKS, Wscale,
        x_fp16.stride(0), Qidxs.stride(0), Qidxs.stride(1),
        cb.stride(0), y.stride(0),
        BLOCK_B=block_b, BLOCK_M=block_m,
        BLOCKS_PER_SPLIT=blocks_per_split,
    )
    return y


# ────────────────────────────────────────────────────────────────
# 2D-tiled dequant kernel (for dequant-then-cuBLAS approach)
# ────────────────────────────────────────────────────────────────

@triton.jit
def _dequant_2d_kernel(
    qidxs_ptr, codebook_ptr, out_ptr,
    M, N_BLOCKS, Wscale,
    stride_q_m, stride_q_k, stride_cb_k, stride_out_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_range = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_range < M
    d_range = tl.arange(0, 8)
    scale = Wscale.to(tl.float32)

    for j_off in tl.static_range(BLOCK_N):
        j = n_start + j_off
        j_mask = j < N_BLOCKS

        indices = tl.load(
            qidxs_ptr + m_range * stride_q_m + j * stride_q_k,
            mask=m_mask & j_mask, other=0,
        )
        indices = (indices.to(tl.int32) & 0xFFFF)

        cb_vecs = tl.load(
            codebook_ptr + indices[:, None] * stride_cb_k + d_range[None, :],
            mask=m_mask[:, None] & j_mask, other=0.0,
        ).to(tl.float32)

        scaled = (cb_vecs * scale).to(tl.float16)
        tl.store(
            out_ptr + m_range[:, None] * stride_out_m + j * 8 + d_range[None, :],
            scaled,
            mask=m_mask[:, None] & j_mask,
        )


def dequant_2d(Qidxs, codebook_half, Wscale, block_m=64, block_n=64):
    """2D-tiled dequant to maximize SM utilization."""
    M, N_BLOCKS = Qidxs.shape
    N = N_BLOCKS * 8
    W = torch.empty(M, N, dtype=torch.float16, device=Qidxs.device)
    cb = codebook_half.contiguous()

    grid = (triton.cdiv(M, block_m), triton.cdiv(N_BLOCKS, block_n))
    _dequant_2d_kernel[grid](
        Qidxs, cb, W, M, N_BLOCKS, Wscale,
        Qidxs.stride(0), Qidxs.stride(1), cb.stride(0), W.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n,
    )
    return W


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
    Wscale = 0.01

    shapes = [
        (3072, 3072),
        (9216, 3072),
        (3072, 9216),
    ]

    # ================================================================
    # Test 1: Split-K matvec (B=1)
    # ================================================================
    print("=" * 90)
    print("Test 1: Split-K matvec (B=1) — varying blocks_per_split")
    print("=" * 90)

    for M, N in shapes:
        m_pad = 1 << (M - 1).bit_length()
        n_pad = 1 << (N - 1).bit_length()
        N_BLOCKS = n_pad // 8
        Qidxs = torch.randint(0, K, (m_pad, N_BLOCKS),
                               device="cuda", dtype=torch.int32).to(torch.int16)
        x = torch.randn(1, n_pad, device="cuda", dtype=torch.float16)

        # Baseline: current fused kernel
        baseline_us = cuda_median_us(
            lambda: glq_dequant_matmul(x, Qidxs, cb_half, Wscale))

        print(f"\n  {M}x{N} (padded {m_pad}x{n_pad}, N_BLOCKS={N_BLOCKS})")
        print(f"  Baseline fused: {baseline_us:.1f}us")
        print(f"  {'bps':>5s} {'CTAs':>6s}  {'Time':>10s}  {'vs base':>8s}")
        print(f"  " + "-" * 36)

        for bps in [32, 64, 128, 256]:
            if bps > N_BLOCKS:
                continue
            n_splits = (N_BLOCKS + bps - 1) // bps
            grid_m = triton.cdiv(m_pad, 64)
            total_ctas = grid_m * n_splits

            us = cuda_median_us(
                lambda bps=bps: splitk_matvec(x, Qidxs, cb_half, Wscale,
                                              block_m=64, blocks_per_split=bps))
            ratio = us / baseline_us
            print(f"  {bps:5d} {total_ctas:6d}  {us:9.1f}us  {ratio:7.2f}x")

    # ================================================================
    # Test 2: Split-K TC matmul (B=8, B=32)
    # ================================================================
    print()
    print("=" * 90)
    print("Test 2: Split-K TC matmul (B=8, B=32)")
    print("=" * 90)

    for M, N in shapes:
        m_pad = 1 << (M - 1).bit_length()
        n_pad = 1 << (N - 1).bit_length()
        N_BLOCKS = n_pad // 8
        Qidxs = torch.randint(0, K, (m_pad, N_BLOCKS),
                               device="cuda", dtype=torch.int32).to(torch.int16)

        for B in [8, 32]:
            x = torch.randn(B, n_pad, device="cuda", dtype=torch.float16)

            baseline_us = cuda_median_us(
                lambda: glq_dequant_matmul(x, Qidxs, cb_half, Wscale))

            print(f"\n  {M}x{N} B={B} (N_BLOCKS={N_BLOCKS})")
            print(f"  Baseline fused: {baseline_us:.1f}us")
            print(f"  {'bps':>5s} {'CTAs':>6s}  {'Time':>10s}  {'vs base':>8s}")
            print(f"  " + "-" * 36)

            for bps in [32, 64, 128]:
                if bps > N_BLOCKS:
                    continue
                # Ensure even for TC pairing
                bps_even = max(2, bps - (bps % 2))
                n_splits = (N_BLOCKS + bps_even - 1) // bps_even

                us = cuda_median_us(
                    lambda bps=bps_even: splitk_matmul(
                        x, Qidxs, cb_half, Wscale,
                        block_b=32, block_m=64, blocks_per_split=bps))
                ratio = us / baseline_us
                grid_size = triton.cdiv(B, 32) * triton.cdiv(m_pad, 64) * n_splits
                print(f"  {bps_even:5d} {grid_size:6d}  {us:9.1f}us  {ratio:7.2f}x")

    # ================================================================
    # Test 3: 2D dequant + cuBLAS vs fused
    # ================================================================
    print()
    print("=" * 90)
    print("Test 3: 2D dequant + cuBLAS vs baseline fused")
    print("=" * 90)

    for M, N in shapes:
        m_pad = 1 << (M - 1).bit_length()
        n_pad = 1 << (N - 1).bit_length()
        N_BLOCKS = n_pad // 8
        Qidxs = torch.randint(0, K, (m_pad, N_BLOCKS),
                               device="cuda", dtype=torch.int32).to(torch.int16)

        # Dequant timing at various block sizes
        print(f"\n  {M}x{N} — 2D dequant timing")
        print(f"  {'blk_m':>5s} {'blk_n':>5s} {'CTAs':>6s}  {'Time':>10s}")
        print(f"  " + "-" * 32)

        best_dequant_us = float('inf')
        for bm in [32, 64, 128]:
            for bn in [32, 64, 128]:
                grid_size = triton.cdiv(m_pad, bm) * triton.cdiv(N_BLOCKS, bn)
                us = cuda_median_us(
                    lambda bm=bm, bn=bn: dequant_2d(
                        Qidxs, cb_half, Wscale, block_m=bm, block_n=bn))
                best_dequant_us = min(best_dequant_us, us)
                print(f"  {bm:5d} {bn:5d} {grid_size:6d}  {us:9.1f}us")

        # Compare full pipeline: best 2D dequant + cuBLAS vs fused
        W_deq = dequant_2d(Qidxs, cb_half, Wscale)
        W_crop = W_deq[:M, :N].contiguous()
        W_dense = torch.randn(M, N, device="cuda", dtype=torch.float16)

        print(f"\n  Full pipeline (best dequant={best_dequant_us:.1f}us):")
        print(f"  {'B':>3s}  {'Fused':>10s}  {'2D+cuB':>10s}  {'Dense':>10s}  "
              f"{'Fused/D':>8s}  {'2D+C/D':>8s}")
        print(f"  " + "-" * 56)

        for B in [1, 8, 32, 64]:
            x_pad = torch.randn(B, n_pad, device="cuda", dtype=torch.float16)
            x_raw = torch.randn(B, N, device="cuda", dtype=torch.float16)

            fused_us = cuda_median_us(
                lambda: glq_dequant_matmul(x_pad, Qidxs, cb_half, Wscale))
            cublas_us = cuda_median_us(lambda: x_raw @ W_crop.T)
            dense_us = cuda_median_us(lambda: x_raw @ W_dense.T)

            total_2d = best_dequant_us + cublas_us

            print(f"  {B:3d}  {fused_us:9.1f}us  {total_2d:9.1f}us  {dense_us:9.1f}us  "
                  f"{fused_us/dense_us:7.2f}x  {total_2d/dense_us:7.2f}x")

    # ================================================================
    # Test 4: Correctness check
    # ================================================================
    print()
    print("=" * 90)
    print("Test 4: Correctness verification")
    print("=" * 90)

    M, N = 3072, 3072
    m_pad = 1 << (M - 1).bit_length()
    n_pad = 1 << (N - 1).bit_length()
    N_BLOCKS = n_pad // 8
    Qidxs = torch.randint(0, K, (m_pad, N_BLOCKS),
                           device="cuda", dtype=torch.int32).to(torch.int16)

    for B in [1, 8]:
        x = torch.randn(B, n_pad, device="cuda", dtype=torch.float16)

        y_baseline = glq_dequant_matmul(x, Qidxs, cb_half, Wscale)

        if B == 1:
            for bps in [64, 128]:
                y_splitk = splitk_matvec(x, Qidxs, cb_half, Wscale,
                                         blocks_per_split=bps)
                err = (y_baseline - y_splitk).abs().max().item()
                rel = err / y_baseline.abs().mean().item()
                print(f"  B={B} splitk bps={bps}: max_err={err:.6f}, "
                      f"rel_err={rel:.6f} {'OK' if rel < 0.01 else 'FAIL'}")
        else:
            for bps in [64, 128]:
                y_splitk = splitk_matmul(x, Qidxs, cb_half, Wscale,
                                         blocks_per_split=bps)
                err = (y_baseline - y_splitk).abs().max().item()
                rel = err / y_baseline.abs().mean().item()
                print(f"  B={B} splitk_tc bps={bps}: max_err={err:.6f}, "
                      f"rel_err={rel:.6f} {'OK' if rel < 0.01 else 'FAIL'}")

        # 2D dequant correctness
        W_2d = dequant_2d(Qidxs, cb_half, Wscale)
        y_2d = (x.half() @ W_2d.T).float()
        # Note: y_baseline includes Wscale in kernel, W_2d already has Wscale baked in
        err = (y_baseline - y_2d).abs().max().item()
        rel = err / y_baseline.abs().mean().item()
        print(f"  B={B} 2d_dequant+mm: max_err={err:.6f}, "
              f"rel_err={rel:.6f} {'OK' if rel < 0.01 else 'FAIL'}")


if __name__ == "__main__":
    main()
