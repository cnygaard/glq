"""Compare kernel strategies: fused dequant+matmul vs dequant-then-cuBLAS.

The fused Triton kernel interleaves random codebook gathers with Tensor Core
matmuls. This benchmark tests whether separating dequant (gather → contiguous
buffer) + cuBLAS (contiguous matmul) is faster, since cuBLAS is heavily
optimized for contiguous memory access patterns.

Usage:
    python benchmarks/bench_kernel_variants.py
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _triton_available = True
except ImportError:
    _triton_available = False


# ---- Timing utility ----

def cuda_median_us(fn, n_warmup=20, n_iter=200):
    """Time a function using CUDA events, return median in microseconds."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms -> us

    times.sort()
    return times[len(times) // 2]


# ---- Triton dequant-only kernel ----

if _triton_available:

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=4),
        ],
        key=['M', 'N_BLOCKS'],
    )
    @triton.jit
    def _dequant_only_kernel(
        qidxs_ptr,
        codebook_ptr,
        out_ptr,
        M,
        N_BLOCKS,
        stride_q_m,
        stride_q_k,
        stride_cb_k,
        stride_out_m,
        BLOCK_M: tl.constexpr,
    ):
        """Dequant codebook indices to contiguous fp16 weight buffer."""
        pid = tl.program_id(0)
        m_start = pid * BLOCK_M
        m_range = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_range < M

        d_range = tl.arange(0, 8)

        for j in range(N_BLOCKS):
            indices = tl.load(
                qidxs_ptr + m_range * stride_q_m + j * stride_q_k,
                mask=m_mask, other=0,
            )
            indices = (indices.to(tl.int32) & 0xFFFF)

            cb_vecs = tl.load(
                codebook_ptr + indices[:, None] * stride_cb_k + d_range[None, :],
                mask=m_mask[:, None], other=0.0,
            )

            tl.store(
                out_ptr + m_range[:, None] * stride_out_m + j * 8 + d_range[None, :],
                cb_vecs,
                mask=m_mask[:, None],
            )

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=4),
        ],
        key=['M', 'N_BLOCKS'],
    )
    @triton.jit
    def _dequant_scale_kernel(
        qidxs_ptr,
        codebook_ptr,
        out_ptr,
        Wscale,
        M,
        N_BLOCKS,
        stride_q_m,
        stride_q_k,
        stride_cb_k,
        stride_out_m,
        BLOCK_M: tl.constexpr,
    ):
        """Dequant + scale to contiguous fp16 weight buffer."""
        pid = tl.program_id(0)
        m_start = pid * BLOCK_M
        m_range = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_range < M

        d_range = tl.arange(0, 8)
        scale = Wscale.to(tl.float32)

        for j in range(N_BLOCKS):
            indices = tl.load(
                qidxs_ptr + m_range * stride_q_m + j * stride_q_k,
                mask=m_mask, other=0,
            )
            indices = (indices.to(tl.int32) & 0xFFFF)

            cb_vecs = tl.load(
                codebook_ptr + indices[:, None] * stride_cb_k + d_range[None, :],
                mask=m_mask[:, None], other=0.0,
            ).to(tl.float32)

            scaled = (cb_vecs * scale).to(tl.float16)
            tl.store(
                out_ptr + m_range[:, None] * stride_out_m + j * 8 + d_range[None, :],
                scaled,
                mask=m_mask[:, None],
            )


def triton_dequant(Qidxs, codebook_half, Wscale=None):
    """Triton dequant: indices → contiguous fp16 weight matrix."""
    M, N_BLOCKS = Qidxs.shape
    N = N_BLOCKS * 8
    W = torch.empty(M, N, dtype=torch.float16, device=Qidxs.device)

    cb = codebook_half.contiguous()

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)

    if Wscale is not None:
        _dequant_scale_kernel[grid](
            Qidxs, cb, W, Wscale,
            M, N_BLOCKS,
            Qidxs.stride(0), Qidxs.stride(1),
            cb.stride(0), W.stride(0),
        )
    else:
        _dequant_only_kernel[grid](
            Qidxs, cb, W,
            M, N_BLOCKS,
            Qidxs.stride(0), Qidxs.stride(1),
            cb.stride(0), W.stride(0),
        )
    return W


# ---- PyTorch dequant (index_select) ----

def pytorch_dequant(Qidxs, codebook_half, Wscale=None):
    """PyTorch dequant: indices → contiguous fp16 weight matrix."""
    M, N_BLOCKS = Qidxs.shape
    N = N_BLOCKS * 8
    idx = (Qidxs.int() & 0xFFFF).reshape(-1)
    W = codebook_half[idx].reshape(M, N)
    if Wscale is not None:
        W = W * Wscale
    return W


# ---- Benchmark ----

def main():
    from glq.codebook import E8ShellCodebook
    from glq.inference_kernel import glq_dequant_matmul

    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    # ================================================================
    # Part 1: Dequant-only timing (no matmul)
    # ================================================================
    print("=" * 80)
    print("Part 1: Dequant-only timing (codebook gather → contiguous fp16)")
    print("=" * 80)
    print(f"{'Shape':<14s} {'PyTorch':>10s} {'Triton':>10s} {'Triton+Scale':>14s}")
    print("-" * 52)

    for M, N in shapes:
        m_pad = 1 << (M - 1).bit_length()
        n_pad = 1 << (N - 1).bit_length()
        Qidxs = torch.randint(0, K, (m_pad, n_pad // 8),
                               device="cuda", dtype=torch.int32).to(torch.int16)

        pytorch_us = cuda_median_us(lambda: pytorch_dequant(Qidxs, cb_half))
        triton_us = cuda_median_us(lambda: triton_dequant(Qidxs, cb_half))
        triton_s_us = cuda_median_us(lambda: triton_dequant(Qidxs, cb_half, Wscale))

        print(f"{M}x{N:<7d} {pytorch_us:9.1f}us {triton_us:9.1f}us {triton_s_us:13.1f}us")

    # ================================================================
    # Part 2: Full pipeline comparison at each batch size
    # ================================================================
    print()
    print("=" * 80)
    print("Part 2: Full pipeline comparison")
    print("  [A] Fused Triton (current glq_dequant_matmul)")
    print("  [B] Triton dequant + cuBLAS fp16 matmul")
    print("  [C] PyTorch dequant + cuBLAS fp16 matmul")
    print("  [D] cuBLAS fp16 matmul (dense baseline)")
    print("=" * 80)

    for M, N in shapes:
        m_pad = 1 << (M - 1).bit_length()
        n_pad = 1 << (N - 1).bit_length()
        Qidxs = torch.randint(0, K, (m_pad, n_pad // 8),
                               device="cuda", dtype=torch.int32).to(torch.int16)
        W_dense = torch.randn(M, N, device="cuda", dtype=torch.float16)

        print(f"\n--- {M}x{N} ---")
        print(f"  {'B':>3s}  {'Fused':>10s}  {'Tri+cuB':>10s}  {'Py+cuB':>10s}  {'Dense':>10s}  "
              f"{'Fused/D':>8s}  {'T+C/D':>8s}  {'P+C/D':>8s}")
        print("  " + "-" * 80)

        for B in batch_sizes:
            x_pad = torch.randn(B, n_pad, device="cuda", dtype=torch.float16)
            x_raw = torch.randn(B, N, device="cuda", dtype=torch.float16)

            # [A] Fused Triton
            fused_us = cuda_median_us(
                lambda: glq_dequant_matmul(x_pad, Qidxs, cb_half, Wscale))

            # Pre-dequant for pipeline variants (measure dequant + matmul separately)
            # [B] Triton dequant + cuBLAS
            def triton_pipeline():
                W = triton_dequant(Qidxs, cb_half, Wscale)
                return x_raw.half() @ W[:M, :N].T
            triton_pipe_us = cuda_median_us(triton_pipeline)

            # [C] PyTorch dequant + cuBLAS
            def pytorch_pipeline():
                W = pytorch_dequant(Qidxs, cb_half, Wscale)
                return x_raw.half() @ W[:M, :N].T
            pytorch_pipe_us = cuda_median_us(pytorch_pipeline)

            # [D] Dense cuBLAS
            dense_us = cuda_median_us(lambda: x_raw @ W_dense.T)

            print(f"  {B:3d}  {fused_us:9.1f}us  {triton_pipe_us:9.1f}us  "
                  f"{pytorch_pipe_us:9.1f}us  {dense_us:9.1f}us  "
                  f"{fused_us/dense_us:7.2f}x  "
                  f"{triton_pipe_us/dense_us:7.2f}x  "
                  f"{pytorch_pipe_us/dense_us:7.2f}x")

    # ================================================================
    # Part 3: Dequant amortization — reuse dequanted weights across B
    # ================================================================
    print()
    print("=" * 80)
    print("Part 3: Dequant amortization (dequant once, matmul at varying B)")
    print("  Simulates: per-layer dequant cost is fixed, matmul scales with B")
    print("=" * 80)

    for M, N in shapes:
        m_pad = 1 << (M - 1).bit_length()
        n_pad = 1 << (N - 1).bit_length()
        Qidxs = torch.randint(0, K, (m_pad, n_pad // 8),
                               device="cuda", dtype=torch.int32).to(torch.int16)
        W_dense = torch.randn(M, N, device="cuda", dtype=torch.float16)

        # Measure dequant cost once
        dequant_us = cuda_median_us(
            lambda: triton_dequant(Qidxs, cb_half, Wscale))

        # Pre-dequant
        W_deq = triton_dequant(Qidxs, cb_half, Wscale)
        W_deq_crop = W_deq[:M, :N].contiguous()

        print(f"\n--- {M}x{N} (dequant={dequant_us:.1f}us) ---")
        print(f"  {'B':>3s}  {'cuBLAS(deq)':>12s}  {'deq+cuBLAS':>12s}  "
              f"{'Fused':>10s}  {'Dense':>10s}  {'d+c/Dense':>10s}")
        print("  " + "-" * 72)

        for B in batch_sizes:
            x_pad = torch.randn(B, n_pad, device="cuda", dtype=torch.float16)
            x_raw = torch.randn(B, N, device="cuda", dtype=torch.float16)

            # cuBLAS on pre-dequanted weights
            cublas_deq_us = cuda_median_us(lambda: x_raw @ W_deq_crop.T)

            # Total: dequant + cuBLAS
            total_us = dequant_us + cublas_deq_us

            # Fused for comparison
            fused_us = cuda_median_us(
                lambda: glq_dequant_matmul(x_pad, Qidxs, cb_half, Wscale))

            # Dense baseline
            dense_us = cuda_median_us(lambda: x_raw @ W_dense.T)

            print(f"  {B:3d}  {cublas_deq_us:11.1f}us  {total_us:11.1f}us  "
                  f"{fused_us:9.1f}us  {dense_us:9.1f}us  "
                  f"{total_us/dense_us:9.2f}x")


if __name__ == "__main__":
    main()
