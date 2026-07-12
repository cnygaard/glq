/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * E8P (QuIP#) tensor-core decode kernels.
 *
 * Derived from QuIP# (https://github.com/Cornell-RelaxML/quip-sharp), licensed
 * under the GNU General Public License v3 — Copyright (c) the QuIP# authors
 * (Cornell-RelaxML). Ported from quiptools/quiptools_e8p_gemv.cu and
 * quiptools/quiptools.cu, then modified for the GLQ project (2026): host
 * functions renamed glq_* and bound in glq_bindings.cpp, and assembled into
 * this self-contained translation unit. Distributed under GPL-3.0, the same
 * license as the upstream and as this project. Full QuIP# citation in the
 * project README.
 *
 * Technical: a 1KB-L1 sign-symmetric codebook is bit-expanded (lop3 +
 * add.f16x2 + parity/sign XOR + ±1/4) and the dot runs on tensor cores
 * (mma.sync.m16n8k16). The E81B (1-bit residual) lookup-matmul and dense
 * decompress kernels below are ported from quiptools/quiptools.cu.
 */
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/types.h>
#include <torch/extension.h>

#ifndef FULL_MASK
#define FULL_MASK 0xffffffff
#endif

using namespace nvcuda;                 // wmma fragments for the e81b (1-bit residual) lookup-matmul
#define E8P_DECOMPRESS_E81B_BLOCK_SIZE 4

#define E8P_CHECK_CUDA(x)       TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define E8P_CHECK_CONTIG(x)     TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define E8P_CHECK_INPUT(x)      do { E8P_CHECK_CUDA(x); E8P_CHECK_CONTIG(x); } while (false)

__host__ static inline void e8pAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert[%s:%d]: %s\n", file, line, cudaGetErrorString(code));
        exit(code);
    }
}
#define E8P_ERRCHK(ans) do { e8pAssert((ans), __FILE__, __LINE__); } while (false)

__device__ static inline uint32_t e8p_add_as_half2(uint32_t x, uint32_t y) {
    uint32_t z;
    asm("add.f16x2 %0,%1,%2;" : "=r"(z) : "r"(x), "r"(y));
    return z;
}

__device__ static inline uint32_t e8p_mask_lop3(uint32_t x, uint32_t m0, uint32_t m1) {
    uint32_t y;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(y) : "r"(x), "r"(m0), "r"(m1));
    return y;
}

#define E8P_BASE_OFFSET 0xd080d080
#define E8P_XMASK 0x00f000f0
#define E8P_WMASK 0x50085008

// Warps per block that split the K dimension in the batched GEMM. Each warp owns a
// unique scratch plane (k-split) for deterministic reduction; scratch = NW * B * N
// floats, so fewer warps = less scratch traffic at the cost of K parallelism. 8
// matches the shell TC matmul's WARPS and bounds B=32 scratch below the weight traffic.
#define E8P_MM_WARPS 8
// Warps per block for the fast E81B decode kernels (single-linear GEMM: k-split;
// grouped MoE: one warp per m-tile). 8 fills the SM warp slots and keeps per-warp
// Ycache smem (NW * 2KB) + the 4KB shared codebook well under 48KB.
#define E81B_MM_WARPS 8


// Fixed-order reduction over the k-split planes -> deterministic output. One thread per
// (b,m) element sums scratch[0..n_splits-1] in order (local copy of the shell's
// glq_reduce_splits_2d_kernel; kept in this TU since the build has no -rdc=true). Shared
// by the B=1 GEMV (BN=N) and the batched GEMM (BN=B*N).
__global__ static void __launch_bounds__(256) glq_reduce_splits_2d_e8p_kernel(
    float *__restrict__ output, const float *__restrict__ scratch, int BN, int n_splits
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BN) return;
    float sum = 0.0f;
    for (int k = 0; k < n_splits; ++k) sum += scratch[(size_t)k * BN + i];
    output[i] = sum;
}


// B=1 decode GEMV. The 32 warps split K; each warp writes its own scratch plane
// scratch[warpId*N + row] (no atomics → deterministic) and glq_reduce_splits_2d_e8p_kernel
// sums the planes in fixed order (scratch = 32*N floats, tiny for B=1).
__global__ static void glq_decode_matvec_e8p_kernel(
    float *__restrict__ scratch,           // (32, N) fp32 k-split planes
    const uint2 *__restrict__ input,
    const uint2 *__restrict__ weights_compressed,
    const uint32_t *__restrict__ codebook_abs,
    int N, int K
) {
    int warpId = threadIdx.y;
    int laneId = threadIdx.x;

    for (int iin = blockIdx.x; iin < (N >> 4); iin += gridDim.x) {
        float z0 = 0.0, z1 = 0.0, z2 = 0.0, z3 = 0.0;

        for (int iik = warpId; iik < (K >> 6); iik += 32) {
            uint2 w_compr = weights_compressed[laneId + 32 * iik + (K >> 1) * iin];
            uint32_t a = w_compr.x;
            uint32_t b = w_compr.y;

            uint32_t s = b;
            s = s ^ (s >> 4);
            s = s ^ (s >> 8);
            s = s ^ (s >> 16);
            uint32_t sb = (s & 15);
            s = b ^ sb;
            sb = sb | (sb << 16);

            uint32_t input_to_warp = ((const uint32_t *)(&input[16 * iik]))[laneId];
            uint32_t shifted_laneId = (laneId & 3) << 3;

            /// BLOCK 01
            {
            uint32_t x = codebook_abs[(a >> 0) & 255];
            x = x ^ ((s & 0x11111111) * 14);
            uint32_t o = E8P_BASE_OFFSET | ((sb & 0x00010001) << 4);
            uint32_t w00 = e8p_add_as_half2(e8p_mask_lop3(x << 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w01 = e8p_add_as_half2(e8p_mask_lop3(x << 0, E8P_XMASK, E8P_WMASK), o);
            uint32_t w02 = e8p_add_as_half2(e8p_mask_lop3(x >> 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w03 = e8p_add_as_half2(e8p_mask_lop3(x >> 8, E8P_XMASK, E8P_WMASK), o);
            x = codebook_abs[(a >> 8) & 255];
            x = x ^ ((s & 0x22222222) * 7);
            o = E8P_BASE_OFFSET | ((sb & 0x00020002) << 3);
            uint32_t w10 = e8p_add_as_half2(e8p_mask_lop3(x << 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w11 = e8p_add_as_half2(e8p_mask_lop3(x << 0, E8P_XMASK, E8P_WMASK), o);
            uint32_t w12 = e8p_add_as_half2(e8p_mask_lop3(x >> 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w13 = e8p_add_as_half2(e8p_mask_lop3(x >> 8, E8P_XMASK, E8P_WMASK), o);
            uint32_t x_in0 = __shfl_sync(FULL_MASK, input_to_warp, shifted_laneId | 0);
            uint32_t x_in1 = __shfl_sync(FULL_MASK, input_to_warp, shifted_laneId | 1);
            asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                " { %0, %1, %2, %3 }, { %4, %5, %6, %7 }, { %8, %9 }, { %0, %1, %2, %3 };"
                : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
                : "r"(w00), "r"(w10), "r"(w01), "r"(w11), "r"(x_in0), "r"(x_in1));
            x_in0 = __shfl_sync(FULL_MASK, input_to_warp, shifted_laneId | 2);
            x_in1 = __shfl_sync(FULL_MASK, input_to_warp, shifted_laneId | 3);
            asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                " { %0, %1, %2, %3 }, { %4, %5, %6, %7 }, { %8, %9 }, { %0, %1, %2, %3 };"
                : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
                : "r"(w02), "r"(w12), "r"(w03), "r"(w13), "r"(x_in0), "r"(x_in1));
            }
            /// BLOCK 23
            {
            uint32_t x = codebook_abs[(a >> 16) & 255];
            s = s >> 2;
            x = x ^ ((s & 0x11111111) * 14);
            uint32_t o = E8P_BASE_OFFSET | ((sb & 0x00040004) << 2);
            uint32_t w00 = e8p_add_as_half2(e8p_mask_lop3(x << 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w01 = e8p_add_as_half2(e8p_mask_lop3(x << 0, E8P_XMASK, E8P_WMASK), o);
            uint32_t w02 = e8p_add_as_half2(e8p_mask_lop3(x >> 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w03 = e8p_add_as_half2(e8p_mask_lop3(x >> 8, E8P_XMASK, E8P_WMASK), o);
            x = codebook_abs[(a >> 24) & 255];
            x = x ^ ((s & 0x22222222) * 7);
            o = E8P_BASE_OFFSET | ((sb & 0x00080008) << 1);
            uint32_t w10 = e8p_add_as_half2(e8p_mask_lop3(x << 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w11 = e8p_add_as_half2(e8p_mask_lop3(x << 0, E8P_XMASK, E8P_WMASK), o);
            uint32_t w12 = e8p_add_as_half2(e8p_mask_lop3(x >> 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w13 = e8p_add_as_half2(e8p_mask_lop3(x >> 8, E8P_XMASK, E8P_WMASK), o);
            uint32_t x_in0 = __shfl_sync(FULL_MASK, input_to_warp, shifted_laneId | 4);
            uint32_t x_in1 = __shfl_sync(FULL_MASK, input_to_warp, shifted_laneId | 5);
            asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                " { %0, %1, %2, %3 }, { %4, %5, %6, %7 }, { %8, %9 }, { %0, %1, %2, %3 };"
                : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
                : "r"(w00), "r"(w10), "r"(w01), "r"(w11), "r"(x_in0), "r"(x_in1));
            x_in0 = __shfl_sync(FULL_MASK, input_to_warp, shifted_laneId | 6);
            x_in1 = __shfl_sync(FULL_MASK, input_to_warp, shifted_laneId | 7);
            asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                " { %0, %1, %2, %3 }, { %4, %5, %6, %7 }, { %8, %9 }, { %0, %1, %2, %3 };"
                : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
                : "r"(w02), "r"(w12), "r"(w03), "r"(w13), "r"(x_in0), "r"(x_in1));
            }
        }
        // Plain store to this warp's plane (unique slot) — no atomics → deterministic.
        if ((laneId & 1) == 0) {
            scratch[(size_t)warpId * N + (iin << 4) + (laneId >> 1)] = (laneId & 2) ? z2 : z0;
        }
    }
}


torch::Tensor glq_decode_matvec_e8p(
    torch::Tensor x, torch::Tensor weights_compressed, torch::Tensor codebook_abs
) {
    E8P_CHECK_INPUT(x);
    E8P_CHECK_INPUT(weights_compressed);
    E8P_CHECK_INPUT(codebook_abs);
    TORCH_CHECK(x.dim() == 1);
    TORCH_CHECK(weights_compressed.dim() == 4 && weights_compressed.size(3) == 4 &&
                weights_compressed.size(2) == 8);
    TORCH_CHECK(x.scalar_type() == torch::kFloat16);
    TORCH_CHECK(weights_compressed.scalar_type() == torch::kInt64);
    TORCH_CHECK(codebook_abs.scalar_type() == torch::kInt32 && codebook_abs.size(-1) == 256);
    TORCH_CHECK(x.size(-1) == weights_compressed.size(1) << 6);

    int64_t N = weights_compressed.size(0) * 16;
    int64_t K = x.size(-1);
    TORCH_CHECK(K % 64 == 0 && N % 16 == 0 && K < 65536 && N < 65536);

    at::DeviceGuard guard(x.device());
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor output = torch::empty({N}, options);  // reduce overwrites every elem

    static int64_t grid_size = 0;
    if (grid_size == 0) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, x.get_device());
        grid_size = static_cast<int64_t>(deviceProp.multiProcessorCount);
    }
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Deterministic scratch+reduce: 32 k-split planes (one per warp), fixed-order summed.
    size_t scratch_bytes = (size_t)32 * N * sizeof(float);
    float *scratch = (float *)c10::cuda::CUDACachingAllocator::raw_alloc(scratch_bytes);
    glq_decode_matvec_e8p_kernel<<<grid_size, dim3(32, 32), 0, stream>>>(
        scratch,
        (const uint2 *)x.data_ptr<c10::Half>(),
        (const uint2 *)weights_compressed.data_ptr<int64_t>(),
        (const uint32_t *)codebook_abs.data_ptr<int32_t>(),
        N, K);
    E8P_ERRCHK(cudaPeekAtLastError());

    int reduce_threads = 256;
    int reduce_blocks = (int)((N + reduce_threads - 1) / reduce_threads);
    glq_reduce_splits_2d_e8p_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
        output.data_ptr<float>(), scratch, (int)N, 32);
    E8P_ERRCHK(cudaPeekAtLastError());
    c10::cuda::CUDACachingAllocator::raw_delete(scratch);
    return output;
}


// ---- Compressed batched E8P tensor-core GEMM (B>1 decode) ----
// Forks glq_decode_matvec_e8p_kernel. The B=1 GEMV broadcasts one token across all
// 8 mma n-columns and keeps only n=0; here each n-column carries a distinct token, so
// the otherwise-idle n=1..7 lanes do the batch for free. The weight bit-expand path is
// byte-identical to the GEMV; only the B operand, accumulation read-back, and output
// write change. Layout cross-checked against glq_decompress_packed_e8p_kernel:
//   mma-row groupID (w0x) -> natural output row 2*groupID   (harvested from z0/z1)
//   mma-row groupID+8 (w1x) -> natural output row 2*groupID+1 (harvested from z2/z3)
//   mma-k 2t <-> natural-k 64*iik+16t  =>  token's 8 consecutive uint32 at 32*iik+8t+[0..7]
// n-column g == token (base + g); base = 8 * blockIdx.y (one n-tile of <=8 tokens).
// Determinism: the E8P_MM_WARPS warps split K; each warp writes its own scratch plane
// scratch[warpId*B*N + ...] (no atomics, planes disjoint by warpId, rows disjoint by
// block) and glq_reduce_splits_2d_e8p_kernel sums the planes in fixed order.
__global__ static void glq_decode_matmul_e8p_kernel(
    float *__restrict__ scratch,           // (E8P_MM_WARPS, B, N) fp32 k-split planes
    const uint32_t *__restrict__ input,    // (B, K) fp16 viewed as (B, K/2) uint32
    const uint2 *__restrict__ weights_compressed,
    const uint32_t *__restrict__ codebook_abs,
    int N, int K, int B
) {
    int warpId = threadIdx.y;
    int laneId = threadIdx.x;
    int g = laneId >> 2;                    // groupID 0..7  -> feeds n-column g
    int t = laneId & 3;                     // 0..3
    int K2 = K >> 1;                        // uint32 (half2) per token row
    int base = blockIdx.y << 3;             // first token of this n-tile
    int tok_n = base + g;                   // token this lane loads into n-column g
    bool active = tok_n < B;
    const uint32_t *tok = input + (size_t)tok_n * K2;
    size_t kplane = (size_t)warpId * B * N; // this warp's unique scratch plane

    for (int iin = blockIdx.x; iin < (N >> 4); iin += gridDim.x) {
        float z0 = 0.0, z1 = 0.0, z2 = 0.0, z3 = 0.0;

        for (int iik = warpId; iik < (K >> 6); iik += E8P_MM_WARPS) {
            uint2 w_compr = weights_compressed[laneId + 32 * iik + K2 * iin];
            uint32_t a = w_compr.x;
            uint32_t b = w_compr.y;

            uint32_t s = b;
            s = s ^ (s >> 4);
            s = s ^ (s >> 8);
            s = s ^ (s >> 16);
            uint32_t sb = (s & 15);
            s = b ^ sb;
            sb = sb | (sb << 16);

            // 8 consecutive uint32 of this lane's token (mma-k pairs for the 4 mmas)
            const uint32_t *xp = tok + 32 * iik + 8 * t;
            uint32_t xr0 = active ? xp[0] : 0u;
            uint32_t xr1 = active ? xp[1] : 0u;
            uint32_t xr2 = active ? xp[2] : 0u;
            uint32_t xr3 = active ? xp[3] : 0u;
            uint32_t xr4 = active ? xp[4] : 0u;
            uint32_t xr5 = active ? xp[5] : 0u;
            uint32_t xr6 = active ? xp[6] : 0u;
            uint32_t xr7 = active ? xp[7] : 0u;

            /// BLOCK 01
            {
            uint32_t x = codebook_abs[(a >> 0) & 255];
            x = x ^ ((s & 0x11111111) * 14);
            uint32_t o = E8P_BASE_OFFSET | ((sb & 0x00010001) << 4);
            uint32_t w00 = e8p_add_as_half2(e8p_mask_lop3(x << 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w01 = e8p_add_as_half2(e8p_mask_lop3(x << 0, E8P_XMASK, E8P_WMASK), o);
            uint32_t w02 = e8p_add_as_half2(e8p_mask_lop3(x >> 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w03 = e8p_add_as_half2(e8p_mask_lop3(x >> 8, E8P_XMASK, E8P_WMASK), o);
            x = codebook_abs[(a >> 8) & 255];
            x = x ^ ((s & 0x22222222) * 7);
            o = E8P_BASE_OFFSET | ((sb & 0x00020002) << 3);
            uint32_t w10 = e8p_add_as_half2(e8p_mask_lop3(x << 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w11 = e8p_add_as_half2(e8p_mask_lop3(x << 0, E8P_XMASK, E8P_WMASK), o);
            uint32_t w12 = e8p_add_as_half2(e8p_mask_lop3(x >> 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w13 = e8p_add_as_half2(e8p_mask_lop3(x >> 8, E8P_XMASK, E8P_WMASK), o);
            asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                " { %0, %1, %2, %3 }, { %4, %5, %6, %7 }, { %8, %9 }, { %0, %1, %2, %3 };"
                : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
                : "r"(w00), "r"(w10), "r"(w01), "r"(w11), "r"(xr0), "r"(xr1));
            asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                " { %0, %1, %2, %3 }, { %4, %5, %6, %7 }, { %8, %9 }, { %0, %1, %2, %3 };"
                : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
                : "r"(w02), "r"(w12), "r"(w03), "r"(w13), "r"(xr2), "r"(xr3));
            }
            /// BLOCK 23
            {
            uint32_t x = codebook_abs[(a >> 16) & 255];
            s = s >> 2;
            x = x ^ ((s & 0x11111111) * 14);
            uint32_t o = E8P_BASE_OFFSET | ((sb & 0x00040004) << 2);
            uint32_t w00 = e8p_add_as_half2(e8p_mask_lop3(x << 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w01 = e8p_add_as_half2(e8p_mask_lop3(x << 0, E8P_XMASK, E8P_WMASK), o);
            uint32_t w02 = e8p_add_as_half2(e8p_mask_lop3(x >> 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w03 = e8p_add_as_half2(e8p_mask_lop3(x >> 8, E8P_XMASK, E8P_WMASK), o);
            x = codebook_abs[(a >> 24) & 255];
            x = x ^ ((s & 0x22222222) * 7);
            o = E8P_BASE_OFFSET | ((sb & 0x00080008) << 1);
            uint32_t w10 = e8p_add_as_half2(e8p_mask_lop3(x << 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w11 = e8p_add_as_half2(e8p_mask_lop3(x << 0, E8P_XMASK, E8P_WMASK), o);
            uint32_t w12 = e8p_add_as_half2(e8p_mask_lop3(x >> 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w13 = e8p_add_as_half2(e8p_mask_lop3(x >> 8, E8P_XMASK, E8P_WMASK), o);
            asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                " { %0, %1, %2, %3 }, { %4, %5, %6, %7 }, { %8, %9 }, { %0, %1, %2, %3 };"
                : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
                : "r"(w00), "r"(w10), "r"(w01), "r"(w11), "r"(xr4), "r"(xr5));
            asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                " { %0, %1, %2, %3 }, { %4, %5, %6, %7 }, { %8, %9 }, { %0, %1, %2, %3 };"
                : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
                : "r"(w02), "r"(w12), "r"(w03), "r"(w13), "r"(xr6), "r"(xr7));
            }
        }
        // z0=C[g,2t] z1=C[g,2t+1] z2=C[g+8,2t] z3=C[g+8,2t+1]; col n==token, row 2g/2g+1.
        // Plain stores to this warp's plane (unique slot) — no atomics → deterministic.
        int row_e = (iin << 4) + (g << 1);     // mma-row g    -> output feature 2g
        int row_o = row_e + 1;                 // mma-row g+8  -> output feature 2g+1
        int tok0 = base + (t << 1);            // n = 2t   -> z0,z2
        int tok1 = tok0 + 1;                   // n = 2t+1 -> z1,z3
        if (tok0 < B) {
            scratch[kplane + (size_t)tok0 * N + row_e] = z0;
            scratch[kplane + (size_t)tok0 * N + row_o] = z2;
        }
        if (tok1 < B) {
            scratch[kplane + (size_t)tok1 * N + row_e] = z1;
            scratch[kplane + (size_t)tok1 * N + row_o] = z3;
        }
    }
}


torch::Tensor glq_matmul_e8p(
    torch::Tensor x, torch::Tensor weights_compressed, torch::Tensor codebook_abs
) {
    E8P_CHECK_INPUT(x);
    E8P_CHECK_INPUT(weights_compressed);
    E8P_CHECK_INPUT(codebook_abs);
    TORCH_CHECK(x.dim() == 2);
    TORCH_CHECK(weights_compressed.dim() == 4 && weights_compressed.size(3) == 4 &&
                weights_compressed.size(2) == 8);
    TORCH_CHECK(x.scalar_type() == torch::kFloat16);
    TORCH_CHECK(weights_compressed.scalar_type() == torch::kInt64);
    TORCH_CHECK(codebook_abs.scalar_type() == torch::kInt32 && codebook_abs.size(-1) == 256);
    TORCH_CHECK(x.size(-1) == weights_compressed.size(1) << 6);
    x = x.contiguous();

    int64_t B = x.size(0);
    int64_t N = weights_compressed.size(0) * 16;
    int64_t K = x.size(-1);
    TORCH_CHECK(K % 64 == 0 && N % 16 == 0 && K < 65536 && N < 65536);

    at::DeviceGuard guard(x.device());
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor output = torch::empty({B, N}, options);  // reduce overwrites every elem

    static int64_t grid_size = 0;
    if (grid_size == 0) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, x.get_device());
        grid_size = static_cast<int64_t>(deviceProp.multiProcessorCount);
    }
    int64_t n_tiles = (B + 7) >> 3;
    dim3 grid(grid_size, n_tiles);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Deterministic scratch+reduce: E8P_MM_WARPS k-split planes, fixed-order summed.
    size_t scratch_bytes = (size_t)E8P_MM_WARPS * B * N * sizeof(float);
    float *scratch = (float *)c10::cuda::CUDACachingAllocator::raw_alloc(scratch_bytes);
    glq_decode_matmul_e8p_kernel<<<grid, dim3(32, E8P_MM_WARPS), 0, stream>>>(
        scratch,
        (const uint32_t *)x.data_ptr<c10::Half>(),
        (const uint2 *)weights_compressed.data_ptr<int64_t>(),
        (const uint32_t *)codebook_abs.data_ptr<int32_t>(),
        N, K, B);
    E8P_ERRCHK(cudaPeekAtLastError());

    int64_t BN = B * N;
    int reduce_threads = 256;
    int reduce_blocks = (int)((BN + reduce_threads - 1) / reduce_threads);
    glq_reduce_splits_2d_e8p_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
        output.data_ptr<float>(), scratch, (int)BN, E8P_MM_WARPS);
    E8P_ERRCHK(cudaPeekAtLastError());
    c10::cuda::CUDACachingAllocator::raw_delete(scratch);
    return output;
}


// ---- Grouped (per-expert) E8P TC-GEMM for fused MoE serving ----
// Fork of glq_decode_matmul_e8p_kernel with a uniform per-block expert route
// (m_indices, like glq_matmul_tc_grouped_scratch_kernel) + per-expert Wscale
// folded into the scratch store. The bit-expand + mma decode body is byte-
// identical to the single kernel; only the weight base, wscale, and pad-tile
// early-return are added. Input/scratch are pre-grouped over M_sum_max token
// slots (8-token n-tiles; the host's 16-pad makes each tile expert-uniform).
__global__ static void glq_decode_matmul_e8p_grouped_kernel(
    float *__restrict__ scratch,           // (E8P_MM_WARPS, M_sum_max, N) fp32 k-split planes
    const uint32_t *__restrict__ input,    // (M_sum_max, K) fp16 as (M_sum_max, K/2) uint32, GROUPED
    const uint2 *__restrict__ weights_compressed,  // (E, N/16, K/64, 8, 4); per-expert via w_estride
    const uint32_t *__restrict__ codebook_abs,
    const int *__restrict__ m_indices,     // (M_sum_max,) expert per slot, -1 = pad
    const float *__restrict__ wscale_dev,  // (E,) per-expert Wscale
    const float *__restrict__ inv_rs_dev,  // (E,) or null — residual stage folds wscale*inv_rs
    long w_estride,                        // uint2 elements per expert = (N/16)*(K/2)
    int N, int K, int B                    // B = M_sum_max
) {
    int warpId = threadIdx.y;
    int laneId = threadIdx.x;
    int g = laneId >> 2;                    // groupID 0..7 -> feeds n-column g
    int t = laneId & 3;
    int K2 = K >> 1;
    int base = blockIdx.y << 3;             // first token slot of this 8-token n-tile
    int eidx = (base < B) ? m_indices[base] : -1;
    if (eidx < 0) return;                   // fully-pad tile -> skip (output discarded)
    const uint2 *wexp = weights_compressed + (size_t)eidx * w_estride;
    float wscale = wscale_dev[eidx];
    if (inv_rs_dev != nullptr) wscale *= inv_rs_dev[eidx];   // residual stage: wscale * inv_resid_scale

    int tok_n = base + g;
    bool active = tok_n < B;
    const uint32_t *tok = input + (size_t)tok_n * K2;
    size_t kplane = (size_t)warpId * B * N;

    for (int iin = blockIdx.x; iin < (N >> 4); iin += gridDim.x) {
        float z0 = 0.0, z1 = 0.0, z2 = 0.0, z3 = 0.0;

        for (int iik = warpId; iik < (K >> 6); iik += E8P_MM_WARPS) {
            uint2 w_compr = wexp[laneId + 32 * iik + K2 * iin];
            uint32_t a = w_compr.x;
            uint32_t b = w_compr.y;

            uint32_t s = b;
            s = s ^ (s >> 4);
            s = s ^ (s >> 8);
            s = s ^ (s >> 16);
            uint32_t sb = (s & 15);
            s = b ^ sb;
            sb = sb | (sb << 16);

            const uint32_t *xp = tok + 32 * iik + 8 * t;
            uint32_t xr0 = active ? xp[0] : 0u;
            uint32_t xr1 = active ? xp[1] : 0u;
            uint32_t xr2 = active ? xp[2] : 0u;
            uint32_t xr3 = active ? xp[3] : 0u;
            uint32_t xr4 = active ? xp[4] : 0u;
            uint32_t xr5 = active ? xp[5] : 0u;
            uint32_t xr6 = active ? xp[6] : 0u;
            uint32_t xr7 = active ? xp[7] : 0u;

            /// BLOCK 01
            {
            uint32_t x = codebook_abs[(a >> 0) & 255];
            x = x ^ ((s & 0x11111111) * 14);
            uint32_t o = E8P_BASE_OFFSET | ((sb & 0x00010001) << 4);
            uint32_t w00 = e8p_add_as_half2(e8p_mask_lop3(x << 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w01 = e8p_add_as_half2(e8p_mask_lop3(x << 0, E8P_XMASK, E8P_WMASK), o);
            uint32_t w02 = e8p_add_as_half2(e8p_mask_lop3(x >> 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w03 = e8p_add_as_half2(e8p_mask_lop3(x >> 8, E8P_XMASK, E8P_WMASK), o);
            x = codebook_abs[(a >> 8) & 255];
            x = x ^ ((s & 0x22222222) * 7);
            o = E8P_BASE_OFFSET | ((sb & 0x00020002) << 3);
            uint32_t w10 = e8p_add_as_half2(e8p_mask_lop3(x << 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w11 = e8p_add_as_half2(e8p_mask_lop3(x << 0, E8P_XMASK, E8P_WMASK), o);
            uint32_t w12 = e8p_add_as_half2(e8p_mask_lop3(x >> 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w13 = e8p_add_as_half2(e8p_mask_lop3(x >> 8, E8P_XMASK, E8P_WMASK), o);
            asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                " { %0, %1, %2, %3 }, { %4, %5, %6, %7 }, { %8, %9 }, { %0, %1, %2, %3 };"
                : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
                : "r"(w00), "r"(w10), "r"(w01), "r"(w11), "r"(xr0), "r"(xr1));
            asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                " { %0, %1, %2, %3 }, { %4, %5, %6, %7 }, { %8, %9 }, { %0, %1, %2, %3 };"
                : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
                : "r"(w02), "r"(w12), "r"(w03), "r"(w13), "r"(xr2), "r"(xr3));
            }
            /// BLOCK 23
            {
            uint32_t x = codebook_abs[(a >> 16) & 255];
            s = s >> 2;
            x = x ^ ((s & 0x11111111) * 14);
            uint32_t o = E8P_BASE_OFFSET | ((sb & 0x00040004) << 2);
            uint32_t w00 = e8p_add_as_half2(e8p_mask_lop3(x << 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w01 = e8p_add_as_half2(e8p_mask_lop3(x << 0, E8P_XMASK, E8P_WMASK), o);
            uint32_t w02 = e8p_add_as_half2(e8p_mask_lop3(x >> 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w03 = e8p_add_as_half2(e8p_mask_lop3(x >> 8, E8P_XMASK, E8P_WMASK), o);
            x = codebook_abs[(a >> 24) & 255];
            x = x ^ ((s & 0x22222222) * 7);
            o = E8P_BASE_OFFSET | ((sb & 0x00080008) << 1);
            uint32_t w10 = e8p_add_as_half2(e8p_mask_lop3(x << 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w11 = e8p_add_as_half2(e8p_mask_lop3(x << 0, E8P_XMASK, E8P_WMASK), o);
            uint32_t w12 = e8p_add_as_half2(e8p_mask_lop3(x >> 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w13 = e8p_add_as_half2(e8p_mask_lop3(x >> 8, E8P_XMASK, E8P_WMASK), o);
            asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                " { %0, %1, %2, %3 }, { %4, %5, %6, %7 }, { %8, %9 }, { %0, %1, %2, %3 };"
                : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
                : "r"(w00), "r"(w10), "r"(w01), "r"(w11), "r"(xr4), "r"(xr5));
            asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                " { %0, %1, %2, %3 }, { %4, %5, %6, %7 }, { %8, %9 }, { %0, %1, %2, %3 };"
                : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
                : "r"(w02), "r"(w12), "r"(w03), "r"(w13), "r"(xr6), "r"(xr7));
            }
        }
        // col n == token (base + 2t/2t+1), row 2g/2g+1; fold per-expert Wscale.
        int row_e = (iin << 4) + (g << 1);
        int row_o = row_e + 1;
        int tok0 = base + (t << 1);
        int tok1 = tok0 + 1;
        if (tok0 < B) {
            scratch[kplane + (size_t)tok0 * N + row_e] = z0 * wscale;
            scratch[kplane + (size_t)tok0 * N + row_o] = z2 * wscale;
        }
        if (tok1 < B) {
            scratch[kplane + (size_t)tok1 * N + row_e] = z1 * wscale;
            scratch[kplane + (size_t)tok1 * N + row_o] = z3 * wscale;
        }
    }
}


// Accumulating variant of the deterministic reduce: output[i] += sum_k scratch[k,i].
// Lets RVQ stages compose by linearity (stage-0 overwrite, stage-k>=1 accumulate).
__global__ static void __launch_bounds__(256) glq_reduce_splits_2d_e8p_accum_kernel(
    float *__restrict__ output, const float *__restrict__ scratch, int BN, int n_splits
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BN) return;
    float sum = 0.0f;
    for (int k = 0; k < n_splits; ++k) sum += scratch[(size_t)k * BN + i];
    output[i] += sum;
}


// Grouped E8P matmul over a PRE-GROUPED input (no gather), for the fused MoE
// host entry in glq_cuda.cu. Non-static (cross-TU). RVQ stages compose by
// linearity: stage-0 (qb0, wscale) overwrites y_out; E8P residual stages 1-3
// (qb1..qb3, wscale*cumulative-inv_rs) accumulate. Each qb{k}/inv_rs{k} pair is
// null for an absent stage (bpw 4 = qb0+qb1; bpw 6 = +qb2; bpw 8 = +qb3). The
// decode kernel is stage-agnostic, so higher even-bpw is just more accumulate
// passes. (Odd-bpw E81B residual stages are handled separately by the host
// entry via launch_grouped_lookupmatmul_e81b.) Caller passes pre-allocated
// scratch (capture-safe).
void launch_grouped_matmul_e8p(
    float *scratch, float *y_out, const half *x_grouped,
    const uint2 *qb0, const uint2 *qb1, const uint2 *qb2, const uint2 *qb3,
    const uint32_t *codebook_abs,
    int M_sum_max, int N, int K, long w_estride,
    const int *m_indices, const float *wscale_dev,
    const float *inv_rs_dev, const float *inv_rs2_dev, const float *inv_rs3_dev,
    cudaStream_t stream
) {
    dim3 grid((N + 15) / 16, (M_sum_max + 7) / 8);
    dim3 block(32, E8P_MM_WARPS);
    int64_t BN = (int64_t)M_sum_max * N;
    int rblocks = (int)((BN + 255) / 256);

    // stage 0 (primary): wscale only, overwrite.
    glq_decode_matmul_e8p_grouped_kernel<<<grid, block, 0, stream>>>(
        scratch, (const uint32_t *)x_grouped, qb0, codebook_abs, m_indices,
        wscale_dev, nullptr, w_estride, N, K, M_sum_max);
    glq_reduce_splits_2d_e8p_kernel<<<rblocks, 256, 0, stream>>>(
        y_out, scratch, (int)BN, E8P_MM_WARPS);

    // E8P residual stages 1-3: wscale * cumulative inv_resid_scale, accumulate.
    const uint2 *qb[3] = {qb1, qb2, qb3};
    const float *irs[3] = {inv_rs_dev, inv_rs2_dev, inv_rs3_dev};
    for (int s = 0; s < 3; ++s) {
        if (qb[s] == nullptr) continue;
        glq_decode_matmul_e8p_grouped_kernel<<<grid, block, 0, stream>>>(
            scratch, (const uint32_t *)x_grouped, qb[s], codebook_abs, m_indices,
            wscale_dev, irs[s], w_estride, N, K, M_sum_max);
        glq_reduce_splits_2d_e8p_accum_kernel<<<rblocks, 256, 0, stream>>>(
            y_out, scratch, (int)BN, E8P_MM_WARPS);
    }
}


__global__ static void glq_decompress_packed_e8p_kernel(
    uint32_t *__restrict__ output,
    const uint2 *__restrict__ weights_compressed,
    const uint32_t *__restrict__ codebook_abs,
    int N, int K
) {
    int warpId = threadIdx.y;
    int laneId = threadIdx.x;
    for (int iin = blockIdx.x; iin < (N >> 4); iin += gridDim.x) {
        for (int iik = warpId; iik < (K >> 6); iik += 32) {
            uint2 w_compr = weights_compressed[laneId + 32 * iik + (K >> 1) * iin];
            uint32_t a = w_compr.x;
            uint32_t b = w_compr.y;
            uint32_t s = b;
            s = s ^ (s >> 4); s = s ^ (s >> 8); s = s ^ (s >> 16);
            uint32_t sb = (s & 15);
            s = b ^ sb;
            sb = sb | (sb << 16);
            {
            uint32_t x = codebook_abs[(a >> 0) & 255];
            x = x ^ ((s & 0x11111111) * 14);
            uint32_t o = E8P_BASE_OFFSET | ((sb & 0x00010001) << 4);
            uint32_t w00 = e8p_add_as_half2(e8p_mask_lop3(x << 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w01 = e8p_add_as_half2(e8p_mask_lop3(x << 0, E8P_XMASK, E8P_WMASK), o);
            uint32_t w02 = e8p_add_as_half2(e8p_mask_lop3(x >> 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w03 = e8p_add_as_half2(e8p_mask_lop3(x >> 8, E8P_XMASK, E8P_WMASK), o);
            x = codebook_abs[(a >> 8) & 255];
            x = x ^ ((s & 0x22222222) * 7);
            o = E8P_BASE_OFFSET | ((sb & 0x00020002) << 3);
            uint32_t w10 = e8p_add_as_half2(e8p_mask_lop3(x << 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w11 = e8p_add_as_half2(e8p_mask_lop3(x << 0, E8P_XMASK, E8P_WMASK), o);
            uint32_t w12 = e8p_add_as_half2(e8p_mask_lop3(x >> 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w13 = e8p_add_as_half2(e8p_mask_lop3(x >> 8, E8P_XMASK, E8P_WMASK), o);
            output[iin*8*K + (laneId>>2)*K + 0*(K>>1) + iik*32 + 0*4 + ((laneId&3)<<3) + 0] = w00;
            output[iin*8*K + (laneId>>2)*K + 0*(K>>1) + iik*32 + 0*4 + ((laneId&3)<<3) + 1] = w01;
            output[iin*8*K + (laneId>>2)*K + 1*(K>>1) + iik*32 + 0*4 + ((laneId&3)<<3) + 0] = w10;
            output[iin*8*K + (laneId>>2)*K + 1*(K>>1) + iik*32 + 0*4 + ((laneId&3)<<3) + 1] = w11;
            output[iin*8*K + (laneId>>2)*K + 0*(K>>1) + iik*32 + 0*4 + ((laneId&3)<<3) + 2] = w02;
            output[iin*8*K + (laneId>>2)*K + 0*(K>>1) + iik*32 + 0*4 + ((laneId&3)<<3) + 3] = w03;
            output[iin*8*K + (laneId>>2)*K + 1*(K>>1) + iik*32 + 0*4 + ((laneId&3)<<3) + 2] = w12;
            output[iin*8*K + (laneId>>2)*K + 1*(K>>1) + iik*32 + 0*4 + ((laneId&3)<<3) + 3] = w13;
            }
            {
            uint32_t x = codebook_abs[(a >> 16) & 255];
            s = s >> 2;
            x = x ^ ((s & 0x11111111) * 14);
            uint32_t o = E8P_BASE_OFFSET | ((sb & 0x00040004) << 2);
            uint32_t w00 = e8p_add_as_half2(e8p_mask_lop3(x << 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w01 = e8p_add_as_half2(e8p_mask_lop3(x << 0, E8P_XMASK, E8P_WMASK), o);
            uint32_t w02 = e8p_add_as_half2(e8p_mask_lop3(x >> 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w03 = e8p_add_as_half2(e8p_mask_lop3(x >> 8, E8P_XMASK, E8P_WMASK), o);
            x = codebook_abs[(a >> 24) & 255];
            x = x ^ ((s & 0x22222222) * 7);
            o = E8P_BASE_OFFSET | ((sb & 0x00080008) << 1);
            uint32_t w10 = e8p_add_as_half2(e8p_mask_lop3(x << 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w11 = e8p_add_as_half2(e8p_mask_lop3(x << 0, E8P_XMASK, E8P_WMASK), o);
            uint32_t w12 = e8p_add_as_half2(e8p_mask_lop3(x >> 4, E8P_XMASK, E8P_WMASK), o);
            uint32_t w13 = e8p_add_as_half2(e8p_mask_lop3(x >> 8, E8P_XMASK, E8P_WMASK), o);
            output[iin*8*K + (laneId>>2)*K + 0*(K>>1) + iik*32 + 1*4 + ((laneId&3)<<3) + 0] = w00;
            output[iin*8*K + (laneId>>2)*K + 0*(K>>1) + iik*32 + 1*4 + ((laneId&3)<<3) + 1] = w01;
            output[iin*8*K + (laneId>>2)*K + 1*(K>>1) + iik*32 + 1*4 + ((laneId&3)<<3) + 0] = w10;
            output[iin*8*K + (laneId>>2)*K + 1*(K>>1) + iik*32 + 1*4 + ((laneId&3)<<3) + 1] = w11;
            output[iin*8*K + (laneId>>2)*K + 0*(K>>1) + iik*32 + 1*4 + ((laneId&3)<<3) + 2] = w02;
            output[iin*8*K + (laneId>>2)*K + 0*(K>>1) + iik*32 + 1*4 + ((laneId&3)<<3) + 3] = w03;
            output[iin*8*K + (laneId>>2)*K + 1*(K>>1) + iik*32 + 1*4 + ((laneId&3)<<3) + 2] = w12;
            output[iin*8*K + (laneId>>2)*K + 1*(K>>1) + iik*32 + 1*4 + ((laneId&3)<<3) + 3] = w13;
            }
        }
    }
}


torch::Tensor glq_decompress_packed_e8p(torch::Tensor weights_compressed, torch::Tensor codebook_abs) {
    E8P_CHECK_INPUT(weights_compressed);
    E8P_CHECK_INPUT(codebook_abs);
    TORCH_CHECK(weights_compressed.dim() == 4 && weights_compressed.size(3) == 4 &&
                weights_compressed.size(2) == 8);
    TORCH_CHECK(weights_compressed.scalar_type() == torch::kInt64);
    TORCH_CHECK(codebook_abs.scalar_type() == torch::kInt32 && codebook_abs.size(-1) == 256);

    int64_t N = weights_compressed.size(0) * 16;
    int64_t K = weights_compressed.size(1) << 6;
    TORCH_CHECK(K % 64 == 0 && N % 16 == 0 && K < 65536 && N < 65536);

    at::DeviceGuard guard(codebook_abs.device());
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    torch::Tensor output = torch::zeros({N, K}, options);

    static int64_t grid_size = 0;
    if (grid_size == 0) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, weights_compressed.get_device());
        grid_size = static_cast<int64_t>(deviceProp.multiProcessorCount);
    }
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    glq_decompress_packed_e8p_kernel<<<grid_size, dim3(32, 32), 0, stream>>>(
        (uint32_t *)output.data_ptr<c10::Half>(),
        (const uint2 *)weights_compressed.data_ptr<int64_t>(),
        (const uint32_t *)codebook_abs.data_ptr<int32_t>(),
        N, K);
    E8P_ERRCHK(cudaPeekAtLastError());
    return output;
}


// ---- E81B (3-bit RVQ stage-2 residual) kernels: WMMA lookup-matmul (B=1/k<=8) + dense decompress ----
// Ported verbatim from quiptools.cu:541-675. CB is 256x8 fp16 (the 255-entry grid padded to 256).

__global__ static void glq_lookupmatmul_e81b_k8_kernel(
    const c10::Half *__restrict__ X,      // k x n
    const int64_t *__restrict__ YIs,      // m x (n/64)
    const c10::Half *__restrict__ CB,     // 256 x 8
    float *__restrict__ Z,                // k x m, accumulated into (pre-zeroed by caller)
    size_t K, size_t M, size_t N
) {
    long m1 = blockIdx.x;
    long k1 = blockIdx.y;
    __shared__ c10::Half Y_cache0[32 * 16];
    wmma::fragment<wmma::matrix_a, 8, 32, 16, __half, wmma::row_major> a0;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, __half, wmma::col_major> b0;
    __shared__ c10::Half Y_cache1[32 * 16];
    wmma::fragment<wmma::matrix_a, 8, 32, 16, __half, wmma::row_major> a1;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, __half, wmma::col_major> b1;
    wmma::fragment<wmma::accumulator, 8, 32, 16, float> c;
    fill_fragment(c, 0.0);
#pragma unroll
    for (long jn = 0; jn < N / 32; jn++) {
        uint32_t packed = ((uint32_t *)YIs)[(m1 * 32 + threadIdx.x) * (N / 32) + jn];
#pragma unroll
        for (long r = 0; r < 2; r++) {
            uint32_t yidx = packed & 255;
            ((uint64_t *)Y_cache0)[(threadIdx.x * 2 + r) * 2] = ((uint64_t *)CB)[yidx * 2];
            ((uint64_t *)Y_cache0)[(threadIdx.x * 2 + r) * 2 + 1] = ((uint64_t *)CB)[yidx * 2 + 1];
            packed = packed >> 8;
        }
#pragma unroll
        for (long r = 0; r < 2; r++) {
            uint32_t yidx = packed & 255;
            ((uint64_t *)Y_cache1)[(threadIdx.x * 2 + r) * 2] = ((uint64_t *)CB)[yidx * 2];
            ((uint64_t *)Y_cache1)[(threadIdx.x * 2 + r) * 2 + 1] = ((uint64_t *)CB)[yidx * 2 + 1];
            packed = packed >> 8;
        }
        load_matrix_sync(a0, (const __half *)(X + 8 * N * k1 + 32 * jn), N);
        load_matrix_sync(b0, (const __half *)Y_cache0, 16);
        mma_sync(c, a0, b0, c);
        load_matrix_sync(a1, (const __half *)(X + 8 * N * k1 + 32 * jn + 16), N);
        load_matrix_sync(b1, (const __half *)Y_cache1, 16);
        mma_sync(c, a1, b1, c);
    }
    store_matrix_sync(&Z[8 * M * k1 + 32 * m1], c, M, wmma::mem_row_major);
}


void glq_lookupmatmul_e81b_k8(torch::Tensor X, torch::Tensor YIs, torch::Tensor CB, torch::Tensor Z) {
    E8P_CHECK_INPUT(X); E8P_CHECK_INPUT(YIs); E8P_CHECK_INPUT(CB); E8P_CHECK_INPUT(Z);
    auto k = X.size(0), m = YIs.size(0), n = X.size(1);
    TORCH_CHECK(Z.size(0) == k && Z.size(1) == m && YIs.size(1) * 64 == n);
    TORCH_CHECK(k <= 8 && m % 32 == 0 && n % 32 == 0);
    TORCH_CHECK(X.scalar_type() == torch::kFloat16 && CB.scalar_type() == torch::kFloat16);
    TORCH_CHECK(YIs.scalar_type() == torch::kInt64 && Z.scalar_type() == torch::kFloat32);
    at::DeviceGuard guard(CB.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    glq_lookupmatmul_e81b_k8_kernel<<<dim3(m / 32, k / 8), dim3(32), 0, stream>>>(
        X.data_ptr<c10::Half>(), YIs.data_ptr<int64_t>(), CB.data_ptr<c10::Half>(),
        Z.data_ptr<float>(), k, m, n);
    E8P_ERRCHK(cudaPeekAtLastError());
}


__global__ static void glq_decompress_e81b_packed_kernel(
    const int64_t *__restrict__ YIs, const c10::Half *__restrict__ CB, c10::Half *__restrict__ Y
) {
    const long i = threadIdx.x + E8P_DECOMPRESS_E81B_BLOCK_SIZE * blockIdx.x;
    uint64_t packed = YIs[i];
#pragma unroll
    for (long j = 0; j < 8; j++) {
        uint64_t yidx = packed & 255;
        ((uint64_t *)Y)[(i * 8 + j) * 2] = ((uint64_t *)CB)[yidx * 2];
        ((uint64_t *)Y)[(i * 8 + j) * 2 + 1] = ((uint64_t *)CB)[yidx * 2 + 1];
        packed = packed >> 8;
    }
}


void glq_decompress_e81b_packed(torch::Tensor YIs, torch::Tensor CB, torch::Tensor Y) {
    E8P_CHECK_INPUT(YIs); E8P_CHECK_INPUT(CB); E8P_CHECK_INPUT(Y);
    size_t m = Y.size(0), n = Y.size(1);
    TORCH_CHECK(YIs.size(0) == m && YIs.size(1) * 64 == n);
    TORCH_CHECK(CB.size(0) == 256 && CB.size(1) == 8);
    at::DeviceGuard guard(CB.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    glq_decompress_e81b_packed_kernel<<<dim3(m * n / (64 * E8P_DECOMPRESS_E81B_BLOCK_SIZE)),
                                        dim3(E8P_DECOMPRESS_E81B_BLOCK_SIZE), 0, stream>>>(
        YIs.data_ptr<int64_t>(), CB.data_ptr<c10::Half>(), Y.data_ptr<c10::Half>());
    E8P_ERRCHK(cudaPeekAtLastError());
}


// ---- Grouped E81B residual stage for the fused MoE (odd bpw 5/7) ----
// Graft of glq_lookupmatmul_e81b_k8_kernel with device-side per-expert dispatch
// (m_indices, like the E8P grouped kernel) + per-expert wscale*cumulative-inv_rs
// folded in, ACCUMULATING into y_out. No K-split: one block computes the full-N
// dot product for its (8-token tile m1? no: 32-row tile, 8-token tile), so each
// (token,row) output is owned by exactly one block -> plain load+FMA+store, no
// atomics, deterministic. Requires M (out features) % 32 == 0 and M_sum_max % 8
// == 0 (host rounds M_sum_max up to a multiple of 8 when an E81B stage is used).
__global__ static void glq_lookupmatmul_e81b_grouped_kernel(
    const c10::Half *__restrict__ x_grouped,   // (M_sum_max, N) fp16, grouped tokens
    const int64_t *__restrict__ YIs,           // (E, M, N/64) int64 -> per-expert via w_estride
    const c10::Half *__restrict__ CB,          // 256 x 8 E81B codebook
    float *__restrict__ y_out,                 // (M_sum_max, M) fp32 -- ACCUMULATE
    const int *__restrict__ m_indices,         // (M_sum_max,) expert per slot, -1 = pad
    const float *__restrict__ wscale_dev,      // (E,)
    const float *__restrict__ inv_rs_dev,      // (E,) cumulative residual scale (or null)
    long w_estride,                            // int64 / expert = M * (N/64)
    int M_sum_max, int M, int N
) {
    // E81B_MM_WARPS warps per block, each owning a DISTINCT m-tile (M-parallel).
    // Every (token-tile, m-tile) output is still written by exactly one warp, so
    // the in-place accumulate stays atomic-free/deterministic; the extra warps +
    // the shared codebook lift occupancy off the old 1-warp-per-block launch.
    const int warpId = threadIdx.y;
    const int laneId = threadIdx.x;
    const long k1 = blockIdx.y;                 // token-tile (8 grouped tokens)
    __shared__ __half smem_CB[256 * 8];
    __shared__ c10::Half Ycache[E81B_MM_WARPS][2][32 * 16];
    for (int i = warpId * 32 + laneId; i < 256 * 8; i += E81B_MM_WARPS * 32)
        smem_CB[i] = ((const __half *)CB)[i];
    __syncthreads();

    const long m1 = (long)blockIdx.x * E81B_MM_WARPS + warpId;  // this warp's m-tile
    if (m1 >= M / 32) return;                   // extra warps (M/32 not a multiple of NW)
    const int base = (int)(k1 << 3);
    const int eidx = (base < M_sum_max) ? m_indices[base] : -1;
    if (eidx < 0) return;                       // fully-pad tile -> skip (output discarded)
    const int64_t *YIe = YIs + (size_t)eidx * w_estride;
    const float scale = wscale_dev[eidx] * (inv_rs_dev != nullptr ? inv_rs_dev[eidx] : 1.0f);
    c10::Half *Y_cache0 = Ycache[warpId][0];
    c10::Half *Y_cache1 = Ycache[warpId][1];

    wmma::fragment<wmma::matrix_a, 8, 32, 16, __half, wmma::row_major> a0, a1;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, __half, wmma::col_major> b0, b1;
    wmma::fragment<wmma::accumulator, 8, 32, 16, float> c;
    fill_fragment(c, 0.0f);
    for (long jn = 0; jn < N / 32; jn++) {
        uint32_t packed = ((const uint32_t *)YIe)[(m1 * 32 + laneId) * (N / 32) + jn];
#pragma unroll
        for (long r = 0; r < 2; r++) {
            uint32_t yidx = packed & 255;
            ((uint64_t *)Y_cache0)[(laneId * 2 + r) * 2]     = ((uint64_t *)smem_CB)[yidx * 2];
            ((uint64_t *)Y_cache0)[(laneId * 2 + r) * 2 + 1] = ((uint64_t *)smem_CB)[yidx * 2 + 1];
            packed = packed >> 8;
        }
#pragma unroll
        for (long r = 0; r < 2; r++) {
            uint32_t yidx = packed & 255;
            ((uint64_t *)Y_cache1)[(laneId * 2 + r) * 2]     = ((uint64_t *)smem_CB)[yidx * 2];
            ((uint64_t *)Y_cache1)[(laneId * 2 + r) * 2 + 1] = ((uint64_t *)smem_CB)[yidx * 2 + 1];
            packed = packed >> 8;
        }
        __syncwarp();
        load_matrix_sync(a0, (const __half *)(x_grouped + (size_t)8 * N * k1 + 32 * jn), N);
        load_matrix_sync(b0, (const __half *)Y_cache0, 16);
        mma_sync(c, a0, b0, c);
        load_matrix_sync(a1, (const __half *)(x_grouped + (size_t)8 * N * k1 + 32 * jn + 16), N);
        load_matrix_sync(b1, (const __half *)Y_cache1, 16);
        mma_sync(c, a1, b1, c);
        __syncwarp();
    }
    // accumulate c*scale into the (8-token, 32-row) y_out tile; single owner -> no atomics.
    wmma::fragment<wmma::accumulator, 8, 32, 16, float> y_frag;
    load_matrix_sync(y_frag, &y_out[(size_t)8 * M * k1 + 32 * m1], M, wmma::mem_row_major);
#pragma unroll
    for (int i = 0; i < c.num_elements; i++) c.x[i] = c.x[i] * scale + y_frag.x[i];
    store_matrix_sync(&y_out[(size_t)8 * M * k1 + 32 * m1], c, M, wmma::mem_row_major);
}


// Launch one grouped E81B residual stage, accumulating wscale*cum_inv_rs * (W_e81b @ x)
// into y_out (which must already hold the lower-stage E8P sum). NON-static (cross-TU).
void launch_grouped_lookupmatmul_e81b(
    const c10::Half *x_grouped, const int64_t *YIs, const c10::Half *CB, float *y_out,
    const int *m_indices, const float *wscale_dev, const float *inv_rs_dev,
    long w_estride, int M_sum_max, int M, int N, cudaStream_t stream
) {
    dim3 grid((M / 32 + E81B_MM_WARPS - 1) / E81B_MM_WARPS, M_sum_max / 8);
    glq_lookupmatmul_e81b_grouped_kernel<<<grid, dim3(32, E81B_MM_WARPS), 0, stream>>>(
        x_grouped, YIs, CB, y_out, m_indices, wscale_dev, inv_rs_dev,
        w_estride, M_sum_max, M, N);
}


// ============================================================================
// Fast E81B decode (odd-bpw 3/5/7). The stages above (glq_lookupmatmul_e81b_k8,
// grid (m/32, k/8), one warp per block, no K-split, batch padded to 8) decode
// the whole weight matrix at a fraction of the E8P GEMV/GEMM occupancy, so any
// odd-bpw layer is ~7-8x slower than even-bpw. These two kernels mirror the E8P
// SM-saturating design: a 4 KB shared-memory-resident 256x8 codebook gather in
// place of the E8P in-register lattice expand; direct fp32 accumulate; no host
// allocations. Same int64 (m_pad, n_pad/64) pack_e81b layout -> full checkpoint
// back-compat, no re-quant. Determinism: GEMV = one warp per output row + fixed
// shfl reduce; GEMM = disjoint k-split scratch planes + fixed-order reduce.
// NOTE on axes (differs from E8P): here N = reduction dim, M = output features.
// ============================================================================


// B=1 E81B decode GEMV. Warp-per-output-row FMA dot; the 32 lanes split the
// reduction (each int64 word packs 8 groups = 64 reduction dims). smem CB shared
// by all warps in the block. One warp owns a row -> deterministic, no scratch.
__global__ static void glq_decode_matvec_e81b_kernel(
    const c10::Half *__restrict__ x,     // (N,) fp16
    const int64_t *__restrict__ YIs,     // (M, N/64) int64
    const c10::Half *__restrict__ CB,    // (256, 8) fp16
    float *__restrict__ out,             // (M,) fp32
    int M, int N
) {
    const int warpId = threadIdx.y;
    const int laneId = threadIdx.x;
    const int NW = blockDim.y;
    __shared__ __half smem_CB[256 * 8];
    for (int i = warpId * 32 + laneId; i < 256 * 8; i += NW * 32)
        smem_CB[i] = ((const __half *)CB)[i];
    __syncthreads();

    const int words = N >> 6;                       // int64 per output row
    for (int row = blockIdx.x * NW + warpId; row < M; row += gridDim.x * NW) {
        const int64_t *rowp = YIs + (size_t)row * words;
        float acc = 0.0f;
        for (int w = laneId; w < words; w += 32) {
            uint64_t packed = (uint64_t)rowp[w];
            const __half *xw = (const __half *)x + (size_t)w * 64;
#pragma unroll
            for (int i = 0; i < 8; i++) {           // 8 groups (8 reduction dims each)
                uint32_t yidx = (uint32_t)(packed & 255);
                packed >>= 8;
                const __half *cbrow = smem_CB + yidx * 8;
                const __half *xg = xw + i * 8;
#pragma unroll
                for (int j = 0; j < 8; j++)
                    acc += __half2float(xg[j]) * __half2float(cbrow[j]);
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(FULL_MASK, acc, off);
        if (laneId == 0) out[row] = acc;
    }
}


torch::Tensor glq_decode_matvec_e81b(torch::Tensor x, torch::Tensor YIs, torch::Tensor CB) {
    E8P_CHECK_INPUT(x); E8P_CHECK_INPUT(YIs); E8P_CHECK_INPUT(CB);
    TORCH_CHECK(x.dim() == 1 && x.scalar_type() == torch::kFloat16);
    TORCH_CHECK(YIs.scalar_type() == torch::kInt64 && CB.scalar_type() == torch::kFloat16);
    TORCH_CHECK(CB.size(0) == 256 && CB.size(1) == 8);
    int64_t M = YIs.size(0), N = x.size(0);
    TORCH_CHECK(YIs.size(1) * 64 == N && N % 64 == 0);
    at::DeviceGuard guard(x.device());
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor output = torch::empty({M}, opt);
    static int64_t grid_size = 0;
    if (grid_size == 0) {
        cudaDeviceProp p; cudaGetDeviceProperties(&p, x.get_device());
        grid_size = p.multiProcessorCount;
    }
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    glq_decode_matvec_e81b_kernel<<<grid_size, dim3(32, E81B_MM_WARPS), 0, stream>>>(
        x.data_ptr<c10::Half>(), YIs.data_ptr<int64_t>(), CB.data_ptr<c10::Half>(),
        output.data_ptr<float>(), (int)M, (int)N);
    E8P_ERRCHK(cudaPeekAtLastError());
    return output;
}


// B>1 E81B decode GEMM. Reuses the WMMA gather+mma body of the k8 kernel but
// wraps it in the E8P GEMM scaffolding: persistent blocks stride output-row
// tiles (blockIdx.x), grid.y = 8-token tiles, E81B_MM_WARPS warps split the
// reduction (jn = warpId; += NW), each writes its own scratch plane, then
// glq_reduce_splits_2d_e8p_kernel sums the planes in fixed order.
__global__ static void glq_decode_matmul_e81b_kernel(
    float *__restrict__ scratch,         // (E81B_MM_WARPS, Bpad, M) fp32 k-split planes
    const c10::Half *__restrict__ X,     // (Bpad, N) fp16
    const int64_t *__restrict__ YIs,     // (M, N/64) int64
    const c10::Half *__restrict__ CB,    // (256, 8) fp16
    int M, int N, int Bpad
) {
    const int warpId = threadIdx.y;
    const int laneId = threadIdx.x;
    const long k1 = blockIdx.y;                     // 8-token tile
    __shared__ __half smem_CB[256 * 8];
    __shared__ c10::Half Ycache[E81B_MM_WARPS][2][32 * 16];
    for (int i = warpId * 32 + laneId; i < 256 * 8; i += E81B_MM_WARPS * 32)
        smem_CB[i] = ((const __half *)CB)[i];
    __syncthreads();

    c10::Half *Y_cache0 = Ycache[warpId][0];
    c10::Half *Y_cache1 = Ycache[warpId][1];
    const size_t kplane = (size_t)warpId * Bpad * M;

    for (long m1 = blockIdx.x; m1 < M / 32; m1 += gridDim.x) {
        wmma::fragment<wmma::matrix_a, 8, 32, 16, __half, wmma::row_major> a0, a1;
        wmma::fragment<wmma::matrix_b, 8, 32, 16, __half, wmma::col_major> b0, b1;
        wmma::fragment<wmma::accumulator, 8, 32, 16, float> c;
        fill_fragment(c, 0.0f);
        for (long jn = warpId; jn < N / 32; jn += E81B_MM_WARPS) {
            uint32_t packed = ((const uint32_t *)YIs)[(m1 * 32 + laneId) * (N / 32) + jn];
#pragma unroll
            for (long r = 0; r < 2; r++) {
                uint32_t yidx = packed & 255;
                ((uint64_t *)Y_cache0)[(laneId * 2 + r) * 2]     = ((uint64_t *)smem_CB)[yidx * 2];
                ((uint64_t *)Y_cache0)[(laneId * 2 + r) * 2 + 1] = ((uint64_t *)smem_CB)[yidx * 2 + 1];
                packed = packed >> 8;
            }
#pragma unroll
            for (long r = 0; r < 2; r++) {
                uint32_t yidx = packed & 255;
                ((uint64_t *)Y_cache1)[(laneId * 2 + r) * 2]     = ((uint64_t *)smem_CB)[yidx * 2];
                ((uint64_t *)Y_cache1)[(laneId * 2 + r) * 2 + 1] = ((uint64_t *)smem_CB)[yidx * 2 + 1];
                packed = packed >> 8;
            }
            __syncwarp();                           // ensure Y_cache writes visible to wmma load
            load_matrix_sync(a0, (const __half *)(X + (size_t)8 * N * k1 + 32 * jn), N);
            load_matrix_sync(b0, (const __half *)Y_cache0, 16);
            mma_sync(c, a0, b0, c);
            load_matrix_sync(a1, (const __half *)(X + (size_t)8 * N * k1 + 32 * jn + 16), N);
            load_matrix_sync(b1, (const __half *)Y_cache1, 16);
            mma_sync(c, a1, b1, c);
            __syncwarp();                           // Y_cache reused next iter
        }
        store_matrix_sync(&scratch[kplane + (size_t)8 * M * k1 + 32 * m1], c, M, wmma::mem_row_major);
    }
}


torch::Tensor glq_decode_matmul_e81b(torch::Tensor x, torch::Tensor YIs, torch::Tensor CB) {
    E8P_CHECK_INPUT(x); E8P_CHECK_INPUT(YIs); E8P_CHECK_INPUT(CB);
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == torch::kFloat16);
    TORCH_CHECK(YIs.scalar_type() == torch::kInt64 && CB.scalar_type() == torch::kFloat16);
    TORCH_CHECK(CB.size(0) == 256 && CB.size(1) == 8);
    int64_t B = x.size(0), N = x.size(-1), M = YIs.size(0);
    TORCH_CHECK(YIs.size(1) * 64 == N && N % 64 == 0 && M % 32 == 0 && N % 32 == 0);
    x = x.contiguous();
    at::DeviceGuard guard(x.device());
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    int64_t Bpad = (B + 7) & ~7L;                   // WMMA matrix_a tile is 8 rows
    torch::Tensor xp = x;
    if (Bpad != B) {
        xp = torch::zeros({Bpad, N}, x.options());
        xp.narrow(0, 0, B).copy_(x);
    }
    torch::Tensor output = torch::empty({Bpad, M}, opt);  // reduce overwrites every elem
    static int64_t grid_size = 0;
    if (grid_size == 0) {
        cudaDeviceProp p; cudaGetDeviceProperties(&p, x.get_device());
        grid_size = p.multiProcessorCount;
    }
    dim3 grid(grid_size, Bpad / 8);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    size_t scratch_bytes = (size_t)E81B_MM_WARPS * Bpad * M * sizeof(float);
    float *scratch = (float *)c10::cuda::CUDACachingAllocator::raw_alloc(scratch_bytes);
    glq_decode_matmul_e81b_kernel<<<grid, dim3(32, E81B_MM_WARPS), 0, stream>>>(
        scratch, xp.data_ptr<c10::Half>(), YIs.data_ptr<int64_t>(), CB.data_ptr<c10::Half>(),
        (int)M, (int)N, (int)Bpad);
    E8P_ERRCHK(cudaPeekAtLastError());
    int64_t BM = Bpad * M;
    int reduce_threads = 256;
    int reduce_blocks = (int)((BM + reduce_threads - 1) / reduce_threads);
    glq_reduce_splits_2d_e8p_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
        output.data_ptr<float>(), scratch, (int)BM, E81B_MM_WARPS);
    E8P_ERRCHK(cudaPeekAtLastError());
    c10::cuda::CUDACachingAllocator::raw_delete(scratch);
    return (Bpad != B) ? output.narrow(0, 0, B).contiguous() : output;
}
