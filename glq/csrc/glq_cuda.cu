/*
 * GLQ CUDA dequant+matmul kernels
 *
 * Fused codebook-gather + matmul for E8 lattice quantized weights.
 * Each 8-weight block is stored as a 16-bit codebook index into a
 * 65536×8 fp16 codebook (2bpw). Optional second-stage residual
 * codebook adds 3/4bpw support.
 *
 * For the packed path, the codebook is compressed to 65536 uint32
 * entries (4 nibbles per half2 pair), reducing L2 gather traffic 4×.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#define FULL_MASK 0xffffffff
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while(false)

/* ─────────────────────────────────────────────────────────────────────
 * B=1 Matvec kernel (decode path) — 4 rows per warp
 *
 * Computes y[m] = sum_j  codebook[Qidxs[m, j]] · x[j*8 : j*8+8]  * Wscale
 *
 * Grid:  (num_sms,)   — persistent-style, each block loops over M rows
 * Block: (32, WARPS)  — 32 threads/warp × WARPS warps
 *
 * Each warp processes 4 output rows simultaneously:
 *   Lanes  0-7:  row A
 *   Lanes  8-15: row B
 *   Lanes 16-23: row C
 *   Lanes 24-31: row D
 *
 * x is loaded once by lanes 0-7 and broadcast via __shfl_sync to all.
 * Each 8-lane group loads its own codebook index and gathers cb[idx][0..7].
 * Dot product reduced within each 8-lane group via __shfl_down_sync.
 * ───────────────────────────────────────────────────────────────────── */

#define ROWS_PER_WARP 4

template <bool HAS_STAGE2>
__global__ void glq_matvec_kernel(
    float* __restrict__ output,         // (M,) fp32
    const half* __restrict__ x,         // (N,) fp16 input vector
    const int16_t* __restrict__ qidxs,  // (M, N_BLOCKS) primary indices
    const half* __restrict__ codebook,  // (65536, 8) primary codebook
    const int16_t* __restrict__ qidxs2, // (M, N_BLOCKS) secondary indices [if HAS_STAGE2]
    const half* __restrict__ codebook2, // (K2, 8) secondary codebook [if HAS_STAGE2]
    float inv_resid_scale,              // 1/resid_scale [if HAS_STAGE2]
    float wscale,                       // global scale factor
    int M,                              // output rows
    int N_BLOCKS                        // N / 8
) {
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int n_warps = blockDim.y;

    // Which of the 4 rows within this warp does this thread belong to?
    const int row_in_warp = lane_id >> 3;   // 0, 1, 2, or 3
    const int elem = lane_id & 7;           // 0..7 within the 8-element group

    // Grid-stride loop over groups of 4 rows
    for (int base_row = (blockIdx.x * n_warps + warp_id) * ROWS_PER_WARP;
         base_row < M;
         base_row += gridDim.x * n_warps * ROWS_PER_WARP) {

        int my_row = base_row + row_in_warp;
        bool valid = (my_row < M);

        float acc = 0.0f;

        for (int j = 0; j < N_BLOCKS; j++) {
            // 1. Load x once (lanes 0-7), broadcast to all 4 groups
            float x_val;
            if (lane_id < 8) {
                x_val = __half2float(x[j * 8 + lane_id]);
            }
            // Broadcast: each lane gets x[j*8 + (lane_id % 8)]
            x_val = __shfl_sync(FULL_MASK, x_val, elem);

            // 2. Each 8-thread group loads its row's index
            //    All 8 threads in the group need the same idx, so lane
            //    row_in_warp*8 loads it and we broadcast within the group.
            uint16_t idx = 0;
            if (valid && elem == 0) {
                idx = (uint16_t)qidxs[my_row * N_BLOCKS + j];
            }
            // Broadcast idx within the 8-lane group
            idx = __shfl_sync(FULL_MASK, idx, row_in_warp * 8);

            // 3. Gather codebook[idx][elem]
            float cb_val = 0.0f;
            if (valid) {
                cb_val = __half2float(codebook[idx * 8 + elem]);
            }

            // 4. Two-stage residual
            if (HAS_STAGE2) {
                uint16_t idx2 = 0;
                if (valid && elem == 0) {
                    idx2 = (uint16_t)qidxs2[my_row * N_BLOCKS + j];
                }
                idx2 = __shfl_sync(FULL_MASK, idx2, row_in_warp * 8);
                if (valid) {
                    cb_val += __half2float(codebook2[idx2 * 8 + elem]) * inv_resid_scale;
                }
            }

            // 5. Dot product within each 8-lane group
            float prod = cb_val * x_val;

            // Reduce within the 8-lane group using __shfl_xor_sync
            // This stays within the 8-lane group naturally
            prod += __shfl_xor_sync(FULL_MASK, prod, 4);
            prod += __shfl_xor_sync(FULL_MASK, prod, 2);
            prod += __shfl_xor_sync(FULL_MASK, prod, 1);

            // All lanes in the group have the full dot product; only elem==0 accumulates
            if (elem == 0) {
                acc += prod;
            }
        }

        // Write 4 outputs (one per 8-lane group leader)
        if (elem == 0 && valid) {
            atomicAdd(&output[my_row], acc * wscale);
        }
    }
}


/* ─────────────────────────────────────────────────────────────────────
 * Split-K Matvec — distributes N reduction across CTAs
 *
 * Grid: 2D (ceil(M/ROWS_PER_BLOCK), ceil(N_BLOCKS/BPS))
 * Each CTA processes BPS codebook blocks for its row tile.
 * Partial sums accumulated via atomicAdd (output must be pre-zeroed).
 *
 * This parallelizes the inner loop: instead of 1 warp doing 1152
 * iterations (3072×9216), we split into 18 CTAs of 64 iterations each.
 * More CTAs in flight = more L2 requests = better latency hiding.
 * ───────────────────────────────────────────────────────────────────── */

#define BLOCKS_PER_SPLIT 64

template <bool HAS_STAGE2>
__global__ void __launch_bounds__(256, 2)
glq_matvec_splitk_kernel(
    float* __restrict__ output,         // (M,) fp32, pre-zeroed
    const half* __restrict__ x,
    const int16_t* __restrict__ qidxs,
    const half* __restrict__ codebook,
    const int16_t* __restrict__ qidxs2,
    const half* __restrict__ codebook2,
    float inv_resid_scale,
    float wscale,
    int M,
    int N_BLOCKS
) {
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int row_in_warp = lane_id >> 3;
    const int elem = lane_id & 7;

    // 2D grid: dim0 = row tiles, dim1 = K splits
    const int rows_per_block = ROWS_PER_WARP * blockDim.y;  // 4 * 8 = 32
    int base_row = (blockIdx.x * blockDim.y + warp_id) * ROWS_PER_WARP;
    int my_row = base_row + row_in_warp;
    bool valid = (my_row < M);

    int j_start = blockIdx.y * BLOCKS_PER_SPLIT;
    int j_end = min(j_start + BLOCKS_PER_SPLIT, N_BLOCKS);

    float acc = 0.0f;

    for (int j = j_start; j < j_end; j++) {
        // Load x once, broadcast
        float x_val;
        if (lane_id < 8) {
            x_val = __half2float(x[j * 8 + lane_id]);
        }
        x_val = __shfl_sync(FULL_MASK, x_val, elem);

        // Each group loads its index
        uint16_t idx = 0;
        if (valid && elem == 0) {
            idx = (uint16_t)qidxs[my_row * N_BLOCKS + j];
        }
        idx = __shfl_sync(FULL_MASK, idx, row_in_warp * 8);

        // Gather codebook
        float cb_val = 0.0f;
        if (valid) {
            cb_val = __half2float(codebook[idx * 8 + elem]);
        }

        // Two-stage residual
        if (HAS_STAGE2) {
            uint16_t idx2 = 0;
            if (valid && elem == 0) {
                idx2 = (uint16_t)qidxs2[my_row * N_BLOCKS + j];
            }
            idx2 = __shfl_sync(FULL_MASK, idx2, row_in_warp * 8);
            if (valid) {
                cb_val += __half2float(codebook2[idx2 * 8 + elem]) * inv_resid_scale;
            }
        }

        // Dot product within 8-lane group
        float prod = cb_val * x_val;
        prod += __shfl_xor_sync(FULL_MASK, prod, 4);
        prod += __shfl_xor_sync(FULL_MASK, prod, 2);
        prod += __shfl_xor_sync(FULL_MASK, prod, 1);

        if (elem == 0) {
            acc += prod;
        }
    }

    // atomicAdd partial sum (output pre-zeroed by host)
    if (elem == 0 && valid) {
        atomicAdd(&output[my_row], acc * wscale);
    }
}


/* ─────────────────────────────────────────────────────────────────────
 * Split-K Packed variant
 * ───────────────────────────────────────────────────────────────────── */

__global__ void __launch_bounds__(256, 2)
glq_matvec_splitk_packed_kernel(
    float* __restrict__ output,
    const half* __restrict__ x,
    const int16_t* __restrict__ qidxs,
    const uint32_t* __restrict__ codebook_packed,
    float wscale,
    int M,
    int N_BLOCKS
) {
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int row_in_warp = lane_id >> 3;
    const int elem = lane_id & 7;

    int base_row = (blockIdx.x * blockDim.y + warp_id) * ROWS_PER_WARP;
    int my_row = base_row + row_in_warp;
    bool valid = (my_row < M);

    int j_start = blockIdx.y * BLOCKS_PER_SPLIT;
    int j_end = min(j_start + BLOCKS_PER_SPLIT, N_BLOCKS);

    float acc = 0.0f;

    for (int j = j_start; j < j_end; j++) {
        float x_val;
        if (lane_id < 8) {
            x_val = __half2float(x[j * 8 + lane_id]);
        }
        x_val = __shfl_sync(FULL_MASK, x_val, elem);

        uint16_t idx = 0;
        if (valid && elem == 0) {
            idx = (uint16_t)qidxs[my_row * N_BLOCKS + j];
        }
        idx = __shfl_sync(FULL_MASK, idx, row_in_warp * 8);

        float cb_val = 0.0f;
        if (valid) {
            uint32_t packed = codebook_packed[idx];
            uint32_t nibble = (packed >> (elem * 4)) & 0xF;
            cb_val = (float)nibble * 0.5f - 3.0f;
        }

        float prod = cb_val * x_val;
        prod += __shfl_xor_sync(FULL_MASK, prod, 4);
        prod += __shfl_xor_sync(FULL_MASK, prod, 2);
        prod += __shfl_xor_sync(FULL_MASK, prod, 1);

        if (elem == 0) {
            acc += prod;
        }
    }

    if (elem == 0 && valid) {
        atomicAdd(&output[my_row], acc * wscale);
    }
}


/* ─────────────────────────────────────────────────────────────────────
 * Packed uint32 variant — 4 rows per warp
 *
 * Same 4-row layout as the fp16 kernel but uses packed codebook:
 *   nibble[i] = (packed >> (i*4)) & 0xF
 *   value[i] = nibble[i] * 0.5f - 3.0f
 *
 * 4 bytes per codebook entry vs 16 → better L2 utilization.
 * ───────────────────────────────────────────────────────────────────── */

__global__ void glq_matvec_packed_kernel(
    float* __restrict__ output,
    const half* __restrict__ x,
    const int16_t* __restrict__ qidxs,
    const uint32_t* __restrict__ codebook_packed, // (65536,) uint32
    float wscale,
    int M,
    int N_BLOCKS
) {
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int n_warps = blockDim.y;

    const int row_in_warp = lane_id >> 3;
    const int elem = lane_id & 7;

    for (int base_row = (blockIdx.x * n_warps + warp_id) * ROWS_PER_WARP;
         base_row < M;
         base_row += gridDim.x * n_warps * ROWS_PER_WARP) {

        int my_row = base_row + row_in_warp;
        bool valid = (my_row < M);

        float acc = 0.0f;

        for (int j = 0; j < N_BLOCKS; j++) {
            // Load x once, broadcast
            float x_val;
            if (lane_id < 8) {
                x_val = __half2float(x[j * 8 + lane_id]);
            }
            x_val = __shfl_sync(FULL_MASK, x_val, elem);

            // Each group loads its index and packed codebook entry
            uint16_t idx = 0;
            if (valid && elem == 0) {
                idx = (uint16_t)qidxs[my_row * N_BLOCKS + j];
            }
            idx = __shfl_sync(FULL_MASK, idx, row_in_warp * 8);

            // Load packed uint32 and decode nibble
            float cb_val = 0.0f;
            if (valid) {
                uint32_t packed = codebook_packed[idx];
                uint32_t nibble = (packed >> (elem * 4)) & 0xF;
                cb_val = (float)nibble * 0.5f - 3.0f;
            }

            // Dot product within 8-lane group (xor stays within group)
            float prod = cb_val * x_val;
            prod += __shfl_xor_sync(FULL_MASK, prod, 4);
            prod += __shfl_xor_sync(FULL_MASK, prod, 2);
            prod += __shfl_xor_sync(FULL_MASK, prod, 1);

            if (elem == 0) {
                acc += prod;
            }
        }

        if (elem == 0 && valid) {
            atomicAdd(&output[my_row], acc * wscale);
        }
    }
}


/* ─────────────────────────────────────────────────────────────────────
 * Host wrappers (pybind11 entry points)
 * ───────────────────────────────────────────────────────────────────── */

torch::Tensor glq_dequant_matvec_cuda(
    torch::Tensor x,           // (N,) fp16
    torch::Tensor qidxs,       // (M, N_BLOCKS) int16
    torch::Tensor codebook,    // (65536, 8) fp16
    float wscale,
    torch::Tensor qidxs2,      // (M, N_BLOCKS) int16 or empty
    torch::Tensor codebook2,   // (K2, 8) fp16 or empty
    float inv_resid_scale
) {
    CHECK_INPUT(x);
    CHECK_INPUT(qidxs);
    CHECK_INPUT(codebook);

    int M = qidxs.size(0);
    int N_BLOCKS = qidxs.size(1);

    at::DeviceGuard guard(x.device());
    // Split-K uses atomicAdd → output must be zeroed
    auto output = torch::zeros({M}, torch::dtype(torch::kFloat32).device(x.device()));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, x.get_device());
    int num_sms = prop.multiProcessorCount;

    const int WARPS = 8;
    const int rows_per_block = ROWS_PER_WARP * WARPS;  // 32
    dim3 block(32, WARPS);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    bool has_stage2 = (qidxs2.numel() > 0 && inv_resid_scale != 0.0f);

    // Always use split-K: it handles all cases well (large M just gets k_splits=1)
    int m_blocks = (M + rows_per_block - 1) / rows_per_block;
    bool use_splitk = (N_BLOCKS >= BLOCKS_PER_SPLIT);

    if (use_splitk) {
        int k_splits = (N_BLOCKS + BLOCKS_PER_SPLIT - 1) / BLOCKS_PER_SPLIT;
        dim3 grid(m_blocks, k_splits);

        if (has_stage2) {
            CHECK_INPUT(qidxs2);
            CHECK_INPUT(codebook2);
            glq_matvec_splitk_kernel<true><<<grid, block, 0, stream>>>(
                output.data_ptr<float>(),
                (const half*)x.data_ptr<c10::Half>(),
                qidxs.data_ptr<int16_t>(),
                (const half*)codebook.data_ptr<c10::Half>(),
                qidxs2.data_ptr<int16_t>(),
                (const half*)codebook2.data_ptr<c10::Half>(),
                inv_resid_scale,
                wscale, M, N_BLOCKS
            );
        } else {
            glq_matvec_splitk_kernel<false><<<grid, block, 0, stream>>>(
                output.data_ptr<float>(),
                (const half*)x.data_ptr<c10::Half>(),
                qidxs.data_ptr<int16_t>(),
                (const half*)codebook.data_ptr<c10::Half>(),
                nullptr, nullptr, 0.0f,
                wscale, M, N_BLOCKS
            );
        }
    } else {
        dim3 grid(num_sms);

        if (has_stage2) {
            CHECK_INPUT(qidxs2);
            CHECK_INPUT(codebook2);
            glq_matvec_kernel<true><<<grid, block, 0, stream>>>(
                output.data_ptr<float>(),
                (const half*)x.data_ptr<c10::Half>(),
                qidxs.data_ptr<int16_t>(),
                (const half*)codebook.data_ptr<c10::Half>(),
                qidxs2.data_ptr<int16_t>(),
                (const half*)codebook2.data_ptr<c10::Half>(),
                inv_resid_scale,
                wscale, M, N_BLOCKS
            );
        } else {
            glq_matvec_kernel<false><<<grid, block, 0, stream>>>(
                output.data_ptr<float>(),
                (const half*)x.data_ptr<c10::Half>(),
                qidxs.data_ptr<int16_t>(),
                (const half*)codebook.data_ptr<c10::Half>(),
                nullptr, nullptr, 0.0f,
                wscale, M, N_BLOCKS
            );
        }
    }

    return output;
}


torch::Tensor glq_dequant_matvec_packed_cuda(
    torch::Tensor x,
    torch::Tensor qidxs,
    torch::Tensor codebook_packed, // (65536,) uint32
    float wscale
) {
    CHECK_INPUT(x);
    CHECK_INPUT(qidxs);
    CHECK_INPUT(codebook_packed);

    int M = qidxs.size(0);
    int N_BLOCKS = qidxs.size(1);

    at::DeviceGuard guard(x.device());
    auto output = torch::zeros({M}, torch::dtype(torch::kFloat32).device(x.device()));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, x.get_device());
    int num_sms = prop.multiProcessorCount;

    const int WARPS = 8;
    dim3 block(32, WARPS);
    dim3 grid(num_sms);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    glq_matvec_packed_kernel<<<grid, block, 0, stream>>>(
        output.data_ptr<float>(),
        (const half*)x.data_ptr<c10::Half>(),
        qidxs.data_ptr<int16_t>(),
        (const uint32_t*)codebook_packed.data_ptr<int32_t>(),
        wscale, M, N_BLOCKS
    );

    return output;
}


/* ─────────────────────────────────────────────────────────────────────
 * Shared-memory Fast Hadamard Transform (FHT)
 *
 * Replaces the Triton _input_rht_kernel and _output_rht_kernel which
 * use global memory barriers (store→debug_barrier→load per butterfly
 * stage). Shared memory eliminates global round-trips: ~5ns per access
 * vs ~100-200ns, reducing FHT from ~28-36μs to ~3-5μs.
 *
 * Grid: (B,) — one block per batch row
 * Block: min(n_pad, 1024) threads
 * Shared memory: n_pad * sizeof(float) bytes (dynamically allocated)
 * ───────────────────────────────────────────────────────────────────── */

__global__ void glq_input_rht_kernel(
    const half* __restrict__ x,      // (B, in_features) fp16 input
    const half* __restrict__ sv,     // (n_pad,) fp16 sign vector
    float* __restrict__ out,         // (B, n_pad) fp32 output (x_rht)
    int in_features,
    int stride_x,                    // x.stride(0) in elements
    float rsqrt_n,                   // 1.0 / sqrt(n_pad)
    int n_pad,
    int log_n
) {
    // Double-buffered shared memory: read from one, write to the other
    extern __shared__ float smem[];  // 2 * n_pad floats
    float* buf_a = smem;
    float* buf_b = smem + n_pad;

    int b = blockIdx.x;
    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    // Step 1: Load x with zero-padding, multiply by SV signs → buf_a
    for (int i = tid; i < n_pad; i += n_threads) {
        float x_val = (i < in_features) ? __half2float(x[b * stride_x + i]) : 0.0f;
        float s = __half2float(sv[i]);
        buf_a[i] = x_val * s;
    }
    __syncthreads();

    // Step 2: FHT butterfly stages with double buffering
    float* src = buf_a;
    float* dst = buf_b;
    for (int k = 0; k < log_n; k++) {
        for (int i = tid; i < n_pad; i += n_threads) {
            int partner = i ^ (1 << k);
            float my_val = src[i];
            float partner_val = src[partner];
            bool lo = (i & (1 << k)) == 0;
            dst[i] = lo ? (my_val + partner_val) : (partner_val - my_val);
        }
        __syncthreads();
        // Swap buffers
        float* tmp = src; src = dst; dst = tmp;
    }

    // Step 3: Normalize and store (result is in src after last swap)
    for (int i = tid; i < n_pad; i += n_threads) {
        out[b * n_pad + i] = src[i] * rsqrt_n;
    }
}


__global__ void glq_output_rht_kernel(
    const float* __restrict__ y_rht, // (B, m_pad) fp32 input
    const half* __restrict__ su,     // (m_pad,) fp16 sign vector
    half* __restrict__ out,          // (B, out_features) fp16 output
    int out_features,
    int m_pad,
    int log_m,
    float rsqrt_m                    // 1.0 / sqrt(m_pad)
) {
    extern __shared__ float smem[];
    float* buf_a = smem;
    float* buf_b = smem + m_pad;

    int b = blockIdx.x;
    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    // Step 1: Load y_rht into buf_a
    for (int i = tid; i < m_pad; i += n_threads) {
        buf_a[i] = y_rht[b * m_pad + i];
    }
    __syncthreads();

    // Step 2: FHT butterfly stages with double buffering
    float* src = buf_a;
    float* dst = buf_b;
    for (int k = 0; k < log_m; k++) {
        for (int i = tid; i < m_pad; i += n_threads) {
            int partner = i ^ (1 << k);
            float my_val = src[i];
            float partner_val = src[partner];
            bool lo = (i & (1 << k)) == 0;
            dst[i] = lo ? (my_val + partner_val) : (partner_val - my_val);
        }
        __syncthreads();
        float* tmp = src; src = dst; dst = tmp;
    }

    // Step 3: Normalize, apply SU signs, unpad, cast to fp16
    for (int i = tid; i < out_features; i += n_threads) {
        float val = src[i] * rsqrt_m;
        float s = __half2float(su[i]);
        out[b * out_features + i] = __float2half(val * s);
    }
}


/* ─────────────────────────────────────────────────────────────────────
 * RHT host wrappers
 * ───────────────────────────────────────────────────────────────────── */

void glq_input_rht_cuda(
    torch::Tensor x,        // (B, in_features) fp16
    torch::Tensor sv,       // (n_pad,) fp16
    torch::Tensor out,      // (B, n_pad) fp32 — pre-allocated
    int in_features,
    int stride_x,
    float rsqrt_n,
    int n_pad,
    int log_n
) {
    CHECK_INPUT(x);
    CHECK_INPUT(sv);

    int B = x.size(0);
    int threads = min(n_pad, 1024);
    int smem = 2 * n_pad * sizeof(float);  // double-buffered

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Request extended shared memory if needed (>48KB)
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(glq_input_rht_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }

    glq_input_rht_kernel<<<B, threads, smem, stream>>>(
        (const half*)x.data_ptr<c10::Half>(),
        (const half*)sv.data_ptr<c10::Half>(),
        out.data_ptr<float>(),
        in_features, stride_x, rsqrt_n, n_pad, log_n
    );
}


void glq_output_rht_cuda(
    torch::Tensor y_rht,    // (B, m_pad) fp32
    torch::Tensor su,       // (m_pad,) fp16
    torch::Tensor out,      // (B, out_features) fp16 — pre-allocated
    int out_features,
    int m_pad,
    int log_m,
    float rsqrt_m
) {
    CHECK_INPUT(y_rht);
    CHECK_INPUT(su);

    int B = y_rht.size(0);
    int threads = min(m_pad, 1024);
    int smem = 2 * m_pad * sizeof(float);  // double-buffered

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(glq_output_rht_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }

    glq_output_rht_kernel<<<B, threads, smem, stream>>>(
        y_rht.data_ptr<float>(),
        (const half*)su.data_ptr<c10::Half>(),
        (half*)out.data_ptr<c10::Half>(),
        out_features, m_pad, log_m, rsqrt_m
    );
}
