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
#include <c10/cuda/CUDACachingAllocator.h>
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

template <int NUM_STAGES>
__global__ void glq_matvec_kernel(
    float* __restrict__ output,         // (M,) fp32
    const half* __restrict__ x,         // (N,) fp16 input vector
    const int16_t* __restrict__ qidxs,  // (M, N_BLOCKS) primary indices
    const half* __restrict__ codebook,  // (65536, 8) primary codebook
    const int16_t* __restrict__ qidxs2, // (M, N_BLOCKS) secondary indices [stage 2+]
    const half* __restrict__ codebook2, // (K2, 8) secondary codebook [stage 2+]
    float inv_resid_scale,              // 1/resid_scale [stage 2+]
    const int16_t* __restrict__ qidxs3, // (M, N_BLOCKS) tertiary indices [stage 3+]
    const half* __restrict__ codebook3, // tertiary codebook [stage 3+]
    float inv_resid_scale2,             // 1/resid_scale2 [stage 3+]
    const int16_t* __restrict__ qidxs4, // (M, N_BLOCKS) quaternary indices [stage 4]
    const half* __restrict__ codebook4, // quaternary codebook [stage 4]
    float inv_resid_scale3,             // 1/resid_scale3 [stage 4]
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

            // 4. Residual stages
            if constexpr (NUM_STAGES >= 2) {
                uint16_t idx2 = 0;
                if (valid && elem == 0) {
                    idx2 = (uint16_t)qidxs2[my_row * N_BLOCKS + j];
                }
                idx2 = __shfl_sync(FULL_MASK, idx2, row_in_warp * 8);
                if (valid) {
                    cb_val += __half2float(codebook2[idx2 * 8 + elem]) * inv_resid_scale;
                }
            }
            if constexpr (NUM_STAGES >= 3) {
                uint16_t idx3 = 0;
                if (valid && elem == 0) {
                    idx3 = (uint16_t)qidxs3[my_row * N_BLOCKS + j];
                }
                idx3 = __shfl_sync(FULL_MASK, idx3, row_in_warp * 8);
                if (valid) {
                    cb_val += __half2float(codebook3[idx3 * 8 + elem]) * inv_resid_scale2;
                }
            }
            if constexpr (NUM_STAGES >= 4) {
                uint16_t idx4 = 0;
                if (valid && elem == 0) {
                    idx4 = (uint16_t)qidxs4[my_row * N_BLOCKS + j];
                }
                idx4 = __shfl_sync(FULL_MASK, idx4, row_in_warp * 8);
                if (valid) {
                    cb_val += __half2float(codebook4[idx4 * 8 + elem]) * inv_resid_scale3;
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

#define BLOCKS_PER_SPLIT_DEFAULT 64

template <int NUM_STAGES>
__global__ void __launch_bounds__(256)
glq_matvec_splitk_kernel(
    float* __restrict__ output,         // (M,) fp32, pre-zeroed
    const half* __restrict__ x,
    const int16_t* __restrict__ qidxs,
    const half* __restrict__ codebook,  // (65536, 8) fp16
    const int16_t* __restrict__ qidxs2,
    const half* __restrict__ codebook2,
    float inv_resid_scale,
    const int16_t* __restrict__ qidxs3,
    const half* __restrict__ codebook3,
    float inv_resid_scale2,
    const int16_t* __restrict__ qidxs4,
    const half* __restrict__ codebook4,
    float inv_resid_scale3,
    float wscale,
    int M,
    int N_BLOCKS,
    int bps,                            // blocks per split (adaptive)
    int cb2_size                        // secondary codebook entries (0 if no stage2)
) {
    // Stage small secondary codebook into shared memory (256 entries = 4KB).
    // Stages 3+ always use the primary (65536-entry) codebook, so no smem
    // staging there — cb3/cb4 go through the global path.
    extern __shared__ half smem_cb2[];

    if (NUM_STAGES >= 2 && cb2_size > 0 && cb2_size <= 256) {
        int tid_flat = threadIdx.y * 32 + threadIdx.x;
        int total = cb2_size * 8;
        for (int i = tid_flat; i < total; i += 256) {
            smem_cb2[i] = codebook2[i];
        }
        __syncthreads();
    }

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int row_in_warp = lane_id >> 3;
    const int elem = lane_id & 7;

    // 2D grid: dim0 = row tiles, dim1 = K splits
    const int rows_per_block = ROWS_PER_WARP * blockDim.y;  // 4 * 8 = 32
    int base_row = (blockIdx.x * blockDim.y + warp_id) * ROWS_PER_WARP;
    int my_row = base_row + row_in_warp;
    bool valid = (my_row < M);

    int j_start = blockIdx.y * bps;
    int j_end = min(j_start + bps, N_BLOCKS);

    float acc = 0.0f;

    // Unrolled-by-2: issue two codebook gathers back-to-back to overlap L2 latency
    int j = j_start;
    for (; j + 1 < j_end; j += 2) {
        float x_val_0, x_val_1;
        if (lane_id < 8) {
            x_val_0 = __half2float(x[j * 8 + lane_id]);
            x_val_1 = __half2float(x[(j + 1) * 8 + lane_id]);
        }
        x_val_0 = __shfl_sync(FULL_MASK, x_val_0, elem);
        x_val_1 = __shfl_sync(FULL_MASK, x_val_1, elem);

        uint16_t idx_0 = 0, idx_1 = 0;
        if (valid && elem == 0) {
            idx_0 = (uint16_t)qidxs[my_row * N_BLOCKS + j];
            idx_1 = (uint16_t)qidxs[my_row * N_BLOCKS + j + 1];
        }
        idx_0 = __shfl_sync(FULL_MASK, idx_0, row_in_warp * 8);
        idx_1 = __shfl_sync(FULL_MASK, idx_1, row_in_warp * 8);

        float cb_val_0 = 0.0f, cb_val_1 = 0.0f;
        if (valid) {
            cb_val_0 = __half2float(codebook[idx_0 * 8 + elem]);
            cb_val_1 = __half2float(codebook[idx_1 * 8 + elem]);
        }

        if constexpr (NUM_STAGES >= 2) {
            const half* cb2 = (cb2_size > 0 && cb2_size <= 256) ? smem_cb2 : codebook2;
            uint16_t idx2_0 = 0, idx2_1 = 0;
            if (valid && elem == 0) {
                idx2_0 = (uint16_t)qidxs2[my_row * N_BLOCKS + j];
                idx2_1 = (uint16_t)qidxs2[my_row * N_BLOCKS + j + 1];
            }
            idx2_0 = __shfl_sync(FULL_MASK, idx2_0, row_in_warp * 8);
            idx2_1 = __shfl_sync(FULL_MASK, idx2_1, row_in_warp * 8);
            if (valid) {
                cb_val_0 += __half2float(cb2[idx2_0 * 8 + elem]) * inv_resid_scale;
                cb_val_1 += __half2float(cb2[idx2_1 * 8 + elem]) * inv_resid_scale;
            }
        }
        if constexpr (NUM_STAGES >= 3) {
            uint16_t idx3_0 = 0, idx3_1 = 0;
            if (valid && elem == 0) {
                idx3_0 = (uint16_t)qidxs3[my_row * N_BLOCKS + j];
                idx3_1 = (uint16_t)qidxs3[my_row * N_BLOCKS + j + 1];
            }
            idx3_0 = __shfl_sync(FULL_MASK, idx3_0, row_in_warp * 8);
            idx3_1 = __shfl_sync(FULL_MASK, idx3_1, row_in_warp * 8);
            if (valid) {
                cb_val_0 += __half2float(codebook3[idx3_0 * 8 + elem]) * inv_resid_scale2;
                cb_val_1 += __half2float(codebook3[idx3_1 * 8 + elem]) * inv_resid_scale2;
            }
        }
        if constexpr (NUM_STAGES >= 4) {
            uint16_t idx4_0 = 0, idx4_1 = 0;
            if (valid && elem == 0) {
                idx4_0 = (uint16_t)qidxs4[my_row * N_BLOCKS + j];
                idx4_1 = (uint16_t)qidxs4[my_row * N_BLOCKS + j + 1];
            }
            idx4_0 = __shfl_sync(FULL_MASK, idx4_0, row_in_warp * 8);
            idx4_1 = __shfl_sync(FULL_MASK, idx4_1, row_in_warp * 8);
            if (valid) {
                cb_val_0 += __half2float(codebook4[idx4_0 * 8 + elem]) * inv_resid_scale3;
                cb_val_1 += __half2float(codebook4[idx4_1 * 8 + elem]) * inv_resid_scale3;
            }
        }

        float prod_0 = cb_val_0 * x_val_0;
        prod_0 += __shfl_xor_sync(FULL_MASK, prod_0, 4);
        prod_0 += __shfl_xor_sync(FULL_MASK, prod_0, 2);
        prod_0 += __shfl_xor_sync(FULL_MASK, prod_0, 1);

        float prod_1 = cb_val_1 * x_val_1;
        prod_1 += __shfl_xor_sync(FULL_MASK, prod_1, 4);
        prod_1 += __shfl_xor_sync(FULL_MASK, prod_1, 2);
        prod_1 += __shfl_xor_sync(FULL_MASK, prod_1, 1);

        if (elem == 0) {
            acc += prod_0 + prod_1;
        }
    }
    // Tail: handle odd remaining iteration
    if (j < j_end) {
        float x_val;
        if (lane_id < 8) { x_val = __half2float(x[j * 8 + lane_id]); }
        x_val = __shfl_sync(FULL_MASK, x_val, elem);
        uint16_t idx = 0;
        if (valid && elem == 0) { idx = (uint16_t)qidxs[my_row * N_BLOCKS + j]; }
        idx = __shfl_sync(FULL_MASK, idx, row_in_warp * 8);
        float cb_val = 0.0f;
        if (valid) { cb_val = __half2float(codebook[idx * 8 + elem]); }
        if constexpr (NUM_STAGES >= 2) {
            const half* cb2 = (cb2_size > 0 && cb2_size <= 256) ? smem_cb2 : codebook2;
            uint16_t idx2 = 0;
            if (valid && elem == 0) { idx2 = (uint16_t)qidxs2[my_row * N_BLOCKS + j]; }
            idx2 = __shfl_sync(FULL_MASK, idx2, row_in_warp * 8);
            if (valid) { cb_val += __half2float(cb2[idx2 * 8 + elem]) * inv_resid_scale; }
        }
        if constexpr (NUM_STAGES >= 3) {
            uint16_t idx3 = 0;
            if (valid && elem == 0) { idx3 = (uint16_t)qidxs3[my_row * N_BLOCKS + j]; }
            idx3 = __shfl_sync(FULL_MASK, idx3, row_in_warp * 8);
            if (valid) { cb_val += __half2float(codebook3[idx3 * 8 + elem]) * inv_resid_scale2; }
        }
        if constexpr (NUM_STAGES >= 4) {
            uint16_t idx4 = 0;
            if (valid && elem == 0) { idx4 = (uint16_t)qidxs4[my_row * N_BLOCKS + j]; }
            idx4 = __shfl_sync(FULL_MASK, idx4, row_in_warp * 8);
            if (valid) { cb_val += __half2float(codebook4[idx4 * 8 + elem]) * inv_resid_scale3; }
        }
        float prod = cb_val * x_val;
        prod += __shfl_xor_sync(FULL_MASK, prod, 4);
        prod += __shfl_xor_sync(FULL_MASK, prod, 2);
        prod += __shfl_xor_sync(FULL_MASK, prod, 1);
        if (elem == 0) { acc += prod; }
    }

    // atomicAdd partial sum (output pre-zeroed by host)
    if (elem == 0 && valid) {
        atomicAdd(&output[my_row], acc * wscale);
    }
}


/* ─────────────────────────────────────────────────────────────────────
 * Deterministic split-K matvec: writes per-split partial sums to a
 * scratch[k_splits, M] buffer without atomicAdd. A follow-up reduction
 * kernel sums the scratch rows in fixed order, giving bit-exact
 * run-to-run results while preserving the SM-saturation benefit of
 * split-K. Body is identical to glq_matvec_splitk_kernel except the
 * final write: one unique (k_split, m) slot per CTA, plain store.
 * ───────────────────────────────────────────────────────────────────── */
template <int NUM_STAGES>
__global__ void __launch_bounds__(256)
glq_matvec_splitk_scratch_kernel(
    float* __restrict__ scratch,        // (k_splits, M) fp32, overwritten
    const half* __restrict__ x,
    const int16_t* __restrict__ qidxs,
    const half* __restrict__ codebook,
    const int16_t* __restrict__ qidxs2,
    const half* __restrict__ codebook2,
    float inv_resid_scale,
    const int16_t* __restrict__ qidxs3,
    const half* __restrict__ codebook3,
    float inv_resid_scale2,
    const int16_t* __restrict__ qidxs4,
    const half* __restrict__ codebook4,
    float inv_resid_scale3,
    float wscale,
    int M,
    int N_BLOCKS,
    int bps,
    int cb2_size
) {
    extern __shared__ half smem_cb2[];

    if (NUM_STAGES >= 2 && cb2_size > 0 && cb2_size <= 256) {
        int tid_flat = threadIdx.y * 32 + threadIdx.x;
        int total = cb2_size * 8;
        for (int i = tid_flat; i < total; i += 256) {
            smem_cb2[i] = codebook2[i];
        }
        __syncthreads();
    }

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int row_in_warp = lane_id >> 3;
    const int elem = lane_id & 7;

    int base_row = (blockIdx.x * blockDim.y + warp_id) * ROWS_PER_WARP;
    int my_row = base_row + row_in_warp;
    bool valid = (my_row < M);

    int j_start = blockIdx.y * bps;
    int j_end = min(j_start + bps, N_BLOCKS);

    float acc = 0.0f;

    int j = j_start;
    for (; j + 1 < j_end; j += 2) {
        float x_val_0, x_val_1;
        if (lane_id < 8) {
            x_val_0 = __half2float(x[j * 8 + lane_id]);
            x_val_1 = __half2float(x[(j + 1) * 8 + lane_id]);
        }
        x_val_0 = __shfl_sync(FULL_MASK, x_val_0, elem);
        x_val_1 = __shfl_sync(FULL_MASK, x_val_1, elem);

        uint16_t idx_0 = 0, idx_1 = 0;
        if (valid && elem == 0) {
            idx_0 = (uint16_t)qidxs[my_row * N_BLOCKS + j];
            idx_1 = (uint16_t)qidxs[my_row * N_BLOCKS + j + 1];
        }
        idx_0 = __shfl_sync(FULL_MASK, idx_0, row_in_warp * 8);
        idx_1 = __shfl_sync(FULL_MASK, idx_1, row_in_warp * 8);

        float cb_val_0 = 0.0f, cb_val_1 = 0.0f;
        if (valid) {
            cb_val_0 = __half2float(codebook[idx_0 * 8 + elem]);
            cb_val_1 = __half2float(codebook[idx_1 * 8 + elem]);
        }

        if constexpr (NUM_STAGES >= 2) {
            const half* cb2 = (cb2_size > 0 && cb2_size <= 256) ? smem_cb2 : codebook2;
            uint16_t idx2_0 = 0, idx2_1 = 0;
            if (valid && elem == 0) {
                idx2_0 = (uint16_t)qidxs2[my_row * N_BLOCKS + j];
                idx2_1 = (uint16_t)qidxs2[my_row * N_BLOCKS + j + 1];
            }
            idx2_0 = __shfl_sync(FULL_MASK, idx2_0, row_in_warp * 8);
            idx2_1 = __shfl_sync(FULL_MASK, idx2_1, row_in_warp * 8);
            if (valid) {
                cb_val_0 += __half2float(cb2[idx2_0 * 8 + elem]) * inv_resid_scale;
                cb_val_1 += __half2float(cb2[idx2_1 * 8 + elem]) * inv_resid_scale;
            }
        }
        if constexpr (NUM_STAGES >= 3) {
            uint16_t idx3_0 = 0, idx3_1 = 0;
            if (valid && elem == 0) {
                idx3_0 = (uint16_t)qidxs3[my_row * N_BLOCKS + j];
                idx3_1 = (uint16_t)qidxs3[my_row * N_BLOCKS + j + 1];
            }
            idx3_0 = __shfl_sync(FULL_MASK, idx3_0, row_in_warp * 8);
            idx3_1 = __shfl_sync(FULL_MASK, idx3_1, row_in_warp * 8);
            if (valid) {
                cb_val_0 += __half2float(codebook3[idx3_0 * 8 + elem]) * inv_resid_scale2;
                cb_val_1 += __half2float(codebook3[idx3_1 * 8 + elem]) * inv_resid_scale2;
            }
        }
        if constexpr (NUM_STAGES >= 4) {
            uint16_t idx4_0 = 0, idx4_1 = 0;
            if (valid && elem == 0) {
                idx4_0 = (uint16_t)qidxs4[my_row * N_BLOCKS + j];
                idx4_1 = (uint16_t)qidxs4[my_row * N_BLOCKS + j + 1];
            }
            idx4_0 = __shfl_sync(FULL_MASK, idx4_0, row_in_warp * 8);
            idx4_1 = __shfl_sync(FULL_MASK, idx4_1, row_in_warp * 8);
            if (valid) {
                cb_val_0 += __half2float(codebook4[idx4_0 * 8 + elem]) * inv_resid_scale3;
                cb_val_1 += __half2float(codebook4[idx4_1 * 8 + elem]) * inv_resid_scale3;
            }
        }

        float prod_0 = cb_val_0 * x_val_0;
        prod_0 += __shfl_xor_sync(FULL_MASK, prod_0, 4);
        prod_0 += __shfl_xor_sync(FULL_MASK, prod_0, 2);
        prod_0 += __shfl_xor_sync(FULL_MASK, prod_0, 1);

        float prod_1 = cb_val_1 * x_val_1;
        prod_1 += __shfl_xor_sync(FULL_MASK, prod_1, 4);
        prod_1 += __shfl_xor_sync(FULL_MASK, prod_1, 2);
        prod_1 += __shfl_xor_sync(FULL_MASK, prod_1, 1);

        if (elem == 0) {
            acc += prod_0 + prod_1;
        }
    }
    if (j < j_end) {
        float x_val;
        if (lane_id < 8) { x_val = __half2float(x[j * 8 + lane_id]); }
        x_val = __shfl_sync(FULL_MASK, x_val, elem);
        uint16_t idx = 0;
        if (valid && elem == 0) { idx = (uint16_t)qidxs[my_row * N_BLOCKS + j]; }
        idx = __shfl_sync(FULL_MASK, idx, row_in_warp * 8);
        float cb_val = 0.0f;
        if (valid) { cb_val = __half2float(codebook[idx * 8 + elem]); }
        if constexpr (NUM_STAGES >= 2) {
            const half* cb2 = (cb2_size > 0 && cb2_size <= 256) ? smem_cb2 : codebook2;
            uint16_t idx2 = 0;
            if (valid && elem == 0) { idx2 = (uint16_t)qidxs2[my_row * N_BLOCKS + j]; }
            idx2 = __shfl_sync(FULL_MASK, idx2, row_in_warp * 8);
            if (valid) { cb_val += __half2float(cb2[idx2 * 8 + elem]) * inv_resid_scale; }
        }
        if constexpr (NUM_STAGES >= 3) {
            uint16_t idx3 = 0;
            if (valid && elem == 0) { idx3 = (uint16_t)qidxs3[my_row * N_BLOCKS + j]; }
            idx3 = __shfl_sync(FULL_MASK, idx3, row_in_warp * 8);
            if (valid) { cb_val += __half2float(codebook3[idx3 * 8 + elem]) * inv_resid_scale2; }
        }
        if constexpr (NUM_STAGES >= 4) {
            uint16_t idx4 = 0;
            if (valid && elem == 0) { idx4 = (uint16_t)qidxs4[my_row * N_BLOCKS + j]; }
            idx4 = __shfl_sync(FULL_MASK, idx4, row_in_warp * 8);
            if (valid) { cb_val += __half2float(codebook4[idx4 * 8 + elem]) * inv_resid_scale3; }
        }
        float prod = cb_val * x_val;
        prod += __shfl_xor_sync(FULL_MASK, prod, 4);
        prod += __shfl_xor_sync(FULL_MASK, prod, 2);
        prod += __shfl_xor_sync(FULL_MASK, prod, 1);
        if (elem == 0) { acc += prod; }
    }

    // Deterministic: each (k_split, m) slot is written by exactly one CTA.
    // No atomics, no race. Scratch is uninitialized; we write it here.
    if (elem == 0 && valid) {
        scratch[blockIdx.y * M + my_row] = acc * wscale;
    }
}


/* Deterministic reduction: sum scratch[0:k_splits, m] → output[m] in
 * fixed loop order. One thread per output row, 256-thread blocks. */
__global__ void __launch_bounds__(256)
glq_reduce_splits_kernel(
    float* __restrict__ output,
    const float* __restrict__ scratch,
    int M,
    int k_splits
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    float sum = 0.0f;
    for (int k = 0; k < k_splits; ++k) {
        sum += scratch[k * M + m];
    }
    output[m] = sum;
}


/* ─────────────────────────────────────────────────────────────────────
 * Tensor Core matmul kernel (B>=2) — inline PTX mma.m16n8k16
 *
 * Y(B×M) = X(B×N) @ dequant(W)^T(N×M) * Wscale
 *
 * Direct register loading — no shared memory staging needed.
 * Each thread loads its specific codebook and x elements directly
 * into the mma register layout per the PTX ISA spec:
 *
 * groupID = laneid / 4  (0..7, selects output row for B and batch pair for A)
 * tid_g   = laneid % 4  (0..3, selects k-pair within each register)
 *
 * B fragment: thread loads from ONE output row (m_start + groupID)
 *   reg0 = {cb[idx_j][2*tid_g], cb[idx_j][2*tid_g+1]}
 *   reg1 = {cb[idx_j1][2*tid_g], cb[idx_j1][2*tid_g+1]}
 *
 * A fragment: thread loads from TWO batch rows (groupID, groupID+8)
 *   reg0 = {x[b+groupID,   j*8+2*tid_g],     x[b+groupID,   j*8+2*tid_g+1]}
 *   reg1 = {x[b+groupID+8, j*8+2*tid_g],     x[b+groupID+8, j*8+2*tid_g+1]}
 *   reg2 = {x[b+groupID,   (j+1)*8+2*tid_g], x[b+groupID,   (j+1)*8+2*tid_g+1]}
 *   reg3 = {x[b+groupID+8, (j+1)*8+2*tid_g], x[b+groupID+8, (j+1)*8+2*tid_g+1]}
 *
 * Output: z0..z3 map to Y[b+groupID, m+2*tid_g], Y[b+groupID, m+2*tid_g+1],
 *         Y[b+groupID+8, m+2*tid_g], Y[b+groupID+8, m+2*tid_g+1]
 *
 * Grid: 3D (ceil(B/16), ceil(M/(8*WARPS)), ceil(N_BLOCKS/TC_BPS))
 * Block: (32, WARPS)
 * ───────────────────────────────────────────────────────────────────── */

#define TC_BPS_DEFAULT 64

template <int NUM_STAGES>
__global__ void __launch_bounds__(256, 2)
glq_matmul_tc_kernel(
    float* __restrict__ output,         // (B_dim, M) fp32, pre-zeroed
    const half* __restrict__ x,         // (B_dim, N) fp16 input
    const int16_t* __restrict__ qidxs,  // (M, N_BLOCKS) primary indices
    const half* __restrict__ codebook,  // (65536, 8) fp16
    const int16_t* __restrict__ qidxs2,
    const half* __restrict__ codebook2,
    float inv_resid_scale,
    const int16_t* __restrict__ qidxs3,
    const half* __restrict__ codebook3,
    float inv_resid_scale2,
    const int16_t* __restrict__ qidxs4,
    const half* __restrict__ codebook4,
    float inv_resid_scale3,
    float wscale,
    int B_dim,
    int M,
    int N,
    int N_BLOCKS,
    int stride_x,
    int bps,                            // blocks per K-split (adaptive)
    int cb2_size                        // secondary codebook entries (0 if no stage2)
) {
    // Stage small secondary codebook into shared memory (256 entries = 4KB).
    // Stages 3+ always use the primary (65536-entry) codebook — no smem staging.
    extern __shared__ half smem_cb2[];
    if (NUM_STAGES >= 2 && cb2_size > 0 && cb2_size <= 256) {
        int tid_flat = threadIdx.y * 32 + threadIdx.x;
        int total = cb2_size * 8;
        for (int i = tid_flat; i < total; i += 256) {
            smem_cb2[i] = codebook2[i];
        }
        __syncthreads();
    }

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int groupID = lane_id >> 2;           // 0..7
    const int tid_g = lane_id & 3;              // 0..3

    int b_start = blockIdx.x * 16;
    int m_start = (blockIdx.y * blockDim.y + warp_id) * 8;
    int k_split = blockIdx.z;

    if (b_start >= B_dim || m_start >= M) return;

    int j_start = k_split * bps;
    int j_end = min(j_start + bps, N_BLOCKS);

    // Accumulators
    float z0 = 0.0f, z1 = 0.0f, z2 = 0.0f, z3 = 0.0f;

    // Pre-compute row indices
    int b_row0 = b_start + groupID;       // batch row for this thread (top half)
    int b_row1 = b_start + groupID + 8;   // batch row (bottom half)
    int m_row = m_start + groupID;        // output row for B fragment
    bool b0_valid = (b_row0 < B_dim);
    bool b1_valid = (b_row1 < B_dim);
    bool m_valid = (m_row < M);
    int k_elem = tid_g * 2;              // k-position pair: 0,2,4,6

    for (int j = j_start; j < j_end; j += 2) {
        bool j1_valid = (j + 1 < N_BLOCKS);

        // ---- B fragment: codebook gather for output row m_row ----
        uint16_t idx_j = 0, idx_j1 = 0;
        if (m_valid) {
            idx_j = (uint16_t)qidxs[m_row * N_BLOCKS + j];
            if (j1_valid) idx_j1 = (uint16_t)qidxs[m_row * N_BLOCKS + j + 1];
        }

        half bv0 = m_valid ? codebook[idx_j * 8 + k_elem]     : __float2half(0.0f);
        half bv1 = m_valid ? codebook[idx_j * 8 + k_elem + 1] : __float2half(0.0f);
        half bv2 = (m_valid && j1_valid) ? codebook[idx_j1 * 8 + k_elem]     : __float2half(0.0f);
        half bv3 = (m_valid && j1_valid) ? codebook[idx_j1 * 8 + k_elem + 1] : __float2half(0.0f);

        if constexpr (NUM_STAGES >= 2) {
            if (m_valid) {
                // Use shared memory for small codebook2, global for large
                const half* cb2 = (cb2_size > 0 && cb2_size <= 256) ? smem_cb2 : codebook2;
                uint16_t idx2_j = (uint16_t)qidxs2[m_row * N_BLOCKS + j];
                bv0 = __float2half(__half2float(bv0) + __half2float(cb2[idx2_j * 8 + k_elem]) * inv_resid_scale);
                bv1 = __float2half(__half2float(bv1) + __half2float(cb2[idx2_j * 8 + k_elem + 1]) * inv_resid_scale);
                if (j1_valid) {
                    uint16_t idx2_j1 = (uint16_t)qidxs2[m_row * N_BLOCKS + j + 1];
                    bv2 = __float2half(__half2float(bv2) + __half2float(cb2[idx2_j1 * 8 + k_elem]) * inv_resid_scale);
                    bv3 = __float2half(__half2float(bv3) + __half2float(cb2[idx2_j1 * 8 + k_elem + 1]) * inv_resid_scale);
                }
            }
        }
        if constexpr (NUM_STAGES >= 3) {
            if (m_valid) {
                uint16_t idx3_j = (uint16_t)qidxs3[m_row * N_BLOCKS + j];
                bv0 = __float2half(__half2float(bv0) + __half2float(codebook3[idx3_j * 8 + k_elem]) * inv_resid_scale2);
                bv1 = __float2half(__half2float(bv1) + __half2float(codebook3[idx3_j * 8 + k_elem + 1]) * inv_resid_scale2);
                if (j1_valid) {
                    uint16_t idx3_j1 = (uint16_t)qidxs3[m_row * N_BLOCKS + j + 1];
                    bv2 = __float2half(__half2float(bv2) + __half2float(codebook3[idx3_j1 * 8 + k_elem]) * inv_resid_scale2);
                    bv3 = __float2half(__half2float(bv3) + __half2float(codebook3[idx3_j1 * 8 + k_elem + 1]) * inv_resid_scale2);
                }
            }
        }
        if constexpr (NUM_STAGES >= 4) {
            if (m_valid) {
                uint16_t idx4_j = (uint16_t)qidxs4[m_row * N_BLOCKS + j];
                bv0 = __float2half(__half2float(bv0) + __half2float(codebook4[idx4_j * 8 + k_elem]) * inv_resid_scale3);
                bv1 = __float2half(__half2float(bv1) + __half2float(codebook4[idx4_j * 8 + k_elem + 1]) * inv_resid_scale3);
                if (j1_valid) {
                    uint16_t idx4_j1 = (uint16_t)qidxs4[m_row * N_BLOCKS + j + 1];
                    bv2 = __float2half(__half2float(bv2) + __half2float(codebook4[idx4_j1 * 8 + k_elem]) * inv_resid_scale3);
                    bv3 = __float2half(__half2float(bv3) + __half2float(codebook4[idx4_j1 * 8 + k_elem + 1]) * inv_resid_scale3);
                }
            }
        }

        __half2 b_h0 = __halves2half2(bv0, bv1);
        __half2 b_h1 = __halves2half2(bv2, bv3);
        uint32_t b_reg0 = *reinterpret_cast<uint32_t*>(&b_h0);
        uint32_t b_reg1 = *reinterpret_cast<uint32_t*>(&b_h1);

        // ---- A fragment: x values from 2 batch rows ----
        // Block j: x[b_row, j*8 + k_elem], x[b_row, j*8 + k_elem + 1]
        int x_off_j = j * 8 + k_elem;
        int x_off_j1 = (j + 1) * 8 + k_elem;

        half av0 = b0_valid ? x[b_row0 * stride_x + x_off_j]     : __float2half(0.0f);
        half av1 = b0_valid ? x[b_row0 * stride_x + x_off_j + 1] : __float2half(0.0f);
        half av2 = b1_valid ? x[b_row1 * stride_x + x_off_j]     : __float2half(0.0f);
        half av3 = b1_valid ? x[b_row1 * stride_x + x_off_j + 1] : __float2half(0.0f);

        half av4 = (b0_valid && j1_valid) ? x[b_row0 * stride_x + x_off_j1]     : __float2half(0.0f);
        half av5 = (b0_valid && j1_valid) ? x[b_row0 * stride_x + x_off_j1 + 1] : __float2half(0.0f);
        half av6 = (b1_valid && j1_valid) ? x[b_row1 * stride_x + x_off_j1]     : __float2half(0.0f);
        half av7 = (b1_valid && j1_valid) ? x[b_row1 * stride_x + x_off_j1 + 1] : __float2half(0.0f);

        __half2 a_h0 = __halves2half2(av0, av1);
        __half2 a_h1 = __halves2half2(av2, av3);
        __half2 a_h2 = __halves2half2(av4, av5);
        __half2 a_h3 = __halves2half2(av6, av7);

        uint32_t a_reg0 = *reinterpret_cast<uint32_t*>(&a_h0);
        uint32_t a_reg1 = *reinterpret_cast<uint32_t*>(&a_h1);
        uint32_t a_reg2 = *reinterpret_cast<uint32_t*>(&a_h2);
        uint32_t a_reg3 = *reinterpret_cast<uint32_t*>(&a_h3);

        // ---- mma.sync.aligned.m16n8k16 ----
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            " { %0, %1, %2, %3 },"
            " { %4, %5, %6, %7 },"
            " { %8, %9 },"
            " { %0, %1, %2, %3 };"
            : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
            : "r"(a_reg0), "r"(a_reg1), "r"(a_reg2), "r"(a_reg3),
              "r"(b_reg0), "r"(b_reg1)
        );
    }

    // ---- Write output: z0..z3 → Y[b_row, m_col] ----
    // z0 = Y[b_row0, m_start + 2*tid_g]
    // z1 = Y[b_row0, m_start + 2*tid_g + 1]
    // z2 = Y[b_row1, m_start + 2*tid_g]
    // z3 = Y[b_row1, m_start + 2*tid_g + 1]
    int m_col0 = m_start + tid_g * 2;
    int m_col1 = m_col0 + 1;

    if (b0_valid && m_col0 < M)
        atomicAdd(&output[b_row0 * M + m_col0], z0 * wscale);
    if (b0_valid && m_col1 < M)
        atomicAdd(&output[b_row0 * M + m_col1], z1 * wscale);
    if (b1_valid && m_col0 < M)
        atomicAdd(&output[b_row1 * M + m_col0], z2 * wscale);
    if (b1_valid && m_col1 < M)
        atomicAdd(&output[b_row1 * M + m_col1], z3 * wscale);
}


/* ─────────────────────────────────────────────────────────────────────
 * Deterministic scratch variant of glq_matmul_tc_kernel.
 * scratch layout: (k_splits, B_dim, M). Each thread writes 4 unique
 * (k_split, b_row, m_col) slots → no atomic, no race. A follow-up
 * glq_reduce_splits_2d_kernel sums across the k_splits axis in fixed
 * order, giving bit-exact run-to-run results.
 * ───────────────────────────────────────────────────────────────────── */
template <int NUM_STAGES>
__global__ void __launch_bounds__(256, 2)
glq_matmul_tc_scratch_kernel(
    float* __restrict__ scratch,        // (k_splits * B_dim * M) fp32, overwritten
    const half* __restrict__ x,
    const int16_t* __restrict__ qidxs,
    const half* __restrict__ codebook,
    const int16_t* __restrict__ qidxs2,
    const half* __restrict__ codebook2,
    float inv_resid_scale,
    const int16_t* __restrict__ qidxs3,
    const half* __restrict__ codebook3,
    float inv_resid_scale2,
    const int16_t* __restrict__ qidxs4,
    const half* __restrict__ codebook4,
    float inv_resid_scale3,
    float wscale,
    int B_dim,
    int M,
    int N,
    int N_BLOCKS,
    int stride_x,
    int bps,
    int cb2_size
) {
    extern __shared__ half smem_cb2[];
    if (NUM_STAGES >= 2 && cb2_size > 0 && cb2_size <= 256) {
        int tid_flat = threadIdx.y * 32 + threadIdx.x;
        int total = cb2_size * 8;
        for (int i = tid_flat; i < total; i += 256) {
            smem_cb2[i] = codebook2[i];
        }
        __syncthreads();
    }

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int groupID = lane_id >> 2;
    const int tid_g = lane_id & 3;

    int b_start = blockIdx.x * 16;
    int m_start = (blockIdx.y * blockDim.y + warp_id) * 8;
    int k_split = blockIdx.z;

    if (b_start >= B_dim || m_start >= M) return;

    int j_start = k_split * bps;
    int j_end = min(j_start + bps, N_BLOCKS);

    float z0 = 0.0f, z1 = 0.0f, z2 = 0.0f, z3 = 0.0f;

    int b_row0 = b_start + groupID;
    int b_row1 = b_start + groupID + 8;
    int m_row = m_start + groupID;
    bool b0_valid = (b_row0 < B_dim);
    bool b1_valid = (b_row1 < B_dim);
    bool m_valid = (m_row < M);
    int k_elem = tid_g * 2;

    for (int j = j_start; j < j_end; j += 2) {
        bool j1_valid = (j + 1 < N_BLOCKS);

        uint16_t idx_j = 0, idx_j1 = 0;
        if (m_valid) {
            idx_j = (uint16_t)qidxs[m_row * N_BLOCKS + j];
            if (j1_valid) idx_j1 = (uint16_t)qidxs[m_row * N_BLOCKS + j + 1];
        }

        half bv0 = m_valid ? codebook[idx_j * 8 + k_elem]     : __float2half(0.0f);
        half bv1 = m_valid ? codebook[idx_j * 8 + k_elem + 1] : __float2half(0.0f);
        half bv2 = (m_valid && j1_valid) ? codebook[idx_j1 * 8 + k_elem]     : __float2half(0.0f);
        half bv3 = (m_valid && j1_valid) ? codebook[idx_j1 * 8 + k_elem + 1] : __float2half(0.0f);

        if constexpr (NUM_STAGES >= 2) {
            if (m_valid) {
                const half* cb2 = (cb2_size > 0 && cb2_size <= 256) ? smem_cb2 : codebook2;
                uint16_t idx2_j = (uint16_t)qidxs2[m_row * N_BLOCKS + j];
                bv0 = __float2half(__half2float(bv0) + __half2float(cb2[idx2_j * 8 + k_elem]) * inv_resid_scale);
                bv1 = __float2half(__half2float(bv1) + __half2float(cb2[idx2_j * 8 + k_elem + 1]) * inv_resid_scale);
                if (j1_valid) {
                    uint16_t idx2_j1 = (uint16_t)qidxs2[m_row * N_BLOCKS + j + 1];
                    bv2 = __float2half(__half2float(bv2) + __half2float(cb2[idx2_j1 * 8 + k_elem]) * inv_resid_scale);
                    bv3 = __float2half(__half2float(bv3) + __half2float(cb2[idx2_j1 * 8 + k_elem + 1]) * inv_resid_scale);
                }
            }
        }
        if constexpr (NUM_STAGES >= 3) {
            if (m_valid) {
                uint16_t idx3_j = (uint16_t)qidxs3[m_row * N_BLOCKS + j];
                bv0 = __float2half(__half2float(bv0) + __half2float(codebook3[idx3_j * 8 + k_elem]) * inv_resid_scale2);
                bv1 = __float2half(__half2float(bv1) + __half2float(codebook3[idx3_j * 8 + k_elem + 1]) * inv_resid_scale2);
                if (j1_valid) {
                    uint16_t idx3_j1 = (uint16_t)qidxs3[m_row * N_BLOCKS + j + 1];
                    bv2 = __float2half(__half2float(bv2) + __half2float(codebook3[idx3_j1 * 8 + k_elem]) * inv_resid_scale2);
                    bv3 = __float2half(__half2float(bv3) + __half2float(codebook3[idx3_j1 * 8 + k_elem + 1]) * inv_resid_scale2);
                }
            }
        }
        if constexpr (NUM_STAGES >= 4) {
            if (m_valid) {
                uint16_t idx4_j = (uint16_t)qidxs4[m_row * N_BLOCKS + j];
                bv0 = __float2half(__half2float(bv0) + __half2float(codebook4[idx4_j * 8 + k_elem]) * inv_resid_scale3);
                bv1 = __float2half(__half2float(bv1) + __half2float(codebook4[idx4_j * 8 + k_elem + 1]) * inv_resid_scale3);
                if (j1_valid) {
                    uint16_t idx4_j1 = (uint16_t)qidxs4[m_row * N_BLOCKS + j + 1];
                    bv2 = __float2half(__half2float(bv2) + __half2float(codebook4[idx4_j1 * 8 + k_elem]) * inv_resid_scale3);
                    bv3 = __float2half(__half2float(bv3) + __half2float(codebook4[idx4_j1 * 8 + k_elem + 1]) * inv_resid_scale3);
                }
            }
        }

        __half2 b_h0 = __halves2half2(bv0, bv1);
        __half2 b_h1 = __halves2half2(bv2, bv3);
        uint32_t b_reg0 = *reinterpret_cast<uint32_t*>(&b_h0);
        uint32_t b_reg1 = *reinterpret_cast<uint32_t*>(&b_h1);

        int x_off_j = j * 8 + k_elem;
        int x_off_j1 = (j + 1) * 8 + k_elem;

        half av0 = b0_valid ? x[b_row0 * stride_x + x_off_j]     : __float2half(0.0f);
        half av1 = b0_valid ? x[b_row0 * stride_x + x_off_j + 1] : __float2half(0.0f);
        half av2 = b1_valid ? x[b_row1 * stride_x + x_off_j]     : __float2half(0.0f);
        half av3 = b1_valid ? x[b_row1 * stride_x + x_off_j + 1] : __float2half(0.0f);

        half av4 = (b0_valid && j1_valid) ? x[b_row0 * stride_x + x_off_j1]     : __float2half(0.0f);
        half av5 = (b0_valid && j1_valid) ? x[b_row0 * stride_x + x_off_j1 + 1] : __float2half(0.0f);
        half av6 = (b1_valid && j1_valid) ? x[b_row1 * stride_x + x_off_j1]     : __float2half(0.0f);
        half av7 = (b1_valid && j1_valid) ? x[b_row1 * stride_x + x_off_j1 + 1] : __float2half(0.0f);

        __half2 a_h0 = __halves2half2(av0, av1);
        __half2 a_h1 = __halves2half2(av2, av3);
        __half2 a_h2 = __halves2half2(av4, av5);
        __half2 a_h3 = __halves2half2(av6, av7);

        uint32_t a_reg0 = *reinterpret_cast<uint32_t*>(&a_h0);
        uint32_t a_reg1 = *reinterpret_cast<uint32_t*>(&a_h1);
        uint32_t a_reg2 = *reinterpret_cast<uint32_t*>(&a_h2);
        uint32_t a_reg3 = *reinterpret_cast<uint32_t*>(&a_h3);

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            " { %0, %1, %2, %3 },"
            " { %4, %5, %6, %7 },"
            " { %8, %9 },"
            " { %0, %1, %2, %3 };"
            : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
            : "r"(a_reg0), "r"(a_reg1), "r"(a_reg2), "r"(a_reg3),
              "r"(b_reg0), "r"(b_reg1)
        );
    }

    // Deterministic: each (k_split, b_row, m_col) slot has exactly one writer.
    int m_col0 = m_start + tid_g * 2;
    int m_col1 = m_col0 + 1;
    size_t k_base = (size_t)k_split * B_dim * M;

    if (b0_valid && m_col0 < M)
        scratch[k_base + (size_t)b_row0 * M + m_col0] = z0 * wscale;
    if (b0_valid && m_col1 < M)
        scratch[k_base + (size_t)b_row0 * M + m_col1] = z1 * wscale;
    if (b1_valid && m_col0 < M)
        scratch[k_base + (size_t)b_row1 * M + m_col0] = z2 * wscale;
    if (b1_valid && m_col1 < M)
        scratch[k_base + (size_t)b_row1 * M + m_col1] = z3 * wscale;
}


/* Deterministic 2D reduction: sum scratch[0:k_splits, 0:B_dim*M] →
 * output[0:B_dim*M] in fixed loop order. One thread per output element. */
__global__ void __launch_bounds__(256)
glq_reduce_splits_2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ scratch,
    int BM,       // B_dim * M
    int k_splits
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BM) return;
    float sum = 0.0f;
    for (int k = 0; k < k_splits; ++k) {
        sum += scratch[(size_t)k * BM + i];
    }
    output[i] = sum;
}


/* ─────────────────────────────────────────────────────────────────────
 * Packed TC matmul kernel (B>=2) — uint32 codebook, 2bpw only
 *
 * Same mma.m16n8k16 layout as glq_matmul_tc_kernel but gathers from
 * packed uint32 codebook (4 bytes vs 16 bytes per entry = 4× less BW).
 * Decode: nibble = (packed >> (i*4)) & 0xF; val = nibble * 0.5 - 3.0
 * ───────────────────────────────────────────────────────────────────── */

__global__ void __launch_bounds__(256, 2)
glq_matmul_tc_packed_kernel(
    float* __restrict__ output,
    const half* __restrict__ x,
    const int16_t* __restrict__ qidxs,
    const uint32_t* __restrict__ codebook_packed,
    float wscale,
    int B_dim, int M, int N, int N_BLOCKS, int stride_x, int bps
) {
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int groupID = lane_id >> 2;
    const int tid_g = lane_id & 3;

    int b_start = blockIdx.x * 16;
    int m_start = (blockIdx.y * blockDim.y + warp_id) * 8;
    int k_split = blockIdx.z;

    if (b_start >= B_dim || m_start >= M) return;

    int j_start = k_split * bps;
    int j_end = min(j_start + bps, N_BLOCKS);

    float z0 = 0.0f, z1 = 0.0f, z2 = 0.0f, z3 = 0.0f;

    int b_row0 = b_start + groupID;
    int b_row1 = b_start + groupID + 8;
    int m_row = m_start + groupID;
    bool b0_valid = (b_row0 < B_dim);
    bool b1_valid = (b_row1 < B_dim);
    bool m_valid = (m_row < M);
    int k_elem = tid_g * 2;

    for (int j = j_start; j < j_end; j += 2) {
        bool j1_valid = (j + 1 < N_BLOCKS);

        // B fragment: packed codebook gather + nibble decode
        uint16_t idx_j = 0, idx_j1 = 0;
        if (m_valid) {
            idx_j = (uint16_t)qidxs[m_row * N_BLOCKS + j];
            if (j1_valid) idx_j1 = (uint16_t)qidxs[m_row * N_BLOCKS + j + 1];
        }

        uint32_t packed_j = m_valid ? codebook_packed[idx_j] : 0;
        uint32_t packed_j1 = (m_valid && j1_valid) ? codebook_packed[idx_j1] : 0;

        half bv0 = __float2half(((packed_j >> (k_elem * 4)) & 0xF) * 0.5f - 3.0f);
        half bv1 = __float2half(((packed_j >> ((k_elem + 1) * 4)) & 0xF) * 0.5f - 3.0f);
        half bv2 = __float2half(((packed_j1 >> (k_elem * 4)) & 0xF) * 0.5f - 3.0f);
        half bv3 = __float2half(((packed_j1 >> ((k_elem + 1) * 4)) & 0xF) * 0.5f - 3.0f);

        __half2 b_h0 = __halves2half2(bv0, bv1);
        __half2 b_h1 = __halves2half2(bv2, bv3);
        uint32_t b_reg0 = *reinterpret_cast<uint32_t*>(&b_h0);
        uint32_t b_reg1 = *reinterpret_cast<uint32_t*>(&b_h1);

        // A fragment: x values from 2 batch rows
        int x_off_j = j * 8 + k_elem;
        int x_off_j1 = (j + 1) * 8 + k_elem;

        half a00 = b0_valid ? x[b_row0 * stride_x + x_off_j]     : __float2half(0.0f);
        half a01 = b0_valid ? x[b_row0 * stride_x + x_off_j + 1] : __float2half(0.0f);
        half a10 = b1_valid ? x[b_row1 * stride_x + x_off_j]     : __float2half(0.0f);
        half a11 = b1_valid ? x[b_row1 * stride_x + x_off_j + 1] : __float2half(0.0f);
        half a02 = (b0_valid && j1_valid) ? x[b_row0 * stride_x + x_off_j1]     : __float2half(0.0f);
        half a03 = (b0_valid && j1_valid) ? x[b_row0 * stride_x + x_off_j1 + 1] : __float2half(0.0f);
        half a12 = (b1_valid && j1_valid) ? x[b_row1 * stride_x + x_off_j1]     : __float2half(0.0f);
        half a13 = (b1_valid && j1_valid) ? x[b_row1 * stride_x + x_off_j1 + 1] : __float2half(0.0f);

        __half2 ah0 = __halves2half2(a00, a01);
        __half2 ah1 = __halves2half2(a10, a11);
        __half2 ah2 = __halves2half2(a02, a03);
        __half2 ah3 = __halves2half2(a12, a13);
        uint32_t a_reg0 = *reinterpret_cast<uint32_t*>(&ah0);
        uint32_t a_reg1 = *reinterpret_cast<uint32_t*>(&ah1);
        uint32_t a_reg2 = *reinterpret_cast<uint32_t*>(&ah2);
        uint32_t a_reg3 = *reinterpret_cast<uint32_t*>(&ah3);

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            " { %0, %1, %2, %3 },"
            " { %4, %5, %6, %7 },"
            " { %8, %9 },"
            " { %0, %1, %2, %3 };"
            : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
            : "r"(a_reg0), "r"(a_reg1), "r"(a_reg2), "r"(a_reg3),
              "r"(b_reg0), "r"(b_reg1)
        );
    }

    int m_col0 = m_start + tid_g * 2;
    int m_col1 = m_col0 + 1;

    if (b0_valid && m_col0 < M)
        atomicAdd(&output[b_row0 * M + m_col0], z0 * wscale);
    if (b0_valid && m_col1 < M)
        atomicAdd(&output[b_row0 * M + m_col1], z1 * wscale);
    if (b1_valid && m_col0 < M)
        atomicAdd(&output[b_row1 * M + m_col0], z2 * wscale);
    if (b1_valid && m_col1 < M)
        atomicAdd(&output[b_row1 * M + m_col1], z3 * wscale);
}


/* ─────────────────────────────────────────────────────────────────────
 * Deterministic scratch variant of glq_matmul_tc_packed_kernel.
 * Same pattern as glq_matmul_tc_scratch_kernel but using the packed
 * uint32 codebook. 2bpw only, no stage2.
 * ───────────────────────────────────────────────────────────────────── */
__global__ void __launch_bounds__(256, 2)
glq_matmul_tc_packed_scratch_kernel(
    float* __restrict__ scratch,
    const half* __restrict__ x,
    const int16_t* __restrict__ qidxs,
    const uint32_t* __restrict__ codebook_packed,
    float wscale,
    int B_dim, int M, int N, int N_BLOCKS, int stride_x, int bps
) {
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int groupID = lane_id >> 2;
    const int tid_g = lane_id & 3;

    int b_start = blockIdx.x * 16;
    int m_start = (blockIdx.y * blockDim.y + warp_id) * 8;
    int k_split = blockIdx.z;

    if (b_start >= B_dim || m_start >= M) return;

    int j_start = k_split * bps;
    int j_end = min(j_start + bps, N_BLOCKS);

    float z0 = 0.0f, z1 = 0.0f, z2 = 0.0f, z3 = 0.0f;

    int b_row0 = b_start + groupID;
    int b_row1 = b_start + groupID + 8;
    int m_row = m_start + groupID;
    bool b0_valid = (b_row0 < B_dim);
    bool b1_valid = (b_row1 < B_dim);
    bool m_valid = (m_row < M);
    int k_elem = tid_g * 2;

    for (int j = j_start; j < j_end; j += 2) {
        bool j1_valid = (j + 1 < N_BLOCKS);

        uint16_t idx_j = 0, idx_j1 = 0;
        if (m_valid) {
            idx_j = (uint16_t)qidxs[m_row * N_BLOCKS + j];
            if (j1_valid) idx_j1 = (uint16_t)qidxs[m_row * N_BLOCKS + j + 1];
        }

        uint32_t packed_j = m_valid ? codebook_packed[idx_j] : 0;
        uint32_t packed_j1 = (m_valid && j1_valid) ? codebook_packed[idx_j1] : 0;

        half bv0 = __float2half(((packed_j >> (k_elem * 4)) & 0xF) * 0.5f - 3.0f);
        half bv1 = __float2half(((packed_j >> ((k_elem + 1) * 4)) & 0xF) * 0.5f - 3.0f);
        half bv2 = __float2half(((packed_j1 >> (k_elem * 4)) & 0xF) * 0.5f - 3.0f);
        half bv3 = __float2half(((packed_j1 >> ((k_elem + 1) * 4)) & 0xF) * 0.5f - 3.0f);

        __half2 b_h0 = __halves2half2(bv0, bv1);
        __half2 b_h1 = __halves2half2(bv2, bv3);
        uint32_t b_reg0 = *reinterpret_cast<uint32_t*>(&b_h0);
        uint32_t b_reg1 = *reinterpret_cast<uint32_t*>(&b_h1);

        int x_off_j = j * 8 + k_elem;
        int x_off_j1 = (j + 1) * 8 + k_elem;

        half a00 = b0_valid ? x[b_row0 * stride_x + x_off_j]     : __float2half(0.0f);
        half a01 = b0_valid ? x[b_row0 * stride_x + x_off_j + 1] : __float2half(0.0f);
        half a10 = b1_valid ? x[b_row1 * stride_x + x_off_j]     : __float2half(0.0f);
        half a11 = b1_valid ? x[b_row1 * stride_x + x_off_j + 1] : __float2half(0.0f);
        half a02 = (b0_valid && j1_valid) ? x[b_row0 * stride_x + x_off_j1]     : __float2half(0.0f);
        half a03 = (b0_valid && j1_valid) ? x[b_row0 * stride_x + x_off_j1 + 1] : __float2half(0.0f);
        half a12 = (b1_valid && j1_valid) ? x[b_row1 * stride_x + x_off_j1]     : __float2half(0.0f);
        half a13 = (b1_valid && j1_valid) ? x[b_row1 * stride_x + x_off_j1 + 1] : __float2half(0.0f);

        __half2 ah0 = __halves2half2(a00, a01);
        __half2 ah1 = __halves2half2(a10, a11);
        __half2 ah2 = __halves2half2(a02, a03);
        __half2 ah3 = __halves2half2(a12, a13);
        uint32_t a_reg0 = *reinterpret_cast<uint32_t*>(&ah0);
        uint32_t a_reg1 = *reinterpret_cast<uint32_t*>(&ah1);
        uint32_t a_reg2 = *reinterpret_cast<uint32_t*>(&ah2);
        uint32_t a_reg3 = *reinterpret_cast<uint32_t*>(&ah3);

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            " { %0, %1, %2, %3 },"
            " { %4, %5, %6, %7 },"
            " { %8, %9 },"
            " { %0, %1, %2, %3 };"
            : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
            : "r"(a_reg0), "r"(a_reg1), "r"(a_reg2), "r"(a_reg3),
              "r"(b_reg0), "r"(b_reg1)
        );
    }

    int m_col0 = m_start + tid_g * 2;
    int m_col1 = m_col0 + 1;
    size_t k_base = (size_t)k_split * B_dim * M;

    if (b0_valid && m_col0 < M)
        scratch[k_base + (size_t)b_row0 * M + m_col0] = z0 * wscale;
    if (b0_valid && m_col1 < M)
        scratch[k_base + (size_t)b_row0 * M + m_col1] = z1 * wscale;
    if (b1_valid && m_col0 < M)
        scratch[k_base + (size_t)b_row1 * M + m_col0] = z2 * wscale;
    if (b1_valid && m_col1 < M)
        scratch[k_base + (size_t)b_row1 * M + m_col1] = z3 * wscale;
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

    int j_start = blockIdx.y * BLOCKS_PER_SPLIT_DEFAULT;
    int j_end = min(j_start + BLOCKS_PER_SPLIT_DEFAULT, N_BLOCKS);

    float acc = 0.0f;

    // Unrolled-by-2: overlap packed codebook gathers
    int j = j_start;
    for (; j + 1 < j_end; j += 2) {
        float x_val_0, x_val_1;
        if (lane_id < 8) {
            x_val_0 = __half2float(x[j * 8 + lane_id]);
            x_val_1 = __half2float(x[(j + 1) * 8 + lane_id]);
        }
        x_val_0 = __shfl_sync(FULL_MASK, x_val_0, elem);
        x_val_1 = __shfl_sync(FULL_MASK, x_val_1, elem);

        uint16_t idx_0 = 0, idx_1 = 0;
        if (valid && elem == 0) {
            idx_0 = (uint16_t)qidxs[my_row * N_BLOCKS + j];
            idx_1 = (uint16_t)qidxs[my_row * N_BLOCKS + j + 1];
        }
        idx_0 = __shfl_sync(FULL_MASK, idx_0, row_in_warp * 8);
        idx_1 = __shfl_sync(FULL_MASK, idx_1, row_in_warp * 8);

        float cb_val_0 = 0.0f, cb_val_1 = 0.0f;
        if (valid) {
            uint32_t packed_0 = codebook_packed[idx_0];
            uint32_t packed_1 = codebook_packed[idx_1];
            cb_val_0 = (float)((packed_0 >> (elem * 4)) & 0xF) * 0.5f - 3.0f;
            cb_val_1 = (float)((packed_1 >> (elem * 4)) & 0xF) * 0.5f - 3.0f;
        }

        float prod_0 = cb_val_0 * x_val_0;
        prod_0 += __shfl_xor_sync(FULL_MASK, prod_0, 4);
        prod_0 += __shfl_xor_sync(FULL_MASK, prod_0, 2);
        prod_0 += __shfl_xor_sync(FULL_MASK, prod_0, 1);

        float prod_1 = cb_val_1 * x_val_1;
        prod_1 += __shfl_xor_sync(FULL_MASK, prod_1, 4);
        prod_1 += __shfl_xor_sync(FULL_MASK, prod_1, 2);
        prod_1 += __shfl_xor_sync(FULL_MASK, prod_1, 1);

        if (elem == 0) { acc += prod_0 + prod_1; }
    }
    // Tail
    if (j < j_end) {
        float x_val;
        if (lane_id < 8) { x_val = __half2float(x[j * 8 + lane_id]); }
        x_val = __shfl_sync(FULL_MASK, x_val, elem);
        uint16_t idx = 0;
        if (valid && elem == 0) { idx = (uint16_t)qidxs[my_row * N_BLOCKS + j]; }
        idx = __shfl_sync(FULL_MASK, idx, row_in_warp * 8);
        float cb_val = 0.0f;
        if (valid) {
            uint32_t packed = codebook_packed[idx];
            cb_val = (float)((packed >> (elem * 4)) & 0xF) * 0.5f - 3.0f;
        }
        float prod = cb_val * x_val;
        prod += __shfl_xor_sync(FULL_MASK, prod, 4);
        prod += __shfl_xor_sync(FULL_MASK, prod, 2);
        prod += __shfl_xor_sync(FULL_MASK, prod, 1);
        if (elem == 0) { acc += prod; }
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

// Dispatch macro: instantiate the templated kernel for NUM_STAGES ∈ {1..4}.
// KERNEL_CALL(NS) must be a statement that launches the kernel using NS
// as the NUM_STAGES template parameter.
#define DISPATCH_NUM_STAGES(NS_VAL, KERNEL_CALL) \
    do { \
        switch (NS_VAL) { \
            case 1: KERNEL_CALL(1); break; \
            case 2: KERNEL_CALL(2); break; \
            case 3: KERNEL_CALL(3); break; \
            case 4: KERNEL_CALL(4); break; \
            default: TORCH_CHECK(false, "num_stages must be 1-4"); \
        } \
    } while (0)

torch::Tensor glq_dequant_matvec_cuda(
    torch::Tensor x,           // (N,) fp16
    torch::Tensor qidxs,       // (M, N_BLOCKS) int16
    torch::Tensor codebook,    // (65536, 8) fp16
    float wscale,
    torch::Tensor qidxs2,      // (M, N_BLOCKS) int16 or empty
    torch::Tensor codebook2,   // (K2, 8) fp16 or empty
    float inv_resid_scale,
    torch::Tensor codebook_abs // unused, kept for ABI compat
) {
    CHECK_INPUT(x);
    CHECK_INPUT(qidxs);
    CHECK_INPUT(codebook);

    int M = qidxs.size(0);
    int N_BLOCKS = qidxs.size(1);

    at::DeviceGuard guard(x.device());
    // Output zeroed for the non-splitk branch (glq_matvec_kernel uses atomicAdd).
    // The splitk branch overwrites output via the reduction kernel, so the
    // zero-init is unnecessary there but the cost is negligible.
    auto output = torch::zeros({M}, torch::dtype(torch::kFloat32).device(x.device()));

    static int num_sms = 0;
    if (num_sms == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, x.get_device());
        num_sms = prop.multiProcessorCount;
    }

    const int WARPS = 8;
    const int rows_per_block = ROWS_PER_WARP * WARPS;  // 32
    dim3 block(32, WARPS);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    bool has_stage2 = (qidxs2.numel() > 0 && inv_resid_scale != 0.0f);

    int m_blocks = (M + rows_per_block - 1) / rows_per_block;
    bool use_splitk = (N_BLOCKS >= BLOCKS_PER_SPLIT_DEFAULT);

    if (use_splitk) {
        // Deterministic split-K via scratch buffer + reduction kernel.
        // Each CTA writes its partial sum to a unique (k_split, m) slot in
        // `scratch` with a plain store (no atomicAdd). A follow-up reduction
        // kernel sums scratch in fixed loop order → bit-exact across runs.
        // This preserves the SM-saturation benefit of adaptive split-K.
        int bps = BLOCKS_PER_SPLIT_DEFAULT;
        int k_splits = (N_BLOCKS + bps - 1) / bps;
        int total_ctas = m_blocks * k_splits;
        if (total_ctas < num_sms * 2 && bps > 16) {
            bps = max(16, bps / 2);
            k_splits = (N_BLOCKS + bps - 1) / bps;
        }
        dim3 grid(m_blocks, k_splits);

        // Allocate scratch via the caching allocator (stream-safe, fast reuse).
        size_t scratch_bytes = (size_t)k_splits * M * sizeof(float);
        float* scratch = (float*)c10::cuda::CUDACachingAllocator::raw_alloc(scratch_bytes);

        if (has_stage2) {
            CHECK_INPUT(qidxs2);
            CHECK_INPUT(codebook2);
            int cb2_size = codebook2.size(0);
            int smem = (cb2_size <= 256) ? cb2_size * 8 * sizeof(half) : 0;
            glq_matvec_splitk_scratch_kernel<2><<<grid, block, smem, stream>>>(
                scratch,
                (const half*)x.data_ptr<c10::Half>(),
                qidxs.data_ptr<int16_t>(),
                (const half*)codebook.data_ptr<c10::Half>(),
                qidxs2.data_ptr<int16_t>(),
                (const half*)codebook2.data_ptr<c10::Half>(),
                inv_resid_scale,
                nullptr, nullptr, 0.0f,
                nullptr, nullptr, 0.0f,
                wscale, M, N_BLOCKS, bps, cb2_size
            );
        } else {
            glq_matvec_splitk_scratch_kernel<1><<<grid, block, 0, stream>>>(
                scratch,
                (const half*)x.data_ptr<c10::Half>(),
                qidxs.data_ptr<int16_t>(),
                (const half*)codebook.data_ptr<c10::Half>(),
                nullptr, nullptr, 0.0f,
                nullptr, nullptr, 0.0f,
                nullptr, nullptr, 0.0f,
                wscale, M, N_BLOCKS, bps, 0
            );
        }

        // Deterministic reduction: sum scratch[0:k_splits, m] → output[m]
        int reduce_threads = 256;
        int reduce_blocks = (M + reduce_threads - 1) / reduce_threads;
        glq_reduce_splits_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
            output.data_ptr<float>(), scratch, M, k_splits
        );

        c10::cuda::CUDACachingAllocator::raw_delete(scratch);
    } else {
        dim3 grid(num_sms);

        if (has_stage2) {
            CHECK_INPUT(qidxs2);
            CHECK_INPUT(codebook2);
            glq_matvec_kernel<2><<<grid, block, 0, stream>>>(
                output.data_ptr<float>(),
                (const half*)x.data_ptr<c10::Half>(),
                qidxs.data_ptr<int16_t>(),
                (const half*)codebook.data_ptr<c10::Half>(),
                qidxs2.data_ptr<int16_t>(),
                (const half*)codebook2.data_ptr<c10::Half>(),
                inv_resid_scale,
                nullptr, nullptr, 0.0f,
                nullptr, nullptr, 0.0f,
                wscale, M, N_BLOCKS
            );
        } else {
            glq_matvec_kernel<1><<<grid, block, 0, stream>>>(
                output.data_ptr<float>(),
                (const half*)x.data_ptr<c10::Half>(),
                qidxs.data_ptr<int16_t>(),
                (const half*)codebook.data_ptr<c10::Half>(),
                nullptr, nullptr, 0.0f,
                nullptr, nullptr, 0.0f,
                nullptr, nullptr, 0.0f,
                wscale, M, N_BLOCKS
            );
        }
    }

    return output;
}


torch::Tensor glq_dequant_matmul_cuda(
    torch::Tensor x,           // (B, N) fp16
    torch::Tensor qidxs,       // (M, N_BLOCKS) int16
    torch::Tensor codebook,    // (65536, 8) fp16
    float wscale,
    torch::Tensor qidxs2,      // (M, N_BLOCKS) int16 or empty
    torch::Tensor codebook2,   // (K2, 8) fp16 or empty
    float inv_resid_scale,
    torch::Tensor codebook_abs, // unused, kept for ABI compat
    torch::Tensor qidxs3,      // stage 3 indices or empty
    torch::Tensor codebook3,   // stage 3 codebook (typically same as primary) or empty
    float inv_resid_scale2,    // 1/resid_scale2
    torch::Tensor qidxs4,      // stage 4 indices or empty
    torch::Tensor codebook4,   // stage 4 codebook (typically same as primary) or empty
    float inv_resid_scale3     // 1/resid_scale3
) {
    CHECK_INPUT(x);
    CHECK_INPUT(qidxs);
    CHECK_INPUT(codebook);

    int B_dim = x.size(0);
    int N = x.size(1);
    int M = qidxs.size(0);
    int N_BLOCKS = qidxs.size(1);

    at::DeviceGuard guard(x.device());
    auto output = torch::zeros({B_dim, M}, torch::dtype(torch::kFloat32).device(x.device()));

    static int num_sms = 0;
    if (num_sms == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, x.get_device());
        num_sms = prop.multiProcessorCount;
    }

    const int WARPS = 8;
    dim3 block(32, WARPS);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    bool has_stage2 = (qidxs2.numel() > 0 && inv_resid_scale != 0.0f);
    bool has_stage3 = has_stage2 && (qidxs3.numel() > 0 && inv_resid_scale2 != 0.0f);
    bool has_stage4 = has_stage3 && (qidxs4.numel() > 0 && inv_resid_scale3 != 0.0f);
    int num_stages = 1 + (has_stage2 ? 1 : 0) + (has_stage3 ? 1 : 0) + (has_stage4 ? 1 : 0);

    const half* x_ptr = (const half*)x.data_ptr<c10::Half>();
    const half* cb_ptr = (const half*)codebook.data_ptr<c10::Half>();
    const int16_t* q_ptr = qidxs.data_ptr<int16_t>();
    float* out_ptr = output.data_ptr<float>();

    const int16_t* q2_ptr = has_stage2 ? qidxs2.data_ptr<int16_t>() : nullptr;
    const half* cb2_ptr = has_stage2 ? (const half*)codebook2.data_ptr<c10::Half>() : nullptr;
    const int16_t* q3_ptr = has_stage3 ? qidxs3.data_ptr<int16_t>() : nullptr;
    const half* cb3_ptr = has_stage3 ? (const half*)codebook3.data_ptr<c10::Half>() : nullptr;
    const int16_t* q4_ptr = has_stage4 ? qidxs4.data_ptr<int16_t>() : nullptr;
    const half* cb4_ptr = has_stage4 ? (const half*)codebook4.data_ptr<c10::Half>() : nullptr;
    float irs = has_stage2 ? inv_resid_scale : 0.0f;
    float irs2 = has_stage3 ? inv_resid_scale2 : 0.0f;
    float irs3 = has_stage4 ? inv_resid_scale3 : 0.0f;

    if (has_stage2) {
        CHECK_INPUT(qidxs2);
        CHECK_INPUT(codebook2);
    }
    if (has_stage3) {
        CHECK_INPUT(qidxs3);
        CHECK_INPUT(codebook3);
    }
    if (has_stage4) {
        CHECK_INPUT(qidxs4);
        CHECK_INPUT(codebook4);
    }

    // Deterministic scratch+reduce split-K for the B>=2 path. Same pattern as
    // the B=1 matvec fix: adaptive k_splits for SM saturation, each CTA writes
    // to a unique (k_split, b_row, m_col) scratch slot (no atomicAdd), then a
    // reduction kernel sums across k_splits in fixed order → bit-exact runs.
    int b_tiles = (B_dim + 15) / 16;
    int m_tiles = (M + 7) / 8;
    int m_tiles_per_block = WARPS;
    int m_grid = (m_tiles + m_tiles_per_block - 1) / m_tiles_per_block;

    int bps = TC_BPS_DEFAULT;
    int k_splits = (N_BLOCKS + bps - 1) / bps;
    int total_ctas = b_tiles * m_grid * k_splits;
    if (total_ctas < num_sms * 2 && bps > 16) {
        bps = max(16, bps / 2);
        k_splits = (N_BLOCKS + bps - 1) / bps;
    }

    dim3 tc_grid(b_tiles, m_grid, k_splits);
    dim3 tc_block(32, WARPS);

    size_t scratch_bytes = (size_t)k_splits * B_dim * M * sizeof(float);
    float* scratch = (float*)c10::cuda::CUDACachingAllocator::raw_alloc(scratch_bytes);

    int cb2_size = has_stage2 ? codebook2.size(0) : 0;
    int smem = (has_stage2 && cb2_size <= 256) ? cb2_size * 8 * (int)sizeof(half) : 0;

#define LAUNCH_TC_MATMUL(NS)                                                              \
    glq_matmul_tc_scratch_kernel<NS><<<tc_grid, tc_block, smem, stream>>>(                \
        scratch, x_ptr, q_ptr, cb_ptr,                                                    \
        q2_ptr, cb2_ptr, irs,                                                             \
        q3_ptr, cb3_ptr, irs2,                                                            \
        q4_ptr, cb4_ptr, irs3,                                                            \
        wscale, B_dim, M, N, N_BLOCKS, N, bps, cb2_size)
    DISPATCH_NUM_STAGES(num_stages, LAUNCH_TC_MATMUL);
#undef LAUNCH_TC_MATMUL

    int BM = B_dim * M;
    int reduce_threads = 256;
    int reduce_blocks = (BM + reduce_threads - 1) / reduce_threads;
    glq_reduce_splits_2d_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
        out_ptr, scratch, BM, k_splits
    );

    c10::cuda::CUDACachingAllocator::raw_delete(scratch);

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


torch::Tensor glq_dequant_matmul_packed_cuda(
    torch::Tensor x,               // (B, N) fp16
    torch::Tensor qidxs,           // (M, N_BLOCKS) int16
    torch::Tensor codebook_packed, // (65536,) uint32
    float wscale
) {
    CHECK_INPUT(x);
    CHECK_INPUT(qidxs);
    CHECK_INPUT(codebook_packed);

    int B_dim = x.size(0);
    int N = x.size(1);
    int M = qidxs.size(0);
    int N_BLOCKS = qidxs.size(1);

    at::DeviceGuard guard(x.device());
    auto output = torch::zeros({B_dim, M}, torch::dtype(torch::kFloat32).device(x.device()));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, x.get_device());
    int num_sms = prop.multiProcessorCount;

    const int WARPS = 8;
    dim3 tc_block(32, WARPS);

    int b_tiles = (B_dim + 15) / 16;
    int m_tiles = (M + 7) / 8;
    int m_grid = (m_tiles + WARPS - 1) / WARPS;

    // Deterministic scratch+reduce split-K (packed 2bpw variant).
    int bps = TC_BPS_DEFAULT;
    int k_splits = (N_BLOCKS + bps - 1) / bps;
    int total_ctas = b_tiles * m_grid * k_splits;
    if (total_ctas < num_sms * 2 && bps > 16) {
        bps = max(16, bps / 2);
        k_splits = (N_BLOCKS + bps - 1) / bps;
    }

    dim3 tc_grid(b_tiles, m_grid, k_splits);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    size_t scratch_bytes = (size_t)k_splits * B_dim * M * sizeof(float);
    float* scratch = (float*)c10::cuda::CUDACachingAllocator::raw_alloc(scratch_bytes);

    glq_matmul_tc_packed_scratch_kernel<<<tc_grid, tc_block, 0, stream>>>(
        scratch,
        (const half*)x.data_ptr<c10::Half>(),
        qidxs.data_ptr<int16_t>(),
        (const uint32_t*)codebook_packed.data_ptr<int32_t>(),
        wscale, B_dim, M, N, N_BLOCKS, N, bps
    );

    int BM = B_dim * M;
    int reduce_threads = 256;
    int reduce_blocks = (BM + reduce_threads - 1) / reduce_threads;
    glq_reduce_splits_2d_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
        output.data_ptr<float>(), scratch, BM, k_splits
    );

    c10::cuda::CUDACachingAllocator::raw_delete(scratch);

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
    float* __restrict__ out,         // (B, stride_out) fp32 output (x_rht)
    int in_features,
    int stride_x,                    // x.stride(0) in elements
    float rsqrt_n,                   // 1.0 / sqrt(n_pad)
    int n_pad,
    int log_n,
    int use_single_buffer,           // 1 if smem too small for double-buffer
    int col_offset,                  // offset within row for block-diagonal
    int stride_out                   // output row stride (total n_pad for block-diag)
) {
    extern __shared__ float smem[];

    int b = blockIdx.x;
    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    // Step 1: Load x with zero-padding, multiply by SV signs
    float* buf = smem;
    for (int i = tid; i < n_pad; i += n_threads) {
        float x_val = (col_offset + i < in_features) ? __half2float(x[b * stride_x + col_offset + i]) : 0.0f;
        float s = __half2float(sv[col_offset + i]);
        buf[i] = x_val * s;
    }
    __syncthreads();

    // Step 2: FHT butterfly stages
    if (use_single_buffer) {
        // Single-buffer: read all elements into registers, sync, write back.
        // n_pad=16384, n_threads=1024 → 16 elements per thread in registers.
        float reg[16];  // max elements per thread
        int elems_per_thread = n_pad / n_threads;

        for (int k = 0; k < log_n; k++) {
            // Phase 1: all threads read and compute into registers
            for (int e = 0; e < elems_per_thread; e++) {
                int i = tid + e * n_threads;
                int partner = i ^ (1 << k);
                float my_val = buf[i];
                float partner_val = buf[partner];
                bool lo = (i & (1 << k)) == 0;
                reg[e] = lo ? (my_val + partner_val) : (partner_val - my_val);
            }
            __syncthreads();
            // Phase 2: write registers back to shared memory
            for (int e = 0; e < elems_per_thread; e++) {
                int i = tid + e * n_threads;
                buf[i] = reg[e];
            }
            __syncthreads();
        }
    } else {
        // Double-buffer: ping-pong between buf_a and buf_b
        float* buf_b = smem + n_pad;
        float* src = buf;
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
            float* tmp = src; src = dst; dst = tmp;
        }
        // If odd number of stages, result is in buf_b — copy to buf
        if (log_n % 2 == 1) {
            for (int i = tid; i < n_pad; i += n_threads) buf[i] = buf_b[i];
            __syncthreads();
        }
    }

    // Step 3: Normalize and store
    for (int i = tid; i < n_pad; i += n_threads) {
        out[b * stride_out + col_offset + i] = buf[i] * rsqrt_n;
    }
}


/* ─────────────────────────────────────────────────────────────────────
 * Two-pass FHT kernels for n_pad/m_pad > 16384 (exceeds smem limit).
 *
 * Butterfly stage k swaps pairs at distance 2^k. For n=32768 (15 stages):
 *   Stages 0-13: distance ≤ 8192 → within each 16384-element half
 *   Stage 14:    distance = 16384 → cross-boundary (a+b, a-b)
 *
 * Pass 1: Two blocks per batch row, each processes one 16384-element half
 *         with 14 butterfly stages in 64KB shared memory.
 * Pass 2: One block per batch row, applies the final cross-boundary stage.
 * ───────────────────────────────────────────────────────────────────── */

// Input RHT pass 1: 14 butterfly stages within each 16384-element half
__global__ void glq_input_rht_twopass_kernel(
    const half* __restrict__ x,
    const half* __restrict__ sv,
    float* __restrict__ temp,       // (B, n_pad) fp32 intermediate
    int in_features,
    int stride_x,
    int n_pad
) {
    extern __shared__ float smem[];
    float* buf = smem;

    const int half_size = n_pad / 2;  // 16384
    const int half_id = blockIdx.x % 2;
    const int b = blockIdx.x / 2;
    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;  // 1024
    const int offset = half_id * half_size;

    // Load half of x with zero-padding, multiply by SV signs
    for (int i = tid; i < half_size; i += n_threads) {
        int global_i = offset + i;
        float x_val = (global_i < in_features) ? __half2float(x[b * stride_x + global_i]) : 0.0f;
        buf[i] = x_val * __half2float(sv[global_i]);
    }
    __syncthreads();

    // 14 butterfly stages (single-buffer with registers)
    const int log_half = 14;  // log2(16384)
    const int elems_per_thread = half_size / n_threads;  // 16
    float reg[16];

    for (int k = 0; k < log_half; k++) {
        for (int e = 0; e < elems_per_thread; e++) {
            int i = tid + e * n_threads;
            int partner = i ^ (1 << k);
            float my_val = buf[i];
            float partner_val = buf[partner];
            bool lo = (i & (1 << k)) == 0;
            reg[e] = lo ? (my_val + partner_val) : (partner_val - my_val);
        }
        __syncthreads();
        for (int e = 0; e < elems_per_thread; e++) {
            buf[tid + e * n_threads] = reg[e];
        }
        __syncthreads();
    }

    // Write to global temp buffer (NOT normalized yet — pass 2 does that)
    for (int i = tid; i < half_size; i += n_threads) {
        temp[b * n_pad + offset + i] = buf[i];
    }
}

// Input RHT pass 2: cross-boundary butterfly (stage 14) + normalize
__global__ void glq_input_rht_cross_kernel(
    const float* __restrict__ temp,   // (B, n_pad) from pass 1
    float* __restrict__ out,          // (B, n_pad) final output
    float rsqrt_n,
    int n_pad
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;
    const int half_size = n_pad / 2;

    for (int i = tid; i < half_size; i += n_threads) {
        float a = temp[b * n_pad + i];
        float b_val = temp[b * n_pad + half_size + i];
        out[b * n_pad + i] = (a + b_val) * rsqrt_n;
        out[b * n_pad + half_size + i] = (a - b_val) * rsqrt_n;
    }
}

// Output RHT pass 1: cross-boundary butterfly (stage 14) — run FIRST for output
__global__ void glq_output_rht_cross_kernel(
    const float* __restrict__ y_rht,  // (B, m_pad) input
    float* __restrict__ temp,         // (B, m_pad) intermediate
    int m_pad
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;
    const int half_size = m_pad / 2;

    for (int i = tid; i < half_size; i += n_threads) {
        float a = y_rht[b * m_pad + i];
        float b_val = y_rht[b * m_pad + half_size + i];
        temp[b * m_pad + i] = a + b_val;
        temp[b * m_pad + half_size + i] = a - b_val;
    }
}

// Output RHT pass 2: 14 butterfly stages within each half + SU signs + unpad + cast
__global__ void glq_output_rht_twopass_kernel(
    const float* __restrict__ temp,   // (B, m_pad) from cross kernel
    const half* __restrict__ su,
    half* __restrict__ out,           // (B, out_features) fp16
    int out_features,
    int m_pad,
    float rsqrt_m
) {
    extern __shared__ float smem[];
    float* buf = smem;

    const int half_size = m_pad / 2;
    const int half_id = blockIdx.x % 2;
    const int b = blockIdx.x / 2;
    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;
    const int offset = half_id * half_size;

    // Load half from temp into smem
    for (int i = tid; i < half_size; i += n_threads) {
        buf[i] = temp[b * m_pad + offset + i];
    }
    __syncthreads();

    // 14 butterfly stages (single-buffer with registers)
    const int log_half = 14;
    const int elems_per_thread = half_size / n_threads;
    float reg[16];

    for (int k = 0; k < log_half; k++) {
        for (int e = 0; e < elems_per_thread; e++) {
            int i = tid + e * n_threads;
            int partner = i ^ (1 << k);
            float my_val = buf[i];
            float partner_val = buf[partner];
            bool lo = (i & (1 << k)) == 0;
            reg[e] = lo ? (my_val + partner_val) : (partner_val - my_val);
        }
        __syncthreads();
        for (int e = 0; e < elems_per_thread; e++) {
            buf[tid + e * n_threads] = reg[e];
        }
        __syncthreads();
    }

    // Normalize, apply SU signs, unpad, cast to fp16
    for (int i = tid; i < half_size; i += n_threads) {
        int global_i = offset + i;
        if (global_i < out_features) {
            float val = buf[i] * rsqrt_m;
            float s = __half2float(su[global_i]);
            out[b * out_features + global_i] = __float2half(val * s);
        }
    }
}


__global__ void glq_output_rht_kernel(
    const float* __restrict__ y_rht, // (B, stride_in) fp32 input
    const half* __restrict__ su,     // (total_m_pad,) fp16 sign vector
    half* __restrict__ out,          // (B, stride_out) fp16 output
    int out_features,
    int m_pad,
    int log_m,
    float rsqrt_m,                   // 1.0 / sqrt(m_pad)
    int use_single_buffer,
    int col_offset,                  // offset within row for block-diagonal
    int stride_in,                   // y_rht row stride
    int stride_out                   // output row stride
) {
    extern __shared__ float smem[];
    float* buf = smem;

    int b = blockIdx.x;
    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    // Step 1: Load y_rht from this block's column range
    for (int i = tid; i < m_pad; i += n_threads) {
        buf[i] = y_rht[b * stride_in + col_offset + i];
    }
    __syncthreads();

    // Step 2: FHT butterfly stages
    if (use_single_buffer) {
        float reg[16];
        int elems_per_thread = m_pad / n_threads;
        for (int k = 0; k < log_m; k++) {
            for (int e = 0; e < elems_per_thread; e++) {
                int i = tid + e * n_threads;
                int partner = i ^ (1 << k);
                float my_val = buf[i];
                float partner_val = buf[partner];
                bool lo = (i & (1 << k)) == 0;
                reg[e] = lo ? (my_val + partner_val) : (partner_val - my_val);
            }
            __syncthreads();
            for (int e = 0; e < elems_per_thread; e++) {
                buf[tid + e * n_threads] = reg[e];
            }
            __syncthreads();
        }
    } else {
        float* buf_b = smem + m_pad;
        float* src = buf;
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
        if (log_m % 2 == 1) {
            for (int i = tid; i < m_pad; i += n_threads) buf[i] = buf_b[i];
            __syncthreads();
        }
    }

    // Step 3: Normalize, apply SU signs, unpad, cast to fp16
    for (int i = tid; i < m_pad; i += n_threads) {
        if (col_offset + i < out_features) {
            float val = buf[i] * rsqrt_m;
            float s = __half2float(su[col_offset + i]);
            out[b * stride_out + col_offset + i] = __float2half(val * s);
        }
    }
}


/* ─────────────────────────────────────────────────────────────────────
 * Multi-block FHT kernels (block-diagonal fast path)
 *
 * Collapse N per-sub-block input/output RHT launches into a single kernel
 * with gridDim.y = num_blocks. blockIdx.y selects the sub-block; per-CTA
 * sub-block metadata is read from a packed int4 array.
 *
 * Double-buffer only. Gated by host to max_bs ≤ 8192 (64KB smem per CTA).
 * ───────────────────────────────────────────────────────────────────── */

__global__ void glq_input_rht_multiblock_kernel(
    const half* __restrict__ x,        // (B, in_features) fp16
    const half* __restrict__ sv,       // (n_pad,) fp16 — full sign vector
    float* __restrict__ out,           // (B, stride_out) fp32 output
    int in_features,
    int stride_x,
    int stride_out,
    const int4* __restrict__ block_meta  // per sub-block: {col_offset, bs, log_bs, _}
) {
    extern __shared__ float smem[];

    int b = blockIdx.x;
    int blk = blockIdx.y;
    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    int4 meta = block_meta[blk];
    int col_offset = meta.x;
    int bs = meta.y;
    int log_bs = meta.z;
    float rsqrt_bs = rsqrtf((float)bs);

    // Step 1: Load x with zero-pad, multiply by SV
    float* buf = smem;
    for (int i = tid; i < bs; i += n_threads) {
        float x_val = (col_offset + i < in_features)
            ? __half2float(x[b * stride_x + col_offset + i]) : 0.0f;
        float s = __half2float(sv[col_offset + i]);
        buf[i] = x_val * s;
    }
    __syncthreads();

    // Step 2: FHT butterfly — double-buffer (buf_b is after buf, max_bs elems apart)
    float* buf_b = smem + bs;
    float* src = buf;
    float* dst = buf_b;
    for (int k = 0; k < log_bs; k++) {
        for (int i = tid; i < bs; i += n_threads) {
            int partner = i ^ (1 << k);
            float my_val = src[i];
            float partner_val = src[partner];
            bool lo = (i & (1 << k)) == 0;
            dst[i] = lo ? (my_val + partner_val) : (partner_val - my_val);
        }
        __syncthreads();
        float* tmp = src; src = dst; dst = tmp;
    }
    if (log_bs % 2 == 1) {
        for (int i = tid; i < bs; i += n_threads) buf[i] = buf_b[i];
        __syncthreads();
    }

    // Step 3: Normalize and store
    for (int i = tid; i < bs; i += n_threads) {
        out[b * stride_out + col_offset + i] = buf[i] * rsqrt_bs;
    }
}

__global__ void glq_output_rht_multiblock_kernel(
    const float* __restrict__ y_rht,   // (B, stride_in) fp32
    const half* __restrict__ su,       // (m_pad,) fp16 — full sign vector
    half* __restrict__ out,            // (B, stride_out) fp16
    int out_features,
    int stride_in,
    int stride_out,
    const int4* __restrict__ block_meta
) {
    extern __shared__ float smem[];

    int b = blockIdx.x;
    int blk = blockIdx.y;
    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    int4 meta = block_meta[blk];
    int col_offset = meta.x;
    int bs = meta.y;
    int log_bs = meta.z;
    float rsqrt_bs = rsqrtf((float)bs);

    float* buf = smem;
    for (int i = tid; i < bs; i += n_threads) {
        buf[i] = y_rht[b * stride_in + col_offset + i];
    }
    __syncthreads();

    float* buf_b = smem + bs;
    float* src = buf;
    float* dst = buf_b;
    for (int k = 0; k < log_bs; k++) {
        for (int i = tid; i < bs; i += n_threads) {
            int partner = i ^ (1 << k);
            float my_val = src[i];
            float partner_val = src[partner];
            bool lo = (i & (1 << k)) == 0;
            dst[i] = lo ? (my_val + partner_val) : (partner_val - my_val);
        }
        __syncthreads();
        float* tmp = src; src = dst; dst = tmp;
    }
    if (log_bs % 2 == 1) {
        for (int i = tid; i < bs; i += n_threads) buf[i] = buf_b[i];
        __syncthreads();
    }

    for (int i = tid; i < bs; i += n_threads) {
        if (col_offset + i < out_features) {
            float val = buf[i] * rsqrt_bs;
            float s = __half2float(su[col_offset + i]);
            out[b * stride_out + col_offset + i] = __float2half(val * s);
        }
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
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    if (n_pad > 16384) {
        // Two-pass path for n_pad=32768 (128KB > max smem)
        auto temp = torch::empty({B, n_pad}, torch::dtype(torch::kFloat32).device(x.device()));
        int half_smem = (n_pad / 2) * sizeof(float);  // 64KB
        cudaFuncSetAttribute(glq_input_rht_twopass_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, half_smem);
        // Pass 1: 14 butterfly stages within each half (2 blocks per batch row)
        glq_input_rht_twopass_kernel<<<2 * B, 1024, half_smem, stream>>>(
            (const half*)x.data_ptr<c10::Half>(),
            (const half*)sv.data_ptr<c10::Half>(),
            temp.data_ptr<float>(),
            in_features, stride_x, n_pad
        );
        // Pass 2: cross-boundary butterfly + normalize
        glq_input_rht_cross_kernel<<<B, 1024, 0, stream>>>(
            temp.data_ptr<float>(),
            out.data_ptr<float>(),
            rsqrt_n, n_pad
        );
        return;
    }

    int threads = min(n_pad, 1024);
    int double_buf_smem = 2 * n_pad * sizeof(float);
    int single_buf_smem = n_pad * sizeof(float);
    int use_single = (double_buf_smem > 96 * 1024) ? 1 : 0;
    int smem = use_single ? single_buf_smem : double_buf_smem;

    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(glq_input_rht_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }

    glq_input_rht_kernel<<<B, threads, smem, stream>>>(
        (const half*)x.data_ptr<c10::Half>(),
        (const half*)sv.data_ptr<c10::Half>(),
        out.data_ptr<float>(),
        in_features, stride_x, rsqrt_n, n_pad, log_n, use_single,
        /*col_offset=*/0, /*stride_out=*/n_pad
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
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    if (m_pad > 16384) {
        // Two-pass path for m_pad=32768
        auto temp = torch::empty({B, m_pad}, torch::dtype(torch::kFloat32).device(y_rht.device()));
        int half_smem = (m_pad / 2) * sizeof(float);  // 64KB
        // Pass 1: cross-boundary butterfly (stage 14)
        glq_output_rht_cross_kernel<<<B, 1024, 0, stream>>>(
            y_rht.data_ptr<float>(),
            temp.data_ptr<float>(),
            m_pad
        );
        // Pass 2: 14 butterfly stages within each half + SU + unpad + cast
        cudaFuncSetAttribute(glq_output_rht_twopass_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, half_smem);
        glq_output_rht_twopass_kernel<<<2 * B, 1024, half_smem, stream>>>(
            temp.data_ptr<float>(),
            (const half*)su.data_ptr<c10::Half>(),
            (half*)out.data_ptr<c10::Half>(),
            out_features, m_pad, rsqrt_m
        );
        return;
    }

    int threads = min(m_pad, 1024);
    int double_buf_smem = 2 * m_pad * sizeof(float);
    int single_buf_smem = m_pad * sizeof(float);
    int use_single = (double_buf_smem > 96 * 1024) ? 1 : 0;
    int smem = use_single ? single_buf_smem : double_buf_smem;

    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(glq_output_rht_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }

    glq_output_rht_kernel<<<B, threads, smem, stream>>>(
        y_rht.data_ptr<float>(),
        (const half*)su.data_ptr<c10::Half>(),
        (half*)out.data_ptr<c10::Half>(),
        out_features, m_pad, log_m, rsqrt_m, use_single,
        /*col_offset=*/0, /*stride_in=*/m_pad, /*stride_out=*/out_features
    );
}


// Forward declaration (defined later after glq_fused_moe_cuda helpers).
static void launch_matvec_splitk(
    float* output, const half* x, const int16_t* qidxs,
    const half* codebook, float wscale,
    const int16_t* qidxs2, const half* codebook2,
    float inv_resid_scale,
    const int16_t* qidxs3, const half* codebook3,
    float inv_resid_scale2,
    const int16_t* qidxs4, const half* codebook4,
    float inv_resid_scale3,
    int num_stages,
    int M, int N_BLOCKS, int cb2_size,
    int num_sms, cudaStream_t stream
);


/* ─────────────────────────────────────────────────────────────────────
 * Fused linear: input_rht → dequant+matmul → output_rht in one host call
 *
 * Eliminates 2 Python→CUDA round-trips per linear layer.
 * All 3 kernels launch on the same stream back-to-back (~1μs gap vs ~140μs).
 * ───────────────────────────────────────────────────────────────────── */

torch::Tensor glq_fused_linear_cuda(
    torch::Tensor x,           // (B, in_features) fp16, contiguous
    torch::Tensor sv,          // (n_pad,) fp16 — input RHT sign vector
    torch::Tensor su,          // (m_pad,) fp16 — output RHT sign vector
    torch::Tensor qidxs,       // (M_pad, N_BLOCKS) int16
    torch::Tensor codebook,    // (65536, 8) fp16
    float wscale,
    int in_features,
    int out_features,
    int n_pad, int m_pad,
    int log_n, int log_m,
    torch::Tensor qidxs2,      // empty or (M_pad, N_BLOCKS) int16
    torch::Tensor codebook2,   // empty or (K2, 8) fp16
    float inv_resid_scale,
    torch::Tensor qidxs3,      // stage 3 indices or empty
    torch::Tensor codebook3,   // stage 3 codebook or empty
    float inv_resid_scale2,
    torch::Tensor qidxs4,      // stage 4 indices or empty
    torch::Tensor codebook4,   // stage 4 codebook or empty
    float inv_resid_scale3
) {
    CHECK_INPUT(x);
    CHECK_INPUT(qidxs);
    CHECK_INPUT(codebook);

    int B = x.size(0);
    int M = qidxs.size(0);
    int N_BLOCKS = qidxs.size(1);
    bool has_stage2 = (qidxs2.numel() > 0 && inv_resid_scale != 0.0f);
    bool has_stage3 = has_stage2 && (qidxs3.numel() > 0 && inv_resid_scale2 != 0.0f);
    bool has_stage4 = has_stage3 && (qidxs4.numel() > 0 && inv_resid_scale3 != 0.0f);
    int num_stages = 1 + (has_stage2 ? 1 : 0) + (has_stage3 ? 1 : 0) + (has_stage4 ? 1 : 0);

    const int16_t* q2_ptr = has_stage2 ? qidxs2.data_ptr<int16_t>() : nullptr;
    const half* cb2_ptr = has_stage2 ? (const half*)codebook2.data_ptr<c10::Half>() : nullptr;
    const int16_t* q3_ptr = has_stage3 ? qidxs3.data_ptr<int16_t>() : nullptr;
    const half* cb3_ptr = has_stage3 ? (const half*)codebook3.data_ptr<c10::Half>() : nullptr;
    const int16_t* q4_ptr = has_stage4 ? qidxs4.data_ptr<int16_t>() : nullptr;
    const half* cb4_ptr = has_stage4 ? (const half*)codebook4.data_ptr<c10::Half>() : nullptr;
    float irs = has_stage2 ? inv_resid_scale : 0.0f;
    float irs2 = has_stage3 ? inv_resid_scale2 : 0.0f;
    float irs3 = has_stage4 ? inv_resid_scale3 : 0.0f;

    at::DeviceGuard guard(x.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // ---- Step 1: Input RHT ----
    auto x_rht = torch::empty({B, n_pad}, torch::dtype(torch::kFloat32).device(x.device()));
    {
        float rsqrt_n = 1.0f / sqrtf((float)n_pad);

        if (n_pad > 16384) {
            // Two-pass for large n_pad (e.g., 32768)
            auto temp = torch::empty({B, n_pad}, torch::dtype(torch::kFloat32).device(x.device()));
            int half_smem = (n_pad / 2) * sizeof(float);
            cudaFuncSetAttribute(glq_input_rht_twopass_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, half_smem);
            glq_input_rht_twopass_kernel<<<2 * B, 1024, half_smem, stream>>>(
                (const half*)x.data_ptr<c10::Half>(),
                (const half*)sv.data_ptr<c10::Half>(),
                temp.data_ptr<float>(),
                in_features, in_features, n_pad
            );
            glq_input_rht_cross_kernel<<<B, 1024, 0, stream>>>(
                temp.data_ptr<float>(),
                x_rht.data_ptr<float>(),
                rsqrt_n, n_pad
            );
        } else {
            int threads = min(n_pad, 1024);
            int double_buf = 2 * n_pad * sizeof(float);
            int single_buf = n_pad * sizeof(float);
            int use_single = (double_buf > 96 * 1024) ? 1 : 0;
            int smem = use_single ? single_buf : double_buf;

            if (smem > 48 * 1024) {
                cudaFuncSetAttribute(glq_input_rht_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            }

            glq_input_rht_kernel<<<B, threads, smem, stream>>>(
                (const half*)x.data_ptr<c10::Half>(),
                (const half*)sv.data_ptr<c10::Half>(),
                x_rht.data_ptr<float>(),
                in_features, in_features, rsqrt_n, n_pad, log_n, use_single,
                /*col_offset=*/0, /*stride_out=*/n_pad
            );
        }
    }

    // ---- Step 2: Dequant + matmul ----
    // Convert x_rht from fp32 to fp16 for the matmul kernels
    auto x_rht_half = x_rht.to(torch::kFloat16);
    auto y_rht = torch::zeros({B, M}, torch::dtype(torch::kFloat32).device(x.device()));
    {
        static int num_sms = 0;
        if (num_sms == 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, x.get_device());
            num_sms = prop.multiProcessorCount;
        }

        if (B == 1) {
            // B=1: deterministic scratch+reduce split-K matvec
            int cb2_size = has_stage2 ? codebook2.size(0) : 0;
            launch_matvec_splitk(
                y_rht.data_ptr<float>(),
                (const half*)x_rht_half.data_ptr<c10::Half>(),
                qidxs.data_ptr<int16_t>(),
                (const half*)codebook.data_ptr<c10::Half>(),
                wscale,
                q2_ptr, cb2_ptr, irs,
                q3_ptr, cb3_ptr, irs2,
                q4_ptr, cb4_ptr, irs3,
                num_stages,
                M, N_BLOCKS, cb2_size, num_sms, stream
            );
        } else {
            // B>=2: deterministic scratch+reduce TC matmul
            const int WARPS = 8;
            dim3 tc_block(32, WARPS);
            int b_tiles = (B + 15) / 16;
            int m_tiles = (M + 7) / 8;
            int m_grid = (m_tiles + WARPS - 1) / WARPS;

            int bps = TC_BPS_DEFAULT;
            int k_splits = (N_BLOCKS + bps - 1) / bps;
            int total_ctas = b_tiles * m_grid * k_splits;
            if (total_ctas < num_sms * 2 && bps > 16) {
                bps = max(16, bps / 2);
                k_splits = (N_BLOCKS + bps - 1) / bps;
            }

            dim3 tc_grid(b_tiles, m_grid, k_splits);

            size_t scratch_bytes = (size_t)k_splits * B * M * sizeof(float);
            float* scratch = (float*)c10::cuda::CUDACachingAllocator::raw_alloc(scratch_bytes);

            int cb2_size = has_stage2 ? codebook2.size(0) : 0;
            int smem = (has_stage2 && cb2_size <= 256) ? cb2_size * 8 * (int)sizeof(half) : 0;

#define LAUNCH_FUSED_TC(NS)                                                               \
            glq_matmul_tc_scratch_kernel<NS><<<tc_grid, tc_block, smem, stream>>>(        \
                scratch,                                                                  \
                (const half*)x_rht_half.data_ptr<c10::Half>(),                            \
                qidxs.data_ptr<int16_t>(),                                                \
                (const half*)codebook.data_ptr<c10::Half>(),                              \
                q2_ptr, cb2_ptr, irs,                                                     \
                q3_ptr, cb3_ptr, irs2,                                                    \
                q4_ptr, cb4_ptr, irs3,                                                    \
                wscale, B, M, n_pad, N_BLOCKS, n_pad, bps, cb2_size)
            DISPATCH_NUM_STAGES(num_stages, LAUNCH_FUSED_TC);
#undef LAUNCH_FUSED_TC

            int BM = B * M;
            int reduce_threads = 256;
            int reduce_blocks = (BM + reduce_threads - 1) / reduce_threads;
            glq_reduce_splits_2d_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
                y_rht.data_ptr<float>(), scratch, BM, k_splits
            );

            c10::cuda::CUDACachingAllocator::raw_delete(scratch);
        }
    }

    // ---- Step 3: Output RHT ----
    auto y = torch::empty({B, out_features}, torch::dtype(torch::kFloat16).device(x.device()));
    {
        float rsqrt_m = 1.0f / sqrtf((float)m_pad);

        if (m_pad > 16384) {
            auto temp = torch::empty({B, m_pad}, torch::dtype(torch::kFloat32).device(x.device()));
            int half_smem = (m_pad / 2) * sizeof(float);
            glq_output_rht_cross_kernel<<<B, 1024, 0, stream>>>(
                y_rht.data_ptr<float>(),
                temp.data_ptr<float>(),
                m_pad
            );
            cudaFuncSetAttribute(glq_output_rht_twopass_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, half_smem);
            glq_output_rht_twopass_kernel<<<2 * B, 1024, half_smem, stream>>>(
                temp.data_ptr<float>(),
                (const half*)su.data_ptr<c10::Half>(),
                (half*)y.data_ptr<c10::Half>(),
                out_features, m_pad, rsqrt_m
            );
        } else {
            int threads = min(m_pad, 1024);
            int double_buf = 2 * m_pad * sizeof(float);
            int single_buf = m_pad * sizeof(float);
            int use_single = (double_buf > 96 * 1024) ? 1 : 0;
            int smem = use_single ? single_buf : double_buf;

            if (smem > 48 * 1024) {
                cudaFuncSetAttribute(glq_output_rht_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            }

            glq_output_rht_kernel<<<B, threads, smem, stream>>>(
                y_rht.data_ptr<float>(),
                (const half*)su.data_ptr<c10::Half>(),
                (half*)y.data_ptr<c10::Half>(),
                out_features, m_pad, log_m, rsqrt_m, use_single,
                /*col_offset=*/0, /*stride_in=*/m_pad, /*stride_out=*/out_features
            );
        }
    }

    return y;
}


/* ─────────────────────────────────────────────────────────────────────
 * Block-diagonal fused linear: input_rht(blocks) + dequant_matmul + output_rht(blocks)
 * ───────────────────────────────────────────────────────────────────── */

torch::Tensor glq_fused_linear_block_diag_cuda(
    torch::Tensor x,           // (B, in_features) fp16, contiguous
    torch::Tensor sv,          // (n_pad,) fp16 — input RHT sign vector
    torch::Tensor su,          // (m_pad,) fp16 — output RHT sign vector
    torch::Tensor qidxs,       // (M, N_BLOCKS) int16
    torch::Tensor codebook,    // (65536, 8) fp16
    float wscale,
    int in_features,
    int out_features,
    int n_pad, int m_pad,
    torch::Tensor blocks_n,    // 1D int64 CPU: [2048, 512, 128]
    torch::Tensor blocks_m,    // 1D int64 CPU: [2048, 512, 128]
    torch::Tensor blocks_n_meta,  // (num_n_blocks, 4) int32 GPU — packed {col_offset, bs, log_bs, _}
    torch::Tensor blocks_m_meta,  // (num_m_blocks, 4) int32 GPU
    torch::Tensor qidxs2,
    torch::Tensor codebook2,
    float inv_resid_scale,
    torch::Tensor qidxs3,      // stage 3 indices or empty
    torch::Tensor codebook3,   // stage 3 codebook or empty
    float inv_resid_scale2,
    torch::Tensor qidxs4,      // stage 4 indices or empty
    torch::Tensor codebook4,   // stage 4 codebook or empty
    float inv_resid_scale3
) {
    int B = x.size(0);
    int M = qidxs.size(0);
    int N_BLOCKS = qidxs.size(1);
    bool has_stage2 = (qidxs2.numel() > 0 && inv_resid_scale != 0.0f);
    bool has_stage3 = has_stage2 && (qidxs3.numel() > 0 && inv_resid_scale2 != 0.0f);
    bool has_stage4 = has_stage3 && (qidxs4.numel() > 0 && inv_resid_scale3 != 0.0f);
    int num_stages = 1 + (has_stage2 ? 1 : 0) + (has_stage3 ? 1 : 0) + (has_stage4 ? 1 : 0);

    const int16_t* q2_ptr = has_stage2 ? qidxs2.data_ptr<int16_t>() : nullptr;
    const half* cb2_ptr = has_stage2 ? (const half*)codebook2.data_ptr<c10::Half>() : nullptr;
    const int16_t* q3_ptr = has_stage3 ? qidxs3.data_ptr<int16_t>() : nullptr;
    const half* cb3_ptr = has_stage3 ? (const half*)codebook3.data_ptr<c10::Half>() : nullptr;
    const int16_t* q4_ptr = has_stage4 ? qidxs4.data_ptr<int16_t>() : nullptr;
    const half* cb4_ptr = has_stage4 ? (const half*)codebook4.data_ptr<c10::Half>() : nullptr;
    float irs = has_stage2 ? inv_resid_scale : 0.0f;
    float irs2 = has_stage3 ? inv_resid_scale2 : 0.0f;
    float irs3 = has_stage4 ? inv_resid_scale3 : 0.0f;

    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    // ---- Step 1: Input RHT ----
    auto x_rht = torch::empty({B, n_pad}, torch::dtype(torch::kFloat32).device(x.device()));
    {
        const half* x_ptr = (const half*)x.data_ptr<c10::Half>();
        const half* sv_ptr = (const half*)sv.data_ptr<c10::Half>();
        float* x_rht_ptr = x_rht.data_ptr<float>();
        int64_t* bn = blocks_n.data_ptr<int64_t>();
        int num_n_blocks = blocks_n.size(0);

        // Compute max_bs (cheap, ≤ 5 entries)
        int max_bs_n = 0;
        for (int bi = 0; bi < num_n_blocks; bi++) {
            int bs = (int)bn[bi];
            if (bs > max_bs_n) max_bs_n = bs;
        }

        // Multiblock fast path: max_bs ≤ 8192 keeps double-buffer smem ≤ 64KB;
        // metadata tensor must be provided on GPU.
        bool use_multiblock = (max_bs_n <= 8192) && (blocks_n_meta.numel() > 0)
                              && blocks_n_meta.is_cuda();
        if (use_multiblock) {
            int threads = min(max_bs_n, 1024);
            int smem = 2 * max_bs_n * (int)sizeof(float);
            if (smem > 48 * 1024) {
                cudaFuncSetAttribute(glq_input_rht_multiblock_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            }
            dim3 grid(B, num_n_blocks);
            glq_input_rht_multiblock_kernel<<<grid, threads, smem, stream>>>(
                x_ptr, sv_ptr, x_rht_ptr,
                in_features, in_features, n_pad,
                (const int4*)blocks_n_meta.data_ptr<int32_t>()
            );
        } else {
            int col_offset = 0;
            for (int bi = 0; bi < num_n_blocks; bi++) {
                int bs = (int)bn[bi];
                int log_bs = __builtin_ctz(bs);
                int threads = min(bs, 1024);
                int double_buf = 2 * bs * (int)sizeof(float);
                int single_buf = bs * (int)sizeof(float);
                int use_single = (double_buf > 96 * 1024) ? 1 : 0;
                int smem = use_single ? single_buf : double_buf;
                if (smem > 48 * 1024) {
                    cudaFuncSetAttribute(glq_input_rht_kernel,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                }
                glq_input_rht_kernel<<<B, threads, smem, stream>>>(
                    x_ptr, sv_ptr, x_rht_ptr,
                    in_features, in_features, 1.0f / sqrtf((float)bs),
                    bs, log_bs, use_single,
                    col_offset, n_pad
                );
                col_offset += bs;
            }
        }
    }

    // ---- Step 2: Dequant + matmul (identical to glq_fused_linear_cuda) ----
    auto x_rht_half = x_rht.to(torch::kFloat16);
    auto y_rht = torch::zeros({B, M}, torch::dtype(torch::kFloat32).device(x.device()));
    {
        static int num_sms = 0;
        if (num_sms == 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, x.get_device());
            num_sms = prop.multiProcessorCount;
        }

        if (B == 1) {
            int cb2_size = has_stage2 ? codebook2.size(0) : 0;
            launch_matvec_splitk(
                y_rht.data_ptr<float>(),
                (const half*)x_rht_half.data_ptr<c10::Half>(),
                qidxs.data_ptr<int16_t>(),
                (const half*)codebook.data_ptr<c10::Half>(),
                wscale,
                q2_ptr, cb2_ptr, irs,
                q3_ptr, cb3_ptr, irs2,
                q4_ptr, cb4_ptr, irs3,
                num_stages,
                M, N_BLOCKS, cb2_size, num_sms, stream
            );
        } else {
            const int WARPS = 8;
            dim3 tc_block(32, WARPS);
            int b_tiles = (B + 15) / 16;
            int m_tiles = (M + 7) / 8;
            int m_grid = (m_tiles + WARPS - 1) / WARPS;

            int bps = TC_BPS_DEFAULT;
            int k_splits = (N_BLOCKS + bps - 1) / bps;
            int total_ctas = b_tiles * m_grid * k_splits;
            if (total_ctas < num_sms * 2 && bps > 16) {
                bps = max(16, bps / 2);
                k_splits = (N_BLOCKS + bps - 1) / bps;
            }

            dim3 tc_grid(b_tiles, m_grid, k_splits);
            size_t scratch_bytes = (size_t)k_splits * B * M * sizeof(float);
            float* scratch = (float*)c10::cuda::CUDACachingAllocator::raw_alloc(scratch_bytes);

            int cb2_size = has_stage2 ? codebook2.size(0) : 0;
            int smem = (has_stage2 && cb2_size <= 256) ? cb2_size * 8 * (int)sizeof(half) : 0;

#define LAUNCH_BD_TC(NS)                                                                  \
            glq_matmul_tc_scratch_kernel<NS><<<tc_grid, tc_block, smem, stream>>>(        \
                scratch,                                                                  \
                (const half*)x_rht_half.data_ptr<c10::Half>(),                            \
                qidxs.data_ptr<int16_t>(),                                                \
                (const half*)codebook.data_ptr<c10::Half>(),                              \
                q2_ptr, cb2_ptr, irs,                                                     \
                q3_ptr, cb3_ptr, irs2,                                                    \
                q4_ptr, cb4_ptr, irs3,                                                    \
                wscale, B, M, n_pad, N_BLOCKS, n_pad, bps, cb2_size)
            DISPATCH_NUM_STAGES(num_stages, LAUNCH_BD_TC);
#undef LAUNCH_BD_TC

            int BM = B * M;
            int reduce_threads = 256;
            int reduce_blocks = (BM + reduce_threads - 1) / reduce_threads;
            glq_reduce_splits_2d_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
                y_rht.data_ptr<float>(), scratch, BM, k_splits
            );
            c10::cuda::CUDACachingAllocator::raw_delete(scratch);
        }
    }

    // ---- Step 3: Output RHT ----
    auto y = torch::empty({B, out_features}, torch::dtype(torch::kFloat16).device(x.device()));
    {
        const half* su_ptr = (const half*)su.data_ptr<c10::Half>();
        half* y_ptr = (half*)y.data_ptr<c10::Half>();
        float* y_rht_ptr = y_rht.data_ptr<float>();
        int64_t* bm = blocks_m.data_ptr<int64_t>();
        int num_m_blocks = blocks_m.size(0);

        int max_bs_m = 0;
        for (int bi = 0; bi < num_m_blocks; bi++) {
            int bs = (int)bm[bi];
            if (bs > max_bs_m) max_bs_m = bs;
        }

        bool use_multiblock = (max_bs_m <= 8192) && (blocks_m_meta.numel() > 0)
                              && blocks_m_meta.is_cuda();
        if (use_multiblock) {
            int threads = min(max_bs_m, 1024);
            int smem = 2 * max_bs_m * (int)sizeof(float);
            if (smem > 48 * 1024) {
                cudaFuncSetAttribute(glq_output_rht_multiblock_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            }
            dim3 grid(B, num_m_blocks);
            glq_output_rht_multiblock_kernel<<<grid, threads, smem, stream>>>(
                y_rht_ptr, su_ptr, y_ptr,
                out_features, m_pad, out_features,
                (const int4*)blocks_m_meta.data_ptr<int32_t>()
            );
        } else {
            int col_offset = 0;
            for (int bi = 0; bi < num_m_blocks; bi++) {
                int bs = (int)bm[bi];
                int log_bs = __builtin_ctz(bs);
                int threads = min(bs, 1024);
                int double_buf = 2 * bs * (int)sizeof(float);
                int single_buf = bs * (int)sizeof(float);
                int use_single = (double_buf > 96 * 1024) ? 1 : 0;
                int smem = use_single ? single_buf : double_buf;
                if (smem > 48 * 1024) {
                    cudaFuncSetAttribute(glq_output_rht_kernel,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                }
                glq_output_rht_kernel<<<B, threads, smem, stream>>>(
                    y_rht_ptr, su_ptr, y_ptr,
                    out_features, bs, log_bs, 1.0f / sqrtf((float)bs),
                    use_single,
                    col_offset, m_pad, out_features
                );
                col_offset += bs;
            }
        }
    }

    return y;
}


/* ─────────────────────────────────────────────────────────────────────
 * Helper kernels for fused MoE
 * ───────────────────────────────────────────────────────────────────── */

// Squared ReLU (non-gated): out[i] = max(0, in[i])^2
__global__ void relu2_activation_kernel(
    const half* __restrict__ in,
    half* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(in[i]);
        v = fmaxf(v, 0.0f);
        out[i] = __float2half(v * v);
    }
}

// SiLU gated: out[i] = silu(in[i]) * in[i + half]
__global__ void silu_gated_activation_kernel(
    const half* __restrict__ in,
    half* __restrict__ out,
    int half_n  // output size = half of input size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < half_n) {
        float gate = __half2float(in[i]);
        float up = __half2float(in[i + half_n]);
        float silu = gate / (1.0f + expf(-gate));
        out[i] = __float2half(silu * up);
    }
}

// Weighted accumulate: out[i] += weight * in[i]
__global__ void weighted_add_kernel(
    half* __restrict__ out,
    const half* __restrict__ in,
    float weight,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(out[i]) + weight * __half2float(in[i]);
        out[i] = __float2half(v);
    }
}

// Helper: launch split-K matvec from C++ (deterministic scratch+reduce path).
// Preserves adaptive split-K SM saturation while staying bit-exact across runs.
static void launch_matvec_splitk(
    float* output, const half* x, const int16_t* qidxs,
    const half* codebook, float wscale,
    const int16_t* qidxs2, const half* codebook2,
    float inv_resid_scale,
    const int16_t* qidxs3, const half* codebook3,
    float inv_resid_scale2,
    const int16_t* qidxs4, const half* codebook4,
    float inv_resid_scale3,
    int num_stages,
    int M, int N_BLOCKS, int cb2_size,
    int num_sms, cudaStream_t stream
) {
    const int WARPS = 8;
    const int rows_per_block = ROWS_PER_WARP * WARPS;
    dim3 block(32, WARPS);
    int m_blocks = (M + rows_per_block - 1) / rows_per_block;

    // Adaptive split-K (same sizing logic as glq_dequant_matvec_cuda).
    int bps = BLOCKS_PER_SPLIT_DEFAULT;
    int k_splits = (N_BLOCKS + bps - 1) / bps;
    int total_ctas = m_blocks * k_splits;
    if (total_ctas < num_sms * 2 && bps > 16) {
        bps = max(16, bps / 2);
        k_splits = (N_BLOCKS + bps - 1) / bps;
    }
    dim3 grid(m_blocks, k_splits);

    size_t scratch_bytes = (size_t)k_splits * M * sizeof(float);
    float* scratch = (float*)c10::cuda::CUDACachingAllocator::raw_alloc(scratch_bytes);

    int smem = (num_stages >= 2 && cb2_size > 0 && cb2_size <= 256)
                   ? cb2_size * 8 * (int)sizeof(half)
                   : 0;

#define LAUNCH_SPLITK_MATVEC(NS)                                                 \
    glq_matvec_splitk_scratch_kernel<NS><<<grid, block, smem, stream>>>(         \
        scratch, x, qidxs, codebook,                                             \
        qidxs2, codebook2, inv_resid_scale,                                      \
        qidxs3, codebook3, inv_resid_scale2,                                     \
        qidxs4, codebook4, inv_resid_scale3,                                     \
        wscale, M, N_BLOCKS, bps, cb2_size)
    DISPATCH_NUM_STAGES(num_stages, LAUNCH_SPLITK_MATVEC);
#undef LAUNCH_SPLITK_MATVEC

    // Deterministic reduction: sum scratch[0:k_splits, m] → output[m]
    int reduce_threads = 256;
    int reduce_blocks = (M + reduce_threads - 1) / reduce_threads;
    glq_reduce_splits_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
        output, scratch, M, k_splits
    );

    c10::cuda::CUDACachingAllocator::raw_delete(scratch);
}

// Helper: launch input/output RHT
static void launch_input_rht(
    const half* x, const half* sv, float* out,
    int in_features, int n_pad, int log_n, float rsqrt_n,
    int B, cudaStream_t stream
) {
    if (n_pad > 16384) {
        // Two-pass for large n_pad (e.g., 32768)
        float* temp = (float*)c10::cuda::CUDACachingAllocator::raw_alloc(
            (size_t)B * n_pad * sizeof(float));
        int half_smem = (n_pad / 2) * sizeof(float);
        cudaFuncSetAttribute(glq_input_rht_twopass_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, half_smem);
        glq_input_rht_twopass_kernel<<<2*B, 1024, half_smem, stream>>>(
            x, sv, temp, in_features, in_features, n_pad);
        glq_input_rht_cross_kernel<<<B, 1024, 0, stream>>>(
            temp, out, rsqrt_n, n_pad);
        c10::cuda::CUDACachingAllocator::raw_delete(temp);
        return;
    }
    int threads = min(n_pad, 1024);
    int dbl = 2 * n_pad * sizeof(float);
    int sgl = n_pad * sizeof(float);
    int use_single = (dbl > 96*1024) ? 1 : 0;
    int smem = use_single ? sgl : dbl;
    if (smem > 48*1024)
        cudaFuncSetAttribute(glq_input_rht_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    glq_input_rht_kernel<<<B, threads, smem, stream>>>(
        x, sv, out, in_features, in_features, rsqrt_n, n_pad, log_n, use_single,
        /*col_offset=*/0, /*stride_out=*/n_pad);
}

static void launch_output_rht(
    const float* y_rht, const half* su, half* out,
    int out_features, int m_pad, int log_m, float rsqrt_m,
    int B, cudaStream_t stream
) {
    if (m_pad > 16384) {
        float* temp = (float*)c10::cuda::CUDACachingAllocator::raw_alloc(
            (size_t)B * m_pad * sizeof(float));
        int half_smem = (m_pad / 2) * sizeof(float);
        glq_output_rht_cross_kernel<<<B, 1024, 0, stream>>>(
            y_rht, temp, m_pad);
        cudaFuncSetAttribute(glq_output_rht_twopass_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, half_smem);
        glq_output_rht_twopass_kernel<<<2*B, 1024, half_smem, stream>>>(
            temp, su, out, out_features, m_pad, rsqrt_m);
        c10::cuda::CUDACachingAllocator::raw_delete(temp);
        return;
    }
    int threads = min(m_pad, 1024);
    int dbl = 2 * m_pad * sizeof(float);
    int sgl = m_pad * sizeof(float);
    int use_single = (dbl > 96*1024) ? 1 : 0;
    int smem = use_single ? sgl : dbl;
    if (smem > 48*1024)
        cudaFuncSetAttribute(glq_output_rht_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    glq_output_rht_kernel<<<B, threads, smem, stream>>>(
        y_rht, su, out, out_features, m_pad, log_m, rsqrt_m, use_single,
        /*col_offset=*/0, /*stride_in=*/m_pad, /*stride_out=*/out_features);
}


// Helper: launch block-diagonal input RHT (sibling of launch_input_rht for non-pow2 dims).
// Reuses glq_input_rht_multiblock_kernel / glq_input_rht_kernel — no new GPU kernels.
static void launch_input_rht_block_diag(
    const half* x, const half* sv, float* out,
    int in_features, int n_pad,
    int B,
    const int64_t* blocks_n_host, int num_n_blocks,
    const int4* blocks_n_meta_dev, bool meta_on_gpu,
    cudaStream_t stream
) {
    int max_bs = 0;
    for (int bi = 0; bi < num_n_blocks; bi++) {
        int bs = (int)blocks_n_host[bi];
        if (bs > max_bs) max_bs = bs;
    }
    bool use_multiblock = (max_bs <= 8192) && meta_on_gpu;
    if (use_multiblock) {
        int threads = min(max_bs, 1024);
        int smem = 2 * max_bs * (int)sizeof(float);
        if (smem > 48 * 1024) {
            cudaFuncSetAttribute(glq_input_rht_multiblock_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        }
        dim3 grid(B, num_n_blocks);
        glq_input_rht_multiblock_kernel<<<grid, threads, smem, stream>>>(
            x, sv, out,
            in_features, in_features, n_pad,
            blocks_n_meta_dev
        );
    } else {
        int col_offset = 0;
        for (int bi = 0; bi < num_n_blocks; bi++) {
            int bs = (int)blocks_n_host[bi];
            int log_bs = __builtin_ctz(bs);
            int threads = min(bs, 1024);
            int double_buf = 2 * bs * (int)sizeof(float);
            int single_buf = bs * (int)sizeof(float);
            int use_single = (double_buf > 96 * 1024) ? 1 : 0;
            int smem = use_single ? single_buf : double_buf;
            if (smem > 48 * 1024) {
                cudaFuncSetAttribute(glq_input_rht_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            }
            glq_input_rht_kernel<<<B, threads, smem, stream>>>(
                x, sv, out,
                in_features, in_features, 1.0f / sqrtf((float)bs),
                bs, log_bs, use_single,
                col_offset, n_pad
            );
            col_offset += bs;
        }
    }
}

// Helper: launch block-diagonal output RHT (sibling of launch_output_rht).
static void launch_output_rht_block_diag(
    const float* y_rht, const half* su, half* out,
    int out_features, int m_pad,
    int B,
    const int64_t* blocks_m_host, int num_m_blocks,
    const int4* blocks_m_meta_dev, bool meta_on_gpu,
    cudaStream_t stream
) {
    int max_bs = 0;
    for (int bi = 0; bi < num_m_blocks; bi++) {
        int bs = (int)blocks_m_host[bi];
        if (bs > max_bs) max_bs = bs;
    }
    bool use_multiblock = (max_bs <= 8192) && meta_on_gpu;
    if (use_multiblock) {
        int threads = min(max_bs, 1024);
        int smem = 2 * max_bs * (int)sizeof(float);
        if (smem > 48 * 1024) {
            cudaFuncSetAttribute(glq_output_rht_multiblock_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        }
        dim3 grid(B, num_m_blocks);
        glq_output_rht_multiblock_kernel<<<grid, threads, smem, stream>>>(
            y_rht, su, out,
            out_features, m_pad, out_features,
            blocks_m_meta_dev
        );
    } else {
        int col_offset = 0;
        for (int bi = 0; bi < num_m_blocks; bi++) {
            int bs = (int)blocks_m_host[bi];
            int log_bs = __builtin_ctz(bs);
            int threads = min(bs, 1024);
            int double_buf = 2 * bs * (int)sizeof(float);
            int single_buf = bs * (int)sizeof(float);
            int use_single = (double_buf > 96 * 1024) ? 1 : 0;
            int smem = use_single ? single_buf : double_buf;
            if (smem > 48 * 1024) {
                cudaFuncSetAttribute(glq_output_rht_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            }
            glq_output_rht_kernel<<<B, threads, smem, stream>>>(
                y_rht, su, out,
                out_features, bs, log_bs, 1.0f / sqrtf((float)bs),
                use_single,
                col_offset, m_pad, out_features
            );
            col_offset += bs;
        }
    }
}


/* ─────────────────────────────────────────────────────────────────────
 * Fused MoE: full expert dispatch in one C++ call
 *
 * Eliminates Python dispatch between expert iterations, activation
 * kernel launches, and topk_ids.unique() GPU-CPU sync.
 * ───────────────────────────────────────────────────────────────────── */

torch::Tensor glq_fused_moe_cuda(
    torch::Tensor x,                    // (num_tokens, hidden) fp16
    torch::Tensor topk_ids,             // (num_tokens, top_k) int64
    torch::Tensor topk_weights,         // (num_tokens, top_k) fp32
    // w13 (gate_up_proj) weights — all experts stacked
    torch::Tensor w13_Qidxs,            // (E, m_pad_w13, n_blocks_w13) int16
    torch::Tensor w13_SU,               // (E, m_pad_w13) fp16
    torch::Tensor w13_SV,               // (n_pad_w13,) fp16 — shared
    torch::Tensor w13_Wscale,           // (E,) fp32
    torch::Tensor w13_Qidxs2,           // empty or (E, m_pad_w13, n_blocks_w13)
    torch::Tensor w13_inv_rs,           // (E,) fp32
    // w2 (down_proj) weights — all experts stacked
    torch::Tensor w2_Qidxs,
    torch::Tensor w2_SU,
    torch::Tensor w2_SV,                // (n_pad_w2,) fp16 — shared
    torch::Tensor w2_Wscale,
    torch::Tensor w2_Qidxs2,
    torch::Tensor w2_inv_rs,
    // Codebooks
    torch::Tensor codebook,
    torch::Tensor codebook2,
    // Dimensions
    int hidden_size, int intermediate_size, int w13_out_features,
    int n_pad_w13, int m_pad_w13,
    int n_pad_w2, int m_pad_w2,
    int log_n_w13, int log_m_w13,
    int log_n_w2, int log_m_w2,
    int activation_type,  // 0=silu(gated), 5=relu2_no_mul, etc.
    // Stage-3 RVQ (optional). Pass empty tensors for 2-stage.
    // The stage-3 codebook is shared between w13 and w2 (mirrors glq_fused_linear_cuda).
    torch::Tensor w13_Qidxs3,           // empty or (E, m_pad_w13, n_blocks_w13) int16
    torch::Tensor w13_inv_rs2,          // empty or (E,) fp32 — stage-3 scale per expert
    torch::Tensor w2_Qidxs3,            // empty or (E, m_pad_w2, n_blocks_w2) int16
    torch::Tensor w2_inv_rs2,           // empty or (E,) fp32 — stage-3 scale per expert
    torch::Tensor codebook3             // empty or (K3, 8) fp16 — shared between w13/w2
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w13_Qidxs);
    CHECK_INPUT(codebook);

    int num_tokens = x.size(0);
    int top_k = topk_ids.size(1);
    int M_w13 = w13_Qidxs.size(1);
    int NB_w13 = w13_Qidxs.size(2);
    int M_w2 = w2_Qidxs.size(1);
    int NB_w2 = w2_Qidxs.size(2);
    bool w13_has_s2 = (w13_Qidxs2.numel() > 0);
    bool w2_has_s2 = (w2_Qidxs2.numel() > 0);
    bool w13_has_s3 = (w13_Qidxs3.numel() > 0 && w13_inv_rs2.numel() > 0);
    bool w2_has_s3 = (w2_Qidxs3.numel() > 0 && w2_inv_rs2.numel() > 0);
    int cb2_size = codebook2.numel() > 0 ? (int)codebook2.size(0) : 0;

    at::DeviceGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    static int num_sms = 0;
    if (num_sms == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, x.get_device());
        num_sms = prop.multiProcessorCount;
    }

    float rsqrt_n_w13 = 1.0f / sqrtf((float)n_pad_w13);
    float rsqrt_m_w13 = 1.0f / sqrtf((float)m_pad_w13);
    float rsqrt_n_w2 = 1.0f / sqrtf((float)n_pad_w2);
    float rsqrt_m_w2 = 1.0f / sqrtf((float)m_pad_w2);

    auto opts_f32 = torch::dtype(torch::kFloat32).device(x.device());
    auto opts_f16 = torch::dtype(torch::kFloat16).device(x.device());

    // Output accumulator (zeroed for weighted add)
    auto output = torch::zeros({num_tokens, hidden_size}, opts_f16);

    // Shared input RHT: same x and w13_SV for all experts
    auto x_rht = torch::empty({num_tokens, n_pad_w13}, opts_f32);
    launch_input_rht(
        (const half*)x.data_ptr<c10::Half>(),
        (const half*)w13_SV.data_ptr<c10::Half>(),
        x_rht.data_ptr<float>(),
        hidden_size, n_pad_w13, log_n_w13, rsqrt_n_w13,
        num_tokens, stream);
    auto x_rht_half = x_rht.to(torch::kFloat16);

    // Reusable per-expert buffers
    auto y_rht_w13 = torch::empty({num_tokens, M_w13}, opts_f32);
    auto h_w13 = torch::empty({num_tokens, w13_out_features}, opts_f16);
    auto h_act = torch::empty({num_tokens, intermediate_size}, opts_f16);
    auto h_rht = torch::empty({num_tokens, n_pad_w2}, opts_f32);
    auto y_rht_w2 = torch::empty({num_tokens, M_w2}, opts_f32);
    auto expert_out = torch::empty({num_tokens, hidden_size}, opts_f16);

    // Read expert IDs to CPU (tiny sync — 2 int64s for top-2)
    auto topk_ids_cpu = topk_ids.cpu();
    auto topk_w_cpu = topk_weights.cpu();

    const half* cb_ptr = (const half*)codebook.data_ptr<c10::Half>();
    const half* cb2_ptr = codebook2.numel() > 0 ? (const half*)codebook2.data_ptr<c10::Half>() : nullptr;
    const half* cb3_ptr = codebook3.numel() > 0 ? (const half*)codebook3.data_ptr<c10::Half>() : nullptr;

    for (int t = 0; t < num_tokens; t++) {
        for (int k = 0; k < top_k; k++) {
            int eidx = topk_ids_cpu[t][k].item<int64_t>();
            float ew = topk_w_cpu[t][k].item<float>();
            if (ew == 0.0f) continue;

            float w13_ws = w13_Wscale[eidx].item<float>();
            float w13_irs = w13_has_s2 ? w13_inv_rs[eidx].item<float>() : 0.0f;
            bool w13_s2 = (w13_irs != 0.0f);
            // Stage-3 only valid if stage-2 is active for this expert
            float w13_irs2 = (w13_s2 && w13_has_s3) ? w13_inv_rs2[eidx].item<float>() : 0.0f;
            bool w13_s3 = (w13_irs2 != 0.0f);

            float w2_ws = w2_Wscale[eidx].item<float>();
            float w2_irs = w2_has_s2 ? w2_inv_rs[eidx].item<float>() : 0.0f;
            bool w2_s2 = (w2_irs != 0.0f);
            float w2_irs2 = (w2_s2 && w2_has_s3) ? w2_inv_rs2[eidx].item<float>() : 0.0f;
            bool w2_s3 = (w2_irs2 != 0.0f);

            int w13_num_stages = 1 + (w13_s2 ? 1 : 0) + (w13_s3 ? 1 : 0);
            int w2_num_stages = 1 + (w2_s2 ? 1 : 0) + (w2_s3 ? 1 : 0);

            // ---- w13: dequant+matmul (input RHT already done) ----
            y_rht_w13.zero_();
            launch_matvec_splitk(
                y_rht_w13.data_ptr<float>() + t * M_w13,
                (const half*)x_rht_half.data_ptr<c10::Half>() + t * n_pad_w13,
                w13_Qidxs[eidx].data_ptr<int16_t>(), cb_ptr, w13_ws,
                w13_s2 ? w13_Qidxs2[eidx].data_ptr<int16_t>() : nullptr,
                w13_s2 ? cb2_ptr : nullptr,
                w13_irs,
                w13_s3 ? w13_Qidxs3[eidx].data_ptr<int16_t>() : nullptr,
                w13_s3 ? cb3_ptr : nullptr,
                w13_irs2,
                nullptr, nullptr, 0.0f,
                w13_num_stages,
                M_w13, NB_w13, cb2_size, num_sms, stream);

            // ---- w13: output RHT ----
            launch_output_rht(
                y_rht_w13.data_ptr<float>() + t * M_w13,
                (const half*)w13_SU[eidx].data_ptr<c10::Half>(),
                (half*)h_w13.data_ptr<c10::Half>() + t * w13_out_features,
                w13_out_features, m_pad_w13, log_m_w13, rsqrt_m_w13, 1, stream);

            // ---- Activation ----
            {
                int n_act = (activation_type < 3) ? w13_out_features / 2 : w13_out_features;
                int blocks = (n_act + 255) / 256;
                const half* act_in = (const half*)h_w13.data_ptr<c10::Half>() + t * w13_out_features;
                half* act_out = (half*)h_act.data_ptr<c10::Half>() + t * intermediate_size;

                if (activation_type == 5) {  // relu2_no_mul
                    relu2_activation_kernel<<<blocks, 256, 0, stream>>>(act_in, act_out, n_act);
                } else if (activation_type == 0) {  // silu (gated)
                    silu_gated_activation_kernel<<<blocks, 256, 0, stream>>>(act_in, act_out, n_act);
                } else {
                    // Fallback: relu2_no_mul as default
                    relu2_activation_kernel<<<blocks, 256, 0, stream>>>(act_in, act_out, n_act);
                }
            }

            // ---- w2: input RHT (different per expert — input is h_act) ----
            launch_input_rht(
                (const half*)h_act.data_ptr<c10::Half>() + t * intermediate_size,
                (const half*)w2_SV.data_ptr<c10::Half>(),
                h_rht.data_ptr<float>() + t * n_pad_w2,
                intermediate_size, n_pad_w2, log_n_w2, rsqrt_n_w2, 1, stream);
            auto h_rht_half = h_rht.to(torch::kFloat16);

            // ---- w2: dequant+matmul ----
            y_rht_w2.zero_();
            launch_matvec_splitk(
                y_rht_w2.data_ptr<float>() + t * M_w2,
                (const half*)h_rht_half.data_ptr<c10::Half>() + t * n_pad_w2,
                w2_Qidxs[eidx].data_ptr<int16_t>(), cb_ptr, w2_ws,
                w2_s2 ? w2_Qidxs2[eidx].data_ptr<int16_t>() : nullptr,
                w2_s2 ? cb2_ptr : nullptr,
                w2_irs,
                w2_s3 ? w2_Qidxs3[eidx].data_ptr<int16_t>() : nullptr,
                w2_s3 ? cb3_ptr : nullptr,
                w2_irs2,
                nullptr, nullptr, 0.0f,
                w2_num_stages,
                M_w2, NB_w2, cb2_size, num_sms, stream);

            // ---- w2: output RHT ----
            launch_output_rht(
                y_rht_w2.data_ptr<float>() + t * M_w2,
                (const half*)w2_SU[eidx].data_ptr<c10::Half>(),
                (half*)expert_out.data_ptr<c10::Half>() + t * hidden_size,
                hidden_size, m_pad_w2, log_m_w2, rsqrt_m_w2, 1, stream);

            // ---- Weighted accumulate ----
            {
                int n = hidden_size;
                int blocks = (n + 255) / 256;
                weighted_add_kernel<<<blocks, 256, 0, stream>>>(
                    (half*)output.data_ptr<c10::Half>() + t * hidden_size,
                    (const half*)expert_out.data_ptr<c10::Half>() + t * hidden_size,
                    ew, n);
            }
        }
    }

    return output;
}


/* ─────────────────────────────────────────────────────────────────────
 * Block-diagonal sibling of glq_fused_moe_cuda for non-pow2 expert dims
 * (e.g. Nemotron-Cascade-2: hidden=5280, intermediate=1856, latent=2688).
 *
 * Replaces launch_input_rht / launch_output_rht (which assume a pow2 FHT
 * with a single log_n) by per-projection block decomposition metadata,
 * reusing the same multiblock / per-block FHT kernels as
 * glq_fused_linear_block_diag_cuda. Dequant+matmul (launch_matvec_splitk)
 * is dimension-agnostic and is called identically.
 * ───────────────────────────────────────────────────────────────────── */

torch::Tensor glq_fused_moe_block_diag_cuda(
    torch::Tensor x,                    // (num_tokens, hidden) fp16
    torch::Tensor topk_ids,             // (num_tokens, top_k) int64
    torch::Tensor topk_weights,         // (num_tokens, top_k) fp32
    // w13 (gate_up_proj) weights — all experts stacked
    torch::Tensor w13_Qidxs,            // (E, m_pad_w13, n_blocks_w13) int16
    torch::Tensor w13_SU,               // (E, m_pad_w13) fp16
    torch::Tensor w13_SV,               // (n_pad_w13,) fp16 — shared
    torch::Tensor w13_Wscale,           // (E,) fp32
    torch::Tensor w13_Qidxs2,           // empty or (E, m_pad_w13, n_blocks_w13)
    torch::Tensor w13_inv_rs,           // (E,) fp32
    // w2 (down_proj) weights — all experts stacked
    torch::Tensor w2_Qidxs,
    torch::Tensor w2_SU,
    torch::Tensor w2_SV,                // (n_pad_w2,) fp16 — shared
    torch::Tensor w2_Wscale,
    torch::Tensor w2_Qidxs2,
    torch::Tensor w2_inv_rs,
    // Codebooks
    torch::Tensor codebook,
    torch::Tensor codebook2,
    // Dimensions
    int hidden_size, int intermediate_size, int w13_out_features,
    int n_pad_w13, int m_pad_w13,
    int n_pad_w2, int m_pad_w2,
    // Block-diag metadata — replaces log_n_/log_m_ scalar args.
    // Each projection has separate n-axis (input RHT) and m-axis (output RHT) decomposition.
    torch::Tensor blocks_n_w13,         // 1D int64 CPU
    torch::Tensor blocks_m_w13,         // 1D int64 CPU
    torch::Tensor blocks_n_w13_meta,    // (num_blocks, 4) int32 GPU
    torch::Tensor blocks_m_w13_meta,    // (num_blocks, 4) int32 GPU
    torch::Tensor blocks_n_w2,
    torch::Tensor blocks_m_w2,
    torch::Tensor blocks_n_w2_meta,
    torch::Tensor blocks_m_w2_meta,
    int activation_type,
    // Stage-3 RVQ (optional). Pass empty tensors for 2-stage.
    torch::Tensor w13_Qidxs3,
    torch::Tensor w13_inv_rs2,
    torch::Tensor w2_Qidxs3,
    torch::Tensor w2_inv_rs2,
    torch::Tensor codebook3
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w13_Qidxs);
    CHECK_INPUT(codebook);

    int num_tokens = x.size(0);
    int top_k = topk_ids.size(1);
    int M_w13 = w13_Qidxs.size(1);
    int NB_w13 = w13_Qidxs.size(2);
    int M_w2 = w2_Qidxs.size(1);
    int NB_w2 = w2_Qidxs.size(2);
    bool w13_has_s2 = (w13_Qidxs2.numel() > 0);
    bool w2_has_s2 = (w2_Qidxs2.numel() > 0);
    bool w13_has_s3 = (w13_Qidxs3.numel() > 0 && w13_inv_rs2.numel() > 0);
    bool w2_has_s3 = (w2_Qidxs3.numel() > 0 && w2_inv_rs2.numel() > 0);
    int cb2_size = codebook2.numel() > 0 ? (int)codebook2.size(0) : 0;

    at::DeviceGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    static int num_sms = 0;
    if (num_sms == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, x.get_device());
        num_sms = prop.multiProcessorCount;
    }

    auto opts_f32 = torch::dtype(torch::kFloat32).device(x.device());
    auto opts_f16 = torch::dtype(torch::kFloat16).device(x.device());

    // Output accumulator (zeroed for weighted add)
    auto output = torch::zeros({num_tokens, hidden_size}, opts_f16);

    // Resolve block-diag metadata pointers up front
    const int64_t* bn_w13 = blocks_n_w13.data_ptr<int64_t>();
    int n_n_w13 = (int)blocks_n_w13.size(0);
    const int4* bn_w13_meta = blocks_n_w13_meta.numel() > 0
                                  ? (const int4*)blocks_n_w13_meta.data_ptr<int32_t>()
                                  : nullptr;
    bool bn_w13_meta_gpu = (bn_w13_meta != nullptr) && blocks_n_w13_meta.is_cuda();

    const int64_t* bm_w13 = blocks_m_w13.data_ptr<int64_t>();
    int n_m_w13 = (int)blocks_m_w13.size(0);
    const int4* bm_w13_meta = blocks_m_w13_meta.numel() > 0
                                  ? (const int4*)blocks_m_w13_meta.data_ptr<int32_t>()
                                  : nullptr;
    bool bm_w13_meta_gpu = (bm_w13_meta != nullptr) && blocks_m_w13_meta.is_cuda();

    const int64_t* bn_w2 = blocks_n_w2.data_ptr<int64_t>();
    int n_n_w2 = (int)blocks_n_w2.size(0);
    const int4* bn_w2_meta = blocks_n_w2_meta.numel() > 0
                                 ? (const int4*)blocks_n_w2_meta.data_ptr<int32_t>()
                                 : nullptr;
    bool bn_w2_meta_gpu = (bn_w2_meta != nullptr) && blocks_n_w2_meta.is_cuda();

    const int64_t* bm_w2 = blocks_m_w2.data_ptr<int64_t>();
    int n_m_w2 = (int)blocks_m_w2.size(0);
    const int4* bm_w2_meta = blocks_m_w2_meta.numel() > 0
                                 ? (const int4*)blocks_m_w2_meta.data_ptr<int32_t>()
                                 : nullptr;
    bool bm_w2_meta_gpu = (bm_w2_meta != nullptr) && blocks_m_w2_meta.is_cuda();

    // Shared input RHT: same x and w13_SV for all experts (block-diag along n-axis).
    auto x_rht = torch::empty({num_tokens, n_pad_w13}, opts_f32);
    launch_input_rht_block_diag(
        (const half*)x.data_ptr<c10::Half>(),
        (const half*)w13_SV.data_ptr<c10::Half>(),
        x_rht.data_ptr<float>(),
        hidden_size, n_pad_w13,
        num_tokens,
        bn_w13, n_n_w13,
        bn_w13_meta, bn_w13_meta_gpu,
        stream);
    auto x_rht_half = x_rht.to(torch::kFloat16);

    // Reusable per-expert buffers
    auto y_rht_w13 = torch::empty({num_tokens, M_w13}, opts_f32);
    auto h_w13 = torch::empty({num_tokens, w13_out_features}, opts_f16);
    auto h_act = torch::empty({num_tokens, intermediate_size}, opts_f16);
    auto h_rht = torch::empty({num_tokens, n_pad_w2}, opts_f32);
    auto y_rht_w2 = torch::empty({num_tokens, M_w2}, opts_f32);
    auto expert_out = torch::empty({num_tokens, hidden_size}, opts_f16);

    // Read expert IDs to CPU (tiny sync — 2 int64s for top-2)
    auto topk_ids_cpu = topk_ids.cpu();
    auto topk_w_cpu = topk_weights.cpu();

    const half* cb_ptr = (const half*)codebook.data_ptr<c10::Half>();
    const half* cb2_ptr = codebook2.numel() > 0 ? (const half*)codebook2.data_ptr<c10::Half>() : nullptr;
    const half* cb3_ptr = codebook3.numel() > 0 ? (const half*)codebook3.data_ptr<c10::Half>() : nullptr;

    for (int t = 0; t < num_tokens; t++) {
        for (int k = 0; k < top_k; k++) {
            int eidx = topk_ids_cpu[t][k].item<int64_t>();
            float ew = topk_w_cpu[t][k].item<float>();
            if (ew == 0.0f) continue;

            float w13_ws = w13_Wscale[eidx].item<float>();
            float w13_irs = w13_has_s2 ? w13_inv_rs[eidx].item<float>() : 0.0f;
            bool w13_s2 = (w13_irs != 0.0f);
            float w13_irs2 = (w13_s2 && w13_has_s3) ? w13_inv_rs2[eidx].item<float>() : 0.0f;
            bool w13_s3 = (w13_irs2 != 0.0f);

            float w2_ws = w2_Wscale[eidx].item<float>();
            float w2_irs = w2_has_s2 ? w2_inv_rs[eidx].item<float>() : 0.0f;
            bool w2_s2 = (w2_irs != 0.0f);
            float w2_irs2 = (w2_s2 && w2_has_s3) ? w2_inv_rs2[eidx].item<float>() : 0.0f;
            bool w2_s3 = (w2_irs2 != 0.0f);

            int w13_num_stages = 1 + (w13_s2 ? 1 : 0) + (w13_s3 ? 1 : 0);
            int w2_num_stages = 1 + (w2_s2 ? 1 : 0) + (w2_s3 ? 1 : 0);

            // ---- w13: dequant+matmul (input RHT already done) ----
            y_rht_w13.zero_();
            launch_matvec_splitk(
                y_rht_w13.data_ptr<float>() + t * M_w13,
                (const half*)x_rht_half.data_ptr<c10::Half>() + t * n_pad_w13,
                w13_Qidxs[eidx].data_ptr<int16_t>(), cb_ptr, w13_ws,
                w13_s2 ? w13_Qidxs2[eidx].data_ptr<int16_t>() : nullptr,
                w13_s2 ? cb2_ptr : nullptr,
                w13_irs,
                w13_s3 ? w13_Qidxs3[eidx].data_ptr<int16_t>() : nullptr,
                w13_s3 ? cb3_ptr : nullptr,
                w13_irs2,
                nullptr, nullptr, 0.0f,
                w13_num_stages,
                M_w13, NB_w13, cb2_size, num_sms, stream);

            // ---- w13: output RHT (block-diag along m-axis) ----
            launch_output_rht_block_diag(
                y_rht_w13.data_ptr<float>() + t * M_w13,
                (const half*)w13_SU[eidx].data_ptr<c10::Half>(),
                (half*)h_w13.data_ptr<c10::Half>() + t * w13_out_features,
                w13_out_features, m_pad_w13,
                1,
                bm_w13, n_m_w13,
                bm_w13_meta, bm_w13_meta_gpu,
                stream);

            // ---- Activation ----
            {
                int n_act = (activation_type < 3) ? w13_out_features / 2 : w13_out_features;
                int blocks = (n_act + 255) / 256;
                const half* act_in = (const half*)h_w13.data_ptr<c10::Half>() + t * w13_out_features;
                half* act_out = (half*)h_act.data_ptr<c10::Half>() + t * intermediate_size;

                if (activation_type == 5) {  // relu2_no_mul
                    relu2_activation_kernel<<<blocks, 256, 0, stream>>>(act_in, act_out, n_act);
                } else if (activation_type == 0) {  // silu (gated)
                    silu_gated_activation_kernel<<<blocks, 256, 0, stream>>>(act_in, act_out, n_act);
                } else {
                    relu2_activation_kernel<<<blocks, 256, 0, stream>>>(act_in, act_out, n_act);
                }
            }

            // ---- w2: input RHT (block-diag, per-expert input is h_act) ----
            launch_input_rht_block_diag(
                (const half*)h_act.data_ptr<c10::Half>() + t * intermediate_size,
                (const half*)w2_SV.data_ptr<c10::Half>(),
                h_rht.data_ptr<float>() + t * n_pad_w2,
                intermediate_size, n_pad_w2,
                1,
                bn_w2, n_n_w2,
                bn_w2_meta, bn_w2_meta_gpu,
                stream);
            auto h_rht_half = h_rht.to(torch::kFloat16);

            // ---- w2: dequant+matmul ----
            y_rht_w2.zero_();
            launch_matvec_splitk(
                y_rht_w2.data_ptr<float>() + t * M_w2,
                (const half*)h_rht_half.data_ptr<c10::Half>() + t * n_pad_w2,
                w2_Qidxs[eidx].data_ptr<int16_t>(), cb_ptr, w2_ws,
                w2_s2 ? w2_Qidxs2[eidx].data_ptr<int16_t>() : nullptr,
                w2_s2 ? cb2_ptr : nullptr,
                w2_irs,
                w2_s3 ? w2_Qidxs3[eidx].data_ptr<int16_t>() : nullptr,
                w2_s3 ? cb3_ptr : nullptr,
                w2_irs2,
                nullptr, nullptr, 0.0f,
                w2_num_stages,
                M_w2, NB_w2, cb2_size, num_sms, stream);

            // ---- w2: output RHT (block-diag along m-axis) ----
            launch_output_rht_block_diag(
                y_rht_w2.data_ptr<float>() + t * M_w2,
                (const half*)w2_SU[eidx].data_ptr<c10::Half>(),
                (half*)expert_out.data_ptr<c10::Half>() + t * hidden_size,
                hidden_size, m_pad_w2,
                1,
                bm_w2, n_m_w2,
                bm_w2_meta, bm_w2_meta_gpu,
                stream);

            // ---- Weighted accumulate ----
            {
                int n = hidden_size;
                int blocks = (n + 255) / 256;
                weighted_add_kernel<<<blocks, 256, 0, stream>>>(
                    (half*)output.data_ptr<c10::Half>() + t * hidden_size,
                    (const half*)expert_out.data_ptr<c10::Half>() + t * hidden_size,
                    ew, n);
            }
        }
    }

    return output;
}
