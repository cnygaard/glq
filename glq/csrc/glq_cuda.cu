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

#define BLOCKS_PER_SPLIT_DEFAULT 64

template <bool HAS_STAGE2>
__global__ void __launch_bounds__(256)
glq_matvec_splitk_kernel(
    float* __restrict__ output,         // (M,) fp32, pre-zeroed
    const half* __restrict__ x,
    const int16_t* __restrict__ qidxs,
    const half* __restrict__ codebook,  // (65536, 8) fp16
    const int16_t* __restrict__ qidxs2,
    const half* __restrict__ codebook2,
    float inv_resid_scale,
    float wscale,
    int M,
    int N_BLOCKS,
    int bps,                            // blocks per split (adaptive)
    int cb2_size                        // secondary codebook entries (0 if no stage2)
) {
    // Stage small secondary codebook into shared memory (256 entries = 4KB)
    extern __shared__ half smem_cb2[];

    if (HAS_STAGE2 && cb2_size > 0 && cb2_size <= 256) {
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
            const half* cb2 = (cb2_size > 0 && cb2_size <= 256) ? smem_cb2 : codebook2;
            uint16_t idx2 = 0;
            if (valid && elem == 0) {
                idx2 = (uint16_t)qidxs2[my_row * N_BLOCKS + j];
            }
            idx2 = __shfl_sync(FULL_MASK, idx2, row_in_warp * 8);
            if (valid) {
                cb_val += __half2float(cb2[idx2 * 8 + elem]) * inv_resid_scale;
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

template <bool HAS_STAGE2>
__global__ void __launch_bounds__(256, 2)
glq_matmul_tc_kernel(
    float* __restrict__ output,         // (B_dim, M) fp32, pre-zeroed
    const half* __restrict__ x,         // (B_dim, N) fp16 input
    const int16_t* __restrict__ qidxs,  // (M, N_BLOCKS) primary indices
    const half* __restrict__ codebook,  // (65536, 8) fp16
    const int16_t* __restrict__ qidxs2,
    const half* __restrict__ codebook2,
    float inv_resid_scale,
    float wscale,
    int B_dim,
    int M,
    int N,
    int N_BLOCKS,
    int stride_x,
    int bps,                            // blocks per K-split (adaptive)
    int cb2_size                        // secondary codebook entries (0 if no stage2)
) {
    // Stage small secondary codebook into shared memory (256 entries = 4KB)
    extern __shared__ half smem_cb2[];
    if (HAS_STAGE2 && cb2_size > 0 && cb2_size <= 256) {
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

        if (HAS_STAGE2 && m_valid) {
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
    float inv_resid_scale,
    torch::Tensor codebook_abs // unused, kept for ABI compat
) {
    CHECK_INPUT(x);
    CHECK_INPUT(qidxs);
    CHECK_INPUT(codebook);

    int M = qidxs.size(0);
    int N_BLOCKS = qidxs.size(1);

    at::DeviceGuard guard(x.device());
    // Split-K uses atomicAdd → output must be zeroed
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

    // Always use split-K: it handles all cases well (large M just gets k_splits=1)
    int m_blocks = (M + rows_per_block - 1) / rows_per_block;
    bool use_splitk = (N_BLOCKS >= BLOCKS_PER_SPLIT_DEFAULT);

    if (use_splitk) {
        // Adaptive BPS: shrink BPS when grid is undersaturated to create more CTAs
        int bps = BLOCKS_PER_SPLIT_DEFAULT;
        int k_splits = (N_BLOCKS + bps - 1) / bps;
        int total_ctas = m_blocks * k_splits;
        if (total_ctas < num_sms * 2 && bps > 16) {
            bps = max(16, bps / 2);
            k_splits = (N_BLOCKS + bps - 1) / bps;
        }
        dim3 grid(m_blocks, k_splits);

        if (has_stage2) {
            CHECK_INPUT(qidxs2);
            CHECK_INPUT(codebook2);
            int cb2_size = codebook2.size(0);
            int smem = (cb2_size <= 256) ? cb2_size * 8 * sizeof(half) : 0;
            glq_matvec_splitk_kernel<true><<<grid, block, smem, stream>>>(
                output.data_ptr<float>(),
                (const half*)x.data_ptr<c10::Half>(),
                qidxs.data_ptr<int16_t>(),
                (const half*)codebook.data_ptr<c10::Half>(),
                qidxs2.data_ptr<int16_t>(),
                (const half*)codebook2.data_ptr<c10::Half>(),
                inv_resid_scale,
                wscale, M, N_BLOCKS, bps, cb2_size
            );
        } else {
            glq_matvec_splitk_kernel<false><<<grid, block, 0, stream>>>(
                output.data_ptr<float>(),
                (const half*)x.data_ptr<c10::Half>(),
                qidxs.data_ptr<int16_t>(),
                (const half*)codebook.data_ptr<c10::Half>(),
                nullptr, nullptr, 0.0f,
                wscale, M, N_BLOCKS, bps, 0
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


torch::Tensor glq_dequant_matmul_cuda(
    torch::Tensor x,           // (B, N) fp16
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

    const half* x_ptr = (const half*)x.data_ptr<c10::Half>();
    const half* cb_ptr = (const half*)codebook.data_ptr<c10::Half>();
    const int16_t* q_ptr = qidxs.data_ptr<int16_t>();
    float* out_ptr = output.data_ptr<float>();

    const int16_t* q2_ptr = has_stage2 ? qidxs2.data_ptr<int16_t>() : nullptr;
    const half* cb2_ptr = has_stage2 ? (const half*)codebook2.data_ptr<c10::Half>() : nullptr;

    if (has_stage2) {
        CHECK_INPUT(qidxs2);
        CHECK_INPUT(codebook2);
    }

    // Use TC kernel: single launch for all B rows
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

    if (has_stage2) {
        int cb2_size = codebook2.size(0);
        int smem = (cb2_size <= 256) ? cb2_size * 8 * sizeof(half) : 0;
        glq_matmul_tc_kernel<true><<<tc_grid, tc_block, smem, stream>>>(
            out_ptr, x_ptr, q_ptr, cb_ptr,
            q2_ptr, cb2_ptr, inv_resid_scale,
            wscale, B_dim, M, N, N_BLOCKS, N, bps, cb2_size);
    } else {
        glq_matmul_tc_kernel<false><<<tc_grid, tc_block, 0, stream>>>(
            out_ptr, x_ptr, q_ptr, cb_ptr,
            nullptr, nullptr, 0.0f,
            wscale, B_dim, M, N, N_BLOCKS, N, bps, 0);
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

    int bps = TC_BPS_DEFAULT;
    int k_splits = (N_BLOCKS + bps - 1) / bps;
    int total_ctas = b_tiles * m_grid * k_splits;
    if (total_ctas < num_sms * 2 && bps > 16) {
        bps = max(16, bps / 2);
        k_splits = (N_BLOCKS + bps - 1) / bps;
    }

    dim3 tc_grid(b_tiles, m_grid, k_splits);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    glq_matmul_tc_packed_kernel<<<tc_grid, tc_block, 0, stream>>>(
        output.data_ptr<float>(),
        (const half*)x.data_ptr<c10::Half>(),
        qidxs.data_ptr<int16_t>(),
        (const uint32_t*)codebook_packed.data_ptr<int32_t>(),
        wscale, B_dim, M, N, N_BLOCKS, N, bps
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
    int log_n,
    int use_single_buffer            // 1 if smem too small for double-buffer
) {
    extern __shared__ float smem[];

    int b = blockIdx.x;
    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    // Step 1: Load x with zero-padding, multiply by SV signs
    float* buf = smem;
    for (int i = tid; i < n_pad; i += n_threads) {
        float x_val = (i < in_features) ? __half2float(x[b * stride_x + i]) : 0.0f;
        float s = __half2float(sv[i]);
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
        out[b * n_pad + i] = buf[i] * rsqrt_n;
    }
}


__global__ void glq_output_rht_kernel(
    const float* __restrict__ y_rht, // (B, m_pad) fp32 input
    const half* __restrict__ su,     // (m_pad,) fp16 sign vector
    half* __restrict__ out,          // (B, out_features) fp16 output
    int out_features,
    int m_pad,
    int log_m,
    float rsqrt_m,                   // 1.0 / sqrt(m_pad)
    int use_single_buffer
) {
    extern __shared__ float smem[];
    float* buf = smem;

    int b = blockIdx.x;
    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    // Step 1: Load y_rht
    for (int i = tid; i < m_pad; i += n_threads) {
        buf[i] = y_rht[b * m_pad + i];
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
    for (int i = tid; i < out_features; i += n_threads) {
        float val = buf[i] * rsqrt_m;
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
    int double_buf_smem = 2 * n_pad * sizeof(float);
    int single_buf_smem = n_pad * sizeof(float);
    // Use single-buffer if double doesn't fit in 96KB extended smem
    int use_single = (double_buf_smem > 96 * 1024) ? 1 : 0;
    int smem = use_single ? single_buf_smem : double_buf_smem;

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(glq_input_rht_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }

    glq_input_rht_kernel<<<B, threads, smem, stream>>>(
        (const half*)x.data_ptr<c10::Half>(),
        (const half*)sv.data_ptr<c10::Half>(),
        out.data_ptr<float>(),
        in_features, stride_x, rsqrt_n, n_pad, log_n, use_single
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
    int double_buf_smem = 2 * m_pad * sizeof(float);
    int single_buf_smem = m_pad * sizeof(float);
    int use_single = (double_buf_smem > 96 * 1024) ? 1 : 0;
    int smem = use_single ? single_buf_smem : double_buf_smem;

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(glq_output_rht_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }

    glq_output_rht_kernel<<<B, threads, smem, stream>>>(
        y_rht.data_ptr<float>(),
        (const half*)su.data_ptr<c10::Half>(),
        (half*)out.data_ptr<c10::Half>(),
        out_features, m_pad, log_m, rsqrt_m, use_single
    );
}


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
    float inv_resid_scale
) {
    CHECK_INPUT(x);
    CHECK_INPUT(qidxs);
    CHECK_INPUT(codebook);

    int B = x.size(0);
    int M = qidxs.size(0);
    int N_BLOCKS = qidxs.size(1);
    bool has_stage2 = (qidxs2.numel() > 0 && inv_resid_scale != 0.0f);

    at::DeviceGuard guard(x.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // ---- Step 1: Input RHT ----
    auto x_rht = torch::empty({B, n_pad}, torch::dtype(torch::kFloat32).device(x.device()));
    {
        int threads = min(n_pad, 1024);
        int double_buf = 2 * n_pad * sizeof(float);
        int single_buf = n_pad * sizeof(float);
        int use_single = (double_buf > 96 * 1024) ? 1 : 0;
        int smem = use_single ? single_buf : double_buf;
        float rsqrt_n = 1.0f / sqrtf((float)n_pad);

        if (smem > 48 * 1024) {
            cudaFuncSetAttribute(glq_input_rht_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        }

        glq_input_rht_kernel<<<B, threads, smem, stream>>>(
            (const half*)x.data_ptr<c10::Half>(),
            (const half*)sv.data_ptr<c10::Half>(),
            x_rht.data_ptr<float>(),
            in_features, in_features, rsqrt_n, n_pad, log_n, use_single
        );
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
            // B=1: split-K matvec
            const int WARPS = 8;
            const int rows_per_block = ROWS_PER_WARP * WARPS;
            dim3 block(32, WARPS);
            int m_blocks = (M + rows_per_block - 1) / rows_per_block;

            int bps = BLOCKS_PER_SPLIT_DEFAULT;
            int k_splits = (N_BLOCKS + bps - 1) / bps;
            int total_ctas = m_blocks * k_splits;
            if (total_ctas < num_sms * 2 && bps > 16) {
                bps = max(16, bps / 2);
                k_splits = (N_BLOCKS + bps - 1) / bps;
            }
            dim3 grid(m_blocks, k_splits);

            if (has_stage2) {
                int cb2_size = codebook2.size(0);
                int smem = (cb2_size <= 256) ? cb2_size * 8 * sizeof(half) : 0;
                glq_matvec_splitk_kernel<true><<<grid, block, smem, stream>>>(
                    y_rht.data_ptr<float>(),
                    (const half*)x_rht_half.data_ptr<c10::Half>(),  // x_rht is fp32 but kernel expects fp16 input
                    qidxs.data_ptr<int16_t>(),
                    (const half*)codebook.data_ptr<c10::Half>(),
                    qidxs2.data_ptr<int16_t>(),
                    (const half*)codebook2.data_ptr<c10::Half>(),
                    inv_resid_scale,
                    wscale, M, N_BLOCKS, bps, cb2_size
                );
            } else {
                glq_matvec_splitk_kernel<false><<<grid, block, 0, stream>>>(
                    y_rht.data_ptr<float>(),
                    (const half*)x_rht_half.data_ptr<c10::Half>(),
                    qidxs.data_ptr<int16_t>(),
                    (const half*)codebook.data_ptr<c10::Half>(),
                    nullptr, nullptr, 0.0f,
                    wscale, M, N_BLOCKS, bps, 0
                );
            }
        } else {
            // B>=2: TC matmul
            const int WARPS = 8;
            dim3 tc_block(32, WARPS);
            int b_tiles = (B + 15) / 16;
            int m_tiles = (M + 7) / 8;
            int m_grid = (m_tiles + WARPS - 1) / WARPS;
            int bps = TC_BPS_DEFAULT;
            int k_splits = (N_BLOCKS + bps - 1) / bps;

            dim3 tc_grid(b_tiles, m_grid, k_splits);

            if (has_stage2) {
                int cb2_size = codebook2.size(0);
                int smem = (cb2_size <= 256) ? cb2_size * 8 * sizeof(half) : 0;
                glq_matmul_tc_kernel<true><<<tc_grid, tc_block, smem, stream>>>(
                    y_rht.data_ptr<float>(),
                    (const half*)x_rht_half.data_ptr<c10::Half>(),
                    qidxs.data_ptr<int16_t>(),
                    (const half*)codebook.data_ptr<c10::Half>(),
                    qidxs2.data_ptr<int16_t>(),
                    (const half*)codebook2.data_ptr<c10::Half>(),
                    inv_resid_scale,
                    wscale, B, M, n_pad, N_BLOCKS, n_pad, bps, cb2_size
                );
            } else {
                glq_matmul_tc_kernel<false><<<tc_grid, tc_block, 0, stream>>>(
                    y_rht.data_ptr<float>(),
                    (const half*)x_rht_half.data_ptr<c10::Half>(),
                    qidxs.data_ptr<int16_t>(),
                    (const half*)codebook.data_ptr<c10::Half>(),
                    nullptr, nullptr, 0.0f,
                    wscale, B, M, n_pad, N_BLOCKS, n_pad, bps, 0
                );
            }
        }
    }

    // ---- Step 3: Output RHT ----
    auto y = torch::empty({B, out_features}, torch::dtype(torch::kFloat16).device(x.device()));
    {
        int threads = min(m_pad, 1024);
        int double_buf = 2 * m_pad * sizeof(float);
        int single_buf = m_pad * sizeof(float);
        int use_single = (double_buf > 96 * 1024) ? 1 : 0;
        int smem = use_single ? single_buf : double_buf;
        float rsqrt_m = 1.0f / sqrtf((float)m_pad);

        if (smem > 48 * 1024) {
            cudaFuncSetAttribute(glq_output_rht_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        }

        glq_output_rht_kernel<<<B, threads, smem, stream>>>(
            y_rht.data_ptr<float>(),
            (const half*)su.data_ptr<c10::Half>(),
            (half*)y.data_ptr<c10::Half>(),
            out_features, m_pad, log_m, rsqrt_m, use_single
        );
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

// Helper: launch split-K matvec from C++ (avoids duplicating the grid logic)
static void launch_matvec_splitk(
    float* output, const half* x, const int16_t* qidxs,
    const half* codebook, float wscale,
    const int16_t* qidxs2, const half* codebook2,
    float inv_resid_scale, bool has_stage2,
    int M, int N_BLOCKS, int cb2_size,
    int num_sms, cudaStream_t stream
) {
    const int WARPS = 8;
    const int rows_per_block = ROWS_PER_WARP * WARPS;
    dim3 block(32, WARPS);
    int m_blocks = (M + rows_per_block - 1) / rows_per_block;
    int bps = BLOCKS_PER_SPLIT_DEFAULT;
    int k_splits = (N_BLOCKS + bps - 1) / bps;
    if (m_blocks * k_splits < num_sms * 2 && bps > 16) {
        bps = max(16, bps / 2);
        k_splits = (N_BLOCKS + bps - 1) / bps;
    }
    dim3 grid(m_blocks, k_splits);

    if (has_stage2) {
        int smem = (cb2_size <= 256) ? cb2_size * 8 * sizeof(half) : 0;
        glq_matvec_splitk_kernel<true><<<grid, block, smem, stream>>>(
            output, x, qidxs, codebook, qidxs2, codebook2,
            inv_resid_scale, wscale, M, N_BLOCKS, bps, cb2_size);
    } else {
        glq_matvec_splitk_kernel<false><<<grid, block, 0, stream>>>(
            output, x, qidxs, codebook, nullptr, nullptr, 0.0f,
            wscale, M, N_BLOCKS, bps, 0);
    }
}

// Helper: launch input/output RHT
static void launch_input_rht(
    const half* x, const half* sv, float* out,
    int in_features, int n_pad, int log_n, float rsqrt_n,
    int B, cudaStream_t stream
) {
    int threads = min(n_pad, 1024);
    int dbl = 2 * n_pad * sizeof(float);
    int sgl = n_pad * sizeof(float);
    int use_single = (dbl > 96*1024) ? 1 : 0;
    int smem = use_single ? sgl : dbl;
    if (smem > 48*1024)
        cudaFuncSetAttribute(glq_input_rht_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    glq_input_rht_kernel<<<B, threads, smem, stream>>>(
        x, sv, out, in_features, in_features, rsqrt_n, n_pad, log_n, use_single);
}

static void launch_output_rht(
    const float* y_rht, const half* su, half* out,
    int out_features, int m_pad, int log_m, float rsqrt_m,
    int B, cudaStream_t stream
) {
    int threads = min(m_pad, 1024);
    int dbl = 2 * m_pad * sizeof(float);
    int sgl = m_pad * sizeof(float);
    int use_single = (dbl > 96*1024) ? 1 : 0;
    int smem = use_single ? sgl : dbl;
    if (smem > 48*1024)
        cudaFuncSetAttribute(glq_output_rht_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    glq_output_rht_kernel<<<B, threads, smem, stream>>>(
        y_rht, su, out, out_features, m_pad, log_m, rsqrt_m, use_single);
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
    int activation_type  // 0=silu(gated), 5=relu2_no_mul, etc.
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

    for (int t = 0; t < num_tokens; t++) {
        for (int k = 0; k < top_k; k++) {
            int eidx = topk_ids_cpu[t][k].item<int64_t>();
            float ew = topk_w_cpu[t][k].item<float>();
            if (ew == 0.0f) continue;

            float w13_ws = w13_Wscale[eidx].item<float>();
            float w13_irs = w13_has_s2 ? w13_inv_rs[eidx].item<float>() : 0.0f;
            bool w13_s2 = (w13_irs != 0.0f);

            float w2_ws = w2_Wscale[eidx].item<float>();
            float w2_irs = w2_has_s2 ? w2_inv_rs[eidx].item<float>() : 0.0f;
            bool w2_s2 = (w2_irs != 0.0f);

            // ---- w13: dequant+matmul (input RHT already done) ----
            y_rht_w13.zero_();
            launch_matvec_splitk(
                y_rht_w13.data_ptr<float>() + t * M_w13,
                (const half*)x_rht_half.data_ptr<c10::Half>() + t * n_pad_w13,
                w13_Qidxs[eidx].data_ptr<int16_t>(), cb_ptr, w13_ws,
                w13_s2 ? w13_Qidxs2[eidx].data_ptr<int16_t>() : nullptr,
                w13_s2 ? cb2_ptr : nullptr,
                w13_irs, w13_s2, M_w13, NB_w13, cb2_size, num_sms, stream);

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
                w2_irs, w2_s2, M_w2, NB_w2, cb2_size, num_sms, stream);

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
