/* glq_trellis.cu — QTIP trellis-coded-quantization (TCQ) decode kernels for GLQ.
 *
 * Ported from Cornell-RelaxML/qtip `qtip-kernels/src/inference.cu`
 * (`kernel_decompress_matvec`). QTIP is GPL-3.0, same license as GLQ.
 *
 * Two deviations from the upstream kernel, both deliberate:
 *
 *   1. **Runtime-generic.** Upstream templates on (M, N, K) and JIT-compiles a wrapper per
 *      weight shape (`decompress_matvec_qtip_{m}_1_{numel}_{K}`). GLQ must serve arbitrary
 *      models from one build, so M/K are runtime kernel args and the compile-time
 *      `static_assert`s become host-side TORCH_CHECKs. Only R (bits/weight) stays a template
 *      param — the bit-unpack differs structurally per R.
 *
 *   2. **CUDA-graph safe.** Upstream calls cudaGetDeviceProperties + cudaFuncSetAttribute on
 *      EVERY launch. cudaGetDeviceProperties is slow and stream-order-hostile; both are
 *      hoisted here into a one-time `std::call_once` init so the steady-state decode path is
 *      pure kernel launches on the current stream (capturable by HF/vLLM cudagraphs, which
 *      is the whole point — GLQ's decode win only materializes under a captured graph).
 *
 * Storage layout is QTIP's exactly (see glq/trellis.py: `pack_layer` + `kernel_tile_flip` +
 * `_PERMUTE`), so `decode_layer` in that module is this kernel's bit-exact oracle.
 *
 * Fixed HYB codebook params (the only variant with a shipped kernel):
 *   L=16 (shift-register width) · S=9 (tlut index bits) · V=1 (log2 of the VQ dim, i.e. 2
 *   weights per trellis step). R = bits/weight = K in the Python API (2, 3 or 4).
 */
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <mutex>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x)       TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_INPUT
#define CHECK_INPUT(x)      do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while (false)
#endif

// RHT helpers live in glq_cuda.cu — the trellis fused op reuses them verbatim (Steps 1 & 3),
// exactly like glq_fused_linear_e8p_cuda does, so quant-time and inference-time RHT match.
void glq_input_rht_blockdiag_cuda(torch::Tensor x, torch::Tensor sv, torch::Tensor x_rht,
                                  int in_features, int n_pad,
                                  torch::Tensor blocks_n, torch::Tensor blocks_n_meta);
void glq_output_rht_blockdiag_cuda(torch::Tensor y_rht, torch::Tensor su, torch::Tensor y,
                                   int out_features, int m_pad,
                                   torch::Tensor blocks_m, torch::Tensor blocks_m_meta);

namespace {

constexpr uint32_t TR_WARP_SIZE   = 32;
constexpr uint32_t TR_BLOCK_SIZE  = 1024;
constexpr uint32_t TR_BLOCK_COUNT = 128;
constexpr uint32_t TR_WARPS       = TR_BLOCK_SIZE / TR_WARP_SIZE;   // 32
constexpr uint32_t TR_MMA_M       = 16;
constexpr uint32_t TR_MMA_K       = 16;
constexpr uint32_t TR_L           = 16;   // shift-register width
constexpr uint32_t TR_S           = 9;    // tlut index bits  → 512 entries
constexpr uint32_t TR_V           = 1;    // log2(vq dim)     → 2 weights / step
constexpr uint32_t TR_FULL_MASK   = 0xFFFFFFFFU;
// tlut (512 half2) replicated once per lane → conflict-free smem lookup. 1<<(S+5+V+1) bytes.
constexpr uint32_t TR_SMEM_BYTES  = 1u << (TR_S + 5 + TR_V + 1);    // 65536 = 64 KiB

union ditto2 { uint2 u32x2; uint32_t u32[2]; half2 f16x2[2]; };
union ditto4 { uint4 u32x4; uint32_t u32[4]; half2 f16x2[4]; };

__inline__ __device__ uint2 tr_ld_cs(const uint2 *p) {
    uint2 out;
    asm("ld.global.cs.v2.u32 {%0, %1}, [%2];" : "=r"(out.x), "=r"(out.y) : "l"(p));
    return out;
}
__inline__ __device__ uint3 tr_ld_cs(const uint3 *p) {
    uint3 out;
    asm("ld.global.cs.u32 %0, [%1];"    : "=r"(out.x) : "l"(p));
    asm("ld.global.cs.u32 %0, [%1+4];"  : "=r"(out.y) : "l"(p));
    asm("ld.global.cs.u32 %0, [%1+8];"  : "=r"(out.z) : "l"(p));
    return out;
}
__inline__ __device__ uint4 tr_ld_cs(const uint4 *p) {
    uint4 out;
    asm("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w) : "l"(p));
    return out;
}
__inline__ __device__ uint32_t tr_ld_x(const uint32_t *p) {
    uint32_t out;
    asm("ld.global.L1::evict_last.u32 %0, [%1];" : "=r"(out) : "l"(p));
    return out;
}

/* Warp-shuffle bit-unpack: lane l holds its 16-bit chunks and pulls lane l+1's (tail-biting
 * wraps at lane 31 → lane 0), reconstructing the OVERLAPPING 32-bit windows from which four
 * successive L=16 trellis states are extracted at 4-bit strides. QTIP-verbatim. */
template <uint32_t R>
__device__ inline void tr_load_reg_cs(const uint16_t *__restrict__ compressed, int weight_idx,
                                      uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
    if constexpr (R == 2) {
        ditto2 reg_load; reg_load.u32x2 = tr_ld_cs((const uint2 *)&compressed[weight_idx]);
        uint32_t next1 = __shfl_sync(TR_FULL_MASK, reg_load.u32x2.x, laneId + 1);
        uint32_t next2 = __shfl_sync(TR_FULL_MASK, reg_load.u32x2.y, laneId + 1);
        reg_cs_next.x = __byte_perm(next1, reg_load.u32x2.x, 0x5410);
        reg_cs_next.y = __byte_perm(next1, reg_load.u32x2.x, 0x7632);
        reg_cs_next.z = __byte_perm(next2, reg_load.u32x2.y, 0x5410);
        reg_cs_next.w = __byte_perm(next2, reg_load.u32x2.y, 0x7632);
    } else if constexpr (R == 3) {
        uint3 reg_load = tr_ld_cs((const uint3 *)&compressed[weight_idx]);
        uint32_t r1 = reg_load.x, r2 = reg_load.y, r3 = reg_load.z;
        uint32_t reg_24_1 = r1 & 0xffffff;
        uint32_t reg_24_2 = ((r1 >> 24) | (r2 << 8)) & 0xffffff;
        uint32_t reg_24_3 = ((r2 >> 16) | (r3 << 16)) & 0xffffff;
        uint32_t reg_24_4 = (r3 >> 8) & 0xffffff;
        uint32_t pack1 = (reg_24_1 >> 8) | ((reg_24_2 << 8) & 0xffff0000);
        uint32_t pack3 = (reg_24_3 >> 8) | ((reg_24_4 << 8) & 0xffff0000);
        uint32_t next1 = __shfl_sync(TR_FULL_MASK, pack1, laneId + 1);
        uint32_t next3 = __shfl_sync(TR_FULL_MASK, pack3, laneId + 1);
        reg_cs_next.x = __byte_perm(next1, reg_24_1, 0x6541);
        reg_cs_next.y = __byte_perm(next1, reg_24_2, 0x6543);
        reg_cs_next.z = __byte_perm(next3, reg_24_3, 0x6541);
        reg_cs_next.w = __byte_perm(next3, reg_24_4, 0x6543);
        reg_cs2_next.x = ((next1 >> 6) & 0x3ff) | (reg_24_1 << 10);
        reg_cs2_next.y = ((next1 >> (6 + 16)) & 0x3ff) | (reg_24_2 << 10);
        reg_cs2_next.z = ((next3 >> 6) & 0x3ff) | (reg_24_3 << 10);
        reg_cs2_next.w = ((next3 >> (6 + 16)) & 0x3ff) | (reg_24_4 << 10);
    } else if constexpr (R == 4) {
        uint4 reg_load = tr_ld_cs((const uint4 *)&compressed[weight_idx]);
        uint32_t r1 = reg_load.x, r2 = reg_load.y, r3 = reg_load.z, r4 = reg_load.w;
        uint32_t pack1 = (r1 >> 16) | (r2 & 0xffff0000);
        uint32_t pack3 = (r3 >> 16) | (r4 & 0xffff0000);
        uint32_t next1 = __shfl_sync(TR_FULL_MASK, pack1, laneId + 1);
        uint32_t next3 = __shfl_sync(TR_FULL_MASK, pack3, laneId + 1);
        reg_cs_next.x = r1; reg_cs_next.y = r2; reg_cs_next.z = r3; reg_cs_next.w = r4;
        reg_cs2_next.x = __byte_perm(next1, r1, 0x0041);
        reg_cs2_next.y = __byte_perm(next1, r2, 0x0043);
        reg_cs2_next.z = __byte_perm(next3, r3, 0x0041);
        reg_cs2_next.w = __byte_perm(next3, r4, 0x0043);
    }
}

/* Decode one MMA A-fragment (4 half2 = 8 weights) from a 32-bit code window.
 * Mirrors glq/trellis.py `quantlut_sym`: state → idx*(idx+1) → tlut[(idx>>6) & 0x1ff] →
 * flip the sign of component 0 when bit 15 is set. The (laneId<<1) in `masked_idx` selects
 * this lane's private replica of the tlut entry (the ×32 replication in smem). */
template <uint32_t R>
__device__ inline void tr_decode_regw(uint32_t reg_c, uint32_t reg_c2, uint32_t laneId,
                                      const half2 *__restrict__ smem_codebook, ditto4 &reg_w) {
#pragma unroll
    for (uint32_t j = 0; j < 4; j += 1) {
        uint32_t idx;
        if constexpr (R == 2)      idx = reg_c >> (4 * (4 - j));
        else if constexpr (R == 3) idx = (j < 3) ? (reg_c >> (6 * (2 - j) + 4)) : reg_c2;
        else                       idx = (j < 3) ? (reg_c >> (8 * (2 - j)))     : reg_c2;

        idx = idx * (idx + 1);                                   // the bitshift trellis map
        uint32_t masked_idx = (idx & 0x7FC0u) | (laneId << 1);   // bits 6..14 → tlut index
        reg_w.f16x2[j] = smem_codebook[masked_idx >> 1];
        reg_w.u32[j] ^= (0x00008000u & idx);                     // sign-flip component 0
    }
}

/* ── Fused B=1 GEMV: bit-unpack → trellis decode → tensor-core mma → block reduce ──
 * out (m,) fp32 = W (m,k) @ x (k,). Each block owns a disjoint m-range and reduces across
 * its 32 warps in smem — no atomics, no cross-block split-K → bit-stable output. */
template <uint32_t R>
__global__ static void __launch_bounds__(TR_BLOCK_SIZE, 1)
glq_trellis_matvec_kernel(float *__restrict__ out,
                          const uint32_t *__restrict__ compressed,
                          const half2 *__restrict__ x,
                          const half2 *__restrict__ codebook,
                          uint32_t m, uint32_t k) {
    extern __shared__ __align__(16) half2 smem_codebook[];

    const uint32_t laneId = threadIdx.x % TR_WARP_SIZE;
    const uint32_t warpId = threadIdx.x / TR_WARP_SIZE;

    const uint32_t tileCountM = m / TR_MMA_M;
    const uint32_t tileCountK = k / TR_MMA_K;
    const uint32_t m_per_block = (tileCountM + (2 * TR_BLOCK_COUNT) - 1) / (2 * TR_BLOCK_COUNT);
    const uint32_t k_per_block = tileCountK / (TR_WARPS * 4) * 2;
    const uint32_t this_warp_k =
        (warpId < (tileCountK % (TR_WARPS * 4)) / 4) ? k_per_block + 2 : k_per_block;

    const uint32_t u16_per_tile       = TR_MMA_M * TR_MMA_K * R / 16;   // 32 u16 @ R=2
    const uint32_t u16_per_tile_block = u16_per_tile * 4;               // 2m × 2k tiles
    const uint32_t weight_step        = TR_WARPS * u16_per_tile_block;
    const uint32_t weight_row_step    = tileCountK * u16_per_tile * 2;  // 2 rows of m-tiles
    const uint32_t f16x2_per_x_tile   = TR_MMA_K / 2;                   // 8
    const uint32_t x_half2            = k / 2;

    uint32_t tileIdM = m_per_block * blockIdx.x;

    // tlut → smem, replicated ×32 (one private copy per lane, so the LUT gather is
    // bank-conflict free). Threads t and t+512 cooperate to fill all 32 replicas.
    {
        uint32_t my_cb_idx = threadIdx.x & 0x1ff;
        half2 my_cb = codebook[my_cb_idx];
        for (uint32_t i = 0; i < 32; i += 2)
            smem_codebook[(my_cb_idx << 5) | (i ^ (threadIdx.x & 0x1f) ^ (threadIdx.x >> 9))] = my_cb;
        __syncthreads();
    }

    __shared__ ditto2 x_buf[TR_WARPS][4][4];
    __shared__ float reduce_gather[TR_WARPS][2][16];

    for (uint32_t mi = 0; mi < m_per_block; mi += 1) {
        if (tileIdM * 2 >= tileCountM) return;

        int weight_idx = tileIdM * weight_row_step + warpId * u16_per_tile_block * 2
                       + laneId * (u16_per_tile_block / TR_WARP_SIZE);
        uint4 reg_cs_next = {}, reg_cs2_next = {};
        tr_load_reg_cs<R>((const uint16_t *)compressed, weight_idx, laneId, reg_cs_next, reg_cs2_next);
        uint4 reg_cs, reg_cs2;
        float4 reg_p[2] = {};

        uint32_t x_idx      = warpId * f16x2_per_x_tile * 4 + laneId;
        uint32_t x_idx_step = TR_WARPS * f16x2_per_x_tile * 4;

#pragma unroll 4
        for (uint32_t ki = 0; ki < this_warp_k; ki += 1) {
            if (ki + 1 != this_warp_k && ki % 2 == 1) weight_idx += weight_step * 2;
            reg_cs = reg_cs_next; reg_cs2 = reg_cs2_next;
            tr_load_reg_cs<R>((const uint16_t *)compressed,
                              weight_idx + (1 - ki % 2) * u16_per_tile_block,
                              laneId, reg_cs_next, reg_cs2_next);

            if (ki % 2 == 0) {
                __syncwarp();
                x_buf[warpId][laneId / 8][laneId % 4].u32[(laneId % 8) / 4] =
                    tr_ld_x(reinterpret_cast<const uint32_t *>(x) + x_idx);
                __syncwarp();
                x_idx += x_idx_step;
            }

#pragma unroll 2
            for (uint32_t subki = 0; subki < 2; subki += 1) {
                // m16n8k16 B-fragment: with N==1 only column 0 matters, held by lanes 0-3.
                // The other lanes feed columns 1-7, whose results are never read — but zero
                // them anyway so no uninitialised (possibly NaN) register enters the mma.
                ditto2 reg_a; reg_a.u32[0] = 0u; reg_a.u32[1] = 0u;
                if (laneId < 4) reg_a.u32x2 = x_buf[warpId][ki % 2 * 2 + subki][laneId].u32x2;

#pragma unroll 2
                for (uint32_t submi = 0; submi < 2; submi += 1) {
                    uint32_t reg_c, reg_c2;
                    if      (submi == 0 && subki == 0) { reg_c = reg_cs.x; reg_c2 = reg_cs2.x; }
                    else if (submi == 1 && subki == 0) { reg_c = reg_cs.y; reg_c2 = reg_cs2.y; }
                    else if (submi == 0 && subki == 1) { reg_c = reg_cs.z; reg_c2 = reg_cs2.z; }
                    else                               { reg_c = reg_cs.w; reg_c2 = reg_cs2.w; }

                    ditto4 reg_w;
                    tr_decode_regw<R>(reg_c, reg_c2, laneId, smem_codebook, reg_w);

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                        " {%0, %1, %2, %3},"
                        " {%4, %5, %6, %7},"
                        " {%8, %9},"
                        " {%0, %1, %2, %3};"
                        : "+f"(reg_p[submi].x), "+f"(reg_p[submi].y),
                          "+f"(reg_p[submi].z), "+f"(reg_p[submi].w)
                        : "r"(reg_w.u32[0]), "r"(reg_w.u32[1]),
                          "r"(reg_w.u32[2]), "r"(reg_w.u32[3]),
                          "r"(reg_a.u32[0]), "r"(reg_a.u32[1]));
                }
            }
            // Upstream prefetches unconditionally; bound it so we never touch x past its end.
            if (ki % 2 == 0 && (x_idx + x_idx_step * 4) < x_half2)
                asm("prefetch.global.L1 [%0];" ::"l"(x + x_idx + x_idx_step * 4));
        }

        // m16n8k16 C-fragment: column 0 lives in c0/c2 of the lanes with laneId%4 == 0.
        if (laneId % 4 == 0) {
            for (int pi = 0; pi < 2; pi++) {
                reduce_gather[warpId][pi][laneId / 4]     = reg_p[pi].x;
                reduce_gather[warpId][pi][laneId / 4 + 8] = reg_p[pi].z;
            }
        }
        __syncthreads();
        if (warpId < 1) {
            int pi = laneId / 16;
            float reduced = 0.0f;
            for (uint32_t warpi = 0; warpi < TR_WARPS; warpi++)
                reduced += reduce_gather[warpi][pi][laneId % 16];
            out[(tileIdM * 2) * TR_MMA_M + laneId] = reduced;
        }
        if (m_per_block > 1) __syncthreads();
        tileIdM += 1;
    }
}

/* ── Decompress: identical weight walk + identical decode, but scatter W instead of mma ──
 * Shares tr_load_reg_cs + tr_decode_regw with the matvec kernel, so a bit-exact test of THIS
 * kernel against glq/trellis.py:decode_layer also pins the matvec's decode. Used for B>1
 * (prefill) and as the correctness oracle.
 *
 * Tile mapping (derived from the packed byte layout, cross-checked against the x-index math):
 *   submi ↔ m-tile within the pair, subki ↔ k-tile within the pair
 *   m_tile = tileIdM*2 + submi
 *   k_tile = 4*warpId + 2*(ki%2) + subki + (4*TR_WARPS)*(ki/2)
 * Fragment→(row,col) is the standard m16n8k16 A layout. */
template <uint32_t R>
__global__ static void __launch_bounds__(TR_BLOCK_SIZE, 1)
glq_trellis_decompress_kernel(half *__restrict__ W,
                              const uint32_t *__restrict__ compressed,
                              const half2 *__restrict__ codebook,
                              uint32_t m, uint32_t k) {
    extern __shared__ __align__(16) half2 smem_codebook[];

    const uint32_t laneId = threadIdx.x % TR_WARP_SIZE;
    const uint32_t warpId = threadIdx.x / TR_WARP_SIZE;

    const uint32_t tileCountM = m / TR_MMA_M;
    const uint32_t tileCountK = k / TR_MMA_K;
    const uint32_t m_per_block = (tileCountM + (2 * TR_BLOCK_COUNT) - 1) / (2 * TR_BLOCK_COUNT);
    const uint32_t k_per_block = tileCountK / (TR_WARPS * 4) * 2;
    const uint32_t this_warp_k =
        (warpId < (tileCountK % (TR_WARPS * 4)) / 4) ? k_per_block + 2 : k_per_block;

    const uint32_t u16_per_tile       = TR_MMA_M * TR_MMA_K * R / 16;
    const uint32_t u16_per_tile_block = u16_per_tile * 4;
    const uint32_t weight_step        = TR_WARPS * u16_per_tile_block;
    const uint32_t weight_row_step    = tileCountK * u16_per_tile * 2;

    uint32_t tileIdM = m_per_block * blockIdx.x;

    {
        uint32_t my_cb_idx = threadIdx.x & 0x1ff;
        half2 my_cb = codebook[my_cb_idx];
        for (uint32_t i = 0; i < 32; i += 2)
            smem_codebook[(my_cb_idx << 5) | (i ^ (threadIdx.x & 0x1f) ^ (threadIdx.x >> 9))] = my_cb;
        __syncthreads();
    }

    const uint32_t groupID = laneId >> 2;    // A-fragment row within the 16×16 tile
    const uint32_t tig     = laneId & 3;     // A-fragment column group

    for (uint32_t mi = 0; mi < m_per_block; mi += 1) {
        if (tileIdM * 2 >= tileCountM) return;

        int weight_idx = tileIdM * weight_row_step + warpId * u16_per_tile_block * 2
                       + laneId * (u16_per_tile_block / TR_WARP_SIZE);
        uint4 reg_cs_next = {}, reg_cs2_next = {};
        tr_load_reg_cs<R>((const uint16_t *)compressed, weight_idx, laneId, reg_cs_next, reg_cs2_next);
        uint4 reg_cs, reg_cs2;

        for (uint32_t ki = 0; ki < this_warp_k; ki += 1) {
            if (ki + 1 != this_warp_k && ki % 2 == 1) weight_idx += weight_step * 2;
            reg_cs = reg_cs_next; reg_cs2 = reg_cs2_next;
            tr_load_reg_cs<R>((const uint16_t *)compressed,
                              weight_idx + (1 - ki % 2) * u16_per_tile_block,
                              laneId, reg_cs_next, reg_cs2_next);

            for (uint32_t subki = 0; subki < 2; subki += 1) {
                const uint32_t k_tile = 4 * warpId + 2 * (ki % 2) + subki + (4 * TR_WARPS) * (ki / 2);
                for (uint32_t submi = 0; submi < 2; submi += 1) {
                    uint32_t reg_c, reg_c2;
                    if      (submi == 0 && subki == 0) { reg_c = reg_cs.x; reg_c2 = reg_cs2.x; }
                    else if (submi == 1 && subki == 0) { reg_c = reg_cs.y; reg_c2 = reg_cs2.y; }
                    else if (submi == 0 && subki == 1) { reg_c = reg_cs.z; reg_c2 = reg_cs2.z; }
                    else                               { reg_c = reg_cs.w; reg_c2 = reg_cs2.w; }

                    ditto4 reg_w;
                    tr_decode_regw<R>(reg_c, reg_c2, laneId, smem_codebook, reg_w);

                    const uint32_t m_tile = tileIdM * 2 + submi;
                    const uint32_t r0 = m_tile * TR_MMA_M + groupID;
                    const uint32_t c0 = k_tile * TR_MMA_K + 2 * tig;
                    // a0,a1 → (r0, c0..c0+1)          a2,a3 → (r0+8, c0..c0+1)
                    // a4,a5 → (r0, c0+8..c0+9)        a6,a7 → (r0+8, c0+8..c0+9)
                    W[(size_t)r0 * k + c0]           = reg_w.f16x2[0].x;
                    W[(size_t)r0 * k + c0 + 1]       = reg_w.f16x2[0].y;
                    W[(size_t)(r0 + 8) * k + c0]     = reg_w.f16x2[1].x;
                    W[(size_t)(r0 + 8) * k + c0 + 1] = reg_w.f16x2[1].y;
                    W[(size_t)r0 * k + c0 + 8]       = reg_w.f16x2[2].x;
                    W[(size_t)r0 * k + c0 + 9]       = reg_w.f16x2[2].y;
                    W[(size_t)(r0 + 8) * k + c0 + 8] = reg_w.f16x2[3].x;
                    W[(size_t)(r0 + 8) * k + c0 + 9] = reg_w.f16x2[3].y;
                }
            }
        }
        tileIdM += 1;
    }
}

/* One-time device/func setup. Upstream did this per launch (cudaGetDeviceProperties is slow
 * and hostile to graph capture); doing it once keeps the steady-state path launch-only. */
void tr_init_once() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        cudaFuncSetAttribute(glq_trellis_matvec_kernel<2>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, TR_SMEM_BYTES);
        cudaFuncSetAttribute(glq_trellis_matvec_kernel<3>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, TR_SMEM_BYTES);
        cudaFuncSetAttribute(glq_trellis_matvec_kernel<4>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, TR_SMEM_BYTES);
        cudaFuncSetAttribute(glq_trellis_decompress_kernel<2>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, TR_SMEM_BYTES);
        cudaFuncSetAttribute(glq_trellis_decompress_kernel<3>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, TR_SMEM_BYTES);
        cudaFuncSetAttribute(glq_trellis_decompress_kernel<4>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, TR_SMEM_BYTES);
    });
}

// R (bits/weight) is recoverable from the packed shape: cols == ceil(256*R/16) == 16*R.
int tr_bits_from_packed(const torch::Tensor &packed) {
    TORCH_CHECK(packed.dim() == 2, "trellis_packed must be 2-D [(m/16)*(k/16), 16*R]");
    int R = (int)packed.size(1) / 16;
    TORCH_CHECK(R >= 2 && R <= 4, "trellis kernel supports R (bits/weight) 2-4, got ", R);
    return R;
}

void tr_check_shape(int64_t m, int64_t k, const torch::Tensor &packed, int R) {
    TORCH_CHECK(m % (2 * TR_MMA_M) == 0, "trellis kernel needs m % 32 == 0, got ", m);
    TORCH_CHECK(k % (4 * TR_MMA_K) == 0, "trellis kernel needs k % 64 == 0, got ", k);
    TORCH_CHECK(packed.size(0) == (m / TR_MMA_M) * (k / TR_MMA_K),
                "trellis_packed rows ", packed.size(0), " != (m/16)*(k/16)");
    TORCH_CHECK(packed.scalar_type() == torch::kInt16, "trellis_packed must be int16");
    (void)R;
}

}  // namespace

/* Decode the whole weight: packed → (m, k) fp16. Bit-exact vs glq/trellis.py decode_layer. */
torch::Tensor glq_decompress_trellis_cuda(torch::Tensor trellis_packed, torch::Tensor tlut,
                                          int64_t m, int64_t k) {
    CHECK_INPUT(trellis_packed);
    CHECK_INPUT(tlut);
    TORCH_CHECK(tlut.scalar_type() == torch::kFloat16, "tlut must be fp16");
    TORCH_CHECK(tlut.numel() == (1 << TR_S) * 2, "tlut must be (512, 2) fp16");
    int R = tr_bits_from_packed(trellis_packed);
    tr_check_shape(m, k, trellis_packed, R);
    at::DeviceGuard guard(trellis_packed.device());
    tr_init_once();

    auto W = torch::empty({m, k}, torch::dtype(torch::kFloat16).device(trellis_packed.device()));
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    const uint32_t *cp = (const uint32_t *)trellis_packed.data_ptr<int16_t>();
    const half2 *cb = (const half2 *)tlut.data_ptr<c10::Half>();
    half *wp = (half *)W.data_ptr<c10::Half>();

#define TR_LAUNCH_DECOMP(RBITS)                                                        \
    glq_trellis_decompress_kernel<RBITS><<<TR_BLOCK_COUNT, TR_BLOCK_SIZE, TR_SMEM_BYTES, stream>>>( \
        wp, cp, cb, (uint32_t)m, (uint32_t)k)
    if (R == 2)      { TR_LAUNCH_DECOMP(2); }
    else if (R == 3) { TR_LAUNCH_DECOMP(3); }
    else             { TR_LAUNCH_DECOMP(4); }
#undef TR_LAUNCH_DECOMP
    return W;
}

/* Fused B=1 GEMV: out (m,) fp32 = W(m,k) @ x(k,), weights never materialized. */
torch::Tensor glq_decode_matvec_trellis_cuda(torch::Tensor x, torch::Tensor trellis_packed,
                                             torch::Tensor tlut, int64_t m, int64_t k) {
    CHECK_INPUT(x);
    CHECK_INPUT(trellis_packed);
    CHECK_INPUT(tlut);
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be fp16");
    TORCH_CHECK(x.numel() == k, "x must have k elements, got ", x.numel());
    TORCH_CHECK(tlut.scalar_type() == torch::kFloat16, "tlut must be fp16");
    int R = tr_bits_from_packed(trellis_packed);
    tr_check_shape(m, k, trellis_packed, R);
    at::DeviceGuard guard(x.device());
    tr_init_once();

    auto out = torch::empty({m}, torch::dtype(torch::kFloat32).device(x.device()));
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    const uint32_t *cp = (const uint32_t *)trellis_packed.data_ptr<int16_t>();
    const half2 *xp = (const half2 *)x.data_ptr<c10::Half>();
    const half2 *cb = (const half2 *)tlut.data_ptr<c10::Half>();
    float *op = out.data_ptr<float>();

#define TR_LAUNCH_MATVEC(RBITS)                                                        \
    glq_trellis_matvec_kernel<RBITS><<<TR_BLOCK_COUNT, TR_BLOCK_SIZE, TR_SMEM_BYTES, stream>>>( \
        op, cp, xp, cb, (uint32_t)m, (uint32_t)k)
    if (R == 2)      { TR_LAUNCH_MATVEC(2); }
    else if (R == 3) { TR_LAUNCH_MATVEC(3); }
    else             { TR_LAUNCH_MATVEC(4); }
#undef TR_LAUNCH_MATVEC
    return out;
}

/* Whether the kernel can serve this shape (host-side gate for the Python fallback). */
bool glq_trellis_kernel_supported(int64_t m, int64_t k) {
    return (m % (2 * TR_MMA_M) == 0) && (k % (4 * TR_MMA_K) == 0);
}

/* ── The shippable op: ONE host call per linear ──
 * Step 1 input RHT (block-diag, in-kernel) → Step 2 trellis decode+matmul → ×Wscale →
 * Step 3 output RHT. Mirrors glq_fused_linear_e8p_cuda exactly, so it traces as a single
 * node for cudagraph capture. B==1 keeps the weights compressed end-to-end (the VRAM +
 * decode win); B>1 (prefill) decompresses once and runs a dense GEMM. */
torch::Tensor glq_fused_linear_trellis_cuda(
    torch::Tensor x,               // (B, in_features) fp16, contiguous
    torch::Tensor sv,              // (n_pad,) fp16
    torch::Tensor su,              // (m_pad,) fp16
    torch::Tensor trellis_packed,  // ((m_pad/16)*(n_pad/16), 16*R) int16
    torch::Tensor tlut,            // (512, 2) fp16
    torch::Tensor blocks_n,        // (num_n_blocks,) int64 CPU
    torch::Tensor blocks_m,        // (num_m_blocks,) int64 CPU
    torch::Tensor blocks_n_meta,   // (num_n_blocks, 4) int32 GPU (or empty)
    torch::Tensor blocks_m_meta,   // (num_m_blocks, 4) int32 GPU (or empty)
    double wscale,
    int64_t in_features, int64_t out_features,
    int64_t n_pad, int64_t m_pad
) {
    CHECK_INPUT(x);
    CHECK_INPUT(trellis_packed);
    CHECK_INPUT(tlut);
    int B = x.size(0);
    at::DeviceGuard guard(x.device());

    // ---- Step 1: input RHT → x_rht (B, n_pad) fp32 ----
    auto x_rht = torch::empty({B, (long)n_pad},
                              torch::dtype(torch::kFloat32).device(x.device()));
    glq_input_rht_blockdiag_cuda(x.contiguous(), sv, x_rht,
                                 (int)in_features, (int)n_pad, blocks_n, blocks_n_meta);

    // ---- Step 2: decode + matmul in the RHT domain → y_rht (B, m_pad) fp32 ----
    torch::Tensor y_rht;
    if (B == 1) {
        auto xh = x_rht.view({(long)n_pad}).to(torch::kFloat16);
        auto yv = glq_decode_matvec_trellis_cuda(xh, trellis_packed, tlut, m_pad, n_pad);
        y_rht = (yv * (float)wscale).view({1, (long)m_pad}).contiguous();
    } else {
        // Prefill: decompress once, then one dense fp32 GEMM (matches the S0 reference and
        // the e8p dense fallback). The compressed weight is what lives in VRAM; W here is a
        // transient per-layer buffer from the caching allocator.
        auto W = glq_decompress_trellis_cuda(trellis_packed, tlut, m_pad, n_pad)
                     .to(torch::kFloat32);
        y_rht = (at::matmul(x_rht, W.t()) * (float)wscale).contiguous();
    }

    // ---- Step 3: output RHT → y (B, out_features) fp16 ----
    auto y = torch::empty({B, (long)out_features},
                          torch::dtype(torch::kFloat16).device(x.device()));
    glq_output_rht_blockdiag_cuda(y_rht, su, y, (int)out_features, (int)m_pad,
                                  blocks_m, blocks_m_meta);
    return y;
}
