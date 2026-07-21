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
 * Fixed HYB codebook params:
 *   L=16 (shift-register width) · S=9 (tlut index bits) · V=1 (log2 of the VQ dim, i.e. 2
 *   weights per trellis step). R = bits/weight = K in the Python API (2, 3 or 4).
 *
 * The kernels additionally template on IS_3INST for the **3INST** variant (QTIP's lookup-free
 * codebook, Python V=1): each 16-bit state decodes by ARITHMETIC (`tr_decode_3inst_half`, a
 * uint32 hash + fp16 two-half sum) instead of the smem tlut gather, and states sit at a K-bit
 * (not 2K-bit) stride — 8 states per A-fragment instead of 4. No tlut → **zero dynamic smem**,
 * which removes the L2/smem codebook-gather bottleneck (~35% of matvec stalls under ncu) and
 * lifts occupancy. The packed storage layout is IDENTICAL to HYB (256·K bits per 16×16 tile),
 * so the tile walk, mma, reduce and scatter are shared verbatim; only `tr_load_reg_cs` /
 * `tr_decode_regw` fork. Bit-flow proven bit-exact against `decode_layer` by the CPU mirrors
 * in tests/test_trellis_3inst_kernel.py BEFORE this port (see that file's module docstring).
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
constexpr uint32_t TR_BLOCK_COUNT = 128;   // upstream default; overridden by tr_grid_x()

/* Blocks along the m axis. Upstream QTIP hardcodes 128 (a ~108-SM A100); GLQ runs on cards
 * from 24 GB consumer parts to a 188-SM RTX PRO 6000, where a fixed 128 leaves 60 SMs idle at
 * decode batch sizes. Query once and cache (the kernels read gridDim.x, so nothing else has to
 * change). Mirrors the `static int num_sms` caching in glq_e8p.cu / glq_cuda.cu. */
uint32_t tr_grid_x() {
    static uint32_t g = 0;
    if (g == 0) {
        int dev = 0, sms = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        g = (sms > 0) ? (uint32_t)sms : TR_BLOCK_COUNT;
    }
    return g;
}
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

/* 3INST lookup-free state decode (glq/trellis.py:decode_3inst, bit-exact):
 *   h = s*89226354 + 64248484 (uint32 WRAP) → r = (h & 0x8FFF8FFF) ^ 0x3B603B60 → the two
 *   16-bit halves BIT-CAST to fp16 and summed IN fp16. The bit-cast (__ushort_as_half, not a
 *   convert) and the fp16 add (__hadd, not fp32) are both load-bearing for torch.equal vs the
 *   Python oracle; every output is an exactly-representable fp16. */
__device__ inline half tr_decode_3inst_half(uint32_t s) {
    const uint32_t h = s * 89226354u + 64248484u;
    const uint32_t r = (h & 0x8FFF8FFFu) ^ 0x3B603B60u;
    return __hadd(__ushort_as_half((unsigned short)(r >> 16)),
                  __ushort_as_half((unsigned short)(r & 0xFFFFu)));
}

/* Warp-shuffle bit-unpack: lane l holds its 16-bit chunks and pulls lane l+1's (tail-biting
 * wraps at lane 31 → lane 0), reconstructing the OVERLAPPING 32-bit windows from which four
 * successive L=16 trellis states are extracted at 4-bit strides. QTIP-verbatim.
 *
 * IS_3INST (V=1) departs from HYB only in WHAT is kept per window: instead of HYB's
 * byte-perm'd 32-bit view, keep the RAW 8R-bit chunk of the tail-biting stream (R=2: this
 * lane's u16 · R=3: reg_24_i · R=4: r_i) in reg_cs, and the 16-bit CONTINUATION (the top 16
 * bits of the NEXT lane's chunk — the stream bits immediately below this chunk's bit 0) in
 * reg_cs2. That uniform {chunk, continuation} form sidesteps the per-R overflow special-casing
 * entirely; the CPU mirror (tests/test_trellis_3inst_kernel.py::_load_chunks) is bit-exact. */
template <uint32_t R, bool IS_3INST = false>
__device__ inline void tr_load_reg_cs(const uint16_t *__restrict__ compressed, int weight_idx,
                                      uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
    if constexpr (IS_3INST && R == 2) {
        ditto2 reg_load; reg_load.u32x2 = tr_ld_cs((const uint2 *)&compressed[weight_idx]);
        reg_cs_next.x = reg_load.u32x2.x & 0xFFFFu;      // chunk = one u16 of stream (width 16)
        reg_cs_next.y = reg_load.u32x2.x >> 16;
        reg_cs_next.z = reg_load.u32x2.y & 0xFFFFu;
        reg_cs_next.w = reg_load.u32x2.y >> 16;
        // width 16 → continuation is the next lane's ENTIRE chunk
        reg_cs2_next.x = __shfl_sync(TR_FULL_MASK, reg_cs_next.x, laneId + 1);
        reg_cs2_next.y = __shfl_sync(TR_FULL_MASK, reg_cs_next.y, laneId + 1);
        reg_cs2_next.z = __shfl_sync(TR_FULL_MASK, reg_cs_next.z, laneId + 1);
        reg_cs2_next.w = __shfl_sync(TR_FULL_MASK, reg_cs_next.w, laneId + 1);
    } else if constexpr (IS_3INST && R == 3) {
        uint3 reg_load = tr_ld_cs((const uint3 *)&compressed[weight_idx]);
        uint32_t r1 = reg_load.x, r2 = reg_load.y, r3 = reg_load.z;
        reg_cs_next.x = r1 & 0xffffff;                            // reg_24_i (width 24)
        reg_cs_next.y = ((r1 >> 24) | (r2 << 8)) & 0xffffff;
        reg_cs_next.z = ((r2 >> 16) | (r3 << 16)) & 0xffffff;
        reg_cs_next.w = (r3 >> 8) & 0xffffff;
        reg_cs2_next.x = (__shfl_sync(TR_FULL_MASK, reg_cs_next.x, laneId + 1) >> 8) & 0xFFFFu;
        reg_cs2_next.y = (__shfl_sync(TR_FULL_MASK, reg_cs_next.y, laneId + 1) >> 8) & 0xFFFFu;
        reg_cs2_next.z = (__shfl_sync(TR_FULL_MASK, reg_cs_next.z, laneId + 1) >> 8) & 0xFFFFu;
        reg_cs2_next.w = (__shfl_sync(TR_FULL_MASK, reg_cs_next.w, laneId + 1) >> 8) & 0xFFFFu;
    } else if constexpr (IS_3INST && R == 4) {
        uint4 reg_load = tr_ld_cs((const uint4 *)&compressed[weight_idx]);
        reg_cs_next = reg_load;                                   // chunk = r_i (width 32)
        reg_cs2_next.x = __shfl_sync(TR_FULL_MASK, reg_load.x, laneId + 1) >> 16;
        reg_cs2_next.y = __shfl_sync(TR_FULL_MASK, reg_load.y, laneId + 1) >> 16;
        reg_cs2_next.z = __shfl_sync(TR_FULL_MASK, reg_load.z, laneId + 1) >> 16;
        reg_cs2_next.w = __shfl_sync(TR_FULL_MASK, reg_load.w, laneId + 1) >> 16;
    } else if constexpr (R == 2) {
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
 * HYB — mirrors glq/trellis.py `quantlut_sym`: state → idx*(idx+1) → tlut[(idx>>6) & 0x1ff] →
 * flip the sign of component 0 when bit 15 is set. The (laneId<<1) in `masked_idx` selects
 * this lane's private replica of the tlut entry (the ×32 replication in smem).
 * IS_3INST — V=1: EIGHT states at a K-bit stride from the extended window
 * Ext = chunk‖continuation (reg_c‖reg_c2): state_j = (Ext >> (8R − R·j)) & 0xFFFF, each state
 * decodes arithmetically (`tr_decode_3inst_half`, no smem), consecutive states (s_2j, s_2j+1)
 * pair into f16x2[j] — the same two adjacent columns HYB's V=2 half2 fills. No tlut, no
 * sign-flip, no laneId replica. Bit-exact per the CPU mirror (`_v1_states`). */
template <uint32_t R, bool IS_3INST = false>
__device__ inline void tr_decode_regw(uint32_t reg_c, uint32_t reg_c2, uint32_t laneId,
                                      const half2 *__restrict__ smem_codebook, ditto4 &reg_w) {
    if constexpr (IS_3INST) {
        constexpr uint32_t WIDTH = 8 * R;                        // chunk bit-width
        const uint64_t ext = ((uint64_t)reg_c << 16) | (uint64_t)reg_c2;
#pragma unroll
        for (uint32_t j = 0; j < 4; j += 1) {
            const uint32_t s0 = (uint32_t)(ext >> (WIDTH - R * (2 * j)))     & 0xFFFFu;
            const uint32_t s1 = (uint32_t)(ext >> (WIDTH - R * (2 * j + 1))) & 0xFFFFu;
            reg_w.f16x2[j] = __halves2half2(tr_decode_3inst_half(s0), tr_decode_3inst_half(s1));
        }
        return;
    }
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
template <uint32_t R, bool IS_3INST = false>
__global__ static void __launch_bounds__(TR_BLOCK_SIZE, 1)
glq_trellis_matvec_kernel(float *__restrict__ out,
                          const uint32_t *__restrict__ compressed,
                          const half2 *__restrict__ x,
                          const half2 *__restrict__ codebook,
                          uint32_t m, uint32_t k) {
    extern __shared__ __align__(16) half2 smem_codebook[];   // unused (0 bytes) when IS_3INST

    const uint32_t laneId = threadIdx.x % TR_WARP_SIZE;
    const uint32_t warpId = threadIdx.x / TR_WARP_SIZE;

    const uint32_t tileCountM = m / TR_MMA_M;
    const uint32_t tileCountK = k / TR_MMA_K;
    // Partition the m-tile-pairs across however many blocks the host launched. Upstream
    // hardcodes BLOCK_COUNT=128 (tuned for a ~108-SM A100); read gridDim.x instead so the
    // kernel self-adapts — on a 188-SM card a fixed 128 leaves a third of the GPU idle.
    const uint32_t m_per_block = (tileCountM + (2 * gridDim.x) - 1) / (2 * gridDim.x);
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
    // 3INST decodes arithmetically — no tlut, no smem fill, no barrier.
    if constexpr (!IS_3INST) {
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
        // Idle warps (this_warp_k == 0 — small k relative to 32 warps × 4 tiles) must NOT run
        // this speculative preload: their weight_idx points past the packed tensor (upstream
        // QTIP loads unconditionally — a latent OOB read that MMU-faults (Xid 31) when the
        // caching allocator maps nothing after the tensor; surfaced on Blackwell/sm_120 as an
        // allocation-layout-dependent cudaErrorIllegalAddress). The predicate is warp-uniform,
        // so the __shfl_syncs inside stay converged; the skipped value was never consumed.
        if (this_warp_k > 0)
            tr_load_reg_cs<R, IS_3INST>((const uint16_t *)compressed, weight_idx, laneId, reg_cs_next, reg_cs2_next);
        uint4 reg_cs, reg_cs2;
        float4 reg_p[2] = {};

        uint32_t x_idx      = warpId * f16x2_per_x_tile * 4 + laneId;
        uint32_t x_idx_step = TR_WARPS * f16x2_per_x_tile * 4;

#pragma unroll 4
        for (uint32_t ki = 0; ki < this_warp_k; ki += 1) {
            if (ki + 1 != this_warp_k && ki % 2 == 1) weight_idx += weight_step * 2;
            reg_cs = reg_cs_next; reg_cs2 = reg_cs2_next;
            tr_load_reg_cs<R, IS_3INST>((const uint16_t *)compressed,
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
                    tr_decode_regw<R, IS_3INST>(reg_c, reg_c2, laneId, smem_codebook, reg_w);

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

/* ── Batched GEMM (B>1): out (B,m) fp32 = x (B,k) @ W(m,k).T, weights stay COMPRESSED ──
 *
 * The GEMV above fills only column 0 of the m16n8k16 B-fragment and discards the other 7
 * columns the tensor core already computed. Here each N-column carries a DIFFERENT token, so
 * up to 8 tokens ride along for free: identical weight-decode work, identical mma count as
 * B=1. (Same trick as glq_decode_matmul_e8p, glq_e8p.cu:247-255.)
 *
 * Three changes vs the GEMV, everything else — the weight walk, the bit-unpack, the decode —
 * is byte-identical, which is what makes row-parity with the GEMV bit-exact:
 *   1. B-fragment straight from global (no x_buf staging): lane l → g = l>>2 picks the TOKEN,
 *      t = l&3 picks the k-pair. Every lane wants a different token, so smem staging buys
 *      nothing, and dropping it frees 4 KB (we're already at 64 KB dynamic for the LUT).
 *   2. Ragged tail by PREDICATION (active = tok < B), not padding — we're on raw mma.sync.
 *   3. C-harvest: g/t swap roles between the B and C fragments (in C, g is the row and t the
 *      column pair), so all four accumulators are live. We loop the 8 columns through the
 *      existing 4 KB reduce_gather rather than widening it 8x to 32 KB (64+32 KB would crush
 *      occupancy); the reduce runs once per m-tile-pair, so 8 passes are free vs the k-loop.
 *
 * Grid (tr_grid_x()=#SMs, ceil(B/8)): each (blockIdx.x → m-range, blockIdx.y → token-tile) owns
 * disjoint output, so the in-block fixed-order reduce stays deterministic with NO global
 * scratch and NO allocation → capture-safe by construction. (e8p's split-K scratch uses a
 * per-call raw_alloc/raw_delete, which glq_cuda.cu:3405-3412 itself flags as illegal during
 * capture — we deliberately do not copy that.) */
template <uint32_t R, bool IS_3INST = false>
__global__ static void __launch_bounds__(TR_BLOCK_SIZE, 1)
glq_trellis_matmul_kernel(float *__restrict__ out,
                          const uint32_t *__restrict__ compressed,
                          const half2 *__restrict__ x,
                          const half2 *__restrict__ codebook,
                          uint32_t m, uint32_t k, uint32_t B) {
    extern __shared__ __align__(16) half2 smem_codebook[];

    const uint32_t laneId = threadIdx.x % TR_WARP_SIZE;
    const uint32_t warpId = threadIdx.x / TR_WARP_SIZE;
    const uint32_t g = laneId >> 2;     // B-frag: token within the 8-token tile · C-frag: row
    const uint32_t t = laneId & 3;      // B-frag: k-pair               · C-frag: column pair

    const uint32_t tileCountM = m / TR_MMA_M;
    const uint32_t tileCountK = k / TR_MMA_K;
    // Partition the m-tile-pairs across however many blocks the host launched. Upstream
    // hardcodes BLOCK_COUNT=128 (tuned for a ~108-SM A100); read gridDim.x instead so the
    // kernel self-adapts — on a 188-SM card a fixed 128 leaves a third of the GPU idle.
    const uint32_t m_per_block = (tileCountM + (2 * gridDim.x) - 1) / (2 * gridDim.x);
    const uint32_t k_per_block = tileCountK / (TR_WARPS * 4) * 2;
    const uint32_t this_warp_k =
        (warpId < (tileCountK % (TR_WARPS * 4)) / 4) ? k_per_block + 2 : k_per_block;

    const uint32_t u16_per_tile       = TR_MMA_M * TR_MMA_K * R / 16;
    const uint32_t u16_per_tile_block = u16_per_tile * 4;
    const uint32_t weight_step        = TR_WARPS * u16_per_tile_block;
    const uint32_t weight_row_step    = tileCountK * u16_per_tile * 2;
    const uint32_t x_row_half2        = k / 2;                 // half2 per token row

    const uint32_t tok      = blockIdx.y * 8 + g;              // this lane's token
    const bool     active   = (tok < B);
    const uint32_t *x_u32   = reinterpret_cast<const uint32_t *>(x) + (size_t)tok * x_row_half2;

    uint32_t tileIdM = m_per_block * blockIdx.x;

    if constexpr (!IS_3INST) {   // 3INST: arithmetic decode — no tlut smem fill, no barrier
        uint32_t my_cb_idx = threadIdx.x & 0x1ff;
        half2 my_cb = codebook[my_cb_idx];
        for (uint32_t i = 0; i < 32; i += 2)
            smem_codebook[(my_cb_idx << 5) | (i ^ (threadIdx.x & 0x1f) ^ (threadIdx.x >> 9))] = my_cb;
        __syncthreads();
    }

    __shared__ float reduce_gather[TR_WARPS][2][16];

    for (uint32_t mi = 0; mi < m_per_block; mi += 1) {
        if (tileIdM * 2 >= tileCountM) return;   // block-uniform → no __syncthreads deadlock

        int weight_idx = tileIdM * weight_row_step + warpId * u16_per_tile_block * 2
                       + laneId * (u16_per_tile_block / TR_WARP_SIZE);
        uint4 reg_cs_next = {}, reg_cs2_next = {};
        // Idle warps (this_warp_k == 0 — small k relative to 32 warps × 4 tiles) must NOT run
        // this speculative preload: their weight_idx points past the packed tensor (upstream
        // QTIP loads unconditionally — a latent OOB read that MMU-faults (Xid 31) when the
        // caching allocator maps nothing after the tensor; surfaced on Blackwell/sm_120 as an
        // allocation-layout-dependent cudaErrorIllegalAddress). The predicate is warp-uniform,
        // so the __shfl_syncs inside stay converged; the skipped value was never consumed.
        if (this_warp_k > 0)
            tr_load_reg_cs<R, IS_3INST>((const uint16_t *)compressed, weight_idx, laneId, reg_cs_next, reg_cs2_next);
        uint4 reg_cs, reg_cs2;
        float4 reg_p[2] = {};

#pragma unroll 4
        for (uint32_t ki = 0; ki < this_warp_k; ki += 1) {
            if (ki + 1 != this_warp_k && ki % 2 == 1) weight_idx += weight_step * 2;
            reg_cs = reg_cs_next; reg_cs2 = reg_cs2_next;
            tr_load_reg_cs<R, IS_3INST>((const uint16_t *)compressed,
                                        weight_idx + (1 - ki % 2) * u16_per_tile_block,
                                        laneId, reg_cs_next, reg_cs2_next);

#pragma unroll 2
            for (uint32_t subki = 0; subki < 2; subki += 1) {
                // Absolute k-tile this (warp, ki, subki) covers — the same mapping the
                // decompress kernel uses, and algebraically identical to the GEMV's x_buf
                // indexing at B=1. L1::evict_last: x is re-read for every m-tile.
                const uint32_t k_tile = 4 * warpId + 2 * (ki % 2) + subki + (4 * TR_WARPS) * (ki / 2);
                const uint32_t xo = k_tile * 8 + t;
                ditto2 reg_a; reg_a.u32[0] = 0u; reg_a.u32[1] = 0u;
                if (active) {
                    reg_a.u32[0] = tr_ld_x(x_u32 + xo);
                    reg_a.u32[1] = tr_ld_x(x_u32 + xo + 4);
                }

#pragma unroll 2
                for (uint32_t submi = 0; submi < 2; submi += 1) {
                    uint32_t reg_c, reg_c2;
                    if      (submi == 0 && subki == 0) { reg_c = reg_cs.x; reg_c2 = reg_cs2.x; }
                    else if (submi == 1 && subki == 0) { reg_c = reg_cs.y; reg_c2 = reg_cs2.y; }
                    else if (submi == 0 && subki == 1) { reg_c = reg_cs.z; reg_c2 = reg_cs2.z; }
                    else                               { reg_c = reg_cs.w; reg_c2 = reg_cs2.w; }

                    ditto4 reg_w;
                    tr_decode_regw<R, IS_3INST>(reg_c, reg_c2, laneId, smem_codebook, reg_w);

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
        }

        // C-fragment: lane l holds c0→(row g, col 2t)  c1→(row g, col 2t+1)
        //                          c2→(row g+8, col 2t) c3→(row g+8, col 2t+1)
        // One column (= one token) per pass, reusing the 4 KB reduce_gather.
        for (uint32_t c = 0; c < 8; c += 1) {
            if (t == (c >> 1)) {
                const bool even = ((c & 1) == 0);
                for (int pi = 0; pi < 2; pi++) {
                    reduce_gather[warpId][pi][g]     = even ? reg_p[pi].x : reg_p[pi].y;
                    reduce_gather[warpId][pi][g + 8] = even ? reg_p[pi].z : reg_p[pi].w;
                }
            }
            __syncthreads();
            if (warpId < 1) {
                const uint32_t out_tok = blockIdx.y * 8 + c;
                if (out_tok < B) {
                    int pi = laneId / 16;
                    float reduced = 0.0f;
                    for (uint32_t warpi = 0; warpi < TR_WARPS; warpi++)
                        reduced += reduce_gather[warpi][pi][laneId % 16];
                    out[(size_t)out_tok * m + (tileIdM * 2) * TR_MMA_M + laneId] = reduced;
                }
            }
            __syncthreads();
        }
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
template <uint32_t R, bool IS_3INST = false>
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
    // Partition the m-tile-pairs across however many blocks the host launched. Upstream
    // hardcodes BLOCK_COUNT=128 (tuned for a ~108-SM A100); read gridDim.x instead so the
    // kernel self-adapts — on a 188-SM card a fixed 128 leaves a third of the GPU idle.
    const uint32_t m_per_block = (tileCountM + (2 * gridDim.x) - 1) / (2 * gridDim.x);
    const uint32_t k_per_block = tileCountK / (TR_WARPS * 4) * 2;
    const uint32_t this_warp_k =
        (warpId < (tileCountK % (TR_WARPS * 4)) / 4) ? k_per_block + 2 : k_per_block;

    const uint32_t u16_per_tile       = TR_MMA_M * TR_MMA_K * R / 16;
    const uint32_t u16_per_tile_block = u16_per_tile * 4;
    const uint32_t weight_step        = TR_WARPS * u16_per_tile_block;
    const uint32_t weight_row_step    = tileCountK * u16_per_tile * 2;

    uint32_t tileIdM = m_per_block * blockIdx.x;

    if constexpr (!IS_3INST) {   // 3INST: arithmetic decode — no tlut smem fill, no barrier
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
        // Idle warps (this_warp_k == 0 — small k relative to 32 warps × 4 tiles) must NOT run
        // this speculative preload: their weight_idx points past the packed tensor (upstream
        // QTIP loads unconditionally — a latent OOB read that MMU-faults (Xid 31) when the
        // caching allocator maps nothing after the tensor; surfaced on Blackwell/sm_120 as an
        // allocation-layout-dependent cudaErrorIllegalAddress). The predicate is warp-uniform,
        // so the __shfl_syncs inside stay converged; the skipped value was never consumed.
        if (this_warp_k > 0)
            tr_load_reg_cs<R, IS_3INST>((const uint16_t *)compressed, weight_idx, laneId, reg_cs_next, reg_cs2_next);
        uint4 reg_cs, reg_cs2;

        for (uint32_t ki = 0; ki < this_warp_k; ki += 1) {
            if (ki + 1 != this_warp_k && ki % 2 == 1) weight_idx += weight_step * 2;
            reg_cs = reg_cs_next; reg_cs2 = reg_cs2_next;
            tr_load_reg_cs<R, IS_3INST>((const uint16_t *)compressed,
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
                    tr_decode_regw<R, IS_3INST>(reg_c, reg_c2, laneId, smem_codebook, reg_w);

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
        cudaFuncSetAttribute(glq_trellis_matmul_kernel<2>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, TR_SMEM_BYTES);
        cudaFuncSetAttribute(glq_trellis_matmul_kernel<3>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, TR_SMEM_BYTES);
        cudaFuncSetAttribute(glq_trellis_matmul_kernel<4>,
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
    glq_trellis_decompress_kernel<RBITS><<<tr_grid_x(), TR_BLOCK_SIZE, TR_SMEM_BYTES, stream>>>( \
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
    glq_trellis_matvec_kernel<RBITS><<<tr_grid_x(), TR_BLOCK_SIZE, TR_SMEM_BYTES, stream>>>( \
        op, cp, xp, cb, (uint32_t)m, (uint32_t)k)
    if (R == 2)      { TR_LAUNCH_MATVEC(2); }
    else if (R == 3) { TR_LAUNCH_MATVEC(3); }
    else             { TR_LAUNCH_MATVEC(4); }
#undef TR_LAUNCH_MATVEC
    return out;
}

/* Batched GEMM: out (B, m) fp32 = x (B, k) @ W(m,k).T, weights never materialized. */
torch::Tensor glq_decode_matmul_trellis_cuda(torch::Tensor x, torch::Tensor trellis_packed,
                                             torch::Tensor tlut, int64_t m, int64_t k) {
    CHECK_INPUT(x);
    CHECK_INPUT(trellis_packed);
    CHECK_INPUT(tlut);
    TORCH_CHECK(x.dim() == 2, "x must be (B, k)");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be fp16");
    TORCH_CHECK(x.size(1) == k, "x must have k columns, got ", x.size(1));
    TORCH_CHECK(tlut.scalar_type() == torch::kFloat16, "tlut must be fp16");
    TORCH_CHECK(tlut.numel() == (1 << TR_S) * 2, "tlut must be (512, 2) fp16");
    int R = tr_bits_from_packed(trellis_packed);
    tr_check_shape(m, k, trellis_packed, R);
    at::DeviceGuard guard(x.device());
    tr_init_once();

    const int64_t B = x.size(0);
    auto out = torch::empty({B, m}, torch::dtype(torch::kFloat32).device(x.device()));
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    const uint32_t *cp = (const uint32_t *)trellis_packed.data_ptr<int16_t>();
    const half2 *xp = (const half2 *)x.data_ptr<c10::Half>();
    const half2 *cb = (const half2 *)tlut.data_ptr<c10::Half>();
    float *op = out.data_ptr<float>();
    dim3 grid(tr_grid_x(), (unsigned)((B + 7) / 8));   // 8 tokens per mma N-tile

#define TR_LAUNCH_MATMUL(RBITS)                                                        \
    glq_trellis_matmul_kernel<RBITS><<<grid, TR_BLOCK_SIZE, TR_SMEM_BYTES, stream>>>(  \
        op, cp, xp, cb, (uint32_t)m, (uint32_t)k, (uint32_t)B)
    if (R == 2)      { TR_LAUNCH_MATMUL(2); }
    else if (R == 3) { TR_LAUNCH_MATMUL(3); }
    else             { TR_LAUNCH_MATMUL(4); }
#undef TR_LAUNCH_MATMUL
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
    //
    // Hybrid dispatch. The compressed kernels re-read/re-decode the whole weight per 8-token
    // tile (traffic ∝ ceil(B/8)) — ideal for decode, wasteful for a multi-thousand-token
    // prefill against a single cuBLAS GEMM. So:
    //   B == 1        → GEMV                          compressed
    //   B ≤ BATCH_MAX → batched GEMM                  compressed  ← every captured decode batch
    //   B >  BATCH_MAX → decompress fp16 + one GEMM   ← prefill only (eager, not captured)
    // Net: no dense weight on ANY decode step, and TTFT stays on the cuBLAS path.
    // GLQ_TRELLIS_DENSE forces the dense path everywhere (bit-exact A/B reference, mirrors
    // GLQ_E8P_DENSE_B1). GLQ_TRELLIS_BATCH_MAX tunes the threshold.
    static const int64_t batch_max = [] {
        const char *e = std::getenv("GLQ_TRELLIS_BATCH_MAX");
        return e ? std::max<int64_t>(1, atoll(e)) : 64;
    }();
    static const bool force_dense = (std::getenv("GLQ_TRELLIS_DENSE") != nullptr);

    torch::Tensor y_rht;
    if (B == 1 && !force_dense) {
        auto xh = x_rht.view({(long)n_pad}).to(torch::kFloat16);
        auto yv = glq_decode_matvec_trellis_cuda(xh, trellis_packed, tlut, m_pad, n_pad);
        y_rht = (yv * (float)wscale).view({1, (long)m_pad}).contiguous();
    } else if (B <= batch_max && !force_dense) {
        auto xh = x_rht.to(torch::kFloat16);                             // (B, n_pad)
        auto yb = glq_decode_matmul_trellis_cuda(xh, trellis_packed, tlut, m_pad, n_pad);
        y_rht = (yb * (float)wscale).contiguous();                       // (B, m_pad) fp32
    } else {
        // Prefill: decompress ONCE to fp16 (not fp32 — halves the transient) and let cuBLAS
        // do the GEMM. W is a transient per-layer buffer from the caching allocator; the
        // weight that actually LIVES in VRAM is the compressed one. fp16 accumulation error
        // here (~1e-3 relative) is negligible against 2-bpw quantization noise (~28%).
        auto W = glq_decompress_trellis_cuda(trellis_packed, tlut, m_pad, n_pad);   // fp16
        auto xh = x_rht.to(torch::kFloat16);
        y_rht = (at::matmul(xh, W.t()).to(torch::kFloat32) * (float)wscale).contiguous();
    }

    // ---- Step 3: output RHT → y (B, out_features) fp16 ----
    auto y = torch::empty({B, (long)out_features},
                          torch::dtype(torch::kFloat16).device(x.device()));
    glq_output_rht_blockdiag_cuda(y_rht, su, y, (int)out_features, (int)m_pad,
                                  blocks_m, blocks_m_meta);
    return y;
}

/* ══ 3INST (lookup-free V=1) host entries — no tlut, ZERO dynamic smem ══
 * Same packed storage and grid geometry as HYB; the <R, true> instantiations decode
 * arithmetically, so the codebook pointer is null and the launches pass smem=0 (no
 * cudaFuncSetAttribute needed — 0 ≤ the default opt-in limit; skipping the 64 KB
 * carve-out is precisely the occupancy win). */

torch::Tensor glq_decompress_trellis_3inst_cuda(torch::Tensor trellis_packed,
                                                int64_t m, int64_t k) {
    CHECK_INPUT(trellis_packed);
    int R = tr_bits_from_packed(trellis_packed);
    tr_check_shape(m, k, trellis_packed, R);
    at::DeviceGuard guard(trellis_packed.device());

    auto W = torch::empty({m, k}, torch::dtype(torch::kFloat16).device(trellis_packed.device()));
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    const uint32_t *cp = (const uint32_t *)trellis_packed.data_ptr<int16_t>();
    half *wp = (half *)W.data_ptr<c10::Half>();

#define TR_LAUNCH_DECOMP3(RBITS)                                                       \
    glq_trellis_decompress_kernel<RBITS, true><<<tr_grid_x(), TR_BLOCK_SIZE, 0, stream>>>( \
        wp, cp, (const half2 *)nullptr, (uint32_t)m, (uint32_t)k)
    if (R == 2)      { TR_LAUNCH_DECOMP3(2); }
    else if (R == 3) { TR_LAUNCH_DECOMP3(3); }
    else             { TR_LAUNCH_DECOMP3(4); }
#undef TR_LAUNCH_DECOMP3
    return W;
}

torch::Tensor glq_decode_matvec_trellis_3inst_cuda(torch::Tensor x, torch::Tensor trellis_packed,
                                                   int64_t m, int64_t k) {
    CHECK_INPUT(x);
    CHECK_INPUT(trellis_packed);
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be fp16");
    TORCH_CHECK(x.numel() == k, "x must have k elements, got ", x.numel());
    int R = tr_bits_from_packed(trellis_packed);
    tr_check_shape(m, k, trellis_packed, R);
    at::DeviceGuard guard(x.device());

    auto out = torch::empty({m}, torch::dtype(torch::kFloat32).device(x.device()));
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    const uint32_t *cp = (const uint32_t *)trellis_packed.data_ptr<int16_t>();
    const half2 *xp = (const half2 *)x.data_ptr<c10::Half>();
    float *op = out.data_ptr<float>();

#define TR_LAUNCH_MATVEC3(RBITS)                                                       \
    glq_trellis_matvec_kernel<RBITS, true><<<tr_grid_x(), TR_BLOCK_SIZE, 0, stream>>>( \
        op, cp, xp, (const half2 *)nullptr, (uint32_t)m, (uint32_t)k)
    if (R == 2)      { TR_LAUNCH_MATVEC3(2); }
    else if (R == 3) { TR_LAUNCH_MATVEC3(3); }
    else             { TR_LAUNCH_MATVEC3(4); }
#undef TR_LAUNCH_MATVEC3
    return out;
}

torch::Tensor glq_decode_matmul_trellis_3inst_cuda(torch::Tensor x, torch::Tensor trellis_packed,
                                                   int64_t m, int64_t k) {
    CHECK_INPUT(x);
    CHECK_INPUT(trellis_packed);
    TORCH_CHECK(x.dim() == 2, "x must be (B, k)");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be fp16");
    TORCH_CHECK(x.size(1) == k, "x must have k columns, got ", x.size(1));
    int R = tr_bits_from_packed(trellis_packed);
    tr_check_shape(m, k, trellis_packed, R);
    at::DeviceGuard guard(x.device());

    const int64_t B = x.size(0);
    auto out = torch::empty({B, m}, torch::dtype(torch::kFloat32).device(x.device()));
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    const uint32_t *cp = (const uint32_t *)trellis_packed.data_ptr<int16_t>();
    const half2 *xp = (const half2 *)x.data_ptr<c10::Half>();
    float *op = out.data_ptr<float>();
    dim3 grid(tr_grid_x(), (unsigned)((B + 7) / 8));   // 8 tokens per mma N-tile

#define TR_LAUNCH_MATMUL3(RBITS)                                                       \
    glq_trellis_matmul_kernel<RBITS, true><<<grid, TR_BLOCK_SIZE, 0, stream>>>(        \
        op, cp, xp, (const half2 *)nullptr, (uint32_t)m, (uint32_t)k, (uint32_t)B)
    if (R == 2)      { TR_LAUNCH_MATMUL3(2); }
    else if (R == 3) { TR_LAUNCH_MATMUL3(3); }
    else             { TR_LAUNCH_MATMUL3(4); }
#undef TR_LAUNCH_MATMUL3
    return out;
}

/* Fused 3INST linear: identical bracket + hybrid B-dispatch as the HYB fused op, minus the
 * tlut. Same env knobs (GLQ_TRELLIS_BATCH_MAX / GLQ_TRELLIS_DENSE). One host call →
 * cudagraph-capturable as a single node. */
torch::Tensor glq_fused_linear_trellis_3inst_cuda(
    torch::Tensor x,               // (B, in_features) fp16, contiguous
    torch::Tensor sv,              // (n_pad,) fp16
    torch::Tensor su,              // (m_pad,) fp16
    torch::Tensor trellis_packed,  // ((m_pad/16)*(n_pad/16), 16*R) int16
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
    int B = x.size(0);
    at::DeviceGuard guard(x.device());

    // ---- Step 1: input RHT → x_rht (B, n_pad) fp32 ----
    auto x_rht = torch::empty({B, (long)n_pad},
                              torch::dtype(torch::kFloat32).device(x.device()));
    glq_input_rht_blockdiag_cuda(x.contiguous(), sv, x_rht,
                                 (int)in_features, (int)n_pad, blocks_n, blocks_n_meta);

    // ---- Step 2: decode + matmul in the RHT domain → y_rht (B, m_pad) fp32 ----
    static const int64_t batch_max = [] {
        const char *e = std::getenv("GLQ_TRELLIS_BATCH_MAX");
        return e ? std::max<int64_t>(1, atoll(e)) : 64;
    }();
    static const bool force_dense = (std::getenv("GLQ_TRELLIS_DENSE") != nullptr);

    torch::Tensor y_rht;
    if (B == 1 && !force_dense) {
        auto xh = x_rht.view({(long)n_pad}).to(torch::kFloat16);
        auto yv = glq_decode_matvec_trellis_3inst_cuda(xh, trellis_packed, m_pad, n_pad);
        y_rht = (yv * (float)wscale).view({1, (long)m_pad}).contiguous();
    } else if (B <= batch_max && !force_dense) {
        auto xh = x_rht.to(torch::kFloat16);                             // (B, n_pad)
        auto yb = glq_decode_matmul_trellis_3inst_cuda(xh, trellis_packed, m_pad, n_pad);
        y_rht = (yb * (float)wscale).contiguous();                       // (B, m_pad) fp32
    } else {
        auto W = glq_decompress_trellis_3inst_cuda(trellis_packed, m_pad, n_pad);   // fp16
        auto xh = x_rht.to(torch::kFloat16);
        y_rht = (at::matmul(xh, W.t()).to(torch::kFloat32) * (float)wscale).contiguous();
    }

    // ---- Step 3: output RHT → y (B, out_features) fp16 ----
    auto y = torch::empty({B, (long)out_features},
                          torch::dtype(torch::kFloat16).device(x.device()));
    glq_output_rht_blockdiag_cuda(y_rht, su, y, (int)out_features, (int)m_pad,
                                  blocks_m, blocks_m_meta);
    return y;
}
