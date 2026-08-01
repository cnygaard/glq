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

#include "glq_fht.cuh"

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
// R=1 (the bpw-5 RVQ residual stage) needs only ONE u32 per lane — 4 chunks x 8 bits. The
// warp's 32 loads are one fully-coalesced 128 B segment, so this is the cheapest of the four
// load paths (uint3 at R=3 is the most expensive: PTX has no .v3, hence three scalar loads).
__inline__ __device__ uint32_t tr_ld_cs(const uint32_t *p) {
    uint32_t out;
    asm("ld.global.cs.u32 %0, [%1];" : "=r"(out) : "l"(p));
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
    if constexpr (IS_3INST && R == 1) {
        // Four 8-bit chunks in ONE u32 (vs R=2's uint2 of u16s, R=4's uint4 of u32s); chunk i
        // = byte i, memory order, same rule as every other rate. Per-lane stride is 2 u16 and
        // every index term is even, so the u32 load is naturally aligned.
        const uint32_t r = tr_ld_cs((const uint32_t *)&compressed[weight_idx]);
        reg_cs_next.x =  r        & 0xFFu;
        reg_cs_next.y = (r >>  8) & 0xFFu;
        reg_cs_next.z = (r >> 16) & 0xFFu;
        reg_cs_next.w =  r >> 24;
        // "Continuation = the top 16 bits of the NEXT lane's chunk" is an identity only while
        // 8R >= 16. Here the chunk is 8 bits and cannot supply 16, so the continuation spans
        // the next TWO lanes (R>=2's `next >> (WIDTH-16)` would be a NEGATIVE shift). Mod-32
        // wrap keeps the tail-biting cycle; each fragment slot x/y/z/w is its own stream
        // across the lanes, which is why the shuffle stays within a slot. Two whole-u32
        // shuffles serve all four fragments — R>=2 needs four. Pinned bit-exact by
        // tests/test_trellis_3inst_kernel.py::test_mirror2b_v1_transliteration[1].
        const uint32_t n1 = __shfl_sync(TR_FULL_MASK, r, laneId + 1);
        const uint32_t n2 = __shfl_sync(TR_FULL_MASK, r, laneId + 2);
        reg_cs2_next.x = (( n1        & 0xFFu) << 8) | ( n2        & 0xFFu);
        reg_cs2_next.y = (((n1 >>  8) & 0xFFu) << 8) | ((n2 >>  8) & 0xFFu);
        reg_cs2_next.z = (((n1 >> 16) & 0xFFu) << 8) | ((n2 >> 16) & 0xFFu);
        reg_cs2_next.w = (( n1 >> 24)          << 8) | ( n2 >> 24);
    } else if constexpr (IS_3INST && R == 2) {
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
/* FUSE_IN (RS2b): compute the input RHT *inside* every block instead of a separate 1-block
 * kernel the whole GPU idles behind (Phase-0 ncu: input_rht = grid(1,1,1), SM 0.2%, on 188
 * SMs). Each block redundantly runs the same fp32 butterfly on x in its own smem — ~µs of
 * overlapped work replacing a serialized kernel + fp32 global round-trip + fp16-cast launch.
 * B=1 (matvec) only: a matmul block serves an 8-token tile and would redo 8 FHTs. When
 * FUSE_IN, `x` points at the RAW fp16 input (in_features), `sv`/`rsqrt_n` drive the
 * transform, and the smem layout is [fp32 ping | fp32 pong | half2 result] (2·k·4 + k·2 B).
 * The butterfly is the bit-exact transliteration pinned by test_rs2a_* (ascending distance,
 * lo ? a+b : b−a, single ×rsqrt_n then RN cast — identical fp32 op order to
 * glq_input_rht_kernel + torch .to(fp16)). */
template <uint32_t R, bool IS_3INST = false, bool FUSE_IN = false>
__global__ static void __launch_bounds__(TR_BLOCK_SIZE, 1)
glq_trellis_matvec_kernel(float *__restrict__ out,
                          const uint32_t *__restrict__ compressed,
                          const half2 *__restrict__ x,
                          const half2 *__restrict__ codebook,
                          uint32_t m, uint32_t k, float wscale,
                          const half *__restrict__ sv = nullptr,
                          float rsqrt_n = 0.0f, uint32_t in_features = 0,
                          const int4 *__restrict__ block_meta = nullptr,
                          uint32_t num_blocks = 1, bool accum = false) {
    extern __shared__ __align__(16) half2 smem_codebook[];   // unused (0 bytes) when IS_3INST

    const half2 *xsrc = x;                  // fragment source (global x, or smem under FUSE_IN)
    if constexpr (FUSE_IN) {
        static_assert(IS_3INST, "FUSE_IN is only instantiated for the 3inst (no-tlut) variant");
        // Unified single-/multi-block layout: [persist: k fp32 | scratch: max_bs fp32].
        // Each (sub-)block ping-pongs persist<->scratch through its butterfly; the fp16
        // result then reuses SCRATCH as half2 (k*2 B <= max_bs*4 B always, because the
        // leading pow2 of a binary decomposition is >= k/2). RS3 sub-blocks come from
        // block_meta (int4 {offset, bs, log_bs, 0}, same as the multiblock RHT kernel).
        const half *x_raw = reinterpret_cast<const half *>(x);
        float *persist = reinterpret_cast<float *>(smem_codebook);
        float *scratch = persist + k;                       // host sizes smem to k + max_bs
        half2 *xh = reinterpret_cast<half2 *>(scratch);
        for (uint32_t i = threadIdx.x; i < k; i += TR_BLOCK_SIZE) {
            float xv = (i < in_features) ? __half2float(x_raw[i]) : 0.0f;
            persist[i] = xv * __half2float(sv[i]);
        }
        __syncthreads();
        const uint32_t nb = (num_blocks == 0) ? 1u : num_blocks;
        for (uint32_t blk = 0; blk < nb; ++blk) {
            const uint32_t off = (nb == 1) ? 0u : (uint32_t)block_meta[blk].x;
            const uint32_t bs  = (nb == 1) ? k  : (uint32_t)block_meta[blk].y;
            // RS4a: distance 1..16 as warp shuffles in place (bs pow2 ≥ 32 —
            // trellis sub-blocks are ≥ 256); smem ping-pong resumes at h0.
            // The pointer-tracked copy-back below is already parity-agnostic.
            uint32_t h0 = 1;
            if (bs >= 32) {
                glq_fht_shuffle_low<8>(persist + off, (int)bs);
                __syncthreads();
                h0 = 32;
            }
            float *src = persist + off, *dst = scratch;
            for (uint32_t h = h0; h < bs; h <<= 1) {
                glq_fht_stage_smem(src, dst, (int)h, (int)bs);
                __syncthreads();
                float *t = src; src = dst; dst = t;
            }
            if (src != persist + off) {       // odd smem stage count: result in scratch
                for (uint32_t i = threadIdx.x; i < bs; i += TR_BLOCK_SIZE)
                    (persist + off)[i] = src[i];
                __syncthreads();
            }
        }
        // Normalize + RN-cast per (sub-)block. Single-block multiplies the HOST
        // 1.0f/sqrtf(k) (bit-matches glq_input_rht_kernel); sub-blocks multiply the
        // in-kernel rsqrtf(bs) (bit-matches glq_input_rht_multiblock_kernel). half2
        // pairs never straddle sub-blocks (bs >= 256, offsets even).
        for (uint32_t blk = 0; blk < nb; ++blk) {
            const uint32_t off = (nb == 1) ? 0u : (uint32_t)block_meta[blk].x;
            const uint32_t bs  = (nb == 1) ? k  : (uint32_t)block_meta[blk].y;
            const float r = (nb == 1) ? rsqrt_n : rsqrtf((float)bs);
            for (uint32_t i = threadIdx.x; i < bs / 2; i += TR_BLOCK_SIZE)
                xh[off / 2 + i] = __floats2half2_rn(persist[off + 2 * i] * r,
                                                    persist[off + 2 * i + 1] * r);
        }
        __syncthreads();
        xsrc = xh;
    }

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
                // FUSE_IN reads the in-block FHT result from SHARED memory — a plain load
                // (tr_ld_x is ld.global and would be illegal on an smem pointer).
                x_buf[warpId][laneId / 8][laneId % 4].u32[(laneId % 8) / 4] =
                    FUSE_IN ? reinterpret_cast<const uint32_t *>(xsrc)[x_idx]
                            : tr_ld_x(reinterpret_cast<const uint32_t *>(xsrc) + x_idx);
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
            // (Skipped under FUSE_IN — xsrc is shared memory, global prefetch doesn't apply.)
            if (!FUSE_IN && ki % 2 == 0 && (x_idx + x_idx_step * 4) < x_half2)
                asm("prefetch.global.L1 [%0];" ::"l"(xsrc + x_idx + x_idx_step * 4));
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
            // RS1: fold the ×wscale that used to be a separate elementwise kernel into the
            // store — bit-exact (same fp32 operands), removes one launch per linear.
            // RVQ (5-8 bpw): stage 2 ACCUMULATES onto stage 1's output. No atomics needed —
            // each block owns a disjoint m-range, so every element has exactly one writer per
            // pass, and the two passes are stream-ordered. Stage 1 (accum=false) IS the
            // initializer, so `out` is never read before it is written and needs no zeroing.
            // __fadd_rn (not `+`) forbids FFMA contraction, keeping the summed result exactly
            // equal to the two stages computed separately — which is what lets the parity
            // test use torch.equal instead of a tolerance.
            const uint32_t oi = (tileIdM * 2) * TR_MMA_M + laneId;
            const float v = reduced * wscale;
            out[oi] = accum ? __fadd_rn(out[oi], v) : v;
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
/* GROUPED (fused MoE): the SAME kernel serving one expert per 8-token tile, so the whole
 * routed-expert step is one launch with device-side dispatch and no host sync — the property
 * that lets vLLM capture the MoE decode in a CUDA graph. Deliberately an `if constexpr` arm
 * of this kernel rather than a forked copy: e8p's grouped kernel IS a fork
 * (glq_e8p.cu:437, "byte-identical to the single kernel"), and a body that must stay
 * byte-identical by inspection is a body that eventually won't be. Three additions, all
 * hoisted above the k-loop so the steady state is untouched:
 *   1. expert route — eidx = m_indices[blockIdx.y*8], and the weight base advances by
 *      eidx*w_estride_u16. Reading the expert from the tile's FIRST slot is sound because
 *      the host pads every expert run to GLQ_MOE_GROUP_TILE=16 slots and fills them as a
 *      dense prefix, so an 8-slot tile lies wholly inside one expert's run.
 *   2. pad-tile early return (eidx < 0) — block-uniform, before any __syncthreads.
 *   3. per-expert scale from wscale_dev[eidx], times inv_rs_dev[eidx] on a residual stage
 *      (same fold e8p uses, so stacked-RVQ stage 2 needs no second scale path).
 * The base offset is size_t, not the body's signed int weight_idx: gemma-4's w13 is 2.97 M
 * u16 per expert and 128 experts clears INT_MAX by only 5×. */
template <uint32_t R, bool IS_3INST = false, bool GROUPED = false>
__global__ static void __launch_bounds__(TR_BLOCK_SIZE, 1)
glq_trellis_matmul_kernel(float *__restrict__ out,
                          const uint32_t *__restrict__ compressed,
                          const half2 *__restrict__ x,
                          const half2 *__restrict__ codebook,
                          uint32_t m, uint32_t k, uint32_t B, float wscale,
                          bool accum = false,
                          const int *__restrict__ m_indices = nullptr,
                          const float *__restrict__ wscale_dev = nullptr,
                          const float *__restrict__ inv_rs_dev = nullptr,
                          size_t w_estride_u16 = 0) {
    extern __shared__ __align__(16) half2 smem_codebook[];

    // Grouped route (see the note above). Local copies, not reassigned parameters: the
    // body's `compressed` is __restrict__-qualified and its value must not change under it.
    const uint32_t *cp = compressed;
    float ws = wscale;
    if constexpr (GROUPED) {
        const uint32_t base = blockIdx.y * 8;
        const int eidx = (base < B) ? m_indices[base] : -1;
        if (eidx < 0) return;                       // whole tile is padding — nothing to do
        cp = (const uint32_t *)((const uint16_t *)compressed + (size_t)eidx * w_estride_u16);
        ws = wscale_dev[eidx];
        if (inv_rs_dev != nullptr) ws *= inv_rs_dev[eidx];
    }

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
            tr_load_reg_cs<R, IS_3INST>((const uint16_t *)cp, weight_idx, laneId, reg_cs_next, reg_cs2_next);
        uint4 reg_cs, reg_cs2;
        float4 reg_p[2] = {};

#pragma unroll 4
        for (uint32_t ki = 0; ki < this_warp_k; ki += 1) {
            if (ki + 1 != this_warp_k && ki % 2 == 1) weight_idx += weight_step * 2;
            reg_cs = reg_cs_next; reg_cs2 = reg_cs2_next;
            tr_load_reg_cs<R, IS_3INST>((const uint16_t *)cp,
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
                    // RS1: ×wscale folded into the store; RVQ stage 2 accumulates (see the
                    // matvec note — disjoint (m-range, token-tile) ownership makes `+=` safe
                    // and deterministic, and __fadd_rn keeps it exactly stage1+stage2).
                    // GROUPED: `ws` is this tile's expert scale; ungrouped it IS `wscale`.
                    const size_t oi = (size_t)out_tok * m + (tileIdM * 2) * TR_MMA_M + laneId;
                    const float v = reduced * ws;
                    out[oi] = accum ? __fadd_rn(out[oi], v) : v;
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
        // FUSE_IN (3inst matvec) smem = 2·k·4 + k·2 bytes; opt in up to k=8192 (81920 B) so
        // the >48KB shapes launch. One-time here — never on the steady (capturable) path.
        cudaFuncSetAttribute((const void *)glq_trellis_matvec_kernel<2, true, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 81920);
        cudaFuncSetAttribute((const void *)glq_trellis_matvec_kernel<3, true, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 81920);
        cudaFuncSetAttribute((const void *)glq_trellis_matvec_kernel<4, true, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 81920);
    });
}

// R (bits/weight) is recoverable from the packed shape: cols == ceil(256*R/16) == 16*R.
/* `allow_r1` is opt-in per call site, NOT a widened global bound. R=1 exists only as the
 * stacked-RVQ residual rate (bpw 5 = 4+1) and is instantiated only for 3INST; HYB and every
 * primary-stage call site keep the R>=2 floor, so an R=1 buffer can never be mistaken for a
 * stage-1 tensor. Widening this check alone would be actively dangerous: the launcher ladders
 * end in a bare `else` that runs the R=4 kernel, which over 16-column data reads a
 * neighbour's bits and returns plausible garbage. */
int tr_bits_from_packed(const torch::Tensor &packed, bool allow_r1 = false) {
    TORCH_CHECK(packed.dim() == 2, "trellis_packed must be 2-D [(m/16)*(k/16), 16*R]");
    int R = (int)packed.size(1) / 16;
    const int lo = allow_r1 ? 1 : 2;
    TORCH_CHECK(R >= lo && R <= 4,
                "trellis kernel supports R (bits/weight) ", lo, "-4, got ", R);
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

/* Fused B=1 GEMV: out (m,) fp32 = wscale * (W(m,k) @ x(k,)), weights never materialized. */
torch::Tensor glq_decode_matvec_trellis_cuda(torch::Tensor x, torch::Tensor trellis_packed,
                                             torch::Tensor tlut, int64_t m, int64_t k,
                                             double wscale) {
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
        op, cp, xp, cb, (uint32_t)m, (uint32_t)k, (float)wscale)
    if (R == 2)      { TR_LAUNCH_MATVEC(2); }
    else if (R == 3) { TR_LAUNCH_MATVEC(3); }
    else             { TR_LAUNCH_MATVEC(4); }
#undef TR_LAUNCH_MATVEC
    return out;
}

/* Batched GEMM: out (B, m) fp32 = wscale * (x (B, k) @ W(m,k).T), weights never materialized. */
torch::Tensor glq_decode_matmul_trellis_cuda(torch::Tensor x, torch::Tensor trellis_packed,
                                             torch::Tensor tlut, int64_t m, int64_t k,
                                             double wscale) {
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
        op, cp, xp, cb, (uint32_t)m, (uint32_t)k, (uint32_t)B, (float)wscale)
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
        // RS1: ×wscale happens in the kernel store — no separate elementwise launch.
        auto yv = glq_decode_matvec_trellis_cuda(xh, trellis_packed, tlut, m_pad, n_pad, wscale);
        y_rht = yv.view({1, (long)m_pad});
    } else if (B <= batch_max && !force_dense) {
        auto xh = x_rht.to(torch::kFloat16);                             // (B, n_pad)
        y_rht = glq_decode_matmul_trellis_cuda(xh, trellis_packed, tlut, m_pad, n_pad, wscale);
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
                                                int64_t m, int64_t k, bool allow_r1 = false) {
    CHECK_INPUT(trellis_packed);
    int R = tr_bits_from_packed(trellis_packed, allow_r1);
    tr_check_shape(m, k, trellis_packed, R);
    at::DeviceGuard guard(trellis_packed.device());

    auto W = torch::empty({m, k}, torch::dtype(torch::kFloat16).device(trellis_packed.device()));
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    const uint32_t *cp = (const uint32_t *)trellis_packed.data_ptr<int16_t>();
    half *wp = (half *)W.data_ptr<c10::Half>();

#define TR_LAUNCH_DECOMP3(RBITS)                                                       \
    glq_trellis_decompress_kernel<RBITS, true><<<tr_grid_x(), TR_BLOCK_SIZE, 0, stream>>>( \
        wp, cp, (const half2 *)nullptr, (uint32_t)m, (uint32_t)k)
    if (R == 1)      { TR_LAUNCH_DECOMP3(1); }   // RVQ residual rate (bpw 5); see tr_bits_from_packed
    else if (R == 2) { TR_LAUNCH_DECOMP3(2); }
    else if (R == 3) { TR_LAUNCH_DECOMP3(3); }
    else             { TR_LAUNCH_DECOMP3(4); }
#undef TR_LAUNCH_DECOMP3
    return W;
}

torch::Tensor glq_decode_matvec_trellis_3inst_cuda(torch::Tensor x, torch::Tensor trellis_packed,
                                                   int64_t m, int64_t k, double wscale,
                                                   c10::optional<torch::Tensor> out_opt,
                                                   bool accum = false) {
    CHECK_INPUT(x);
    CHECK_INPUT(trellis_packed);
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be fp16");
    TORCH_CHECK(x.numel() == k, "x must have k elements, got ", x.numel());
    // accum IS the "this is a residual stage" predicate, so it also gates R=1 — the two can
    // never disagree, and a stage-1 launch can never reach the R=1 instantiation.
    TORCH_CHECK(!accum || out_opt.has_value(),
                "accumulate mode needs the caller's `out` (stage 1 is the initializer)");
    int R = tr_bits_from_packed(trellis_packed, /*allow_r1=*/accum);
    tr_check_shape(m, k, trellis_packed, R);
    at::DeviceGuard guard(x.device());

    // S4b: an (m,) fp32 contiguous view (e.g. a shared y_rht row slice) may be supplied
    // so the store lands in place — no extra copy between decode and the shards RHT.
    auto out = out_opt.has_value()
        ? *out_opt
        : torch::empty({m}, torch::dtype(torch::kFloat32).device(x.device()));
    TORCH_CHECK(out.is_contiguous() && out.numel() == m
                    && out.scalar_type() == torch::kFloat32,
                "out must be a contiguous (m,) fp32 tensor");
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    const uint32_t *cp = (const uint32_t *)trellis_packed.data_ptr<int16_t>();
    const half2 *xp = (const half2 *)x.data_ptr<c10::Half>();
    float *op = out.data_ptr<float>();

#define TR_LAUNCH_MATVEC3(RBITS)                                                       \
    glq_trellis_matvec_kernel<RBITS, true><<<tr_grid_x(), TR_BLOCK_SIZE, 0, stream>>>( \
        op, cp, xp, (const half2 *)nullptr, (uint32_t)m, (uint32_t)k, (float)wscale,   \
        (const half *)nullptr, 0.0f, 0u, (const int4 *)nullptr, 1u, accum)
    if (R == 1)      { TR_LAUNCH_MATVEC3(1); }   // residual stage only (gated by `accum`)
    else if (R == 2) { TR_LAUNCH_MATVEC3(2); }
    else if (R == 3) { TR_LAUNCH_MATVEC3(3); }
    else             { TR_LAUNCH_MATVEC3(4); }
#undef TR_LAUNCH_MATVEC3
    return out;
}

/* RS2b/RS3 — fused-input B=1 GEMV: takes the RAW (pre-RHT) fp16 x and performs the input
 * RHT inside every matvec block (see the FUSE_IN kernel note). out (m,) fp32 =
 * wscale * (W(m,k) @ blockRHT(x·sv)). Single-block pow2 shapes use the host 1/sqrtf(k);
 * block-diagonal shapes pass blocks_n_meta (int4 {offset,bs,log_bs,0}, GPU) + num_blocks +
 * max_bs and normalize per sub-block with rsqrtf(bs), matching the multiblock RHT kernel. */
torch::Tensor glq_decode_matvec_trellis_3inst_fusein_cuda(
    torch::Tensor x_raw, torch::Tensor sv, torch::Tensor trellis_packed,
    int64_t m, int64_t k, int64_t in_features, double wscale,
    c10::optional<torch::Tensor> blocks_n_meta_opt, int64_t num_blocks, int64_t max_bs,
    c10::optional<torch::Tensor> out_opt) {
    // optional (not a default-constructed Tensor): pybind cannot round-trip an undefined
    // at::Tensor as a py::arg default — single-block callers simply omit it.
    torch::Tensor blocks_n_meta =
        blocks_n_meta_opt.has_value() ? *blocks_n_meta_opt : torch::Tensor();
    CHECK_INPUT(x_raw);
    CHECK_INPUT(sv);
    CHECK_INPUT(trellis_packed);
    TORCH_CHECK(x_raw.scalar_type() == torch::kFloat16, "x_raw must be fp16");
    TORCH_CHECK(x_raw.numel() == in_features, "x_raw must have in_features elements");
    TORCH_CHECK(sv.scalar_type() == torch::kFloat16 && sv.numel() == k, "sv must be (k,) fp16");
    const bool multiblock = num_blocks > 1;
    if (multiblock) {
        TORCH_CHECK(blocks_n_meta.is_cuda() && blocks_n_meta.numel() >= num_blocks * 4,
                    "block-diag FUSE_IN needs GPU blocks_n_meta");
        TORCH_CHECK(max_bs > 0 && max_bs <= 8192 && (max_bs & (max_bs - 1)) == 0,
                    "block-diag FUSE_IN needs pow2 max_bs <= 8192, got ", max_bs);
    } else {
        TORCH_CHECK((k & (k - 1)) == 0 && k <= 8192,
                    "FUSE_IN needs a single-block pow2 RHT dim <= 8192, got ", k);
        max_bs = k;
    }
    int R = tr_bits_from_packed(trellis_packed);
    tr_check_shape(m, k, trellis_packed, R);
    at::DeviceGuard guard(x_raw.device());
    tr_init_once();

    // S4b: optional caller-provided (m,) fp32 contiguous destination (shared y_rht slice).
    auto out = out_opt.has_value()
        ? *out_opt
        : torch::empty({m}, torch::dtype(torch::kFloat32).device(x_raw.device()));
    TORCH_CHECK(out.is_contiguous() && out.numel() == m
                    && out.scalar_type() == torch::kFloat32,
                "out must be a contiguous (m,) fp32 tensor");
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    const uint32_t *cp = (const uint32_t *)trellis_packed.data_ptr<int16_t>();
    const half2 *xp = (const half2 *)x_raw.data_ptr<c10::Half>();   // RAW fp16 under FUSE_IN
    const half *svp = (const half *)sv.data_ptr<c10::Half>();
    const int4 *bm = multiblock ? (const int4 *)blocks_n_meta.data_ptr<int32_t>() : nullptr;
    float *op = out.data_ptr<float>();
    const float rsqrt_n = 1.0f / sqrtf((float)k);                   // matches single-block RHT
    const size_t smem = (size_t)k * 4 + (size_t)max_bs * 4;         // persist + scratch/half2
    TORCH_CHECK(smem <= 81920, "FUSE_IN smem ", smem, " exceeds the 81920 B opt-in");

#define TR_LAUNCH_MATVEC3F(RBITS)                                                          \
    glq_trellis_matvec_kernel<RBITS, true, true><<<tr_grid_x(), TR_BLOCK_SIZE, smem, stream>>>( \
        op, cp, xp, (const half2 *)nullptr, (uint32_t)m, (uint32_t)k, (float)wscale,       \
        svp, rsqrt_n, (uint32_t)in_features, bm, (uint32_t)(multiblock ? num_blocks : 1))
    if (R == 2)      { TR_LAUNCH_MATVEC3F(2); }
    else if (R == 3) { TR_LAUNCH_MATVEC3F(3); }
    else             { TR_LAUNCH_MATVEC3F(4); }
#undef TR_LAUNCH_MATVEC3F
    return out;
}

torch::Tensor glq_decode_matmul_trellis_3inst_cuda(torch::Tensor x, torch::Tensor trellis_packed,
                                                   int64_t m, int64_t k, double wscale,
                                                   c10::optional<torch::Tensor> out_opt = c10::nullopt,
                                                   bool accum = false) {
    CHECK_INPUT(x);
    CHECK_INPUT(trellis_packed);
    TORCH_CHECK(x.dim() == 2, "x must be (B, k)");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be fp16");
    TORCH_CHECK(x.size(1) == k, "x must have k columns, got ", x.size(1));
    // As in the GEMV: `accum` marks a residual stage, so it alone unlocks R=1, and it
    // requires the caller's buffer because stage 1 is what initialized it.
    TORCH_CHECK(!accum || out_opt.has_value(),
                "accumulate mode needs the caller's `out` (stage 1 is the initializer)");
    int R = tr_bits_from_packed(trellis_packed, /*allow_r1=*/accum);
    tr_check_shape(m, k, trellis_packed, R);
    at::DeviceGuard guard(x.device());

    const int64_t B = x.size(0);
    auto out = out_opt.has_value()
        ? *out_opt
        : torch::empty({B, m}, torch::dtype(torch::kFloat32).device(x.device()));
    TORCH_CHECK(out.is_contiguous() && out.numel() == B * m
                    && out.scalar_type() == torch::kFloat32,
                "out must be a contiguous (B, m) fp32 tensor");
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    const uint32_t *cp = (const uint32_t *)trellis_packed.data_ptr<int16_t>();
    const half2 *xp = (const half2 *)x.data_ptr<c10::Half>();
    float *op = out.data_ptr<float>();
    dim3 grid(tr_grid_x(), (unsigned)((B + 7) / 8));   // 8 tokens per mma N-tile

#define TR_LAUNCH_MATMUL3(RBITS)                                                       \
    glq_trellis_matmul_kernel<RBITS, true><<<grid, TR_BLOCK_SIZE, 0, stream>>>(        \
        op, cp, xp, (const half2 *)nullptr, (uint32_t)m, (uint32_t)k, (uint32_t)B,      \
        (float)wscale, accum)
    if (R == 1)      { TR_LAUNCH_MATMUL3(1); }   // residual stage only (gated by `accum`)
    else if (R == 2) { TR_LAUNCH_MATMUL3(2); }
    else if (R == 3) { TR_LAUNCH_MATMUL3(3); }
    else             { TR_LAUNCH_MATMUL3(4); }
#undef TR_LAUNCH_MATMUL3
    return out;
}

/* Fused 3INST linear: identical bracket + hybrid B-dispatch as the HYB fused op, minus the
 * tlut. Same env knobs (GLQ_TRELLIS_BATCH_MAX / GLQ_TRELLIS_DENSE). One host call →
 * cudagraph-capturable as a single node. */
/* Steps 1-2 of the fused 3INST linear (input RHT + decode/matmul + ×wscale) →
 * y_rht (B, m_pad) fp32. `y_out_slice`: optional contiguous (m_pad,) destination the B=1
 * matvec branches store into DIRECTLY (S4b writes a shared shard buffer row with no copy);
 * batched/dense branches ignore it and return a fresh tensor — the caller compares data
 * pointers to decide whether a copy is still needed. */
static torch::Tensor tr3_forward_yrht(
    torch::Tensor x, torch::Tensor sv, torch::Tensor trellis_packed,
    torch::Tensor blocks_n, torch::Tensor blocks_n_meta,
    double wscale, int64_t in_features, int64_t n_pad, int64_t m_pad,
    c10::optional<torch::Tensor> y_out_slice,
    torch::Tensor trellis_packed2 = torch::Tensor(),   // stacked RVQ (5-8 bpw); empty = absent
    double inv_resid_scale2 = 0.0
) {
    int B = x.size(0);

    // Stacked RVQ: W = decode(stage1) + inv_resid_scale2 * decode(stage2), summed by matmul
    // linearity as y1 + rs2*y2. The buffer's presence IS the stage count, so a present buffer
    // with a zero scale (or vice versa) is a half-configured layer — exactly the shape of the
    // e8p stage-3/4 silent drop, where the flag said "present" while the scale stayed 0.0 and
    // the stage vanished with no error. Refuse instead.
    const bool has_s2 = trellis_packed2.defined() && trellis_packed2.numel() > 0;
    TORCH_CHECK(has_s2 == (inv_resid_scale2 != 0.0),
                "trellis RVQ stage 2 is half-configured: trellis_packed2 is ",
                has_s2 ? "present" : "absent", " but inv_resid_scale2 = ", inv_resid_scale2,
                ". Refusing rather than silently decoding stage 1 only.");
    // Scale folded host-side into ONE double so the kernel store stays a single multiply.
    const double wscale2 = wscale * inv_resid_scale2;

    static const int64_t batch_max = [] {
        const char *e = std::getenv("GLQ_TRELLIS_BATCH_MAX");
        return e ? std::max<int64_t>(1, atoll(e)) : 64;
    }();
    static const bool force_dense = (std::getenv("GLQ_TRELLIS_DENSE") != nullptr);
    // RS2b kill-switch: GLQ_TRELLIS_FUSE_INPUT=0 restores the standalone input-RHT kernel.
    static const bool fuse_in = [] {
        const char *e = std::getenv("GLQ_TRELLIS_FUSE_INPUT");
        return !(e && e[0] == '0');
    }();

    // FUSE_IN eligibility: single-block pow2 (RS2b) or block-diagonal with GPU meta (RS3).
    const int64_t fnb = blocks_n.numel() > 0 ? blocks_n.size(0) : 1;
    int64_t fmax_bs = n_pad;
    if (fnb > 1) fmax_bs = blocks_n.max().item<int64_t>();
    const bool fusein_sb = fnb <= 1 && n_pad <= 8192 && ((n_pad & (n_pad - 1)) == 0);
    const bool fusein_bd = fnb > 1 && fmax_bs <= 8192
                           && blocks_n_meta.numel() > 0 && blocks_n_meta.is_cuda()
                           && ((size_t)n_pad * 4 + (size_t)fmax_bs * 4) <= 81920;

    torch::Tensor y_rht;
    // FUSE_IN computes the input RHT in smem and DISCARDS it, so a second decode pass would
    // have no transformed x to read — re-running the transform would silently garbage stage 2.
    // Stage-2 layers therefore take the explicit-x_rht path, costing one extra launch for the
    // input RHT. (Follow-up: have pass 1 export xh to a caller buffer and restore FUSE_IN.)
    if (fuse_in && B == 1 && !force_dense && !has_s2 && (fusein_sb || fusein_bd)) {
        // ---- RS2b/RS3 fused path: input RHT + cast + decode + ×wscale in ONE kernel ----
        auto yv = glq_decode_matvec_trellis_3inst_fusein_cuda(
            x.contiguous().view({-1}), sv, trellis_packed, m_pad, n_pad, in_features, wscale,
            blocks_n_meta, fnb, fmax_bs, y_out_slice);
        y_rht = yv.view({1, (long)m_pad});
    } else {
        // ---- Step 1: input RHT → x_rht (B, n_pad) fp32 ----
        auto x_rht = torch::empty({B, (long)n_pad},
                                  torch::dtype(torch::kFloat32).device(x.device()));
        glq_input_rht_blockdiag_cuda(x.contiguous(), sv, x_rht,
                                     (int)in_features, (int)n_pad, blocks_n, blocks_n_meta);

        // ---- Step 2: decode + matmul in the RHT domain → y_rht (B, m_pad) fp32 ----
        if (B == 1 && !force_dense) {
            auto xh = x_rht.view({(long)n_pad}).to(torch::kFloat16);
            // RS1: ×wscale happens in the kernel store — no separate elementwise launch.
            auto yv = glq_decode_matvec_trellis_3inst_cuda(xh, trellis_packed, m_pad, n_pad,
                                                           wscale, y_out_slice);
            // Stage 2 accumulates ONTO stage 1's output — same buffer, no temp tensor and no
            // extra pass over y. Stream-ordered after stage 1, disjoint m-range per block.
            if (has_s2)
                glq_decode_matvec_trellis_3inst_cuda(xh, trellis_packed2, m_pad, n_pad,
                                                     wscale2, yv, /*accum=*/true);
            y_rht = yv.view({1, (long)m_pad});
        } else if (B <= batch_max && !force_dense) {
            auto xh = x_rht.to(torch::kFloat16);                             // (B, n_pad)
            y_rht = glq_decode_matmul_trellis_3inst_cuda(xh, trellis_packed, m_pad, n_pad,
                                                          wscale);
            if (has_s2)
                glq_decode_matmul_trellis_3inst_cuda(xh, trellis_packed2, m_pad, n_pad,
                                                     wscale2, y_rht, /*accum=*/true);
        } else {
            auto W = glq_decompress_trellis_3inst_cuda(trellis_packed, m_pad, n_pad);   // fp16
            if (has_s2) {
                // Dense prefill sums the WEIGHTS before one GEMM — same result by linearity,
                // and one GEMM instead of two. The fp16 add keeps opmath in fp32 internally
                // with no m×k fp32 transient, matching this branch's existing fp16 rationale.
                auto W2 = glq_decompress_trellis_3inst_cuda(trellis_packed2, m_pad, n_pad,
                                                            /*allow_r1=*/true);
                W.add_(W2, (float)inv_resid_scale2);
            }
            auto xh = x_rht.to(torch::kFloat16);
            y_rht = (at::matmul(xh, W.t()).to(torch::kFloat32) * (float)wscale).contiguous();
        }
    }
    return y_rht;
}

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

    auto y_rht = tr3_forward_yrht(x, sv, trellis_packed, blocks_n, blocks_n_meta,
                                  wscale, in_features, n_pad, m_pad, c10::nullopt);

    // ---- Step 3: output RHT → y (B, out_features) fp16 ----
    auto y = torch::empty({B, (long)out_features},
                          torch::dtype(torch::kFloat16).device(x.device()));
    glq_output_rht_blockdiag_cuda(y_rht, su, y, (int)out_features, (int)m_pad,
                                  blocks_m, blocks_m_meta);
    return y;
}

/* S4b: the fused 3INST linear STOPPED at the y_rht seam — per-shard calls of a fused
 * QKV/gate_up linear each deposit their (B, m_pad) fp32 result into columns
 * [col, col+m_pad) of the SHARED y_rht_out buffer, and ONE glq_output_rht_shards_cuda
 * launch then replaces the per-shard output RHTs. B=1 matvec branches store the row
 * slice in place (zero copy); batched/dense branches copy their fresh result in. */
void glq_fused_linear_trellis_3inst_yrht_cuda(
    torch::Tensor x, torch::Tensor sv, torch::Tensor trellis_packed,
    torch::Tensor blocks_n, torch::Tensor blocks_n_meta,
    double wscale, int64_t in_features, int64_t n_pad, int64_t m_pad,
    torch::Tensor y_rht_out,       // (B, total_m) fp32, contiguous
    int64_t col                    // this shard's column offset in y_rht_out
) {
    CHECK_INPUT(x);
    CHECK_INPUT(trellis_packed);
    CHECK_INPUT(y_rht_out);
    TORCH_CHECK(y_rht_out.scalar_type() == torch::kFloat32
                    && y_rht_out.dim() == 2 && y_rht_out.size(0) == x.size(0)
                    && col + m_pad <= y_rht_out.size(1),
                "y_rht_out must be (B, >= col+m_pad) fp32");
    int64_t B = x.size(0);
    at::DeviceGuard guard(x.device());

    c10::optional<torch::Tensor> slice;
    if (B == 1)
        slice = y_rht_out.select(0, 0).narrow(0, col, m_pad);   // contiguous row slice
    auto y = tr3_forward_yrht(x, sv, trellis_packed, blocks_n, blocks_n_meta,
                              wscale, in_features, n_pad, m_pad, slice);
    if (!slice.has_value() || y.data_ptr() != slice->data_ptr())
        y_rht_out.narrow(1, col, m_pad).copy_(y.view({B, (long)m_pad}));
}

/* ══ Stacked-RVQ (5-8 bpw) entries — SEPARATE symbols, not widened signatures ══
 * The Python registration is guarded by `hasattr(cuda, "<symbol>")`, which tests a NAME,
 * not an arity. Widening the shipped 13-arg entry would let a STALE .so pass that guard and
 * then bind a 13-arg function to a 15-arg schema — failing deep inside dispatch, possibly
 * mid-capture. A new symbol turns the guard into a true capability probe: absent → the op is
 * never defined → the caller falls back to the eager decode with a warning. Every 2-4 bpw
 * checkpoint keeps the byte-identical old entry (so its tests stay the back-compat gate),
 * and both share ONE core (tr3_forward_yrht), so the 1- and 2-stage math cannot drift. */
torch::Tensor glq_fused_linear_trellis_3inst_rvq2_cuda(
    torch::Tensor x,               // (B, in_features) fp16, contiguous
    torch::Tensor sv,              // (n_pad,) fp16
    torch::Tensor su,              // (m_pad,) fp16
    torch::Tensor trellis_packed,  // stage 1: ((m_pad/16)*(n_pad/16), 16*R1) int16, R1 = 4
    torch::Tensor trellis_packed2, // stage 2: same rows, 16*R2 cols, R2 = bpw-4 in 1..4
    torch::Tensor blocks_n, torch::Tensor blocks_m,
    torch::Tensor blocks_n_meta, torch::Tensor blocks_m_meta,
    double wscale, double inv_resid_scale2,
    int64_t in_features, int64_t out_features,
    int64_t n_pad, int64_t m_pad
) {
    // This entry is 2-stage BY CONSTRUCTION: refusing an empty stage 2 here makes it
    // impossible to reach the fused path in a configuration that would decode stage 1 only.
    TORCH_CHECK(trellis_packed2.numel() > 0 && inv_resid_scale2 != 0.0,
                "the rvq2 entry requires a populated stage 2 (got packed2.numel()=",
                trellis_packed2.numel(), ", inv_resid_scale2=", inv_resid_scale2,
                "); single-stage layers must use glq_fused_linear_trellis_3inst_cuda");
    CHECK_INPUT(x);
    CHECK_INPUT(trellis_packed);
    CHECK_INPUT(trellis_packed2);
    int B = x.size(0);
    at::DeviceGuard guard(x.device());

    auto y_rht = tr3_forward_yrht(x, sv, trellis_packed, blocks_n, blocks_n_meta,
                                  wscale, in_features, n_pad, m_pad, c10::nullopt,
                                  trellis_packed2, inv_resid_scale2);

    auto y = torch::empty({B, (long)out_features},
                          torch::dtype(torch::kFloat16).device(x.device()));
    glq_output_rht_blockdiag_cuda(y_rht, su, y, (int)out_features, (int)m_pad,
                                  blocks_m, blocks_m_meta);
    return y;
}

/* S4b shard-batched variant of the above (see the 1-stage yrht note). */
void glq_fused_linear_trellis_3inst_yrht_rvq2_cuda(
    torch::Tensor x, torch::Tensor sv,
    torch::Tensor trellis_packed, torch::Tensor trellis_packed2,
    torch::Tensor blocks_n, torch::Tensor blocks_n_meta,
    double wscale, double inv_resid_scale2,
    int64_t in_features, int64_t n_pad, int64_t m_pad,
    torch::Tensor y_rht_out, int64_t col
) {
    TORCH_CHECK(trellis_packed2.numel() > 0 && inv_resid_scale2 != 0.0,
                "the rvq2 yrht entry requires a populated stage 2 (got packed2.numel()=",
                trellis_packed2.numel(), ", inv_resid_scale2=", inv_resid_scale2, ")");
    CHECK_INPUT(x);
    CHECK_INPUT(trellis_packed);
    CHECK_INPUT(trellis_packed2);
    CHECK_INPUT(y_rht_out);
    TORCH_CHECK(y_rht_out.scalar_type() == torch::kFloat32
                    && y_rht_out.dim() == 2 && y_rht_out.size(0) == x.size(0)
                    && col + m_pad <= y_rht_out.size(1),
                "y_rht_out must be (B, >= col+m_pad) fp32");
    int64_t B = x.size(0);
    at::DeviceGuard guard(x.device());

    c10::optional<torch::Tensor> slice;
    if (B == 1)
        slice = y_rht_out.select(0, 0).narrow(0, col, m_pad);   // contiguous row slice
    auto y = tr3_forward_yrht(x, sv, trellis_packed, blocks_n, blocks_n_meta,
                              wscale, in_features, n_pad, m_pad, slice,
                              trellis_packed2, inv_resid_scale2);
    if (!slice.has_value() || y.data_ptr() != slice->data_ptr())
        y_rht_out.narrow(1, col, m_pad).copy_(y.view({B, (long)m_pad}));
}

/* ══ Grouped (per-expert) 3INST matmul for the fused MoE — the capturable decode path ══
 *
 * Counterpart of launch_grouped_matmul_e8p (glq_e8p.cu), called from glq_fused_moe_trellis_
 * 3inst_cuda in glq_cuda.cu. Two launches at most and NO scratch, NO reduce kernel and NO
 * allocation: the matmul kernel's in-block fixed-order reduce is already complete, and each
 * (m-range, token-tile) owns disjoint output. That is what makes the whole MoE step safe to
 * capture — e8p needs a split-K scratch plane plus a reduce pass here.
 *
 * Stacked RVQ (5-8 bpw): stage 2 is the same kernel with accum=true and the per-expert scale
 * pre-multiplied by inv_resid_scale2 inside the kernel, exactly as the single-linear path
 * composes y1 + rs2*y2 by matmul linearity. R2 == 0 means "no residual" — and R2 == 1 is
 * reachable ONLY here (bpw 5 = 4+1), which is why the R ladder below drops to 1 for stage 2
 * and not for stage 1.
 *
 * `x_grouped` (M_sum_max, k) fp16 is already in the RHT domain and pad rows are zeroed by
 * glq_moe_gather_rows_kernel; `y_out` (M_sum_max, m) fp32 rows for pad TILES are left
 * untouched and are never read (the grouped output RHT skips m_indices < 0). */
void launch_grouped_matmul_trellis_3inst(
    float *y_out,                  // (M_sum_max, m) fp32
    const int16_t *packed,         // (E, tiles, 16*R1) int16 — stage 1
    const int16_t *packed2,        // (E, tiles, 16*R2) int16 — stage 2, or nullptr
    const half *x_grouped,         // (M_sum_max, k) fp16, RHT domain, grouped
    int R1, int R2,                // bits/weight per stage; R2 == 0 ⇒ no residual
    size_t stride1_u16, size_t stride2_u16,   // int16 elements per expert, per stage
    int M_sum_max, int m, int k,
    const int *m_indices,          // (M_sum_max,) expert per slot, -1 = pad
    const float *wscale_dev,       // (E,) fp32
    const float *inv_rs2_dev,      // (E,) fp32 — read only when packed2 != nullptr
    cudaStream_t stream
) {
    // Both ladders below end in a bare `else` that runs the R=4 kernel, which over
    // narrower data reads a neighbour's bits and returns PLAUSIBLE GARBAGE (the hazard
    // tr_bits_from_packed's comment describes). Bound the rates here so a caller that
    // mis-derives R fails loudly instead.
    TORCH_CHECK(R1 >= 2 && R1 <= 4, "grouped trellis stage 1 needs R 2-4, got ", R1);
    TORCH_CHECK(R2 >= 0 && R2 <= 4, "grouped trellis stage 2 needs R 0-4, got ", R2);
    TORCH_CHECK((packed2 != nullptr) == (R2 > 0),
                "grouped trellis stage 2 is half-configured: packed2 is ",
                packed2 ? "present" : "absent", " but R2 = ", R2);
    TORCH_CHECK(packed2 == nullptr || inv_rs2_dev != nullptr,
                "grouped trellis stage 2 present but inv_resid_scale2 pointer is null");
    // Cap grid.x at the number of m-tile-pairs: the kernel derives m_per_block from
    // gridDim.x, so blocks beyond that would launch only to hit `tileIdM*2 >= tileCountM`
    // and return. Harmless but not free at 128 experts × 2 shards × every layer.
    const uint32_t m_pairs = (uint32_t)((m / TR_MMA_M + 1) / 2);
    const uint32_t gx = m_pairs < tr_grid_x() ? m_pairs : tr_grid_x();
    dim3 grid(gx, (unsigned)((M_sum_max + 7) / 8));   // 8 token-slots per mma N-tile

#define TR_LAUNCH_GROUPED(RBITS, PTR, STRIDE, IRS, ACC)                                 \
    glq_trellis_matmul_kernel<RBITS, true, true><<<grid, TR_BLOCK_SIZE, 0, stream>>>(   \
        y_out, (const uint32_t *)(PTR), (const half2 *)x_grouped, (const half2 *)nullptr, \
        (uint32_t)m, (uint32_t)k, (uint32_t)M_sum_max, 0.0f, (ACC),                     \
        m_indices, wscale_dev, (IRS), (STRIDE))

    if (R1 == 2)      { TR_LAUNCH_GROUPED(2, packed, stride1_u16, nullptr, false); }
    else if (R1 == 3) { TR_LAUNCH_GROUPED(3, packed, stride1_u16, nullptr, false); }
    else              { TR_LAUNCH_GROUPED(4, packed, stride1_u16, nullptr, false); }

    if (packed2 != nullptr && R2 > 0) {
        if (R2 == 1)      { TR_LAUNCH_GROUPED(1, packed2, stride2_u16, inv_rs2_dev, true); }
        else if (R2 == 2) { TR_LAUNCH_GROUPED(2, packed2, stride2_u16, inv_rs2_dev, true); }
        else if (R2 == 3) { TR_LAUNCH_GROUPED(3, packed2, stride2_u16, inv_rs2_dev, true); }
        else              { TR_LAUNCH_GROUPED(4, packed2, stride2_u16, inv_rs2_dev, true); }
    }
#undef TR_LAUNCH_GROUPED
}
