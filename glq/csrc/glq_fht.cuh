/* glq_fht.cuh — shared FHT butterfly device helpers (RS4a).
 *
 * The 3-line butterfly body used to be inlined ~9× across glq_cuda.cu / glq_trellis.cu;
 * new code uses these helpers. NOTE for the JIT build: torch's cpp_extension hashes only
 * the listed .cu/.cpp sources — after editing this header, remove the cached
 * torch_extensions/<py>/glq_cuda build dir to force a rebuild.
 */
#pragma once

#include <cuda_runtime.h>

/* One smem butterfly stage at distance h over n elements: dst[i] = lo ? a+b : b−a.
 * Grid-strides over the calling block's threads; the caller owns the barrier. */
__device__ __forceinline__ void glq_fht_stage_smem(const float *__restrict__ src,
                                                   float *__restrict__ dst,
                                                   int h, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float a = src[i], b = src[i ^ h];
        dst[i] = ((i & h) == 0) ? (a + b) : (b - a);
    }
}

/* Warp-shuffle the LOW butterfly stages (distance 1,2,4,8,16) IN PLACE on buf —
 * five stages with zero smem round-trips and zero __syncthreads (RS4a: these five
 * barriers+passes were ~40% of every FHT's stage count).
 *
 * Ownership: thread t holds elements i = t + e·blockDim.x. For h ≤ 16 the partner
 * i^h lives at the SAME slot e of lane t^h in the SAME warp (h flips only lane bits,
 * and (i & h) == (t & h)), so each stage is one __shfl_xor_sync per element.
 *
 * Requirements: n pow2 with n ≥ 32 — the active threads (t < n when n < blockDim.x)
 * then form whole warps, keeping the full-mask shuffle converged. Bit-exact vs the
 * smem stages it replaces: identical ascending stage order, identical fp32
 * lo ? a+b : b−a. Caller must __syncthreads() after (later stages read across
 * threads); the smem stage loop then starts at h = 32, and double-buffer callers
 * must recompute their copy-back parity from the REMAINING stage count. */
template <int MAX_E>
__device__ __forceinline__ void glq_fht_shuffle_low(float *__restrict__ buf, int n) {
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    if (tid >= n) return;                       // whole idle warps only (n pow2 ≥ 32)
    const int e_cnt = (n > nthr) ? (n / nthr) : 1;
    float r[MAX_E];
#pragma unroll
    for (int e = 0; e < MAX_E; ++e)
        if (e < e_cnt) r[e] = buf[tid + e * nthr];
    const int hmax = ((n >> 1) < 16) ? (n >> 1) : 16;
    for (int h = 1; h <= hmax; h <<= 1) {
#pragma unroll
        for (int e = 0; e < MAX_E; ++e) {
            if (e < e_cnt) {
                float partner = __shfl_xor_sync(0xFFFFFFFFu, r[e], h);
                r[e] = ((tid & h) == 0) ? (r[e] + partner) : (partner - r[e]);
            }
        }
    }
#pragma unroll
    for (int e = 0; e < MAX_E; ++e)
        if (e < e_cnt) buf[tid + e * nthr] = r[e];
}
