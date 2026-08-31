/* glq_trellis_cpu_avx512.cpp — AVX512F+BW+VL tier.
 *
 * The kernel bodies live in glq_trellis_cpu_zmm.inl, shared with the avx512fp16 tier;
 * this TU contributes the target pragma and the decode policy: after the hash's
 * mul/add/and/xor, the two u16 halves convert via VPMOVDW + VCVTPH2PS, add exactly in
 * fp32, and take the LOAD-BEARING VCVTPS2PH(RN)+VCVTPH2PS round-trip — without the fp16
 * rounding the result differs from the oracle on 42% of states. The fp16 tier replaces
 * exactly this tail with one native VADDPH.
 */
#if defined(__x86_64__) || defined(_M_X64)
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512vl,avx512dq,f16c,fma")

#include <immintrin.h>

#include "glq_trellis_cpu.hpp"
#include "glq_trellis_layout.hpp"

namespace glq_cpu {
namespace {

#include "glq_trellis_cpu_zmm.inl"

struct DecodeAvx512 {
    static inline __m512 decode16(__m512i s, bool arith) {
        if (!arith) return zmm_lut_gather(s);
        const __m512i h = _mm512_add_epi32(
            _mm512_mullo_epi32(s, _mm512_set1_epi32(89226354)),
            _mm512_set1_epi32(64248484));
        const __m512i r = _mm512_xor_si512(
            _mm512_and_si512(h, _mm512_set1_epi32((int)0x8FFF8FFFu)),
            _mm512_set1_epi32((int)0x3B603B60u));
        const __m256i lo16 =
            _mm512_cvtepi32_epi16(_mm512_and_si512(r, _mm512_set1_epi32(0xFFFF)));
        const __m256i hi16 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(r, 16));
        const __m512 sum = _mm512_add_ps(_mm512_cvtph_ps(hi16), _mm512_cvtph_ps(lo16));
        return _mm512_cvtph_ps(_mm512_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT));
    }
};

void decompress_avx512(const uint16_t* packed, uint16_t* W, int64_t m, int64_t k, int R) {
    zmm_decompress<DecodeAvx512>(packed, W, m, k, R);
}
void matvec_avx512(const uint16_t* packed, const float* x, float* y, int64_t m, int64_t k,
                   int R, float wscale, bool accum, int64_t blk_begin, int64_t blk_end) {
    zmm_matvec<DecodeAvx512>(packed, x, y, m, k, R, wscale, accum, blk_begin, blk_end);
}
void matmul_avx512(const uint16_t* packed, const float* x, float* y, int64_t B, int64_t m,
                   int64_t k, int R, float wscale, bool accum,
                   int64_t blk_begin, int64_t blk_end) {
    zmm_matmul<DecodeAvx512>(packed, x, y, B, m, k, R, wscale, accum, blk_begin, blk_end);
}

const Kernels kAvx512 = { decompress_avx512, matvec_avx512, matmul_avx512 };

}  // namespace

const Kernels* avx512_kernels() { return &kAvx512; }

}  // namespace glq_cpu

#pragma GCC pop_options
#else
namespace glq_cpu {
struct Kernels;
const Kernels* avx512_kernels() { return nullptr; }
}  // namespace glq_cpu
#endif
