/* glq_trellis_cpu_fp16.cpp — AVX512-FP16 tier: the avx512 tier with native fp16 decode.
 *
 * Shares every kernel body with the avx512 TU via glq_trellis_cpu_zmm.inl; the only
 * difference is the decode policy tail: instead of converting both u16 halves to fp32,
 * adding, and rounding back (VCVTPH2PS x2 + VADDPS + VCVTPS2PH + VCVTPH2PS), one native
 * VADDPH performs the oracle's fp16 add DIRECTLY — decode_3inst's semantics in a single
 * instruction — then one VCVTPH2PS widens for the fp32 FMA.
 *
 * The whole TU sits behind a GCC>=12 guard: __m256h/_mm256_add_ph need the avx512fp16
 * header support (manylinux_2_28's toolset and every box this repo uses qualify); older
 * toolchains compile the null registrar and the tier reports unavailable.
 */
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__GNUC__) && __GNUC__ >= 12
#pragma GCC push_options
#pragma GCC target("avx512fp16,avx512f,avx512bw,avx512vl,avx512dq,f16c,fma")

#include <immintrin.h>

#include "glq_trellis_cpu.hpp"
#include "glq_trellis_layout.hpp"

namespace glq_cpu {
namespace {

#include "glq_trellis_cpu_zmm.inl"

struct DecodeFp16 {
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
        // Native fp16 add IS the oracle (same hi + lo operand order); no rounding
        // round-trip needed — VADDPH's RN result is decode_3inst's value exactly.
        const __m256h sum = _mm256_add_ph(_mm256_castsi256_ph(hi16),
                                          _mm256_castsi256_ph(lo16));
        return _mm512_cvtph_ps(_mm256_castph_si256(sum));
    }
};

void decompress_fp16(const uint16_t* packed, uint16_t* W, int64_t m, int64_t k, int R) {
    zmm_decompress<DecodeFp16>(packed, W, m, k, R);
}
void matvec_fp16(const uint16_t* packed, const float* x, float* y, int64_t m, int64_t k,
                 int R, float wscale, bool accum, int64_t blk_begin, int64_t blk_end) {
    zmm_matvec<DecodeFp16>(packed, x, y, m, k, R, wscale, accum, blk_begin, blk_end);
}
void matmul_fp16(const uint16_t* packed, const float* x, float* y, int64_t B, int64_t m,
                 int64_t k, int R, float wscale, bool accum,
                 int64_t blk_begin, int64_t blk_end) {
    zmm_matmul<DecodeFp16>(packed, x, y, B, m, k, R, wscale, accum, blk_begin, blk_end);
}

const Kernels kFp16 = { decompress_fp16, matvec_fp16, matmul_fp16 };

}  // namespace

const Kernels* avx512fp16_kernels() { return &kFp16; }

}  // namespace glq_cpu

#pragma GCC pop_options
#else
namespace glq_cpu {
struct Kernels;
const Kernels* avx512fp16_kernels() { return nullptr; }
}  // namespace glq_cpu
#endif
