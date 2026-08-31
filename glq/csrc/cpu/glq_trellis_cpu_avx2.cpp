/* glq_trellis_cpu_avx2.cpp — AVX2+F16C+FMA tier.
 *
 * Compiled with an in-source target pragma (never global -mavx2): the JIT build rung has
 * no per-file flags, and baseline TUs must stay baseline. Safe to COMPILE on any x86;
 * dispatch runs it only when the CPU reports avx2+f16c+fma.
 *
 * Structure per 32x32 window-group: the bit-unpack (cheap shift/mask chains) runs scalar
 * into an L1-resident staging array of 16-bit states; the expensive parts — codebook
 * decode and MAC — run 8-wide. Decode micro-variants (runtime-selectable, gate-tested
 * equal):
 *   arith: VPMULLD/VPADDD/VPAND/VPXOR then F16C converts of the two u16 halves, exact
 *          fp32 add, and the LOAD-BEARING VCVTPS2PH(RN)+VCVTPH2PS round-trip — without
 *          the fp16 rounding the result differs from the oracle on 42% of states.
 *   lut:   one VPGATHERDD from the 256 KB fp32 LUT (exact by construction).
 * Accumulation (GEMV): per-lane partial sums in 8 ymm accumulators per m-pair block,
 * folded by VHADDPS pairs — the fixed (l0+l1)+(l2+l3) quad tree — so every output row
 * has one fixed summation order, independent of thread count.
 */
#if defined(__x86_64__) || defined(_M_X64)
#pragma GCC push_options
#pragma GCC target("avx2,f16c,fma,bmi2")

#include <immintrin.h>

#include "glq_trellis_cpu.hpp"
#include "glq_trellis_layout.hpp"

namespace glq_cpu {
namespace {


// ---- decode: 8 states (u32 ymm) -> 8 oracle-exact fp32 weights ----------------------
inline __m256 decode8_lut(__m256i idx) {
    return _mm256_i32gather_ps(g_lut32, idx, 4);
}

inline __m256 decode8_arith(__m256i s) {
    const __m256i h = _mm256_add_epi32(_mm256_mullo_epi32(s, _mm256_set1_epi32(89226354)),
                                       _mm256_set1_epi32(64248484));
    const __m256i r = _mm256_xor_si256(
        _mm256_and_si256(h, _mm256_set1_epi32((int)0x8FFF8FFFu)),
        _mm256_set1_epi32((int)0x3B603B60u));
    const __m256i lo = _mm256_and_si256(r, _mm256_set1_epi32(0xFFFF));
    const __m256i hi = _mm256_srli_epi32(r, 16);
    // pack 8 u32 (each holding a u16 bit pattern) -> 8 u16 in one xmm, order-preserving
    const __m128i lo16 = _mm_packus_epi32(_mm256_castsi256_si128(lo),
                                          _mm256_extracti128_si256(lo, 1));
    const __m128i hi16 = _mm_packus_epi32(_mm256_castsi256_si128(hi),
                                          _mm256_extracti128_si256(hi, 1));
    const __m256 sum = _mm256_add_ps(_mm256_cvtph_ps(hi16), _mm256_cvtph_ps(lo16));
    // exact fp32 sum -> oracle fp16 (RN) -> exact fp32 widening
    return _mm256_cvtph_ps(_mm256_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT));
}

inline __m256 decode8(__m256i s, bool arith) {
    return arith ? decode8_arith(s) : decode8_lut(s);
}

inline bool use_arith() {
    // auto: arith — gathers serialize on most cores; the microbench revisits per tier.
    return g_decode_variant != DECODE_LUT;
}

// ---- decompress -----------------------------------------------------------------------
template <int R>
void decompress_impl(const uint16_t* packed, uint16_t* W, int64_t m, int64_t k) {
    const Geom g = Geom::make(m, k, R);
    const bool arith = use_arith();
    const int64_t blocks = g.tileCountM / 2;
    alignas(32) uint32_t states[4][8][32];
    alignas(32) float dec[8][32];
    for (int64_t p = 0; p < blocks; ++p) {
        for (int w = 0; w < 32; ++w) {
            const int64_t this_warp_k = g.k_per_block + (w < g.warp_rem ? 2 : 0);
            const uint16_t* base = packed + p * g.weight_row_step + (int64_t)w * 2 * g.utb;
            for (int64_t ki = 0; ki < this_warp_k; ++ki) {
                const uint16_t* buf = base + (ki / 2) * 2 * g.weight_step + (ki % 2) * g.utb;
                unpack_group_states<R>(buf, states);
                for (int subki = 0; subki < 2; ++subki) {
                    const int64_t k_tile =
                        4 * (int64_t)w + 2 * (ki % 2) + subki + (4 * 32) * (ki / 2);
                    for (int submi = 0; submi < 2; ++submi) {
                        const auto& st = states[kSlotMap[submi][subki]];
                        for (int j = 0; j < 8; ++j)
                            for (int q = 0; q < 4; ++q)
                                _mm256_store_ps(
                                    dec[j] + 8 * q,
                                    decode8(_mm256_load_si256(
                                                (const __m256i*)(st[j] + 8 * q)),
                                            arith));
                        const int64_t m_tile = p * 2 + submi;
                        for (int l = 0; l < 32; ++l) {
                            const int64_t r0 = m_tile * 16 + (l >> 2);
                            const int64_t c0 = k_tile * 16 + 2 * (l & 3);
                            uint16_t* w00 = W + r0 * k + c0;
                            uint16_t* w80 = W + (r0 + 8) * k + c0;
                            w00[0] = float_to_half_rn(dec[0][l]);
                            w00[1] = float_to_half_rn(dec[1][l]);
                            w80[0] = float_to_half_rn(dec[2][l]);
                            w80[1] = float_to_half_rn(dec[3][l]);
                            w00[8] = float_to_half_rn(dec[4][l]);
                            w00[9] = float_to_half_rn(dec[5][l]);
                            w80[8] = float_to_half_rn(dec[6][l]);
                            w80[9] = float_to_half_rn(dec[7][l]);
                        }
                    }
                }
            }
        }
    }
}

void decompress_avx2(const uint16_t* packed, uint16_t* W, int64_t m, int64_t k, int R) {
    switch (R) {
        case 1: decompress_impl<1>(packed, W, m, k); break;
        case 2: decompress_impl<2>(packed, W, m, k); break;
        case 3: decompress_impl<3>(packed, W, m, k); break;
        default: decompress_impl<4>(packed, W, m, k); break;
    }
}

// ---- fused GEMV -----------------------------------------------------------------------
/* Lane-parallel accumulation: acc_g / acc_g8 hold per-LANE partials (4 quarters x 8
 * lanes) for the g and g+8 row halves of one m_tile; each output row is the quad-tree
 * fold of its four t-lanes. x fragments: the column pattern {2t, 2t+1, 8+2t, 9+2t} has
 * period 4, so one 16-float tile slice serves all quarters via four constant VPERMPS. */
template <int R>
void matvec_impl(const uint16_t* packed, const float* x, float* y, int64_t m, int64_t k,
                 float wscale, bool accum, int64_t blk_begin, int64_t blk_end) {
    const Geom g = Geom::make(m, k, R);
    const bool arith = use_arith();
    const __m256i pidx0 = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
    const __m256i pidx1 = _mm256_setr_epi32(1, 3, 5, 7, 1, 3, 5, 7);
    alignas(32) uint32_t states[4][8][32];

    for (int64_t p = blk_begin; p < blk_end; ++p) {
        // acc[submi][half][quarter] as stack ymms; 2 submi x 2 half x 4 quarters
        alignas(32) float accbuf[2][2][4][8] = {};
        for (int w = 0; w < 32; ++w) {
            const int64_t this_warp_k = g.k_per_block + (w < g.warp_rem ? 2 : 0);
            const uint16_t* base = packed + p * g.weight_row_step + (int64_t)w * 2 * g.utb;
            for (int64_t ki = 0; ki < this_warp_k; ++ki) {
                const uint16_t* buf = base + (ki / 2) * 2 * g.weight_step + (ki % 2) * g.utb;
                unpack_group_states<R>(buf, states);
                for (int subki = 0; subki < 2; ++subki) {
                    const int64_t k_tile =
                        4 * (int64_t)w + 2 * (ki % 2) + subki + (4 * 32) * (ki / 2);
                    const float* xc = x + k_tile * 16;
                    const __m256 xlo = _mm256_loadu_ps(xc);
                    const __m256 xhi = _mm256_loadu_ps(xc + 8);
                    const __m256 xp0 = _mm256_permutevar8x32_ps(xlo, pidx0);
                    const __m256 xp1 = _mm256_permutevar8x32_ps(xlo, pidx1);
                    const __m256 xp2 = _mm256_permutevar8x32_ps(xhi, pidx0);
                    const __m256 xp3 = _mm256_permutevar8x32_ps(xhi, pidx1);
                    for (int submi = 0; submi < 2; ++submi) {
                        const auto& st = states[kSlotMap[submi][subki]];
                        for (int q = 0; q < 4; ++q) {
                            const auto ld = [&](int j) {
                                return _mm256_load_si256((const __m256i*)(st[j] + 8 * q));
                            };
                            __m256 ag = _mm256_load_ps(accbuf[submi][0][q]);
                            __m256 a8 = _mm256_load_ps(accbuf[submi][1][q]);
                            ag = _mm256_fmadd_ps(decode8(ld(0), arith), xp0, ag);
                            ag = _mm256_fmadd_ps(decode8(ld(1), arith), xp1, ag);
                            ag = _mm256_fmadd_ps(decode8(ld(4), arith), xp2, ag);
                            ag = _mm256_fmadd_ps(decode8(ld(5), arith), xp3, ag);
                            a8 = _mm256_fmadd_ps(decode8(ld(2), arith), xp0, a8);
                            a8 = _mm256_fmadd_ps(decode8(ld(3), arith), xp1, a8);
                            a8 = _mm256_fmadd_ps(decode8(ld(6), arith), xp2, a8);
                            a8 = _mm256_fmadd_ps(decode8(ld(7), arith), xp3, a8);
                            _mm256_store_ps(accbuf[submi][0][q], ag);
                            _mm256_store_ps(accbuf[submi][1][q], a8);
                        }
                    }
                }
            }
        }
        // Epilogue: lane quads -> rows. Quarter q covers global lanes 8q..8q+7, i.e.
        // rows 2q (lanes t=0..3) and 2q+1 (lanes t=4..7); fold (l0+l1)+(l2+l3).
        float* yp = y + p * 32;
        for (int submi = 0; submi < 2; ++submi) {
            for (int half = 0; half < 2; ++half) {
                for (int q = 0; q < 4; ++q) {
                    const float* a = accbuf[submi][half][q];
                    const float r_even = (a[0] + a[1]) + (a[2] + a[3]);
                    const float r_odd  = (a[4] + a[5]) + (a[6] + a[7]);
                    const int64_t row = submi * 16 + half * 8 + 2 * q;
                    const float ve = r_even * wscale;
                    const float vo = r_odd * wscale;
                    yp[row]     = accum ? yp[row] + ve : ve;
                    yp[row + 1] = accum ? yp[row + 1] + vo : vo;
                }
            }
        }
    }
}

void matvec_avx2(const uint16_t* packed, const float* x, float* y, int64_t m, int64_t k,
                 int R, float wscale, bool accum, int64_t blk_begin, int64_t blk_end) {
    switch (R) {
        case 1: matvec_impl<1>(packed, x, y, m, k, wscale, accum, blk_begin, blk_end); break;
        case 2: matvec_impl<2>(packed, x, y, m, k, wscale, accum, blk_begin, blk_end); break;
        case 3: matvec_impl<3>(packed, x, y, m, k, wscale, accum, blk_begin, blk_end); break;
        default: matvec_impl<4>(packed, x, y, m, k, wscale, accum, blk_begin, blk_end); break;
    }
}

/* Batched GEMM: decode each fragment vector ONCE, FMA it into every token's lane-parallel
 * accumulators. Per (b, lane) the FMA order (j = 0,1,4,5 into g; 2,3,6,7 into g+8, per
 * (w, ki, subki, submi)) and the quad-tree epilogue are IDENTICAL to matvec_impl — row b
 * of the GEMM is bit-identical to the GEMV on x[b]. accbuf is L1-resident (<=4 KB). */
template <int R>
void matmul_impl(const uint16_t* packed, const float* x, float* y, int64_t B, int64_t m,
                 int64_t k, float wscale, bool accum, int64_t blk_begin, int64_t blk_end) {
    const Geom g = Geom::make(m, k, R);
    const bool arith = use_arith();
    const __m256i pidx0 = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
    const __m256i pidx1 = _mm256_setr_epi32(1, 3, 5, 7, 1, 3, 5, 7);
    alignas(32) uint32_t states[4][8][32];
    alignas(32) float accbuf[8][2][2][4][8];

    for (int64_t p = blk_begin; p < blk_end; ++p) {
        for (int64_t b = 0; b < B; ++b)
            for (int s0 = 0; s0 < 2; ++s0)
                for (int h = 0; h < 2; ++h)
                    for (int q = 0; q < 4; ++q)
                        _mm256_store_ps(accbuf[b][s0][h][q], _mm256_setzero_ps());
        for (int w = 0; w < 32; ++w) {
            const int64_t this_warp_k = g.k_per_block + (w < g.warp_rem ? 2 : 0);
            const uint16_t* base = packed + p * g.weight_row_step + (int64_t)w * 2 * g.utb;
            for (int64_t ki = 0; ki < this_warp_k; ++ki) {
                const uint16_t* buf = base + (ki / 2) * 2 * g.weight_step + (ki % 2) * g.utb;
                unpack_group_states<R>(buf, states);
                for (int subki = 0; subki < 2; ++subki) {
                    const int64_t k_tile =
                        4 * (int64_t)w + 2 * (ki % 2) + subki + (4 * 32) * (ki / 2);
                    for (int submi = 0; submi < 2; ++submi) {
                        const auto& st = states[kSlotMap[submi][subki]];
                        for (int q = 0; q < 4; ++q) {
                            const auto ld = [&](int j) {
                                return _mm256_load_si256((const __m256i*)(st[j] + 8 * q));
                            };
                            const __m256 w0 = decode8(ld(0), arith);
                            const __m256 w1 = decode8(ld(1), arith);
                            const __m256 w2 = decode8(ld(2), arith);
                            const __m256 w3 = decode8(ld(3), arith);
                            const __m256 w4 = decode8(ld(4), arith);
                            const __m256 w5 = decode8(ld(5), arith);
                            const __m256 w6 = decode8(ld(6), arith);
                            const __m256 w7 = decode8(ld(7), arith);
                            for (int64_t b = 0; b < B; ++b) {
                                const float* xc = x + b * k + k_tile * 16;
                                const __m256 xlo = _mm256_loadu_ps(xc);
                                const __m256 xhi = _mm256_loadu_ps(xc + 8);
                                const __m256 xp0 = _mm256_permutevar8x32_ps(xlo, pidx0);
                                const __m256 xp1 = _mm256_permutevar8x32_ps(xlo, pidx1);
                                const __m256 xp2 = _mm256_permutevar8x32_ps(xhi, pidx0);
                                const __m256 xp3 = _mm256_permutevar8x32_ps(xhi, pidx1);
                                __m256 ag = _mm256_load_ps(accbuf[b][submi][0][q]);
                                __m256 a8 = _mm256_load_ps(accbuf[b][submi][1][q]);
                                ag = _mm256_fmadd_ps(w0, xp0, ag);
                                ag = _mm256_fmadd_ps(w1, xp1, ag);
                                ag = _mm256_fmadd_ps(w4, xp2, ag);
                                ag = _mm256_fmadd_ps(w5, xp3, ag);
                                a8 = _mm256_fmadd_ps(w2, xp0, a8);
                                a8 = _mm256_fmadd_ps(w3, xp1, a8);
                                a8 = _mm256_fmadd_ps(w6, xp2, a8);
                                a8 = _mm256_fmadd_ps(w7, xp3, a8);
                                _mm256_store_ps(accbuf[b][submi][0][q], ag);
                                _mm256_store_ps(accbuf[b][submi][1][q], a8);
                            }
                        }
                    }
                }
            }
        }
        for (int64_t b = 0; b < B; ++b) {
            float* yp = y + b * m + p * 32;
            for (int submi = 0; submi < 2; ++submi) {
                for (int half = 0; half < 2; ++half) {
                    for (int q = 0; q < 4; ++q) {
                        const float* a = accbuf[b][submi][half][q];
                        const float r_even = (a[0] + a[1]) + (a[2] + a[3]);
                        const float r_odd  = (a[4] + a[5]) + (a[6] + a[7]);
                        const int64_t row = submi * 16 + half * 8 + 2 * q;
                        const float ve = r_even * wscale;
                        const float vo = r_odd * wscale;
                        yp[row]     = accum ? yp[row] + ve : ve;
                        yp[row + 1] = accum ? yp[row + 1] + vo : vo;
                    }
                }
            }
        }
    }
}

void matmul_avx2(const uint16_t* packed, const float* x, float* y, int64_t B, int64_t m,
                 int64_t k, int R, float wscale, bool accum,
                 int64_t blk_begin, int64_t blk_end) {
    switch (R) {
        case 1: matmul_impl<1>(packed, x, y, B, m, k, wscale, accum, blk_begin, blk_end); break;
        case 2: matmul_impl<2>(packed, x, y, B, m, k, wscale, accum, blk_begin, blk_end); break;
        case 3: matmul_impl<3>(packed, x, y, B, m, k, wscale, accum, blk_begin, blk_end); break;
        default: matmul_impl<4>(packed, x, y, B, m, k, wscale, accum, blk_begin, blk_end); break;
    }
}

const Kernels kAvx2 = { decompress_avx2, matvec_avx2, matmul_avx2 };

}  // namespace

const Kernels* avx2_kernels() { return &kAvx2; }

}  // namespace glq_cpu

#pragma GCC pop_options
#else   // non-x86 build: no AVX2 tier
namespace glq_cpu {
struct Kernels;
const Kernels* avx2_kernels() { return nullptr; }
}  // namespace glq_cpu
#endif
