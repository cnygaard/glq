/* glq_trellis_cpu_avx512.cpp — AVX512F+BW+VL tier: the AVX2 tier widened to 16 lanes.
 *
 * Same in-source target pragma discipline, same scalar bit-unpack staging, same decode
 * variants and fixed accumulation orders (per-lane partials + the (l0+l1)+(l2+l3) quad
 * tree), so every determinism/parity contract carries over verbatim. What AVX-512 buys
 * beyond width: VPMOVDW replaces the AVX2 pack/extract dance for u32->u16, one 16-float
 * load + VPERMPS-var serves a whole k-tile column slice, and a 32x32 window-group is two
 * 16-lane sweeps instead of four 8-lane ones.
 */
#if defined(__x86_64__) || defined(_M_X64)
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512vl,avx512dq,f16c,fma")

#include <immintrin.h>

#include "glq_trellis_cpu.hpp"
#include "glq_trellis_layout.hpp"

namespace glq_cpu {
namespace {

inline __m512 decode16_lut(__m512i idx) {
    return _mm512_i32gather_ps(idx, g_lut32, 4);
}

inline __m512 decode16_arith(__m512i s) {
    const __m512i h = _mm512_add_epi32(_mm512_mullo_epi32(s, _mm512_set1_epi32(89226354)),
                                       _mm512_set1_epi32(64248484));
    const __m512i r = _mm512_xor_si512(
        _mm512_and_si512(h, _mm512_set1_epi32((int)0x8FFF8FFFu)),
        _mm512_set1_epi32((int)0x3B603B60u));
    const __m256i lo16 = _mm512_cvtepi32_epi16(_mm512_and_si512(r, _mm512_set1_epi32(0xFFFF)));
    const __m256i hi16 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(r, 16));
    const __m512 sum = _mm512_add_ps(_mm512_cvtph_ps(hi16), _mm512_cvtph_ps(lo16));
    // exact fp32 sum -> oracle fp16 (RN) -> exact fp32 widening (the load-bearing round)
    return _mm512_cvtph_ps(_mm512_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT));
}

inline __m512 decode16(__m512i s, bool arith) {
    return arith ? decode16_arith(s) : decode16_lut(s);
}

inline bool use_arith() { return g_decode_variant != DECODE_LUT; }

// x-fragment permutes: lanes 16h..16h+15 have t = l&3 -> a period-4 index pattern.
inline __m512i pidx(int base) {
    return _mm512_setr_epi32(base, base + 2, base + 4, base + 6,
                             base, base + 2, base + 4, base + 6,
                             base, base + 2, base + 4, base + 6,
                             base, base + 2, base + 4, base + 6);
}

// ---- decompress -----------------------------------------------------------------------
template <int R>
void decompress_impl(const uint16_t* packed, uint16_t* W, int64_t m, int64_t k) {
    const Geom g = Geom::make(m, k, R);
    const bool arith = use_arith();
    const int64_t blocks = g.tileCountM / 2;
    alignas(64) uint32_t states[4][8][32];
    alignas(64) float dec[8][32];
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
                            for (int h = 0; h < 2; ++h)
                                _mm512_store_ps(
                                    dec[j] + 16 * h,
                                    decode16(_mm512_load_si512(
                                                 (const void*)(st[j] + 16 * h)),
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

void decompress_avx512(const uint16_t* packed, uint16_t* W, int64_t m, int64_t k, int R) {
    switch (R) {
        case 1: decompress_impl<1>(packed, W, m, k); break;
        case 2: decompress_impl<2>(packed, W, m, k); break;
        case 3: decompress_impl<3>(packed, W, m, k); break;
        default: decompress_impl<4>(packed, W, m, k); break;
    }
}

// ---- fused GEMV -----------------------------------------------------------------------
template <int R>
void matvec_impl(const uint16_t* packed, const float* x, float* y, int64_t m, int64_t k,
                 float wscale, bool accum, int64_t blk_begin, int64_t blk_end) {
    const Geom g = Geom::make(m, k, R);
    const bool arith = use_arith();
    const __m512i px0 = pidx(0), px1 = pidx(1), px2 = pidx(8), px3 = pidx(9);
    alignas(64) uint32_t states[4][8][32];

    for (int64_t p = blk_begin; p < blk_end; ++p) {
        alignas(64) float accbuf[2][2][2][16] = {};   // [submi][rowhalf][lanehalf][16]
        for (int w = 0; w < 32; ++w) {
            const int64_t this_warp_k = g.k_per_block + (w < g.warp_rem ? 2 : 0);
            const uint16_t* base = packed + p * g.weight_row_step + (int64_t)w * 2 * g.utb;
            for (int64_t ki = 0; ki < this_warp_k; ++ki) {
                const uint16_t* buf = base + (ki / 2) * 2 * g.weight_step + (ki % 2) * g.utb;
                unpack_group_states<R>(buf, states);
                for (int subki = 0; subki < 2; ++subki) {
                    const int64_t k_tile =
                        4 * (int64_t)w + 2 * (ki % 2) + subki + (4 * 32) * (ki / 2);
                    const __m512 xt = _mm512_loadu_ps(x + k_tile * 16);
                    const __m512 xp0 = _mm512_permutexvar_ps(px0, xt);
                    const __m512 xp1 = _mm512_permutexvar_ps(px1, xt);
                    const __m512 xp2 = _mm512_permutexvar_ps(px2, xt);
                    const __m512 xp3 = _mm512_permutexvar_ps(px3, xt);
                    for (int submi = 0; submi < 2; ++submi) {
                        const auto& st = states[kSlotMap[submi][subki]];
                        for (int h = 0; h < 2; ++h) {
                            const auto ld = [&](int j) {
                                return _mm512_load_si512((const void*)(st[j] + 16 * h));
                            };
                            __m512 ag = _mm512_load_ps(accbuf[submi][0][h]);
                            __m512 a8 = _mm512_load_ps(accbuf[submi][1][h]);
                            ag = _mm512_fmadd_ps(decode16(ld(0), arith), xp0, ag);
                            ag = _mm512_fmadd_ps(decode16(ld(1), arith), xp1, ag);
                            ag = _mm512_fmadd_ps(decode16(ld(4), arith), xp2, ag);
                            ag = _mm512_fmadd_ps(decode16(ld(5), arith), xp3, ag);
                            a8 = _mm512_fmadd_ps(decode16(ld(2), arith), xp0, a8);
                            a8 = _mm512_fmadd_ps(decode16(ld(3), arith), xp1, a8);
                            a8 = _mm512_fmadd_ps(decode16(ld(6), arith), xp2, a8);
                            a8 = _mm512_fmadd_ps(decode16(ld(7), arith), xp3, a8);
                            _mm512_store_ps(accbuf[submi][0][h], ag);
                            _mm512_store_ps(accbuf[submi][1][h], a8);
                        }
                    }
                }
            }
        }
        // Epilogue: lane-half h covers lanes 16h..16h+15 -> rows 4h..4h+3, four t-lanes
        // per row; fold (l0+l1)+(l2+l3), identical tree to every other tier.
        float* yp = y + p * 32;
        for (int submi = 0; submi < 2; ++submi) {
            for (int half = 0; half < 2; ++half) {
                for (int h = 0; h < 2; ++h) {
                    const float* a = accbuf[submi][half][h];
                    for (int rq = 0; rq < 4; ++rq) {
                        const float* q4 = a + 4 * rq;
                        const float v = ((q4[0] + q4[1]) + (q4[2] + q4[3])) * wscale;
                        const int64_t row = submi * 16 + half * 8 + 4 * h + rq;
                        yp[row] = accum ? yp[row] + v : v;
                    }
                }
            }
        }
    }
}

void matvec_avx512(const uint16_t* packed, const float* x, float* y, int64_t m, int64_t k,
                   int R, float wscale, bool accum, int64_t blk_begin, int64_t blk_end) {
    switch (R) {
        case 1: matvec_impl<1>(packed, x, y, m, k, wscale, accum, blk_begin, blk_end); break;
        case 2: matvec_impl<2>(packed, x, y, m, k, wscale, accum, blk_begin, blk_end); break;
        case 3: matvec_impl<3>(packed, x, y, m, k, wscale, accum, blk_begin, blk_end); break;
        default: matvec_impl<4>(packed, x, y, m, k, wscale, accum, blk_begin, blk_end); break;
    }
}

// ---- batched GEMM ---------------------------------------------------------------------
template <int R>
void matmul_impl(const uint16_t* packed, const float* x, float* y, int64_t B, int64_t m,
                 int64_t k, float wscale, bool accum, int64_t blk_begin, int64_t blk_end) {
    const Geom g = Geom::make(m, k, R);
    const bool arith = use_arith();
    const __m512i px0 = pidx(0), px1 = pidx(1), px2 = pidx(8), px3 = pidx(9);
    alignas(64) uint32_t states[4][8][32];
    alignas(64) float accbuf[8][2][2][2][16];

    for (int64_t p = blk_begin; p < blk_end; ++p) {
        for (int64_t b = 0; b < B; ++b)
            for (int s0 = 0; s0 < 2; ++s0)
                for (int rh = 0; rh < 2; ++rh)
                    for (int h = 0; h < 2; ++h)
                        _mm512_store_ps(accbuf[b][s0][rh][h], _mm512_setzero_ps());
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
                        for (int h = 0; h < 2; ++h) {
                            const auto ld = [&](int j) {
                                return _mm512_load_si512((const void*)(st[j] + 16 * h));
                            };
                            const __m512 w0 = decode16(ld(0), arith);
                            const __m512 w1 = decode16(ld(1), arith);
                            const __m512 w2 = decode16(ld(2), arith);
                            const __m512 w3 = decode16(ld(3), arith);
                            const __m512 w4 = decode16(ld(4), arith);
                            const __m512 w5 = decode16(ld(5), arith);
                            const __m512 w6 = decode16(ld(6), arith);
                            const __m512 w7 = decode16(ld(7), arith);
                            for (int64_t b = 0; b < B; ++b) {
                                const __m512 xt = _mm512_loadu_ps(x + b * k + k_tile * 16);
                                const __m512 xp0 = _mm512_permutexvar_ps(px0, xt);
                                const __m512 xp1 = _mm512_permutexvar_ps(px1, xt);
                                const __m512 xp2 = _mm512_permutexvar_ps(px2, xt);
                                const __m512 xp3 = _mm512_permutexvar_ps(px3, xt);
                                __m512 ag = _mm512_load_ps(accbuf[b][submi][0][h]);
                                __m512 a8 = _mm512_load_ps(accbuf[b][submi][1][h]);
                                ag = _mm512_fmadd_ps(w0, xp0, ag);
                                ag = _mm512_fmadd_ps(w1, xp1, ag);
                                ag = _mm512_fmadd_ps(w4, xp2, ag);
                                ag = _mm512_fmadd_ps(w5, xp3, ag);
                                a8 = _mm512_fmadd_ps(w2, xp0, a8);
                                a8 = _mm512_fmadd_ps(w3, xp1, a8);
                                a8 = _mm512_fmadd_ps(w6, xp2, a8);
                                a8 = _mm512_fmadd_ps(w7, xp3, a8);
                                _mm512_store_ps(accbuf[b][submi][0][h], ag);
                                _mm512_store_ps(accbuf[b][submi][1][h], a8);
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
                    for (int h = 0; h < 2; ++h) {
                        const float* a = accbuf[b][submi][half][h];
                        for (int rq = 0; rq < 4; ++rq) {
                            const float* q4 = a + 4 * rq;
                            const float v = ((q4[0] + q4[1]) + (q4[2] + q4[3])) * wscale;
                            const int64_t row = submi * 16 + half * 8 + 4 * h + rq;
                            yp[row] = accum ? yp[row] + v : v;
                        }
                    }
                }
            }
        }
    }
}

void matmul_avx512(const uint16_t* packed, const float* x, float* y, int64_t B, int64_t m,
                   int64_t k, int R, float wscale, bool accum,
                   int64_t blk_begin, int64_t blk_end) {
    switch (R) {
        case 1: matmul_impl<1>(packed, x, y, B, m, k, wscale, accum, blk_begin, blk_end); break;
        case 2: matmul_impl<2>(packed, x, y, B, m, k, wscale, accum, blk_begin, blk_end); break;
        case 3: matmul_impl<3>(packed, x, y, B, m, k, wscale, accum, blk_begin, blk_end); break;
        default: matmul_impl<4>(packed, x, y, B, m, k, wscale, accum, blk_begin, blk_end); break;
    }
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
