/* glq_trellis_cpu_zmm.inl — shared 16-lane (zmm) kernel bodies for the AVX-512 tiers.
 *
 * Include INSIDE a TU's target-pragma'd `namespace glq_cpu { namespace { ... } }` after
 * defining a decode policy:
 *
 *     struct Decode { static __m512 decode16(__m512i states, bool arith); };
 *
 * The avx512 and avx512fp16 tiers differ ONLY in that policy (double-rounded F16C
 * converts vs native VADDPH); everything else — the unpack staging, the fixed FMA
 * orders, the (l0+l1)+(l2+l3) epilogue tree — is one copy here, so the two tiers cannot
 * drift and every determinism/parity contract proven for one holds for the other.
 * No includes of its own: immintrin.h and the layout/cpu headers come from the TU.
 */

inline bool zmm_use_arith() { return g_decode_variant != DECODE_LUT; }

inline __m512 zmm_lut_gather(__m512i idx) {
    return _mm512_i32gather_ps(idx, g_lut32, 4);
}

// x-fragment permutes: lanes 16h..16h+15 have t = l&3 -> a period-4 index pattern.
inline __m512i zmm_pidx(int base) {
    return _mm512_setr_epi32(base, base + 2, base + 4, base + 6,
                             base, base + 2, base + 4, base + 6,
                             base, base + 2, base + 4, base + 6,
                             base, base + 2, base + 4, base + 6);
}

// ---- decompress -----------------------------------------------------------------------
template <typename D, int R>
void zmm_decompress_impl(const uint16_t* packed, uint16_t* W, int64_t m, int64_t k) {
    const Geom g = Geom::make(m, k, R);
    const bool arith = zmm_use_arith();
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
                                    D::decode16(_mm512_load_si512(
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

// ---- fused GEMV -----------------------------------------------------------------------
template <typename D, int R>
void zmm_matvec_impl(const uint16_t* packed, const float* x, float* y, int64_t m,
                     int64_t k, float wscale, bool accum,
                     int64_t blk_begin, int64_t blk_end) {
    const Geom g = Geom::make(m, k, R);
    const bool arith = zmm_use_arith();
    const __m512i px0 = zmm_pidx(0), px1 = zmm_pidx(1);
    const __m512i px2 = zmm_pidx(8), px3 = zmm_pidx(9);
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
                            ag = _mm512_fmadd_ps(D::decode16(ld(0), arith), xp0, ag);
                            ag = _mm512_fmadd_ps(D::decode16(ld(1), arith), xp1, ag);
                            ag = _mm512_fmadd_ps(D::decode16(ld(4), arith), xp2, ag);
                            ag = _mm512_fmadd_ps(D::decode16(ld(5), arith), xp3, ag);
                            a8 = _mm512_fmadd_ps(D::decode16(ld(2), arith), xp0, a8);
                            a8 = _mm512_fmadd_ps(D::decode16(ld(3), arith), xp1, a8);
                            a8 = _mm512_fmadd_ps(D::decode16(ld(6), arith), xp2, a8);
                            a8 = _mm512_fmadd_ps(D::decode16(ld(7), arith), xp3, a8);
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

// ---- batched GEMM ---------------------------------------------------------------------
template <typename D, int R>
void zmm_matmul_impl(const uint16_t* packed, const float* x, float* y, int64_t B,
                     int64_t m, int64_t k, float wscale, bool accum,
                     int64_t blk_begin, int64_t blk_end) {
    const Geom g = Geom::make(m, k, R);
    const bool arith = zmm_use_arith();
    const __m512i px0 = zmm_pidx(0), px1 = zmm_pidx(1);
    const __m512i px2 = zmm_pidx(8), px3 = zmm_pidx(9);
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
                            const __m512 w0 = D::decode16(ld(0), arith);
                            const __m512 w1 = D::decode16(ld(1), arith);
                            const __m512 w2 = D::decode16(ld(2), arith);
                            const __m512 w3 = D::decode16(ld(3), arith);
                            const __m512 w4 = D::decode16(ld(4), arith);
                            const __m512 w5 = D::decode16(ld(5), arith);
                            const __m512 w6 = D::decode16(ld(6), arith);
                            const __m512 w7 = D::decode16(ld(7), arith);
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

// ---- R-dispatch wrappers (vtable-signature functions) ---------------------------------
template <typename D>
void zmm_decompress(const uint16_t* packed, uint16_t* W, int64_t m, int64_t k, int R) {
    switch (R) {
        case 1: zmm_decompress_impl<D, 1>(packed, W, m, k); break;
        case 2: zmm_decompress_impl<D, 2>(packed, W, m, k); break;
        case 3: zmm_decompress_impl<D, 3>(packed, W, m, k); break;
        default: zmm_decompress_impl<D, 4>(packed, W, m, k); break;
    }
}

template <typename D>
void zmm_matvec(const uint16_t* packed, const float* x, float* y, int64_t m, int64_t k,
                int R, float wscale, bool accum, int64_t blk_begin, int64_t blk_end) {
    switch (R) {
        case 1: zmm_matvec_impl<D, 1>(packed, x, y, m, k, wscale, accum, blk_begin, blk_end); break;
        case 2: zmm_matvec_impl<D, 2>(packed, x, y, m, k, wscale, accum, blk_begin, blk_end); break;
        case 3: zmm_matvec_impl<D, 3>(packed, x, y, m, k, wscale, accum, blk_begin, blk_end); break;
        default: zmm_matvec_impl<D, 4>(packed, x, y, m, k, wscale, accum, blk_begin, blk_end); break;
    }
}

template <typename D>
void zmm_matmul(const uint16_t* packed, const float* x, float* y, int64_t B, int64_t m,
                int64_t k, int R, float wscale, bool accum,
                int64_t blk_begin, int64_t blk_end) {
    switch (R) {
        case 1: zmm_matmul_impl<D, 1>(packed, x, y, B, m, k, wscale, accum, blk_begin, blk_end); break;
        case 2: zmm_matmul_impl<D, 2>(packed, x, y, B, m, k, wscale, accum, blk_begin, blk_end); break;
        case 3: zmm_matmul_impl<D, 3>(packed, x, y, B, m, k, wscale, accum, blk_begin, blk_end); break;
        default: zmm_matmul_impl<D, 4>(packed, x, y, B, m, k, wscale, accum, blk_begin, blk_end); break;
    }
}
