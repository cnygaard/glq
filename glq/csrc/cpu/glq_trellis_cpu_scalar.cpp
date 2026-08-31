/* glq_trellis_cpu_scalar.cpp — portable scalar tier: the C++ port of the numpy mirror.
 *
 * No intrinsics, no ISA assumptions: this tier is the aarch64/any-CPU fallback AND the
 * in-process oracle every SIMD tier is gated against (on top of the Python decode_layer
 * gate). Decode goes through the shared LUT (g_lut16), which bakes in the fp16-rounding
 * of the oracle's fp16 add — see glq_trellis_cpu.hpp::float_to_half_rn.
 */
#include "glq_trellis_cpu.hpp"
#include "glq_trellis_layout.hpp"

namespace glq_cpu {
namespace {

// (submi, subki) -> window chunk slot, mirroring _KEY = {(0,0):x,(1,0):y,(0,1):z,(1,1):w}.
constexpr int kSlot[2][2] = {{0, 2}, {1, 3}};   // kSlot[submi][subki]

template <int R>
void decompress_impl(const uint16_t* packed, uint16_t* W, int64_t m, int64_t k) {
    const Geom g = Geom::make(m, k, R);
    const int64_t blocks = g.tileCountM / 2;
    for (int64_t p = 0; p < blocks; ++p) {
        for (int w = 0; w < 32; ++w) {
            const int64_t this_warp_k = g.k_per_block + (w < g.warp_rem ? 2 : 0);
            const uint16_t* base = packed + p * g.weight_row_step + (int64_t)w * 2 * g.utb;
            for (int64_t ki = 0; ki < this_warp_k; ++ki) {
                const uint16_t* buf =
                    base + (ki / 2) * 2 * g.weight_step + (ki % 2) * g.utb;
                // De-interleave the window-group into per-slot 32-lane chunk arrays.
                uint32_t chunks[4][32];
                for (int l = 0; l < 32; ++l) {
                    uint32_t c[4];
                    lane_chunks<R>(buf, l, c);
                    chunks[0][l] = c[0]; chunks[1][l] = c[1];
                    chunks[2][l] = c[2]; chunks[3][l] = c[3];
                }
                for (int subki = 0; subki < 2; ++subki) {
                    const int64_t k_tile =
                        4 * (int64_t)w + 2 * (ki % 2) + subki + (4 * 32) * (ki / 2);
                    for (int submi = 0; submi < 2; ++submi) {
                        const uint32_t* ch = chunks[kSlot[submi][subki]];
                        const int64_t m_tile = p * 2 + submi;
                        for (int l = 0; l < 32; ++l) {
                            uint16_t s[8];
                            lane_states<R>(ch[l], lane_cont<R>(ch, l), s);
                            const int64_t r0 = m_tile * 16 + (l >> 2);
                            const int64_t c0 = k_tile * 16 + 2 * (l & 3);
                            uint16_t* w00 = W + r0 * k + c0;
                            uint16_t* w80 = W + (r0 + 8) * k + c0;
                            w00[0] = g_lut16[s[0]]; w00[1] = g_lut16[s[1]];
                            w80[0] = g_lut16[s[2]]; w80[1] = g_lut16[s[3]];
                            w00[8] = g_lut16[s[4]]; w00[9] = g_lut16[s[5]];
                            w80[8] = g_lut16[s[6]]; w80[9] = g_lut16[s[7]];
                        }
                    }
                }
            }
        }
    }
}

void decompress_scalar(const uint16_t* packed, uint16_t* W, int64_t m, int64_t k, int R) {
    switch (R) {
        case 1: decompress_impl<1>(packed, W, m, k); break;
        case 2: decompress_impl<2>(packed, W, m, k); break;
        case 3: decompress_impl<3>(packed, W, m, k); break;
        default: decompress_impl<4>(packed, W, m, k); break;
    }
}

/* Fused GEMV over m-pair blocks [blk_begin, blk_end). Accumulation order per output row
 * is FIXED (w ascending, ki ascending, subki, submi, lane, then the four column terms
 * c0, c0+1, c0+8, c0+9) and rows are never split across calls — deterministic and
 * thread-count independent by construction. */
template <int R>
void matvec_impl(const uint16_t* packed, const float* x, float* y, int64_t m, int64_t k,
                 float wscale, bool accum, int64_t blk_begin, int64_t blk_end) {
    const Geom g = Geom::make(m, k, R);
    for (int64_t p = blk_begin; p < blk_end; ++p) {
        float acc[32] = {};                                 // rows 32p .. 32p+31
        for (int w = 0; w < 32; ++w) {
            const int64_t this_warp_k = g.k_per_block + (w < g.warp_rem ? 2 : 0);
            const uint16_t* base = packed + p * g.weight_row_step + (int64_t)w * 2 * g.utb;
            for (int64_t ki = 0; ki < this_warp_k; ++ki) {
                const uint16_t* buf =
                    base + (ki / 2) * 2 * g.weight_step + (ki % 2) * g.utb;
                uint32_t chunks[4][32];
                for (int l = 0; l < 32; ++l) {
                    uint32_t c[4];
                    lane_chunks<R>(buf, l, c);
                    chunks[0][l] = c[0]; chunks[1][l] = c[1];
                    chunks[2][l] = c[2]; chunks[3][l] = c[3];
                }
                for (int subki = 0; subki < 2; ++subki) {
                    const int64_t k_tile =
                        4 * (int64_t)w + 2 * (ki % 2) + subki + (4 * 32) * (ki / 2);
                    const float* xc = x + k_tile * 16;
                    for (int submi = 0; submi < 2; ++submi) {
                        const uint32_t* ch = chunks[kSlot[submi][subki]];
                        float* arow = acc + submi * 16;     // rows of m_tile 2p+submi
                        for (int l = 0; l < 32; ++l) {
                            uint16_t s[8];
                            lane_states<R>(ch[l], lane_cont<R>(ch, l), s);
                            const int gi = l >> 2;
                            const int c0 = 2 * (l & 3);
                            arow[gi] += g_lut32[s[0]] * xc[c0]
                                      + g_lut32[s[1]] * xc[c0 + 1]
                                      + g_lut32[s[4]] * xc[c0 + 8]
                                      + g_lut32[s[5]] * xc[c0 + 9];
                            arow[gi + 8] += g_lut32[s[2]] * xc[c0]
                                          + g_lut32[s[3]] * xc[c0 + 1]
                                          + g_lut32[s[6]] * xc[c0 + 8]
                                          + g_lut32[s[7]] * xc[c0 + 9];
                        }
                    }
                }
            }
        }
        float* yp = y + p * 32;
        for (int r = 0; r < 32; ++r) {
            const float v = acc[r] * wscale;
            yp[r] = accum ? yp[r] + v : v;
        }
    }
}

void matvec_scalar(const uint16_t* packed, const float* x, float* y, int64_t m, int64_t k,
                   int R, float wscale, bool accum, int64_t blk_begin, int64_t blk_end) {
    switch (R) {
        case 1: matvec_impl<1>(packed, x, y, m, k, wscale, accum, blk_begin, blk_end); break;
        case 2: matvec_impl<2>(packed, x, y, m, k, wscale, accum, blk_begin, blk_end); break;
        case 3: matvec_impl<3>(packed, x, y, m, k, wscale, accum, blk_begin, blk_end); break;
        default: matvec_impl<4>(packed, x, y, m, k, wscale, accum, blk_begin, blk_end); break;
    }
}

const Kernels kScalar = { decompress_scalar, matvec_scalar };

}  // namespace

const Kernels* scalar_kernels() { return &kScalar; }

}  // namespace glq_cpu
