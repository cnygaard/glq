/* glq_trellis_layout.hpp — the kernel-layout walk for the 3INST (V=1) packed stream.
 *
 * Faithful C++ port of the numpy mirror in tests/test_trellis_3inst_kernel.py
 * (_load_chunks :179, _v1_states :201, _mirror_decompress :221), which is gated
 * bit-exact against glq/trellis.py decode_layer for K=1..4. Variable names follow the
 * mirror / the CUDA kernel so the three can be diffed by eye.
 *
 * Geometry, for R = K bits/weight (V=1):
 *   u16_per_tile   = 16R          (one 16x16 tile: 256 weights * R bits)
 *   utb            = 64R          (a 2x2 tile block, u16 units)
 *   weight_step    = 32*utb       (one full 32-warp sweep)
 *   weight_row_step= (k/16)*32R   (one 32-row m-pair block)
 * Lane l's window at (p, w, ki):
 *   addr_u16(l) = p*WRS + w*2*utb + (ki/2)*2*WS + (ki%2)*utb + l*(utb/32)
 * The 32 lanes of one (p,w,ki) window-group are CONTIGUOUS (utb u16 total, gapless),
 * and consecutive (ki, w) walk strictly ascending addresses — stream-friendly.
 *
 * Each lane block (2R u16, little-endian u32 assembly) splits into 4 chunks of 8R bits
 * (window slots x,y,z,w = (submi,subki) = (0,0),(1,0),(0,1),(1,1)). Tail-biting: the 16
 * stream bits below chunk l live at the TOP of chunk (l+1)%32 (R>=2); R=1's 8-bit chunks
 * span TWO lanes — the known trap. States: s_j = (Ext >> (8R - R*j)) & 0xFFFF, j=0..7,
 * Ext = (chunk << 16) | cont.
 *
 * Scatter (tile (m_tile, k_tile), lane l, g=l>>2, t=l&3, r0=m_tile*16+g, c0=k_tile*16+2t):
 *   (s0,s1)->(r0,c0..c0+1)   (s2,s3)->(r0+8,c0..c0+1)
 *   (s4,s5)->(r0,c0+8..c0+9) (s6,s7)->(r0+8,c0+8..c0+9)
 */
#pragma once

#include <cstdint>

namespace glq_cpu {

struct Geom {
    int64_t tileCountM, tileCountK;
    int64_t u16_per_tile, utb, weight_step, weight_row_step;
    int64_t k_per_block, warp_rem;   // this_warp_k = k_per_block + (w < warp_rem ? 2 : 0)
    static Geom make(int64_t m, int64_t k, int R) {
        Geom g;
        g.tileCountM = m / 16;
        g.tileCountK = k / 16;
        g.u16_per_tile = 16 * R;
        g.utb = g.u16_per_tile * 4;
        g.weight_step = 32 * g.utb;
        g.weight_row_step = g.tileCountK * g.u16_per_tile * 2;
        g.k_per_block = g.tileCountK / (32 * 4) * 2;
        g.warp_rem = (g.tileCountK % (32 * 4)) / 4;
        return g;
    }
};

// Extract the four 8R-bit window chunks of lane l from the window-group base pointer
// (buf = packed_u16 + addr_u16(lane 0); lane stride = 2R u16). Mirrors _load_chunks.
template <int R>
inline void lane_chunks(const uint16_t* buf, int l, uint32_t out[4]) {
    const uint16_t* p = buf + (int64_t)l * 2 * R;
    if (R == 1) {
        const uint32_t r1 = (uint32_t)p[0] | ((uint32_t)p[1] << 16);
        out[0] = r1 & 0xFF; out[1] = (r1 >> 8) & 0xFF;
        out[2] = (r1 >> 16) & 0xFF; out[3] = r1 >> 24;
    } else if (R == 2) {
        out[0] = p[0]; out[1] = p[1]; out[2] = p[2]; out[3] = p[3];
    } else if (R == 3) {
        const uint32_t r1 = (uint32_t)p[0] | ((uint32_t)p[1] << 16);
        const uint32_t r2 = (uint32_t)p[2] | ((uint32_t)p[3] << 16);
        const uint32_t r3 = (uint32_t)p[4] | ((uint32_t)p[5] << 16);
        out[0] = r1 & 0xFFFFFF;
        out[1] = ((r1 >> 24) | (r2 << 8)) & 0xFFFFFF;
        out[2] = ((r2 >> 16) | (r3 << 16)) & 0xFFFFFF;
        out[3] = r3 >> 8;
    } else {  // R == 4
        out[0] = (uint32_t)p[0] | ((uint32_t)p[1] << 16);
        out[1] = (uint32_t)p[2] | ((uint32_t)p[3] << 16);
        out[2] = (uint32_t)p[4] | ((uint32_t)p[5] << 16);
        out[3] = (uint32_t)p[6] | ((uint32_t)p[7] << 16);
    }
}

// Tail-biting continuation for window slot `chunk[32]` (one slot's 32-lane stream).
// Mirrors _v1_states' cont computation, including the R=1 two-lane form.
template <int R>
inline uint32_t lane_cont(const uint32_t chunk[32], int l) {
    if constexpr (R == 1) {
        return ((chunk[(l + 1) & 31] & 0xFF) << 8) | (chunk[(l + 2) & 31] & 0xFF);
    } else {
        return (chunk[(l + 1) & 31] >> (8 * R - 16)) & 0xFFFF;
    }
}

// The eight 16-bit states of lane l's window chunk. Mirrors _v1_states.
template <int R>
inline void lane_states(uint32_t chunk, uint32_t cont, uint16_t s[8]) {
    const uint64_t ext = ((uint64_t)chunk << 16) | cont;
    for (int j = 0; j < 8; ++j)
        s[j] = (uint16_t)((ext >> (8 * R - R * j)) & 0xFFFF);
}

// (submi, subki) -> window chunk slot, mirroring _KEY = {(0,0):x,(1,0):y,(0,1):z,(1,1):w}.
constexpr int kSlotMap[2][2] = {{0, 2}, {1, 3}};

// Unpack one window-group into states[slot][j][lane] (u32 storage so SIMD tiers can load
// vectors straight from the staging array). Shared by every tier — one copy, no drift.
template <int R>
inline void unpack_group_states(const uint16_t* buf, uint32_t states[4][8][32]) {
    uint32_t chunks[4][32];
    for (int l = 0; l < 32; ++l) {
        uint32_t c[4];
        lane_chunks<R>(buf, l, c);
        chunks[0][l] = c[0]; chunks[1][l] = c[1]; chunks[2][l] = c[2]; chunks[3][l] = c[3];
    }
    for (int slot = 0; slot < 4; ++slot) {
        for (int l = 0; l < 32; ++l) {
            uint16_t s[8];
            lane_states<R>(chunks[slot][l], lane_cont<R>(chunks[slot], l), s);
            for (int j = 0; j < 8; ++j) states[slot][j][l] = s[j];
        }
    }
}

}  // namespace glq_cpu
