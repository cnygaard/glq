/* glq_trellis_cpu.hpp — CPU fused-decode for the 3INST trellis codebook: shared decls.
 *
 * Tier model: one vtable of kernel entry points per ISA tier. Tiers are compiled in
 * separate TUs behind `#pragma GCC target` (NOT global -m flags: the JIT build rung has
 * no per-file flags, and letting the compiler emit AVX-512 into baseline code paths is a
 * SIGILL on AVX2 machines). A tier is "available" only if BOTH its TU was compiled in
 * (vtable slot non-null) AND the CPU reports the feature bits. `scalar` always exists —
 * it is the portable C++ port of the numpy mirror in tests/test_trellis_3inst_kernel.py
 * and doubles as the in-process oracle for the SIMD tiers.
 */
#pragma once

#include <cstdint>
#include <cstring>

namespace glq_cpu {

// ---- portable fp16 <-> fp32 (bit-exact, no F16C dependency — the scalar tier must run
// on any CPU, and the LUT build must not depend on the host ISA) ----------------------
inline float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1F;
    const uint32_t man  = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;                                   // +-0
        } else {                                           // subnormal: normalize
            uint32_t e = 127 - 15 + 1;
            uint32_t m = man;
            while (!(m & 0x400)) { m <<= 1; --e; }
            bits = sign | (e << 23) | ((m & 0x3FF) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);           // inf / nan
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// Round-to-nearest-even fp32 -> fp16. Load-bearing for the decode oracle: the fp32 sum
// of the two decoded halves is EXACT, but it differs from decode_3inst's fp16 add on
// 27,707/65,536 states unless rounded back to fp16 (RN). Verified exhaustively.
inline uint16_t float_to_half_rn(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    x &= 0x7FFFFFFFu;
    if (x >= 0x47800000u)                                  // overflow / inf / nan
        return (uint16_t)(sign | (x > 0x7F800000u ? 0x7E00u : 0x7C00u));
    if (x < 0x38800000u) {                                 // subnormal fp16 (or zero)
        if (x < 0x33000000u) return (uint16_t)sign;        // rounds to +-0
        const uint32_t shift = 126 - (x >> 23);
        uint32_t m = (x & 0x7FFFFFu) | 0x800000u;
        const uint32_t q = m >> (shift + 13);
        const uint32_t rem = m & ((1u << (shift + 13)) - 1);
        const uint32_t half = 1u << (shift + 12);
        uint32_t r = q + ((rem > half) || (rem == half && (q & 1)));
        return (uint16_t)(sign | r);
    }
    const uint32_t e = ((x >> 23) - 112) << 10;
    const uint32_t m = (x >> 13) & 0x3FFu;
    const uint32_t rem = x & 0x1FFFu;
    uint16_t out = (uint16_t)(sign | e | m);
    out = (uint16_t)(out + ((rem > 0x1000u) || (rem == 0x1000u && (out & 1))));
    return out;
}

// ---- the 3INST oracle LUT: 65,536 fp16 values (and their exact fp32 widening),
// built once at extension init from the decode_3inst recipe -------------------------
extern uint16_t g_lut16[65536];
extern float    g_lut32[65536];
void init_lut();

// ---- decode micro-variant (SIMD tiers implement both; scalar is LUT-only) ---------
enum DecodeVariant { DECODE_AUTO = 0, DECODE_ARITH = 1, DECODE_LUT = 2 };
extern DecodeVariant g_decode_variant;

// ---- tier vtable ------------------------------------------------------------------
struct Kernels {
    // decompress packed -> W (m x k) fp16 bit patterns (row-major, stride k).
    void (*decompress)(const uint16_t* packed_u16, uint16_t* W, int64_t m, int64_t k,
                       int R);
    // fused GEMV: y (m fp32) (+)= wscale * (W @ x), weights never materialized.
    // One call handles rows [row_begin, row_end) in units of 32-row m-pair blocks so
    // at::parallel_for can partition rows without splitting any row across threads.
    void (*matvec)(const uint16_t* packed_u16, const float* x, float* y,
                   int64_t m, int64_t k, int R, float wscale, bool accum,
                   int64_t blk_begin, int64_t blk_end);
    // fused small-batch GEMM: y (B x m, row-major) (+)= wscale * (x @ W.T); decode runs
    // once per fragment and is amortized over the B tokens. Row b's accumulation chain
    // is IDENTICAL to matvec on x[b] — bit-exact row parity is the contract.
    void (*matmul)(const uint16_t* packed_u16, const float* x, float* y, int64_t B,
                   int64_t m, int64_t k, int R, float wscale, bool accum,
                   int64_t blk_begin, int64_t blk_end);
};

enum Tier { TIER_SCALAR = 0, TIER_AVX2 = 1, TIER_AVX512 = 2, TIER_AVX512FP16 = 3,
            TIER_COUNT = 4 };

extern const Kernels* g_tier_tables[TIER_COUNT];   // null slot = TU not compiled in
extern const char* const g_tier_names[TIER_COUNT];

bool tier_cpu_supported(Tier t);   // CPU feature bits only
bool tier_available(Tier t);       // compiled-in AND cpu-supported
Tier resolve_auto();
const Kernels& active();           // current tier's table
void set_tier_by_name(const char* name);   // "auto" or a tier name; throws if unavailable
const char* active_name();

}  // namespace glq_cpu
