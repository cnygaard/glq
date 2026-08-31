/* glq_cpu_dispatch.cpp — LUT storage/init and runtime ISA tier selection.
 *
 * Resolution order: an explicit set_tier_by_name() (tests sweep tiers in one process)
 * beats the GLQ_CPU_ISA env (read once, the headless override) beats auto-detection
 * (highest compiled-in tier the CPU supports). Requesting an unavailable tier THROWS —
 * a silent downgrade would let a CI tier-sweep pass while testing the wrong code.
 */
#include "glq_trellis_cpu.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

namespace glq_cpu {

uint16_t g_lut16[65536];
float    g_lut32[65536];

void init_lut() {
    static std::once_flag once;
    std::call_once(once, [] {
        for (uint32_t s = 0; s < 65536; ++s) {
            const uint32_t h = s * 89226354u + 64248484u;
            const uint32_t r = (h & 0x8FFF8FFFu) ^ 0x3B603B60u;
            // fp32 sum of the two fp16 halves is exact; RN16 makes it the oracle's
            // fp16 add (differs on 27,707/65,536 states without the rounding).
            const float f = half_to_float((uint16_t)(r >> 16))
                          + half_to_float((uint16_t)(r & 0xFFFFu));
            g_lut16[s] = float_to_half_rn(f);
            g_lut32[s] = half_to_float(g_lut16[s]);
        }
    });
}

const Kernels* scalar_kernels();   // defined in the scalar TU (always compiled)
const Kernels* avx2_kernels();     // avx2 TU (returns null on non-x86 builds)
const Kernels* avx512_kernels();   // avx512 TU (returns null on non-x86 builds)
const Kernels* avx512fp16_kernels();   // fp16 TU (null below GCC 12 / non-x86)

const Kernels* g_tier_tables[TIER_COUNT] = { nullptr, nullptr, nullptr, nullptr };
const char* const g_tier_names[TIER_COUNT] = { "scalar", "avx2", "avx512", "avx512fp16" };

DecodeVariant g_decode_variant = DECODE_AUTO;

namespace {
std::atomic<int> g_active{-1};   // -1 = unresolved

void ensure_tables() {
    static std::once_flag once;
    std::call_once(once, [] {
        init_lut();
        g_tier_tables[TIER_SCALAR] = scalar_kernels();
        g_tier_tables[TIER_AVX2] = avx2_kernels();
        g_tier_tables[TIER_AVX512] = avx512_kernels();
        g_tier_tables[TIER_AVX512FP16] = avx512fp16_kernels();
    });
}
}  // namespace

bool tier_cpu_supported(Tier t) {
    switch (t) {
        case TIER_SCALAR: return true;
#if defined(__x86_64__) || defined(_M_X64)
        case TIER_AVX2:
            return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c")
                && __builtin_cpu_supports("fma");
        case TIER_AVX512:
            return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")
                && __builtin_cpu_supports("avx512vl");
        case TIER_AVX512FP16:
#if defined(__GNUC__) && __GNUC__ >= 12
            return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")
                && __builtin_cpu_supports("avx512vl")
                && __builtin_cpu_supports("avx512fp16");
#else
            return false;
#endif
#endif
        default: return false;
    }
}

bool tier_available(Tier t) {
    ensure_tables();
    return g_tier_tables[t] != nullptr && tier_cpu_supported(t);
}

Tier resolve_auto() {
    for (int t = TIER_COUNT - 1; t >= 0; --t)
        if (tier_available((Tier)t)) return (Tier)t;
    return TIER_SCALAR;
}

void set_tier_by_name(const char* name) {
    ensure_tables();
    if (std::strcmp(name, "auto") == 0) {
        g_active.store((int)resolve_auto());
        return;
    }
    for (int t = 0; t < TIER_COUNT; ++t) {
        if (std::strcmp(name, g_tier_names[t]) == 0) {
            if (!tier_available((Tier)t))
                throw std::runtime_error(std::string("glq CPU isa tier unavailable: ") + name);
            g_active.store(t);
            return;
        }
    }
    throw std::runtime_error(std::string("unknown glq CPU isa tier: ") + name
                             + " (want scalar|avx2|avx512|avx512fp16|auto)");
}

const Kernels& active() {
    ensure_tables();
    int t = g_active.load();
    if (t < 0) {
        const char* env = std::getenv("GLQ_CPU_ISA");
        if (env && env[0]) set_tier_by_name(env);
        else g_active.store((int)resolve_auto());
        t = g_active.load();
    }
    return *g_tier_tables[t];
}

const char* active_name() {
    active();   // force resolution
    return g_tier_names[g_active.load()];
}

}  // namespace glq_cpu
