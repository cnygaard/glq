/* glq_bindings_cpu.cpp — PYBIND11_MODULE for glq._C_cpu (and the JIT twin "glq_cpu").
 *
 * Naming mirrors the CUDA extension with a `_cpu` suffix so call sites read the same.
 * Input validation mirrors tr_bits_from_packed / tr_check_shape in glq_trellis.cu:
 * the R=1 rate is reachable only behind allow_r1 (it is the stacked-RVQ residual rate,
 * and the bare ladder would otherwise read a neighbour's bits and return plausible
 * garbage for a mis-sized buffer).
 */
#include <torch/extension.h>

#include <ATen/Parallel.h>

#include "glq_trellis_cpu.hpp"

namespace glq_cpu {
// glq_fht_cpu.cpp
void blockdiag_fht_rows(float* x, int64_t rows, int64_t n, const int32_t* meta,
                        int64_t nblocks);
// glq_moe_cpu.cpp — the fused MoE entry registers itself, so the MoE orchestration
// (grouping, per-expert indexing, gated activation, weighted reduce) stays in its own TU.
void register_moe_bindings(pybind11::module& m);
}  // namespace glq_cpu

namespace {

int bits_from_packed(const torch::Tensor& packed, bool allow_r1) {
    TORCH_CHECK(packed.dim() == 2, "trellis_packed must be 2-D [(m/16)*(k/16), 16*R]");
    const int R = (int)(packed.size(1) / 16);
    const int lo = allow_r1 ? 1 : 2;
    TORCH_CHECK(R >= lo && R <= 4,
                "trellis CPU kernel supports R (bits/weight) ", lo, "-4, got ", R);
    TORCH_CHECK(packed.size(1) == 16 * R, "trellis_packed columns must be a multiple of 16");
    return R;
}

bool kernel_supported(int64_t m, int64_t k) {
    return m > 0 && k > 0 && (m % 32 == 0) && (k % 64 == 0);
}

torch::Tensor decompress_trellis_3inst_cpu(torch::Tensor packed, int64_t m, int64_t k,
                                           bool allow_r1) {
    TORCH_CHECK(!packed.is_cuda(), "trellis_packed must be a CPU tensor");
    TORCH_CHECK(packed.scalar_type() == torch::kInt16, "trellis_packed must be int16");
    packed = packed.contiguous();
    const int R = bits_from_packed(packed, allow_r1);
    TORCH_CHECK(kernel_supported(m, k), "CPU trellis kernel needs m % 32 == 0 and "
                "k % 64 == 0, got (", m, ", ", k, ")");
    TORCH_CHECK(packed.size(0) == (m / 16) * (k / 16),
                "trellis_packed rows ", packed.size(0), " != (m/16)*(k/16)");
    glq_cpu::init_lut();

    auto W = torch::empty({m, k}, torch::dtype(torch::kFloat16));
    glq_cpu::active().decompress(
        reinterpret_cast<const uint16_t*>(packed.data_ptr<int16_t>()),
        reinterpret_cast<uint16_t*>(W.data_ptr<at::Half>()), m, k, R);
    return W;
}

torch::Tensor lut_cpu() {
    glq_cpu::init_lut();
    auto t = torch::empty({65536}, torch::dtype(torch::kFloat16));
    std::memcpy(t.data_ptr<at::Half>(), glq_cpu::g_lut16, sizeof(glq_cpu::g_lut16));
    return t;
}

torch::Tensor matvec_trellis_3inst_cpu(torch::Tensor x, torch::Tensor packed,
                                       int64_t m, int64_t k, double wscale,
                                       c10::optional<torch::Tensor> out_opt,
                                       bool accum) {
    TORCH_CHECK(!x.is_cuda() && x.scalar_type() == torch::kFloat32,
                "x must be a CPU fp32 tensor");
    TORCH_CHECK(x.numel() == k, "x must have k elements, got ", x.numel());
    TORCH_CHECK(!accum || out_opt.has_value(),
                "accumulate mode needs the caller's `out` (stage 1 is the initializer)");
    x = x.contiguous();
    packed = packed.contiguous();
    // accum IS the residual-stage predicate, so it alone unlocks R=1 — mirroring the
    // CUDA entry's coupling of the two.
    const int R = bits_from_packed(packed, /*allow_r1=*/accum);
    TORCH_CHECK(kernel_supported(m, k), "CPU trellis kernel needs m % 32 == 0 and "
                "k % 64 == 0, got (", m, ", ", k, ")");
    TORCH_CHECK(packed.size(0) == (m / 16) * (k / 16),
                "trellis_packed rows ", packed.size(0), " != (m/16)*(k/16)");
    glq_cpu::init_lut();

    auto out = out_opt.has_value() ? *out_opt
                                   : torch::empty({m}, torch::dtype(torch::kFloat32));
    TORCH_CHECK(out.is_contiguous() && out.numel() == m
                    && out.scalar_type() == torch::kFloat32 && !out.is_cuda(),
                "out must be a contiguous CPU (m,) fp32 tensor");
    const auto& kern = glq_cpu::active();
    const uint16_t* pp = reinterpret_cast<const uint16_t*>(packed.data_ptr<int16_t>());
    const float* xp = x.data_ptr<float>();
    float* yp = out.data_ptr<float>();
    const int64_t blocks = m / 32;
    at::parallel_for(0, blocks, 1, [&](int64_t b0, int64_t b1) {
        kern.matvec(pp, xp, yp, m, k, R, (float)wscale, accum, b0, b1);
    });
    return out;
}

torch::Tensor matmul_trellis_3inst_cpu(torch::Tensor x, torch::Tensor packed,
                                       int64_t m, int64_t k, double wscale,
                                       c10::optional<torch::Tensor> out_opt,
                                       bool accum) {
    TORCH_CHECK(!x.is_cuda() && x.scalar_type() == torch::kFloat32,
                "x must be a CPU fp32 tensor");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == k, "x must be (B, k)");
    const int64_t B = x.size(0);
    TORCH_CHECK(B >= 1 && B <= 8, "fused CPU GEMM handles B <= 8 (got ", B,
                "); larger batches take the dense-transient path");
    TORCH_CHECK(!accum || out_opt.has_value(),
                "accumulate mode needs the caller's `out` (stage 1 is the initializer)");
    x = x.contiguous();
    packed = packed.contiguous();
    const int R = bits_from_packed(packed, /*allow_r1=*/accum);
    TORCH_CHECK(kernel_supported(m, k), "CPU trellis kernel needs m % 32 == 0 and "
                "k % 64 == 0, got (", m, ", ", k, ")");
    TORCH_CHECK(packed.size(0) == (m / 16) * (k / 16),
                "trellis_packed rows ", packed.size(0), " != (m/16)*(k/16)");
    glq_cpu::init_lut();

    auto out = out_opt.has_value() ? *out_opt
                                   : torch::empty({B, m}, torch::dtype(torch::kFloat32));
    TORCH_CHECK(out.is_contiguous() && out.numel() == B * m
                    && out.scalar_type() == torch::kFloat32 && !out.is_cuda(),
                "out must be a contiguous CPU (B, m) fp32 tensor");
    const auto& kern = glq_cpu::active();
    const uint16_t* pp = reinterpret_cast<const uint16_t*>(packed.data_ptr<int16_t>());
    const float* xp = x.data_ptr<float>();
    float* yp = out.data_ptr<float>();
    at::parallel_for(0, m / 32, 1, [&](int64_t b0, int64_t b1) {
        kern.matmul(pp, xp, yp, B, m, k, R, (float)wscale, accum, b0, b1);
    });
    return out;
}

// ---- the fused linear bracket: signs -> in-FHT -> decode-GEMM -> out-FHT -> signs ----
int64_t cpu_batch_max() { return glq_cpu::batch_max(); }

torch::Tensor fused_linear_impl(torch::Tensor x2d, torch::Tensor sv, torch::Tensor su,
                                torch::Tensor packed,
                                c10::optional<torch::Tensor> packed2, double inv_rs2,
                                torch::Tensor meta_n, torch::Tensor meta_m,
                                double wscale, int64_t in_features, int64_t out_features,
                                int64_t n_pad, int64_t m_pad) {
    TORCH_CHECK(!x2d.is_cuda() && x2d.dim() == 2, "x must be a CPU (B, in) tensor");
    // Trellis quantization never pads (block-diagonal RHT sized to the layer), so the
    // bracket needs no pad/slice steps — assert the invariant instead of handling it.
    TORCH_CHECK(n_pad == in_features && m_pad == out_features,
                "trellis layers are unpadded; got n_pad ", n_pad, " != in ", in_features,
                " or m_pad ", m_pad, " != out ", out_features);
    TORCH_CHECK(sv.numel() == n_pad && su.numel() == m_pad, "sv/su size mismatch");
    const bool has_s2 = packed2.has_value() && packed2->numel() > 0;
    TORCH_CHECK(has_s2 == (inv_rs2 != 0.0),
                "trellis RVQ stage 2 is half-configured: packed2 is ",
                has_s2 ? "present" : "absent", " but inv_resid_scale2 = ", inv_rs2,
                ". Refusing rather than silently decoding stage 1 only.");
    glq_cpu::init_lut();

    const int64_t B = x2d.size(0);
    // Step 1: x * sv, then the in-op block-diagonal FHT (fp32 throughout — an
    // intentional divergence from the GPU path's fp16 activations; accuracy only gains).
    auto xr = (x2d.to(torch::kFloat32) * sv.to(torch::kFloat32)).contiguous();
    auto mn = meta_n.contiguous();
    glq_cpu::blockdiag_fht_rows(xr.data_ptr<float>(), B, n_pad,
                                mn.data_ptr<int32_t>(), mn.size(0));

    // Step 2: decode + matmul, weights never materialized for B <= batch_max.
    torch::Tensor y;
    const double ws2 = wscale * inv_rs2;
    if (B <= cpu_batch_max()) {
        y = (B == 1)
            ? matvec_trellis_3inst_cpu(xr.view({n_pad}), packed, m_pad, n_pad, wscale,
                                       c10::nullopt, false).view({1, m_pad})
            : matmul_trellis_3inst_cpu(xr, packed, m_pad, n_pad, wscale,
                                       c10::nullopt, false);
        if (has_s2) {
            if (B == 1)
                matvec_trellis_3inst_cpu(xr.view({n_pad}), *packed2, m_pad, n_pad, ws2,
                                         y.view({m_pad}), true);
            else
                matmul_trellis_3inst_cpu(xr, *packed2, m_pad, n_pad, ws2, y, true);
        }
    } else {
        // Prefill-sized batches: decompress ONCE (transient — the resident weight stays
        // compressed) and let the tuned torch GEMM run; RVQ folds by weight linearity.
        auto W = decompress_trellis_3inst_cpu(packed, m_pad, n_pad, false).to(torch::kFloat32);
        if (has_s2) {
            auto W2 = decompress_trellis_3inst_cpu(*packed2, m_pad, n_pad, true);
            W.add_(W2.to(torch::kFloat32), inv_rs2);
        }
        y = at::matmul(xr, W.t()).mul_((float)wscale).contiguous();
    }

    // Step 3: out-FHT + row signs.
    auto mm = meta_m.contiguous();
    glq_cpu::blockdiag_fht_rows(y.data_ptr<float>(), B, m_pad,
                                mm.data_ptr<int32_t>(), mm.size(0));
    y.mul_(su.to(torch::kFloat32));
    return y;
}

torch::Tensor fused_linear_trellis_3inst_cpu(torch::Tensor x2d, torch::Tensor sv,
                                             torch::Tensor su, torch::Tensor packed,
                                             torch::Tensor meta_n, torch::Tensor meta_m,
                                             double wscale, int64_t in_features,
                                             int64_t out_features, int64_t n_pad,
                                             int64_t m_pad) {
    return fused_linear_impl(x2d, sv, su, packed, c10::nullopt, 0.0, meta_n, meta_m,
                             wscale, in_features, out_features, n_pad, m_pad);
}

torch::Tensor fused_linear_trellis_3inst_rvq2_cpu(
    torch::Tensor x2d, torch::Tensor sv, torch::Tensor su, torch::Tensor packed,
    torch::Tensor packed2, torch::Tensor meta_n, torch::Tensor meta_m, double wscale,
    double inv_resid_scale2, int64_t in_features, int64_t out_features, int64_t n_pad,
    int64_t m_pad) {
    return fused_linear_impl(x2d, sv, su, packed, packed2, inv_resid_scale2, meta_n,
                             meta_m, wscale, in_features, out_features, n_pad, m_pad);
}

torch::Tensor blockdiag_fht_cpu(torch::Tensor x, torch::Tensor meta) {
    TORCH_CHECK(!x.is_cuda() && x.scalar_type() == torch::kFloat32,
                "x must be a CPU fp32 tensor");
    TORCH_CHECK(x.dim() == 2, "x must be (rows, n)");
    TORCH_CHECK(meta.scalar_type() == torch::kInt32 && meta.dim() == 2
                    && meta.size(1) == 4 && !meta.is_cuda(),
                "meta must be CPU int32 (nblocks, 4) {offset, bs, log_bs, pad}");
    auto out = x.contiguous().clone();     // never mutate the caller's activations
    auto mc = meta.contiguous();
    glq_cpu::blockdiag_fht_rows(out.data_ptr<float>(), out.size(0), out.size(1),
                                mc.data_ptr<int32_t>(), mc.size(0));
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("glq_decompress_trellis_3inst_cpu", &decompress_trellis_3inst_cpu,
          py::arg("trellis_packed"), py::arg("m"), py::arg("k"),
          py::arg("allow_r1") = false);
    m.def("glq_trellis_3inst_lut_cpu", &lut_cpu);
    m.def("glq_trellis_cpu_kernel_supported", &kernel_supported,
          py::arg("m"), py::arg("k"));
    glq_cpu::register_moe_bindings(m);
    m.def("glq_cpu_active_isa", [] { return std::string(glq_cpu::active_name()); });
    m.def("glq_cpu_isa_available", [](const std::string& name) {
        for (int t = 0; t < glq_cpu::TIER_COUNT; ++t)
            if (name == glq_cpu::g_tier_names[t])
                return glq_cpu::tier_available((glq_cpu::Tier)t);
        return false;
    }, py::arg("name"));
    m.def("glq_cpu_set_isa", [](const std::string& name) {
        glq_cpu::set_tier_by_name(name.c_str());
    }, py::arg("name"));
    m.def("glq_decode_matvec_trellis_3inst_cpu", &matvec_trellis_3inst_cpu,
          py::arg("x"), py::arg("trellis_packed"), py::arg("m"), py::arg("k"),
          py::arg("wscale") = 1.0, py::arg("out") = py::none(),
          py::arg("accum") = false);
    m.def("glq_blockdiag_fht_cpu", &blockdiag_fht_cpu, py::arg("x"), py::arg("meta"));
    m.def("glq_decode_matmul_trellis_3inst_cpu", &matmul_trellis_3inst_cpu,
          py::arg("x"), py::arg("trellis_packed"), py::arg("m"), py::arg("k"),
          py::arg("wscale") = 1.0, py::arg("out") = py::none(),
          py::arg("accum") = false);
    m.def("glq_fused_linear_trellis_3inst_cpu", &fused_linear_trellis_3inst_cpu,
          py::arg("x"), py::arg("sv"), py::arg("su"), py::arg("trellis_packed"),
          py::arg("blocks_n_meta"), py::arg("blocks_m_meta"), py::arg("wscale"),
          py::arg("in_features"), py::arg("out_features"), py::arg("n_pad"),
          py::arg("m_pad"));
    m.def("glq_fused_linear_trellis_3inst_rvq2_cpu", &fused_linear_trellis_3inst_rvq2_cpu,
          py::arg("x"), py::arg("sv"), py::arg("su"), py::arg("trellis_packed"),
          py::arg("trellis_packed2"), py::arg("blocks_n_meta"), py::arg("blocks_m_meta"),
          py::arg("wscale"), py::arg("inv_resid_scale2"), py::arg("in_features"),
          py::arg("out_features"), py::arg("n_pad"), py::arg("m_pad"));
    m.def("glq_cpu_set_decode_variant", [](const std::string& v) {
        if (v == "auto") glq_cpu::g_decode_variant = glq_cpu::DECODE_AUTO;
        else if (v == "arith") glq_cpu::g_decode_variant = glq_cpu::DECODE_ARITH;
        else if (v == "lut") glq_cpu::g_decode_variant = glq_cpu::DECODE_LUT;
        else throw std::runtime_error("unknown decode variant: " + v +
                                      " (want auto|arith|lut)");
    }, py::arg("variant"));
}
