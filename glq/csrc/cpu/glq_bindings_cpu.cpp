/* glq_bindings_cpu.cpp — PYBIND11_MODULE for glq._C_cpu (and the JIT twin "glq_cpu").
 *
 * Naming mirrors the CUDA extension with a `_cpu` suffix so call sites read the same.
 * Input validation mirrors tr_bits_from_packed / tr_check_shape in glq_trellis.cu:
 * the R=1 rate is reachable only behind allow_r1 (it is the stacked-RVQ residual rate,
 * and the bare ladder would otherwise read a neighbour's bits and return plausible
 * garbage for a mis-sized buffer).
 */
#include <torch/extension.h>

#include "glq_trellis_cpu.hpp"

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

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("glq_decompress_trellis_3inst_cpu", &decompress_trellis_3inst_cpu,
          py::arg("trellis_packed"), py::arg("m"), py::arg("k"),
          py::arg("allow_r1") = false);
    m.def("glq_trellis_3inst_lut_cpu", &lut_cpu);
    m.def("glq_trellis_cpu_kernel_supported", &kernel_supported,
          py::arg("m"), py::arg("k"));
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
}
