/* glq_moe_cpu.cpp — fused MoE decode for the 3INST trellis on CPU.
 *
 * Mirrors `glq_fused_moe_trellis_3inst_cuda` (glq_cuda.cu:5042) semantically, with the
 * same conventions, because a CPU/GPU disagreement must mean a bug rather than a
 * difference of opinion:
 *
 *   * `SU` and `Wscale` are PER EXPERT; `SV` and the block-diagonal metadata are SHARED
 *     across experts (one RHT basis per shard, not per expert).
 *   * `w13` holds gate and up concatenated and **gate is the first half** of w13_out
 *     (glq_cuda.cu:4543-4559): h = act(y[:, :inter]) * y[:, inter:].
 *   * The reduce runs over k in FIXED order into a per-routing scratch, exactly as the
 *     GPU's weighted scatter-reduce does (glq_cuda.cu:4520-4537). Accumulating in expert
 *     order instead would be just as "correct" and would silently produce different
 *     floating-point results from the GPU on any token whose experts are not already
 *     sorted — fp addition is not associative.
 *
 * The heavy work reuses what already exists and is already gated bit-exactly: the
 * ISA-dispatched decode vtable (`glq_cpu::active()`), the block-diagonal FHT
 * (`blockdiag_fht_rows`), and ATen for the elementwise activation — ATen deliberately,
 * so the activation is bit-identical to the torch reference rather than a second
 * implementation of tanh that agrees to within an ULP.
 */
#include <torch/extension.h>

#include <ATen/Parallel.h>

#include <vector>

#include "glq_trellis_cpu.hpp"

namespace glq_cpu {
void blockdiag_fht_rows(float* x, int64_t rows, int64_t n, const int32_t* meta,
                        int64_t nblocks);
}  // namespace glq_cpu

namespace {

int moe_bits(const torch::Tensor& packed) {
    TORCH_CHECK(packed.dim() == 3, "packed MoE weights must be (E, tiles, 16*R)");
    const int R = (int)(packed.size(2) / 16);
    TORCH_CHECK(R >= 2 && R <= 4 && packed.size(2) == 16 * R,
                "trellis CPU MoE supports R (bits/weight) 2-4, got ", packed.size(2) / 16.0);
    return R;
}

/* One projection for one expert's rows: signs -> FHT -> decode-GEMM -> FHT -> signs.
 * The same bracket as the dense path (glq_bindings_cpu.cpp fused_linear_impl); kept in
 * step with it by tests/test_moe_cpu_kernel.py::test_single_expert_equals_the_dense_bracket,
 * which asserts a one-expert MoE equals the dense op exactly. */
torch::Tensor expert_bracket(const torch::Tensor& rows_in,   // (n, k) fp32 contiguous
                             const torch::Tensor& sv,        // (k,) fp16, shared
                             const torch::Tensor& su_e,      // (m,) fp16, this expert
                             const torch::Tensor& packed_e,  // (tiles, 16R) int16
                             const torch::Tensor& meta_n, const torch::Tensor& meta_m,
                             float wscale, int64_t m, int64_t k, int R) {
    const int64_t n = rows_in.size(0);
    auto xr = (rows_in * sv.to(torch::kFloat32)).contiguous();
    auto mn = meta_n.contiguous();
    glq_cpu::blockdiag_fht_rows(xr.data_ptr<float>(), n, k, mn.data_ptr<int32_t>(),
                                mn.size(0));

    const auto& kern = glq_cpu::active();
    const uint16_t* pp = reinterpret_cast<const uint16_t*>(packed_e.data_ptr<int16_t>());
    torch::Tensor y;
    if (n <= glq_cpu::batch_max()) {
        y = torch::empty({n, m}, torch::dtype(torch::kFloat32));
        const float* xp = xr.data_ptr<float>();
        float* yp = y.data_ptr<float>();
        // Parallel over 32-row output blocks, the same partitioning the dense op uses, so
        // a row's accumulation order does not depend on the thread count.
        at::parallel_for(0, m / 32, 1, [&](int64_t b0, int64_t b1) {
            kern.matmul(pp, xp, yp, n, m, k, R, wscale, /*accum=*/false, b0, b1);
        });
    } else {
        // Above the fused batch cap the SIMD kernels have no accumulator room (their
        // per-row state is a fixed 8-deep array), so the dense path decompresses ONCE and
        // hands the tuned GEMM the transient weight — the resident weight stays packed.
        // Passing a bigger batch to matmul() writes past that array: it segfaults rather
        // than degrading, which is how this branch was (re)discovered.
        auto W = torch::empty({m, k}, torch::dtype(torch::kFloat16));
        kern.decompress(pp, reinterpret_cast<uint16_t*>(W.data_ptr<at::Half>()), m, k, R);
        y = at::matmul(xr, W.to(torch::kFloat32).t()).mul_(wscale).contiguous();
    }

    auto mm = meta_m.contiguous();
    glq_cpu::blockdiag_fht_rows(y.data_ptr<float>(), n, m, mm.data_ptr<int32_t>(),
                                mm.size(0));
    y.mul_(su_e.to(torch::kFloat32));
    return y;
}

torch::Tensor apply_gated_activation(const torch::Tensor& y, int64_t inter,
                                     int64_t activation) {
    using torch::indexing::Slice;
    auto gate = y.index({Slice(), Slice(0, inter)});
    auto up = y.index({Slice(), Slice(inter, torch::indexing::None)});
    torch::Tensor g;
    if (activation == 0) {
        g = at::silu(gate);
    } else if (activation == 1) {
        g = at::gelu(gate, "tanh");
    } else {
        g = at::relu(gate).pow(2);
    }
    return (g * up).contiguous();
}

torch::Tensor fused_moe_trellis_3inst_cpu(
    torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
    torch::Tensor w13_packed, torch::Tensor w13_SU, torch::Tensor w13_SV,
    torch::Tensor w13_Wscale,
    torch::Tensor w2_packed, torch::Tensor w2_SU, torch::Tensor w2_SV,
    torch::Tensor w2_Wscale,
    int64_t hidden, int64_t inter, int64_t w13_out,
    torch::Tensor meta_n_w13, torch::Tensor meta_m_w13,
    torch::Tensor meta_n_w2, torch::Tensor meta_m_w2,
    int64_t activation) {
    TORCH_CHECK(!x.is_cuda(), "x must be a CPU tensor");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == hidden, "x must be (tokens, hidden)");
    TORCH_CHECK(topk_ids.dim() == 2 && topk_weights.sizes() == topk_ids.sizes(),
                "topk_ids and topk_weights must both be (tokens, top_k)");
    TORCH_CHECK(topk_ids.size(0) == x.size(0), "routing rows must match tokens");
    TORCH_CHECK(w13_out == 2 * inter,
                "this entry is gated-only: w13_out must be 2*intermediate, got ", w13_out,
                " vs ", inter);
    TORCH_CHECK(activation >= 0 && activation < 3,
                "activation must be 0 (silu), 1 (gelu-tanh) or 2 (relu^2); the *_no_mul "
                "variants have no gated fused path");
    const int64_t E = w13_packed.size(0);
    TORCH_CHECK(w2_packed.size(0) == E && w13_SU.size(0) == E && w2_SU.size(0) == E
                    && w13_Wscale.numel() == E && w2_Wscale.numel() == E,
                "per-expert tensors disagree on the number of experts");
    TORCH_CHECK(w13_SV.numel() == hidden && w2_SV.numel() == inter,
                "SV is shared across experts and sized by the shard's INPUT dim");

    const int R13 = moe_bits(w13_packed), R2 = moe_bits(w2_packed);
    glq_cpu::init_lut();

    const int64_t T = x.size(0), topk = topk_ids.size(1), routings = T * topk;
    auto xf = x.to(torch::kFloat32).contiguous();
    auto ids = topk_ids.to(torch::kLong).contiguous();
    auto wts = topk_weights.to(torch::kFloat32).contiguous();
    const int64_t* idp = ids.data_ptr<int64_t>();

    // Group routings by expert. Ascending routing index within an expert keeps the gather
    // deterministic; the REDUCE order is fixed separately, below.
    std::vector<std::vector<int64_t>> by_expert((size_t)E);
    for (int64_t r = 0; r < routings; ++r) {
        const int64_t e = idp[r];
        if (e >= 0 && e < E) by_expert[(size_t)e].push_back(r);   // out-of-range = dropped
    }

    // Per-routing scratch, exactly like the GPU's expert_out: the weighted sum over k
    // happens afterwards in k order, so the result cannot depend on expert ordering.
    auto expert_out = torch::zeros({routings, hidden}, torch::dtype(torch::kFloat32));

    for (int64_t e = 0; e < E; ++e) {
        const auto& rs = by_expert[(size_t)e];
        if (rs.empty()) continue;                      // an unrouted expert decodes nothing
        const int64_t n = (int64_t)rs.size();

        auto rows = torch::empty({n, hidden}, torch::dtype(torch::kFloat32));
        auto sel = torch::empty({n}, torch::dtype(torch::kLong));
        int64_t* selp = sel.data_ptr<int64_t>();
        for (int64_t i = 0; i < n; ++i) selp[i] = rs[(size_t)i] / topk;   // routing -> token
        at::index_select_out(rows, xf, 0, sel);

        auto h = expert_bracket(rows, w13_SV, w13_SU[e], w13_packed[e], meta_n_w13,
                                meta_m_w13, w13_Wscale[e].item<float>(), w13_out, hidden,
                                R13);
        auto act = apply_gated_activation(h, inter, activation);
        auto z = expert_bracket(act, w2_SV, w2_SU[e], w2_packed[e], meta_n_w2, meta_m_w2,
                                w2_Wscale[e].item<float>(), hidden, inter, R2);

        auto dst = torch::from_blob(const_cast<int64_t*>(rs.data()), {n},
                                    torch::dtype(torch::kLong));
        expert_out.index_copy_(0, dst, z);
    }

    // out[t] = sum_k wts[t,k] * expert_out[t*topk + k], k ascending — the GPU's order.
    auto grouped = expert_out.view({T, topk, hidden});
    return (grouped * wts.unsqueeze(-1)).sum(/*dim=*/1);
}

}  // namespace

namespace glq_cpu {
void register_moe_bindings(pybind11::module& m) {
    m.def("glq_fused_moe_trellis_3inst_cpu", &fused_moe_trellis_3inst_cpu,
          py::arg("x"), py::arg("topk_ids"), py::arg("topk_weights"),
          py::arg("w13_trellis_packed"), py::arg("w13_SU"), py::arg("w13_SV"),
          py::arg("w13_Wscale"),
          py::arg("w2_trellis_packed"), py::arg("w2_SU"), py::arg("w2_SV"),
          py::arg("w2_Wscale"),
          py::arg("hidden_size"), py::arg("intermediate_size"), py::arg("w13_out_features"),
          py::arg("blocks_n_w13_meta"), py::arg("blocks_m_w13_meta"),
          py::arg("blocks_n_w2_meta"), py::arg("blocks_m_w2_meta"),
          py::arg("activation_type"));
}
}  // namespace glq_cpu
