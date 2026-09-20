/* glq_gdn_cpu.cpp — fused GatedDeltaNet decode for CPU.
 *
 * 36 of Qwen3.8-Flash-Next's 48 layers are `linear_attention`, and on CPU they run
 * transformers' reference PyTorch: every CPU log carries "fused_recurrent_gated_delta_rule
 * is falling back to its reference PyTorch implementation". Nobody has optimised that path
 * for CPU because both upstream packages are GPU-only -- flash-linear-attention dispatches
 * to Triton (measured: "RuntimeError: 0 active drivers"), and causal-conv1d ships one .cpp
 * binding over three .cu kernels. So this is not a faster version of something; it is the
 * first non-reference implementation.
 *
 * Scope is DECODE ONLY (seq_len == 1), which is the branch
 * `use_precomputed_states and seq_len == 1` selects (modeling_qwen4_exp.py:586). Prefill
 * goes through the chunked rule, a different and much larger function, and is not where
 * decode time goes.
 *
 * Tiering reuses glq_trellis_cpu.hpp's tier machinery rather than adding a second one, so
 * GLQ_CPU_ISA / glq_cpu_set_isa steer this kernel too and the test sweep gets every tier
 * for free. The bodies come from glq_gdn_cpu.inl, included once per target pragma.
 */
#include <torch/extension.h>

#include <ATen/Parallel.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "glq_trellis_cpu.hpp"

namespace glq_cpu {
namespace gdn {

// ---- one instantiation of the body per tier ----------------------------------------
namespace tier_scalar {
#include "glq_gdn_cpu.inl"
}  // namespace tier_scalar

#if defined(__x86_64__) || defined(_M_X64)
#pragma GCC push_options
#pragma GCC target("avx2,fma")
namespace tier_avx2 {
#include "glq_gdn_cpu.inl"
}  // namespace tier_avx2
#pragma GCC pop_options

#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512vl,avx512dq,fma")
namespace tier_avx512 {
#include "glq_gdn_cpu.inl"
}  // namespace tier_avx512
#pragma GCC pop_options
#endif

using StepFn = void (*)(float*, const float*, const float*, const float*, float, float,
                        float*, float*, int64_t, int64_t);

/* Which body to run. Deliberately keyed off the SAME tier resolution the trellis kernel
 * uses (`glq_cpu::active_name()`), so one `glq_cpu_set_isa("avx2")` steers both and a tier
 * sweep in the tests covers this kernel without a second control surface. There is no
 * fp16 body: the state is fp32 end to end, so avx512fp16 and avx512 run identical code. */
static StepFn resolve_step() {
#if defined(__x86_64__) || defined(_M_X64)
    const char* n = active_name();
    if (n != nullptr) {
        if (std::strcmp(n, "avx512fp16") == 0 || std::strcmp(n, "avx512") == 0)
            return &tier_avx512::gdn_head_step;
        if (std::strcmp(n, "avx2") == 0) return &tier_avx2::gdn_head_step;
    }
#endif
    return &tier_scalar::gdn_head_step;
}

/* Fused recurrent gated delta rule, one decode token.
 *
 * Mirrors torch_recurrent_gated_delta_rule (modeling_qwen4_exp.py:400) for
 * sequence_length == 1, with the caller having already done the cheap per-token prep --
 * the transposes, the l2 normalisation and the 1/sqrt(K) scaling -- which at S=1 are on
 * tiny tensors and not worth moving into C++.
 *
 *   state (B, H, K, V) fp32 contiguous, MUTATED IN PLACE and returned
 *   q, k  (B, H, K) fp32
 *   v     (B, H, V) fp32
 *   g     (B, H) fp32, the RAW decay; exponentiated here so the caller allocates no temp
 *   beta  (B, H) fp32
 *
 * Returns the (B, H, V) output. The state is updated in place because eliminating the
 * reference's four per-token temporaries is the entire point; the caller writes the same
 * object back into the cache, so the observable contract is unchanged.
 */
torch::Tensor gdn_recurrent_step_cpu(torch::Tensor state, torch::Tensor q, torch::Tensor k,
                                     torch::Tensor v, torch::Tensor g, torch::Tensor beta) {
    TORCH_CHECK(!state.is_cuda(), "gdn_recurrent_step_cpu: CPU tensors only");
    TORCH_CHECK(state.dim() == 4, "state must be (B, H, K, V), got ", state.sizes());
    TORCH_CHECK(state.scalar_type() == torch::kFloat32, "state must be fp32");
    TORCH_CHECK(state.is_contiguous(), "state must be contiguous");

    const int64_t B = state.size(0), H = state.size(1);
    const int64_t K = state.size(2), V = state.size(3);

    TORCH_CHECK(q.sizes() == k.sizes(), "q and k must match: ", q.sizes(), " vs ", k.sizes());
    TORCH_CHECK(q.dim() == 3 && q.size(0) == B && q.size(1) == H && q.size(2) == K,
                "q must be (B, H, K) = (", B, ", ", H, ", ", K, "), got ", q.sizes());
    TORCH_CHECK(v.dim() == 3 && v.size(0) == B && v.size(1) == H && v.size(2) == V,
                "v must be (B, H, V) = (", B, ", ", H, ", ", V, "), got ", v.sizes());
    TORCH_CHECK(g.numel() == B * H && beta.numel() == B * H,
                "g and beta must be (B, H)");

    auto qc = q.to(torch::kFloat32).contiguous();
    auto kc = k.to(torch::kFloat32).contiguous();
    auto vc = v.to(torch::kFloat32).contiguous();
    auto gc = g.to(torch::kFloat32).contiguous().view({B * H});
    auto bc = beta.to(torch::kFloat32).contiguous().view({B * H});

    auto out = torch::empty({B, H, V}, torch::dtype(torch::kFloat32));

    float* Sp = state.data_ptr<float>();
    const float* qp = qc.data_ptr<float>();
    const float* kp = kc.data_ptr<float>();
    const float* vp = vc.data_ptr<float>();
    const float* gp = gc.data_ptr<float>();
    const float* bp = bc.data_ptr<float>();
    float* op = out.data_ptr<float>();

    const StepFn step = resolve_step();

    /* ONE parallel region for the whole layer, partitioned over (batch, head).
     *
     * Not one per operation. Splitting the five reference passes into five parallel_fors
     * is the shape that makes torbi's CPU Viterbi 300,000x slower than its GPU path, and
     * it is the same trap as glq_moe_cpu.cpp:197's inner-parallel branch. Each (b, h) owns
     * a disjoint K x V slab and its own output row, so nothing races and no element's
     * accumulation order depends on the thread count. */
    at::parallel_for(0, B * H, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> scratch((size_t)V);
        for (int64_t bh = begin; bh < end; ++bh) {
            step(Sp + bh * K * V, qp + bh * K, kp + bh * K, vp + bh * V,
                 std::exp(gp[bh]), bp[bh], op + bh * V, scratch.data(), K, V);
        }
    });

    return out;
}

}  // namespace gdn

void register_gdn_bindings(pybind11::module& m) {
    m.def("glq_gdn_recurrent_step_cpu", &gdn::gdn_recurrent_step_cpu,
          py::arg("state"), py::arg("query"), py::arg("key"), py::arg("value"),
          py::arg("g"), py::arg("beta"),
          "Fused GatedDeltaNet decode step (seq_len==1). Mutates `state` in place and "
          "returns the (B, H, V) output.");
}

}  // namespace glq_cpu
