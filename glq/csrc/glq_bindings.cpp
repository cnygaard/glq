/*
 * pybind11 bindings for GLQ CUDA kernels
 */

#include <torch/extension.h>

// Forward declarations (defined in glq_cuda.cu)
torch::Tensor glq_dequant_matvec_cuda(
    torch::Tensor x,
    torch::Tensor qidxs,
    torch::Tensor codebook,
    float wscale,
    torch::Tensor qidxs2,
    torch::Tensor codebook2,
    float inv_resid_scale,
    torch::Tensor codebook_abs  // unused, kept for ABI compat
);

torch::Tensor glq_dequant_matmul_cuda(
    torch::Tensor x,
    torch::Tensor qidxs,
    torch::Tensor codebook,
    float wscale,
    torch::Tensor qidxs2,
    torch::Tensor codebook2,
    float inv_resid_scale,
    torch::Tensor codebook_abs,  // unused, kept for ABI compat
    torch::Tensor qidxs3,
    torch::Tensor codebook3,
    float inv_resid_scale2,
    torch::Tensor qidxs4,
    torch::Tensor codebook4,
    float inv_resid_scale3
);

torch::Tensor glq_dequant_matvec_packed_cuda(
    torch::Tensor x,
    torch::Tensor qidxs,
    torch::Tensor codebook_packed,
    float wscale
);

torch::Tensor glq_dequant_matmul_packed_cuda(
    torch::Tensor x,
    torch::Tensor qidxs,
    torch::Tensor codebook_packed,
    float wscale
);

void glq_input_rht_cuda(
    torch::Tensor x,
    torch::Tensor sv,
    torch::Tensor out,
    int in_features,
    int stride_x,
    float rsqrt_n,
    int n_pad,
    int log_n
);

void glq_output_rht_cuda(
    torch::Tensor y_rht,
    torch::Tensor su,
    torch::Tensor out,
    int out_features,
    int m_pad,
    int log_m,
    float rsqrt_m
);

torch::Tensor glq_fused_moe_cuda(
    torch::Tensor x,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor w13_Qidxs, torch::Tensor w13_SU, torch::Tensor w13_SV,
    torch::Tensor w13_Wscale, torch::Tensor w13_Qidxs2, torch::Tensor w13_inv_rs,
    torch::Tensor w2_Qidxs, torch::Tensor w2_SU, torch::Tensor w2_SV,
    torch::Tensor w2_Wscale, torch::Tensor w2_Qidxs2, torch::Tensor w2_inv_rs,
    torch::Tensor codebook, torch::Tensor codebook2,
    int hidden_size, int intermediate_size, int w13_out_features,
    int n_pad_w13, int m_pad_w13, int n_pad_w2, int m_pad_w2,
    int log_n_w13, int log_m_w13, int log_n_w2, int log_m_w2,
    int activation_type
);

torch::Tensor glq_fused_linear_cuda(
    torch::Tensor x,
    torch::Tensor sv,
    torch::Tensor su,
    torch::Tensor qidxs,
    torch::Tensor codebook,
    float wscale,
    int in_features,
    int out_features,
    int n_pad, int m_pad,
    int log_n, int log_m,
    torch::Tensor qidxs2,
    torch::Tensor codebook2,
    float inv_resid_scale,
    torch::Tensor qidxs3,
    torch::Tensor codebook3,
    float inv_resid_scale2,
    torch::Tensor qidxs4,
    torch::Tensor codebook4,
    float inv_resid_scale3
);

torch::Tensor glq_fused_linear_block_diag_cuda(
    torch::Tensor x,
    torch::Tensor sv,
    torch::Tensor su,
    torch::Tensor qidxs,
    torch::Tensor codebook,
    float wscale,
    int in_features,
    int out_features,
    int n_pad, int m_pad,
    torch::Tensor blocks_n,
    torch::Tensor blocks_m,
    torch::Tensor blocks_n_meta,
    torch::Tensor blocks_m_meta,
    torch::Tensor qidxs2,
    torch::Tensor codebook2,
    float inv_resid_scale,
    torch::Tensor qidxs3,
    torch::Tensor codebook3,
    float inv_resid_scale2,
    torch::Tensor qidxs4,
    torch::Tensor codebook4,
    float inv_resid_scale3
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("glq_fused_linear_cuda", &glq_fused_linear_cuda,
          "GLQ fused input_rht + dequant_matmul + output_rht (CUDA)");
    m.def("glq_fused_linear_block_diag_cuda", &glq_fused_linear_block_diag_cuda,
          "GLQ fused block-diagonal input_rht + dequant_matmul + output_rht (CUDA)");
    m.def("glq_fused_moe_cuda", &glq_fused_moe_cuda,
          "GLQ fused MoE expert dispatch (CUDA)");
    m.def("glq_dequant_matvec_cuda", &glq_dequant_matvec_cuda,
          "GLQ dequant+matvec B=1 (CUDA)");
    m.def("glq_dequant_matmul_cuda", &glq_dequant_matmul_cuda,
          "GLQ dequant+matmul any B (CUDA, loops B matvecs)");
    m.def("glq_dequant_matvec_packed_cuda", &glq_dequant_matvec_packed_cuda,
          "GLQ dequant+matvec B=1 packed codebook (CUDA)");
    m.def("glq_dequant_matmul_packed_cuda", &glq_dequant_matmul_packed_cuda,
          "GLQ dequant+matmul B>=2 packed codebook (CUDA)");
    m.def("glq_input_rht_cuda", &glq_input_rht_cuda,
          "GLQ input RHT (pad+SV+FHT) shared-memory (CUDA)");
    m.def("glq_output_rht_cuda", &glq_output_rht_cuda,
          "GLQ output RHT (FHT+SU+unpad) shared-memory (CUDA)");
}
