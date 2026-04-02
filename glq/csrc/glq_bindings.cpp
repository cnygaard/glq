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
    torch::Tensor codebook_abs  // unused, kept for ABI compat
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
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
