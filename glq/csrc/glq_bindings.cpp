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
    float inv_resid_scale
);

torch::Tensor glq_dequant_matvec_packed_cuda(
    torch::Tensor x,
    torch::Tensor qidxs,
    torch::Tensor codebook_packed,
    float wscale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("glq_dequant_matvec_cuda", &glq_dequant_matvec_cuda,
          "GLQ dequant+matvec B=1 (CUDA)");
    m.def("glq_dequant_matvec_packed_cuda", &glq_dequant_matvec_packed_cuda,
          "GLQ dequant+matvec B=1 packed codebook (CUDA)");
}
