"""Compile-check the glq CUDA extension (verbose) after editing glq_cuda.cu."""
import os

import glq
from torch.utils.cpp_extension import load

csrc = os.path.join(os.path.dirname(glq.__file__), "csrc")
m = load(
    "glq_cuda",
    sources=[os.path.join(csrc, "glq_cuda.cu"), os.path.join(csrc, "glq_bindings.cpp")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)
print("BUILD_OK")
print("glq_moe_grouped_matmul:", hasattr(m, "glq_moe_grouped_matmul"))
print("glq_moe_build_grouping:", hasattr(m, "glq_moe_build_grouping"))
