"""Register GLQ CUDA C kernels as torch custom ops for torch.compile + CUDA graphs.

vLLM's torch.compile tracing can't see through JIT-compiled pybind11 extensions.
Registering them as custom ops with fake implementations allows the compiler to
trace through GLQ's apply() methods and capture CUDA graphs.

Uses torch.library directly (pybind11 methods lack __globals__ so
vLLM's direct_register_custom_op/infer_schema can't introspect them).
"""

import torch
from torch.library import Library

_glq_lib = Library("glq", "FRAGMENT")
_registered = False


def _ensure_registered():
    """Register GLQ CUDA C kernels as torch custom ops. Idempotent."""
    global _registered
    if _registered:
        return
    _registered = True

    from glq.inference_kernel import _try_load_cuda_ext
    from glq import inference_kernel as _ik

    if not _try_load_cuda_ext() or _ik._glq_cuda is None:
        return

    from vllm.platforms import current_platform
    dispatch_key = current_platform.dispatch_key

    cuda = _ik._glq_cuda

    # -- 1. dequant_matvec: (N,) fp16 → (M,) fp32, with optional E8P codebook_abs --
    _glq_lib.define(
        "dequant_matvec(Tensor x, Tensor qidxs, Tensor codebook, float wscale, "
        "Tensor qidxs2, Tensor codebook2, float inv_resid_scale, "
        "Tensor codebook_abs) -> Tensor")
    _glq_lib.impl("dequant_matvec", cuda.glq_dequant_matvec_cuda, dispatch_key)
    _glq_lib._register_fake("dequant_matvec", _dequant_matvec_fake)

    # -- 2. dequant_matmul: (B, N) fp16 → (B, M) fp32 --
    _glq_lib.define(
        "dequant_matmul(Tensor x, Tensor qidxs, Tensor codebook, float wscale, "
        "Tensor qidxs2, Tensor codebook2, float inv_resid_scale) -> Tensor")
    _glq_lib.impl("dequant_matmul", cuda.glq_dequant_matmul_cuda, dispatch_key)
    _glq_lib._register_fake("dequant_matmul", _dequant_matmul_fake)

    # -- 3. dequant_matvec_packed: (N,) fp16 → (M,) fp32 --
    _glq_lib.define(
        "dequant_matvec_packed(Tensor x, Tensor qidxs, Tensor codebook_packed, "
        "float wscale) -> Tensor")
    _glq_lib.impl("dequant_matvec_packed", cuda.glq_dequant_matvec_packed_cuda, dispatch_key)
    _glq_lib._register_fake("dequant_matvec_packed", _dequant_matvec_packed_fake)

    # -- 4. dequant_matmul_packed: (B, N) fp16 → (B, M) fp32 --
    _glq_lib.define(
        "dequant_matmul_packed(Tensor x, Tensor qidxs, Tensor codebook_packed, "
        "float wscale) -> Tensor")
    _glq_lib.impl("dequant_matmul_packed", cuda.glq_dequant_matmul_packed_cuda, dispatch_key)
    _glq_lib._register_fake("dequant_matmul_packed", _dequant_matmul_packed_fake)

    # -- 5. input_rht: mutates pre-allocated out --
    _glq_lib.define(
        "input_rht(Tensor x, Tensor sv, Tensor(a!) out, int in_features, "
        "int stride_x, float rsqrt_n, int n_pad, int log_n) -> ()")
    _glq_lib.impl("input_rht", cuda.glq_input_rht_cuda, dispatch_key)
    _glq_lib._register_fake("input_rht", _input_rht_fake)

    # -- 6. output_rht: mutates pre-allocated out --
    _glq_lib.define(
        "output_rht(Tensor y_rht, Tensor su, Tensor(a!) out, int out_features, "
        "int m_pad, int log_m, float rsqrt_m) -> ()")
    _glq_lib.impl("output_rht", cuda.glq_output_rht_cuda, dispatch_key)
    _glq_lib._register_fake("output_rht", _output_rht_fake)


# --- Fake implementations for torch.compile tracing ---

def _dequant_matvec_fake(x, qidxs, codebook, wscale, qidxs2, codebook2, inv_resid_scale, codebook_abs):
    M = qidxs.shape[0]
    return torch.empty(M, dtype=torch.float32, device=x.device)


def _dequant_matmul_fake(x, qidxs, codebook, wscale, qidxs2, codebook2, inv_resid_scale):
    B = x.shape[0]
    M = qidxs.shape[0]
    return torch.empty(B, M, dtype=torch.float32, device=x.device)


def _dequant_matvec_packed_fake(x, qidxs, codebook_packed, wscale):
    M = qidxs.shape[0]
    return torch.empty(M, dtype=torch.float32, device=x.device)


def _dequant_matmul_packed_fake(x, qidxs, codebook_packed, wscale):
    B = x.shape[0]
    M = qidxs.shape[0]
    return torch.empty(B, M, dtype=torch.float32, device=x.device)


def _input_rht_fake(x, sv, out, in_features, stride_x, rsqrt_n, n_pad, log_n):
    pass  # mutates `out` in-place


def _output_rht_fake(y_rht, su, out, out_features, m_pad, log_m, rsqrt_m):
    pass  # mutates `out` in-place
