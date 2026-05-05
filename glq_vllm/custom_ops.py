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

    # -- 1. dequant_matvec: (N,) fp16 → (M,) fp32 --
    _glq_lib.define(
        "dequant_matvec(Tensor x, Tensor qidxs, Tensor codebook, float wscale, "
        "Tensor qidxs2, Tensor codebook2, float inv_resid_scale, "
        "Tensor codebook_abs) -> Tensor")
    _glq_lib.impl("dequant_matvec", cuda.glq_dequant_matvec_cuda, dispatch_key)
    _glq_lib._register_fake("dequant_matvec", _dequant_matvec_fake)

    # -- 2. dequant_matmul: (B, N) fp16 → (B, M) fp32 --
    _glq_lib.define(
        "dequant_matmul(Tensor x, Tensor qidxs, Tensor codebook, float wscale, "
        "Tensor qidxs2, Tensor codebook2, float inv_resid_scale, "
        "Tensor codebook_abs, "
        "Tensor qidxs3, Tensor codebook3, float inv_resid_scale2, "
        "Tensor qidxs4, Tensor codebook4, float inv_resid_scale3) -> Tensor")
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

    # -- 7. fused_linear (pow2): input_rht + dequant_matmul + output_rht in
    #       one kernel. (B, N) fp16 → (B, out_features) fp16. --
    _glq_lib.define(
        "fused_linear(Tensor x, Tensor sv, Tensor su, "
        "Tensor qidxs, Tensor codebook, float wscale, "
        "int in_features, int out_features, "
        "int n_pad, int m_pad, int log_n, int log_m, "
        "Tensor qidxs2, Tensor codebook2, float inv_resid_scale, "
        "Tensor qidxs3, Tensor codebook3, float inv_resid_scale2, "
        "Tensor qidxs4, Tensor codebook4, float inv_resid_scale3) -> Tensor")
    _glq_lib.impl("fused_linear", cuda.glq_fused_linear_cuda, dispatch_key)
    _glq_lib._register_fake("fused_linear", _fused_linear_fake)

    # -- 8. fused_linear_block_diag (non-pow2): same as above plus 4 block
    #       metadata tensors between m_pad and qidxs2. --
    _glq_lib.define(
        "fused_linear_block_diag(Tensor x, Tensor sv, Tensor su, "
        "Tensor qidxs, Tensor codebook, float wscale, "
        "int in_features, int out_features, int n_pad, int m_pad, "
        "Tensor blocks_n, Tensor blocks_m, "
        "Tensor blocks_n_meta, Tensor blocks_m_meta, "
        "Tensor qidxs2, Tensor codebook2, float inv_resid_scale, "
        "Tensor qidxs3, Tensor codebook3, float inv_resid_scale2, "
        "Tensor qidxs4, Tensor codebook4, float inv_resid_scale3) -> Tensor")
    _glq_lib.impl("fused_linear_block_diag",
                  cuda.glq_fused_linear_block_diag_cuda, dispatch_key)
    _glq_lib._register_fake("fused_linear_block_diag",
                            _fused_linear_block_diag_fake)

    # -- 9. fused_moe_block_diag: per-expert dispatch + dequant + matmul +
    #       activation + output project, in one kernel. (B, hidden) fp16 →
    #       (B, hidden) fp16. --
    _glq_lib.define(
        "fused_moe_block_diag(Tensor x, Tensor topk_ids, Tensor topk_weights, "
        "Tensor w13_Qidxs, Tensor w13_SU, Tensor w13_SV, "
        "Tensor w13_Wscale, Tensor w13_Qidxs2, Tensor w13_inv_rs, "
        "Tensor w2_Qidxs, Tensor w2_SU, Tensor w2_SV, "
        "Tensor w2_Wscale, Tensor w2_Qidxs2, Tensor w2_inv_rs, "
        "Tensor codebook, Tensor codebook2, "
        "int hidden_size, int intermediate_size, int w13_out_features, "
        "int n_pad_w13, int m_pad_w13, int n_pad_w2, int m_pad_w2, "
        "Tensor blocks_n_w13, Tensor blocks_m_w13, "
        "Tensor blocks_n_w13_meta, Tensor blocks_m_w13_meta, "
        "Tensor blocks_n_w2, Tensor blocks_m_w2, "
        "Tensor blocks_n_w2_meta, Tensor blocks_m_w2_meta, "
        "int activation_type, "
        "Tensor w13_Qidxs3, Tensor w13_inv_rs2, "
        "Tensor w2_Qidxs3, Tensor w2_inv_rs2, "
        "Tensor codebook3) -> Tensor")
    _glq_lib.impl("fused_moe_block_diag",
                  cuda.glq_fused_moe_block_diag_cuda, dispatch_key)
    _glq_lib._register_fake("fused_moe_block_diag",
                            _fused_moe_block_diag_fake)


# --- Fake implementations for torch.compile tracing ---

def _dequant_matvec_fake(x, qidxs, codebook, wscale, qidxs2, codebook2, inv_resid_scale, codebook_abs):
    M = qidxs.shape[0]
    return torch.empty(M, dtype=torch.float32, device=x.device)


def _dequant_matmul_fake(x, qidxs, codebook, wscale, qidxs2, codebook2, inv_resid_scale, codebook_abs,
                          qidxs3, codebook3, inv_resid_scale2,
                          qidxs4, codebook4, inv_resid_scale3):
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


def _fused_linear_fake(x, sv, su, qidxs, codebook, wscale,
                       in_features, out_features, n_pad, m_pad, log_n, log_m,
                       qidxs2, codebook2, inv_resid_scale,
                       qidxs3, codebook3, inv_resid_scale2,
                       qidxs4, codebook4, inv_resid_scale3):
    # Output is trimmed to out_features (qidxs.shape[0] may be m_pad >= out_features
    # for block-diag, but the kernel always emits exactly out_features cols).
    return torch.empty((*x.shape[:-1], out_features),
                       dtype=torch.float16, device=x.device)


def _fused_linear_block_diag_fake(x, sv, su, qidxs, codebook, wscale,
                                   in_features, out_features, n_pad, m_pad,
                                   blocks_n, blocks_m,
                                   blocks_n_meta, blocks_m_meta,
                                   qidxs2, codebook2, inv_resid_scale,
                                   qidxs3, codebook3, inv_resid_scale2,
                                   qidxs4, codebook4, inv_resid_scale3):
    return torch.empty((*x.shape[:-1], out_features),
                       dtype=torch.float16, device=x.device)


def _fused_moe_block_diag_fake(x, topk_ids, topk_weights,
                                w13_Qidxs, w13_SU, w13_SV,
                                w13_Wscale, w13_Qidxs2, w13_inv_rs,
                                w2_Qidxs, w2_SU, w2_SV,
                                w2_Wscale, w2_Qidxs2, w2_inv_rs,
                                codebook, codebook2,
                                hidden_size, intermediate_size, w13_out_features,
                                n_pad_w13, m_pad_w13, n_pad_w2, m_pad_w2,
                                blocks_n_w13, blocks_m_w13,
                                blocks_n_w13_meta, blocks_m_w13_meta,
                                blocks_n_w2, blocks_m_w2,
                                blocks_n_w2_meta, blocks_m_w2_meta,
                                activation_type,
                                w13_Qidxs3, w13_inv_rs2,
                                w2_Qidxs3, w2_inv_rs2,
                                codebook3):
    # MoE projects back to hidden_size after w2.
    return torch.empty((*x.shape[:-1], hidden_size),
                       dtype=torch.float16, device=x.device)
