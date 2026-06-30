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

    # -- 9a. input_rht_triton: same schema as input_rht (C++), but
    #         dispatches to the Triton fallback in glq.quantized_linear.
    #         Fires when n_pad > 16384 (e.g., 70B-class MLPs). Kept
    #         symmetric so dynamo sees a clean torch.ops.glq.* call on
    #         both branches of the n_pad <= 16384 shape gate. --
    _glq_lib.define(
        "input_rht_triton(Tensor x, Tensor sv, Tensor(a!) out, int in_features, "
        "int stride_x, float rsqrt_n, int n_pad, int log_n) -> ()")
    _glq_lib.impl("input_rht_triton", _input_rht_triton_impl, dispatch_key)
    _glq_lib._register_fake("input_rht_triton", _input_rht_fake)

    # -- 9b. output_rht_triton: same schema as output_rht (C++). --
    _glq_lib.define(
        "output_rht_triton(Tensor y_rht, Tensor su, Tensor(a!) out, "
        "int out_features, int m_pad, int log_m, float rsqrt_m) -> ()")
    _glq_lib.impl("output_rht_triton", _output_rht_triton_impl, dispatch_key)
    _glq_lib._register_fake("output_rht_triton", _output_rht_fake)

    # -- 9c. gather_kv_paged_dequant: fused Triton gather + E8 dequant for
    #         the v0.3.0 KV cache hot path. Mutates the pre-allocated
    #         ``out`` tensor in-place. Preparatory work for full-graph
    #         capture; today the attention region is already a piecewise
    #         graph-break boundary (because of ``.unique()`` in
    #         ``kv_compression.py``), so this op doesn't unblock any
    #         further capture by itself. --
    _glq_lib.define(
        "gather_kv_paged_dequant("
        "Tensor idx1, Tensor idx2, Tensor idx3, Tensor idx_s1, Tensor scale, "
        "Tensor cb_primary, Tensor cb_secondary, Tensor h_mat, "
        "Tensor block_indices, Tensor(a!) out, "
        "int NB_sel, int BS, int H, int G, "
        "int idx_st_nb, int idx_st_bs, int idx_st_h, "
        "int idxu_st_nb, int idxu_st_bs, int idxu_st_h, "
        "int sc_st_nb, int sc_st_bs, int sc_st_h, "
        "int out_st_nb, int out_st_bs, int out_st_h, "
        "float rs, int n_primary, int n_secondary) -> ()")
    _glq_lib.impl("gather_kv_paged_dequant",
                  _gather_kv_paged_dequant_impl, dispatch_key)
    _glq_lib._register_fake("gather_kv_paged_dequant",
                            _gather_kv_paged_dequant_fake)

    # -- 9d. scatter_kv_paged_quant: fused Triton write-side scatter.
    #         Aliases all five destination cache buffers as outputs. When
    #         ``n_primary < 3`` or ``n_secondary < 1`` the corresponding
    #         dst args are passed as aliases of ``cache_idx1`` (matching
    #         the existing Python launcher); the kernel's constexpr
    #         ``N_PRIMARY`` / ``N_SECONDARY`` guards then skip those
    #         writes. --
    _glq_lib.define(
        "scatter_kv_paged_quant("
        "Tensor idx1_src, Tensor idx2_src, Tensor idx3_src, Tensor idx_s1_src, "
        "Tensor scale_src, "
        "Tensor(a!) cache_idx1, Tensor(b!) cache_idx2, Tensor(c!) cache_idx3, "
        "Tensor(d!) cache_idx_s1, Tensor(e!) cache_scale, "
        "Tensor slot_mapping, "
        "int nT, int H, int G, int block_size, "
        "int src_st_t, int src_st_h, "
        "int srcu_st_t, int srcu_st_h, "
        "int scsrc_st_t, int scsrc_st_h, "
        "int cache_st_nb, int cache_st_bs, int cache_st_h, "
        "int cacheu_st_nb, int cacheu_st_bs, int cacheu_st_h, "
        "int scache_st_nb, int scache_st_bs, int scache_st_h, "
        "int n_primary, int n_secondary) -> ()")
    _glq_lib.impl("scatter_kv_paged_quant",
                  _scatter_kv_paged_quant_impl, dispatch_key)
    _glq_lib._register_fake("scatter_kv_paged_quant",
                            _scatter_kv_paged_quant_fake)

    # -- 10. fused_moe_block_diag: per-expert dispatch + dequant + matmul +
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

    # -- 10b. fused_moe_grouped_gemm: grouped-GEMM MoE (sort tokens by expert +
    #         batched tensor-core GEMM per expert). IDENTICAL signature + output
    #         shape to fused_moe_block_diag, so it reuses the same fake impl.
    #         Gated activations only (the dispatch routes non-gated to block-diag). --
    if hasattr(cuda, "glq_fused_moe_grouped_gemm_cuda"):
        _glq_lib.define(
            "fused_moe_grouped_gemm(Tensor x, Tensor topk_ids, Tensor topk_weights, "
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
        _glq_lib.impl("fused_moe_grouped_gemm",
                      cuda.glq_fused_moe_grouped_gemm_cuda, dispatch_key)
        _glq_lib._register_fake("fused_moe_grouped_gemm",
                                _fused_moe_block_diag_fake)

    # -- 10c. fused_moe_e8p: grouped-GEMM MoE with the e8p tensor-core decode
    #         (int64 TC-packed Qidxs_e8p + codebook_abs grid). v1: 4bpw (E8P+E8P).
    #         Output shape identical to the shell MoE ops -> own fake (different sig). --
    if hasattr(cuda, "glq_fused_moe_e8p_cuda"):
        _glq_lib.define(
            "fused_moe_e8p(Tensor x, Tensor topk_ids, Tensor topk_weights, "
            "Tensor w13_Qidxs_e8p, Tensor w13_Qidxs2_e8p, "
            "Tensor w13_SU, Tensor w13_SV, Tensor w13_Wscale, Tensor w13_inv_rs, "
            "Tensor w2_Qidxs_e8p, Tensor w2_Qidxs2_e8p, "
            "Tensor w2_SU, Tensor w2_SV, Tensor w2_Wscale, Tensor w2_inv_rs, "
            "Tensor codebook_abs, "
            "int hidden_size, int intermediate_size, int w13_out_features, "
            "int n_pad_w13, int m_pad_w13, int n_pad_w2, int m_pad_w2, "
            "Tensor blocks_n_w13, Tensor blocks_m_w13, "
            "Tensor blocks_n_w13_meta, Tensor blocks_m_w13_meta, "
            "Tensor blocks_n_w2, Tensor blocks_m_w2, "
            "Tensor blocks_n_w2_meta, Tensor blocks_m_w2_meta, "
            "int activation_type, "
            "Tensor w13_Qidxs3_e8p, Tensor w13_Qidxs4_e8p, "
            "Tensor w13_inv_rs2, Tensor w13_inv_rs3, "
            "Tensor w2_Qidxs3_e8p, Tensor w2_Qidxs4_e8p, "
            "Tensor w2_inv_rs2, Tensor w2_inv_rs3, "
            "Tensor w13_Qidxs2_e81b, Tensor w13_Qidxs3_e81b, Tensor w13_Qidxs4_e81b, "
            "Tensor w2_Qidxs2_e81b, Tensor w2_Qidxs3_e81b, Tensor w2_Qidxs4_e81b, "
            "Tensor e81b_grid) -> Tensor")
        _glq_lib.impl("fused_moe_e8p", cuda.glq_fused_moe_e8p_cuda, dispatch_key)
        _glq_lib._register_fake("fused_moe_e8p", _fused_moe_e8p_fake)

    # -- 11. embedding_dequant: GLQ per-row embedding lookup (gather + dequant +
    #         inverse RHT) for the Gemma-4 GLQ-quantized PLE embedding. The shared
    #         helper's raw ``fast_hadamard_transform`` is a kernel dynamo can't
    #         trace, which was the last path breaking vLLM's torch.compile; wrap
    #         it as one opaque op so the embedding lookup compiles + cudagraph-
    #         captures. Real impl is the same ``_dequant_embedding_rows`` the HF
    #         path uses (out_dtype carried as ScalarType). --
    from glq.quantized_linear import _dequant_embedding_rows
    _glq_lib.define(
        "embedding_dequant(Tensor input_ids, Tensor qidxs, Tensor sv, "
        "Tensor wscale, Tensor codebook, Tensor? qidxs2, "
        "Tensor? inv_resid_scale, Tensor? codebook2, int n_pad, "
        "int embedding_dim, float embed_scale, ScalarType? out_dtype) -> Tensor")
    _glq_lib.impl("embedding_dequant", _dequant_embedding_rows, dispatch_key)
    _glq_lib._register_fake("embedding_dequant", _embedding_dequant_fake)

    # -- 12. E8P (--codebook e8p) tensor-core decode ops. Pybind-only otherwise,
    #        so torch.dynamo can't trace the e8p forward; registering them as
    #        torch.ops.glq.* lets the e8p decode path compile + cudagraph-capture
    #        like the shell. decode_matvec_e8p / decompress_packed_e8p return a
    #        tensor; the e81b ops accumulate into a pre-zeroed out arg. --
    if hasattr(cuda, "glq_decode_matvec_e8p"):
        _glq_lib.define(
            "decode_matvec_e8p(Tensor x, Tensor weights_compressed, "
            "Tensor codebook_abs) -> Tensor")
        _glq_lib.impl("decode_matvec_e8p", cuda.glq_decode_matvec_e8p, dispatch_key)
        _glq_lib._register_fake("decode_matvec_e8p", _decode_matvec_e8p_fake)

        _glq_lib.define(
            "decompress_packed_e8p(Tensor weights_compressed, "
            "Tensor codebook_abs) -> Tensor")
        _glq_lib.impl("decompress_packed_e8p", cuda.glq_decompress_packed_e8p, dispatch_key)
        _glq_lib._register_fake("decompress_packed_e8p", _decompress_packed_e8p_fake)

        _glq_lib.define(
            "lookupmatmul_e81b_k8(Tensor X, Tensor YIs, Tensor CB, "
            "Tensor(a!) Z) -> ()")
        _glq_lib.impl("lookupmatmul_e81b_k8", cuda.glq_lookupmatmul_e81b_k8, dispatch_key)
        _glq_lib._register_fake("lookupmatmul_e81b_k8", _lookupmatmul_e81b_k8_fake)

        _glq_lib.define(
            "decompress_e81b_packed(Tensor YIs, Tensor CB, Tensor(a!) Y) -> ()")
        _glq_lib.impl("decompress_e81b_packed", cuda.glq_decompress_e81b_packed, dispatch_key)
        _glq_lib._register_fake("decompress_e81b_packed", _decompress_e81b_packed_fake)

        # Fused E8P linear: the whole input_rht + N-stage decode/matmul + ×wscale +
        # output_rht as ONE opaque op (collapses ~5 dispatches/linear → 1), so the
        # decode path traces as a single node like the shell's fused_linear.
        if hasattr(cuda, "glq_fused_linear_e8p_cuda"):
            _glq_lib.define(
                "fused_linear_e8p(Tensor x, Tensor sv, Tensor su, "
                "Tensor qidxs_e8p, Tensor qidxs2_e8p, Tensor qidxs2_e81b, "
                "Tensor qidxs3_e8p, Tensor qidxs3_e81b, Tensor qidxs4_e8p, Tensor qidxs4_e81b, "
                "Tensor codebook_abs, Tensor e81b_codebook, "
                "Tensor blocks_n, Tensor blocks_m, Tensor blocks_n_meta, Tensor blocks_m_meta, "
                "float wscale, float inv_resid_scale, float inv_resid_scale2, float inv_resid_scale3, "
                "int in_features, int out_features, "
                "int n_pad, int m_pad, int log_n, int log_m) -> Tensor")
            _glq_lib.impl("fused_linear_e8p", cuda.glq_fused_linear_e8p_cuda, dispatch_key)
            _glq_lib._register_fake("fused_linear_e8p", _fused_linear_e8p_fake)


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


def _embedding_dequant_fake(input_ids, qidxs, sv, wscale, codebook, qidxs2,
                            inv_resid_scale, codebook2, n_pad, embedding_dim,
                            embed_scale, out_dtype):
    # Output: [*input_ids.shape, embedding_dim] in out_dtype (default sv.dtype),
    # on input_ids' device — matches _dequant_embedding_rows' real return.
    dt = out_dtype if out_dtype is not None else sv.dtype
    return input_ids.new_empty((*input_ids.shape, embedding_dim), dtype=dt)


def _decode_matvec_e8p_fake(x, weights_compressed, codebook_abs):
    # weights_compressed is (m_pad//16, n_pad//64, 8, 4) → output (m_pad,) fp32.
    M = weights_compressed.shape[0] * 16
    return torch.empty(M, dtype=torch.float32, device=x.device)


def _decompress_packed_e8p_fake(weights_compressed, codebook_abs):
    # (m_pad//16, n_pad//64, 8, 4) → dense (m_pad, n_pad) fp16.
    m = weights_compressed.shape[0] * 16
    n = weights_compressed.shape[1] * 64
    return torch.empty(m, n, dtype=torch.float16, device=weights_compressed.device)


def _lookupmatmul_e81b_k8_fake(X, YIs, CB, Z):
    return None  # accumulates into the pre-zeroed Z (mutating op, no return)


def _decompress_e81b_packed_fake(YIs, CB, Y):
    return None  # fills the pre-allocated Y (mutating op, no return)


def _fused_linear_e8p_fake(x, sv, su, qidxs_e8p, qidxs2_e8p, qidxs2_e81b,
                           qidxs3_e8p, qidxs3_e81b, qidxs4_e8p, qidxs4_e81b,
                           codebook_abs, e81b_codebook,
                           blocks_n, blocks_m, blocks_n_meta, blocks_m_meta,
                           wscale, inv_resid_scale, inv_resid_scale2, inv_resid_scale3,
                           in_features, out_features,
                           n_pad, m_pad, log_n, log_m):
    # (B, in_features) → (B, out_features) fp16.
    return torch.empty((x.shape[0], out_features), dtype=torch.float16, device=x.device)


def _gather_kv_paged_dequant_impl(idx1, idx2, idx3, idx_s1, scale,
                                   cb_primary, cb_secondary, h_mat,
                                   block_indices, out,
                                   NB_sel, BS, H, G,
                                   idx_st_nb, idx_st_bs, idx_st_h,
                                   idxu_st_nb, idxu_st_bs, idxu_st_h,
                                   sc_st_nb, sc_st_bs, sc_st_h,
                                   out_st_nb, out_st_bs, out_st_h,
                                   rs, n_primary, n_secondary):
    """Real-device impl for ``glq::gather_kv_paged_dequant``. Launches the
    fused Triton kernel; mutates ``out`` in place."""
    from glq_vllm.e8_paged_cache import _fused_gather_dequant_kernel
    grid = (NB_sel, BS, H)
    _fused_gather_dequant_kernel[grid](
        idx1, idx2, idx3, idx_s1, scale,
        cb_primary, cb_secondary, h_mat,
        block_indices, out,
        NB_sel,
        BS, H, G,
        idx_st_nb, idx_st_bs, idx_st_h,
        idxu_st_nb, idxu_st_bs, idxu_st_h,
        sc_st_nb, sc_st_bs, sc_st_h,
        out_st_nb, out_st_bs, out_st_h,
        RS=rs,
        N_PRIMARY=n_primary,
        N_SECONDARY=n_secondary,
    )


def _gather_kv_paged_dequant_fake(idx1, idx2, idx3, idx_s1, scale,
                                   cb_primary, cb_secondary, h_mat,
                                   block_indices, out,
                                   NB_sel, BS, H, G,
                                   idx_st_nb, idx_st_bs, idx_st_h,
                                   idxu_st_nb, idxu_st_bs, idxu_st_h,
                                   sc_st_nb, sc_st_bs, sc_st_h,
                                   out_st_nb, out_st_bs, out_st_h,
                                   rs, n_primary, n_secondary):
    pass  # mutates ``out`` in place


def _scatter_kv_paged_quant_impl(idx1_src, idx2_src, idx3_src, idx_s1_src,
                                  scale_src,
                                  cache_idx1, cache_idx2, cache_idx3,
                                  cache_idx_s1, cache_scale,
                                  slot_mapping,
                                  nT, H, G, block_size,
                                  src_st_t, src_st_h,
                                  srcu_st_t, srcu_st_h,
                                  scsrc_st_t, scsrc_st_h,
                                  cache_st_nb, cache_st_bs, cache_st_h,
                                  cacheu_st_nb, cacheu_st_bs, cacheu_st_h,
                                  scache_st_nb, scache_st_bs, scache_st_h,
                                  n_primary, n_secondary):
    """Real-device impl for ``glq::scatter_kv_paged_quant``. Launches the
    fused Triton scatter kernel; mutates the destination cache buffers
    in place."""
    from glq_vllm.e8_paged_cache import _fused_scatter_qt_kernel
    grid = (nT, H)
    _fused_scatter_qt_kernel[grid](
        idx1_src, idx2_src, idx3_src, idx_s1_src, scale_src,
        cache_idx1, cache_idx2, cache_idx3, cache_idx_s1, cache_scale,
        slot_mapping,
        nT, H, G, block_size,
        src_st_t, src_st_h,
        srcu_st_t, srcu_st_h,
        scsrc_st_t, scsrc_st_h,
        cache_st_nb, cache_st_bs, cache_st_h,
        cacheu_st_nb, cacheu_st_bs, cacheu_st_h,
        scache_st_nb, scache_st_bs, scache_st_h,
        N_PRIMARY=n_primary,
        N_SECONDARY=n_secondary,
    )


def _scatter_kv_paged_quant_fake(*args, **kwargs):
    pass  # mutates destination cache buffers in place


def _input_rht_triton_impl(x, sv, out, in_features, stride_x, rsqrt_n,
                            n_pad, log_n):
    """Real-device impl for ``glq::input_rht_triton``. Same semantics as
    the C++ kernel but uses the Triton fallback that handles the
    n_pad > 16384 path. Mutates ``out`` in place."""
    from glq.quantized_linear import _input_rht_kernel
    B = x.shape[0]
    _input_rht_kernel[(B,)](
        x, sv, out, in_features, stride_x,
        rsqrt_n, N=n_pad, LOG_N=log_n, num_warps=8)


def _output_rht_triton_impl(y_rht, su, out, out_features, m_pad, log_m,
                             rsqrt_m):
    """Real-device impl for ``glq::output_rht_triton``. Mutates ``out``."""
    from glq.quantized_linear import _output_rht_kernel
    B = y_rht.shape[0]
    output_fp16 = (out.dtype == torch.float16)
    _output_rht_kernel[(B,)](
        y_rht, su, out, out_features,
        y_rht.stride(0), out.stride(0),
        rsqrt_m, OUTPUT_FP16=output_fp16,
        M=m_pad, LOG_M=log_m, num_warps=8)


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


def _fused_moe_e8p_fake(x, topk_ids, topk_weights,
                        w13_Qidxs_e8p, w13_Qidxs2_e8p,
                        w13_SU, w13_SV, w13_Wscale, w13_inv_rs,
                        w2_Qidxs_e8p, w2_Qidxs2_e8p,
                        w2_SU, w2_SV, w2_Wscale, w2_inv_rs,
                        codebook_abs,
                        hidden_size, intermediate_size, w13_out_features,
                        n_pad_w13, m_pad_w13, n_pad_w2, m_pad_w2,
                        blocks_n_w13, blocks_m_w13,
                        blocks_n_w13_meta, blocks_m_w13_meta,
                        blocks_n_w2, blocks_m_w2,
                        blocks_n_w2_meta, blocks_m_w2_meta,
                        activation_type,
                        w13_Qidxs3_e8p, w13_Qidxs4_e8p, w13_inv_rs2, w13_inv_rs3,
                        w2_Qidxs3_e8p, w2_Qidxs4_e8p, w2_inv_rs2, w2_inv_rs3,
                        w13_Qidxs2_e81b, w13_Qidxs3_e81b, w13_Qidxs4_e81b,
                        w2_Qidxs2_e81b, w2_Qidxs3_e81b, w2_Qidxs4_e81b, e81b_grid):
    # Same output contract as the shell grouped MoE op: (num_tokens, hidden) fp16.
    return torch.empty((*x.shape[:-1], hidden_size),
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
