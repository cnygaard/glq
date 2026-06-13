"""GLQ FusedMoE method for vLLM — compressed MoE expert weights.

Stores per-expert GLQ compressed buffers (Qidxs, SU, SV, Wscale)
as 3D tensors (num_experts, padded_dim, n_blocks). Uses fused
dequant+matmul CUDA kernels (same as non-MoE layers) for each
active expert instead of materializing dense weights.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)

from .linear_method import (
    _ensure_codebook, _glq_pad, _is_pow2, _make_glq_param, _glq_apply_shard,
    _detect_block_diag, _pack_block_meta,
)


def _round8(n: int) -> int:
    """Round up to a multiple of 8 (GLQ packs 8 weights per E8 block)."""
    return ((n + 7) // 8) * 8


def _apply_activation(h, activation):
    """Apply MoE activation function (gated or non-gated).

    activation can be a string (vLLM 0.16) or MoEActivation enum (vLLM 0.18+).
    """
    act = activation.value if hasattr(activation, 'value') else str(activation) if activation else "silu"

    # Gated activations: split input into gate and up projections
    if act == "silu":
        gate, up = h.chunk(2, dim=-1)
        return F.silu(gate) * up
    elif act in ("gelu", "gelu_tanh", "gelu_pytorch_tanh"):
        # gemma-4 experts use gelu_pytorch_tanh (vLLM names it "gelu_tanh").
        # Matching only the bare "gelu" string sent these to the non-gated
        # silu fallback below -> structurally wrong output (garbage).
        gate, up = h.chunk(2, dim=-1)
        approx = "tanh" if "tanh" in act else "none"
        return F.gelu(gate, approximate=approx) * up
    elif act == "relu2":
        gate, up = h.chunk(2, dim=-1)
        return F.relu(gate).square() * up

    # Non-gated activations
    elif act == "silu_no_mul":
        return F.silu(h)
    elif act == "gelu_no_mul":
        return F.gelu(h)
    elif act == "relu2_no_mul":
        return F.relu(h).square()

    # Fallback
    return F.silu(h)


class GLQFusedMoEMethod(FusedMoEMethodBase):
    """GLQ quantized FusedMoE method.

    Stores per-expert compressed GLQ buffers. On apply(), uses fused
    dequant+matmul CUDA kernels for active experts only.

    Weight naming convention (matches NemotronH expert_params_mapping):
    - w13_Qidxs: (num_experts, m_pad_w13, n_blocks) int16
    - w13_SU: (num_experts, m_pad_w13) fp16
    - w13_SV: (n_pad) fp16  — shared across experts
    - w13_Wscale: (num_experts,) fp32
    - w2_Qidxs, w2_SU, w2_SV, w2_Wscale: same pattern for down_proj
    """

    def __init__(self, quant_config, moe=None):
        super().__init__(moe)
        self.quant_config = quant_config

    def get_fused_moe_quant_config(self, layer):
        return None

    def create_weights(
        self,
        layer: nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_loader = extra_weight_attrs.get("weight_loader")

        # w13 = gate_up_proj (or just up_proj for non-gated)
        is_gated = getattr(self.moe, 'is_act_and_mul', None)
        if is_gated is None:
            is_gated = getattr(layer, 'is_act_and_mul', False)
        w13_out = 2 * intermediate_size_per_partition if is_gated else intermediate_size_per_partition
        # GLQ (post-v0.2.9) stores BLOCK-DIAGONAL artifacts: m_pad/n_pad equal the
        # TRUE out/in dims (n rounded to a multiple of 8), NOT padded to a power of
        # 2. Allocate the buffers at those true dims so (a) the gate/up loader's
        # `loaded*2 == slot` row-split lands correctly when 2*intermediate isn't a
        # power of 2 (e.g. gemma-4: 2*704=1408), and (b) the per-expert artifact
        # shapes match. (For power-of-2 MoE dims, e.g. sarvam, this is identical to
        # the old _glq_pad and the fast fused kernel still applies.)
        m_pad_w13 = w13_out
        n_pad_w13 = _round8(hidden_size)
        n_blocks_w13 = n_pad_w13 // 8

        # w2 = down_proj
        m_pad_w2 = hidden_size
        n_pad_w2 = _round8(intermediate_size_per_partition)
        n_blocks_w2 = n_pad_w2 // 8

        # Store metadata for apply()
        layer.glq_num_experts = num_experts
        layer.glq_hidden_size = hidden_size
        layer.glq_intermediate_size = intermediate_size_per_partition
        layer.glq_w13_out = w13_out
        layer.glq_is_gated = is_gated
        layer.glq_m_pad_w13 = m_pad_w13
        layer.glq_m_pad_w2 = m_pad_w2
        layer.glq_n_pad_w13 = n_pad_w13
        layer.glq_n_pad_w2 = n_pad_w2

        # w13 compressed buffers
        layer.w13_Qidxs = _make_glq_param(
            torch.zeros(num_experts, m_pad_w13, n_blocks_w13, dtype=torch.int16))
        layer.w13_SU = _make_glq_param(
            torch.ones(num_experts, m_pad_w13, dtype=torch.float16))
        layer.w13_SV = _make_glq_param(
            torch.ones(n_pad_w13, dtype=torch.float16))
        layer.w13_Wscale = _make_glq_param(
            torch.ones(num_experts, dtype=torch.float32))

        # w2 compressed buffers
        layer.w2_Qidxs = _make_glq_param(
            torch.zeros(num_experts, m_pad_w2, n_blocks_w2, dtype=torch.int16))
        layer.w2_SU = _make_glq_param(
            torch.ones(num_experts, m_pad_w2, dtype=torch.float16))
        layer.w2_SV = _make_glq_param(
            torch.ones(n_pad_w2, dtype=torch.float16))
        layer.w2_Wscale = _make_glq_param(
            torch.ones(num_experts, dtype=torch.float32))

        # Stage 2 buffers (3/4bpw)
        layer.w13_Qidxs2 = _make_glq_param(
            torch.zeros(num_experts, m_pad_w13, n_blocks_w13, dtype=torch.int16))
        layer.w13_inv_resid_scale = _make_glq_param(
            torch.zeros(num_experts, dtype=torch.float32))
        layer.w2_Qidxs2 = _make_glq_param(
            torch.zeros(num_experts, m_pad_w2, n_blocks_w2, dtype=torch.int16))
        layer.w2_inv_resid_scale = _make_glq_param(
            torch.zeros(num_experts, dtype=torch.float32))

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Ensure codebook is on device. Cache per-expert metadata for fast apply()."""
        device = layer.w13_Qidxs.device
        bpw = getattr(self.quant_config, 'bpw', 2)

        # Update PADDED dims from actual loaded weight shapes
        if layer.w13_Qidxs.dim() == 3 and layer.w13_Qidxs.shape[1] > 0:
            layer.glq_m_pad_w13 = layer.w13_Qidxs.shape[1]
            layer.glq_n_pad_w13 = layer.w13_Qidxs.shape[2] * 8
            if layer.glq_m_pad_w13 <= layer.glq_intermediate_size:
                layer.glq_is_gated = False
                layer.glq_w13_out = layer.glq_intermediate_size
        if layer.w2_Qidxs.dim() == 3 and layer.w2_Qidxs.shape[1] > 0:
            layer.glq_m_pad_w2 = layer.w2_Qidxs.shape[1]
            layer.glq_n_pad_w2 = layer.w2_Qidxs.shape[2] * 8

        # Check if any expert has stage2
        max_bpw = 2
        for i in range(layer.glq_num_experts):
            if layer.w13_inv_resid_scale[i].item() != 0.0:
                max_bpw = max(max_bpw, bpw)
                break

        _ensure_codebook(device, max_bpw=max_bpw)

        # Ensure all weight tensors are on GPU
        for attr in ['w13_Qidxs', 'w13_SU', 'w13_SV', 'w13_Wscale',
                     'w2_Qidxs', 'w2_SU', 'w2_SV', 'w2_Wscale',
                     'w13_Qidxs2', 'w13_inv_resid_scale',
                     'w2_Qidxs2', 'w2_inv_resid_scale']:
            t = getattr(layer, attr, None)
            if t is not None and t.device != device:
                setattr(layer, attr, torch.nn.Parameter(t.data.to(device), requires_grad=False))

        # Cache log2 values and per-expert scalars for fast apply().
        # CRITICAL: for non-power-of-2 (block-diagonal) dims the log is 0 — the
        # C++ RHT ops treat log==0 as the "decompose m_pad/n_pad into pow2 blocks"
        # sentinel. Passing int(log2(non_pow2)) (a truncated log) makes the op
        # index the buffer as if it were the smaller 2^log size -> out-of-bounds
        # illegal memory access. Mirror the linear path (linear_method.py:784).
        def _glog(p):
            return int(math.log2(p)) if _is_pow2(p) else 0
        layer.glq_log_n_w13 = _glog(layer.glq_n_pad_w13)
        layer.glq_log_m_w13 = _glog(layer.glq_m_pad_w13)
        layer.glq_log_n_w2 = _glog(layer.glq_n_pad_w2)
        layer.glq_log_m_w2 = _glog(layer.glq_m_pad_w2)

        # Block-diagonal metadata for non-pow2 dims. The per-expert fallback
        # `_glq_apply_shard` ONLY applies the (block-diagonal) Hadamard via the
        # fused_linear_block_diag op when this meta is supplied — without it the
        # legacy path runs with log==0 (= no transform) and the weights decode to
        # ~zero. Build it exactly like the non-MoE linear path (linear_method.py:775).
        def _bd_meta(m_pad, n_pad):
            is_bd, blocks_m, blocks_n = _detect_block_diag(m_pad, n_pad)
            if not is_bd:
                return None
            return {
                "blocks_n_tensor": torch.tensor(blocks_n, dtype=torch.int64, device='cpu'),
                "blocks_m_tensor": torch.tensor(blocks_m, dtype=torch.int64, device='cpu'),
                "blocks_n_meta_gpu": _pack_block_meta(blocks_n).to(device, non_blocking=True),
                "blocks_m_meta_gpu": _pack_block_meta(blocks_m).to(device, non_blocking=True),
            }
        layer.glq_bd_meta_w13 = _bd_meta(layer.glq_m_pad_w13, layer.glq_n_pad_w13)
        layer.glq_bd_meta_w2 = _bd_meta(layer.glq_m_pad_w2, layer.glq_n_pad_w2)

        n = layer.glq_num_experts
        layer.glq_w13_wscale = [layer.w13_Wscale[i].item() for i in range(n)]
        layer.glq_w2_wscale = [layer.w2_Wscale[i].item() for i in range(n)]
        layer.glq_w13_inv_rs = [layer.w13_inv_resid_scale[i].item() for i in range(n)]
        layer.glq_w2_inv_rs = [layer.w2_inv_resid_scale[i].item() for i in range(n)]

        # Remove weight_loader from params (prevents vLLM v1 serialization issues)
        for name, param in layer.named_parameters():
            if hasattr(param, 'weight_loader'):
                del param.weight_loader

    @staticmethod
    def _activation_type(activation) -> int:
        """Map activation string/enum to integer for C++ dispatch."""
        act = activation.value if hasattr(activation, 'value') else str(activation) if activation else "silu"
        return {"silu": 0, "gelu": 1, "gelu_tanh": 1, "gelu_pytorch_tanh": 1,
                "relu2": 2, "silu_no_mul": 3, "gelu_no_mul": 4,
                "relu2_no_mul": 5}.get(act, 5)

    @torch.compiler.disable
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: nn.Module | None = None,
        shared_experts_input: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fused dequant+matmul for active experts via CUDA kernels.

        ``shared_experts`` (vLLM >= 0.22 FusedMoE) is accepted but ignored: GLQ
        does not fuse the shared expert into the routed-expert kernel, so the
        runner computes it separately (its Linears are served by
        GLQLinearMethod via the layer_bpw whitelist). ``**kwargs`` absorbs any
        further runner-only kwargs added by newer vLLM versions.
        """
        from . import linear_method as _lm
        from glq import inference_kernel as _ik

        cb = _lm._codebook
        cb2 = _lm._codebook2_small
        dtype = x.dtype
        device = x.device

        num_tokens, hidden = x.shape
        out_dim = hidden
        inter_dim = layer.glq_intermediate_size
        w13_out = layer.glq_w13_out
        activation = getattr(layer, 'activation', None)

        # Try fused C++ MoE path (GLQ_MOE_FORCE_FALLBACK=1 forces the per-expert
        # Python loop instead — used to A/B-isolate the fused kernel vs the
        # weight load during bring-up of a new MoE architecture).
        # The fused MoE kernel assumes power-of-2 padded dims (it shifts by
        # log2(pad)). Block-diagonal experts (non-pow2 true dims, e.g. gemma-4:
        # hidden 2816, 2*inter 1408, inter 704) must take the block-diag-capable
        # per-expert fallback (`_glq_apply_shard`, the same path the non-MoE GLQ
        # linears use for o_proj/down_proj). Fused-kernel block-diag is a perf TODO.
        import os as _os
        _pow2_dims = (_is_pow2(layer.glq_n_pad_w13) and _is_pow2(layer.glq_m_pad_w13)
                      and _is_pow2(layer.glq_n_pad_w2) and _is_pow2(layer.glq_m_pad_w2))
        if (_os.environ.get("GLQ_MOE_FORCE_FALLBACK", "0") == "0"
                and _pow2_dims
                and _ik._try_load_cuda_ext()
                and hasattr(_ik._glq_cuda, 'glq_fused_moe_cuda')
                and layer.glq_n_pad_w13 <= 16384
                and layer.glq_m_pad_w13 <= 16384
                and layer.glq_n_pad_w2 <= 16384
                and layer.glq_m_pad_w2 <= 16384):
            _empty_i16 = torch.empty(0, dtype=torch.int16, device=device)
            _empty_f16 = torch.empty(0, dtype=torch.float16, device=device)
            _empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
            cb_half = cb.codebook_half if cb is not None else _empty_f16
            cb2_half = cb2.codebook_half if cb2 is not None else _empty_f16

            output = _ik._glq_cuda.glq_fused_moe_cuda(
                x.half().contiguous(),
                topk_ids.to(torch.int64),   # vLLM >=0.22 gives int32; kernel wants int64
                topk_weights,
                # w13
                layer.w13_Qidxs, layer.w13_SU, layer.w13_SV,
                layer.w13_Wscale, layer.w13_Qidxs2, layer.w13_inv_resid_scale,
                # w2
                layer.w2_Qidxs, layer.w2_SU, layer.w2_SV,
                layer.w2_Wscale, layer.w2_Qidxs2, layer.w2_inv_resid_scale,
                # Codebooks
                cb_half, cb2_half,
                # Dims
                hidden, inter_dim, w13_out,
                layer.glq_n_pad_w13, layer.glq_m_pad_w13,
                layer.glq_n_pad_w2, layer.glq_m_pad_w2,
                layer.glq_log_n_w13, layer.glq_log_m_w13,
                layer.glq_log_n_w2, layer.glq_log_m_w2,
                self._activation_type(activation),
                # stage-3 RVQ: pass empty tensors (2-stage). pybind requires real
                # tensors here — the C++ `= None` default can't bind to torch::Tensor.
                _empty_i16, _empty_f32, _empty_i16, _empty_f32, _empty_f16,
            )
            if dtype != torch.float16:
                output = output.to(dtype)
            return output

        # Block-diagonal fused path (non-pow2 expert dims, e.g. gemma-4: hidden
        # 2816, 2*inter 1408, inter 704). Single host call into the SAME
        # glq_fused_moe_block_diag_cuda kernel the HF E8RHTFusedExperts path uses,
        # routed through the registered torch.ops.glq.fused_moe_block_diag so it is
        # one opaque, host-sync-free op (no topk_ids.unique()/.item()/boolean-mask
        # loop). That sync-freedom is what lets vLLM's FULL cudagraph capture the
        # MoE decode step — the Python fallback below cannot be captured.
        #
        # The kernel iterates (token, expert) pairs at B=1, so it wins decode but
        # loses multi-token prefill (where the Python loop's per-expert batched
        # matmul is faster — see glq.fused_experts._FUSED_KERNEL_MAX_TOKENS). Gate
        # on a token cap (env-tunable) so larger batches take the per-expert
        # batched fallback while small-batch decode takes the fused op.
        # GLQ_MOE_FORCE_FALLBACK=1 disables this path entirely (A/B isolation).
        #
        # EAGER default cap = 4 (matches glq.fused_experts._FUSED_KERNEL_MAX_TOKENS):
        # the fused kernel iterates (token, expert) at B=1, so it wins small-batch
        # decode (measured 26B-A4B b1: 11.4 vs 4.8 tok/s Python loop, 2.4x) but
        # loses multi-token batches (b32: 39.3 vs 45.4), where grouping tokens
        # per-expert (the Python loop) is faster. Raising the cap only pays off
        # once the path is cudagraph-captured (decode replays at ~0 launch cost) —
        # that lands with the Stage-2 device-dispatched kernel; until then keep it
        # to decode-sized batches so large-batch/prefill throughput never regresses.
        _bd_cap = int(_os.environ.get("GLQ_MOE_BD_MAX_TOKENS", "4"))
        if (_os.environ.get("GLQ_MOE_FORCE_FALLBACK", "0") == "0"
                and not _pow2_dims
                and layer.glq_bd_meta_w13 is not None
                and layer.glq_bd_meta_w2 is not None
                and num_tokens <= _bd_cap
                and _ik._try_load_cuda_ext()
                and hasattr(_ik._glq_cuda, 'glq_fused_moe_block_diag_cuda')
                and hasattr(torch.ops, 'glq')
                and hasattr(torch.ops.glq, 'fused_moe_block_diag')
                and layer.glq_n_pad_w13 <= 16384
                and layer.glq_m_pad_w13 <= 16384
                and layer.glq_n_pad_w2 <= 16384
                and layer.glq_m_pad_w2 <= 16384):
            bd13 = layer.glq_bd_meta_w13
            bd2 = layer.glq_bd_meta_w2
            _empty_i16 = torch.empty(0, dtype=torch.int16, device=device)
            _empty_f16 = torch.empty(0, dtype=torch.float16, device=device)
            _empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
            cb_half = cb.codebook_half if cb is not None else _empty_f16
            cb2_half = cb2.codebook_half if cb2 is not None else _empty_f16

            output = torch.ops.glq.fused_moe_block_diag(
                x.half().contiguous(),
                topk_ids.to(torch.int64),
                topk_weights.float().contiguous(),
                # w13
                layer.w13_Qidxs, layer.w13_SU, layer.w13_SV,
                layer.w13_Wscale, layer.w13_Qidxs2, layer.w13_inv_resid_scale,
                # w2
                layer.w2_Qidxs, layer.w2_SU, layer.w2_SV,
                layer.w2_Wscale, layer.w2_Qidxs2, layer.w2_inv_resid_scale,
                # Codebooks
                cb_half, cb2_half,
                # Dims
                hidden, inter_dim, w13_out,
                layer.glq_n_pad_w13, layer.glq_m_pad_w13,
                layer.glq_n_pad_w2, layer.glq_m_pad_w2,
                # Block-diagonal meta (CPU block-size tensors + GPU packed meta),
                # same dict the linear fallback already consumes successfully.
                bd13["blocks_n_tensor"], bd13["blocks_m_tensor"],
                bd13["blocks_n_meta_gpu"], bd13["blocks_m_meta_gpu"],
                bd2["blocks_n_tensor"], bd2["blocks_m_tensor"],
                bd2["blocks_n_meta_gpu"], bd2["blocks_m_meta_gpu"],
                self._activation_type(activation),
                # stage-3 RVQ: empty tensors (2-stage path).
                _empty_i16, _empty_f32, _empty_i16, _empty_f32, _empty_f16,
            )
            if dtype != torch.float16:
                output = output.to(dtype)
            return output

        # Fallback: Python expert loop (block-diag-capable via _glq_apply_shard).
        # Defensive: the shared SV vectors must be on the compute device (the
        # per-expert dequant kernel asserts sv.is_cuda).
        if layer.w13_SV.device != device:
            layer.w13_SV = nn.Parameter(layer.w13_SV.data.to(device), requires_grad=False)
        if layer.w2_SV.device != device:
            layer.w2_SV = nn.Parameter(layer.w2_SV.data.to(device), requires_grad=False)
        output = torch.zeros(num_tokens, out_dim, dtype=dtype, device=device)
        active_experts = topk_ids.unique()

        for expert_idx_t in active_experts:
            expert_idx = expert_idx_t.item()

            mask = (topk_ids == expert_idx)
            token_mask = mask.any(dim=1)
            expert_weights = (topk_weights * mask.float()).sum(dim=1)
            selected_tokens = x[token_mask]
            selected_weights = expert_weights[token_mask]

            inv_rs_w13 = layer.glq_w13_inv_rs[expert_idx]
            has_s2_w13 = inv_rs_w13 != 0.0
            h = _glq_apply_shard(
                selected_tokens, device, cb, cb2,
                Qidxs=layer.w13_Qidxs[expert_idx],
                SU=layer.w13_SU[expert_idx],
                SV=layer.w13_SV,
                wscale=layer.glq_w13_wscale[expert_idx],
                has_stage2=has_s2_w13,
                inv_rs=inv_rs_w13,
                Qidxs2=layer.w13_Qidxs2[expert_idx] if has_s2_w13 else None,
                out_features=w13_out, in_features=hidden,
                m_pad=layer.glq_m_pad_w13, n_pad=layer.glq_n_pad_w13,
                log_n=layer.glq_log_n_w13, log_m=layer.glq_log_m_w13,
                block_diag_meta=layer.glq_bd_meta_w13,
            )

            h = _apply_activation(h, activation)

            inv_rs_w2 = layer.glq_w2_inv_rs[expert_idx]
            has_s2_w2 = inv_rs_w2 != 0.0
            h = _glq_apply_shard(
                h, device, cb, cb2,
                Qidxs=layer.w2_Qidxs[expert_idx],
                SU=layer.w2_SU[expert_idx],
                SV=layer.w2_SV,
                wscale=layer.glq_w2_wscale[expert_idx],
                has_stage2=has_s2_w2,
                inv_rs=inv_rs_w2,
                Qidxs2=layer.w2_Qidxs2[expert_idx] if has_s2_w2 else None,
                out_features=out_dim, in_features=inter_dim,
                m_pad=layer.glq_m_pad_w2, n_pad=layer.glq_n_pad_w2,
                log_n=layer.glq_log_n_w2, log_m=layer.glq_log_m_w2,
                block_diag_meta=layer.glq_bd_meta_w2,
            )

            output[token_mask] += h.to(dtype) * selected_weights.unsqueeze(-1)

        return output
