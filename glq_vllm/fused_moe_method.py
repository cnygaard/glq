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
    _ensure_codebook, _glq_pad, _make_glq_param, _glq_apply_shard,
)


def _apply_activation(h, activation):
    """Apply MoE activation function (gated or non-gated).

    activation can be a string (vLLM 0.16) or MoEActivation enum (vLLM 0.18+).
    """
    act = activation.value if hasattr(activation, 'value') else str(activation) if activation else "silu"

    # Gated activations: split input into gate and up projections
    if act == "silu":
        gate, up = h.chunk(2, dim=-1)
        return F.silu(gate) * up
    elif act == "gelu":
        gate, up = h.chunk(2, dim=-1)
        return F.gelu(gate) * up
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
        m_pad_w13 = _glq_pad(w13_out)
        n_pad_w13 = _glq_pad(hidden_size)
        n_blocks_w13 = n_pad_w13 // 8

        # w2 = down_proj
        m_pad_w2 = _glq_pad(hidden_size)
        n_pad_w2 = _glq_pad(intermediate_size_per_partition)
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

        # Cache log2 values and per-expert scalars for fast apply()
        layer.glq_log_n_w13 = int(math.log2(layer.glq_n_pad_w13))
        layer.glq_log_m_w13 = int(math.log2(layer.glq_m_pad_w13))
        layer.glq_log_n_w2 = int(math.log2(layer.glq_n_pad_w2))
        layer.glq_log_m_w2 = int(math.log2(layer.glq_m_pad_w2))

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
        return {"silu": 0, "gelu": 1, "relu2": 2,
                "silu_no_mul": 3, "gelu_no_mul": 4, "relu2_no_mul": 5}.get(act, 5)

    @torch.compiler.disable
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fused dequant+matmul for active experts via CUDA kernels."""
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

        # Try fused C++ MoE path
        if (_ik._try_load_cuda_ext()
                and hasattr(_ik._glq_cuda, 'glq_fused_moe_cuda')
                and layer.glq_n_pad_w13 <= 16384
                and layer.glq_m_pad_w13 <= 16384
                and layer.glq_n_pad_w2 <= 16384
                and layer.glq_m_pad_w2 <= 16384):
            _empty_i16 = torch.empty(0, dtype=torch.int16, device=device)
            _empty_f16 = torch.empty(0, dtype=torch.float16, device=device)
            cb_half = cb.codebook_half if cb is not None else _empty_f16
            cb2_half = cb2.codebook_half if cb2 is not None else _empty_f16

            output = _ik._glq_cuda.glq_fused_moe_cuda(
                x.half().contiguous(),
                topk_ids,
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
            )
            if dtype != torch.float16:
                output = output.to(dtype)
            return output

        # Fallback: Python expert loop
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
            )

            output[token_mask] += h.to(dtype) * selected_weights.unsqueeze(-1)

        return output
