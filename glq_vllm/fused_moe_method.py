"""GLQ FusedMoE method for vLLM — compressed MoE expert weights.

Stores per-expert GLQ compressed buffers (Qidxs, SU, SV, Wscale)
as 3D tensors (num_experts, padded_dim, n_blocks). Dequantizes
selected experts on-the-fly during apply().
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)

from glq.codebook import E8ShellCodebook
from glq.hadamard import fast_hadamard_transform

from .linear_method import _ensure_codebook, _glq_pad


def _dequant_expert_weight(Qidxs, SU, SV, Wscale, cb, out_features, in_features,
                           Qidxs2=None, inv_rs=0.0, cb2=None):
    """Dequantize one expert's GLQ weight to dense fp16."""
    m_pad, n_blocks = Qidxs.shape
    n_pad = n_blocks * 8

    W_rht = cb.decode(Qidxs.long().reshape(-1)).reshape(m_pad, n_pad).float()
    if Qidxs2 is not None and inv_rs != 0.0 and cb2 is not None:
        W_rht2 = cb2.decode(Qidxs2.long().reshape(-1)).reshape(m_pad, n_pad).float()
        W_rht = W_rht + W_rht2 * inv_rs
    W_rht = W_rht * Wscale.float()

    W = fast_hadamard_transform(W_rht.clone())
    W = W * SV.float().unsqueeze(0)
    W = fast_hadamard_transform(W.T.clone()).T
    W = W * SU.float().unsqueeze(1)

    return W[:out_features, :in_features].half()


class GLQFusedMoEMethod(FusedMoEMethodBase):
    """GLQ quantized FusedMoE method.

    Stores per-expert compressed GLQ buffers. On apply(), dequantizes
    selected experts to dense and performs the MoE computation.

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
        is_gated = getattr(layer, 'is_act_and_mul', True)
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
        layer.w13_Qidxs = nn.Parameter(
            torch.zeros(num_experts, m_pad_w13, n_blocks_w13, dtype=torch.int16),
            requires_grad=False)
        layer.w13_SU = nn.Parameter(
            torch.ones(num_experts, m_pad_w13, dtype=torch.float16),
            requires_grad=False)
        layer.w13_SV = nn.Parameter(
            torch.ones(n_pad_w13, dtype=torch.float16),
            requires_grad=False)
        layer.w13_Wscale = nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32),
            requires_grad=False)

        # w2 compressed buffers
        layer.w2_Qidxs = nn.Parameter(
            torch.zeros(num_experts, m_pad_w2, n_blocks_w2, dtype=torch.int16),
            requires_grad=False)
        layer.w2_SU = nn.Parameter(
            torch.ones(num_experts, m_pad_w2, dtype=torch.float16),
            requires_grad=False)
        layer.w2_SV = nn.Parameter(
            torch.ones(n_pad_w2, dtype=torch.float16),
            requires_grad=False)
        layer.w2_Wscale = nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32),
            requires_grad=False)

        # Stage 2 buffers (3/4bpw)
        layer.w13_Qidxs2 = nn.Parameter(
            torch.zeros(num_experts, m_pad_w13, n_blocks_w13, dtype=torch.int16),
            requires_grad=False)
        layer.w13_inv_resid_scale = nn.Parameter(
            torch.zeros(num_experts, dtype=torch.float32),
            requires_grad=False)
        layer.w2_Qidxs2 = nn.Parameter(
            torch.zeros(num_experts, m_pad_w2, n_blocks_w2, dtype=torch.int16),
            requires_grad=False)
        layer.w2_inv_resid_scale = nn.Parameter(
            torch.zeros(num_experts, dtype=torch.float32),
            requires_grad=False)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Ensure codebook is on device."""
        device = layer.w13_Qidxs.device
        bpw = getattr(self.quant_config, 'bpw', 2)

        # Check if any expert has stage2
        max_bpw = 2
        for i in range(layer.glq_num_experts):
            if layer.w13_inv_resid_scale[i].item() != 0.0:
                max_bpw = max(max_bpw, bpw)
                break

        _ensure_codebook(device, max_bpw=max_bpw)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """Dequant selected experts, apply MoE computation."""
        from .linear_method import _codebook, _codebook2_small

        cb = _codebook
        cb2 = _codebook2_small
        dtype = x.dtype
        device = x.device

        # x shape: (num_tokens, hidden_size)
        num_tokens, hidden = x.shape
        num_experts = layer.glq_num_experts
        out_dim = layer.glq_hidden_size
        inter_dim = layer.glq_intermediate_size
        w13_out = layer.glq_w13_out
        is_gated = layer.glq_is_gated
        topk = topk_ids.shape[1]

        # Output accumulator
        output = torch.zeros(num_tokens, out_dim, dtype=dtype, device=device)

        # Process each expert
        for expert_idx in range(num_experts):
            # Find tokens assigned to this expert
            mask = (topk_ids == expert_idx)  # (num_tokens, topk)
            token_mask = mask.any(dim=1)  # (num_tokens,)

            if not token_mask.any():
                continue

            # Get weights for selected tokens
            # For each token, sum the topk_weights where this expert is selected
            expert_weights = (topk_weights * mask.float()).sum(dim=1)  # (num_tokens,)
            selected_tokens = x[token_mask]  # (n_selected, hidden)
            selected_weights = expert_weights[token_mask]  # (n_selected,)

            # Dequant w13 (gate_up_proj)
            inv_rs_w13 = layer.w13_inv_resid_scale[expert_idx].item()
            has_s2_w13 = inv_rs_w13 != 0.0
            w13 = _dequant_expert_weight(
                layer.w13_Qidxs[expert_idx], layer.w13_SU[expert_idx],
                layer.w13_SV, layer.w13_Wscale[expert_idx],
                cb, w13_out, hidden,
                Qidxs2=layer.w13_Qidxs2[expert_idx] if has_s2_w13 else None,
                inv_rs=inv_rs_w13, cb2=cb2,
            )

            # Dequant w2 (down_proj)
            inv_rs_w2 = layer.w2_inv_resid_scale[expert_idx].item()
            has_s2_w2 = inv_rs_w2 != 0.0
            w2 = _dequant_expert_weight(
                layer.w2_Qidxs[expert_idx], layer.w2_SU[expert_idx],
                layer.w2_SV, layer.w2_Wscale[expert_idx],
                cb, hidden, inter_dim,
                Qidxs2=layer.w2_Qidxs2[expert_idx] if has_s2_w2 else None,
                inv_rs=inv_rs_w2, cb2=cb2,
            )

            # MLP forward: w13 → activation → w2
            h = F.linear(selected_tokens.to(w13.dtype), w13)
            if is_gated:
                gate, up = h.chunk(2, dim=-1)
                h = F.silu(gate) * up
            else:
                h = F.silu(h)
            h = F.linear(h, w2)

            # Accumulate weighted output
            output[token_mask] += h.to(dtype) * selected_weights.unsqueeze(-1)

        # Handle shared experts if present
        if shared_experts_input is not None:
            return output, shared_experts_input

        return output
