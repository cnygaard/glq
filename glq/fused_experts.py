"""Drop-in replacement for `transformers.models.nemotron_h.modeling_nemotron_h.NemotronHExperts`.

The native HF implementation packs all `n_routed_experts` expert MLPs into
stacked tensors (`up_proj` shape `(num_experts, intermediate, hidden)` and
`down_proj`) and dispatches via a per-expert Python loop in `.forward`. GLQ's
quantized weights are stored per-expert (one set of `Qidxs/SU/SV/Wscale/...`
per expert), so we mirror the native interface but back it with a
`ModuleList` of per-expert `_ExpertPair`s. Each pair is two `E8RHTLinear`s,
one for `up_proj` and one for `down_proj`.

This keeps the native model's routing intact and lets each expert use the
full N-stage RVQ inference path that `E8RHTLinear` already supports
(`glq_fused_linear_cuda` handles 1–4 stages). The per-expert Python overhead
is acceptable as a v1: the sensitivity allocator can put 5bpw experts
(3-stage RVQ) next to 4bpw experts (2-stage RVQ) within the same MoE layer,
and the existing `glq_fused_moe_cuda` kernel only handles 2-stage. A v2
fused-kernel path can be added later for layers with uniform-bpw experts.

The internal child-module layout matches the trust-remote-code checkpoint
key layout exactly: `experts.{i}.up_proj.Qidxs`, `experts.{i}.down_proj.SU`,
etc., so HF's normal state-dict loader can install the buffers without any
key remapping.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .quantized_linear import E8RHTLinear


class _ExpertPair(nn.Module):
    """Single MoE expert: up_proj -> activation -> down_proj. Non-gated."""

    def __init__(self, input_dim: int, intermediate_dim: int,
                 output_dim: int, block_diagonal: bool = True):
        super().__init__()
        self.up_proj = E8RHTLinear(
            input_dim, intermediate_dim,
            bias=False, block_diagonal=block_diagonal,
        )
        self.down_proj = E8RHTLinear(
            intermediate_dim, output_dim,
            bias=False, block_diagonal=block_diagonal,
        )


class E8RHTFusedExperts(nn.Module):
    """Mimics `NemotronHExperts.forward(hidden_states, top_k_index, top_k_weights)`.

    Per-expert `_ExpertPair`s live at the top level under integer-string
    keys (`self._modules["0"]`, `"1"`, ... `"127"`), so state-dict keys
    load straight as `experts.{i}.up_proj.Qidxs` matching the
    trust-remote-code checkpoint layout. We hand-populate `_modules` rather
    than subclassing `nn.ModuleList` because `nn.ModuleList`'s `__setattr__`
    auto-numbers any submodule attribute (including `act_fn`), which would
    collide with the expert indices.

    NemotronH's MoE is **non-gated** (only up_proj followed by activation
    followed by down_proj), unlike Mixtral.

    State-dict keys accepted (matched directly through HF's normal loader):

        {i}.up_proj.Qidxs / SU / SV / Wscale / Qidxs2 / inv_resid_scale / ...
        {i}.down_proj.Qidxs / SU / SV / Wscale / ...
    """

    def __init__(self, config, block_diagonal: bool = True):
        super().__init__()
        self.num_experts = int(config.n_routed_experts)
        self.hidden_dim = int(config.hidden_size)
        self.intermediate_dim = int(config.moe_intermediate_size)
        latent = getattr(config, "moe_latent_size", None)
        self.input_dim = int(latent) if latent is not None else self.hidden_dim
        self.output_dim = self.input_dim

        # Lazy-import to avoid hard-deps at unit-test time.
        from transformers.activations import ACT2FN
        # `act_fn` is a regular submodule attribute. Activations like
        # ReLUSquaredActivation are `nn.Module` instances but carry no
        # learnable parameters, so they don't add state-dict keys.
        self.act_fn = ACT2FN[config.mlp_hidden_act]

        # Populate `_modules` directly so the per-expert children appear
        # under integer-string keys at the top level. nn.Module's
        # __setattr__ machinery then exposes `self["0"]`-style access via
        # `__getattr__` (and we provide explicit `__getitem__` below).
        for i in range(self.num_experts):
            pair = _ExpertPair(
                self.input_dim, self.intermediate_dim, self.output_dim,
                block_diagonal=block_diagonal,
            )
            self._modules[str(i)] = pair

    def __getitem__(self, idx: int) -> _ExpertPair:
        return self._modules[str(idx)]  # type: ignore[return-value]

    def __iter__(self):
        for i in range(self.num_experts):
            yield self._modules[str(i)]

    def __len__(self) -> int:
        return self.num_experts

    def forward(self, hidden_states: torch.Tensor,
                top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        """Mirror of `NemotronHExperts.forward`.

        hidden_states: ``(num_tokens, hidden_or_latent)``
        top_k_index:   ``(num_tokens, top_k)`` int64
        top_k_weights: ``(num_tokens, top_k)`` float — summed routing weights
        """
        final_hidden_states = torch.zeros_like(
            hidden_states, dtype=top_k_weights.dtype)

        # Build the same expert_mask the native impl uses:
        #   shape (num_experts, top_k, num_tokens)
        with torch.no_grad():
            expert_mask = nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero().squeeze(-1)

        for expert_idx_t in expert_hit:
            expert_idx = int(expert_idx_t)
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue

            current_state = hidden_states[token_idx]
            pair = self[expert_idx]  # ModuleList __getitem__
            # up_proj -> activation -> down_proj. No gating.
            h = pair.up_proj(current_state)
            h = self.act_fn(h)
            h = pair.down_proj(h)

            # Apply this expert's routing weight per token.
            h = h * top_k_weights[token_idx, top_k_pos, None]

            final_hidden_states.index_add_(
                0, token_idx, h.to(final_hidden_states.dtype))

        return final_hidden_states.to(hidden_states.dtype)


def _replace_nemotron_h_experts(model: nn.Module, block_diagonal: bool = True) -> int:
    """Walk `model` and replace each `NemotronHExperts` with `E8RHTFusedExperts`.

    Returns the number of substitutions performed. No-op if the native
    `nemotron_h` integration isn't installed (i.e. we're loading via
    `trust_remote_code` where experts are already individual `nn.Linear`s).
    """
    try:
        from transformers.models.nemotron_h.modeling_nemotron_h import (
            NemotronHExperts,
        )
    except Exception:
        return 0

    # Walk a snapshot since we'll mutate during iteration.
    targets = [(name, m) for name, m in model.named_modules()
               if isinstance(m, NemotronHExperts)]
    for name, _ in targets:
        cfg = model.config
        new_mod = E8RHTFusedExperts(cfg, block_diagonal=block_diagonal)
        new_mod.requires_grad_(False)
        model.set_submodule(name, new_mod)
    return len(targets)
