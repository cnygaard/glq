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
(`glq_fused_linear_cuda` handles 1–4 stages).

**Forward path selection** (in priority order):

1. ``glq_fused_moe_block_diag_cuda`` — single C++ call dispatching all active
   experts in one host-side step. Block-diagonal RHT, supports stages 1–3.
   Used when the kernel ext is available, the codebook(s) are attached,
   the block-decomposition metadata is on GPU, and no expert exceeds
   stage-3 (which holds for our 4.5bpw checkpoint).
2. Per-expert Python loop fallback — calls each expert's
   `_ExpertPair.{up,down}_proj` (each E8RHTLinear), which itself uses
   the well-optimized single-linear `glq_fused_linear_block_diag_cuda`.
   Always works; slower because of the per-expert dispatch overhead.

The internal child-module layout matches the trust-remote-code checkpoint
key layout exactly: `experts.{i}.up_proj.Qidxs`, `experts.{i}.down_proj.SU`,
etc., so HF's normal state-dict loader can install the buffers without any
key remapping.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .quantized_linear import E8RHTLinear


def _is_relu2(act_fn) -> bool:
    """Heuristic: detect whether an activation is the relu² (squared-relu)
    that NemotronH uses in its non-gated MoE. Used to pick the kernel's
    activation_type=5 (relu2_no_mul) path."""
    name = type(act_fn).__name__.lower()
    return "relu2" in name or "relusquared" in name


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

    # ------------------------------------------------------------------ fused-kernel path

    def _try_build_stacked(self) -> bool:
        """Lazily stack per-expert GLQ buffers into ``(E, ...)`` tensors that
        the fused MoE kernel consumes. Idempotent: rebuilds only on first
        call after weights are loaded.

        Returns True if the stacked buffers are ready.
        """
        if getattr(self, "_stacked_ready", False):
            return True
        # All per-expert E8RHTLinears must be on a CUDA device with codebook(s)
        # attached (set by GLQQuantizer._process_model_after_weight_loading).
        e0 = self[0]
        if e0.up_proj.codebook is None:
            return False
        if not e0.up_proj.Qidxs.is_cuda:
            return False

        device = e0.up_proj.Qidxs.device
        E = self.num_experts

        def _stack(attr: str, target_dtype=None):
            tensors = [getattr(self[i].up_proj, attr) for i in range(E)]
            return torch.stack(tensors).to(device=device).contiguous()

        def _stack_down(attr: str):
            tensors = [getattr(self[i].down_proj, attr) for i in range(E)]
            return torch.stack(tensors).to(device=device).contiguous()

        # w13 (= up_proj) stacked tensors
        self._w13_Qidxs = _stack("Qidxs")
        self._w13_SU = _stack("SU")
        self._w13_SV = e0.up_proj.SV.contiguous()  # shared across experts
        self._w13_Wscale = _stack("Wscale").float()
        self._w13_Qidxs2 = _stack("Qidxs2")
        self._w13_inv_resid_scale = _stack("inv_resid_scale").float()
        self._w13_Qidxs3 = _stack("Qidxs3")
        self._w13_inv_resid_scale2 = _stack("inv_resid_scale2").float()

        # w2 (= down_proj) stacked tensors
        self._w2_Qidxs = _stack_down("Qidxs")
        self._w2_SU = _stack_down("SU")
        self._w2_SV = e0.down_proj.SV.contiguous()  # shared across experts
        self._w2_Wscale = _stack_down("Wscale").float()
        self._w2_Qidxs2 = _stack_down("Qidxs2")
        self._w2_inv_resid_scale = _stack_down("inv_resid_scale").float()
        self._w2_Qidxs3 = _stack_down("Qidxs3")
        self._w2_inv_resid_scale2 = _stack_down("inv_resid_scale2").float()

        # Block-decomposition metadata — same for every expert in this layer
        # because the dims (in/out) match. Take from expert 0.
        self._w13_blocks_n = e0.up_proj._blocks_n_tensor
        self._w13_blocks_m = e0.up_proj._blocks_m_tensor
        self._w13_blocks_n_meta = e0.up_proj._blocks_n_meta_cpu.to(device, non_blocking=True)
        self._w13_blocks_m_meta = e0.up_proj._blocks_m_meta_cpu.to(device, non_blocking=True)
        self._w2_blocks_n = e0.down_proj._blocks_n_tensor
        self._w2_blocks_m = e0.down_proj._blocks_m_tensor
        self._w2_blocks_n_meta = e0.down_proj._blocks_n_meta_cpu.to(device, non_blocking=True)
        self._w2_blocks_m_meta = e0.down_proj._blocks_m_meta_cpu.to(device, non_blocking=True)

        # Padded dim cache (used to pick fused vs fallback path)
        self._n_pad_w13 = e0.up_proj.n_pad
        self._m_pad_w13 = e0.up_proj.m_pad
        self._n_pad_w2 = e0.down_proj.n_pad
        self._m_pad_w2 = e0.down_proj.m_pad

        # Codebook references — shared global instances attached by
        # _process_model_after_weight_loading on every E8RHTLinear.
        self._codebook = e0.up_proj.codebook
        self._codebook2 = e0.up_proj.codebook2  # may be None for pure 2bpw

        self._stacked_ready = True
        return True

    # Above this many tokens, the per-expert Python loop wins because it
    # batches all tokens routed to the same expert into one Tensor-Core
    # matmul (B=N), while the fused kernel iterates (token, top_k) pairs
    # sequentially with B=1. Empirically: kernel wins for B=1 decode,
    # loses on multi-token prefill (RTX PRO 6000 Blackwell, Cascade-2).
    _FUSED_KERNEL_MAX_TOKENS = 4

    def _try_fused_forward(self, hidden_states, top_k_index, top_k_weights):
        """Single-call kernel path. Returns the output tensor on success,
        ``None`` on any unsupported case (caller falls back to Python loop)."""
        if hidden_states.dim() != 2:
            return None
        if not hidden_states.is_cuda:
            return None
        # Skip the kernel for multi-token batches — see _FUSED_KERNEL_MAX_TOKENS.
        if hidden_states.shape[0] > self._FUSED_KERNEL_MAX_TOKENS:
            return None
        if not self._try_build_stacked():
            return None

        # Padded dims must be ≤ 16384 (kernel constraint).
        if max(self._n_pad_w13, self._m_pad_w13,
               self._n_pad_w2, self._m_pad_w2) > 16384:
            return None

        from . import inference_kernel as _ik
        if not _ik._try_load_cuda_ext():
            return None
        glq_cuda = _ik._glq_cuda
        if not hasattr(glq_cuda, "glq_fused_moe_block_diag_cuda"):
            return None

        # NemotronH uses non-gated relu² → activation_type=5.
        activation_type = 5 if _is_relu2(self.act_fn) else 0

        # Codebook tensors — reuse shared E8 codebook for stages 1 & 3
        # (same convention as glq_fused_linear_cuda's stage-3 path).
        cb_half = self._codebook.codebook_half
        cb2_half = (self._codebook2.codebook_half
                    if self._codebook2 is not None else torch.empty(0, dtype=torch.float16, device=cb_half.device))
        # Stage-3 always uses the primary 65536-entry codebook.
        cb3_half = cb_half

        return glq_cuda.glq_fused_moe_block_diag_cuda(
            hidden_states.half().contiguous(),
            top_k_index.contiguous(),
            top_k_weights.float().contiguous(),
            self._w13_Qidxs, self._w13_SU, self._w13_SV,
            self._w13_Wscale, self._w13_Qidxs2, self._w13_inv_resid_scale,
            self._w2_Qidxs, self._w2_SU, self._w2_SV,
            self._w2_Wscale, self._w2_Qidxs2, self._w2_inv_resid_scale,
            cb_half, cb2_half,
            self.input_dim, self.intermediate_dim, self.intermediate_dim,
            self._n_pad_w13, self._m_pad_w13,
            self._n_pad_w2, self._m_pad_w2,
            self._w13_blocks_n, self._w13_blocks_m,
            self._w13_blocks_n_meta, self._w13_blocks_m_meta,
            self._w2_blocks_n, self._w2_blocks_m,
            self._w2_blocks_n_meta, self._w2_blocks_m_meta,
            activation_type,
            self._w13_Qidxs3, self._w13_inv_resid_scale2,
            self._w2_Qidxs3, self._w2_inv_resid_scale2,
            cb3_half,
        )

    def forward(self, hidden_states: torch.Tensor,
                top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        """Mirror of `NemotronHExperts.forward`.

        hidden_states: ``(num_tokens, hidden_or_latent)``
        top_k_index:   ``(num_tokens, top_k)`` int64
        top_k_weights: ``(num_tokens, top_k)`` float — summed routing weights
        """
        # Try the fused single-call path first.
        fused_out = self._try_fused_forward(hidden_states, top_k_index, top_k_weights)
        if fused_out is not None:
            return fused_out.to(hidden_states.dtype)

        # ---------- Python per-expert fallback ----------
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
