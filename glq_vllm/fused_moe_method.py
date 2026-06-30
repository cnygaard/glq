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
from ._dispatch import _grouped_enabled


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
        # e8p MoE: register int64 TC-packed buffers + decode via the e8p path.
        # Loading is codebook-agnostic (vLLM's prefix expert-mapping carries the
        # `Qidxs_e8p` suffix; GLQ's shape-based _glq_weight_loader handles the
        # 16-row-tiled gate/up split), so only create_weights/process/apply differ.
        self.codebook_type = getattr(quant_config, 'codebook', 'e8_shell')

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

        if self.codebook_type == "e8p":
            self._create_weights_e8p(layer, num_experts, hidden_size,
                                     intermediate_size_per_partition, is_gated, w13_out)
            return

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

        # Stage 3 buffers (5/6 bpw). Unlike the LINEAR path (which registers a tiny
        # sentinel resized on load by `_glq_weight_loader`), the MoE path loads each
        # expert through vLLM's FusedMoE default weight_loader, which writes
        # `param.data[expert_id] = loaded` WITHOUT resizing — so the slot must be
        # pre-sized at the full (E, m_pad, n_blocks) shape, exactly like Qidxs2 above.
        # A (E,1,1) sentinel here makes the loaded (m_pad, n_blocks) tensor land in a
        # 1-element slot, and the stage-3 kernel then reads out of bounds (illegal
        # memory access during the profiling forward). Only >4 bpw checkpoints carry
        # Qidxs3, so pre-allocate the full buffer only then; <=4 bpw keeps a tiny
        # placeholder whose zero inv_resid_scale2 makes the kernel skip stage 3
        # per-expert (and avoids the per-expert VRAM cost on 2-stage models). Stage 3
        # reuses the PRIMARY 65536-entry codebook (mirrors the linear path), not cb2.
        _has_stage3 = float(getattr(self.quant_config, 'bpw', 2)) > 4.0
        if _has_stage3:
            layer.w13_Qidxs3 = _make_glq_param(
                torch.zeros(num_experts, m_pad_w13, n_blocks_w13, dtype=torch.int16))
            layer.w2_Qidxs3 = _make_glq_param(
                torch.zeros(num_experts, m_pad_w2, n_blocks_w2, dtype=torch.int16))
        else:
            layer.w13_Qidxs3 = _make_glq_param(
                torch.zeros(num_experts, 1, 1, dtype=torch.int16))
            layer.w2_Qidxs3 = _make_glq_param(
                torch.zeros(num_experts, 1, 1, dtype=torch.int16))
        layer.w13_inv_resid_scale2 = _make_glq_param(
            torch.zeros(num_experts, dtype=torch.float32))
        layer.w2_inv_resid_scale2 = _make_glq_param(
            torch.zeros(num_experts, dtype=torch.float32))

    def _create_weights_e8p(self, layer, num_experts, hidden_size, inter, is_gated, w13_out):
        """e8p MoE buffers: int64 TC-packed Qidxs_e8p (+ RVQ residual stages) per expert,
        registered FULL-SIZE so vLLM's FusedMoE loader copies in-place per expert. Names
        carry the `_e8p`/`_e81b` suffixes so vLLM's prefix expert-mapping + the shape-based
        `_glq_weight_loader` route `gate_proj.Qidxs_e8p` -> `w13_Qidxs_e8p` (gate/up split on
        the per-expert slot's mp16 axis-0). Mirrors the single-linear e8p create_weights."""
        from .linear_method import _e8p_pad

        def _dims(out_sz, in_sz):
            m_pad, n_pad = _e8p_pad(out_sz, 16), _e8p_pad(in_sz, 64)
            return m_pad, n_pad, max(m_pad // 16, 1), max(n_pad // 64, 1)

        m_pad_w13, n_pad_w13, mp16_w13, nb64_w13 = _dims(w13_out, hidden_size)
        m_pad_w2, n_pad_w2, mp16_w2, nb64_w2 = _dims(hidden_size, inter)

        layer.glq_num_experts = num_experts
        layer.glq_hidden_size = hidden_size
        layer.glq_intermediate_size = inter
        layer.glq_w13_out = w13_out
        layer.glq_is_gated = is_gated
        layer.glq_m_pad_w13, layer.glq_n_pad_w13 = m_pad_w13, n_pad_w13
        layer.glq_m_pad_w2, layer.glq_n_pad_w2 = m_pad_w2, n_pad_w2
        layer.glq_is_e8p = True

        bpw = int(float(getattr(self.quant_config, 'bpw', 4)))
        # Residual stage codebook per stage k (0-indexed): E8P unless E81B-final of odd bpw.
        _nf = bpw // 2
        _st = [False] * _nf + ([True] if bpw % 2 == 1 else [])

        def _reg(prefix, mp16, nb64, m_pad, n_pad):
            setattr(layer, f"{prefix}_Qidxs_e8p", _make_glq_param(
                torch.zeros(num_experts, mp16, nb64, 8, 4, dtype=torch.int64)))
            setattr(layer, f"{prefix}_SU", _make_glq_param(
                torch.ones(num_experts, m_pad, dtype=torch.float16)))
            setattr(layer, f"{prefix}_SV", _make_glq_param(
                torch.ones(n_pad, dtype=torch.float16)))
            setattr(layer, f"{prefix}_Wscale", _make_glq_param(
                torch.ones(num_experts, dtype=torch.float32)))
            setattr(layer, f"{prefix}_inv_resid_scale", _make_glq_param(
                torch.zeros(num_experts, dtype=torch.float32)))
            setattr(layer, f"{prefix}_inv_resid_scale2", _make_glq_param(
                torch.zeros(num_experts, dtype=torch.float32)))
            setattr(layer, f"{prefix}_inv_resid_scale3", _make_glq_param(
                torch.zeros(num_experts, dtype=torch.float32)))
            # Residual stages k=1,2,3 -> Qidxs{k+1}_(e8p|e81b). Register the variant the
            # recipe uses at FULL (E,...) shape; the other variant + inactive stages are
            # (E,1,...) sentinels collapsed to numel-0 in process (the decode numel-gates them).
            for k in (1, 2, 3):
                active = len(_st) > k
                is_e81b = active and _st[k]
                e8p_t = (torch.zeros(num_experts, mp16, nb64, 8, 4, dtype=torch.int64)
                         if active and not is_e81b
                         else torch.zeros(num_experts, 1, 1, 1, 1, dtype=torch.int64))
                e81b_t = (torch.zeros(num_experts, m_pad, nb64, dtype=torch.int64)
                          if is_e81b
                          else torch.zeros(num_experts, 1, 1, dtype=torch.int64))
                setattr(layer, f"{prefix}_Qidxs{k + 1}_e8p", _make_glq_param(e8p_t))
                setattr(layer, f"{prefix}_Qidxs{k + 1}_e81b", _make_glq_param(e81b_t))

        _reg("w13", mp16_w13, nb64_w13, m_pad_w13, n_pad_w13)
        _reg("w2", mp16_w2, nb64_w2, m_pad_w2, n_pad_w2)

    def _process_e8p(self, layer):
        """e8p MoE: cache grids + block-diag RHT tensors + per-expert scalars; collapse
        unused residual sentinels to numel-0. Mirrors linear_method._setup_e8p_weights."""
        from .linear_method import _ensure_codebook, _try_load_cuda_ext
        from glq.hadamard import _block_decompose as _bd
        from glq.quantized_linear import _pack_block_meta as _pbm

        dev = layer.w13_Qidxs_e8p.device
        grid_cb, e81b_cb = _ensure_codebook(dev, max_bpw=8, codebook_type="e8p")
        _try_load_cuda_ext()
        layer._glq_e8p_grid = grid_cb.grid_packed_abs.to(dev)
        layer._glq_e81b_grid = e81b_cb.e81b_grid.to(dev) if e81b_cb is not None else None

        # Re-read padded dims from the loaded stage-0 buffers (pow2/block-diag agnostic).
        layer.glq_m_pad_w13 = layer.w13_Qidxs_e8p.shape[1] * 16
        layer.glq_n_pad_w13 = layer.w13_Qidxs_e8p.shape[2] * 64
        layer.glq_m_pad_w2 = layer.w2_Qidxs_e8p.shape[1] * 16
        layer.glq_n_pad_w2 = layer.w2_Qidxs_e8p.shape[2] * 64

        # Move e8p buffers to device IN PLACE (mutate .data — never reassign Parameter);
        # collapse residual sentinels (per-expert slot is 1 element) to numel-0 so the
        # decode's numel gate skips the inactive stage/variant.
        for prefix in ("w13", "w2"):
            for attr in [f"{prefix}_Qidxs_e8p", f"{prefix}_SU", f"{prefix}_SV",
                         f"{prefix}_Wscale", f"{prefix}_inv_resid_scale",
                         f"{prefix}_inv_resid_scale2", f"{prefix}_inv_resid_scale3"]:
                t = getattr(layer, attr, None)
                if t is not None and t.device != dev:
                    t.data = t.data.to(dev)
            for k in (2, 3, 4):
                for var in ("e8p", "e81b"):
                    t = getattr(layer, f"{prefix}_Qidxs{k}_{var}", None)
                    if t is None:
                        continue
                    if t.dim() >= 2 and t.shape[1] <= 1:           # sentinel -> numel-0
                        t.data = torch.empty(0, dtype=torch.int64, device=dev)
                    elif t.device != dev:
                        t.data = t.data.to(dev)

        # Block-diagonal RHT tensors — shared across experts (same shape per prefix).
        def _blocks(m_pad, n_pad):
            bn, bm = _bd(n_pad), _bd(m_pad)
            return {'_bn': torch.tensor(bn, dtype=torch.int64, device='cpu'),
                    '_bm': torch.tensor(bm, dtype=torch.int64, device='cpu'),
                    '_bnm': _pbm(bn).to(dev), '_bmm': _pbm(bm).to(dev)}
        layer._glq_e8p_blocks_w13 = _blocks(layer.glq_m_pad_w13, layer.glq_n_pad_w13)
        layer._glq_e8p_blocks_w2 = _blocks(layer.glq_m_pad_w2, layer.glq_n_pad_w2)

        # Per-expert scalars (avoid a GPU->CPU sync each forward).
        n = layer.glq_num_experts
        for pfx in ("w13", "w2"):
            setattr(layer, f"glq_{pfx}_wscale",
                    [getattr(layer, f"{pfx}_Wscale")[i].item() for i in range(n)])
            for s, attr in ((1, "inv_resid_scale"), (2, "inv_resid_scale2"), (3, "inv_resid_scale3")):
                setattr(layer, f"glq_{pfx}_inv_rs{'' if s == 1 else s}",
                        [getattr(layer, f"{pfx}_{attr}")[i].item() for i in range(n)])

        # Eligibility for the fused grouped-e8p op (v1 = 4bpw: E8P stage-0 + optional
        # E8P stage-1). The fused op consumes only Qidxs_e8p + Qidxs2_e8p; recipes with
        # an E81B stage or a stage-2/3 E8P residual (odd / 5-8 bpw) fall back to the
        # per-expert loop. Computed once at load (the sentinels are already collapsed).
        def _fused_ok(pfx):
            # Full N-stage (E8P stages 0-3 + an optional E81B residual) is fused.
            # The E81B grouped WMMA needs m_pad % 32 == 0 (E8P needs only 16); a layer
            # with an E81B stage whose m_pad isn't a multiple of 32 falls back to the
            # per-expert loop (correct, eager).
            has_e81b = any((getattr(layer, f"{pfx}_Qidxs{k}_e81b", None) is not None
                            and getattr(layer, f"{pfx}_Qidxs{k}_e81b").numel() > 0)
                           for k in (2, 3, 4))
            if has_e81b and getattr(layer, f"glq_m_pad_{pfx}") % 32 != 0:
                return False
            return True
        layer.glq_e8p_fused_ok = _fused_ok("w13") and _fused_ok("w2")

        for _name, param in layer.named_parameters():
            if hasattr(param, 'weight_loader'):
                del param.weight_loader

    def _apply_e8p(self, layer, x, topk_weights, topk_ids):
        """e8p MoE baseline: per-expert loop reusing E8RHTLinear._e8p_linear_apply (the
        proven single-linear e8p decode). Correct + compressed; eager (not cudagraph).
        The fused grouped-e8p op replaces this in Phase B."""
        from glq.quantized_linear import E8RHTLinear
        _apply = E8RHTLinear._e8p_linear_apply
        dtype, device = x.dtype, x.device
        grid, e81b_grid = layer._glq_e8p_grid, layer._glq_e81b_grid
        n = layer.glq_num_experts
        hidden, inter, w13_out = (layer.glq_hidden_size, layer.glq_intermediate_size,
                                  layer.glq_w13_out)
        activation = getattr(layer, 'activation', None)
        _empty = torch.empty(0, dtype=torch.int64, device=device)

        def _sl(buf, e):
            return buf[e] if (buf is not None and buf.numel() > 0 and buf.shape[0] == n) else _empty

        def _shard(pfx, xin, in_f, out_f, e):
            blk = getattr(layer, f"_glq_e8p_blocks_{pfx}")
            q2e, q2b = _sl(getattr(layer, f"{pfx}_Qidxs2_e8p"), e), _sl(getattr(layer, f"{pfx}_Qidxs2_e81b"), e)
            q3e, q3b = _sl(getattr(layer, f"{pfx}_Qidxs3_e8p"), e), _sl(getattr(layer, f"{pfx}_Qidxs3_e81b"), e)
            q4e, q4b = _sl(getattr(layer, f"{pfx}_Qidxs4_e8p"), e), _sl(getattr(layer, f"{pfx}_Qidxs4_e81b"), e)
            return _apply(
                xin, getattr(layer, f"{pfx}_SV"), getattr(layer, f"{pfx}_SU")[e],
                getattr(layer, f"{pfx}_Qidxs_e8p")[e], q2e, q2b, grid, e81b_grid,
                getattr(layer, f"glq_{pfx}_wscale")[e],
                getattr(layer, f"glq_{pfx}_inv_rs")[e],
                q2e.numel() > 0, q2b.numel() > 0,
                in_f, out_f, getattr(layer, f"glq_n_pad_{pfx}"), getattr(layer, f"glq_m_pad_{pfx}"),
                blk['_bn'], blk['_bm'], blk['_bnm'], blk['_bmm'],
                Qidxs3_e8p=q3e, Qidxs3_e81b=q3b, Qidxs4_e8p=q4e, Qidxs4_e81b=q4b,
                has_e8p3=q3e.numel() > 0, has_e81b3=q3b.numel() > 0,
                has_e8p4=q4e.numel() > 0, has_e81b4=q4b.numel() > 0,
                inv_rs2=getattr(layer, f"glq_{pfx}_inv_rs2")[e],
                inv_rs3=getattr(layer, f"glq_{pfx}_inv_rs3")[e],
                bias=None, out_dtype=torch.float16)

        out = torch.zeros(x.shape[0], hidden, dtype=dtype, device=device)
        for et in topk_ids.unique():
            e = int(et.item())
            mask = (topk_ids == e)
            token_mask = mask.any(dim=1)
            ew = (topk_weights * mask.float()).sum(dim=1)[token_mask]
            xt = x[token_mask].to(torch.float16)
            h = _shard("w13", xt, hidden, w13_out, e)
            h = _apply_activation(h, activation)
            y = _shard("w2", h, inter, hidden, e)
            out[token_mask] += y.to(dtype) * ew.unsqueeze(-1)
        return out

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Ensure codebook is on device. Cache per-expert metadata for fast apply()."""
        if getattr(layer, 'glq_is_e8p', False):
            self._process_e8p(layer)
            return
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
                     'w2_Qidxs2', 'w2_inv_resid_scale',
                     'w13_Qidxs3', 'w13_inv_resid_scale2',
                     'w2_Qidxs3', 'w2_inv_resid_scale2']:
            t = getattr(layer, attr, None)
            if t is not None and t.device != device:
                setattr(layer, attr, torch.nn.Parameter(t.data.to(device), requires_grad=False))

        # Disable stage 3 on <=4 bpw layers. create_weights registers a (E,1,1)
        # Qidxs3 placeholder for 2-stage checkpoints (no Qidxs3 key in the
        # checkpoint, so it's never written). The C++ MoE entries gate stage 3 on
        # `Qidxs3.numel() > 0 && inv_resid_scale2.numel() > 0` — a (E,1,1) sentinel
        # has numel E>0, which would WRONGLY enable stage 3 and make the kernel read
        # qidxs3[m_row*N_BLOCKS+j] out of bounds of the 1-element-per-expert buffer
        # (illegal memory access). Collapse the sentinel to a numel-0 tensor so the
        # check is False. A real stage-3 buffer has shape (E, m_pad, n_blocks) with
        # m_pad>1, so it is left intact.
        for qn in ('w13_Qidxs3', 'w2_Qidxs3'):
            q = getattr(layer, qn, None)
            if q is not None and q.dim() == 3 and q.shape[1] <= 1:
                setattr(layer, qn, torch.nn.Parameter(
                    torch.empty(0, dtype=torch.int16, device=device), requires_grad=False))

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
        layer.glq_w13_inv_rs2 = [layer.w13_inv_resid_scale2[i].item() for i in range(n)]
        layer.glq_w2_inv_rs2 = [layer.w2_inv_resid_scale2[i].item() for i in range(n)]

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
        if getattr(layer, 'glq_is_e8p', False):
            import os as _os
            from glq import inference_kernel as _ik
            num_tokens_e8p = x.shape[0]
            activation = getattr(layer, 'activation', None)
            _bd_cap = int(_os.environ.get("GLQ_MOE_BD_MAX_TOKENS", "256"))
            # Fused grouped-e8p op: device-side expert dispatch (no host sync) so the
            # whole MoE decode step is capturable under FULL cudagraph; per-expert
            # tensor-core e8p decode. v1 = 4bpw (E8P stage-0 + optional E8P stage-1) —
            # `glq_e8p_fused_ok` is False for E81B/5-8bpw recipes, which take the loop.
            # GLQ_MOE_FORCE_FALLBACK=1 forces the loop (A/B isolation). Tokens > cap
            # (prefill) also take the loop, which is faster for many tokens (eager).
            if (getattr(layer, 'glq_e8p_fused_ok', False)
                    and _os.environ.get("GLQ_MOE_FORCE_FALLBACK", "0") == "0"
                    and num_tokens_e8p <= _bd_cap
                    and self._activation_type(activation) < 3
                    and _ik._try_load_cuda_ext()
                    and hasattr(_ik._glq_cuda, 'glq_fused_moe_e8p_cuda')
                    and hasattr(torch.ops, 'glq')
                    and hasattr(torch.ops.glq, 'fused_moe_e8p')
                    and layer.glq_n_pad_w13 <= 16384 and layer.glq_m_pad_w13 <= 16384
                    and layer.glq_n_pad_w2 <= 16384 and layer.glq_m_pad_w2 <= 16384):
                dtype = x.dtype
                blk13 = layer._glq_e8p_blocks_w13
                blk2 = layer._glq_e8p_blocks_w2
                e81b_grid = layer._glq_e8p_grid.new_empty(0) \
                    if getattr(layer, '_glq_e81b_grid', None) is None else layer._glq_e81b_grid
                output = torch.ops.glq.fused_moe_e8p(
                    x.half().contiguous(),
                    topk_ids.to(torch.int64),
                    topk_weights.float().contiguous(),
                    layer.w13_Qidxs_e8p, layer.w13_Qidxs2_e8p,
                    layer.w13_SU, layer.w13_SV, layer.w13_Wscale, layer.w13_inv_resid_scale,
                    layer.w2_Qidxs_e8p, layer.w2_Qidxs2_e8p,
                    layer.w2_SU, layer.w2_SV, layer.w2_Wscale, layer.w2_inv_resid_scale,
                    layer._glq_e8p_grid,
                    layer.glq_hidden_size, layer.glq_intermediate_size, layer.glq_w13_out,
                    layer.glq_n_pad_w13, layer.glq_m_pad_w13,
                    layer.glq_n_pad_w2, layer.glq_m_pad_w2,
                    blk13['_bn'], blk13['_bm'], blk13['_bnm'], blk13['_bmm'],
                    blk2['_bn'], blk2['_bm'], blk2['_bnm'], blk2['_bmm'],
                    self._activation_type(activation),
                    # E8P residual stages 2-3 (even bpw 6/8): numel-0 Qidxs / zero
                    # inv_rs for absent stages (4bpw); the op gates per-Qidxs-numel.
                    layer.w13_Qidxs3_e8p, layer.w13_Qidxs4_e8p,
                    layer.w13_inv_resid_scale2, layer.w13_inv_resid_scale3,
                    layer.w2_Qidxs3_e8p, layer.w2_Qidxs4_e8p,
                    layer.w2_inv_resid_scale2, layer.w2_inv_resid_scale3,
                    # E81B residual stage (odd bpw 5/7): per-stage Qidxs_e81b (numel-0
                    # if absent) + shared 256x8 grid (numel-0 placeholder if pure-E8P).
                    layer.w13_Qidxs2_e81b, layer.w13_Qidxs3_e81b, layer.w13_Qidxs4_e81b,
                    layer.w2_Qidxs2_e81b, layer.w2_Qidxs3_e81b, layer.w2_Qidxs4_e81b,
                    e81b_grid,
                )
                if dtype != torch.float16:
                    output = output.to(dtype)
                return output
            # Fallback: per-expert e8p decode loop (Phase A baseline; eager, correct,
            # compressed). Used for prefill (> cap), E81B/5-8bpw recipes, or forced A/B.
            return self._apply_e8p(layer, x, topk_weights, topk_ids)

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
                # stage-3 RVQ (5/6 bpw): real Qidxs3 + per-expert inv_resid_scale2;
                # codebook3 = PRIMARY codebook (cb_half). Kernel skips stage-3 per-expert
                # when inv_resid_scale2==0 (<=4 bpw), so the 2-stage path is unaffected.
                layer.w13_Qidxs3, layer.w13_inv_resid_scale2,
                layer.w2_Qidxs3, layer.w2_inv_resid_scale2, cb_half,
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
        # Default cap = 256 (>= vLLM's decode cudagraph capture sizes, which top
        # out at max_num_seqs). The Stage-2 kernel is device-dispatched (reads
        # routing on-device, no .cpu()), so under FULL cudagraph the whole decode
        # step — every captured batch size up to the cap — is captured and replays
        # at ~0 launch cost (this is the cudagraph win). Prefill (num_tokens > cap)
        # still takes the per-expert batched Python loop, which is faster for many
        # tokens and runs eager. For pure-eager serving (no cudagraph) at large
        # batch, set GLQ_MOE_BD_MAX_TOKENS=4 to restore the small-batch-only gate.
        _bd_cap = int(_os.environ.get("GLQ_MOE_BD_MAX_TOKENS", "256"))

        # Stage 3 grouped-GEMM path: sorts tokens by expert -> one batched
        # tensor-core GEMM per expert. This is the batched-decode/throughput win
        # (validated 26B-A4B: b32 6.9-8.4x over the per-(token,expert) block-diag
        # matvec; b1 ~1.15x). DEFAULT-ON for batched MoE; b1 stays on the block-diag
        # path below, which is bit-exact to the matvec oracle (grouped uses the TC
        # matmul -> numerically equivalent but not bit-identical to it, only
        # run-to-run deterministic). Gated activations only; non-gated falls through.
        #
        # GLQ_MOE_GROUPED tri-state:
        #   unset / "auto" (default): grouped when num_tokens >= GLQ_MOE_GROUPED_MIN
        #                             (default 2); block-diag keeps b1.
        #   "1"/"on"/"true"         : force grouped whenever the gate holds (incl. b1).
        #   "0"/"off"/"false"       : never grouped (force block-diag; A/B isolation).
        _want_grouped = _grouped_enabled(
            _os.environ.get("GLQ_MOE_GROUPED"),
            int(_os.environ.get("GLQ_MOE_GROUPED_MIN", "2")),
            num_tokens)
        if (_want_grouped
                and _os.environ.get("GLQ_MOE_FORCE_FALLBACK", "0") == "0"
                and not _pow2_dims
                and layer.glq_bd_meta_w13 is not None
                and layer.glq_bd_meta_w2 is not None
                and self._activation_type(activation) < 3
                and num_tokens <= _bd_cap
                and _ik._try_load_cuda_ext()
                and hasattr(_ik._glq_cuda, 'glq_fused_moe_grouped_gemm_cuda')
                and hasattr(torch.ops, 'glq')
                and hasattr(torch.ops.glq, 'fused_moe_grouped_gemm')
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
            output = torch.ops.glq.fused_moe_grouped_gemm(
                x.half().contiguous(),
                topk_ids.to(torch.int64),
                topk_weights.float().contiguous(),
                layer.w13_Qidxs, layer.w13_SU, layer.w13_SV,
                layer.w13_Wscale, layer.w13_Qidxs2, layer.w13_inv_resid_scale,
                layer.w2_Qidxs, layer.w2_SU, layer.w2_SV,
                layer.w2_Wscale, layer.w2_Qidxs2, layer.w2_inv_resid_scale,
                cb_half, cb2_half,
                hidden, inter_dim, w13_out,
                layer.glq_n_pad_w13, layer.glq_m_pad_w13,
                layer.glq_n_pad_w2, layer.glq_m_pad_w2,
                bd13["blocks_n_tensor"], bd13["blocks_m_tensor"],
                bd13["blocks_n_meta_gpu"], bd13["blocks_m_meta_gpu"],
                bd2["blocks_n_tensor"], bd2["blocks_m_tensor"],
                bd2["blocks_n_meta_gpu"], bd2["blocks_m_meta_gpu"],
                self._activation_type(activation),
                layer.w13_Qidxs3, layer.w13_inv_resid_scale2,
                layer.w2_Qidxs3, layer.w2_inv_resid_scale2, cb_half,
            )
            if dtype != torch.float16:
                output = output.to(dtype)
            return output

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
                # stage-3 RVQ (5/6 bpw): per-expert Qidxs3 + inv_resid_scale2;
                # zero-scale experts skip stage 3 in-kernel. codebook3 = PRIMARY.
                layer.w13_Qidxs3, layer.w13_inv_resid_scale2,
                layer.w2_Qidxs3, layer.w2_inv_resid_scale2, cb_half,
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
            # Stage 3 (5/6 bpw): per-expert inv_resid_scale2 != 0 ⇒ Qidxs3 present.
            # Stages 3-4 reuse the PRIMARY 65536-entry codebook (handled inside
            # _glq_apply_shard); cb2 is stage-2 only.
            inv_rs2_w13 = layer.glq_w13_inv_rs2[expert_idx]
            has_s3_w13 = inv_rs2_w13 != 0.0
            h = _glq_apply_shard(
                selected_tokens, device, cb, cb2,
                Qidxs=layer.w13_Qidxs[expert_idx],
                SU=layer.w13_SU[expert_idx],
                SV=layer.w13_SV,
                wscale=layer.glq_w13_wscale[expert_idx],
                has_stage2=has_s2_w13,
                inv_rs=inv_rs_w13,
                Qidxs2=layer.w13_Qidxs2[expert_idx] if has_s2_w13 else None,
                Qidxs3=layer.w13_Qidxs3[expert_idx] if has_s3_w13 else None,
                inv_rs2=inv_rs2_w13,
                out_features=w13_out, in_features=hidden,
                m_pad=layer.glq_m_pad_w13, n_pad=layer.glq_n_pad_w13,
                log_n=layer.glq_log_n_w13, log_m=layer.glq_log_m_w13,
                block_diag_meta=layer.glq_bd_meta_w13,
            )

            h = _apply_activation(h, activation)

            inv_rs_w2 = layer.glq_w2_inv_rs[expert_idx]
            has_s2_w2 = inv_rs_w2 != 0.0
            inv_rs2_w2 = layer.glq_w2_inv_rs2[expert_idx]
            has_s3_w2 = inv_rs2_w2 != 0.0
            h = _glq_apply_shard(
                h, device, cb, cb2,
                Qidxs=layer.w2_Qidxs[expert_idx],
                SU=layer.w2_SU[expert_idx],
                SV=layer.w2_SV,
                wscale=layer.glq_w2_wscale[expert_idx],
                has_stage2=has_s2_w2,
                inv_rs=inv_rs_w2,
                Qidxs2=layer.w2_Qidxs2[expert_idx] if has_s2_w2 else None,
                Qidxs3=layer.w2_Qidxs3[expert_idx] if has_s3_w2 else None,
                inv_rs2=inv_rs2_w2,
                out_features=out_dim, in_features=inter_dim,
                m_pad=layer.glq_m_pad_w2, n_pad=layer.glq_n_pad_w2,
                log_n=layer.glq_log_n_w2, log_m=layer.glq_log_m_w2,
                block_diag_meta=layer.glq_bd_meta_w2,
            )

            output[token_mask] += h.to(dtype) * selected_weights.unsqueeze(-1)

        return output
