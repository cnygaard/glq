"""Buffer shapes for trellis MoE registration — pure int math, no torch, no vllm.

Split out of ``fused_moe_method`` so the sizing can be tested without a vLLM install (and
without a GPU). It is worth isolating: vLLM's FusedMoE loader ``copy_``s each expert into a
pre-registered slot, so a wrong shape here does not raise — it resizes and lands the weights
somewhere else, which surfaces much later as incoherent output.

Layout comes from ``glq.trellis.pack_layer``: one (m, n) matrix packs to
``[(m//16)*(n//16), ceil(256*K/16)]`` int16, and 5-8 bpw stacks a second stage at
K = bpw-4 over the same tile grid (``trellis_rvq_recipe``).
"""
from __future__ import annotations

TD = 16          # glq.trellis.TD — tile side


def _tiles(m: int, n: int) -> int:
    if m % TD or n % TD:
        raise ValueError(
            f"trellis MoE needs both dims divisible by {TD}; got ({m}, {n}). The packed "
            f"layout is one entry per 16x16 tile, so a ragged dim has no representation.")
    return (m // TD) * (n // TD)


def trellis_moe_shapes(num_experts: int, hidden_size: int, inter: int,
                       w13_out: int, bpw: int) -> dict[str, tuple[int, ...]]:
    """Shapes for every per-expert trellis buffer of one MoE layer.

    ``w13`` is [gate; up] stacked (or just up when not gated); ``w2`` is down_proj.
    Stage 2 exists only for bpw >= 5; below that a ``(E, 1, 1)`` sentinel is registered
    rather than nothing, because both the loader and the decode gate on ``numel()`` and a
    missing attribute would be an AttributeError deep inside apply().
    """
    from glq.trellis import trellis_rvq_recipe

    rates = trellis_rvq_recipe(int(bpw))          # [K] or [4, bpw-4]
    k1 = rates[0]
    k2 = rates[1] if len(rates) > 1 else None

    # vLLM's loader splits w13 by halving axis 0 (see _glq_weight_loader). In the
    # row-block-major packed layout that is a 16-row-block boundary, and kernel_tile_flip
    # pairs those blocks — so each half must be 32-row aligned or gate and up bytes
    # interleave into weights that still load and still decode, incorrectly. Same rule
    # split_trellis_packed asserts on the quantize side; assert it here too, where the
    # message can still name the layer's dims.
    if w13_out != inter and w13_out % 32:
        raise ValueError(
            f"trellis MoE gate/up split needs each half 32-row aligned; w13_out={w13_out} "
            f"(inter={inter}) is not. The MMA byte flip pairs 16-row blocks, so a cut "
            f"inside a pair mixes gate and up.")
    if w13_out != inter and inter % 32:
        raise ValueError(
            f"trellis MoE gate/up split needs each half 32-row aligned; inter={inter} is "
            f"not a multiple of 32.")

    t13, t2 = _tiles(w13_out, hidden_size), _tiles(hidden_size, inter)
    out: dict[str, tuple[int, ...]] = {
        "w13_trellis_packed": (num_experts, t13, TD * k1),
        "w2_trellis_packed": (num_experts, t2, TD * k1),
        "w13_trellis_packed2": ((num_experts, t13, TD * k2) if k2 else (num_experts, 1, 1)),
        "w2_trellis_packed2": ((num_experts, t2, TD * k2) if k2 else (num_experts, 1, 1)),
        # SU is per output row; SV is per input column and shared across experts (one RHT
        # seed for the whole layer, exactly as the quantizer produces).
        "w13_SU": (num_experts, w13_out),
        "w2_SU": (num_experts, hidden_size),
        "w13_SV": (hidden_size,),
        "w2_SV": (inter,),
        "w13_Wscale": (num_experts,),
        "w2_Wscale": (num_experts,),
        "w13_inv_resid_scale2": (num_experts,),
        "w2_inv_resid_scale2": (num_experts,),
    }
    return out
