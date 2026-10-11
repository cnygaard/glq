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

#: Bytes per element of each registered buffer, matching the dtypes the registration path
#: uses (`_make_glq_param` in fused_moe_method). Kept here beside the shapes because the
#: arena's byte budget needs both, and a dtype guessed in a second place is a sizing bug
#: that shows up as wasted VRAM rather than an error.
_ITEMSIZE = {
    "trellis_packed": 2,          # int16
    "trellis_packed2": 2,         # int16
    "SU": 2,                      # fp16
    "SV": 2,                      # fp16
    "Wscale": 4,                  # fp32
    "inv_resid_scale2": 4,        # fp32
}


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


def _itemsize(key: str) -> int:
    """Bytes per element for a registered buffer name like ``w13_trellis_packed2``."""
    leaf = key.split("_", 1)[1]
    try:
        return _ITEMSIZE[leaf]
    except KeyError:                                  # pragma: no cover - defensive
        raise KeyError(
            f"no itemsize recorded for buffer {key!r}. A new per-expert buffer must be "
            f"added to _ITEMSIZE, or the arena will mis-size itself silently.") from None


def arena_shapes(n_slots: int, *, hidden_size: int, inter: int, w13_out: int,
                 bpw: int) -> dict[str, tuple[int, ...]]:
    """Shapes for a VRAM expert arena holding ``n_slots`` experts of one MoE layer.

    Every per-expert buffer, with the slot count replacing the expert count and the trailing
    dims untouched — that identity is what lets the existing CUDA kernel read the arena with
    no change: it indexes ``base + eidx * estride``, so passing the arena as the base and a
    slot id as ``eidx`` lands on the right bytes
    (`glq/csrc/glq_cuda.cu`, ``glq_grouped_matmul_trellis_*``).

    ``w13_SV`` / ``w2_SV`` are deliberately absent: one RHT seed serves the whole layer, so
    they are not per-expert and must stay shared. (The *checkpoint* stores a copy per expert,
    which is a different thing — see `tests/models/`.)
    """
    full = trellis_moe_shapes(1, hidden_size, inter, w13_out, bpw)
    return {k: (int(n_slots),) + v[1:] for k, v in full.items() if len(v) > 1 or
            k.endswith(("Wscale", "inv_resid_scale2"))}


def bytes_per_slot(*, hidden_size: int, inter: int, w13_out: int, bpw: int) -> int:
    """Device bytes one arena slot occupies — every per-expert buffer summed.

    Derived from the registered shapes rather than measured once and hard-coded: the
    1.776 MiB figure in `benchmarks/_expert_cache_score.py` came from a checkpoint header and
    includes the per-expert ``SV`` copies that the serving layout shares, so it is ~0.6% high.
    """
    total = 0
    for key, shape in arena_shapes(1, hidden_size=hidden_size, inter=inter,
                                   w13_out=w13_out, bpw=bpw).items():
        n = 1
        for d in shape[1:]:
            n *= d
        total += n * _itemsize(key)
    return total


def slots_for_budget(budget_bytes: int, *, n_layers: int, hidden_size: int, inter: int,
                     w13_out: int, bpw: int, n_experts: int | None = None,
                     max_tokens: int | None = None, top_k: int | None = None) -> int:
    """Slots **per layer** that ``budget_bytes`` of VRAM affords across ``n_layers``.

    Per-layer rather than one shared pool: S0 measured the two as a tie on real routing
    (0.5236 global vs 0.5300 per-layer), so this is not chosen for hit rate — it is chosen
    because a fixed quota per layer makes the budget predictable and cannot starve a layer.
    Strata's 7x gap needs ``slots < n_layers * top_k``, far below anything sized here.

    ``max_tokens``/``top_k``, when given, enforce the **correctness precondition**; see the
    ValueError below.
    """
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {n_layers}")
    per = bytes_per_slot(hidden_size=hidden_size, inter=inter, w13_out=w13_out, bpw=bpw)
    slots = int(budget_bytes) // (per * int(n_layers))
    if n_experts is not None:
        # Past full residency the extra slots can never be filled; capping keeps the arena
        # from holding VRAM it cannot use.
        slots = min(slots, int(n_experts))

    if max_tokens is not None and top_k is not None:
        need = int(max_tokens) * int(top_k)
        if slots < need:
            raise ValueError(
                f"an arena of {slots} slots/layer cannot serve {max_tokens} tokens x top_k "
                f"{top_k} = {need} SIMULTANEOUS routings. Every expert routed in a step must "
                f"be resident at once: the fused op reads the arena after the fill, so a "
                f"later routing evicting an earlier one leaves the matmul reading a slot "
                f"that holds a different expert — plausible output from the wrong weights. "
                f"Raise the budget to >= "
                f"{need * per * n_layers / 1024**3:.2f} GiB, or lower the token cap.")
    return slots
