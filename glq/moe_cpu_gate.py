"""When the fused CPU MoE op can serve a layer — the single origin for that answer.

``glq_fused_moe_trellis_3inst_cpu`` (glq/csrc/cpu/glq_moe_cpu.cpp) is reached from two
places that know nothing about each other: vLLM's ``fused_moe_method._apply_trellis_cpu``
and HF's ``GLQStackedGatedExperts._try_fused_cpu``. Both must refuse the same layers for
the same reasons, so the decision lives here rather than twice.

This module is deliberately dependency-free — no torch, no vLLM, no env reads. Callers
gather the facts; this decides. That keeps the policy unit-testable on a CPU box with
neither a GPU nor vLLM installed, and it is why ``glq_vllm/_dispatch.py`` re-exports from
here instead of owning a second copy.

Every "no" lands on a per-expert loop that is correct on any shape, so this gate only ever
trades speed for reach, never correctness. The reason string exists because the
alternative — a silent drop onto the slower loop — gets diagnosed as "GLQ is slow on CPU"
instead of as a missing gate.
"""
from __future__ import annotations


def moe_cpu_fused_refusal(*, fused_shape_ok: bool, has_stage2: bool, unpadded: bool,
                          activation_type: int, ext_has_entry: bool,
                          force_fallback: bool, cpu_fused_enabled: bool,
                          sv_shared: bool = True, is_3inst: bool = True) -> str | None:
    """Why ``glq_fused_moe_trellis_3inst_cpu`` cannot serve this layer, or None if it can.

    The limits, in the order they are checked:

    * **stage 2** — the CPU op has no ``packed2``/``inv_resid_scale2`` inputs, so 5-8 bpw
      stacked RVQ has no fused CPU MoE path at all.
    * **variant** — the fused entries take no ``tlut``, so a HYB checkpoint would decode
      against a codebook the kernel does not have. Defaults to ``True`` because vLLM
      already refuses HYB MoE at load (``fused_moe_method.py:107``) and so cannot reach
      here with one; the HF container has no such earlier gate and passes its real value.
    * **padding** — the op passes *logical* dims to its per-expert bracket, matching the
      invariant the dense CPU bracket asserts ("trellis layers are unpadded"). A padded
      layer would index the packed tiles wrongly.
    * **shape** — ``m % 32``/``n % 64``/``R`` bounds, already computed at load as
      ``glq_trellis_fused_ok``; the CPU kernel splits work as ``m / 32`` exactly like CUDA.
    * **shared SV** — the op takes ONE ``SV`` per projection and applies it to every
      expert (glq_moe_cpu.cpp:162 checks only its length). That holds because the RHT seed
      is fixed per layer, so one instance serves all experts (glq/rht.py:162) — but it is a
      property of the checkpoint, not of the op, and decoding 511 experts in the wrong RHT
      basis produces finite, plausible, wrong numbers. Callers that can check it cheaply
      should; ``True`` is the default because vLLM's loader allocates a single shared
      ``SV`` by construction and has nothing to compare.
    * **activation** — the op is gated-only (``w13_out == 2 * intermediate``); ids 0/1/2
      are silu / gelu-tanh / relu², and 3+ are the ``*_no_mul`` variants.
    * **extension** — a wheel predating the symbol must fall back, not raise.
    """
    if has_stage2:
        return ("stage-2 residual (5-8 bpw stacked RVQ): the fused CPU MoE op decodes one "
                "stage only")
    if not is_3inst:
        return ("HYB trellis (a fitted tlut is present): the fused CPU MoE entries are "
                "3INST/lookup-free only")
    if not unpadded:
        return ("padded shapes: the fused CPU MoE op assumes trellis' unpadded layout "
                "(n_pad == in_features, m_pad == out_features)")
    if not fused_shape_ok:
        return ("shapes the CPU trellis kernel cannot take (needs m_pad % 32 == 0, "
                "n_pad % 64 == 0 and 2 <= R <= 4)")
    if not sv_shared:
        return ("SV differs across experts: the fused CPU MoE op applies one SV to every "
                "expert, so this layer's experts do not share an RHT basis")
    if not 0 <= activation_type < 3:
        return (f"activation id {activation_type}: the fused CPU MoE op is gated-only "
                f"(0 silu, 1 gelu-tanh, 2 relu^2)")
    if not ext_has_entry:
        return ("the CPU extension has no glq_fused_moe_trellis_3inst_cpu entry "
                "(older glq wheel, or the extension failed to load)")
    if force_fallback:
        return "GLQ_MOE_FORCE_FALLBACK is set"
    if not cpu_fused_enabled:
        return "GLQ_FUSED_TRELLIS_CPU=0 is set"
    return None
