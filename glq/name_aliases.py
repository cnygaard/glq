"""Alias spellings for architectures whose module paths and checkpoint keys disagree.

The single origin for one recurring bug class. transformers and vLLM rename **weight key
prefixes** on the way in, but module paths and `quantization_config` travel untouched — so a
set built from one and matched against the other misses every entry, and the miss is silent:
the layer stays unquantized, its GLQ payload has nowhere to land, and the symptom surfaces
somewhere else entirely.

It has cost four separate sites so far:

* `glq_vllm/config.py` — `layer_bpw` is `backbone.*`, `get_quant_method` asks about `model.*`.
  Surfaced as ``AttributeError: 'MergedColumnParallelLinear' object has no attribute 'data'``
  from inside vLLM's loader, naming nothing (fixed by the vLLM whitelist alias).
* `glq_vllm/config.py` again, in `_is_prefused`, which did not get that alias. The costliest
  of the four: the layer resolved as quantized but not as pre-fused, so every NemotronH Mamba
  `in_proj` loaded a ZERO weight and the model produced fluent garbage while the footprint
  and whitelist checks passed.
* `glq/fused_experts.py` — `_collect_quantized_shapes` keys on `backbone.*` while
  `named_modules()` yields `model.*`; the packed map came back empty, every buffer registered
  at size 0, and accelerate's `infer_auto_device_map` saw an empty model.
* `glq/hf_integration.py` — the nn.Linear walk matched module names against a whitelist of
  checkpoint names, so **every Mamba mixer projection was skipped**. It hid behind the
  footprint (the MoE experts are 10.47 of Lightning-30B's 14.65 GiB, so missing 0.31 GiB of
  mamba projections stays inside any tolerance) and surfaced as
  ``RuntimeError: expected mat1 and mat2 to have the same dtype`` in `in_proj`.

Two of those three had their own inline copy, and the copies had already drifted — the
expert-shape one lacked the ``language_model.`` forms the VL wrapper needs. Hence this module.

Deliberately dependency-free — no torch, no transformers, no vLLM, no env reads — so the
mapping is unit-testable anywhere and `glq` never has to import `glq_vllm`. Same reasoning as
`glq.moe_cpu_gate`.
"""
from __future__ import annotations

#: ``(checkpoint spelling, module spelling)`` prefix pairs, translated in BOTH directions.
#:
#: Both directions because a checkpoint written with native prefixes is equally legal and an
#: alias that matches nothing costs nothing. The nested pair covers the vision-language
#: wrapper, whose ``sd_prefix`` is ``language_model.backbone.layers``.
_PREFIX_PAIRS = (
    ("backbone.", "model."),
    ("language_model.backbone.", "language_model.model."),
)


def backbone_aliases(name: str) -> set[str]:
    """Alternative spellings of ``name``, excluding ``name`` itself.

    Returning a set *without* the input is what lets a caller write
    ``for cand in (name, *backbone_aliases(name))`` with no de-duplication, and leaves a
    non-NemotronH model paying for no extra lookups.

    The nested ``language_model.`` forms are checked first and win: a plain ``model.`` rule
    would otherwise rewrite ``language_model.model.…`` only at its tail, producing a spelling
    no checkpoint and no module tree ever uses.
    """
    for ckpt, module in reversed(_PREFIX_PAIRS):      # longest (nested) prefix first
        if name.startswith(ckpt):
            return {module + name[len(ckpt):]}
        if name.startswith(module):
            return {ckpt + name[len(module):]}
    return set()


def with_aliases(name: str) -> tuple[str, ...]:
    """``name`` followed by its aliases — the lookup order callers want.

    The original is always tried first, so a checkpoint and module tree that already agree
    resolve on the first attempt and the aliases are never consulted.
    """
    return (name, *sorted(backbone_aliases(name)))
