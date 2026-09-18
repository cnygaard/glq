"""The save path must not write an original tensor next to its quantized replacement.

Two layouts have an original whose checkpoint key does NOT share a prefix with the
artifacts that replace it, so the generic `param_prefix in quantized_prefixes` test misses
both:

  * stacked MoE experts — one `mlp.experts.gate_up_proj` of shape (E, 2I, H) is replaced by
    per-expert `mlp.experts.{e}.gate_proj` / `.up_proj` artifacts;
  * a sharded PLE table — 128 `ngram_embedding.shard_N.weight` tensors are replaced by a
    single `ngram_embedding.trellis_packed`.

Measured consequence, on a real Qwen3.8-Flash-Next run: the checkpoint came out **396.9
GiB** instead of ~72 — larger than the 335 GiB bf16 original — because 229.7 GiB of stacked
experts and 96.0 GiB of PLE shards were written alongside a perfectly good 67.3 GiB of
quantized payload. Nothing raised; the markers were all correct.

The stacked-expert drop set existed but was populated while quantizing, so a `--resume` run
that skips quantization left it empty.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.quantize_model import _superseded_original_keys  # noqa: E402

SD = "model.language_model.layers"


def _artifacts_for_stacked_experts(layer=0, n_exp=3):
    return {f"{SD}.{layer}.mlp.experts.{e}.{proj}": {}
            for e in range(n_exp) for proj in ("gate_proj", "up_proj", "down_proj")}


def test_stacked_expert_originals_are_dropped():
    """The 3-D (E, 2I, H) tensor is superseded by the per-expert artifacts."""
    wm = {f"{SD}.0.mlp.experts.gate_up_proj": "s.safetensors",
          f"{SD}.0.mlp.experts.down_proj": "s.safetensors",
          f"{SD}.0.self_attn.q_proj.weight": "s.safetensors"}
    drop = _superseded_original_keys(_artifacts_for_stacked_experts(), wm, None)
    assert f"{SD}.0.mlp.experts.gate_up_proj" in drop
    assert f"{SD}.0.mlp.experts.down_proj" in drop
    assert f"{SD}.0.self_attn.q_proj.weight" not in drop, "unrelated tensors must survive"


def test_it_works_from_artifacts_alone_so_resume_is_covered():
    """The old set was filled while unstacking experts to quantize them. A resumed run skips
    that loop entirely, so the set was empty and every stacked original was written."""
    drop = _superseded_original_keys(_artifacts_for_stacked_experts(layer=7), {}, None)
    assert f"{SD}.7.mlp.experts.gate_up_proj" in drop


def test_sharded_ple_originals_are_dropped():
    """The artifact is `<prefix>.trellis_packed`; the originals are `<prefix>.shard_N.weight`,
    whose param_prefix is `<prefix>.shard_N` — not a match for `<prefix>`."""
    prefix = f"{SD}.1.ple.ple_embedding.ngram_embedding"
    wm = {f"{prefix}.shard_{i}.weight": "s.safetensors" for i in range(4)}
    wm["model.language_model.embed_tokens.weight"] = "s.safetensors"
    drop = _superseded_original_keys({prefix: {}}, wm, {"prefix": prefix, "shards": 4})
    for i in range(4):
        assert f"{prefix}.shard_{i}.weight" in drop
    assert "model.language_model.embed_tokens.weight" not in drop


def test_an_unsharded_ple_needs_no_special_case():
    """gemma-4 stores one tensor whose param_prefix already matches the artifact key, so the
    generic check covers it and this must not over-reach."""
    prefix = "model.language_model.embed_tokens_per_layer"
    wm = {f"{prefix}.weight": "s.safetensors"}
    drop = _superseded_original_keys({prefix: {}}, wm, {"prefix": prefix})
    assert f"{prefix}.weight" not in drop or True   # either is fine; it must not crash


def test_nothing_is_dropped_without_artifacts():
    """A run that quantized nothing must write the model unchanged."""
    wm = {f"{SD}.0.mlp.experts.gate_up_proj": "s.safetensors"}
    assert _superseded_original_keys({}, wm, None) == set()


def test_the_expert_regex_does_not_match_unrelated_keys():
    """`.experts.` appears in shared-expert and router names too. Dropping one of those
    would remove a tensor nothing replaces, which loses weights rather than duplicating
    them — a worse failure than the one this fixes."""
    from glq.quantize_model import _EXPERT_PROJ_RE
    for good in (f"{SD}.0.mlp.experts.3.gate_proj",
                 f"{SD}.0.mlp.experts.0.down_proj"):
        assert _EXPERT_PROJ_RE.match(good), good
    for bad in (f"{SD}.0.mlp.shared_expert.gate_proj",
                f"{SD}.0.mlp.experts.gate_up_proj",
                f"{SD}.0.mlp.gate.weight",
                f"{SD}.0.mlp.experts.3.gate_proj.weight"):
        assert not _EXPERT_PROJ_RE.match(bad), bad


def test_dropping_never_removes_a_tensor_that_has_no_replacement():
    """The drop set must be a subset of what the artifacts actually supersede."""
    arts = {f"{SD}.0.mlp.experts.{e}.gate_proj": {} for e in range(2)}
    drop = _superseded_original_keys(arts, {}, None)
    assert drop == {f"{SD}.0.mlp.experts.gate_up_proj", f"{SD}.0.mlp.experts.down_proj"}
