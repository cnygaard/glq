"""Don't let transformers' in-memory config renaming leak into a saved checkpoint.

GLQ writes ``config.json`` from ``save_cfg.to_dict()`` — the config **as transformers
loaded it**, not as the source repo declared it. Usually those agree. For Qwen4Exp they
do not: transformers 5.17's ``Qwen4ExpTextConfig`` rewrites ``"full_attention"`` to
``"qwen_sparse_attention"`` on load whenever ``indexer_n_heads`` is set.

That name is internal to transformers. vLLM's Qwen4Exp accepts only
``"linear_attention"`` and ``"full_attention"``::

    else:
        raise ValueError(f"Invalid layer_type {layer_type}")

so the saved checkpoint cannot be served at all — it fails during layer construction,
before a single weight is read. A bf16 copy saved the same way fails identically, which
is what makes it easy to misread as a GLQ quantization bug.

Measured on Qwen3.8-Flash-Next: source declares 36 ``linear_attention`` + 12
``full_attention``; the checkpoint we published declared 36 + 12
``qwen_sparse_attention`` and vLLM refused it.

These pin the restore. They are cheap and pure — no model, no GPU — because the failure
they guard costs an 11-hour run and a 76 GiB re-upload to discover.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.quantize_model import _restore_source_layer_types  # noqa: E402

RENAMED = ["linear_attention"] * 2 + ["qwen_sparse_attention"] * 2
SOURCE = ["linear_attention"] * 2 + ["full_attention"] * 2


def _src(tmp_path, raw: dict) -> str:
    with open(os.path.join(tmp_path, "config.json"), "w") as f:
        json.dump(raw, f)
    return str(tmp_path)


def test_the_source_name_is_restored_at_top_level(tmp_path):
    src = _src(tmp_path, {"layer_types": SOURCE})
    cfg = {"layer_types": list(RENAMED)}
    _restore_source_layer_types(cfg, src)
    assert cfg["layer_types"] == SOURCE


def test_the_source_name_is_restored_inside_text_config(tmp_path):
    """Qwen4Exp's real layout: the list lives under ``text_config``."""
    src = _src(tmp_path, {"text_config": {"layer_types": SOURCE}})
    cfg = {"text_config": {"layer_types": list(RENAMED)}}
    _restore_source_layer_types(cfg, src)
    assert cfg["text_config"]["layer_types"] == SOURCE


def test_a_nested_source_restores_a_flat_dict_too(tmp_path):
    """Streaming promotes the text config to top level, so the saved dict can be flat
    while the source repo's is nested. The names must still be restored."""
    src = _src(tmp_path, {"text_config": {"layer_types": SOURCE}})
    cfg = {"layer_types": list(RENAMED)}
    _restore_source_layer_types(cfg, src)
    assert cfg["layer_types"] == SOURCE


# ---- what it must not touch ---------------------------------------------------------

def test_a_config_without_layer_types_is_untouched(tmp_path):
    """Most architectures have no such key; this must be a no-op for them."""
    src = _src(tmp_path, {"hidden_size": 8})
    cfg = {"hidden_size": 8}
    _restore_source_layer_types(cfg, src)
    assert cfg == {"hidden_size": 8}


def test_a_source_without_layer_types_does_not_delete_ours(tmp_path):
    """If the source is silent we have nothing better to say — keep what we have rather
    than dropping a key the model needs."""
    src = _src(tmp_path, {"hidden_size": 8})
    cfg = {"layer_types": list(RENAMED)}
    _restore_source_layer_types(cfg, src)
    assert cfg["layer_types"] == RENAMED


def test_a_differing_length_is_refused(tmp_path):
    """A source list of another length describes a different model — copying it would
    silently mis-declare which layers hold KV. Leave ours alone."""
    src = _src(tmp_path, {"layer_types": SOURCE[:2]})
    cfg = {"layer_types": list(RENAMED)}
    _restore_source_layer_types(cfg, src)
    assert cfg["layer_types"] == RENAMED


def test_an_unreadable_source_is_not_fatal(tmp_path):
    """This runs at the very end of a multi-hour quantize. It must never be the thing
    that loses the run."""
    cfg = {"layer_types": list(RENAMED)}
    _restore_source_layer_types(cfg, str(tmp_path / "does-not-exist"))
    assert cfg["layer_types"] == RENAMED
    _src(tmp_path, {})
    with open(os.path.join(tmp_path, "config.json"), "w") as f:
        f.write("{ not json")
    _restore_source_layer_types(cfg, str(tmp_path))
    assert cfg["layer_types"] == RENAMED


def test_identical_names_are_a_no_op(tmp_path):
    """The common case: transformers renamed nothing."""
    src = _src(tmp_path, {"layer_types": SOURCE})
    cfg = {"layer_types": list(SOURCE)}
    _restore_source_layer_types(cfg, src)
    assert cfg["layer_types"] == SOURCE


# ---- the property that actually matters ---------------------------------------------

@pytest.mark.parametrize("saved", [RENAMED, SOURCE])
def test_the_saved_config_never_names_a_type_vllm_rejects(tmp_path, saved):
    """vLLM's Qwen4Exp raises ``ValueError: Invalid layer_type`` for anything outside
    this pair, so the restored list must contain only names it accepts."""
    src = _src(tmp_path, {"text_config": {"layer_types": SOURCE}})
    cfg = {"text_config": {"layer_types": list(saved)}}
    _restore_source_layer_types(cfg, src)
    assert set(cfg["text_config"]["layer_types"]) <= {"linear_attention", "full_attention"}
