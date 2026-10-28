"""Which base model a benchmark record belongs to — the key every comparison groups on.

`base_model` means "the model this checkpoint was quantized from". The per-(model, GPU) table,
the model-card section and the %-of-bf16 index all group records by it, so a quantized build
that loses its base drops out of its family: a GLQ card would silently show no NVFP4 row.

Measured on the Hub, 2026-10-28:

    xv0y5ncu/gemma-4-31B-it-GLQ-trellis-3inst-4bpw  config._name_or_path = google/gemma-4-31B-it
    nvidia/Gemma-4-31B-IT-NVFP4                     config has NO base; card base_model = google/gemma-4-31B-it
    google/gemma-4-31B-it  (bf16)                   config has NO base; card base_model = google/gemma-4-31B

So NVFP4 needs the card. But the bf16 card names its PRETRAINING parent, and taking that would
pull the bf16 baseline out of its own family. Only a quantized checkpoint's card base is a
quantization parent; an unquantized checkpoint is the base.
"""
from __future__ import annotations

import pytest

from glq.bench import hfmeta


@pytest.fixture
def offline(monkeypatch):
    """model_meta with the network replaced: configs and card come from the test."""
    state = {"cfg": {}, "card": None}
    monkeypatch.setattr(hfmeta, "_load_configs", lambda m, t: (state["cfg"], {}, None))
    monkeypatch.setattr(hfmeta, "_card_base_model", lambda m, d, t: state["card"])
    monkeypatch.setattr(hfmeta, "_disk_gb", lambda d, m, t: None)
    return state


NVFP4_CFG = {"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}


def test_a_quantized_checkpoint_without_a_config_base_takes_its_cards(offline):
    offline["cfg"], offline["card"] = NVFP4_CFG, "google/gemma-4-31B-it"
    assert hfmeta.model_meta("nvidia/Gemma-4-31B-IT-NVFP4").base_model == \
        "google/gemma-4-31B-it"


def test_a_bf16_checkpoint_stays_the_base_whatever_its_card_says(offline):
    offline["cfg"], offline["card"] = {}, "google/gemma-4-31B"
    m = hfmeta.model_meta("google/gemma-4-31B-it")
    assert m.quant_method == "none"
    assert m.base_model is None


def test_the_config_base_wins_over_the_card(offline):
    offline["cfg"] = {"_name_or_path": "google/gemma-4-31B-it",
                      "quantization_config": {"quant_method": "glq"}}
    offline["card"] = "something/else"
    assert hfmeta.model_meta("xv0y5ncu/g-GLQ").base_model == "google/gemma-4-31B-it"


@pytest.mark.parametrize("front,want", [
    ("base_model: google/gemma-4-31B-it\n", "google/gemma-4-31B-it"),
    ("base_model:\n- google/gemma-4-31B-it\n", "google/gemma-4-31B-it"),
    ("base_model: ['google/gemma-4-31B-it']\n", "google/gemma-4-31B-it"),
    ("license: apache-2.0\n", None),
])
def test_a_local_checkpoints_card_frontmatter_is_read(tmp_path, front, want):
    """GLQ output dirs carry the card build_card wrote, with base_model in its frontmatter."""
    (tmp_path / "README.md").write_text(f"---\n{front}---\n\n# card\n", encoding="utf-8")
    assert hfmeta._card_base_model(str(tmp_path), tmp_path, None) == want


def test_a_local_checkpoint_without_a_card_has_no_card_base(tmp_path):
    assert hfmeta._card_base_model(str(tmp_path), tmp_path, None) is None
