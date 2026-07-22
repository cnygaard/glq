"""GLQ model-card generation (glq/model_card.py + templates/model_card.md.j2).

The card is the *original* base-model card with GLQ sections injected on top: it
inherits the original YAML frontmatter (license, language, tags) + adds GLQ tags
and base_model relation, renders the install/vLLM/Transformers/coding-agent/E8-KV
sections, an optional benchmarks table, and appends the original body verbatim
under a collapsible block. These tests render against a synthetic quant_out dir
(no network: the original README fetch fails gracefully to an empty body) and
assert the structure + frontmatter merge.
"""

import json
from pathlib import Path

import pytest

jinja2 = pytest.importorskip("jinja2")

from glq.model_card import (  # noqa: E402
    _merged_frontmatter, _split_frontmatter, _bpw_label, build_card,
)


def _write_quant_dir(tmp_path, *, bpw=4, layer_bpw=None, arch="LlamaForCausalLM",
                     auto_map=None, extra_cfg=None, codebook=None, variant=None):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    qmeta = {"bpw": bpw, "avg_sqnr_db": 21.5, "n_quantized_layers": 224,
             "nsamples": 128, "seqlen": 2048}
    if codebook:
        qmeta["codebook"] = codebook
    if variant:
        qmeta["variant"] = variant
    (tmp_path / "quantize_config.json").write_text(json.dumps(qmeta))
    cfg = {"architectures": [arch]}
    if auto_map:
        cfg["auto_map"] = auto_map
    qc = {"bpw": bpw}
    if layer_bpw:
        qc["layer_bpw"] = layer_bpw
    if codebook:
        qc["codebook"] = codebook
    cfg["quantization_config"] = qc
    if extra_cfg:
        cfg.update(extra_cfg)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return tmp_path


def test_split_frontmatter_roundtrip():
    pytest.importorskip("yaml")
    text = "---\nlicense: apache-2.0\ntags:\n- foo\n---\n# Title\n\nBody here.\n"
    fm, body = _split_frontmatter(text)
    assert fm["license"] == "apache-2.0"
    assert fm["tags"] == ["foo"]
    assert body.startswith("# Title")


def test_split_frontmatter_no_frontmatter():
    fm, body = _split_frontmatter("# Just a body\n")
    assert fm == {}
    assert body == "# Just a body\n"


def test_merged_frontmatter_inherits_and_adds():
    orig = {"license": "apache-2.0", "language": ["en"], "tags": ["text-generation"]}
    fm = _merged_frontmatter(orig, "google/gemma-4-e4b-it")
    assert fm["license"] == "apache-2.0"           # inherited, not relabeled
    assert fm["language"] == ["en"]
    assert fm["base_model"] == "google/gemma-4-e4b-it"
    assert fm["base_model_relation"] == "quantized"
    for t in ("glq", "quantization", "e8-lattice"):
        assert t in fm["tags"]
    assert "text-generation" in fm["tags"]          # original tag preserved


def test_merged_frontmatter_default_license():
    fm = _merged_frontmatter({}, "some/model")
    assert fm["license"] == "other"                 # safe fallback when none given


def test_bpw_label():
    assert _bpw_label(4.0, False, 4, 4) == "4bpw"
    assert _bpw_label(5.0, True, 3, 8) == "5.0bpw (mixed 3–8)"


def test_build_card_uniform(tmp_path):
    out = _write_quant_dir(tmp_path, bpw=4)
    card = build_card(out, "google/gemma-4-e4b-it",
                      repo_id="xv0y5ncu/Test-GLQ-4bpw", write=True)
    # frontmatter prepended
    assert card.startswith("---\n")
    fm, body = _split_frontmatter(card)
    assert fm["base_model"] == "google/gemma-4-e4b-it"
    assert "glq" in fm["tags"]
    # GLQ sections present
    assert "## Install" in body
    assert "pip install glq" in body
    assert 'quantization="glq"' in body
    assert "## Use with Transformers" in body
    assert "AutoModelForCausalLM" in body          # not multimodal
    assert "GLQ_KV_QUANT=e8_relaxed:2" in body     # E8 KV recipe
    assert "xv0y5ncu/Test-GLQ-4bpw" in body        # repo id threaded into examples
    assert "GLQ on GitHub" in body                 # footer
    # written to disk
    assert (out / "README.md").exists()


def test_build_card_no_benchmarks_table_when_absent(tmp_path):
    out = _write_quant_dir(tmp_path, bpw=3)
    card = build_card(out, "x/y", benchmarks=None, write=False)
    assert "## Benchmarks" not in card


def test_build_card_with_benchmarks(tmp_path):
    out = _write_quant_dir(tmp_path, bpw=4)
    bench = [{"task": "MMLU-Pro", "metric": "exact_match", "n": 247, "value": "65.2%"}]
    card = build_card(out, "x/y", benchmarks=bench, write=False)
    assert "## Benchmarks" in card
    assert "MMLU-Pro" in card
    assert "65.2%" in card
    assert "n=247" in card


def test_build_card_mixed_precision(tmp_path):
    layer_bpw = {"model.layers.0.self_attn.q_proj": 3,
                 "model.layers.0.mlp.down_proj": 8}
    out = _write_quant_dir(tmp_path, bpw=5.0, layer_bpw=layer_bpw)
    card = build_card(out, "x/y", write=False)
    assert "mixed" in card
    assert "3–8" in card or "3-8" in card


def test_build_card_multimodal_flags(tmp_path):
    out = _write_quant_dir(tmp_path, bpw=4, arch="Gemma4ForConditionalGeneration",
                           auto_map={"AutoModel": "x"})
    card = build_card(out, "google/gemma-4-e4b-it", write=False)
    assert "AutoModelForImageTextToText" in card    # multimodal auto class
    assert "trust_remote_code=True" in card         # auto_map -> trust remote
    assert "limit_mm_per_prompt" in card            # text-only serving note


def test_build_card_trellis(tmp_path):
    # A trellis (TCQ) checkpoint must describe the trellis method, NOT the E8-shell one.
    out = _write_quant_dir(tmp_path, bpw=4, codebook="trellis", variant="hyb")
    card = build_card(out, "google/gemma-4-12B-it",
                      repo_id="xv0y5ncu/gemma-4-12B-it-trellis-4bpw", write=False)
    fm, body = _split_frontmatter(card)
    # trellis-correct method prose
    low = body.lower()
    assert "trellis" in low and ("tcq" in low or "trellis-coded" in low)
    # must NOT claim the E8-shell weight codebook (the KV-cache section legitimately says E8)
    assert "65,536-point subset" not in body
    assert "E8 shell codebook" not in body
    # codebook-aware tags
    assert "trellis" in fm["tags"] and "e8-lattice" not in fm["tags"]
    # shared GLQ scaffolding still present
    assert "pip install glq" in body and "GLQ on GitHub" in body


def test_build_card_shell_unchanged_default(tmp_path):
    # no `codebook` key (legacy checkpoints) → E8-shell prose + e8-lattice tag, unchanged.
    out = _write_quant_dir(tmp_path, bpw=4)
    card = build_card(out, "x/y", write=False)
    fm, _ = _split_frontmatter(card)
    assert "e8-lattice" in fm["tags"] and "trellis" not in fm["tags"]
    assert "E8" in card  # the E8-shell method prose


def test_build_card_sweet_spot_callout(tmp_path):
    # >4 bpw -> steer toward E8 KV cache
    out_hi = _write_quant_dir(tmp_path / "hi", bpw=8)
    card_hi = build_card(out_hi, "x/y", write=False)
    assert "E8 KV cache" in card_hi
    # 2-4 bpw -> sweet spot
    out_lo = _write_quant_dir(tmp_path / "lo", bpw=3)
    card_lo = build_card(out_lo, "x/y", write=False)
    assert "2–4" in card_lo or "2-4" in card_lo
