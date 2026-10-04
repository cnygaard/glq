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
    # The INSTALLABLE command, not any occurrence of the substring: the block also says
    # "a plain `pip install glq` leaves `vllm` not a command", so a looser assertion passes
    # on the sentence warning against it even if the card gives no working command.
    assert "pip install 'glq[vllm]'" in body
    assert 'quantization="glq"' in body
    assert "## Use with Transformers" in body
    assert "AutoModelForCausalLM" in body          # not multimodal
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
    assert "pip install 'glq[vllm]'" in body and "GLQ on GitHub" in body


def test_build_card_shell_unchanged_default(tmp_path):
    # no `codebook` key (legacy checkpoints) → E8-shell prose + e8-lattice tag, unchanged.
    out = _write_quant_dir(tmp_path, bpw=4)
    card = build_card(out, "x/y", write=False)
    fm, _ = _split_frontmatter(card)
    assert "e8-lattice" in fm["tags"] and "trellis" not in fm["tags"]
    assert "E8" in card  # the E8-shell method prose


def test_build_card_sweet_spot_callout(tmp_path):
    # >4 bpw -> says what the rung IS for, now that it can no longer point at the E8 KV
    # cache (which does not start on vLLM >= 0.27).
    out_hi = _write_quant_dir(tmp_path / "hi", bpw=8)
    card_hi = build_card(out_hi, "x/y", write=False)
    assert "8.0 bpw" in card_hi
    assert "close to bf16" in card_hi
    # 2-4 bpw -> sweet spot
    out_lo = _write_quant_dir(tmp_path / "lo", bpw=3)
    card_lo = build_card(out_lo, "x/y", write=False)
    assert "2–4" in card_lo or "2-4" in card_lo


# ---- E8 KV cache: removed from the card --------------------------------------------

def test_no_card_advertises_the_e8_kv_cache(tmp_path):
    """The E8 KV cache does not start on vLLM >= 0.27.

    Every stage still announces itself and then EngineCore exits on
    ``kv_cache_stride_order``. It was removed from the README on 2026-08-16 for exactly
    that reason, but the card template kept a whole section plus a pitch in the opening
    blockquote — so every generated card shipped serve flags that produce a dead engine,
    on repos users reach before they reach the README.

    Both bpw branches are checked: the >4.05 branch made the KV cache the *only*
    justification it offered for a high-bpw checkpoint.
    """
    for bpw in (3, 5):
        out = _write_quant_dir(tmp_path / f"b{bpw}", bpw=bpw)
        card = build_card(out, "x/y", write=False)
        for probe in ("E8 KV cache", "GLQ_KV_QUANT", "e8_relaxed:2",
                      "GLQ_KV_E8_SIDECAR", "Smaller KV cache"):
            assert probe not in card, f"{probe!r} still advertised at {bpw} bpw"


def test_the_high_bpw_branch_still_says_something_useful(tmp_path):
    """Deleting the KV-cache clause must not leave ">4 bpw: savings are modest" dangling
    with no follow-up — that reads as a reason not to use the checkpoint at all."""
    out = _write_quant_dir(tmp_path, bpw=6)
    card = build_card(out, "x/y", write=False)
    assert "2–4 bits/weight" in card
    head = card[: card.find("## Install")]
    assert "modest" not in head or len(head.split("modest")[1].split("\n")[0]) > 40


def _card(tmp_path, **kw):
    """A built card body for the Install-block tests."""
    out = _write_quant_dir(tmp_path, **kw)
    return build_card(out, "google/gemma-4-e4b-it",
                      repo_id="xv0y5ncu/Test-GLQ-4bpw", write=True)


# ----------------------------------------------- the Install block must cover what it shows

# Reported from a real attempt: the card said `pip install glq`, the reader followed it, and
# then `vllm` was not a command. glq's own dependencies are `torch` and `numpy` -- vLLM is
# not among them, and `transformers`/`accelerate` are behind the `hf` extra whose own
# pyproject comment already records that `pip install glq` alone fails that snippet. So the
# single line was insufficient for EVERY usage section on the card.

def _install_block(body: str) -> str:
    """The Install section, up to the next heading."""
    assert "## Install" in body
    after = body.split("## Install", 1)[1]
    return after.split("\n## ", 1)[0]


def test_the_install_block_installs_vllm_because_the_card_tells_you_to_use_it(tmp_path):
    body = _split_frontmatter(_card(tmp_path))[1]
    block = _install_block(body)
    assert "vllm serve" in body or "from vllm import" in body, "no vLLM usage to support"
    assert "glq[vllm]" in block, (
        f"the card shows vLLM usage but never installs it:\n{block}")


def test_the_install_block_creates_a_virtual_environment(tmp_path):
    """`vllm` has to land on PATH for `vllm serve` to resolve, and a venv is what makes the
    console scripts reachable without touching the system interpreter."""
    block = _install_block(_split_frontmatter(_card(tmp_path))[1])
    assert "python -m venv" in block
    assert "activate" in block


def test_vllm_and_glq_are_installed_in_one_transaction(tmp_path):
    """Two separate pip installs let pip resolve torch twice and silently move it under the
    other package. The extra makes it one command and one resolution."""
    block = _install_block(_split_frontmatter(_card(tmp_path))[1])
    assert "glq[vllm]" in block
    assert "pip install vllm\n" not in block, "vLLM installed on its own"


def test_the_transformers_path_gets_its_extra(tmp_path):
    """`import glq.hf_integration` needs transformers, and device_map="auto" needs
    accelerate -- that is exactly what the `hf` extra is for."""
    body = _split_frontmatter(_card(tmp_path))[1]
    if "## Use with Transformers" not in body:
        return
    block = _install_block(body)
    assert "glq[hf]" in block, f"transformers path shown but its extra never named:\n{block}"


def test_the_venv_path_is_consistent_between_create_and_activate(tmp_path):
    """A card that creates `env` and activates `venv` leaves the reader in the system
    interpreter, which is the failure this whole block exists to prevent."""
    import re
    block = _install_block(_split_frontmatter(_card(tmp_path))[1])
    created = re.search(r"python -m venv (\S+)", block)
    activated = re.search(r"source (\S+)/bin/activate", block)
    assert created and activated, block
    assert created.group(1) == activated.group(1), (
        f"creates {created.group(1)} but activates {activated.group(1)}")
