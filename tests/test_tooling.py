"""Family-aware tool-calling serve args — the knowledge that drifted.

Which parser (and which template) a model family needs for vLLM tool calling lived in
two places and disagreed once already: the installer printed `hermes` for every model,
which matches SmolLM3/Qwen-style ``<tool_call>`` markup and silently mangles gemma-4's.
`glq/tooling.py` is now the single source; the installer and `glq-code` both read it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq import tooling as T  # noqa: E402


def test_gemma4_gets_its_own_parser_and_the_external_template(tmp_path):
    args = T.tool_serve_args("xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw",
                             templates_dir=tmp_path)
    joined = " ".join(args)
    assert "--enable-auto-tool-choice" in joined
    assert "--tool-call-parser gemma4" in joined
    assert "--reasoning-parser gemma4" in joined
    assert f"--chat-template {tmp_path / T.GEMMA4_TOOL_TEMPLATE}" in joined
    # Without enable_thinking the template never opens a thought section, but the
    # RL-trained model thinks anyway — measured live: <|thought|> markers and tool-call
    # syntax leaking into prose, plus a "thoughtthoughtthought" repetition loop in the
    # reasoning field. The README's validated recipe always carried this kwarg; compact
    # JSON (no spaces) so the printed shell command stays copy-pasteable unquoted.
    assert '--default-chat-template-kwargs {"enable_thinking":true}' in joined


def test_smollm3_uses_hermes():
    """SmolLM3 renders tool calls as JSON inside <tool_call> — hermes' format. Confirmed
    from the published checkpoint's own chat_template.jinja: `"arguments"` x2, `<tool_call>`
    x2, `<function=` x0. SmolLM3+hermes is also the pairing Terminal-Bench validated."""
    args = T.tool_serve_args("xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel")
    joined = " ".join(args)
    assert "--tool-call-parser hermes" in joined
    assert "--enable-auto-tool-choice" in joined
    assert "--chat-template" not in joined


@pytest.mark.parametrize("model", [
    "xv0y5ncu/Qwen3.5-0.8B-GLQ-trellis-3inst-4bpw",
    "xv0y5ncu/Qwen3.5-2B-GLQ-trellis-3inst-4bpw",
    "xv0y5ncu/Qwen3.8-27B-GLQ-trellis-3inst-4bpw",
    "xv0y5ncu/Qwen3.8-27B-GLQ-trellis-3inst-3bpw",
    "xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4",
])
def test_qwen_does_not_use_hermes(model):
    """hermes was wrong for every Qwen we publish, and wrong in the silent direction: it
    parses JSON inside <tool_call>, while all five of these templates emit XML tags.

    Checked against each published checkpoint's own chat_template.jinja — `<function=` x5 and
    `"arguments"` x0 for all five. Qwen3.8-Flash-Next's template both INSTRUCTS the format
    ("If you choose to call a function ONLY reply in the following format ... <tool_call>
    <function=example_function_name> <parameter=...") and renders past calls in it, so there
    is no reading under which a JSON parser would see a tool call."""
    joined = " ".join(T.tool_serve_args(model))
    assert "--tool-call-parser hermes" not in joined
    assert "--enable-auto-tool-choice" in joined
    assert "--chat-template" not in joined


@pytest.mark.parametrize("model", [
    "xv0y5ncu/Qwen3.5-2B-GLQ-trellis-3inst-4bpw",
    "xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4",
])
def test_qwen_uses_the_parser_for_its_xml_tool_markup(model):
    """`qwen3_xml` is the name in Qwen's own serving recipe for Qwen3.8-27B. In this vLLM it
    and `qwen3_coder` are aliases — `tool_parsers/__init__.py` maps both to
    `Qwen3EngineToolParser`, one `Qwen3Parser` whose docstring is exactly the
    `<tool_call>/<function=/<parameter=` format these templates emit — so either name works
    and the vendor's is the one to carry."""
    joined = " ".join(T.tool_serve_args(model))
    assert "--tool-call-parser qwen3_xml" in joined
    assert "--reasoning-parser qwen3" in joined


@pytest.mark.parametrize("model", [
    "xv0y5ncu/Qwen3.5-2B-GLQ-trellis-3inst-4bpw",
    "xv0y5ncu/Qwen3.8-27B-GLQ-trellis-3inst-4bpw",
])
def test_qwen_gets_the_qwen3_reasoning_parser_and_text_only_load(model):
    """Qwen3.x are thinking models: without --reasoning-parser qwen3 the <think> block
    stays in `content` and leaks into the agent's prose — the same failure class as
    gemma-4's enable_thinking leak, and it reads as rambling/repetition in pi. Parser
    name per vLLM's official Qwen3.5 recipe. --language-model-only skips the multimodal
    wrapper's bf16 vision tower, which a text-only agent/chat never uses."""
    joined = " ".join(T.tool_serve_args(model))
    assert "--reasoning-parser qwen3" in joined
    assert "--language-model-only" in joined


def test_smollm3_keeps_the_validated_pairing_without_a_reasoning_parser():
    """SmolLM3+hermes WITHOUT a reasoning parser is what Terminal-Bench validated
    end to end; qwen3's parser is not known to match SmolLM3's markup. Don't drift it
    as a side effect of the Qwen fix."""
    joined = " ".join(T.tool_serve_args("xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel"))
    assert "--reasoning-parser" not in joined
    assert "--language-model-only" not in joined


def test_unknown_families_get_none_not_a_guess(tmp_path):
    """A silently wrong parser produces tool calls that never parse — the worst failure
    mode, because it looks like a bad model rather than a bad flag."""
    assert T.tool_serve_args("mistralai/Devstral-Small-2-24B") is None


def test_template_is_returned_from_cache_without_fetching(tmp_path):
    tpl = tmp_path / T.GEMMA4_TOOL_TEMPLATE
    tpl.write_text("{% macro x %}")

    def no_fetch(url):
        raise AssertionError("fetched despite a cached template")

    assert T.ensure_gemma4_template(templates_dir=tmp_path, fetch=no_fetch) == tpl


def test_template_is_fetched_and_cached_when_missing(tmp_path):
    seen = []

    def fetch(url):
        seen.append(url)
        return b"{% macro format_parameters %}"

    tpl = T.ensure_gemma4_template(templates_dir=tmp_path, fetch=fetch)
    assert tpl.read_bytes() == b"{% macro format_parameters %}"
    assert seen == [T.GEMMA4_TOOL_TEMPLATE_URL]


def test_a_failed_fetch_raises_with_the_manual_command(tmp_path):
    """glq-code must work on installs that predate the installer download — and when the
    fetch fails there too, the error has to say what to run, not just that it failed."""
    def fetch(url):
        raise OSError("could not resolve host")

    with pytest.raises(RuntimeError, match="curl"):
        T.ensure_gemma4_template(templates_dir=tmp_path, fetch=fetch)


# ============================================ family-aware sampling defaults

# The model card names sampling the client cannot always send. `top_k` is not in the OpenAI
# schema at all, so pi drops it silently and Qwen3.8-Flash-Next samples at vLLM's top_k=0
# (off) instead of the 20 its card asks for. Verified against the serving vLLM:
# `--override-generation-config` lands in `default_sampling_params`, and
# `chat_completion/protocol.py:to_sampling_params` resolves
# "user -> server default -> OpenAI default" over fields that are None on the request.

import json


def _cfg(model_id):
    """The JSON dict a model's sampling args carry, or None when nothing is emitted."""
    args = T.sampling_serve_args(model_id)
    if args is None:
        return None
    assert args[0] == "--override-generation-config", args
    return json.loads(args[1])


def test_qwen_gets_the_top_k_its_card_asks_for():
    """20, not the 64 that was being applied to every family because gemma-4's numbers
    were the only ones in the table."""
    assert _cfg("xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4")["top_k"] == 20


def test_gemma4_keeps_the_values_that_were_already_shipping():
    cfg = _cfg("xv0y5ncu/gemma-4-26B-A4B-it-GLQ-4bpw")
    assert (cfg["temperature"], cfg["top_p"], cfg["top_k"]) == (1.0, 0.95, 64)


def test_smollm3_asks_for_no_top_k_at_all():
    """Its card specifies temperature 0.6 and no top_k. Emitting top_k=0 would be the same
    thing to vLLM, but emitting 64 — as the shared default did — is a different model."""
    cfg = _cfg("xv0y5ncu/SmolLM3-3B-trellis-3inst-6bpw")
    assert cfg["temperature"] == 0.6
    assert "top_k" not in cfg


def test_an_unknown_family_is_left_entirely_alone():
    """Unlike the tool parser, a missing sampling override has no silent-failure mode: the
    checkpoint's own generation_config.json is a better answer than a guess."""
    assert T.sampling_serve_args("some-org/unheard-of-model") is None


def test_only_parameters_vllm_actually_reads_are_emitted():
    """`get_diff_sampling_param` filters to a fixed list; a key outside it is silently
    dropped, which would make this function look like it was doing something it was not."""
    understood = {"repetition_penalty", "presence_penalty", "frequency_penalty",
                  "temperature", "top_k", "top_p", "min_p", "max_new_tokens"}
    for model in ("org/Qwen3.8", "org/gemma-4-26B", "org/SmolLM3-3B"):
        assert set(_cfg(model)) <= understood, model


def test_the_json_is_compact_enough_to_paste_into_a_shell():
    """These args get printed as copy-pasteable commands by the installer; spaces inside an
    unquoted JSON argument break the paste."""
    args = T.sampling_serve_args("org/Qwen3.8")
    assert " " not in args[1], args[1]


def test_neutral_parameters_are_not_restated():
    """min_p=0.0, presence_penalty=0.0 and repetition_penalty=1.0 on Qwen's card are already
    vLLM's own OpenAI-path defaults (`_DEFAULT_SAMPLING_PARAMS`). Setting them changes
    nothing, and a config that lists them implies the opposite."""
    cfg = _cfg("org/Qwen3.8")
    for neutral in ("min_p", "presence_penalty", "repetition_penalty",
                    "frequency_penalty"):
        assert neutral not in cfg, f"{neutral} is already vLLM's default"


def test_the_chat_slider_defaults_follow_the_same_table():
    """glq-chat hardcoded gemma-4's 1.0/0.95/64 for every model, so a Qwen session started
    with the wrong top_k unless the user knew to move the slider."""
    from glq.chat import recommended_sampling
    assert recommended_sampling("org/Qwen3.8-Flash-Next")["top_k"] == 20
    assert recommended_sampling("org/gemma-4-26B")["top_k"] == 64
    assert recommended_sampling("org/SmolLM3-3B")["temperature"] == 0.6
    # 0 = off in the UI, which is how "no top_k" is expressed on a slider.
    assert recommended_sampling("org/SmolLM3-3B")["top_k"] == 0


def test_an_unknown_model_still_gets_usable_slider_values():
    """The UI cannot show None on a slider. Unknown falls back to neutral values rather
    than to one family's numbers."""
    cfg = recommended = None
    from glq.chat import recommended_sampling
    cfg = recommended_sampling("some-org/unheard-of-model")
    assert set(cfg) == {"temperature", "top_p", "top_k"}
    assert all(v is not None for v in cfg.values())
