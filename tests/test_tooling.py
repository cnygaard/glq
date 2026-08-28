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


@pytest.mark.parametrize("model", [
    "xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel",
    "xv0y5ncu/Qwen3.5-2B-GLQ-trellis-3inst-4bpw",
    "xv0y5ncu/Qwen3.8-27B-GLQ-trellis-3inst-4bpw",
])
def test_smollm3_and_qwen_use_hermes(model):
    """Both families emit hermes-style <tool_call> markup; SmolLM3+hermes is the pairing
    the Terminal-Bench integration validated."""
    args = T.tool_serve_args(model)
    joined = " ".join(args)
    assert "--tool-call-parser hermes" in joined
    assert "--enable-auto-tool-choice" in joined
    assert "--chat-template" not in joined


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
