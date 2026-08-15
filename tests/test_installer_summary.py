"""The "what now?" summary the installer prints when it finishes.

This is the entire user manual for someone who arrived via `curl … | bash`: they have no
repo checkout, no docs open, and no idea what got installed where. Every command printed
has to be copy-pasteable *as printed*.

The bug these tests were written against: the summary pointed at
`python examples/chat/app.py`, but `examples/` is not in the wheel
(`pyproject.toml` ships only `glq*` and `glq_vllm*`), so that path does not exist on a
pip-installed machine — the instruction was broken for precisely the audience that needs it.
Hence `test_no_command_references_the_unshipped_examples_directory`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.installer.__main__ import next_steps  # noqa: E402

MODEL = "xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel"
VENV = "/home/u/.glq/venv"
ALL = ("core", "vllm", "picode", "chat")


def _text(components=ALL, model=MODEL, size_gib=1.8):
    return next_steps(venv=VENV, model=model, components=components, port=8000,
                      size_gib=size_gib)


def test_it_names_the_model_that_was_chosen():
    """Nine checkpoints were on offer; the summary is where the user finds out which one
    they are about to download."""
    assert MODEL in _text()


def test_serve_command_is_complete_and_copy_pasteable():
    t = _text()
    assert f"{VENV}/bin/vllm serve {MODEL}" in t
    assert "--quantization glq" in t          # without this vLLM cannot load a GLQ repo
    assert "--port 8000" in t


def test_no_command_references_the_unshipped_examples_directory():
    """`examples/` is not in the wheel, so a curl|bash user has no such path."""
    assert "examples/chat" not in _text()
    assert "examples/" not in _text()


def test_chat_uses_an_installed_console_command():
    assert f"{VENV}/bin/glq-chat" in _text()
    assert "7860" in _text()


def test_chat_is_omitted_when_not_installed():
    t = _text(components=("core", "vllm"))
    assert "glq-chat" not in t


def test_picode_line_sources_nvm_because_it_is_not_on_the_path():
    """nvm is a shell function, absent from a non-login shell — the documented trap from
    benchmarks/harbor_pi_glq.py. Printing a bare `pi …` gives command-not-found."""
    t = _text()
    assert "~/.nvm/nvm.sh" in t
    assert f"pi --provider glq --model {MODEL}" in t


def test_picode_is_omitted_when_not_installed():
    assert "pi --provider" not in _text(components=("core", "vllm", "chat"))


def test_serve_command_enables_tool_choice_when_picode_is_installed():
    """pi is a tool-using agent. Against a server started without these flags every request
    dies with:

        400 "auto" tool choice requires --enable-auto-tool-choice and
            --tool-call-parser to be set

    Measured on the box: the plain serve line the summary used to print made picode
    unusable. The flags are only added when picode was actually installed — they change
    server behaviour, so nobody who did not ask for the agent should get them."""
    t = _text()
    assert "--enable-auto-tool-choice" in t
    assert "--tool-call-parser" in t


def test_serve_command_stays_plain_without_picode():
    t = _text(components=("core", "vllm", "chat"))
    assert "--enable-auto-tool-choice" not in t
    assert "--tool-call-parser" not in t


def test_the_tool_parser_family_is_called_out_as_model_dependent():
    """`hermes` matches SmolLM3's <tool_call> markup; other families need another parser,
    and a silently wrong one produces tool calls that never parse."""
    assert "hermes" in _text()
    assert "parser" in _text().lower()


def test_it_warns_that_the_first_serve_downloads_the_weights():
    """The single most surprising thing about the first run: `vllm serve` sits there for
    minutes pulling GiB. Saying so up front stops it looking like a hang."""
    t = _text(size_gib=22.4)
    assert "22.4" in t
    assert "download" in t.lower()


def test_it_says_how_to_change_the_model():
    assert "glq-setup" in _text()


def test_it_mentions_quantizing_your_own_model():
    """Half of what GLQ does. A user who only ever runs published checkpoints never
    discovers `glq-quantize` exists."""
    assert "glq-quantize" in _text()


def test_it_offers_a_way_to_check_the_server_is_up():
    """Distinguishes 'still loading' from 'broken' without reading vLLM's log."""
    t = _text()
    assert "/v1/models" in t


def test_it_records_where_the_config_went():
    assert "config.json" in _text()


def test_every_venv_command_is_absolute():
    """The venv is never activated by the installer, so a bare `vllm` or `glq-chat` would
    resolve to something else or nothing at all."""
    for line in _text().splitlines():
        stripped = line.strip()
        for cmd in ("vllm ", "glq-chat", "glq-setup", "glq-quantize"):
            if stripped.startswith(cmd):
                raise AssertionError(f"non-absolute command in summary: {stripped!r}")
