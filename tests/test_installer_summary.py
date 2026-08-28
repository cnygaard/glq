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


def test_picode_step_is_the_one_command_not_an_incantation():
    """The old step printed `. ~/.nvm/nvm.sh && pi …` — one dropped dot and Ubuntu
    suggests installing the unrelated Raspberry-Pi package. glq-code resolves pi, serves
    with family-correct tool flags, and frees the GPU when pi exits; the step is one
    word."""
    t = _text()
    assert "glq-code" in t
    assert "~/.nvm/nvm.sh" not in t


def test_picode_is_omitted_when_not_installed():
    assert "glq-code" not in _text(components=("core", "vllm", "chat"))


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


def test_it_shows_a_request_that_actually_generates_text():
    """`/v1/models` proves the port is open, not that the stack can decode a token.

    Those are different failures and only the second one is the interesting one: GLQ builds
    its CUDA kernels on the user's machine, so a server can start, list the model, and still
    be unable to run a forward pass. Someone who arrived via `curl … | bash` has no repo and
    no client wired up, so without a generation request printed here their first real
    inference is whenever they get round to writing one — and any failure lands far away
    from the install that caused it.

    The distro suite gates on exactly this completion ("The capital of France is" -> Paris);
    the text handed to the user should let them run the same check.
    """
    t = _text()
    assert "/v1/completions" in t or "/v1/chat/completions" in t, (
        "next_steps never shows a request that returns generated text")
    assert "prompt" in t or "messages" in t, "the request carries no prompt"


def test_the_generation_request_names_the_installed_model():
    """A copy-pasted body with a placeholder model id comes back 404 from vLLM, which reads
    as a broken install rather than a wrong argument."""
    t = _text(model="xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel")
    body = [ln for ln in t.splitlines()
            if "/v1/completions" in ln or "/v1/chat/completions" in ln or '"model"' in ln]
    assert any("xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel" in ln for ln in body), (
        f"the generation request does not name the model:\n" + "\n".join(body))


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


# The handoff. `glq-chat` now starts vLLM itself, waits for it, opens a browser and stops
# the server again on exit — so it is the one command a new user needs, and printing
# `vllm serve` above it sends them to a blocking terminal and a second window instead.

def test_the_chat_command_is_the_first_step_when_it_is_installed():
    first = [ln for ln in _text().splitlines() if ln.startswith("1. ")]
    assert first, "the summary has no numbered steps"
    assert "chat" in first[0].lower(), f"step 1 is not the chat: {first[0]!r}"


def test_the_chat_step_says_it_starts_the_server_itself():
    """Otherwise the user starts `vllm serve` first out of habit and wonders why the second
    one fails on a port collision."""
    t = _text()
    chat_block = t.split("1. ", 1)[1].split("\n2. ", 1)[0]
    assert "glq-chat" in chat_block
    assert "start" in chat_block.lower()
    assert "vllm" in chat_block.lower()


def test_the_first_run_warning_sits_on_the_step_the_user_runs_first():
    """The weight download and the kernel build happen on whichever command they run first;
    a warning attached to a command they never type does not warn anyone."""
    t = _text(size_gib=22.4)
    first_block = t.split("1. ", 1)[1].split("\n2. ", 1)[0]
    assert "glq-chat" in first_block, "step 1 is not the command they will actually run"
    assert "22.4" in first_block


def test_serving_by_hand_is_still_documented_for_headless_use():
    """`--no-serve`, remote boxes, and anything that is not the Gradio UI still need it."""
    t = _text()
    assert f"{VENV}/bin/vllm serve {MODEL}" in t


def test_serving_is_step_one_when_the_chat_was_not_installed():
    first = [ln for ln in _text(components=("core", "vllm")).splitlines()
             if ln.startswith("1. ")]
    assert "serve" in first[0].lower()


def test_the_summary_is_closed_off_by_its_separator():
    """A missing comma in the closing list once concatenated the sign-off with the rule,
    producing one 1000-character line and no visible end to the output."""
    lines = _text().splitlines()
    assert lines[-1].strip() == "=" * 74, f"summary does not end with a rule: {lines[-1]!r}"
    longest = max(lines, key=len)
    assert len(longest) < 200, f"runaway line in the summary ({len(longest)} chars)"


def test_the_chat_step_mentions_the_shareable_link():
    """The chat publishes a public gradio.live URL by default. Someone reading only this
    summary should learn that from the summary, not from a surprise line in the log."""
    t = _text()
    chat_block = t.split("1. ", 1)[1].split("\n2. ", 1)[0]
    assert "gradio.live" in chat_block or "share" in chat_block.lower()


# KV compression is a serving-time choice, so every way the summary tells someone to serve
# has to carry it — the `vllm serve` line and the picode line alike. It is set through the
# environment, which means a user who copies the plain command silently gets fp16.

def test_the_serve_line_shows_how_to_turn_the_fp8_cache_on():
    t = next_steps(venv=VENV, model=MODEL, components=ALL, port=8000, size_gib=1.8,
                   fp8_kv=True)
    assert "--kv-cache-dtype fp8" in t
    assert "--kv-cache-dtype-skip-layers sliding_window" in t
    assert f"{VENV}/bin/vllm serve" in t


def test_the_picode_line_carries_it_too():
    t = next_steps(venv=VENV, model=MODEL, components=ALL, port=8000, size_gib=1.8,
                   fp8_kv=True)
    assert "glq-code" in t, "no picode step in the fp8 variant"


def test_the_summary_says_what_it_costs():
    """Someone reading only the summary should learn it is a trade, not a free win."""
    t = next_steps(venv=VENV, model=MODEL, components=ALL, port=8000, size_gib=1.8,
                   fp8_kv=True)
    assert "precision" in t.lower() or "context" in t.lower()


def test_nothing_about_it_appears_when_it_was_not_chosen():
    assert "--kv-cache-dtype" not in _text()


def test_the_e8_cache_is_never_printed_as_an_instruction():
    """It does not serve on vLLM 0.27.1, so no copyable command may suggest it."""
    for chosen in (True, False):
        t = next_steps(venv=VENV, model=MODEL, components=ALL, port=8000, size_gib=1.8,
                       fp8_kv=chosen)
        assert "GLQ_KV_QUANT" not in t


def test_quantize_component_drops_the_manual_pip_step():
    """With the component installed, telling the user to pip-install the extra they just
    got would read as if the install had not worked."""
    txt = _text(components=("core", "vllm", "chat", "quantize"))
    assert "glq-quantize" in txt
    assert "glq[quantize]" not in txt


def test_without_the_component_the_manual_pip_step_remains():
    txt = _text(components=("core", "vllm", "chat"))
    assert "glq[quantize]" in txt
    assert "glq-quantize" in txt


def test_gemma4_models_get_the_gemma4_parser_and_the_downloaded_template():
    """hermes matches SmolLM3-style <tool_call> markup and silently mangles gemma-4 tool
    calls; gemma-4 also needs the external template the picode component now downloads.
    The printed serve command must match the chosen model's family."""
    t = _text(model="xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw")
    assert "--tool-call-parser gemma4" in t
    assert "--reasoning-parser gemma4" in t
    assert "--chat-template" in t and "tool_chat_template_gemma4.jinja" in t
    assert "hermes" not in t


def test_non_gemma_models_keep_the_hermes_parser():
    t = _text()          # the default model is SmolLM3
    assert "--tool-call-parser hermes" in t
    assert "gemma4" not in t
