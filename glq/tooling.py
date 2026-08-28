"""Family-aware vLLM tool-calling serve args — one source for knowledge that drifted.

pi (and any tool-using client) needs the server started with `--enable-auto-tool-choice`
and a parser that matches the model's tool markup. Which parser — and for gemma-4, which
*template*, since its bundled chat template is plain chat and the tool template lives in
vLLM's repo examples — used to be duplicated between the installer's printed commands and
the bench harness, and drifted once already: hermes was printed for every model, which
matches SmolLM3/Qwen-style ``<tool_call>`` markup and silently mangles gemma-4's into
tool calls that never parse. That is the worst failure mode, because it reads as a bad
model rather than a bad flag.

Stdlib only: the installer's core profile has no huggingface_hub and no requests.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

GEMMA4_TOOL_TEMPLATE = "tool_chat_template_gemma4.jinja"
#: The version pin matters: templates track vLLM's parser expectations, and this pairing
#: is what the README's tool-calling recipe was validated against.
GEMMA4_TOOL_TEMPLATE_URL = ("https://raw.githubusercontent.com/vllm-project/vllm/"
                            f"v0.20.2/examples/{GEMMA4_TOOL_TEMPLATE}")


def _default_templates_dir() -> Path:
    return Path(os.environ.get("GLQ_HOME", Path.home() / ".glq")) / "templates"


def tool_serve_args(model_id: str, templates_dir=None) -> list[str] | None:
    """vLLM serve args that make tool calling work for this model's family.

    None for an unknown family — the caller must refuse rather than guess, because a
    wrong parser fails silently at the worst layer.
    """
    name = model_id.lower()
    if "gemma-4" in name:
        tpl = Path(templates_dir or _default_templates_dir()) / GEMMA4_TOOL_TEMPLATE
        return ["--enable-auto-tool-choice",
                "--tool-call-parser", "gemma4",
                "--reasoning-parser", "gemma4",
                "--chat-template", str(tpl),
                # Without this the template never opens a thought section, but the
                # RL-trained model thinks anyway — measured live in a pi session:
                # <|thought|> markers and tool-call syntax leaking into prose, and a
                # "thoughtthoughtthought" repetition loop in the reasoning field. The
                # README's validated recipe always carried it. Compact JSON (no spaces)
                # keeps the printed shell command copy-pasteable without quoting.
                "--default-chat-template-kwargs", '{"enable_thinking":true}']
    if "smollm3" in name or "qwen" in name:
        # Both families emit hermes-style <tool_call> markup; SmolLM3+hermes is the
        # pairing the Terminal-Bench integration validated end to end.
        return ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    return None


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as resp:            # noqa: S310 - pinned https
        return resp.read()


def ensure_gemma4_template(templates_dir=None, fetch=_fetch) -> Path:
    """The cached gemma-4 tool template, fetching it if this install never got one.

    The installer's picode component downloads it, but glq-code must also work on
    installs that predate that or skipped the component. A failed fetch raises with the
    manual command — "it failed" without "do this instead" strands the user exactly
    where the missing-template ValueError from vLLM did.
    """
    tpl = Path(templates_dir or _default_templates_dir()) / GEMMA4_TOOL_TEMPLATE
    if tpl.exists():
        return tpl
    try:
        data = fetch(GEMMA4_TOOL_TEMPLATE_URL)
        tpl.parent.mkdir(parents=True, exist_ok=True)
        tpl.write_bytes(data)
    except OSError as exc:
        raise RuntimeError(
            f"could not fetch gemma-4's tool template ({exc}); get it yourself:\n"
            f"  curl --proto '=https' --tlsv1.2 -fsSL {GEMMA4_TOOL_TEMPLATE_URL} "
            f"-o {tpl}") from exc
    return tpl
