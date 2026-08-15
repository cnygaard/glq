"""Gradio chat UI for a GLQ checkpoint served by vLLM (`glq-chat`).

Lives in the package rather than under examples/ because the installer must point at a
command that exists on a pip-only machine: `examples/` is not in the wheel, so a user who
arrived via `curl … | bash` has no such directory. `examples/chat/app.py` is a shim.

GLQ models serve over vLLM's OpenAI-compatible API, so this is a thin client: no model is
loaded in this process, and switching checkpoints in the dropdown just changes the `model`
field on the request. Start the server first:

    vllm serve xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel --quantization glq

Then:

    pip install 'glq[chat]'
    glq-chat

The model list comes from `~/.glq/config.json` when the installer wrote one, otherwise from
whatever the server reports on /v1/models — so this works whether or not you used install.sh.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
GLQ_CONFIG = Path(os.environ.get("GLQ_HOME", Path.home() / ".glq")) / "config.json"


def _installed_config() -> dict:
    try:
        return json.loads(GLQ_CONFIG.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _served_models(client: OpenAI) -> list[str]:
    """Ask the server what it is actually serving.

    Preferred over the config file because it cannot go stale: if someone restarted vLLM
    with a different checkpoint, this reflects reality.
    """
    try:
        return [m.id for m in client.models.list().data]
    except Exception:                                            # noqa: BLE001
        return []


def make_responder(client):
    """Build the streaming reply generator the chat UI calls.

    Separate from `build_ui` (and free of any gradio import) for two reasons: it is the part
    with behaviour worth testing — see tests/test_chat_streaming.py — and a UI that blocks
    until the whole reply is ready looks hung, which on a reasoning model at a 32k budget
    means a minute of blank screen.

    Yields are **cumulative**, not deltas: Gradio 6 renders the newest yield, so emitting
    deltas would leave the user looking at the last token by itself.
    """
    def respond(message, history, model, temperature, max_tokens):
        messages = []
        for turn in history or []:
            # Gradio 'messages' format: {"role": ..., "content": ...}
            if isinstance(turn, dict):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": message})

        stream = client.chat.completions.create(
            model=model, messages=messages, stream=True,
            temperature=temperature, max_tokens=int(max_tokens))

        out = ""
        for chunk in stream:
            # vLLM emits role-only and finish-reason chunks whose delta.content is None.
            out += chunk.choices[0].delta.content or ""
            yield out

    return respond


def build_ui(base_url: str, models: list[str], api_key: str = "glq"):
    import gradio as gr                                   # noqa: PLC0415 - UI-only dep

    client = OpenAI(base_url=base_url, api_key=api_key)
    respond = make_responder(client)

    # elem_id on the pieces a browser test needs to find. Gradio's own class names are
    # generated and churn between releases, so a selector like `.svelte-1x2y3z` breaks on
    # the next bump and trains everyone to ignore the test. These ids are ours and stable.
    with gr.Blocks(title="GLQ chat") as demo:
        gr.Markdown(f"### GLQ chat\nServing endpoint: `{base_url}`")
        model = gr.Dropdown(choices=models, value=models[0] if models else None,
                            label="GLQ checkpoint", allow_custom_value=True,
                            elem_id="glq-model")
        temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="temperature",
                                elem_id="glq-temperature")
        max_tokens = gr.Slider(64, 8192, value=1024, step=64, label="max tokens",
                               elem_id="glq-max-tokens")
        # No `type=` argument: Gradio 6 removed it and made the messages format the only
        # one (it was required in 5.x). `respond` therefore always receives history as a
        # list of {"role", "content"} dicts — see the `gradio>=6` pin in pyproject.
        gr.ChatInterface(respond,
                         chatbot=gr.Chatbot(elem_id="glq-chatbot", height=420),
                         textbox=gr.Textbox(elem_id="glq-input",
                                            placeholder="Ask the GLQ model something…"),
                         additional_inputs=[model, temperature, max_tokens])
    return demo


def main(argv=None) -> int:
    cfg = _installed_config()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default=cfg.get("base_url", DEFAULT_BASE_URL))
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="public Gradio link")
    args = p.parse_args(argv)

    client = OpenAI(base_url=args.base_url, api_key="glq")
    models = _served_models(client) or cfg.get("available") or []
    if not models:
        print(f"warning: no models found at {args.base_url}. Is vLLM running?\n"
              f"  vllm serve <repo-id> --quantization glq")

    build_ui(args.base_url, models).launch(server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
