"""Gradio chat UI for a GLQ checkpoint served by vLLM (`glq-chat`).

Lives in the package rather than under examples/ because the installer must point at a
command that exists on a pip-only machine: `examples/` is not in the wheel, so a user who
arrived via `curl … | bash` has no such directory. `examples/chat/app.py` is a shim.

GLQ models serve over vLLM's OpenAI-compatible API, so no model is loaded in this process —
but this command *owns* the server rather than assuming one:

    pip install 'glq[chat]'
    glq-chat

starts vLLM if nothing is answering, waits for it, opens the chat, and stops the server
again on the way out. That last part matters on a desktop: vLLM has no idle unload, so a
server left running holds its share of the card until it is killed.

Pass `--no-serve` to go back to being a pure client against a server you started yourself:

    vllm serve xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel --quantization glq

The model comes from `~/.glq/config.json` when the installer wrote one, and the dropdown
from whatever the server reports on /v1/models — so this works whether or not you used
install.sh.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from urllib.parse import urlparse

from glq.supervisor import (DEFAULT_GPU_MEMORY_UTILIZATION,
                            DEFAULT_MAX_MODEL_LEN, VllmSupervisor)

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
GLQ_CONFIG = Path(os.environ.get("GLQ_HOME", Path.home() / ".glq")) / "config.json"


def _installed_config() -> dict:
    try:
        return json.loads(GLQ_CONFIG.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _openai_client(base_url: str, api_key: str = "glq"):
    """Imported on use, not at module scope.

    `import openai` costs ~1.0 s of the 2.2 s this command spends before it can print
    anything, and the very first symptom the user reports is that nothing happens when they
    start it. Nothing here needs the client until the server is already up.
    """
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key)


def _served_models(client) -> list[str]:
    """Ask the server what it is actually serving.

    Preferred over the config file because it cannot go stale: if someone restarted vLLM
    with a different checkpoint, this reflects reality.
    """
    try:
        return [m.id for m in client.models.list().data]
    except Exception:                                            # noqa: BLE001
        return []


def build_ui(base_url: str, models: list[str], api_key: str = "glq"):
    # Imported here, not at module scope: `glq[chat]` is an extra, and `main()` must be able
    # to report a missing gradio itself rather than failing at import — which is also what
    # lets these paths be tested without it.
    import gradio as gr

    client = _openai_client(base_url, api_key)

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
            delta = chunk.choices[0].delta.content or ""
            out += delta
            yield out

    with gr.Blocks(title="GLQ chat") as demo:
        gr.Markdown(f"### GLQ chat\nServing endpoint: `{base_url}`")
        model = gr.Dropdown(choices=models, value=models[0] if models else None,
                            label="GLQ checkpoint", allow_custom_value=True)
        temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="temperature")
        max_tokens = gr.Slider(64, 8192, value=1024, step=64, label="max tokens")
        # No `type=` argument: Gradio 6 removed it and made the messages format the only
        # one (it was required in 5.x). `respond` therefore always receives history as a
        # list of {"role", "content"} dicts — see the `gradio>=6` pin in pyproject.
        gr.ChatInterface(respond,
                         additional_inputs=[model, temperature, max_tokens])
    return demo


def _server_port(base_url: str, default: int = 8000) -> int:
    """The port vLLM should listen on, taken from the URL the client will call.

    Deriving it rather than accepting a second flag means the two cannot disagree — a
    `--port 8001` that leaves the client pointed at 8000 is a confusing way to fail.
    """
    return urlparse(base_url).port or default


def _vram_bytes():
    """Total VRAM, or None where nvidia-smi cannot answer."""
    try:
        from glq.installer.hardware import vram_bytes
        return vram_bytes()
    except Exception:                                       # noqa: BLE001 - no driver, no GPU
        return None


def _checkpoint_bytes(repo_id: str):
    """How much VRAM the weights will want, or None if we cannot find out.

    Reuses the installer's sizing (it sums the repo's `.safetensors` entries, because the
    API's `usedStorage` counts every revision) so the chat and the picker agree on how big a
    checkpoint is. One small HTTP call, right before a download of GiB — but it must never be
    the thing that stops the chat starting, hence the broad except.
    """
    try:
        from glq.installer.discovery import repo_size_bytes
        return repo_size_bytes(repo_id) or None
    except Exception:                                       # noqa: BLE001 - offline, 404, …
        return None


def _display_available() -> bool:
    """Is there a desktop to open a browser on? Over SSH there is not, and `inbrowser=True`
    would either do nothing or launch a text browser in the terminal running the server."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def main(argv=None) -> int:
    cfg = _installed_config()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=cfg.get("model"),
                   help="checkpoint to serve (default: the one install.sh configured)")
    p.add_argument("--base-url", default=cfg.get("base_url", DEFAULT_BASE_URL))
    p.add_argument("--port", type=int, default=7860, help="port for the chat UI itself")
    p.add_argument("--gpu-memory-utilization", type=float, default=None,
                   help=f"fraction of VRAM vLLM may reserve "
                        f"(default {DEFAULT_GPU_MEMORY_UTILIZATION}; the rest stays free "
                        f"for whatever else uses the GPU)")
    p.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN,
                   help=f"context window to serve (default {DEFAULT_MAX_MODEL_LEN}; vLLM "
                        f"would otherwise take the model's own maximum, which for gemma-4 "
                        f"is 262144 and needs GiB of KV cache for a single request)")
    p.add_argument("--fp8-kv-cache", dest="fp8_kv", action="store_true",
                   default=cfg.get("fp8_kv", False),
                   help="vLLM's fp8 KV cache: about twice the context per GiB, at lower "
                        "attention precision. Default from the installer's choice")
    p.add_argument("--no-fp8-kv-cache", dest="fp8_kv", action="store_false",
                   help="keep the KV cache at full precision")
    p.add_argument("--no-serve", dest="serve", action="store_false",
                   help="do not start vLLM; attach to a server you started yourself")
    p.add_argument("--no-browser", dest="browser", action="store_false",
                   help="print the URL instead of opening a browser")
    p.add_argument("--verbose", action="store_true",
                   help="stream vLLM's own output instead of a one-line summary")
    # On by default: a *.gradio.live tunnel is what makes the chat reachable from a phone or
    # another machine with no port forwarding and no firewall rules. It is also a public,
    # unauthenticated URL to a GPU for as long as the chat runs, which is why it is announced
    # rather than silent, and why --no-share exists.
    p.add_argument("--share", dest="share", action="store_true", default=True,
                   help="public https://….gradio.live link (default)")
    p.add_argument("--no-share", dest="share", action="store_false",
                   help="keep the chat on this machine only")
    args = p.parse_args(argv)

    if args.serve and not args.model:
        print("error: no model to serve. Pass --model <repo-id>, or run the installer,\n"
              "       or use --no-serve to attach to a server you started yourself.\n"
              "       Published checkpoints: https://huggingface.co/xv0y5ncu")
        return 2

    supervisor = VllmSupervisor(
        model=args.model,
        port=_server_port(args.base_url),
        base_url=args.base_url,
        gpu_memory_utilization=args.gpu_memory_utilization,
        serve=args.serve,
        verbose=args.verbose,
        max_model_len=args.max_model_len,
        fp8_kv=args.fp8_kv,
        # Size the KV pool for this checkpoint. Skipped entirely when the user named a
        # fraction themselves, so `--gpu-memory-utilization` costs no network round trip.
        weights_bytes=(None if args.gpu_memory_utilization is not None or not args.model
                       else _checkpoint_bytes(args.model)),
        vram_bytes=None if args.gpu_memory_utilization is not None else _vram_bytes(),
    )

    # Ctrl-C already unwinds through the context manager below, but `kill` and a closed
    # terminal do not: the default disposition for SIGTERM/SIGHUP ends the process without
    # running any cleanup, which would leave vLLM holding the card with nothing driving it.
    # Turning them into SystemExit routes them through the same teardown.
    def _exit_on(signum, _frame):
        raise SystemExit(128 + signum)

    for _sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(_sig, _exit_on)
        except (ValueError, OSError, AttributeError):     # not the main thread, or not POSIX
            pass

    # The context manager is the VRAM-release guarantee: whatever the UI does, including
    # falling over, vLLM is stopped on the way out.
    with supervisor:
        client = _openai_client(args.base_url)
        models = _served_models(client) or cfg.get("available") or []
        if args.model and args.model not in models:
            models.insert(0, args.model)
        if not models:
            print(f"warning: no models found at {args.base_url}. Is vLLM running?\n"
                  f"  vllm serve <repo-id> --quantization glq")

        if args.share:
            print("  publishing a public gradio link — anyone who has it can use this GPU; "
                  "--no-share keeps the chat on this machine", file=sys.stderr, flush=True)

        # gradio announces its own local and public URLs, so we do not repeat them. What was
        # actually broken is that they never arrived: measured on a box, with stdout
        # redirected those lines block-buffer and do not surface until the process exits,
        # nine minutes later. Line-buffering stdout is the fix — every line gradio writes,
        # the share link included, appears as it happens.
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):        # not a real stream (tests, pipes)
            pass
        build_ui(args.base_url, models).launch(
            server_port=args.port, share=args.share,
            inbrowser=args.browser and _display_available())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
