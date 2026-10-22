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
import importlib.util
import json
import os
import signal
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from glq import fit
from glq.supervisor import (DEFAULT_GPU_MEMORY_UTILIZATION,
                            DEFAULT_MAX_MODEL_LEN, DEFAULT_MAX_NUM_SEQS,
                            DEFAULT_READY_TIMEOUT,
                            VllmSupervisor)
from glq.tooling import recommended_sampling as _recommended_sampling

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
GLQ_CONFIG = Path(os.environ.get("GLQ_HOME", Path.home() / ".glq")) / "config.json"


def _installed_config() -> dict:
    try:
        return json.loads(GLQ_CONFIG.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def default_model(cfg: dict, command: str) -> str | None:
    """The model a command serves when --model is absent: the installer's per-command
    pick (`code_model`/`chat_model` — glq-code prefers Qwen, whose own chat template
    emits tool markup, glq-chat prefers gemma-4 for MoE decode speed; see
    installer.recommend.PREFERRED_FAMILIES), else the generic install-time choice.
    Absent keys mean an older config — behavior is then exactly the pre-split default."""
    return cfg.get(f"{command}_model") or cfg.get("model")


#: The `chat` extra. Both are imported lazily — `openai` costs ~1.0 s of import time and
#: gradio far more — so nothing here fails at module scope, and the rest of glq works
#: without either.
CHAT_DEPS = ("gradio", "openai")


def missing_chat_deps() -> list[str]:
    """Which of the `chat` extra's packages are not installed.

    Checked up front, because both imports used to sit *inside* the block that owns the
    running server: on a plain `pip install glq` the user waited out a multi-minute weight
    load and was then handed a bare ModuleNotFoundError, with the load thrown away on
    teardown. `find_spec` answers the question without paying the import.
    """
    return [name for name in CHAT_DEPS if importlib.util.find_spec(name) is None]


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


def max_tokens_ceiling(max_model_len: int) -> int:
    """The largest useful `max_tokens` for a server with this context window.

    `max_tokens` caps the *output*; the window has to hold prompt + history + output. A
    slider that runs to the full window therefore offers a value that fails as soon as the
    user types anything — which is what it did, once the context was capped at 8192 to keep
    gemma-4 servable. Half the window, floored so a small window still allows an answer.
    """
    return max(256, max_model_len // 2)


#: The slider defaults, per model family — see `tooling._FAMILY_SAMPLING` for the values and
#: which of them vLLM would have applied anyway. Sending all three keeps one visible,
#: consistent answer to "what am I sampling with"; the old defaults were the worst of both,
#: overriding temperature with 0.7 while leaving top_p and top_k to the checkpoint.
#:
#: This used to be a single dict of gemma-4's numbers applied to every model, so a Qwen
#: session sampled at top_k=64 where its card asks for 20. The sliders mean a user CAN fix
#: that by dragging, but a default nobody's card asked for is not a default.
recommended_sampling = _recommended_sampling

#: Kept as gemma-4's row for anything still importing the constant.
RECOMMENDED_SAMPLING = _recommended_sampling("gemma-4")


def completion_kwargs(*, model, messages, temperature, top_p, top_k, max_tokens):
    """Build the `chat.completions.create` call.

    `top_k` is not in the OpenAI schema. The client drops unknown keyword arguments silently,
    so passing it directly looks correct and samples with top_k disabled; vLLM accepts it as
    an extension, which `extra_body` is the supported way to reach.

    top_k <= 0 sends nothing at all rather than a literal 0, so the server falls back to the
    checkpoint's own generation_config instead of being pinned to a value no card asked for.
    """
    kwargs = {
        "model": model, "messages": messages, "stream": True,
        "temperature": float(temperature), "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }
    if int(top_k) > 0:
        kwargs["extra_body"] = {"top_k": int(top_k)}
    return kwargs


def show_model_picker(models: list[str]) -> bool:
    """Is there anything to pick between?

    `glq-chat` starts one server with one model, so the dropdown is normally a list of one —
    a control that costs a row of screen and can only be set to what it already is. It earns
    its place only when the server reports several models.
    """
    return len(models) > 1


def build_ui(base_url: str, models: list[str], api_key: str = "glq",
             max_model_len: int = DEFAULT_MAX_MODEL_LEN):
    # Imported here, not at module scope: `glq[chat]` is an extra, and `main()` must be able
    # to report a missing gradio itself rather than failing at import — which is also what
    # lets these paths be tested without it.
    import gradio as gr

    client = _openai_client(base_url, api_key)

    def respond(message, history, model, temperature, top_p, top_k, max_tokens):
        messages = []
        for turn in history or []:
            # Gradio 'messages' format: {"role": ..., "content": ...}
            if isinstance(turn, dict):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": message})

        stream = client.chat.completions.create(
            **completion_kwargs(model=model, messages=messages, temperature=temperature,
                                top_p=top_p, top_k=top_k, max_tokens=max_tokens))

        out = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            out += delta
            yield out

    served = models[0] if models else None
    with gr.Blocks(title="GLQ chat") as demo:
        # One line, not three: the model is what the user cares about and the endpoint is
        # only interesting when something is wrong.
        gr.Markdown(f"**GLQ chat** — {served or 'no model'} · `{base_url}`")

        # Created inside the accordion, because gradio renders `additional_inputs` wherever
        # they are constructed — building them in the open layout is exactly what stacked
        # three controls above every conversation.
        with gr.Accordion("Settings", open=False):
            model = gr.Dropdown(choices=models, value=served, label="GLQ checkpoint",
                                allow_custom_value=True,
                                visible=show_model_picker(models))
            # The served model's own card, not one family's numbers for everybody.
            sampling = recommended_sampling(served)
            temperature = gr.Slider(0.0, 2.0, value=sampling["temperature"],
                                    step=0.05, label="temperature")
            top_p = gr.Slider(0.0, 1.0, value=sampling["top_p"], step=0.01,
                              label="top_p")
            # 0 = off, so a model whose card asks for no top_k (SmolLM3) can be served from
            # the same UI by dragging this to zero rather than editing a flag.
            top_k = gr.Slider(0, 200, value=sampling["top_k"], step=1,
                              label="top_k (0 = off)")
            ceiling = max_tokens_ceiling(max_model_len)
            max_tokens = gr.Slider(64, ceiling, value=min(1024, ceiling), step=64,
                                   label="max tokens")

        # No `type=` argument: Gradio 6 removed it and made the messages format the only
        # one (it was required in 5.x). `respond` therefore always receives history as a
        # list of {"role", "content"} dicts — see the `gradio>=6` pin in pyproject.
        gr.ChatInterface(respond,
                         additional_inputs=[model, temperature, top_p, top_k, max_tokens])
    return demo



def positive_seconds(text: str) -> float:
    """An argparse type for a wait that must actually wait.

    Zero or negative would mean "give up before asking", which surfaces as an instant and
    inexplicable startup failure rather than as the configuration error it is.
    """
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {value:g}")
    return value


def writable_workdir() -> Path:
    """A directory gradio can write into, since it puts `.gradio` in the *current* one.

    Measured on ubuntu:26.04 with the repo mounted read-only: launching from there gives
    "Could not create share link. [Errno 13] Permission denied: '.gradio'" — so where the
    user happened to be standing decided whether the public link worked.

    `~/.glq` already holds config.json and vllm.log, so it is where this belongs. Falls back
    to the temp directory rather than raising: losing the share link is a nuisance, refusing
    to start the chat at all is worse.
    """
    home = Path(os.environ.get("GLQ_HOME", Path.home() / ".glq"))
    try:
        home.mkdir(parents=True, exist_ok=True)
        if os.access(home, os.W_OK):
            return home
    except OSError:
        pass
    return Path(tempfile.gettempdir())


def _server_port(base_url: str, default: int = 8000) -> int:
    """The port vLLM should listen on, taken from the URL the client will call.

    Deriving it rather than accepting a second flag means the two cannot disagree — a
    `--port 8001` that leaves the client pointed at 8000 is a confusing way to fail.
    """
    return urlparse(base_url).port or default


def _ram_bytes():
    """Total host RAM, or None if it cannot be read.

    The supervisor needs this because offloaded expert weights are **pinned**: pinned pages
    cannot be swapped out, so planning more than the host can hold does not run slowly, it
    fails to allocate. Without it `_PINNED_RAM_FRACTION` cannot bind — measured on a 121 GiB
    box, `glq-code` planned `--cpu-offload-gb 42` for Flash-Next, which beside the 23.84 GiB
    PLE table is 66 GiB of pinned memory and impossible on the 32-64 GiB a consumer has.

    Same never-block contract as `_vram_bytes`: a failure means the caps simply do not bind,
    which is the behaviour that existed before this was wired in.

    Shared by glq-chat and glq-code, for the same reason `sizing_weights_bytes` is.
    """
    try:
        from glq.installer.hardware import ram_bytes
        return ram_bytes()
    except Exception:                                       # noqa: BLE001 - never block
        return None


def _vram_bytes():
    """Total VRAM, or None where nvidia-smi cannot answer."""
    try:
        from glq.installer.hardware import vram_bytes
        return vram_bytes()
    except Exception:                                       # noqa: BLE001 - no driver, no GPU
        return None


def _swap_bytes():
    """Total swap, or None if it cannot be read. **0 is an answer, not an unknown.**

    Read for the pre-flight fit check rather than for planning. Pinned expert pages cannot be
    swapped out, but the allocations *around* them can, so a box with swap survives a pinned
    budget that a swapless box does not — and both measured data points behind that budget are
    swapless, which is why the distinction has to be visible. See `glq.fit`.
    """
    try:
        from glq.installer.hardware import swap_bytes
        return swap_bytes()
    except Exception:                                       # noqa: BLE001 - never block
        return None


def _gpu_name():
    """The card's marketing name, or None. Cosmetic — it only labels a row in a message.

    Worth having there: "this card 31.86 GiB" tells a user less than "RTX PRO 4500 Blackwell
    31.86 GiB", which they can match against what they think they bought.
    """
    try:
        from glq.installer.hardware import gpu_name
        return gpu_name()
    except Exception:                                       # noqa: BLE001 - never block
        return None


def _available_ram_bytes():
    """RAM a new process can have without swapping, or None. Advisory only.

    Reported in a refusal, never gated on: a refusal must be a statement about the machine,
    so the gate uses MemTotal. Refusing because a browser happens to be open would be wrong.
    """
    try:
        from glq.installer.hardware import available_ram_bytes
        return available_ram_bytes()
    except Exception:                                       # noqa: BLE001 - never block
        return None


def _sizing_fetch(model: str):
    """Where the sizing lookups should read from: a local checkpoint directory, or the Hub.

    `--model` takes either a repo id or a path, and the three lookups below were HTTP-only.
    A path interpolates into the URL, so it could never resolve — and because all three
    swallow failure, the window, PLE offload, expert offload and the expert arena collapsed
    together in silence. Measured on an L4: 16384 instead of 131072 tokens, and 6.4 instead
    of 15.2 tok/s with no message explaining either.

    A token does not fix that half; no authorization makes the URL valid. What does is
    noticing the bytes are already on disk — `config.json` sits beside the weights in any
    cached snapshot, so for a model that is present this never needed the network at all.
    """
    try:
        if model and Path(model).is_dir():
            from glq.installer.discovery import local_fetcher
            return local_fetcher(model)
    except OSError:                                         # unreadable path, or not a path
        pass
    # Read the attribute at call time, not as a default argument: every function here binds
    # `fetch=_fetch_json` at import, which is why patching the module attribute cannot reach
    # them. (The same default-argument trap hides `VllmSupervisor`'s `spawn` seam.)
    from glq.installer import discovery
    return discovery._fetch_json


def _model_max_len(repo_id: str):
    """The model's declared context maximum, or None — sizes the auto window's clamp.

    Same never-block contract as _checkpoint_bytes: one small lookup whose failure must not
    stop the chat from starting at the conservative floor.
    """
    try:
        from glq.installer.discovery import model_max_len
        return model_max_len(repo_id, fetch=_sizing_fetch(repo_id)) or None
    except Exception:                                       # noqa: BLE001 - offline, 404, …
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
        return repo_size_bytes(repo_id, fetch=_sizing_fetch(repo_id)) or None
    except Exception:                                       # noqa: BLE001 - offline, 404, …
        return None


def checkpoint_offload_bytes(repo_id: str) -> tuple[int, int, int]:
    """(ple, expert, nontext) non-resident bytes for this checkpoint, 0s if unknown.

    The first two can live in HOST memory; the third (MTP head, vision tower) is not loaded
    at all by a text-only serve, so it costs neither VRAM nor pinned RAM.

    Same never-block contract as `_checkpoint_bytes`: one small config.json read, and any
    failure means "assume every weight is resident" — today's behaviour — rather than
    stopping the chat or, worse, planning an offload the checkpoint cannot support.

    Shared by glq-chat and glq-code so the two cannot drift apart on it, which is the same
    reason `sizing_weights_bytes` is shared.
    """
    if not repo_id:
        return 0, 0, 0
    try:
        from glq.installer.discovery import offload_bytes
        return offload_bytes(_sizing_fetch(repo_id)(
            f"https://huggingface.co/{repo_id}/resolve/main/config.json"))
    except Exception:                                       # noqa: BLE001 - offline, 404, …
        return 0, 0, 0


def checkpoint_mtp(repo_id: str) -> tuple[int, bool, int]:
    """``(mtp_layers, head_is_quantized, decoder_layers)`` for this checkpoint, zeros if
    unknown — what the planner needs to budget for speculative decoding.

    `mtp_layers` drives whether `--mtp` is offerable at all. **Test the VALUE, not the key:**
    `mtp`, `mtp_num_hidden_layers` and `mtp_use_dedicated_embeddings` are all present on
    Flash-Next builds that have no usable head, with `null` values (measured on
    `...-3bpw-ple4`, 2026-10-13).

    `head_is_quantized` is whether `layer_bpw` carries MTP entries — 1,545 of them on
    `...-ple4-3mtp`, 0 on `...-ple4`, whose head is bf16 and therefore costs the whole of
    `nontext_bytes` (4.86 GiB) to load. Matched on dotted SEGMENTS, following the warning on
    `quantize_model._is_nontext`: `model.layers.0.mtpool.weight` contains "mtp" and is an
    ordinary text weight.

    `decoder_layers` is what makes the MTP block's expert bytes derivable rather than guessed:
    `_offloadable_bytes` drops that layer from every total, so the planner reconstructs it as
    one layer's share of `expert_offload_bytes`.

    Same never-block contract as `checkpoint_offload_bytes`, and shared by glq-chat and
    glq-code so the two cannot drift apart on it.
    """
    if not repo_id:
        return 0, False, 0
    try:
        cfg = _sizing_fetch(repo_id)(
            f"https://huggingface.co/{repo_id}/resolve/main/config.json")
        text = cfg.get("text_config") or {}

        def _get(key):
            return cfg.get(key) or text.get(key)

        mtp = _get("mtp_num_hidden_layers") or _get("num_nextn_predict_layers")
        if not mtp:
            block = _get("mtp")
            if isinstance(block, dict):
                mtp = block.get("num_hidden_layers") or 1
        layers = int(_get("num_hidden_layers") or 0)
        lb = (cfg.get("quantization_config") or {}).get("layer_bpw") or {}
        quantized = any(k.split(".")[0] == "mtp" for k in lb)
        return int(mtp or 0), quantized, layers
    except Exception:                                       # noqa: BLE001 - offline, 404, …
        return 0, False, 0


def resolve_mtp_tokens(requested, mtp_layers: int, head_is_quantized: bool):
    """Draft tokens to speculate with, or a message explaining the refusal.

    Returns ``(tokens, error)`` — exactly one of them meaningful.

    **The default is OFF.** Speculation is opt-in via `--mtp N`, for two reasons measured
    rather than assumed:

    * It changes output. Three speculative arms on an L4 produced three distinct digests
      where every non-speculative arm agreed, so it is not lossless at temperature 0 and a
      default-on would silently change what users get.
    * It loads the draft head, which withholds the `nontext_bytes` subtraction and deepens
      the expert offload — and host-memory accounting for this checkpoint is not yet
      trustworthy: a glq-code session on a 120 GiB box held **102.8 GiB** of shared host
      memory against a planned 65.3 GiB and was OOM-killed by the kernel. Until that is
      understood, speculation does not get to raise the offload budget unasked.

    Worth having behind the flag: with the expert arena it measured 19.692 tok/s on an L4
    (3.060x a 6.436 baseline) and 37.957 on a 32 GiB Blackwell against 29.523 without. Which
    N wins is hardware-dependent — 1 beat 3 on the L4, 3 beat 1 on the Blackwell — so neither
    is a safe constant.

    `head_is_quantized` is still reported by `checkpoint_mtp` and still worth knowing (a bf16
    head costs the whole of `nontext_bytes` to load, two window tiers on a 24 GB card), but it
    no longer selects a default now that there is none.

    An explicit request on a checkpoint with no head is an error rather than a silent 0,
    because the alternative is vLLM refusing the `--speculative-config` minutes into a weight
    load — and because the user asked for something this checkpoint cannot do.
    """
    if requested is None:
        return 0, None
    n = int(requested)
    if n < 0:
        return 0, f"--mtp must be 0 or more, not {n}"
    if n and not mtp_layers:
        return 0, ("this checkpoint declares no MTP draft head, so --mtp cannot speculate "
                   "with it. Re-run without --mtp, or pick a checkpoint whose name carries "
                   "an mtp suffix.")
    return n, None


def sizing_weights_bytes(args, device=None):
    """The checkpoint size the supervisor should size against, or None.

    `--gpu-memory-utilization` used to suppress the lookup outright: on a GPU that flag IS
    the answer weights_bytes would have produced, so the HTTP call was pure cost. On CPU it
    is a flag that does nothing — the supervisor reports it as ignored — while the KV pool
    still has to be sized against the weights, because there they share one pool of RAM
    with the runtime and the page cache. Suppressing the lookup there restores exactly the
    overcommit that hangs a swapless machine, through a flag that has no effect.

    Shared by glq-chat and glq-code so the two cannot drift apart on it.
    """
    if not getattr(args, "model", None):
        return None
    if args.gpu_memory_utilization is not None:
        from glq.supervisor import detect_device
        if (device or detect_device()) != "cpu":
            return None
    return _checkpoint_bytes(args.model)


def _display_available() -> bool:
    """Is there a desktop to open a browser on? Over SSH there is not, and `inbrowser=True`
    would either do nothing or launch a text browser in the terminal running the server."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def main(argv=None) -> int:
    cfg = _installed_config()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=default_model(cfg, "chat"),
                   help="checkpoint to serve (default: the installer's chat pick, "
                        "then its generic one)")
    p.add_argument("--base-url", default=cfg.get("base_url", DEFAULT_BASE_URL))
    p.add_argument("--port", type=int, default=7860, help="port for the chat UI itself")
    p.add_argument("--gpu-memory-utilization", type=float, default=None,
                   help=f"fraction of VRAM vLLM may reserve "
                        f"(default {DEFAULT_GPU_MEMORY_UTILIZATION}; the rest stays free "
                        f"for whatever else uses the GPU)")
    # glq-chat deliberately does NOT trade decode speed for a longer auto window the way
    # glq-code does (`window_offload_extra_gib` stays 0): a conversation that never passes a
    # few thousand tokens would pay PCIe on every token for context it will not fill, and it
    # prices its window at 8 concurrent requests, so a tier costs eight times what it costs a
    # single-stream agent. A window the user PINS with --max-model-len still drives the budget.
    # This flag is how to ask for more anyway; `0` serves resident.
    p.add_argument("--cpu-offload-gb", type=int, default=None, metavar="N",
                   help="GiB of MoE experts to keep in host RAM, overriding the plan "
                        "(0 = none; raising it buys context and costs PCIe decode time)")
    p.add_argument("--max-model-len", type=int, default=None,
                   help="context window to serve (default: sized from VRAM headroom in "
                        "tiers 8192-65536, clamped to the model's own maximum; pass a "
                        "number to pin it — vLLM would otherwise take the model's "
                        "declared maximum, which for gemma-4 is 262144 and needs GiB of "
                        "KV cache for a single request)")
    p.add_argument("--mtp", type=int, default=None, metavar="N",
                   help="speculate N draft tokens per step with the checkpoint's MTP draft "
                        "head (default OFF). Which N wins is hardware-dependent: 1 beat 3 on "
                        "an L4, 3 beat 1 on a 32 GiB Blackwell. Output is NOT byte-identical "
                        "with speculation on, even at temperature 0.")
    p.add_argument("--max-num-seqs", type=int, default=None,
                   help=f"concurrent sequences vLLM plans for (default "
                        f"{DEFAULT_MAX_NUM_SEQS} on GPU, 4 on the CPU backend; vLLM's own "
                        f"default is 1024 — a batch-server number that hybrid-GDN models "
                        f"cannot even start under, since every sequence reserves a Mamba "
                        f"cache block)")
    p.add_argument("--fp8-kv-cache", dest="fp8_kv", action="store_true",
                   default=cfg.get("fp8_kv", False),
                   help="vLLM's fp8 KV cache: about twice the context per GiB, at lower "
                        "attention precision. Default from the installer's choice")
    p.add_argument("--no-fp8-kv-cache", dest="fp8_kv", action="store_false",
                   help="keep the KV cache at full precision")
    # Weight load plus CUDA-graph capture, and both scale with the model and the disk. The
    # default suits one checkpoint on an idle card; a large MoE from cold storage, or several
    # servers starting at once, legitimately need longer. Measured in the distro matrix: at
    # 5-way concurrency, 12 legs of 44 failed purely because startup outran this window.
    p.add_argument("--ready-timeout", type=positive_seconds, default=DEFAULT_READY_TIMEOUT,
                   metavar="SECONDS",
                   help=f"give up after this long without progress — no log output and no "
                        f"weight-download movement; a slow but moving download never trips "
                        f"it (default {DEFAULT_READY_TIMEOUT:g})")
    p.add_argument("--no-serve", dest="serve", action="store_false",
                   help="do not start vLLM; attach to a server you started yourself")
    p.add_argument("--no-browser", dest="browser", action="store_false",
                   help="print the URL instead of opening a browser")
    p.add_argument("--force", action="store_true",
                   help="serve a checkpoint this machine was measured too small for "
                        "(vLLM will most likely fail during the weight load)")
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

    # Before the supervisor, deliberately: starting vLLM first would spend minutes loading
    # weights only to discard them when the UI cannot be built.
    missing = missing_chat_deps()
    if missing:
        print(f"error: the chat UI needs {', '.join(missing)}, which "
              f"{'is' if len(missing) == 1 else 'are'} not installed.\n"
              f"       pip install 'glq[chat]'\n"
              f"       (glq itself does not need {'it' if len(missing) == 1 else 'them'}; "
              f"they ship as an extra so a serving-only install stays small.)")
        return 3

    # Declared host-offloadable bytes: the supervisor sizes the pool and window from
    # what actually lands in VRAM, and turns PLE offload on for a checkpoint that
    # cannot serve without it. (0, 0) for everything that declares nothing.
    _offload = checkpoint_offload_bytes(args.model)
    # With the gradio guard above, and for the same reason: a `--speculative-config` this
    # checkpoint cannot honour makes vLLM refuse minutes into a weight load.
    _mtp_layers, _mtp_quantized, _decoder_layers = checkpoint_mtp(args.model)
    _mtp, _mtp_err = resolve_mtp_tokens(args.mtp, _mtp_layers, _mtp_quantized)
    if _mtp_err:
        print(f"error: {_mtp_err}", file=sys.stderr)
        return 2
    supervisor = VllmSupervisor(
        model=args.model,
        port=_server_port(args.base_url),
        base_url=args.base_url,
        gpu_memory_utilization=args.gpu_memory_utilization,
        serve=args.serve,
        verbose=args.verbose,
        max_model_len=args.max_model_len,
        # The declared-max lookup only runs in auto mode — an explicit flag costs no
        # network round trip, same principle as the pool plan below.
        model_max_len=(None if args.max_model_len is not None or not args.model
                       else _model_max_len(args.model)),
        max_num_seqs=args.max_num_seqs,
        fp8_kv=args.fp8_kv,
        timeout=args.ready_timeout,
        # Size the KV pool for this checkpoint — on CPU that is what keeps the pool,
        # the weights and the page cache inside one RAM budget.
        weights_bytes=sizing_weights_bytes(args),
        vram_bytes=None if args.gpu_memory_utilization is not None else _vram_bytes(),
        # Host RAM, so the pinnable-expert cap can bind: offloaded weights are pinned and
        # cannot be swapped, so an offload larger than the host fails to allocate.
        ram_bytes=_ram_bytes(),
        # Declared host-offloadable bytes, so the supervisor sizes the pool and the window
        # from what actually lands in VRAM and turns PLE offload on for a checkpoint that
        # cannot serve without it. 0/0 for everything that declares nothing, which is every
        # checkpoint today except Qwen3.8-Flash-Next.
        ple_offload_bytes=_offload[0], expert_offload_bytes=_offload[1],
        nontext_bytes=_offload[2],
        # Speculation changes the budget three ways, all handled in the supervisor: the MTP
        # head is loaded (so `nontext_bytes` stops being subtracted), its own MoE layer
        # becomes offloadable, and every decode step widens to `1 + mtp` tokens per stream.
        mtp_tokens=_mtp, mtp_layers=_mtp_layers, decoder_layers=_decoder_layers,
        expert_offload_gib=args.cpu_offload_gb,
        # One stream, because this command serves one person: it starts vLLM, opens a chat
        # UI, and stops the server on the way out. Without this it inherited
        # `_WINDOW_CONCURRENCY = 8` and had to afford its window eight times over, which is a
        # server's requirement. Measured on a 22.49 GiB card with Flash-Next: 8192 at
        # concurrency 8 against 65536 at 1, same offload, slightly larger arena.
        #
        # `max_num_seqs` deliberately stays at its default rather than following suit. Beyond
        # the pool vLLM queues rather than failing, so a second browser tab waits instead of
        # being refused — the over-subscription `_WINDOW_CONCURRENCY` documents.
        window_concurrency=1,
    )

    # The last guard, and the most expensive failure it prevents: a checkpoint larger than
    # this machine used to reach vLLM and die with a CUDA out-of-memory several minutes into
    # the weight load, after the download, with nothing naming the cause.
    #
    # Asked of the supervisor, not of the arguments, because by here the offload, the window
    # and the arena are planned and it is THAT footprint which either fits. Construction is
    # pure planning, so there is nothing to undo when the answer is no.
    #
    # The same function glq-code calls, deliberately: every other sizing helper is already
    # shared between the two commands, and a gate that lived in only one of them would mean
    # the fix applied to whichever command the author happened to be using. `command=` is
    # threaded through so the shared renderer does not tell a glq-chat user to re-run
    # glq-code.
    if args.serve and not args.force:
        _verdict = fit.check_fit(supervisor, ram_bytes=_ram_bytes(),
                                 swap_bytes=_swap_bytes(),
                                 available_ram_bytes=_available_ram_bytes(),
                                 floor=DEFAULT_MAX_MODEL_LEN)
        if _verdict.fits is False:
            print(*fit.refusal_lines(_verdict, command="glq-chat",
                                     gpu_name=_gpu_name()),
                  sep="\n", file=sys.stderr)
            return 2

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
        # gradio writes `.gradio` into the current directory; stand somewhere writable
        # first, or the share tunnel dies wherever the user happened to launch from.
        os.chdir(writable_workdir())
        build_ui(args.base_url, models,
                 max_model_len=supervisor.max_model_len).launch(
            server_port=args.port, share=args.share,
            inbrowser=args.browser and _display_available())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
