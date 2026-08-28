"""Second stage of the GLQ installer — everything after the venv exists.

`install.sh` creates the venv and `pip install glq`s into it, then hands over here. Keeping
the logic in Python rather than bash is what makes it testable: see `tests/test_installer_*`.

Run directly with `glq-setup` to re-run the wizard against an existing install.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from glq import kv_compression as kv_env

from . import configure, discovery, hardware, prompt, verify
from .recommend import rank

GLQ_HOME = Path(os.environ.get("GLQ_HOME", Path.home() / ".glq"))
DEFAULT_PORT = 8000


def _venv_bin(venv: Path, exe: str) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin") / exe


class Runner:
    """Runs commands, or prints them under --dry-run."""

    def __init__(self, dry_run: bool):
        self.dry_run = dry_run

    def __call__(self, cmd, **kw) -> int:
        printable = " ".join(shlex.quote(str(c)) for c in cmd)
        if self.dry_run:
            print(f"  [dry-run] {printable}")
            return 0
        print(f"  $ {printable}")
        return subprocess.call([str(c) for c in cmd], **kw)


#: Both ends of this range exist because of gemma-4, and vLLM declares only
#: `transformers>=5.5.3`, so without it pip resolves the newest and every gemma-4 checkpoint
#: dies before a weight is loaded.
#:
#:   floor   5.13.1 — earlier transformers has no gemma-4 at all.
#:   ceiling <5.15  — 5.15.0 moved gemma-4 to a per-layer config and made the global
#:                    `config.head_dim` raise AmbiguousGlobalPerLayerAttributeError, which
#:                    vLLM 0.27.1 reads while building its ModelConfig. Bisected on an L4:
#:                    5.15.0 fails, 5.14.1 / 5.13.1 / 5.12.1 / 5.11.0 / 5.10.4 all give
#:                    head_size=512 and load.
#:
#: Not a GLQ bug — stock bf16 google/gemma-4-E2B-it fails the same way with no GLQ in the
#: process — and not fixable from the config: forcing the documented
#: `allow_global_per_layer_attribute_access` yields head_size=256 for a model whose layers
#: are 256 *and* 512, and the weight loader dies on a size assert. Lift the ceiling when
#: vLLM builds gemma-4 from per-layer configs.
GEMMA4_TRANSFORMERS = "transformers>=5.13.1,<5.15"

# Template constants live in glq.tooling — the single source both this installer and
# glq-code read, so the family knowledge cannot drift again.
from glq.tooling import (GEMMA4_TOOL_TEMPLATE,  # noqa: E402
                         GEMMA4_TOOL_TEMPLATE_URL, tool_serve_args)


def _install_python_extras(run: Runner, venv: Path, components) -> None:
    pip = _venv_bin(venv, "pip")
    wanted = []
    if "vllm" in components:
        wanted += ["vllm", GEMMA4_TRANSFORMERS]
    if "chat" in components:
        wanted += ["gradio", "openai"]
    if wanted:
        print(f"\n== installing: {', '.join(wanted)}")
        run([pip, "install", "--upgrade", *wanted])
    if "quantize" in components:
        # Deliberately a separate command with NO --upgrade: `glq[quantize]` names glq
        # itself, and --upgrade would replace a --glq-source dev install with the PyPI
        # release. Plain install leaves an already-satisfied glq alone and resolves only
        # the extra's missing deps (pyproject stays the single source of truth for them).
        print("\n== installing: glq[quantize]")
        run([pip, "install", "glq[quantize]"])


def _install_open_webui(run: Runner) -> Path | None:
    """Open WebUI goes in its OWN venv, always.

    It pins 119 dependencies exactly, including `transformers==5.5.4`, while GLQ needs
    >=5.13.1 to serve gemma-4. Sharing a venv silently downgrades transformers and breaks
    GLQ. It also requires Python >=3.11,<3.13 where glq supports 3.10.
    """
    webui_venv = GLQ_HOME / "venv-webui"
    print("\n== Open WebUI (separate venv — it pins transformers==5.5.4, which would "
          "break GLQ's gemma-4 support if shared)")
    print("   license: 'Open WebUI License' (not an OSI-standard licence)")
    if sys.version_info < (3, 11) or sys.version_info >= (3, 13):
        print(f"   skipped: needs Python >=3.11,<3.13, this is "
              f"{sys.version_info.major}.{sys.version_info.minor}")
        return None
    run([sys.executable, "-m", "venv", str(webui_venv)])
    run([_venv_bin(webui_venv, "pip"), "install", "--upgrade", "open-webui"])
    return webui_venv


def _install_picode(run: Runner) -> None:
    """Install the pi coding agent via nvm.

    nvm is a shell function, not a binary, and is NOT on the PATH of a non-login shell — so
    every npm invocation must source it first. `--force` is needed because the legacy
    package may already own the `pi` bin (both traps are documented in
    `benchmarks/harbor_pi_glq.py`).
    """
    print("\n== pi coding agent (installs node via nvm)")
    nvm_sh = Path.home() / ".nvm" / "nvm.sh"
    if not nvm_sh.exists():
        run(["bash", "-c",
             "curl --proto '=https' --tlsv1.2 -fsSL "
             "https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash"])
    run(["bash", "-c",
         ". ~/.nvm/nvm.sh && nvm install --lts >/dev/null && "
         "npm install -g --force @earendil-works/pi-coding-agent"])
    # gemma-4's tool template is not in the model checkpoint (its bundled template is
    # plain chat) and not in the vLLM wheel — it lives in vLLM's repo examples. Measured:
    # the printed serve command referenced a path that did not exist and vLLM refused to
    # start. Fetched here, once, for every install that picks picode: the chosen model is
    # known at configure time but users switch models later, and the file is a few KB.
    # mkdir rides inside the same command so --dry-run creates nothing.
    tpl = GLQ_HOME / "templates" / GEMMA4_TOOL_TEMPLATE
    print("== gemma-4 tool-calling template (from vLLM's examples/)")
    try:
        run(["bash", "-c",
             f"mkdir -p {tpl.parent} && "
             f"curl --proto '=https' --tlsv1.2 -fsSL {GEMMA4_TOOL_TEMPLATE_URL} "
             f"-o {tpl}"])
    except Exception as exc:                                  # noqa: BLE001
        print(f"   fetch failed ({exc}) — tool calling on gemma-4 models needs it; "
              f"get it yourself later:\n"
              f"   curl --proto '=https' --tlsv1.2 -fsSL {GEMMA4_TOOL_TEMPLATE_URL} "
              f"-o {tpl}")


def _start_chat(venv) -> None:
    """Hand the terminal over to `glq-chat`, replacing this process.

    `exec` rather than a subprocess so Ctrl-C reaches the chat directly and there is no
    installer left sitting in the process tree waiting on it. Everything the user needs to
    read has already been printed by the time this runs — nothing after it happens.
    """
    chat = str(_venv_bin(Path(venv), "glq-chat"))
    print(f"\nStarting GLQ — {chat}\n")
    try:
        os.execv(chat, [chat])
    except OSError as exc:
        # "GLQ is installed." is already on screen, so a traceback here would read as a
        # failed install when only the handoff failed. Say what broke and stop.
        print(f"could not start {chat}: {exc}\n"
              f"Everything is installed — run it yourself when you are ready.")


def next_steps(*, venv, model: str, components, port: int, size_gib: float = 0.0,
               fp8_kv: bool = False) -> str:
    """The whole user manual for someone who arrived via `curl … | bash`.

    They have no repo checkout and no docs open, so every command must be copy-pasteable
    exactly as printed — which means absolute paths into the venv (the installer never
    activates it) and nothing under `examples/`, which is not in the wheel.
    """
    venv = Path(venv)
    py, out, n = _venv_bin(venv, "python"), [], 0

    def step(title, *cmds):
        nonlocal n
        n += 1
        out.append(f"{n}. {title}")
        out.extend(f"     {c}" for c in cmds)
        out.append("")

    out += ["", "=" * 74, "GLQ is installed.", ""]

    # The first start is the one that looks broken: it downloads GiB of weights and then
    # JIT-builds the CUDA extension, in silence. Say so, or it reads as a hang.
    first_run = (f"downloads ~{size_gib:.1f} GiB of weights and then " if size_gib
                 else "downloads the weights and then ")

    # pi is a tool-using agent: without these flags every request it makes fails with
    # `400 "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser`.
    # Only added when picode was installed — they change server behaviour, so someone who
    # just wants to chat should not silently get them. The parser follows the model's
    # FAMILY: hermes matches SmolLM3/Qwen-style <tool_call> markup and silently mangles
    # gemma-4's, which needs its own parser plus the external template the picode
    # component downloads (gemma-4's bundled chat template is plain chat).
    is_gemma4 = "gemma-4" in model.lower()
    tools = ""
    if "picode" in components:
        # hermes for families the shared helper does not know: the by-hand command is
        # advisory, and printing SOMETHING beats printing nothing — glq-code, which
        # actually starts servers, refuses unknown families instead.
        args = tool_serve_args(model, templates_dir=GLQ_HOME / "templates") or             ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
        tools = " " + " ".join(args)
    # A serving-time choice, so it belongs on the command the user copies — vLLM's own
    # flags, which is why they can simply be appended.
    kv_flags = kv_env.shell_suffix(fp8_kv)
    # "~30 s" was a guess copied from a source comment. Measured in a container on 8 cores,
    # cold cache, one arch: 38.3 s — and the build parallelises over MAX_JOBS, so a 4-core
    # machine is proportionally slower. Quote the range rather than the best case; someone
    # who was promised 30 s and waits 90 reasonably concludes it has hung. Whichever command
    # they run first pays this, so the warning goes on step 1 wherever step 1 lands.
    slow_start = (f"The first run {first_run}JIT-builds the CUDA"
                  f"\n   extension if no prebuilt kernel matches this Python (about a minute,"
                  f"\n   longer on fewer cores), so it is slow to start; later runs are not.")

    if "chat" in components:
        # One command, one terminal. glq-chat starts vLLM itself, waits for it, opens the
        # browser, and stops the server again when the chat is closed — which is the part
        # that matters on a desktop, because vLLM never releases VRAM on its own.
        step(f"Chat in a browser. This starts vLLM for you, opens "
             f"http://localhost:7860,\n   and stops the server again when you press Ctrl-C."
             f"\n   It also prints a public https://….gradio.live link so you can use it "
             f"from\n   a phone or another machine — anyone with that link can use this GPU,"
             f"\n   so pass --no-share to keep it on this machine."
             f"\n   {slow_start}",
             f"{_venv_bin(venv, 'glq-chat')}")

    step(("Serve it by hand instead — for a headless box, another client, or picode."
          if "chat" in components else f"Serve the model. {slow_start}"),
         f"{_venv_bin(venv, 'vllm')} serve {model} --quantization glq "
         f"--port {port}{kv_flags}{tools}")
    if tools and is_gemma4:
        out.insert(len(out) - 1,
                   "   (the tool-choice flags are what pi needs; the chat template is "
                   "gemma-4's tool\n    template, downloaded by the installer — the "
                   "model's own template is plain chat)\n")
    elif tools:
        out.insert(len(out) - 1,
                   "   (the tool-choice flags are what pi needs; `hermes` matches "
                   "SmolLM3-style\n    <tool_call> markup — other model families need a "
                   "different parser)\n")

    step("Check it is up (from another terminal):",
         f"curl -s http://127.0.0.1:{port}/v1/models")

    # /v1/models answers "is the port open", which is not the question that matters here.
    # GLQ compiles its CUDA kernels on this machine, so the server can start and list the
    # model while still being unable to run a forward pass — and someone who arrived via
    # `curl … | bash` has no client wired up, so without this their first real inference is
    # whenever they write one, far from the install that caused the failure. Same completion
    # the distro suite gates on, so "Paris" means the whole stack works.
    step("Generate a token — the check that the model actually decodes:",
         f"curl -s http://127.0.0.1:{port}/v1/completions \\",
         "       -H 'Content-Type: application/json' \\",
         f"       -d '{{\"model\": \"{model}\", \"prompt\": \"The capital of France is\", "
         f"\"max_tokens\": 8, \"temperature\": 0}}'",
         "",
         "   A working install completes it with Paris.")

    if "picode" in components:
        # One command: glq-code resolves pi (no nvm sourcing — one dropped dot there and
        # Ubuntu suggests the unrelated Raspberry-Pi package), serves vLLM with the
        # family-correct tool flags, and frees the GPU when pi exits.
        step("Code with the pi agent — starts its own tool-calling server, stops it "
             "when pi exits:",
             f"{_venv_bin(venv, 'glq-code')}   # --model <repo-id> for another "
             f"checkpoint")

    if fp8_kv:
        out += ["   KV cache: fp8 (vLLM's own) — about twice the context per GiB, at lower",
                "   attention precision; sliding-window layers are left alone.",
                "   `glq-chat --no-fp8-kv-cache` serves it at full precision.", ""]

    step("Serve a different checkpoint (--list shows all of them):",
         f"{_venv_bin(venv, 'glq-setup')} --list",
         f"{_venv_bin(venv, 'glq-setup')} --model <repo-id>")

    if "quantize" in components:
        step("Quantize a model of your own:",
             f"{_venv_bin(venv, 'glq-quantize')} --model <hf-repo> --output ./out "
             f"--bpw 4 --nsamples 128")
    else:
        step("Quantize a model of your own:",
             f"{_venv_bin(venv, 'pip')} install 'glq[quantize]'",
             f"{_venv_bin(venv, 'glq-quantize')} --model <hf-repo> --output ./out "
             f"--bpw 4 --nsamples 128")

    out += [f"Python in this venv: {py}",
            f"Config written to:   {GLQ_HOME / 'config.json'}",
            "Docs: https://github.com/cnygaard/glq",
            "🐚🐬🤿 GLQ READY",
            "=" * 74]
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="glq-setup", description="Set up GLQ serving, a chat UI and picode.")
    p.add_argument("--components", help="comma-separated: core,vllm,picode,chat,quantize")
    p.add_argument("--model", help="HF repo id to serve (default: chosen interactively)")
    p.add_argument("--chat", choices=("gradio", "openwebui", "none"), default="gradio")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--venv", default=str(GLQ_HOME / "venv"))
    p.add_argument("--yes", action="store_true", help="accept defaults, never prompt")
    p.add_argument("--fp8-kv-cache", dest="fp8_kv", action="store_true", default=None,
                   help="vLLM's fp8 KV cache: about twice the context per GiB")
    p.add_argument("--no-fp8-kv-cache", dest="fp8_kv", action="store_false",
                   help="keep the KV cache at full precision")
    p.add_argument("--start", dest="start", action="store_true", default=None,
                   help="start GLQ and open the chat when the install succeeds")
    p.add_argument("--no-start", dest="start", action="store_false",
                   help="never start GLQ, just print the steps")
    p.add_argument("--list", action="store_true", help="list checkpoints and exit")
    p.add_argument("--verify", action="store_true",
                   help="self-check an existing install and exit (no network)")
    p.add_argument("--dry-run", action="store_true", help="print commands, change nothing")
    args = p.parse_args(argv)

    venv = Path(args.venv)
    run = Runner(args.dry_run)

    if args.verify:
        components = tuple(c.strip() for c in (args.components or "core,vllm").split(","))
        checks = verify.run_checks(components)
        print(verify.render(checks))
        return 0 if verify.all_ok(checks) else 1

    gpu, vram = hardware.gpu_name(), hardware.vram_bytes()
    print(f"GPU:  {gpu or 'none detected'}"
          + (f"  ({vram / 1024**3:.1f} GiB)" if vram else ""))

    try:
        checkpoints = discovery.discover()
    except discovery.DiscoveryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    ranked = rank(checkpoints, vram)
    print(f"Found {len(checkpoints)} checkpoints in the GLQ collection.")

    if args.list:
        for r in ranked:
            fit = "recommended" if r.recommended else (
                "fits" if r.fits else "too large" if r.fits is False else "unknown")
            print(f"  {r.checkpoint.size_gib:6.1f} GiB  {r.checkpoint.repo_id}  [{fit}]")
        return 0

    tty = None if (args.yes or args.dry_run) else prompt.open_tty()
    try:
        if args.components:
            components = tuple(c.strip() for c in args.components.split(",") if c.strip())
        else:
            components = prompt.select_components(prompt.DEFAULT_COMPONENTS, tty=tty)

        kv_on = args.fp8_kv
        if kv_on is None:
            kv_on = prompt.confirm(
                "\nUse vLLM's fp8 KV cache? The cache holds 8 bits per element instead of "
                "16, so about twice the context fits in the same VRAM, at lower attention "
                "precision. Sliding-window layers are left alone",
                default=False, tty=tty)

        if args.model:
            chosen = next((c for c in checkpoints if c.repo_id == args.model), None)
            if chosen is None:                      # not in the collection: still allowed
                chosen = discovery.Checkpoint(args.model, 0)
        else:
            chosen = prompt.select_model(ranked, tty=tty)
    finally:
        if tty is not None:
            tty.close()

    print(f"\nComponents: {', '.join(components)}")
    print(f"Model:      {chosen.repo_id}")

    _install_python_extras(run, venv, components)
    if "picode" in components:
        _install_picode(run)
    if args.chat == "openwebui":
        _install_open_webui(run)

    base_url = f"http://127.0.0.1:{args.port}/v1"
    if not args.dry_run:
        configure.write_glq_config(
            GLQ_HOME / "config.json", model=chosen.repo_id, base_url=base_url,
            components=components, available=[c.repo_id for c in checkpoints],
            fp8_kv=bool(kv_on))
        if "picode" in components:
            configure.write_pi_models(
                Path.home() / ".pi" / "agent" / "models.json", base_url,
                [chosen.repo_id])
    else:
        print(f"  [dry-run] would write {GLQ_HOME / 'config.json'}")

    # Make the pip CUDA wheels usable by every compiler that will meet them, not just GLQ's.
    # They ship no `libcudart.so` and no `lib64/`, so `-lcudart` cannot resolve — measured in
    # a container, this stops **vLLM** dead (it JIT-builds flashinfer) long after GLQ's own
    # kernels have built and the self-check has gone green. Runs once, here, because this is
    # the point where the toolchain is known to be installed.
    if not args.dry_run:
        try:
            from glq import inference_kernel as _ik
            _ik.repair_cuda_wheel_layout()
        except Exception as exc:                                      # noqa: BLE001
            print(f"  note: could not normalise the CUDA wheel layout ({exc})")

    # Assert the install can do what next_steps is about to promise. Printing
    # "GLQ is installed." over a venv whose plugin does not resolve sends the user to
    # "Unknown quantization method: glq" with no clue the installer already knew.
    if not args.dry_run:
        checks = verify.run_checks(components)
        print(verify.render(checks))
        if not verify.all_ok(checks):
            print("\n" + "=" * 74)
            print("Install INCOMPLETE — the checks above failed, so serving would not "
                  "work.\nFix the FAIL lines and re-run, or run "
                  f"{_venv_bin(venv, 'glq-setup')} --verify to re-check.")
            print("=" * 74)
            return 1

    print(next_steps(venv=venv, model=chosen.repo_id, components=components,
                     port=args.port, size_gib=chosen.size_gib, fp8_kv=bool(kv_on)))

    # The handoff. `glq-chat` starts vLLM, opens the browser and stops the server again on
    # exit, so this turns "installed, now read four steps" into "installed, here is your
    # model" — which is where Ollama and LM Studio have been all along.
    #
    # It is deliberately hard to trigger by accident: only after a green self-check, only
    # when the chat was installed, and never from a non-interactive run. `--yes` is what
    # CI and `docker build` use, and an installer that seizes a GPU there is worse than one
    # that prints instructions.
    if "chat" in components and not args.dry_run and args.start is not False:
        start = args.start
        if start is None:
            # `confirm`'s default answers for a *user* who pressed Enter. With no terminal
            # there is no user, and defaulting to yes there would start a server in CI and
            # in `docker build` — so absence of a terminal means no, not the default.
            tty = None if args.yes else prompt.open_tty()
            if tty is None:
                start = False
            else:
                try:
                    start = prompt.confirm(
                        "\nStart GLQ now and open the chat? "
                        "(Ctrl-C stops it and frees the GPU)", default=True, tty=tty)
                finally:
                    tty.close()
        if start:
            _start_chat(venv)               # replaces this process
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
