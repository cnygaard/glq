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


def _install_python_extras(run: Runner, venv: Path, components) -> None:
    pip = _venv_bin(venv, "pip")
    wanted = []
    if "vllm" in components:
        wanted.append("vllm")
    if "chat" in components:
        wanted += ["gradio", "openai"]
    if wanted:
        print(f"\n== installing: {', '.join(wanted)}")
        run([pip, "install", "--upgrade", *wanted])


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


def next_steps(*, venv, model: str, components, port: int, size_gib: float = 0.0) -> str:
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

    # pi is a tool-using agent: without these two flags every request it makes fails with
    # `400 "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser`.
    # Only added when picode was installed — they change server behaviour, so someone who
    # just wants to chat should not silently get them.
    tools = (" --enable-auto-tool-choice --tool-call-parser hermes"
             if "picode" in components else "")
    # "~30 s" was a guess copied from a source comment. Measured in a container on 8 cores,
    # cold cache, one arch: 38.3 s — and the build parallelises over MAX_JOBS, so a 4-core
    # machine is proportionally slower. Quote the range rather than the best case; someone
    # who was promised 30 s and waits 90 reasonably concludes it has hung.
    step(f"Serve the model. The first run {first_run}JIT-builds the CUDA"
         f"\n   extension (about a minute, longer on fewer cores), so it is slow to start;"
         f"\n   later runs are not.",
         f"{_venv_bin(venv, 'vllm')} serve {model} --quantization glq --port {port}{tools}")
    if tools:
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

    if "chat" in components:
        step(f"Chat in a browser — then open http://localhost:7860",
             f"{_venv_bin(venv, 'glq-chat')}")

    if "picode" in components:
        # nvm is a shell function and is NOT on a non-login shell's PATH; a bare `pi`
        # gives command-not-found. Same trap as benchmarks/harbor_pi_glq.py.
        step("Use the pi coding agent against it (nvm must be sourced first):",
             f". ~/.nvm/nvm.sh && pi --provider glq --model {model}")

    step("Serve a different checkpoint (--list shows all of them):",
         f"{_venv_bin(venv, 'glq-setup')} --list",
         f"{_venv_bin(venv, 'glq-setup')} --model <repo-id>")

    step("Quantize a model of your own:",
         f"{_venv_bin(venv, 'pip')} install 'glq[quantize]'",
         f"{_venv_bin(venv, 'glq-quantize')} --model <hf-repo> --output ./out "
         f"--bpw 4 --nsamples 128")

    out += [f"Python in this venv: {py}",
            f"Config written to:   {GLQ_HOME / 'config.json'}",
            "Docs: https://github.com/cnygaard/glq",
            "=" * 74]
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="glq-setup", description="Set up GLQ serving, a chat UI and picode.")
    p.add_argument("--components", help="comma-separated: core,vllm,picode,chat")
    p.add_argument("--model", help="HF repo id to serve (default: chosen interactively)")
    p.add_argument("--chat", choices=("gradio", "openwebui", "none"), default="gradio")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--venv", default=str(GLQ_HOME / "venv"))
    p.add_argument("--yes", action="store_true", help="accept defaults, never prompt")
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
            components=components, available=[c.repo_id for c in checkpoints])
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
                     port=args.port, size_gib=chosen.size_gib))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
