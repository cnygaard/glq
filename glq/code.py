"""Run the pi coding agent against a supervised, tool-calling-correct vLLM server.

The manual sequence this replaces failed four separate ways in one evening: `pi` only
resolves after sourcing nvm (and a dropped dot makes Ubuntu suggest the unrelated
Raspberry-Pi package), the server needs `--enable-auto-tool-choice` plus a parser that
matches the model's tool markup (hermes silently mangles gemma-4's), gemma-4 needs an
external tool template that is in neither the checkpoint nor the vLLM wheel, and
`~/.pi/agent/models.json` has to agree with what is being served.

Same architecture as glq-chat: `VllmSupervisor` owns the server's lifetime, pi is the
foreground, and quitting pi frees the GPU — which matters more here than in the chat,
because coding sessions are long and people walk away from them.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from glq.chat import (DEFAULT_BASE_URL, _installed_config, _model_max_len,
                      _server_port, _vram_bytes, default_model,
                      positive_seconds, sizing_weights_bytes)
from glq.installer.configure import write_pi_models
from glq.supervisor import (DEFAULT_MAX_NUM_SEQS, DEFAULT_READY_TIMEOUT,
                            VllmSupervisor)
from glq.tooling import ensure_gemma4_template, tool_serve_args

#: A coding agent carries file contents, diffs and multi-turn tool results — glq-chat's
#: 8192 is a conversation, not a working set. Still far below gemma-4's declared 262144,
#: for the same reason chat caps it: nobody's KV pool should pay for a window the session
#: will not use. Raise it with --max-model-len.
DEFAULT_CODE_MAX_MODEL_LEN = 16384


def _find_pi() -> Path | None:
    """The pi binary, preferring nvm's node installs over PATH.

    Order matters: a bare `which pi` can resolve the unrelated Raspberry-Pi `pi` from
    apt/snap — exactly what Ubuntu suggests installing when the real one is missing.
    Newest node version first, matching what `nvm use default` would put on PATH.
    """
    candidates = sorted(glob.glob(str(Path.home() / ".nvm" / "versions" / "node"
                                      / "*" / "bin" / "pi")), reverse=True)
    if candidates:
        return Path(candidates[0])
    found = shutil.which("pi")
    return Path(found) if found else None


def _run_pi(cmd, env) -> int:
    return subprocess.call(cmd, env=env)


def main(argv=None) -> int:
    cfg = _installed_config()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=default_model(cfg, "code"),
                   help="checkpoint to serve (default: the installer's code pick, "
                        "then its generic one)")
    p.add_argument("--base-url", default=cfg.get("base_url", DEFAULT_BASE_URL))
    p.add_argument("--gpu-memory-utilization", type=float, default=None,
                   help="fraction of VRAM vLLM may reserve (default: sized from the "
                        "checkpoint)")
    p.add_argument("--max-model-len", type=int, default=None,
                   help=f"context window to serve (default: sized from VRAM headroom, "
                        f"floor {DEFAULT_CODE_MAX_MODEL_LEN} — a coding agent carries "
                        f"file contents and diffs; pass a number to pin it)")
    p.add_argument("--max-num-seqs", type=int, default=None,
                   help="concurrent sequences (default: 16 on GPU, 4 on the CPU backend)")
    p.add_argument("--ready-timeout", type=positive_seconds,
                   default=DEFAULT_READY_TIMEOUT, metavar="SECONDS")
    p.add_argument("--no-serve", dest="serve", action="store_false",
                   help="do not start vLLM; attach to a server you started yourself")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("pi_args", nargs=argparse.REMAINDER,
                   help="everything after `--` goes to pi verbatim")
    args = p.parse_args(argv)

    if not args.model:
        print("error: no model to serve — pass --model <repo-id> or run the installer",
              file=sys.stderr)
        return 2

    # Both guards run BEFORE the supervisor: starting vLLM costs minutes of weight
    # loading, all wasted if the agent cannot run or the parser would be wrong.
    pi = _find_pi()
    if pi is None:
        print("error: the pi coding agent is not installed. Install the picode "
              "component:\n  glq-setup --components picode   (or re-run install.sh "
              "and pick it)", file=sys.stderr)
        return 3

    tool_args = tool_serve_args(args.model)
    if tool_args is None:
        print(f"error: no known tool-calling setup for {args.model}. A wrong parser "
              f"fails silently\n(tool calls that never parse), so glq-code refuses to "
              f"guess. Supported families:\ngemma-4, SmolLM3, Qwen.", file=sys.stderr)
        return 2
    if "gemma4" in tool_args:
        # The template is in neither the checkpoint nor the vLLM wheel; the installer
        # downloads it, but this install may predate that or have skipped picode.
        ensure_gemma4_template()

    supervisor = VllmSupervisor(
        model=args.model,
        port=_server_port(args.base_url),
        base_url=args.base_url,
        gpu_memory_utilization=args.gpu_memory_utilization,
        serve=args.serve,
        verbose=args.verbose,
        max_model_len=args.max_model_len,
        max_model_len_floor=DEFAULT_CODE_MAX_MODEL_LEN,
        model_max_len=(None if args.max_model_len is not None or not args.model
                       else _model_max_len(args.model)),
        max_num_seqs=args.max_num_seqs,
        timeout=args.ready_timeout,
        extra_args=tool_args,
        weights_bytes=sizing_weights_bytes(args),
        vram_bytes=None if args.gpu_memory_utilization is not None else _vram_bytes(),
    )

    # pi resolves `glq/<model>` through ~/.pi/agent/models.json; refresh it so the
    # provider always points at the server this process is about to own (merge-safe:
    # other providers are preserved). After supervisor construction, because in auto
    # mode the served window is the supervisor's choice, not an args value.
    # maxTokens = window/4: pi treats it as the per-turn output ask, and the transcript
    # grows with every tool round-trip — window/2 fits the first turn and 400s later ones.
    write_pi_models(Path.home() / ".pi" / "agent" / "models.json",
                    args.base_url, [args.model],
                    context_window=supervisor.max_model_len,
                    max_tokens=max(1024, supervisor.max_model_len // 4))

    # `kill` and a closed terminal end the process without unwinding the context
    # manager below; turn them into SystemExit so the server still comes down.
    def _exit_on(signum, _frame):
        raise SystemExit(128 + signum)

    for _sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(_sig, _exit_on)
        except (ValueError, OSError, AttributeError):
            pass

    with supervisor:
        if supervisor.proc is None and args.serve:
            # We attached to a server someone else started — most likely glq-chat's,
            # which deliberately serves WITHOUT tool flags. pi's requests will then fail
            # with 400 "auto tool choice requires --enable-auto-tool-choice".
            print("  warning: attached to an already-running server — if it was not "
                  "started with tool\n  support, pi's requests will fail with a 400; "
                  "stop it and re-run glq-code.", file=sys.stderr)
        # Drop a literal leading "--" from REMAINDER; everything else goes to pi as-is.
        passthrough = [a for i, a in enumerate(args.pi_args)
                       if not (i == 0 and a == "--")]
        # npm bin shims are `#!/usr/bin/env node`: pi's own bin dir must be on the
        # child's PATH or the shebang cannot resolve node.
        env = {**os.environ,
               "PATH": f"{pi.parent}{os.pathsep}{os.environ.get('PATH', '')}"}
        return _run_pi([str(pi), "--provider", "glq", "--model", args.model,
                        *passthrough], env)


if __name__ == "__main__":
    raise SystemExit(main())
