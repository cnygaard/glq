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
                      _server_port, _vram_bytes, checkpoint_offload_bytes,
                      default_model, positive_seconds, sizing_weights_bytes)
from glq.installer.configure import pi_max_tokens, write_pi_models
from glq.supervisor import (DEFAULT_MAX_NUM_SEQS, DEFAULT_READY_TIMEOUT,
                            VllmSupervisor)
from glq.tooling import (ensure_gemma4_template, sampling_serve_args,
                         tool_serve_args)

#: The window a coding session falls back to when the card cannot be sized — a CPU box, or
#: a `--model` whose config we could not read. A coding agent carries file contents, diffs
#: and multi-turn tool results, so glq-chat's 8192 is a conversation, not a working set.
#:
#: This stays small deliberately. It is a FLOOR, not a target: the supervisor tiers up from
#: here to the model's declared maximum when it knows the VRAM, and raising the floor itself
#: would hand a CPU server a quarter-million-token window against an 8 GiB pool.
DEFAULT_CODE_MAX_MODEL_LEN = 16384

#: pi issues one request at a time, so one stream is both what the server should admit and
#: what the window should be priced at. The chat default of 16 (priced at 8) describes a
#: server nobody is running here, and it is expensive: the same 96 GB card and checkpoint
#: reach 131072 at eight streams and the model's full 262144 at one, because the window's KV
#: cost is multiplied by the concurrency it must hold.
#:
#: The cost of 1 is that a second client — glq-chat pointed at this server, say — queues
#: instead of batching. Correct for an agent, and `--max-num-seqs` re-prices the window with
#: it for anyone who wants otherwise.
DEFAULT_CODE_MAX_NUM_SEQS = 1


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


#: Worked examples, because the `--` convention lived in one `help=` string on a positional
#: and the README never showed a glq-code command line at all.
_EPILOG = """examples:
  glq-code                               serve, and start a fresh pi session
  glq-code -- --continue                 = pi --continue   (resume the last session)
  glq-code -- --resume                   = pi --resume     (pick a session to resume)
  glq-code --max-model-len 65536 -- -c   glq-code's own flags first, pi's after `--`

Everything after `--` is handed to pi verbatim; run `pi --help` for its flags. Note they
take two dashes or one letter -- `--resume` or `-r`, never `-resume`."""


class _PiAwareParser(argparse.ArgumentParser):
    """argparse, plus a hint that names the `--` separator when a pi flag was meant.

    `nargs=REMAINDER` cannot absorb a leading-dash token: argparse's option branch claims it
    first, so `glq-code --continue` exits 2 on `unrecognized arguments: --continue` and leaves
    the reader to guess that a separator exists. This appends the fix to exactly that message.

    Conditional on the extras looking like flags, deliberately. A hint on every parse failure
    is standing noise, and standing noise trains a reader past the error that matters -- the
    same reasoning `spot_scout.partition_scannable` records for `AuthFailure`.
    """

    def error(self, message):                                   # noqa: D102 - argparse hook
        if message.startswith("unrecognized arguments:"):
            flags = [a for a in message.split(":", 1)[1].split() if a.startswith("-")]
            if flags:
                message = (f"{message}\n  pi's own flags go after a `--` separator:  "
                           f"glq-code -- {' '.join(flags)}")
        super().error(message)


def main(argv=None) -> int:
    cfg = _installed_config()
    # allow_abbrev=False because prefix matching SILENTLY redirects abbreviated pi flags into
    # glq-code's own options, and several are unique prefixes: `--c` -> --cpu-offload-gb (which
    # then eats the next token as its int), `--r`/`--re` -> --ready-timeout (likewise),
    # `--v` -> --verbose (pi never gets it), `--no-s` -> --no-serve (vLLM is not started at
    # all). `glq-code --c 3` parsed cleanly and did the wrong thing, which is the failure class
    # this project treats as worst. Off, each of those becomes an error the hint explains.
    #
    # The cost is that glq-code's own flags must now be spelled in full: `--max-model` used to
    # work and no longer does.
    p = _PiAwareParser(description=__doc__.splitlines()[0], epilog=_EPILOG,
                       allow_abbrev=False,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=default_model(cfg, "code"),
                   help="checkpoint to serve (default: the installer's code pick, "
                        "then its generic one)")
    p.add_argument("--base-url", default=cfg.get("base_url", DEFAULT_BASE_URL))
    p.add_argument("--gpu-memory-utilization", type=float, default=None,
                   help="fraction of VRAM vLLM may reserve (default: sized from the "
                        "checkpoint)")
    # The offload policy stops once resident fits WEIGHT_FRACTION of VRAM, without asking
    # whether the headroom it left affords a useful window -- measured on a 23 GiB L4 serving
    # Flash-Next, it picks 34 GiB and a 32768 context where 42 GiB reaches 262144 and lowers
    # utilization. This is how to ask for the longer window. `0` serves resident.
    p.add_argument("--cpu-offload-gb", type=int, default=None, metavar="N",
                   help="GiB of MoE experts to keep in host RAM, overriding the plan "
                        "(0 = none; raising it buys context and costs PCIe decode time)")
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

    # The model card's sampling, pinned server-side because pi cannot send `top_k` at all
    # (not an OpenAI field, and the pi config has no slot for it). Appended, not substituted:
    # correct sampling with no tool parser is a useless coding session.
    tool_args = tool_args + (sampling_serve_args(args.model) or [])

    # Declared host-offloadable bytes: the supervisor sizes the pool and window from
    # what actually lands in VRAM, and turns PLE offload on for a checkpoint that
    # cannot serve without it. (0, 0) for everything that declares nothing.
    _offload = checkpoint_offload_bytes(args.model)
    # One value for both knobs, so what the server admits and what the window was priced for
    # can never disagree. Letting them drift is how a pool sized for one request ends up
    # admitting eight.
    _seqs = args.max_num_seqs or DEFAULT_CODE_MAX_NUM_SEQS
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
        max_num_seqs=_seqs,
        window_concurrency=_seqs,
        timeout=args.ready_timeout,
        extra_args=tool_args,
        weights_bytes=sizing_weights_bytes(args),
        vram_bytes=None if args.gpu_memory_utilization is not None else _vram_bytes(),
        # Declared host-offloadable bytes, so the supervisor sizes the pool and the window
        # from what actually lands in VRAM and turns PLE offload on for a checkpoint that
        # cannot serve without it. 0/0 for everything that declares nothing, which is every
        # checkpoint today except Qwen3.8-Flash-Next.
        ple_offload_bytes=_offload[0], expert_offload_bytes=_offload[1],
        nontext_bytes=_offload[2],
        expert_offload_gib=args.cpu_offload_gb,
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
                    max_tokens=pi_max_tokens(supervisor.max_model_len))

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
