"""Own a vLLM server's lifetime, so stopping the chat frees the GPU.

`glq-chat` used to be a pure client: the user started `vllm serve` in one terminal and the
UI in another, and if they ran the UI first they got a chat window with an empty model
dropdown. Worse, the documented serve command passed no `--gpu-memory-utilization`, so vLLM
took its default 0.9 — on a 24 GB card serving a 1.8 GB model that reserves ~21.6 GB of KV
pool and holds it until the process dies. On a machine that also plays games, that is not a
background service, it is the whole GPU.

vLLM has neither lazy load nor idle unload (Ollama has both), so "just leave it running" is
not a kindness here. One process therefore owns the whole lifetime: start the server, use
it, and on exit — including the exception path — take it down.

Lives apart from `glq/chat.py` so it can be tested without gradio and without a GPU: the
process factory, the health probe and the clock are all injected.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

from glq import kv_compression

#: vLLM's own default is 0.9 of *total* VRAM, which is the right call for a dedicated
#: inference box and the wrong one for a desktop. A GLQ checkpoint is small by construction —
#: that is the point of the project — so a modest pool still leaves a usable context while
#: keeping most of the card free for whatever else the user is doing.
DEFAULT_GPU_MEMORY_UTILIZATION = 0.45

#: Weight load plus CUDA-graph capture. Minutes, not seconds, and not something a wheel
#: fixes — the prebuilt kernels remove the *compile*, not the load.
DEFAULT_READY_TIMEOUT = 900.0

#: How often to say something while waiting. Measured on an L4: `glq-chat` printed three
#: lines and then nothing for nine minutes, which is indistinguishable from a hang.
DEFAULT_REPORT_EVERY = 5.0


#: What vLLM needs on top of the weights before a single KV block exists: activations,
#: workspace, CUDA-graph capture, and whatever the loaded model costs beyond the sum of its
#: `.safetensors`. Budgeting 2 GiB was measured wrong on an L4 — a 13.9 GiB checkpoint in an
#: 18.0 GiB budget left 0.42 GiB of KV, so the real overhead was ~3.7 GiB. 4 GiB, from that
#: measurement rather than from taste.
_RUNTIME_OVERHEAD_BYTES = 4 * 1024 ** 3

#: A KV pool below this serves no useful context, so it is part of what must fit rather than
#: something to leave to chance.
_MIN_KV_BYTES = 2 * 1024 ** 3

#: vLLM defaults `max_model_len` to whatever the model declares, and gemma-4 declares
#: 262144 — which needs 6.15 GiB of KV for one request, measured. A chat does not need a
#: quarter-million-token window, and asking for one means no card of this size can serve a
#: model of this size. 8192 is a conversation; raise it with --max-model-len.
DEFAULT_MAX_MODEL_LEN = 8192

#: Leave a slice of the card for the desktop. Above this, vLLM competes with the compositor
#: and a display server can fail to allocate.
_MAX_UTILIZATION = 0.92


def plan_gpu_memory_utilization(*, weights_bytes, vram_bytes):
    """How much of the card vLLM may reserve, sized from the checkpoint.

    A fixed fraction cannot be right for every model: measured on a 23 GB L4, the 0.45 that
    leaves a 1.8 GB SmolLM3 politely sharing the card gives a 15 GB MoE a 10.4 GB budget, so
    the weights alone overflow it and vLLM dies with "No available memory for the cache
    blocks". Ask instead for weights + runtime overhead + a usable cache, which is small when
    the model is small — the point of not simply taking vLLM's 0.9 — and large when it must
    be.

    Returns the documented default when either input is unknown: `--model` can point at a
    repo whose size we failed to look up, and guessing large would seize the whole GPU.
    """
    if not weights_bytes or not vram_bytes:
        return DEFAULT_GPU_MEMORY_UTILIZATION
    needed = weights_bytes + _RUNTIME_OVERHEAD_BYTES + _MIN_KV_BYTES
    return min(max(needed / vram_bytes, DEFAULT_GPU_MEMORY_UTILIZATION), _MAX_UTILIZATION)


def default_log_path():
    return Path(os.environ.get("GLQ_HOME", Path.home() / ".glq")) / "vllm.log"


def server_up(base_url: str, timeout: float = 1.5) -> bool:
    """Is something already answering the OpenAI API at `base_url`?

    Deliberately urllib rather than the `openai` client: this runs before the UI is built and
    must not depend on it, and a bare GET is all the question needs.
    """
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:      # noqa: S310
            return 200 <= r.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _ninja_present(which=shutil.which) -> bool:
    """Is ninja reachable? FlashInfer's JIT shells out to it **by name**, with no override.

    `install.sh` installs it into the venv, but running a venv binary by absolute path does
    not put the venv's bin/ on PATH — which is why `child_env` prepends it.
    """
    return bool(which("ninja") or which(
        "ninja", path=os.path.dirname(sys.executable)))


def _nvcc_present(which=shutil.which) -> bool:
    """Is the NVIDIA CUDA Toolkit's compiler on PATH (or under CUDA_HOME)?

    `nvcc` is the component that decides this: FlashInfer shells out to it. The toolkit is
    what the user installs; nvcc is how we detect it.
    """
    if which("nvcc"):
        return True
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    return bool(home) and os.path.exists(os.path.join(home, "bin", "nvcc"))


def _compute_cap(run=subprocess.run):
    """The GPU's compute capability as reported by nvidia-smi, e.g. "12.0", or None."""
    try:
        out = run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                  capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    line = (out.stdout or "").strip().splitlines()
    return line[0].strip() if line else None


#: Below this, FlashInfer ships prebuilt kernels and needs no toolkit.
_FLASHINFER_PREBUILT_BELOW = 12.0


def flashinfer_env(compute_cap=None, have_nvcc=None, have_ninja=None) -> dict:
    """Environment overrides needed for vLLM to start on this machine.

    Measured on an RTX PRO 6000 (sm_120), vLLM 0.27.1, no CUDA Toolkit: GLQ's own prebuilt
    kernel loads and decodes fine, but EngineCore dies before generating a token because
    vLLM's sampler backend JIT-compiles for `120f` and cannot find nvcc. The same container
    with VLLM_USE_FLASHINFER_SAMPLER=0 answered "Paris".

    Scoped deliberately to new architectures: FlashInfer ships prebuilt kernels for sm_86
    and sm_89, so switching samplers there would trade speed away to fix nothing. Unknown
    capability changes nothing either — a silent sampler swap on a guess is worse than the
    error, which at least names its own cause.
    """
    if compute_cap is None:
        compute_cap = _compute_cap()
    if have_nvcc is None:
        have_nvcc = _nvcc_present()
    if have_ninja is None:
        have_ninja = _ninja_present()
    # nvcc alone is not enough. Measured on an RTX PRO 6000 with a CUDA 13.2 toolkit on
    # PATH: FlashInfer still died with `FileNotFoundError: 'ninja'`, because the JIT shells
    # out to ninja and the venv's bin/ was not on the child's PATH. A toolchain missing
    # either half cannot build, so either half is grounds to fall back.
    if (have_nvcc and have_ninja) or not compute_cap:
        return {}
    try:
        cap = float(compute_cap)
    except (TypeError, ValueError):
        return {}
    return {"VLLM_USE_FLASHINFER_SAMPLER": "0"} if cap >= _FLASHINFER_PREBUILT_BELOW else {}


def child_env() -> dict:
    """The environment `vllm serve` is started with.

    PYTHONUNBUFFERED: without it the child block-buffers into the log file, so the lines
    explaining a failure arrive long after we have given up — or never, if it dies with them
    unflushed.

    PATH: the venv's own bin/ goes first. We launch vLLM by absolute path, which does NOT
    put that directory on PATH the way `source activate` would — so a child that shells out
    to a sibling tool cannot find it. Measured: FlashInfer JIT-compiles its sampler on
    sm_120, runs `ninja` by name, and dies with FileNotFoundError while
    ~/.glq/venv/bin/ninja sits there unused, installed by install.sh for this exact purpose.
    """
    bindir = os.path.dirname(sys.executable)
    path = os.environ.get("PATH", "")
    parts = [p for p in path.split(os.pathsep) if p and p != bindir]
    env = {**os.environ, "PYTHONUNBUFFERED": "1",
           "PATH": os.pathsep.join([bindir, *parts])}
    # After PATH, so the probe sees the tools the child will actually have.
    env.update(flashinfer_env())
    return env


class VllmSupervisor:
    """Start vLLM if nothing is serving, wait for it, and stop it again on the way out.

    Use it as a context manager; `__exit__` is the VRAM-release guarantee and has to hold on
    the exception path, which is exactly where it matters.
    """

    def __init__(self, *, model, port=8000, base_url=None, gpu_memory_utilization=None,
                 vllm_bin=None, extra_args=(), serve=True,
                 spawn=subprocess.Popen, probe=server_up,
                 sleep=time.sleep, monotonic=time.monotonic,
                 timeout=DEFAULT_READY_TIMEOUT, out=None,
                 log_path=None, verbose=False,
                 report_every=DEFAULT_REPORT_EVERY,
                 weights_bytes=None, vram_bytes=None,
                 max_model_len=DEFAULT_MAX_MODEL_LEN, fp8_kv=False):
        self.model = model
        self.port = int(port)
        self.base_url = base_url or f"http://127.0.0.1:{self.port}/v1"
        # An explicit flag always wins; otherwise size the pool from the checkpoint, because
        # a fixed fraction starves anything bigger than it.
        self.gpu_memory_utilization = (
            float(gpu_memory_utilization) if gpu_memory_utilization is not None
            else plan_gpu_memory_utilization(weights_bytes=weights_bytes,
                                             vram_bytes=vram_bytes))
        self.vllm_bin = vllm_bin or os.path.join(os.path.dirname(sys.executable), "vllm")
        self.extra_args = list(extra_args)
        self.max_model_len = int(max_model_len)
        self.fp8_kv = bool(fp8_kv)
        self.serve = serve
        self._spawn, self._probe = spawn, probe
        self._sleep, self._monotonic = sleep, monotonic
        self.timeout = float(timeout)
        self._out = out if out is not None else sys.stderr
        self.proc = None                 #: set only when *we* started it
        self.log_path = Path(log_path) if log_path else default_log_path()
        self.verbose = verbose
        self.report_every = float(report_every)
        self._log_fh = None              #: the child writes here
        self._reader = None              #: we read the same file to report progress
        self._tail = deque(maxlen=40)
        self._last_output_at = None      #: when the child last said anything

    # ------------------------------------------------------------------ lifecycle

    def argv(self) -> list[str]:
        return [self.vllm_bin, "serve", self.model,
                "--quantization", "glq",
                "--port", str(self.port),
                "--gpu-memory-utilization", str(self.gpu_memory_utilization),
                "--max-model-len", str(self.max_model_len),
                *kv_compression.serve_args(self.fp8_kv),
                *self.extra_args]

    def start(self) -> bool:
        """True if we started a server, False if we attached to a running one.

        Attaching matters twice over: a second vLLM on the same port would fail anyway, and
        reusing a warm server is the only thing that makes a repeat `glq-chat` instant.
        """
        if self._probe(self.base_url):
            self._say(f"attaching to the server already on {self.base_url}")
            return False
        if not self.serve:
            raise RuntimeError(
                f"no server at {self.base_url} and --no-serve was given. Start one with:\n"
                f"  {' '.join(self.argv())}")

        self._say(f"starting vLLM for {self.model}")
        self._say(f"  reserving {self.gpu_memory_utilization:.0%} of VRAM for the weights "
                  f"and KV cache, with a {self.max_model_len} token context")
        # Say how long this takes *before* going quiet. Minutes of silence you were warned
        # about is patience; the same silence unannounced is indistinguishable from a hang,
        # and that is what a first run looks like today.
        self._say("  the first run downloads the weights and loads the model — expect a few "
                  "minutes")

        # A file, not a pipe. A pipe holds ~64 KiB and then *blocks the writer*, so a child
        # this chatty could wedge on its own logging with nobody reading. It also gives the
        # user something to tail, and us something to quote when startup fails.
        log = self._open_log()
        self._say(f"  vLLM log: {self.log_path}")

        # PYTHONUNBUFFERED: without it the child block-buffers into the file, so the lines
        # explaining a failure arrive long after we have given up — or never, if it dies
        # with them unflushed.
        if self.fp8_kv:
            self._say("  KV cache in fp8 (vLLM's own) — about twice the context per GiB")
        env = child_env()
        if "VLLM_USE_FLASHINFER_SAMPLER" in env:
            self._say("  the NVIDIA CUDA Toolkit is not installed, and FlashInfer ships no "
                      "prebuilt sampler for this GPU, so it")
            self._say("  cannot build one — falling back to vLLM's built-in sampler. "
                      "Please install the CUDA Toolkit for the faster path:")
            self._say("  https://developer.nvidia.com/cuda-downloads")
        self.proc = self._spawn(self.argv(), stdout=log,
                                stderr=subprocess.STDOUT, env=env)
        try:
            self._wait_until_ready()
        except BaseException:
            self.stop()                  # never leave a half-started server holding the card
            raise
        return True

    def _open_log(self):
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self.log_path, "w", buffering=1)
        except OSError:                  # unwritable GLQ_HOME: a log is not worth failing over
            import tempfile
            self.log_path = Path(tempfile.mkstemp(prefix="glq-vllm-", suffix=".log")[1])
            self._log_fh = open(self.log_path, "w", buffering=1)
        return self._log_fh

    def stop(self) -> None:
        """Terminate the child, escalating to kill. A server we merely attached to is left
        alone — we did not start it, so it is not ours to end."""
        proc, self.proc = self.proc, None
        if proc is None:
            return
        self._say("stopping vLLM — the GPU is free again")
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:                # noqa: BLE001 - TimeoutExpired, or a fake's stand-in
            proc.kill()
            try:
                proc.wait(timeout=10)
            except Exception:            # noqa: BLE001 - nothing further we can do
                pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_exc):
        self.stop()
        return False

    # ------------------------------------------------------------------ internals

    def _say(self, msg: str) -> None:
        print(f"  {msg}", file=self._out, flush=True)

    def _wait_until_ready(self) -> None:
        """Poll until the endpoint answers, the child dies, or we run out of patience.

        Reports progress in vLLM's own words while it waits. Its log says what it is doing —
        downloading shards, loading weights, capturing CUDA graphs — which is both more
        accurate and more reassuring than anything we could invent, and the difference
        between a terminal that looks busy and one that looks hung.
        """
        started = self._monotonic()
        deadline = started + self.timeout
        last_report = started
        while True:
            rc = self.proc.poll()
            self._collect()
            if rc is not None:
                raise RuntimeError(
                    f"vLLM exited with status {rc} before it was ready.\n" + self._drain())
            if self._probe(self.base_url):
                self._say(f"ready in {self._monotonic() - started:.0f}s — {self.base_url}")
                return
            now = self._monotonic()
            if now >= deadline:
                raise RuntimeError(
                    f"vLLM did not become ready within {self.timeout:.0f}s.\n" + self._drain())
            if now - last_report >= self.report_every:
                self._report(now - started)
                last_report = now
            self._sleep(2.0)

    @staticmethod
    def _informative(line: str) -> bool:
        """Is this line worth showing as "what vLLM is doing"?

        Python prints warnings over two lines — the message, then the offending source line —
        and the source line lands last, so a naive "latest line" reports
        `warnings.warn('resource_tracker: ...'` and keeps reporting it while the real work
        goes on silently. Measured on an L4: ten identical progress lines over a minute.
        """
        s = line.strip()
        return bool(s) and not s.startswith(("warnings.warn", "warnings.simplefilter",
                                             "self._", "return ", "File \""))

    def _collect(self) -> None:
        """Pull whatever the child has written since last time into the tail buffer."""
        if self._reader is None:
            try:
                self._reader = open(self.log_path, "r", errors="replace")
            except OSError:
                return
        try:
            fresh = self._reader.read()
        except OSError:                  # noqa: BLE001 - nothing to add
            return
        for line in fresh.splitlines():
            if line.strip():
                self._tail.append(line)
                self._last_output_at = self._monotonic()
                if self.verbose:
                    print(f"  | {line}", file=self._out, flush=True)

    def _report(self, elapsed: float) -> None:
        """One line: how long we have been waiting, and the child's latest word on it."""
        latest = next((ln for ln in reversed(self._tail) if self._informative(ln)), None)
        latest = (latest or "waiting for vLLM to come up").strip()
        if len(latest) > 96:
            latest = latest[:93] + "..."

        # Distinguish "still working" from "has said nothing for a while". Without this the
        # same line reappears on a timer and the two look identical.
        quiet = ""
        if self._last_output_at is not None:
            since = self._monotonic() - self._last_output_at
            if since >= 30:
                quiet = f"  (no new output for {since:.0f}s)"
        self._say(f"[{elapsed:4.0f}s] {latest}{quiet}")

    def _drain(self, keep: int = 25) -> str:
        """The child's last lines, plus where to read the rest — the reason a startup failed
        is often further up than any tail we would print."""
        self._collect()
        tail = "\n".join(list(self._tail)[-keep:]) or "(vLLM produced no output)"

        # vLLM names the flag but not a number, and the user cannot see what we computed.
        # This is the one startup failure with a one-line fix, so spell it out.
        hint = ""
        if "No available memory for the cache blocks" in tail:
            hint = (f"\n\nThe model did not fit in {self.gpu_memory_utilization:.0%} of this "
                    f"GPU. Retry with a larger share, e.g.\n"
                    f"  glq-chat --gpu-memory-utilization "
                    f"{min(self.gpu_memory_utilization + 0.2, 0.95):.2f}")
        return f"{tail}{hint}\n\nFull log: {self.log_path}"
