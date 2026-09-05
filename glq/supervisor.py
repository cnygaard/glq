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
import signal
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

#: vLLM defaults max_num_seqs to 1024 — sized for a batch server, absurd for a single-user
#: chat — and on hybrid-GDN models every decode sequence reserves its own Mamba cache
#: block before a single request exists. Measured with a 27B GDN hybrid on a 96 GB card:
#: "max_num_seqs (1024) exceeds available Mamba cache blocks (399)", startup refused, with
#: a pool that held ~24 GiB of cache. 16 covers regenerations and a couple of parallel
#: tabs, and shrinks CUDA-graph capture and KV pressure for every model, not just hybrids.
DEFAULT_MAX_NUM_SEQS = 16

#: The CPU-backend default. At single-digit tok/s TOTAL, 16 concurrent decodes is thrash;
#: 4 covers a regeneration plus one parallel tab. Explicit --max-num-seqs always wins.
DEFAULT_CPU_MAX_NUM_SEQS = 4

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


#: gemma-4's measured KV cost — 6.15 GiB for one 262,144-token request — is the
#: worst-case anchor among the served families: hybrid-GDN models keep constant-size
#: state for most layers and sliding-window layers page smaller, so sizing against this
#: number only over-reserves. Exact per-model KV math is vLLM's job; duplicating it
#: client-side breaks on every backend change.
_KV_BYTES_PER_TOKEN = int(6.15 * 1024**3 / 262144)

_WINDOW_TIERS = (8192, 16384, 32768, 65536)

#: The window must be affordable at realistic chat concurrency, not for one request —
#: half of DEFAULT_MAX_NUM_SEQS. This is the constant that keeps a 23 GiB L4 serving a
#: 14.4 GiB 26B at 8192 (headroom ≈ 2.1 GiB; 16384×8×24.6 KiB ≈ 3.1 GiB does not fit)
#: while a 96 GiB card reaches 65536 (headroom ≈ 24 GiB ≥ 12.3 GiB).
_WINDOW_CONCURRENCY = 8


def plan_max_model_len(*, weights_bytes, vram_bytes, model_max_len,
                       floor=DEFAULT_MAX_MODEL_LEN):
    """The served context window, tiered from the KV headroom the pool plan leaves.

    A fixed 8192 was designed for 24 GiB desktops and wastes a 96 GiB card; the model's
    declared maximum (gemma-4: 262,144) drowns any card. Pick the largest tier whose
    full window, at chat concurrency and the worst-case per-token anchor, fits inside
    the pool `plan_gpu_memory_utilization` is already going to reserve — this feature
    grabs no extra VRAM.

    Any unknown input returns the floor: tiering up blind is strictly worse than a
    small window, because vLLM refuses a --max-model-len above the model's declared
    maximum (SmolLM2-class models declare 8192) and then nothing serves at all.
    """
    if not weights_bytes or not vram_bytes or not model_max_len:
        return floor
    util = plan_gpu_memory_utilization(weights_bytes=weights_bytes,
                                       vram_bytes=vram_bytes)
    headroom = util * vram_bytes - weights_bytes - _RUNTIME_OVERHEAD_BYTES
    chosen = floor
    for tier in _WINDOW_TIERS:
        if tier * _WINDOW_CONCURRENCY * _KV_BYTES_PER_TOKEN <= headroom:
            chosen = max(chosen, tier)
    return min(chosen, int(model_max_len))


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


def _installed_vllm_version() -> str:
    from importlib.metadata import version
    return version("vllm")


def detect_device(vllm_version=_installed_vllm_version, vram=None) -> str:
    """"cpu" or "cuda" — the backend this venv can actually serve with.

    The INSTALLED WHEEL wins over the live GPU: a `+cpu` vLLM cannot use a GPU however
    the flags read (real case: a GPU box where the user installed the CPU stack), and
    reading the wheel's local version is a metadata lookup, not an import. Only when the
    version is unreadable does the GPU probe decide.
    """
    try:
        if "+cpu" in vllm_version():
            return "cpu"
    except Exception:                                    # noqa: BLE001 - fall to the probe
        pass
    if vram is None:
        from glq.installer.hardware import vram_bytes as vram_probe
        vram = vram_probe
    return "cuda" if vram() is not None else "cpu"


#: VLLM_CPU_KVCACHE_SPACE bounds, GiB. 8 is the value the CPU serving path was validated
#: with; at the ~24.6 KiB/token worst-case anchor it already holds ~340k tokens, far past
#: any served window at CPU concurrency — more would only crowd out the weights.
_CPU_KV_MIN_GIB, _CPU_KV_MAX_GIB = 2, 8


#: What the CPU serving stack costs beyond the weights and the KV pool. Measured on an
#: 8-vCPU box serving the 13.9 GiB 26B-A4B with a 4 GiB pool: the VLLM::Worker held
#: 18.56 GiB (so ~0.7 GiB of activations above weights + pool) and the driver and engine
#: processes another ~1.4 GiB.
_CPU_RUNTIME_OVERHEAD_BYTES = 2 * 1024 ** 3

#: Share of RAM the *anonymous* demand — weights + pool + runtime — may reach. The rest is
#: for the page cache, and on CPU that is not a luxury: the loader streams the whole
#: checkpoint through it, and a box with no swap configured (the AWS default) can reclaim
#: nothing else. Anchored to two observed configurations on a 30.8 GiB box holding the
#: 13.9 GiB 26B-A4B: a 4 GiB pool (≈65% anonymous) served for hours across many runs, and
#: a 7 GiB pool (≈77%) left kswapd0 pinned at 100% with 85% iowait, buff/cache down to
#: 100 MiB, and sshd unable to complete a banner exchange. 0.70 sits between them, nearer
#: the configuration that worked. Refine it when the failure is reproduced under
#: instrumentation — it is bounded by observation, not measured to a knife edge.
_CPU_ANON_FRACTION = 0.70

#: The ceiling when the checkpoint size could not be looked up, so the arithmetic above is
#: unavailable. 4 GiB is the pool that served the 13.9 GiB 26B-A4B across every run of this
#: work without pressure; 8 would be betting that the model is small.
_CPU_KV_UNKNOWN_MAX_GIB = 4


def plan_cpu_kvcache_gib(ram_bytes, weights_bytes=None) -> int:
    """The CPU KV pool in GiB, clamped to [2, 8].

    On a GPU the weights and the KV cache come out of VRAM while the loader, the runtime
    and the page cache come out of RAM, so sizing the pool against one number works. On CPU
    they all come out of the *same* RAM, which is why this needs `weights_bytes`: a quarter
    of RAM is 7 GiB on a 32 GiB box whether the checkpoint is 1.8 GiB or 13.9, and in the
    second case that overcommits the machine.

    Without a checkpoint size — an unknown repo, a local path we could not measure — it
    keeps the old fraction-of-RAM answer. That is the input we have, and a pool that is
    too small costs context rather than the machine.
    """
    if not ram_bytes:
        return _CPU_KV_MAX_GIB
    if not weights_bytes:
        # Size unknown — an offline box, a local path, a 404. A quarter of RAM was the old
        # rule and it is a gamble here: on a 32 GiB machine it hands out 7 GiB whether the
        # checkpoint is 1.8 GiB or 13.9. Cap the guess at the largest pool observed serving
        # comfortably, because the two errors are not symmetric — too small costs context,
        # too large hangs a machine with no swap.
        quarter = int(ram_bytes / 2**30) // 4
        return max(_CPU_KV_MIN_GIB, min(_CPU_KV_UNKNOWN_MAX_GIB, quarter))
    budget = (_CPU_ANON_FRACTION * ram_bytes - weights_bytes
              - _CPU_RUNTIME_OVERHEAD_BYTES)
    return max(_CPU_KV_MIN_GIB, min(_CPU_KV_MAX_GIB, int(budget / 2**30)))


def child_env(device: str = "cuda", ram_bytes=None, weights_bytes=None) -> dict:
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
    if device == "cpu":
        # The CPU backend takes its KV pool from this env var (GiB). A user-set value
        # always wins; the flashinfer probe is skipped — it answers a GPU question with
        # an nvidia-smi subprocess this path has no use for.
        env.setdefault("VLLM_CPU_KVCACHE_SPACE",
                       str(plan_cpu_kvcache_gib(ram_bytes, weights_bytes)))
        # vLLM's auto CPU binding takes one logical CPU per physical core and then holds
        # one back for itself, so a 4-core machine runs GLQ's kernels on 3. Decode is
        # memory-bound and scales with cores until the bandwidth saturates: serving the
        # 26B-A4B on an 8-vCPU Sapphire Rapids measured 3.0 tok/s on 3 cores and 3.3 on 4
        # (binding all 8 logical CPUs added nothing — SMT siblings share the same
        # bandwidth). Nothing is competing for that core during single-stream chat, which
        # is what this supervisor serves, so claim it. setdefault: a busy box can hold
        # cores back by setting this itself.
        env.setdefault("VLLM_CPU_NUM_OF_RESERVED_CPU", "0")
        return env
    # After PATH, so the probe sees the tools the child will actually have.
    env.update(flashinfer_env())
    return env


def _hf_download_bytes() -> int:
    """Bytes of in-flight HF downloads: the sizes of `*.incomplete` blobs in the hub cache.

    A checkpoint download writes NOTHING to vLLM's log, so this is the only sign of life
    during the phase that takes longest on a first run. Shallow glob on the hub layout
    (`models--*/blobs/*.incomplete`) — cheap enough to poll every 2 s even on a large cache.
    """
    root = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    total = 0
    for p in (root / "hub").glob("models--*/blobs/*.incomplete"):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


class VllmSupervisor:
    """Start vLLM if nothing is serving, wait for it, and stop it again on the way out.

    Use it as a context manager; `__exit__` is the VRAM-release guarantee and has to hold on
    the exception path, which is exactly where it matters.
    """

    def __init__(self, *, model, port=8000, base_url=None, gpu_memory_utilization=None,
                 vllm_bin=None, extra_args=(), serve=True,
                 spawn=subprocess.Popen, probe=server_up,
                 sleep=time.sleep, monotonic=time.monotonic,
                 download_bytes=_hf_download_bytes,
                 killpg=os.killpg, getpgid=os.getpgid,
                 timeout=DEFAULT_READY_TIMEOUT, out=None,
                 log_path=None, verbose=False,
                 report_every=DEFAULT_REPORT_EVERY,
                 weights_bytes=None, vram_bytes=None,
                 max_model_len=None, model_max_len=None,
                 max_model_len_floor=DEFAULT_MAX_MODEL_LEN, fp8_kv=False,
                 max_num_seqs=None, device=None, ram_bytes=None):
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
        # max_model_len=None means "size it": the largest window tier the planned
        # pool's KV headroom affords, clamped to the model's declared maximum. With any
        # sizing input unknown this lands exactly on the old fixed default, so callers
        # that never pass the lookups keep today's behavior.
        self._window_note = ""
        if max_model_len is None:
            max_model_len = plan_max_model_len(
                weights_bytes=weights_bytes, vram_bytes=vram_bytes,
                model_max_len=model_max_len, floor=max_model_len_floor)
            if max_model_len > max_model_len_floor:
                self._window_note = (" (sized from KV headroom; "
                                     "--max-model-len overrides)")
        self.max_model_len = int(max_model_len)
        # None = auto: follow the wheel/GPU detection. Explicit "cpu"/"cuda" is for tests
        # and for callers that already decided (the config records the installer's choice,
        # but the wheel in THIS venv is what actually serves — so live detection is the
        # default even then).
        self.device = device if device in ("cpu", "cuda") else detect_device()
        if self.device == "cpu" and ram_bytes is None:
            try:
                from glq.installer.hardware import ram_bytes as _ram
                ram_bytes = _ram()
            except Exception:                            # noqa: BLE001 - pool falls to 8 GiB
                ram_bytes = None
        self._ram_bytes = ram_bytes
        # A --gpu-memory-utilization on the CPU backend would be silently meaningless;
        # remember it so start() can say so once instead of dropping it wordlessly.
        self._ignored_gpu_flag = (self.device == "cpu"
                                  and gpu_memory_utilization is not None)
        self.max_num_seqs = int(max_num_seqs) if max_num_seqs is not None else (
            DEFAULT_CPU_MAX_NUM_SEQS if self.device == "cpu" else DEFAULT_MAX_NUM_SEQS)
        self.fp8_kv = bool(fp8_kv)
        self.serve = serve
        self._spawn, self._probe = spawn, probe
        self._sleep, self._monotonic = sleep, monotonic
        self._download_bytes = download_bytes
        self._killpg, self._getpgid = killpg, getpgid
        self._dl_seen = 0                #: highest in-flight download byte count seen
        self._dl_at = None               #: when the download last grew
        #: expected .safetensors bytes — the download total, when the caller looked it up
        self.weights_bytes = weights_bytes
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
        if self.device == "cpu":
            # --gpu-memory-utilization means nothing to the CPU backend, and its
            # fullgraph compile cannot trace the fused CPU decode path — --enforce-eager
            # is a requirement, not a preference. The KV pool travels in the child env
            # (VLLM_CPU_KVCACHE_SPACE), not a flag.
            return [self.vllm_bin, "serve", self.model,
                    "--quantization", "glq",
                    "--port", str(self.port),
                    "--enforce-eager",
                    "--max-model-len", str(self.max_model_len),
                    "--max-num-seqs", str(self.max_num_seqs),
                    *self.extra_args]
        return [self.vllm_bin, "serve", self.model,
                "--quantization", "glq",
                "--port", str(self.port),
                "--gpu-memory-utilization", str(self.gpu_memory_utilization),
                "--max-model-len", str(self.max_model_len),
                "--max-num-seqs", str(self.max_num_seqs),
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
        if self.device == "cpu":
            self._say(f"  serving on the CPU backend — expect single-digit tokens/s; "
                      f"{self.max_model_len} token context")
            if self._ignored_gpu_flag:
                self._say("  (--gpu-memory-utilization has no effect on the CPU backend "
                          "— ignoring it)")
        else:
            self._say(f"  reserving {self.gpu_memory_utilization:.0%} of VRAM for the weights "
                      f"and KV cache, with a {self.max_model_len} token context"
                      f"{self._window_note}")
        # Say how long this takes *before* going quiet. Minutes of silence you were warned
        # about is patience; the same silence unannounced is indistinguishable from a hang,
        # and that is what a first run looks like today.
        size = (f" (~{self.weights_bytes / 2**30:.1f} GiB)"
                if self.weights_bytes else "")
        self._say(f"  the first run downloads the weights{size} and loads the model — "
                  f"expect a few minutes")

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
        env = child_env(device=self.device, ram_bytes=self._ram_bytes,
                        weights_bytes=self.weights_bytes)
        if self.device == "cpu" and self._ram_bytes and self.weights_bytes:
            # Say the arithmetic out loud. On CPU the weights, the pool, the runtime and the
            # page cache share one pool of RAM, and when it does not fit there is no swap to
            # absorb it on a default cloud box — the machine stops answering rather than
            # OOM-killing one process, which is a much harder failure to read after the fact.
            pool = int(env["VLLM_CPU_KVCACHE_SPACE"])
            anon = self.weights_bytes + pool * 2**30 + _CPU_RUNTIME_OVERHEAD_BYTES
            self._say(f"  RAM plan: {self.weights_bytes / 2**30:.1f} GiB weights + "
                      f"{pool} GiB KV pool + ~{_CPU_RUNTIME_OVERHEAD_BYTES / 2**30:.0f} GiB "
                      f"runtime = {anon / 2**30:.1f} of {self._ram_bytes / 2**30:.1f} GiB")
            if anon > _CPU_ANON_FRACTION * self._ram_bytes:
                self._say("  warning: that leaves little room for the page cache the loader "
                          "streams the checkpoint through. If the machine has no swap it may "
                          "thrash rather than fail cleanly — serve a smaller checkpoint, or "
                          "set VLLM_CPU_KVCACHE_SPACE lower.")
        if "VLLM_USE_FLASHINFER_SAMPLER" in env:
            self._say("  the NVIDIA CUDA Toolkit is not installed, and FlashInfer ships no "
                      "prebuilt sampler for this GPU, so it")
            self._say("  cannot build one — falling back to vLLM's built-in sampler. "
                      "Please install the CUDA Toolkit for the faster path:")
            self._say("  https://developer.nvidia.com/cuda-downloads")
        # Its own session: the engine core is a spawned grandchild, and the group is what
        # holds the VRAM — stop() signals the group, which only works if we lead one.
        self.proc = self._spawn(self.argv(), stdout=log,
                                stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
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
        """End the child's whole process group, escalating to SIGKILL. A server we merely
        attached to is left alone — we did not start it, so it is not ours to end.

        The group, not just the child: vLLM's engine core is a spawned grandchild, and a
        SIGTERM to the API server alone left an EngineCore parked in a weight download
        holding 17 GiB — while this method printed "the GPU is free again". The group is
        the unit that holds VRAM, so the group is the unit stop() ends.
        """
        proc, self.proc = self.proc, None
        if proc is None:
            return
        self._say("stopping vLLM — the GPU is free again")
        self._signal_tree(proc, hard=False)
        try:
            proc.wait(timeout=20)
        except Exception:                # noqa: BLE001 - TimeoutExpired, or a fake's stand-in
            self._signal_tree(proc, hard=True)
            try:
                proc.wait(timeout=10)
            except Exception:            # noqa: BLE001 - nothing further we can do
                pass

    def _signal_tree(self, proc, *, hard: bool) -> None:
        """Signal the child's process group if it leads one; fall back to the child alone.

        The leadership check keeps this safe for anything not spawned by start() with its
        own session — never signal a group we do not own.
        """
        sig = signal.SIGKILL if hard else signal.SIGTERM
        try:
            if self._getpgid(proc.pid) == proc.pid:
                self._killpg(proc.pid, sig)
                return
        except Exception:                # noqa: BLE001 - no such process, or a fake's stand-in
            pass
        proc.kill() if hard else proc.terminate()

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
        last_progress = started
        last_report = started
        dl_baseline = self._safe_download_bytes()
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
            # The timeout is a NO-PROGRESS window, not a stopwatch. A first run downloads
            # the weights, and a download writes nothing to the log — measured on a 26B:
            # 15 minutes parked in snapshot_download, perfectly healthy, and a fixed
            # deadline shot it. Log output and download growth both count as progress; a
            # true hang still ends after `timeout` seconds of neither.
            if self._last_output_at is not None:
                last_progress = max(last_progress, self._last_output_at)
            dl = self._safe_download_bytes()
            if dl > dl_baseline:
                dl_baseline = dl
                self._dl_seen = dl
                self._dl_at = now
                last_progress = now
            if now - last_progress >= self.timeout:
                hint = ""
                if any("unauthenticated requests" in ln for ln in self._tail):
                    hint = ("\nThe weight download is running unauthenticated, which HF "
                            "rate-limits hard — set HF_TOKEN in the environment and retry.")
                raise RuntimeError(
                    f"vLLM made no progress for {self.timeout:.0f}s — no log output and "
                    f"no weight-download movement.\n" + self._drain() + hint)
            if now - last_report >= self.report_every:
                self._report(now - started)
                last_report = now
            self._sleep(2.0)

    def _safe_download_bytes(self) -> int:
        try:
            return int(self._download_bytes())
        except Exception:                # noqa: BLE001 - a broken probe must not kill the wait
            return 0

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
        # A silent log with a growing cache is the most common healthy state of a first
        # run — say what is actually happening instead of quoting a stale line. But the
        # download owns the line only while it is the FRESHEST signal: measured, the old
        # unconditional label froze at "17.8 GiB so far" through the whole load phase.
        if self._dl_seen and (self._last_output_at is None
                              or (self._dl_at or 0) >= self._last_output_at):
            if self.weights_bytes and self._dl_seen >= 0.98 * self.weights_bytes:
                latest = (f"weights downloaded "
                          f"({self.weights_bytes / 2**30:.1f} GiB) — the engine is "
                          f"loading them")
            elif self.weights_bytes:
                latest = (f"downloading weights: {self._dl_seen / 2**30:.1f} / "
                          f"{self.weights_bytes / 2**30:.1f} GiB")
            else:
                latest = f"downloading weights: {self._dl_seen / 2**30:.1f} GiB so far"
            quiet = ""
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
            if self.device == "cpu":
                hint = ("\n\nThe model plus its KV pool did not fit in RAM. Retry with a "
                        "smaller pool, e.g.\n"
                        "  VLLM_CPU_KVCACHE_SPACE=2 glq-chat\n"
                        "or pick a smaller checkpoint.")
            else:
                hint = (f"\n\nThe model did not fit in {self.gpu_memory_utilization:.0%} of this "
                        f"GPU. Retry with a larger share, e.g.\n"
                        f"  glq-chat --gpu-memory-utilization "
                        f"{min(self.gpu_memory_utilization + 0.2, 0.95):.2f}")
        return f"{tail}{hint}\n\nFull log: {self.log_path}"
