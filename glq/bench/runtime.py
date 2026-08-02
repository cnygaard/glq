"""Build a vLLM model handle, record the exact serving command, and capture the
weights footprint at load.

vLLM v1 loads weights in an EngineCore *subprocess*, so neither a Python
stdout-redirect nor a parent-process CUDA mem delta sees the weight allocation.
We pin vLLM's logging to INFO-on-stdout via ``VLLM_LOGGING_CONFIG_PATH`` (inherited
by the subprocess), capture file descriptors 1 and 2 (which the subprocess
inherits) around ``LLM(**kwargs)``, parse vLLM's "Model loading took X GiB" line,
and *replay* the captured bytes so the user still sees the logs. If the line is
absent we report ``None`` — never an nvidia-smi delta, which under vLLM measures the
gpu_memory_utilization KV pool (util x total VRAM), not the model weights.

Imports vllm/torch lazily so this module is import-safe on a CPU box (the
command-builder is unit-tested there).
"""
from __future__ import annotations

import json
import os
import shlex
import tempfile
from contextlib import contextmanager

from .provenance import _nvidia_smi
from .record import ServingMeta
from .tasks.parse import parse_load_mem_gib

# Arches whose vLLM path is a multimodal wrapper (gemma-4 unified/E*B/31B etc.):
# serve text-only so the multimodal profiling forward doesn't fire.
_MULTIMODAL_HINTS = ("ConditionalGeneration",)
_TEXT_ONLY_MM = {"image": 0, "video": 0, "audio": 0}
# Small FULL-cudagraph capture set: safe for GLQ MoE (<=256 cap) and quick.
_DEFAULT_CUDAGRAPH = {"cudagraph_mode": "FULL",
                      "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}


def is_multimodal(arch: str | None) -> bool:
    return bool(arch and any(h in arch for h in _MULTIMODAL_HINTS))


def build_llm_kwargs(model: str, *, quant: str | None = None, dtype: str = "bfloat16",
                     max_model_len: int | None = None, gpu_mem_util: float = 0.9,
                     multimodal: bool = False, cudagraph: bool = True) -> dict:
    kw: dict = dict(model=model, dtype=dtype, trust_remote_code=True,
                    gpu_memory_utilization=gpu_mem_util)
    if max_model_len:
        kw["max_model_len"] = max_model_len
    if quant and quant not in ("none", "bf16"):
        kw["quantization"] = quant
    if multimodal:
        kw["limit_mm_per_prompt"] = dict(_TEXT_ONLY_MM)
    if cudagraph:
        kw["compilation_config"] = dict(_DEFAULT_CUDAGRAPH)
    return kw


def serving_command(model: str, kw: dict) -> str:
    """Reconstruct an equivalent ``vllm serve …`` CLI string from LLM kwargs."""
    parts = ["vllm", "serve", model]
    if kw.get("quantization"):
        parts += ["--quantization", str(kw["quantization"])]
    parts += ["--gpu-memory-utilization", str(kw.get("gpu_memory_utilization", 0.9))]
    if kw.get("max_model_len"):
        parts += ["--max-model-len", str(kw["max_model_len"])]
    if kw.get("dtype"):
        parts += ["--dtype", str(kw["dtype"])]
    if kw.get("trust_remote_code"):
        parts += ["--trust-remote-code"]
    if kw.get("limit_mm_per_prompt"):
        parts += ["--limit-mm-per-prompt", json.dumps(kw["limit_mm_per_prompt"])]
    return " ".join(shlex.quote(p) for p in parts)


@contextmanager
def _capture_fd_tee(fd: int = 2):
    """Capture everything written to ``fd`` (incl. by child processes) into a temp
    file, then replay it to the original fd on exit so logs aren't swallowed."""
    saved = os.dup(fd)
    tf = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tf.fileno(), fd)
    try:
        yield tf
    finally:
        try:
            os.fsync(fd)
        except OSError:
            pass
        os.dup2(saved, fd)
        os.close(saved)
        try:
            tf.seek(0)
            os.write(fd, tf.read())          # replay
        finally:
            tf.close()


@contextmanager
def _vllm_logging_to_stdout():
    """Pin vLLM (and its EngineCore subprocess, which inherits the env at spawn) to
    INFO on **stdout** via ``VLLM_LOGGING_CONFIG_PATH``, so the "Model loading took
    X GiB" line is always emitted to a predictable stream regardless of the user's
    ``VLLM_LOGGING_LEVEL``. Uses a stdlib formatter (no python-json-logger dep); the
    GiB regex matches the message whether the line is plain text or a JSON record.
    See https://docs.vllm.ai/en/latest/examples/others/logging_configuration.html
    """
    cfg = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"glqbench": {"format": "%(levelname)s %(name)s %(message)s"}},
        "handlers": {"glqbench": {"class": "logging.StreamHandler",
                                  "formatter": "glqbench", "level": "INFO",
                                  "stream": "ext://sys.stdout"}},
        "loggers": {"vllm": {"handlers": ["glqbench"], "level": "INFO",
                             "propagate": False}},
    }
    tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(cfg, tf)
    tf.close()
    prev = os.environ.get("VLLM_LOGGING_CONFIG_PATH")
    os.environ["VLLM_LOGGING_CONFIG_PATH"] = tf.name
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("VLLM_LOGGING_CONFIG_PATH", None)
        else:
            os.environ["VLLM_LOGGING_CONFIG_PATH"] = prev
        try:
            os.unlink(tf.name)
        except OSError:
            pass


def _gpu_mem_used_gib() -> float | None:
    rows = _nvidia_smi("memory.used")        # MiB
    try:
        return round(float(rows[0]) / 1024, 2) if rows else None
    except (ValueError, IndexError):
        return None


class LoadedModel:
    """A loaded vLLM engine + its captured serving metadata."""
    def __init__(self, llm, serving: ServingMeta, tokenizer=None):
        self.llm = llm
        self.serving = serving
        self.tokenizer = tokenizer


def load(model: str, *, quant: str | None = None, dtype: str = "bfloat16",
         max_model_len: int | None = None, gpu_mem_util: float = 0.9,
         arch: str | None = None, multimodal: bool | None = None) -> LoadedModel:
    """Construct the vLLM engine and capture load footprint + serving command."""
    from vllm import LLM
    if multimodal is None:
        multimodal = is_multimodal(arch)
    kw = build_llm_kwargs(model, quant=quant, dtype=dtype, max_model_len=max_model_len,
                          gpu_mem_util=gpu_mem_util, multimodal=multimodal)

    # vLLM emits "Model loading took X GiB" (the weights footprint) at INFO from the
    # EngineCore subprocess, on *stdout* once we pin the logging config. Capture both
    # fd 1 and fd 2 (the subprocess inherits them) around LLM(), then parse.
    captured = ""
    with _vllm_logging_to_stdout(), _capture_fd_tee(1) as tf_out, _capture_fd_tee(2) as tf_err:
        llm = LLM(**kw)
        tf_out.seek(0)
        tf_err.seek(0)
        captured = (tf_out.read().decode("utf-8", "replace") + "\n"
                    + tf_err.read().decode("utf-8", "replace"))
    # The model-weights footprint comes from vLLM's own report. We deliberately do
    # NOT fall back to an nvidia-smi delta: under vLLM that captures the
    # gpu_memory_utilization KV-cache pool (util x total VRAM), not the weights —
    # a misleading number. Report None rather than something wrong.
    load_mem = parse_load_mem_gib(captured)

    serving = ServingMeta(
        runtime="vllm",
        command=serving_command(model, kw),
        llm_kwargs={k: v for k, v in kw.items() if k != "model"},
        dtype=dtype,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        load_gpu_mem_gib=load_mem,
    )
    try:
        tok = llm.get_tokenizer()
    except Exception:  # noqa: BLE001
        tok = None
    return LoadedModel(llm, serving, tok)
