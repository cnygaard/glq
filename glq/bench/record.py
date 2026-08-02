"""The unified benchmark record schema + JSON(L) (de)serialization.

One ``BenchRecord`` == one (model x benchmark-task x GPU x runtime x run). Records
are appended as one-JSON-object-per-line (JSONL) to the centralized results repo,
so the schema is intentionally flat-nested and stdlib-only (no torch/vllm import),
which keeps record handling — and its tests — CPU-only.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any

SCHEMA_VERSION = 1

# Recognized quantization-method tags (``none`` == unquantized bf16/fp16 baseline).
QUANT_METHODS = ("none", "glq", "modelopt", "gptq", "awq", "bnb", "fp8", "int8")


def utcnow_iso() -> str:
    """UTC timestamp, second precision, e.g. ``2026-06-20T08:30:00Z``."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ModelMeta:
    """Identity + provenance of the model under test."""
    id: str                                  # HF repo id or local dir path
    hf_url: str | None = None                # https://huggingface.co/<id> (None for local)
    base_model: str | None = None            # parent model it was quantized from
    base_hf_url: str | None = None
    quant_method: str | None = None          # one of QUANT_METHODS
    bpw: float | None = None                 # average bits-per-weight (None for bf16)
    bpw_min: int | None = None               # per-layer floor (mixed precision)
    bpw_max: int | None = None               # per-layer ceiling
    is_mixed: bool | None = None
    weights_disk_gb: float | None = None     # on-disk safetensors size
    architecture: str | None = None          # config.architectures[0]


@dataclass
class EnvMeta:
    """Library + toolchain versions (reproducibility)."""
    python: str | None = None
    torch: str | None = None
    vllm: str | None = None
    transformers: str | None = None
    sglang: str | None = None
    triton: str | None = None
    cuda: str | None = None                  # torch.version.cuda
    driver: str | None = None                # nvidia driver
    glq: str | None = None
    glq_git_sha: str | None = None


@dataclass
class HardwareMeta:
    """The GPU(s) the run executed on."""
    gpu_model: str | None = None             # e.g. "NVIDIA RTX PRO 6000 Blackwell ..."
    gpu_count: int | None = None
    gpu_total_vram_gib: float | None = None  # per-GPU total VRAM


@dataclass
class ServingMeta:
    """How the model was served + memory at load."""
    runtime: str = "vllm"                    # vllm | sglang | hf
    command: str | None = None               # reconstructed `vllm serve ...` string
    llm_kwargs: dict[str, Any] | None = None  # exact LLM(**kwargs) used
    dtype: str | None = None
    kv_cache_dtype: str | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    load_gpu_mem_gib: float | None = None     # weights+ctx resident right after load


@dataclass
class BenchmarkResult:
    """The benchmark outcome. ``value`` is the headline metric (e.g. accuracy 0..1)."""
    task: str                                # mmlu_pro | aime_2024 | aime_2026 | wikitext2_ppl | throughput | ...
    metric: str                              # accuracy | perplexity | tokens_per_s | ...
    value: float
    standardized: bool = False               # counts toward the weighted quality index
    config: dict[str, Any] = field(default_factory=dict)   # n, budget, thinking, seed, sampling
    extra: dict[str, Any] = field(default_factory=dict)    # correct, n, truncated, no_answer, mean_gen_tokens, ci95


@dataclass
class ThroughputResult:
    """Decode/serving throughput (tok/s), when measured."""
    output_tok_s: float | None = None
    input_tok_s: float | None = None
    batch: int | None = None
    measure: str | None = None               # vllm_bench_throughput | in_run_tqdm | ...


@dataclass
class BenchRecord:
    """One benchmark run, fully self-describing."""
    model: ModelMeta
    benchmark: BenchmarkResult
    env: EnvMeta = field(default_factory=EnvMeta)
    hardware: HardwareMeta = field(default_factory=HardwareMeta)
    serving: ServingMeta = field(default_factory=ServingMeta)
    throughput: ThroughputResult | None = None
    timestamp_utc: str = field(default_factory=utcnow_iso)
    schema_version: int = SCHEMA_VERSION

    # ---- serialization ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        """Single-line JSON (JSONL row)."""
        return json.dumps(self.to_dict(), default=str, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BenchRecord":
        return _from_dict(cls, d)

    @classmethod
    def from_json(cls, line: str) -> "BenchRecord":
        return cls.from_dict(json.loads(line))


# Nested-dataclass reconstruction: only pass keys the dataclass declares (forward-
# compatible with newer-schema records that add fields) and recurse into the typed
# dataclass / Optional[dataclass] sub-objects.
_SUBOBJECTS: dict[str, type] = {
    "model": ModelMeta, "env": EnvMeta, "hardware": HardwareMeta,
    "serving": ServingMeta, "benchmark": BenchmarkResult, "throughput": ThroughputResult,
}


def _from_dict(cls: type, d: dict[str, Any]) -> Any:
    if not isinstance(d, dict):
        return d
    field_names = {f.name for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for k, v in d.items():
        if k not in field_names:
            continue                          # drop unknown (future) keys
        sub = _SUBOBJECTS.get(k)
        if sub is not None and isinstance(v, dict):
            kwargs[k] = _from_dict(sub, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


def write_jsonl(records: list[BenchRecord], path, *, append: bool = False) -> None:
    """Write records to a JSONL file (one object per line).

    Overwrites by default (least-surprising for ``write_*``); pass ``append=True``
    to accumulate. The results store (``store.append``) reads + merges + overwrites,
    so it controls history explicitly rather than relying on append mode.
    """
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json() + "\n")


def read_jsonl(path) -> list[BenchRecord]:
    """Read all records from a JSONL file, skipping blank/comment lines."""
    out: list[BenchRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(BenchRecord.from_json(line))
    return out
