"""glq.bench — generalized model-performance benchmarking toolkit.

Runs model benchmarks (quality + throughput), captures full reproducibility
provenance (exact serving command, library/GPU versions, GPU memory at load,
tokens/sec, base model, quantization method, HF links), and appends structured
records to a durable centralized git results repo. Produces comparison tables,
plots, and a weighted "% of bf16" quality index across models, quant methods,
and GPUs.

The capture/record/index/report layers are import-safe without vLLM or a GPU
(only ``glq-bench run`` needs them); this keeps the unit tests CPU-only.

CLI entry point: ``glq-bench`` (see ``glq.bench.cli``).
"""
from __future__ import annotations

from .record import (
    SCHEMA_VERSION,
    BenchmarkResult,
    BenchRecord,
    EnvMeta,
    HardwareMeta,
    ModelMeta,
    ServingMeta,
    ThroughputResult,
    utcnow_iso,
)

__all__ = [
    "SCHEMA_VERSION",
    "BenchRecord",
    "ModelMeta",
    "EnvMeta",
    "HardwareMeta",
    "ServingMeta",
    "BenchmarkResult",
    "ThroughputResult",
    "utcnow_iso",
]
