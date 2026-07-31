"""Benchmark task adapters for glq-bench.

Each adapter takes a loaded model handle + a config and returns a
``(BenchmarkResult, ThroughputResult | None)``. The registry maps task names to
adapters + whether they count toward the standardized quality index.
"""
from __future__ import annotations

from .registry import TASKS, TaskSpec, get_task, list_tasks

__all__ = ["TASKS", "TaskSpec", "get_task", "list_tasks"]
