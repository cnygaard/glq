"""Standardized throughput (tok/s) via ``vllm bench throughput`` (the user's
chosen cross-GPU measure). Runs as its own subprocess (vLLM bench manages its own
engine), parses output tok/s from stdout. ``kind="throughput"``, not part of the
quality index.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

from ..record import BenchmarkResult, ServingMeta, ThroughputResult
from .parse import parse_vllm_bench_throughput


def _resolve_vllm() -> str:
    """Find the ``vllm`` CLI. Prefer the one next to the running interpreter (so it
    works when the venv's bin/ isn't on a non-interactive shell's PATH), then PATH."""
    cand = os.path.join(os.path.dirname(sys.executable), "vllm")
    if os.path.exists(cand):
        return cand
    found = shutil.which("vllm")
    if found:
        return found
    raise RuntimeError("`vllm` CLI not found (looked next to the Python executable "
                       f"at {cand!r} and on PATH)")


def run(ctx, config: dict):
    il = int(config.get("input_len", 1024))
    ol = int(config.get("output_len", 256))
    npr = int(config.get("num_prompts", 64))
    timeout = int(config.get("timeout", 3600))

    cmd = [_resolve_vllm(), "bench", "throughput", "--model", ctx.model,
           "--input-len", str(il), "--output-len", str(ol),
           "--num-prompts", str(npr), "--trust-remote-code"]
    if ctx.quant and ctx.quant not in ("none", "bf16"):
        cmd += ["--quantization", ctx.quant]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    parsed = parse_vllm_bench_throughput((proc.stdout or "") + "\n" + (proc.stderr or ""))
    out_tps = parsed["output_tok_s"]

    res = BenchmarkResult(
        task=config.get("task_name", "throughput"), metric="tokens_per_s",
        value=out_tps if out_tps is not None else None,
        standardized=False,
        config={"input_len": il, "output_len": ol, "num_prompts": npr,
                "tool": "vllm bench throughput"},
        extra={"rc": proc.returncode, "total_tok_s": parsed["total_tok_s"],
               "ok": out_tps is not None,
               "stderr_tail": (proc.stderr or "")[-500:] if proc.returncode else None})
    tp = ThroughputResult(output_tok_s=out_tps, batch=npr, measure="vllm_bench_throughput")
    ctx.standalone_serving = ServingMeta(runtime="vllm", command=" ".join(cmd))
    return res, tp
