"""Decode speed per concurrency (B=1, B=32, ...) via ``vllm bench sweep serve``.

Deliberately not hand-rolled. The README's published single-stream figures and the
committed ``benchmarks/cmp_results_l4/`` data both came from this exact tool driven by
``benchmarks/_cmp_sweep.sh``; measuring it a different way here would put two incompatible
definitions of "tok/s" in one comparison table. Using the maintained tool also keeps the
numbers comparable with anyone else's vLLM benchmarks.

Two rates come out of each concurrency and they answer different questions:

* ``per_stream_tok_s = 1000 / mean_tpot_ms`` — time *per output token*, so prefill is
  already excluded. This is what "decode at B=1" means and what bf16 is compared against.
* ``output_throughput`` — aggregate tokens/s across all concurrent streams, the
  serving-capacity number.

Distinct from the ``throughput`` task, which shells out to ``vllm bench throughput`` for a
single aggregate rate over a fixed prompt count *including* prefill. Both are kept: old
records stay comparable, and each record's ``config.measure`` says which it is.

``kind="throughput"`` — the sweep starts and stops its own server, so it cannot share the
quality tasks' in-process engine.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

from ..record import BenchmarkResult, ServingMeta, ThroughputResult

# Same arms as benchmarks/_cmp_sweep.sh, so a task record and a hand-run sweep are the
# same measurement: random 128/256, ignore-eos, fixed seed.
_INPUT_LEN = 128
_OUTPUT_LEN = 256
_SEED = 42
# num_prompts per concurrency — enough requests that the tail doesn't dominate, matching
# benchmarks/_cmp_bench_params.json.
_PROMPTS = {1: 16, 4: 32, 8: 64, 16: 96, 32: 128}


def _resolve_vllm() -> str:
    """Find the ``vllm`` CLI. Prefer the one next to the running interpreter (so it works
    when the venv's bin/ isn't on a non-interactive shell's PATH), then PATH."""
    cand = os.path.join(os.path.dirname(sys.executable), "vllm")
    if os.path.exists(cand):
        return cand
    found = shutil.which("vllm")
    if found:
        return found
    raise RuntimeError("`vllm` CLI not found (looked next to the Python executable "
                       f"at {cand!r} and on PATH)")


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _parse_summary(path: str) -> dict[int, dict]:
    """summary.csv -> {concurrency: {...}}, averaging the ``--num-runs`` repeats."""
    rows: dict[int, list[dict]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                conc = int(float(row["max_concurrency"]))
            except (KeyError, TypeError, ValueError):
                continue
            rows.setdefault(conc, []).append(row)

    def num(row, key):
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            return None

    out = {}
    for conc, rs in rows.items():
        tpot = _mean([num(r, "mean_tpot_ms") for r in rs])
        out[conc] = {
            "runs": len(rs),
            "output_throughput": round(_mean([num(r, "output_throughput") for r in rs]), 1)
            if _mean([num(r, "output_throughput") for r in rs]) is not None else None,
            # 1/TPOT is the per-token decode rate with prefill already out of it.
            "per_stream_tok_s": round(1000.0 / tpot, 1) if tpot else None,
            "mean_tpot_ms": round(tpot, 2) if tpot else None,
            "mean_ttft_ms": round(_mean([num(r, "mean_ttft_ms") for r in rs]), 1)
            if _mean([num(r, "mean_ttft_ms") for r in rs]) is not None else None,
        }
    return out


def run(ctx, config: dict):
    concurrencies = [int(c) for c in config.get("concurrencies", [1, 32])]
    num_runs = int(config.get("num_runs", 3))
    max_model_len = int(config.get("max_model_len", 2048))
    timeout = int(config.get("timeout", 7200))
    port = int(config.get("port", 8321))

    # vLLM's default gpu_memory_utilization (0.9) sizes the KV cache to fill the card AFTER
    # weights, which leaves nothing for the warm-up lm_head logits GEMM on a small GPU. On a
    # 22 GiB L4 a 1.8 GiB SmolLM3-3B checkpoint allocated 260,848 KV tokens (~20 GiB) and
    # then OOMed asking for 64 MiB — the sweep reported only "terminated early", because the
    # server's own stderr never surfaces. Expose the knob so 24-32 GB cards (the ones GLQ
    # exists for) are benchmarkable at all; None keeps vLLM's default.
    gpu_mem = config.get("gpu_memory_utilization")
    serve_extra = config.get("serve_extra", "")

    # Fail BEFORE the measurement, not after: vLLM's sweep tool imports pandas only at
    # the very end (writing summary.csv), so a missing dep otherwise discards a completed
    # ~40-minute run. Fresh venvs lack it — vllm[bench] is an extra, and only the quantize
    # extra pulls it transitively.
    import importlib.util
    if importlib.util.find_spec("pandas") is None:
        raise RuntimeError(
            "decode_sweep needs pandas in this venv (vllm's sweep tool imports it when "
            "writing summary.csv — at the END of the run). Install it first: "
            "pip install pandas")

    vllm = _resolve_vllm()
    serve_flags = f"--max-model-len {max_model_len} --port {port}"
    if ctx.quant and ctx.quant not in ("none", "bf16"):
        serve_flags += f" --quantization {ctx.quant}"
    if gpu_mem is not None:
        serve_flags += f" --gpu-memory-utilization {float(gpu_mem)}"
    if serve_extra:
        serve_flags += f" {serve_extra}"

    with tempfile.TemporaryDirectory() as tmp:
        params_path = os.path.join(tmp, "bench_params.json")
        with open(params_path, "w") as fh:
            json.dump([{"max_concurrency": c,
                        "num_prompts": _PROMPTS.get(c, max(16, c * 4))}
                       for c in concurrencies], fh)

        exp = "decode_sweep"
        cmd = [vllm, "bench", "sweep", "serve",
               "--serve-cmd", f"{vllm} serve {ctx.model} {serve_flags}",
               "--bench-cmd", (f"{vllm} bench serve --base-url http://localhost:{port} "
                               f"--dataset-name random --random-input-len {_INPUT_LEN} "
                               f"--random-output-len {_OUTPUT_LEN} --ignore-eos "
                               f"--seed {_SEED} --model {ctx.model}"),
               "--bench-params", params_path,
               "--num-runs", str(num_runs),
               "--server-ready-timeout", "600",
               "-o", tmp, "-e", exp]
        # The venv's bin/ first on PATH. `vllm` is invoked by absolute path, which does not
        # put that directory on PATH the way `source activate` would — and vLLM's own
        # dependencies shell out to siblings by name. Measured on an RTX PRO 6000: the
        # sweep died because FlashInfer JIT-compiles its sm_120 sampler and ran `ninja`,
        # which was installed at ~/.glq/venv/bin/ninja and simply not on the child's PATH.
        bindir = os.path.dirname(sys.executable)
        env = {**os.environ,
               "PATH": os.pathsep.join(
                   [bindir, *[p for p in os.environ.get("PATH", "").split(os.pathsep)
                              if p and p != bindir]])}
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, check=False, env=env)

        summary = os.path.join(tmp, exp, "summary.csv")   # serve.py writes this
        per_conc = _parse_summary(summary) if os.path.exists(summary) else {}

    if not per_conc:
        # 400 characters of stderr lost the actual cause more than once: vLLM's sweep
        # wrapper ends with "The script was terminated early", and the reason the server
        # died is thousands of lines earlier, in stdout.
        raise RuntimeError(
            f"vllm bench sweep serve produced no summary.csv (rc={proc.returncode}).\n"
            f"--- stderr tail ---\n{(proc.stderr or '')[-2000:]}\n"
            f"--- stdout tail ---\n{(proc.stdout or '')[-2000:]}")

    # Headline is single-stream when it was asked for — the number that decides whether a
    # quantized model is usable interactively.
    head_c = 1 if 1 in per_conc else min(per_conc)
    head = per_conc[head_c]["per_stream_tok_s"]

    res = BenchmarkResult(
        task=config.get("task_name", "decode_sweep"), metric="tokens_per_s", value=head,
        standardized=False,                      # GPU-dependent: never in the quality index
        config={"concurrencies": concurrencies, "num_runs": num_runs,
                "input_len": _INPUT_LEN, "output_len": _OUTPUT_LEN, "seed": _SEED,
                "max_model_len": max_model_len,
                # Recorded because they change the KV-cache size, hence scheduling, hence
                # the number: two records at different settings must not look identical.
                "gpu_memory_utilization": gpu_mem,
                "serve_extra": serve_extra or None,
                "measure": "vllm_bench_sweep_serve/1000-over-mean_tpot_ms",
                "tool": "vllm bench sweep serve"},
        extra={"rc": proc.returncode, "headline_concurrency": head_c,
               "per_concurrency": {str(k): v for k, v in sorted(per_conc.items())}})
    tp = ThroughputResult(output_tok_s=head, batch=head_c,
                          measure="vllm_bench_sweep_serve")
    ctx.standalone_serving = ServingMeta(runtime="vllm", command=" ".join(cmd))
    return res, tp
