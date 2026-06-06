#!/usr/bin/env python3
"""Multi-model vLLM benchmark harness (GLQ vs bf16 / AWQ / GPTQ).

Orchestrates the *official* ``vllm bench`` tooling over a set of models and
quantization methods, emitting JSON for each run plus a top-level ``index.json``.
Three regimes per model:

  * ``vllm bench latency``    — offline, batch-size sweep (1 = unbatched, N = batched)
  * ``vllm bench throughput`` — offline max throughput (random dataset)
  * ``vllm bench sweep serve``— online request-rate sweep (server auto-managed),
                                hierarchical output usable by ``vllm bench sweep plot``

This script only SHELLS OUT to ``vllm`` (it never imports vllm), so there is no
spawn/``__main__`` hazard and each run is isolated in its own process. Runs are
best-effort: a failed/timed-out/OOM run is recorded and the sweep continues
(covers HF-gated bf16 Gemma, missing AWQ/GPTQ checkpoints, and 31B OOM on small
GPUs).

Quickstart:
    # see the full command matrix without running anything
    python benchmarks/bench_vllm_sweep.py --dry-run

    # run only the SmolLM3-3B GLQ checkpoint, latency at b=1, on a GPU box
    python benchmarks/bench_vllm_sweep.py --only smol3b_glq --modes latency \
        --batch-sizes 1 --output-dir ./vllm_bench_results

    # full default matrix (3 GLQ + 3 bf16), all three modes
    python benchmarks/bench_vllm_sweep.py --output-dir ./vllm_bench_results

Caveats:
  * bf16 ``google/gemma-4-*`` are HF-gated → set HF_TOKEN + accept the license.
  * ``google/gemma-4-31B-it`` and the 31B GLQ need ~96 GB (Blackwell g7e) and the
    CUDA-12.9 stack (sm_120 requires CUDA >= 12.9).
  * AWQ/GPTQ checkpoints for these exact models likely don't exist publicly; add
    your own with --awq REPO / --gptq REPO. Missing ones are skipped, not fatal.
  * Run on a GPU box; this orchestrator itself is GPU-agnostic. Weight-only GLQ
    serving needs NO ``GLQ_KV_*`` envs.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Run specs
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Spec:
    name: str
    model: str
    quant: str | None = None          # "glq" | "awq" | "gptq" | None (bf16)
    dtype: str | None = None          # e.g. "bfloat16" for the bf16 baselines
    engine_extra: list[str] = dataclasses.field(default_factory=list)
    notes: str = ""


def default_specs() -> list[Spec]:
    """3 GLQ checkpoints + their bf16 originals. AWQ/GPTQ added via CLI."""
    return [
        # --- GLQ (weight-only; quantization=glq, no GLQ_KV_* envs) ---
        Spec("gemma_e4b_glq", "xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw", quant="glq"),
        Spec("gemma_31b_glq", "xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8",
             quant="glq", notes="needs ~96GB Blackwell + cu12.9; publish status unverified"),
        Spec("smol3b_glq", "xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw", quant="glq"),
        # --- bf16 originals (HF-gated for Gemma; 31B is huge) ---
        Spec("gemma_e4b_bf16", "google/gemma-4-E4B-it", dtype="bfloat16",
             notes="HF-gated"),
        Spec("gemma_31b_bf16", "google/gemma-4-31B-it", dtype="bfloat16",
             notes="HF-gated; needs ~96GB Blackwell + cu12.9"),
        Spec("smol3b_bf16", "HuggingFaceTB/SmolLM3-3B", dtype="bfloat16"),
    ]


# --------------------------------------------------------------------------- #
# Command builders
# --------------------------------------------------------------------------- #


def engine_args(spec: Spec, args) -> list[str]:
    """Engine-side flags shared by latency/throughput (and the serve server)."""
    out: list[str] = ["--model", spec.model]
    if spec.quant:
        out += ["--quantization", spec.quant]
    if spec.dtype:
        out += ["--dtype", spec.dtype]
    out += ["--max-model-len", str(args.max_model_len),
            "--gpu-memory-utilization", str(args.gpu_mem)]
    out += spec.engine_extra
    return out


def latency_cmd(spec: Spec, batch: int, out_json: Path, args) -> list[str]:
    return [args.vllm, "bench", "latency",
            *engine_args(spec, args),
            "--input-len", str(args.input_len),
            "--output-len", str(args.output_len),
            "--batch-size", str(batch),
            "--num-iters", str(args.num_iters),
            "--num-iters-warmup", str(args.num_iters_warmup),
            "--output-json", str(out_json)]


def throughput_cmd(spec: Spec, out_json: Path, args) -> list[str]:
    return [args.vllm, "bench", "throughput",
            *engine_args(spec, args),
            "--dataset-name", "random",
            "--input-len", str(args.input_len),
            "--output-len", str(args.output_len),
            "--num-prompts", str(args.num_prompts),
            "--output-json", str(out_json)]


def serve_sweep_cmd(spec: Spec, run_dir: Path, bench_params: Path, args) -> list[str]:
    """Official `vllm bench sweep serve` — request-rate sweep, auto-managed server.

    The server config is fixed in --serve-cmd; the request-rate sweep lives in
    the --bench-params JSON (one combo per rate). Hierarchical output under
    run_dir is consumable by `vllm bench sweep plot`.
    """
    quant = f" --quantization {spec.quant}" if spec.quant else ""
    dtype = f" --dtype {spec.dtype}" if spec.dtype else ""
    extra = (" " + " ".join(spec.engine_extra)) if spec.engine_extra else ""
    serve_cmd = (
        f"{args.vllm} serve {spec.model}{quant}{dtype}"
        f" --max-model-len {args.max_model_len}"
        f" --gpu-memory-utilization {args.gpu_mem} --port {args.serve_port}{extra}"
    )
    bench_cmd = (
        f"{args.vllm} bench serve --model {spec.model}"
        f" --base-url http://localhost:{args.serve_port}"
        f" --dataset-name random --random-input-len {args.input_len}"
        f" --random-output-len {args.output_len} --num-prompts {args.num_prompts}"
        f" --max-concurrency {args.max_concurrency} --save-result"
    )
    return [args.vllm, "bench", "sweep", "serve",
            "--serve-cmd", serve_cmd,
            "--bench-cmd", bench_cmd,
            "--bench-params", str(bench_params),
            "--output-dir", str(run_dir),
            "--experiment-name", spec.name,
            "--num-runs", str(args.num_runs),
            "--server-ready-timeout", str(args.server_ready_timeout)]


def write_bench_params(path: Path, rates: list[str]) -> None:
    """--bench-params JSON: one dict per request rate (Cartesian-producted by the
    sweep against the fixed --bench-cmd). 'inf' stays a string."""
    combos = [{"--request-rate": (r if r == "inf" else float(r))} for r in rates]
    path.write_text(json.dumps(combos, indent=2))


# --------------------------------------------------------------------------- #
# Execution + metric parsing
# --------------------------------------------------------------------------- #


def run(cmd: list[str], args, *, cwd: Path | None = None) -> dict:
    """Run one subprocess best-effort. Returns a status dict (never raises)."""
    printable = " ".join(shlex.quote(c) for c in cmd)
    if args.dry_run:
        print(f"  DRY-RUN: {printable}")
        return {"status": "dry-run", "cmd": printable}
    print(f"  RUN: {printable}", flush=True)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=args.timeout)
        st = "ok" if p.returncode == 0 else "failed"
        return {"status": st, "cmd": printable, "returncode": p.returncode,
                "elapsed_s": round(time.time() - t0, 1),
                "stderr_tail": p.stderr[-1500:] if st == "failed" else ""}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "cmd": printable,
                "elapsed_s": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "cmd": printable, "error": f"{type(e).__name__}: {e}"}


def parse_metrics(mode: str, out_path: Path) -> dict:
    """Pull the headline numbers out of a native vLLM bench JSON (best-effort)."""
    try:
        if mode == "latency":
            d = json.loads(out_path.read_text())
            return {"avg_latency_s": d.get("avg_latency"),
                    "p50_s": (d.get("percentiles") or {}).get("50"),
                    "p99_s": (d.get("percentiles") or {}).get("99")}
        if mode == "throughput":
            d = json.loads(out_path.read_text())
            return {"requests_per_s": d.get("requests_per_second"),
                    "tokens_per_s": d.get("tokens_per_second"),
                    "total_tokens": d.get("total_num_tokens")}
    except Exception:  # noqa: BLE001
        pass
    return {}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default="./vllm_bench_results",
                   help="root dir for all JSON + sweep output (default: %(default)s)")
    p.add_argument("--modes", default="latency,throughput,serve",
                   help="comma list of {latency,throughput,serve} (default: %(default)s)")
    p.add_argument("--only", default="",
                   help="comma list of spec names to run (default: all)")
    p.add_argument("--awq", action="append", default=[], metavar="REPO",
                   help="add an AWQ baseline (repeatable): -q awq on REPO")
    p.add_argument("--gptq", action="append", default=[], metavar="REPO",
                   help="add a GPTQ baseline (repeatable): -q gptq on REPO")
    # shapes / sweep params
    p.add_argument("--batch-sizes", default="1,8,32", help="latency batch sweep")
    p.add_argument("--request-rates", default="1,4,16,inf", help="serve rate sweep")
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--num-prompts", type=int, default=200)
    p.add_argument("--num-iters", type=int, default=10)
    p.add_argument("--num-iters-warmup", type=int, default=3)
    p.add_argument("--max-concurrency", type=int, default=64)
    p.add_argument("--num-runs", type=int, default=1, help="serve sweep runs/combo")
    # engine
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--serve-port", type=int, default=8000)
    p.add_argument("--server-ready-timeout", type=int, default=600)
    p.add_argument("--vllm", default="vllm", help="path to the vllm CLI")
    p.add_argument("--timeout", type=int, default=3600, help="per-run seconds")
    p.add_argument("--dry-run", action="store_true",
                   help="print the full command matrix and exit (no execution)")
    args = p.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    rates = [r.strip() for r in args.request_rates.split(",") if r.strip()]

    specs = default_specs()
    for repo in args.awq:
        specs.append(Spec(f"awq_{Path(repo).name}", repo, quant="awq", notes="AWQ baseline"))
    for repo in args.gptq:
        specs.append(Spec(f"gptq_{Path(repo).name}", repo, quant="gptq", notes="GPTQ baseline"))
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        specs = [s for s in specs if s.name in keep]
        if not specs:
            print(f"No specs match --only={args.only}. "
                  f"Available: {[s.name for s in default_specs()]}", file=sys.stderr)
            return 1

    out_root = Path(args.output_dir)
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []

    print(f"== bench matrix: {len(specs)} specs x modes={modes} "
          f"({'DRY-RUN' if args.dry_run else 'LIVE'}) ==")
    for spec in specs:
        sd = out_root / spec.name
        if not args.dry_run:
            sd.mkdir(parents=True, exist_ok=True)
        print(f"\n# {spec.name}  ({spec.model}, q={spec.quant or 'bf16'})"
              f"{'  -- ' + spec.notes if spec.notes else ''}")

        if "latency" in modes:
            for b in batch_sizes:
                oj = sd / f"latency_b{b}.json"
                r = run(latency_cmd(spec, b, oj, args), args)
                r.update(name=spec.name, model=spec.model, quant=spec.quant,
                         mode="latency", batch=b, output_path=str(oj))
                if r["status"] == "ok":
                    r["metrics"] = parse_metrics("latency", oj)
                index.append(r)

        if "throughput" in modes:
            oj = sd / "throughput.json"
            r = run(throughput_cmd(spec, oj, args), args)
            r.update(name=spec.name, model=spec.model, quant=spec.quant,
                     mode="throughput", output_path=str(oj))
            if r["status"] == "ok":
                r["metrics"] = parse_metrics("throughput", oj)
            index.append(r)

        if "serve" in modes:
            run_dir = sd / "serve_sweep"
            bp = sd / "bench_params.json"
            if args.dry_run:
                print(f"  DRY-RUN: write bench_params {bp} -> "
                      f"rates {rates}")
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                write_bench_params(bp, rates)
            r = run(serve_sweep_cmd(spec, run_dir, bp, args), args)
            r.update(name=spec.name, model=spec.model, quant=spec.quant,
                     mode="serve_sweep", request_rates=rates,
                     output_path=str(run_dir))
            index.append(r)

    if not args.dry_run:
        idx_path = out_root / "index.json"
        idx_path.write_text(json.dumps(
            {"args": vars(args), "runs": index}, indent=2))
        ok = sum(1 for r in index if r.get("status") == "ok")
        print(f"\n== done: {ok}/{len(index)} runs ok; index -> {idx_path} ==")
        print("   serve-sweep dashboards: vllm bench sweep plot "
              f"{out_root}/<spec>/serve_sweep")
    else:
        print(f"\n== dry-run: {len(index)} commands above ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
