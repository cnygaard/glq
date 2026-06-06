#!/usr/bin/env python3
"""Quality-eval harness (lm-evaluation-harness on the vLLM backend) + README
benchmark-table generator for GLQ checkpoints vs their bf16 originals.

Recreates the Gemma-4-E4B-it model-card style benchmarks with `lm_eval[vllm]`:
mmlu_pro, gpqa_diamond, aime24, aime25, winogrande. Two modes:

  --mode run    : per model x task-bucket, shell out to `lm_eval --model vllm …`,
                  writing native lm_eval result JSON. Best-effort (a gated/failed
                  task is recorded and skipped).
  --mode table  : aggregate the lm_eval JSONs (+ optional B=1/B=32 tok/s from a
                  bench_tps_vllm RESULT log) into a Markdown table ready to paste
                  into README.md, with HF-model links and a ~% of bf16 column.

This script only SHELLS OUT to `lm_eval` (never imports it / vllm), so runs are
isolated and a crash in one never sinks the rest.

NOT covered: **livecodebench** — it is not an lm-eval task; recreate it via the
standalone LiveCodeBench repo (https://github.com/LiveCodeBench/LiveCodeBench)
with its code-execution sandbox.

Quickstart:
    # see the full lm_eval command matrix without running anything
    python benchmarks/bench_quality_lm_eval.py --mode run --dry-run

    # run one cheap bucket on the 3B GLQ checkpoint (GPU box)
    python benchmarks/bench_quality_lm_eval.py --mode run --only smol3b_glq \
        --buckets core --output-dir ./quality_results

    # build the README table from whatever JSON exists (+ tok/s log)
    python benchmarks/bench_quality_lm_eval.py --mode table \
        --output-dir ./quality_results --tps-log ./tps.log

Caveats:
  * gpqa_diamond uses the gated `Idavidrein/gpqa` dataset → accept its HF license
    and set HF_TOKEN; bf16 google/gemma-4-* are gated too.
  * The 31B (GLQ + bf16) need ~96 GB (Blackwell g7e) + the CUDA-12.9 stack; bf16
    31B is a ~60 GB download. All best-effort / skip-on-failure.
  * AIME runs long (32k-token) generations even at 30 Q; --max-gen-toks caps it.
    --limit subsets are NOT card-faithful — the table labels them.
  * Run on a GPU box (cu12.9 venv for Blackwell). transformers 5.x. Don't share
    the GPU with another running benchmark.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Run specs + task buckets
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Spec:
    name: str
    model: str
    bpw: str
    quant: str | None = None       # "glq" | None (bf16)
    dtype: str | None = None       # "bfloat16" for bf16 baselines
    bf16_of: str | None = None     # name of the bf16 counterpart (for % of bf16)
    engine_extra: list[str] = dataclasses.field(default_factory=list)


def default_specs() -> list[Spec]:
    return [
        Spec("gemma_e4b_glq", "xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw", "4.0",
             quant="glq", bf16_of="gemma_e4b_bf16"),
        Spec("gemma_31b_glq", "xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8",
             "5.0 (mix)", quant="glq", bf16_of="gemma_31b_bf16"),
        Spec("smol3b_glq", "xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw", "3.5 (mix)",
             quant="glq", bf16_of="smol3b_bf16"),
        Spec("gemma_e4b_bf16", "google/gemma-4-E4B-it", "16", dtype="bfloat16"),
        Spec("gemma_31b_bf16", "google/gemma-4-31B-it", "16", dtype="bfloat16"),
        Spec("smol3b_bf16", "HuggingFaceTB/SmolLM3-3B", "16", dtype="bfloat16"),
    ]


# bucket name -> (tasks, uses_mmlu_limit). Grouped by limit policy so the model
# loads once per bucket and a fragile/gated task is isolated.
def buckets(args) -> dict[str, dict]:
    return {
        "mmlu_pro": {"tasks": "mmlu_pro", "limit": args.mmlu_limit},
        "core": {"tasks": "winogrande,aime24,aime25", "limit": args.core_limit},
        "gpqa": {"tasks": args.gpqa_task, "limit": args.core_limit},
    }


# display column -> lm_eval result key (override gpqa via --gpqa-task)
def task_columns(args) -> list[tuple[str, str]]:
    return [("mmlu_pro", "MMLU-Pro"), (args.gpqa_task, "GPQA-D"),
            ("aime24", "AIME24"), ("aime25", "AIME25"),
            ("winogrande", "WinoGrande")]


PRIMARY_METRICS = ("exact_match", "acc_norm", "acc")


# --------------------------------------------------------------------------- #
# run mode
# --------------------------------------------------------------------------- #


def lm_eval_cmd(spec: Spec, tasks: str, limit: float | None, out_dir: Path, args) -> list[str]:
    margs = [f"pretrained={spec.model}",
             f"max_model_len={args.max_model_len}",
             f"gpu_memory_utilization={args.gpu_mem}",
             "trust_remote_code=True"]
    if spec.quant:
        margs.append(f"quantization={spec.quant}")
    if spec.dtype:
        margs.append(f"dtype={spec.dtype}")
    margs += spec.engine_extra
    cmd = [args.lm_eval, "--model", "vllm", "--model_args", ",".join(margs),
           "--tasks", tasks, "--batch_size", args.batch_size,
           "--gen_kwargs", f"max_gen_toks={args.max_gen_toks}",
           "--output_path", str(out_dir)]
    if args.apply_chat_template:
        cmd.append("--apply_chat_template")
    if limit and float(limit) > 0:
        cmd += ["--limit", str(limit)]
    return cmd


def run_one(cmd: list[str], args) -> dict:
    printable = " ".join(shlex.quote(c) for c in cmd)
    if args.dry_run:
        print(f"  DRY-RUN: {printable}")
        return {"status": "dry-run", "cmd": printable}
    print(f"  RUN: {printable}", flush=True)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        st = "ok" if p.returncode == 0 else "failed"
        return {"status": st, "cmd": printable, "returncode": p.returncode,
                "elapsed_s": round(time.time() - t0, 1),
                "stderr_tail": p.stderr[-1500:] if st == "failed" else ""}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "cmd": printable,
                "elapsed_s": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "cmd": printable, "error": f"{type(e).__name__}: {e}"}


def do_run(args, specs: list[Spec]) -> int:
    out_root = Path(args.output_dir)
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
    want_buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]
    bdefs = buckets(args)
    index: list[dict] = []
    print(f"== quality matrix: {len(specs)} specs x buckets={want_buckets} "
          f"({'DRY-RUN' if args.dry_run else 'LIVE'}) ==")
    for spec in specs:
        print(f"\n# {spec.name}  ({spec.model}, q={spec.quant or 'bf16'})")
        for bname in want_buckets:
            if bname not in bdefs:
                print(f"  ! unknown bucket {bname}; skip", file=sys.stderr)
                continue
            bd = bdefs[bname]
            od = out_root / spec.name / bname
            if not args.dry_run:
                od.mkdir(parents=True, exist_ok=True)
            r = run_one(lm_eval_cmd(spec, bd["tasks"], bd["limit"], od, args), args)
            r.update(name=spec.name, model=spec.model, bucket=bname,
                     tasks=bd["tasks"], limit=bd["limit"], output_path=str(od))
            index.append(r)
    if not args.dry_run:
        (out_root / "run_index.json").write_text(
            json.dumps({"args": vars(args), "runs": index}, indent=2))
        ok = sum(1 for r in index if r.get("status") == "ok")
        print(f"\n== run done: {ok}/{len(index)} buckets ok; "
              f"index -> {out_root/'run_index.json'} ==")
        print(f"   build the table: {sys.argv[0]} --mode table "
              f"--output-dir {out_root} [--tps-log <bench_tps_vllm.log>]")
    return 0


# --------------------------------------------------------------------------- #
# table mode
# --------------------------------------------------------------------------- #


def _primary(taskres: dict) -> float | None:
    for p in PRIMARY_METRICS:
        for k, v in taskres.items():
            if k.split(",")[0] == p and isinstance(v, (int, float)):
                return float(v)
    for k, v in taskres.items():            # fallback: first non-stderr numeric
        if "stderr" not in k and isinstance(v, (int, float)):
            return float(v)
    return None


def _collect_results(out_root: Path) -> dict[str, dict[str, float]]:
    """model_repo -> {result_task_key: primary_metric}, merged over all bucket
    JSONs lm_eval wrote under out_root."""
    by_model: dict[str, dict[str, float]] = {}
    for jf in glob.glob(str(out_root / "**" / "results*.json"), recursive=True):
        try:
            d = json.loads(Path(jf).read_text())
        except Exception:  # noqa: BLE001
            continue
        margs = str((d.get("config") or {}).get("model_args", ""))
        m = re.search(r"pretrained=([^,]+)", margs)
        model = m.group(1) if m else (d.get("model_name") or jf)
        dst = by_model.setdefault(model, {})
        for task, res in (d.get("results") or {}).items():
            if isinstance(res, dict):
                val = _primary(res)
                if val is not None:
                    dst[task] = val
    return by_model


def _parse_tps(log_path: str | None) -> dict[str, dict[int, float]]:
    """bench_tps_vllm RESULT lines -> model -> {batch: total_decode_tokps}."""
    out: dict[str, dict[int, float]] = {}
    if not log_path or not Path(log_path).exists():
        return out
    pat = re.compile(r"RESULT.*model=(\S+).*batch=(\d+).*total_decode_tokps=([\d.]+)")
    for line in Path(log_path).read_text().splitlines():
        mm = pat.search(line)
        if mm:
            out.setdefault(mm.group(1), {})[int(mm.group(2))] = float(mm.group(3))
    return out


def _cell(v: float | None, pct: bool = True) -> str:
    if v is None:
        return "—"
    return f"{v*100:.1f}" if pct else f"{v:.0f}"


def do_table(args, specs: list[Spec]) -> int:
    out_root = Path(args.output_dir)
    results = _collect_results(out_root)
    tps = _parse_tps(args.tps_log)
    cols = task_columns(args)
    by_name = {s.name: s for s in specs}

    def link(repo: str) -> str:
        return f"[`{repo}`](https://huggingface.co/{repo})"

    header = "| Model | bpw | " + " | ".join(c[1] for c in cols) + \
             " | Tok/s B1 | Tok/s B32 | ~% bf16 |"
    sep = "|" + "---|" * (4 + len(cols))
    lines = [header, sep]
    for spec in specs:
        r = results.get(spec.model, {})
        qcells = [_cell(r.get(tkey)) for tkey, _ in cols]
        t = tps.get(spec.model, {})
        b1 = _cell(t.get(1), pct=False)
        b32 = _cell(t.get(32), pct=False)
        # ~% of bf16: mean per-task ratio vs the bf16 counterpart (GLQ rows only)
        pct = "—"
        if spec.bf16_of and spec.bf16_of in by_name:
            base = results.get(by_name[spec.bf16_of].model, {})
            ratios = [r[tk] / base[tk] for tk, _ in cols
                      if r.get(tk) and base.get(tk)]
            if ratios:
                pct = f"{sum(ratios) / len(ratios) * 100:.1f}%"
        lines.append(f"| {link(spec.model)} | {spec.bpw} | "
                     + " | ".join(qcells) + f" | {b1} | {b32} | {pct} |")

    note = ("\n<sub>Quality = lm-evaluation-harness (vLLM backend), "
            "chat-template + thinking; values are %% accuracy/exact-match. "
            "MMLU-Pro run with `--limit` (subset, not card-faithful) unless noted; "
            "GPQA-D/AIME/WinoGrande full. Tok/s = single-stream (B1) / 32-way (B32) "
            "decode on the same GPU (see `bench_tps_vllm.py`). ~%% bf16 = mean "
            "per-task ratio vs the bf16 original. `—` = not run / unavailable.</sub>")
    md = "\n".join(lines) + "\n" + note
    print(md)
    if not args.dry_run:
        (out_root / "quality_table.md").write_text(md + "\n")
        (out_root / "quality_index.json").write_text(json.dumps(
            {"results": results, "tps": {k: v for k, v in tps.items()}}, indent=2))
        print(f"\n# wrote {out_root/'quality_table.md'} + quality_index.json",
              file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["run", "table"], default="run")
    p.add_argument("--output-dir", default="./quality_results")
    p.add_argument("--only", default="", help="comma list of spec names")
    p.add_argument("--buckets", default="mmlu_pro,core,gpqa",
                   help="comma list of {mmlu_pro,core,gpqa}")
    p.add_argument("--gpqa-task", default="gpqa_diamond_cot_zeroshot",
                   help="lm_eval gpqa task name (version-dependent)")
    p.add_argument("--mmlu-limit", type=float, default=0.1,
                   help="--limit for mmlu_pro (fraction or count); 0 = full")
    p.add_argument("--core-limit", type=float, default=0.0,
                   help="--limit for core/gpqa buckets (0 = full)")
    p.add_argument("--max-gen-toks", type=int, default=16384,
                   help="cap generation length for CoT/AIME (cost control)")
    p.add_argument("--batch-size", default="auto")
    p.add_argument("--apply-chat-template", action="store_true", default=True)
    p.add_argument("--no-chat-template", dest="apply_chat_template",
                   action="store_false")
    p.add_argument("--max-model-len", type=int, default=20480)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--lm-eval", default="lm_eval", help="path to lm_eval CLI")
    p.add_argument("--timeout", type=int, default=21600, help="per-bucket seconds")
    p.add_argument("--tps-log", default=None,
                   help="(table mode) bench_tps_vllm RESULT log for B1/B32 tok/s")
    p.add_argument("--no-baselines", action="store_true",
                   help="with --only, do NOT auto-add bf16 counterparts (GLQ-only)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    specs = default_specs()
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        # keep bf16 counterparts so the table can compute % bf16 (unless GLQ-only)
        if not args.no_baselines:
            keep |= {by.bf16_of for by in specs if by.name in keep and by.bf16_of}
        specs = [s for s in specs if s.name in keep]
        if not specs:
            print(f"No specs match --only={args.only}. "
                  f"Available: {[s.name for s in default_specs()]}", file=sys.stderr)
            return 1

    return do_run(args, specs) if args.mode == "run" else do_table(args, specs)


if __name__ == "__main__":
    raise SystemExit(main())
