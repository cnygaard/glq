"""Seed the glq-bench results repo with the gemma-4 numbers measured by hand this
session (RTX PRO 6000, vLLM 0.23.0, thinking, pass@1). One record per
(model x task). Only metrics actually measured are filled in; unmeasured tok/s /
VRAM are left None (honest).

    python scripts/seed_bench_records.py --out seed.jsonl            # write JSONL
    python scripts/seed_bench_records.py --push --results-repo <url> # also push

These become the first comparable records; re-quantized / re-run models append
alongside via `glq-bench run`.
"""
from __future__ import annotations

import argparse

from glq.bench import (BenchmarkResult, BenchRecord, EnvMeta, HardwareMeta,
                       ModelMeta, ServingMeta, ThroughputResult)
from glq.bench.hfmeta import _hf_url
from glq.bench.record import write_jsonl

ENV = EnvMeta(python="3.12.0", torch="2.11.0+cu128", vllm="0.23.0",
              transformers="5.10.0", glq="0.6.3")
HW = HardwareMeta(gpu_model="NVIDIA RTX PRO 6000 Blackwell Server Edition",
                  gpu_count=1, gpu_total_vram_gib=95.6)

B31 = "google/gemma-4-31B-it"
B12 = "google/gemma-4-12B-it"
B4 = "google/gemma-4-E4B-it"

# (model, base, quant, bpw, task, acc, vram_gib, out_tok_s, ts)
ROWS = [
    # ---- 31B (paired GLQ vs bf16, + NVFP4) ----
    (B31, None, "none", None, "mmlu_pro", 0.867, 57.9, None, "2026-06-19T16:50:00Z"),
    (B31, None, "none", None, "aime_2024", 0.867, 57.9, 228.9, "2026-06-19T19:09:00Z"),
    (B31, None, "none", None, "aime_2026", 0.900, 57.9, 188.6, "2026-06-19T21:28:00Z"),
    ("xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8", B31, "glq", 5.0, "mmlu_pro", 0.900, 16.5, None, "2026-06-19T16:22:00Z"),
    ("xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8", B31, "glq", 5.0, "aime_2024", 0.900, 16.5, None, "2026-06-19T18:52:00Z"),
    ("xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8", B31, "glq", 5.0, "aime_2026", 0.833, 16.5, 68.7, "2026-06-19T21:06:00Z"),
    ("nvidia/Gemma-4-31B-IT-NVFP4", B31, "modelopt", 4.0, "aime_2026", 0.933, 30.1, 430.6, "2026-06-19T22:14:00Z"),
    # ---- 12B (paired GLQ vs bf16) ----
    (B12, None, "none", None, "mmlu_pro", 0.783, None, 585.0, "2026-06-19T11:28:00Z"),
    (B12, None, "none", None, "aime_2024", 0.933, None, 375.0, "2026-06-19T11:48:00Z"),
    ("xv0y5ncu/Gemma-4-12B-it-GLQ-5.0bpw", B12, "glq", 5.0, "mmlu_pro", 0.817, 6.94, 187.0, "2026-06-19T10:19:00Z"),
    ("xv0y5ncu/Gemma-4-12B-it-GLQ-5.0bpw", B12, "glq", 5.0, "aime_2024", 0.833, 6.94, 160.0, "2026-06-19T11:13:00Z"),
    # ---- E4B (no bf16 baseline run -> no index, only absolute scores) ----
    ("xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw", B4, "glq", 4.0, "mmlu_pro", 0.750, 5.1, None, "2026-06-19T15:23:00Z"),
    ("xv0y5ncu/Gemma-4-E4B-it-GLQ-3.5bpw-mix3-5", B4, "glq", 3.5, "mmlu_pro", 0.700, 5.23, None, "2026-06-19T15:30:00Z"),
]

_BUDGET = {"mmlu_pro": 16384, "aime_2024": 32768, "aime_2026": 65536}


def build_records() -> list[BenchRecord]:
    out = []
    for mid, base, quant, bpw, task, acc, vram, toks, ts in ROWS:
        out.append(BenchRecord(
            model=ModelMeta(id=mid, hf_url=_hf_url(mid), base_model=base,
                            base_hf_url=_hf_url(base), quant_method=quant, bpw=bpw,
                            architecture="Gemma4ForConditionalGeneration"),
            benchmark=BenchmarkResult(task=task, metric="accuracy", value=acc,
                                      standardized=True,
                                      config={"thinking": True, "budget": _BUDGET.get(task)}),
            env=ENV, hardware=HW,
            serving=ServingMeta(runtime="vllm", load_gpu_mem_gib=vram,
                                gpu_memory_utilization=0.9),
            throughput=(ThroughputResult(output_tok_s=toks, measure="in_run_chat")
                        if toks is not None else None),
            timestamp_utc=ts))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed glq-bench with this session's numbers.")
    ap.add_argument("--out", default="seed_bench_records.jsonl")
    ap.add_argument("--push", action="store_true", help="also append+push to the results repo")
    ap.add_argument("--results-repo", default=None)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    recs = build_records()
    write_jsonl(recs, args.out)
    print(f"wrote {len(recs)} records -> {args.out}")
    if args.push:
        import os
        from glq.bench import store
        repo = args.results_repo or os.environ.get("GLQ_BENCH_RESULTS_REPO")
        store.append_and_push(recs, repo, results_dir=args.results_dir)
        print(f"pushed {len(recs)} records -> {repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
