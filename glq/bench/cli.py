"""``glq-bench`` command-line entry point.

Subcommands:
  run       run benchmark task(s) on a model (vLLM) -> records [+ push]
  index     compute the % -of-bf16 weighted quality index from records
  report    render a markdown performance table from records
  compare   side-by-side table for an explicit set of models
  plot      render performance plots (matplotlib)
  pull/push sync the centralized results repo

Heavy deps (vllm, matplotlib, git) are imported lazily inside each handler so
``glq-bench index``/``report`` run on a CPU box with only the [bench] extras.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_records(args) -> list:
    """Records come from an explicit --records JSONL, or the synced results repo."""
    from .record import read_jsonl
    if getattr(args, "records", None):
        return read_jsonl(args.records)
    from . import store
    return store.load_all(args.results_repo, results_dir=args.results_dir)


def _load_weights(path: str | None) -> dict:
    if not path:
        return {}
    import json
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(text) or {}
    return json.loads(text)


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
def _cmd_run(args) -> int:
    from . import runner
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    recs = runner.run(
        model=args.model, tasks=tasks, quant=args.quant, runtime=args.runtime,
        n=args.n, budget=args.budget, avg_k=args.avg_k, gpu_mem_util=args.gpu_mem_util,
        max_model_len=args.max_model_len, hf_token=None,
    )
    from .record import write_jsonl
    out = args.out or "bench_records.jsonl"
    write_jsonl(recs, out)
    print(f"wrote {len(recs)} record(s) -> {out}", flush=True)
    if args.push:
        from . import store
        store.append_and_push(recs, args.results_repo, results_dir=args.results_dir)
        print(f"pushed {len(recs)} record(s) to {args.results_repo}", flush=True)
    return 0


def _cmd_index(args) -> int:
    from . import index as index_mod
    from . import report
    recs = _load_records(args)
    table = report.index_table(recs, weights=_load_weights(args.weights),
                               only_standardized=not args.all_tasks)
    _emit(table, args.out)
    return 0


def _cmd_report(args) -> int:
    from . import report
    recs = _load_records(args)
    filt = dict(kv.split("=", 1) for kv in (args.filter or []))
    table = report.perf_table(recs, filters=filt)
    _emit(table, args.out)
    return 0


def _cmd_compare(args) -> int:
    from . import report
    recs = _load_records(args)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    table = report.compare_table(recs, models=models, tasks=tasks)
    _emit(table, args.out)
    return 0


def _cmd_leaderboard(args) -> int:
    from . import report
    from .record import utcnow_iso
    recs = _load_records(args)
    md = report.leaderboard(recs, weights=_load_weights(args.weights),
                            generated_utc=None if args.no_timestamp else utcnow_iso())
    _emit(md, args.out or "LEADERBOARD.md")
    return 0


def _cmd_plot(args) -> int:
    from . import plot
    recs = _load_records(args)
    paths = plot.render(recs, kind=args.kind, out_dir=args.out or "plots")
    print("wrote:\n  " + "\n  ".join(str(p) for p in paths), flush=True)
    return 0


def _cmd_pull(args) -> int:
    from . import store
    store.pull(args.results_repo, results_dir=args.results_dir)
    print(f"pulled {args.results_repo}", flush=True)
    return 0


def _cmd_push(args) -> int:
    from . import store
    store.push(args.results_repo, results_dir=args.results_dir)
    print(f"pushed {args.results_repo}", flush=True)
    return 0


def _emit(text: str, out: str | None) -> None:
    if not out or out == "-":
        print(text)
    else:
        Path(out).write_text(text, encoding="utf-8")
        print(f"wrote {out}", flush=True)


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="glq-bench",
                                description="Generalized model-performance benchmarking toolkit.")
    p.add_argument("--results-repo", default=None,
                   help="git URL or local path of the centralized results repo "
                        "(default: $GLQ_BENCH_RESULTS_REPO).")
    p.add_argument("--results-dir", default=None,
                   help="local checkout dir for the results repo (default: cache).")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run benchmark task(s) on a model")
    r.add_argument("--model", required=True, help="HF repo id or local checkpoint dir")
    r.add_argument("--tasks", required=True, help="comma list: mmlu_pro,aime_2026,throughput,...")
    r.add_argument("--quant", default=None, help="quant method override: glq|none|modelopt|gptq|awq")
    r.add_argument("--runtime", default="vllm", choices=["vllm"])
    r.add_argument("--n", type=int, default=None, help="num samples (quality tasks)")
    r.add_argument("--budget", type=int, default=None, help="max gen tokens / thinking budget")
    r.add_argument("--avg-k", dest="avg_k", type=int, default=1,
                   help="samples per problem for avg@k scoring (aime); default 1 = pass@1")
    r.add_argument("--gpu-mem-util", dest="gpu_mem_util", type=float, default=0.9)
    r.add_argument("--max-model-len", dest="max_model_len", type=int, default=None)
    r.add_argument("--out", default=None, help="local JSONL to write records to")
    r.add_argument("--push", action="store_true", help="also append+push to results repo")
    r.set_defaults(func=_cmd_run)

    for name, fn, helptext in (("index", _cmd_index, "weighted percent-of-bf16 quality index"),
                               ("report", _cmd_report, "markdown performance table")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--records", default=None, help="read records from this JSONL instead of the repo")
        s.add_argument("--out", default=None, help="output file (default stdout)")
        if name == "index":
            s.add_argument("--weights", default=None, help="JSON/YAML task->weight map")
            s.add_argument("--all-tasks", action="store_true", help="include non-standardized tasks")
        if name == "report":
            s.add_argument("--filter", action="append", default=[], help="key=value (e.g. base=google/gemma-4-31B-it)")
        s.set_defaults(func=fn)

    lb = sub.add_parser("leaderboard", help="generate the full LEADERBOARD.md (index + per-base tables)")
    lb.add_argument("--records", default=None, help="read records from this JSONL instead of the repo")
    lb.add_argument("--weights", default=None, help="JSON/YAML task->weight map")
    lb.add_argument("--out", default=None, help="output file (default LEADERBOARD.md; '-' for stdout)")
    lb.add_argument("--no-timestamp", action="store_true", help="omit the generated-on timestamp")
    lb.set_defaults(func=_cmd_leaderboard)

    c = sub.add_parser("compare", help="side-by-side table for explicit models")
    c.add_argument("--records", default=None)
    c.add_argument("--models", required=True, help="comma list of model ids")
    c.add_argument("--tasks", default=None, help="comma list (default: all)")
    c.add_argument("--out", default=None)
    c.set_defaults(func=_cmd_compare)

    pl = sub.add_parser("plot", help="render performance plots")
    pl.add_argument("--records", default=None)
    pl.add_argument("--kind", default="all",
                    help="quality-vs-bpw|pareto|tok-s-by-gpu|index|all")
    pl.add_argument("--out", default=None, help="output dir (default: plots/)")
    pl.set_defaults(func=_cmd_plot)

    for name, fn in (("pull", _cmd_pull), ("push", _cmd_push)):
        s = sub.add_parser(name, help=f"{name} the results repo")
        s.set_defaults(func=fn)

    return p


def main(argv: list[str] | None = None) -> int:
    import os
    args = build_parser().parse_args(argv)
    if getattr(args, "results_repo", None) is None:
        args.results_repo = os.environ.get("GLQ_BENCH_RESULTS_REPO")
    try:
        return args.func(args)
    except NotImplementedError as e:
        print(f"glq-bench: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
