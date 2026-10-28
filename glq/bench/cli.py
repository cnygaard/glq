"""``glq-bench`` command-line entry point.

Subcommands:
  run       run benchmark task(s) on a model (vLLM) -> records [+ push]
  index     compute the % -of-bf16 weighted quality index from records
  report    render a markdown performance table from records
  compare   side-by-side table for an explicit set of models
  leaderboard  LEADERBOARD.md, structured by the HF collections
  collections  the HF collections LEADERBOARD.md follows (JSON)
  card      put the per-GPU speed/quality table into a model's HF card
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
    import json

    from . import runner
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    task_config = json.loads(args.task_config) if args.task_config else None
    recs = runner.run(
        model=args.model, tasks=tasks, quant=args.quant, runtime=args.runtime,
        n=args.n, budget=args.budget, avg_k=args.avg_k, gpu_mem_util=args.gpu_mem_util,
        max_num_seqs=args.max_num_seqs, dtype=args.dtype,
        max_model_len=args.max_model_len, kv_cache_dtype=args.kv_cache_dtype,
        kv_transfer_config=args.kv_transfer_config,
        cpu_offload_gb=args.cpu_offload_gb,
        hf_token=None, task_config=task_config,
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


def _collections(args):
    """``(start_here, families)`` from a ``--collections`` snapshot, else live from the Hub."""
    from . import hfcollections
    if getattr(args, "collections", None):
        return hfcollections.from_json(Path(args.collections).read_text(encoding="utf-8"))
    return hfcollections.fetch_collections()


def _cmd_leaderboard(args) -> int:
    from . import report
    from .record import utcnow_iso
    recs = _load_records(args)
    # The structure IS the collections. Without them there is no faithful file to write, and
    # writing a differently shaped one would publish it as if nothing had changed.
    try:
        start, families = _collections(args)
    except Exception as e:                                           # noqa: BLE001
        print(f"glq-bench: could not read the HF collections ({e}); LEADERBOARD.md not "
              f"written. Pass --collections FILE to use a saved snapshot.", file=sys.stderr)
        return 2
    md = report.leaderboard(recs, start_here=start, families=families,
                            generated_utc=None if args.no_timestamp else utcnow_iso())
    _emit(md, args.out or "LEADERBOARD.md")
    return 0


def _cmd_collections(args) -> int:
    """Print or save the collections the leaderboard is structured by."""
    from . import hfcollections
    try:
        start, families = hfcollections.fetch_collections()
    except Exception as e:                                           # noqa: BLE001
        print(f"glq-bench: could not read the HF collections: {e}", file=sys.stderr)
        return 2
    _emit(hfcollections.to_json(start, families), args.out)
    return 0


def _cmd_card(args) -> int:
    """Splice the per-GPU table into a model's HF card. A dry run unless --push."""
    import difflib
    import os

    from . import report
    if args.push and args.card_file:
        print("glq-bench: --push reads the card from the Hub, so it cannot overwrite edits "
              "made there since your local copy; drop --card-file", file=sys.stderr)
        return 2
    recs = _load_records(args)
    try:
        section = report.card_section(recs, args.model)
        if args.card_file:
            old = Path(args.card_file).read_text(encoding="utf-8")
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(args.model, "README.md", force_download=True,
                                   token=os.environ.get("HF_TOKEN"))
            old = Path(path).read_text(encoding="utf-8")
        new = report.splice_card(old, section)
    except ValueError as e:
        print(f"glq-bench: {e}", file=sys.stderr)
        return 2

    if new == old:
        print("card already up to date", flush=True)
    else:
        sys.stdout.writelines(difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            "README.md (current)", "README.md (with glq-bench section)"))
    if args.out:
        _emit(new, args.out)
    if args.push and new != old:
        from huggingface_hub import HfApi
        HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
            path_or_fileobj=new.encode("utf-8"), path_in_repo="README.md",
            repo_id=args.model, commit_message="Model card: speed and quality by GPU")
        print(f"pushed README.md to {args.model}", flush=True)
    elif not args.push:
        print("(dry run — nothing uploaded; re-run with --push to publish)", flush=True)
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
def _json_arg(raw: str):
    """argparse type for a JSON object, so malformed input fails at parse time.

    argparse turns a ValueError/TypeError from a type callable into a clean usage error; any
    other exception would traceback, so the json error is re-raised as ValueError.
    """
    import json as _json
    try:
        return _json.loads(raw)
    except _json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {e}") from None


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
    r.add_argument("--max-num-seqs", dest="max_num_seqs", type=int, default=64,
                   help="engine concurrency. On hybrid-GDN models every decode slot "
                        "reserves a Mamba cache block up front, so this is the knob that "
                        "relieves 'max_num_seqs exceeds available Mamba cache blocks' "
                        "when gpu-mem-util has no room left. Lower also shrinks the "
                        "per-step logits buffer, which matters on large vocabularies.")
    r.add_argument("--max-model-len", dest="max_model_len", type=int, default=None)
    r.add_argument("--dtype", default=None,
                   help="engine dtype for vLLM-backed tasks (float16|bfloat16|float32). "
                        "Default: glq.tooling.preferred_dtype for the model. "
                        "runtime.build_llm_kwargs always accepted this; it simply was "
                        "not reachable, so every vLLM-backed number was bf16 whether or "
                        "not that was the right dtype for the model.")
    # An engine argument, not an env var: selecting a KV dtype any other way
    # runs bf16 and reports it as a KV-quantization result.
    r.add_argument("--kv-cache-dtype", dest="kv_cache_dtype", default=None,
                   help="fp8 | turboquant_4bit_nc | ... (default: engine default)")
    # WEIGHT offload, not the KV offload below. This exists so a model larger than the
    # card can still be measured: Kolibri-1 bf16 is 145.5 GiB against 96 GiB, so without
    # it the GLQ perplexity number has no baseline and cannot be interpreted. Use it for
    # prefill-bound tasks (perplexity); a DECODE rate measured under offload describes
    # PCIe, not the quantization.
    r.add_argument("--cpu-offload-gb", dest="cpu_offload_gb", type=float, default=0,
                   help="GiB of WEIGHTS to map into host RAM so an oversized model "
                        "fits (vLLM --cpu-offload-gb). Needed for a bf16 baseline of a "
                        "model bigger than the card. Note --gpu-mem-util is a share of "
                        "the WHOLE card, so the KV pool grows into the VRAM this frees; "
                        "size the two together. Recorded in ServingMeta.")
    # Parsed by argparse, not by the runner: a typo in a long JSON blob should fail here and
    # not four minutes into an engine start. vLLM's own recipes configure CPU/disk KV offload
    # through this, e.g. the OffloadingConnector config published for Qwen3.8-Flash-Next.
    r.add_argument("--kv-transfer-config", dest="kv_transfer_config", default=None,
                   type=_json_arg,
                   help='JSON passed to vLLM as --kv-transfer-config, e.g. CPU KV offload: '
                        '\'{"kv_connector": "OffloadingConnector", "kv_role": "kv_both", '
                        '"kv_connector_extra_config": {"cpu_bytes_to_use": 68719476736}}\'. '
                        'Recorded in ServingMeta and in the reconstructed serve command, '
                        'because it changes both the memory topology and the speed of a run.')
    r.add_argument("--task-config", dest="task_config", default=None,
                   help='JSON merged into every task config, last word. Model-level knobs '
                        'live here because they belong to the chat template, not the task '
                        '— e.g. SmolLM3 needs \'{"system": null, "temperature": 0.6, '
                        '"top_k": null}\' or it silently runs no-think.')
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

    lb = sub.add_parser("leaderboard", help="generate LEADERBOARD.md: recommended checkpoints, "
                                            "then one section per family collection")
    lb.add_argument("--records", default=None, help="read records from this JSONL instead of the repo")
    lb.add_argument("--collections", default=None,
                    help="read the collections from this snapshot (`glq-bench collections "
                         "--out`) instead of the Hub")
    lb.add_argument("--out", default=None, help="output file (default LEADERBOARD.md; '-' for stdout)")
    lb.add_argument("--no-timestamp", action="store_true", help="omit the generated-on timestamp")
    lb.set_defaults(func=_cmd_leaderboard)

    cl = sub.add_parser("collections", help="show the HF collections LEADERBOARD.md follows "
                                            "(JSON; a snapshot for --collections)")
    cl.add_argument("--out", default=None, help="output file (default stdout)")
    cl.set_defaults(func=_cmd_collections)

    cd = sub.add_parser("card", help="put the per-GPU speed/quality table into a model's HF "
                                     "card (prints a diff; uploads only with --push)")
    cd.add_argument("--model", required=True, help="HF repo id of the model whose card to update")
    cd.add_argument("--records", default=None, help="read records from this JSONL instead of the repo")
    cd.add_argument("--card-file", dest="card_file", default=None,
                    help="read the current card from this file instead of the Hub (offline review)")
    cd.add_argument("--out", default=None, help="also write the new card to this file")
    cd.add_argument("--push", action="store_true",
                    help="upload the new README.md to the model repo (token from HF_TOKEN)")
    cd.set_defaults(func=_cmd_card)

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
