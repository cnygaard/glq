#!/usr/bin/env python3
"""Batch GLQ quantizer — drive ``glq-quantize`` over many models, one config each.

Edit the ``JOBS`` dict below (model id -> quant params) and run; the driver shells
out to the ``glq-quantize`` CLI (``glq/quantize_model.py``) with the right flags
per model, auto-names each output dir, and resumes a partially-finished batch.

Data structure (the easy-to-read surface):

    JOBS = {
        "google/gemma-4-12B-it":     {"bpw": 5.0, "min_bpw": 4, "max_bpw": 8},
        "google/gemma-4-26B-A4B-it": {"bpw": 5.0, "min_bpw": 4, "max_bpw": 8,
                                      "streaming": True},
    }

  * key   = HF model id (or a plain label if you add ``"model": <id>`` to the value,
            so the same base model can appear at two bpws).
  * value = named params, each mapping 1:1 to a ``glq-quantize`` flag. Only ``bpw``
            is required; everything else takes a default. A mistyped key raises.
  * bpw semantics: an *integer* bpw with no min/max  -> uniform quant (1 pass).
                   a fractional bpw, or any min/max  -> mixed precision (2 passes:
                   the avg target is ``bpw``; the allocator stays within [min,max]).

Mixed precision is TWO ``glq-quantize`` invocations (the CLI itself splits them):
    pass 1 (profile):  --bpw <avg> --min-bpw <m> --max-bpw <M>
                       -> writes <dir>/bpw_allocation.json and exits (no model saved)
    pass 2 (quantize): --bpw-map <dir>/bpw_allocation.json
                       -> re-quantizes each layer at its allocated bpw and saves
This driver auto-chains both so you never juggle the allocation JSON. Mixed jobs
therefore cost ~2x a uniform job.

Quickstart:
    # see the exact command matrix without running anything
    python scripts/batch_quantization.py --dry-run

    # quantize just one entry, into ./glq_out/<auto-name>/
    python scripts/batch_quantization.py --only gemma-4-12B --output-dir ./glq_out

    # resume a long batch: already-finished dirs are skipped (use --force to redo)
    python scripts/batch_quantization.py --output-dir /opt/dlami/nvme/glq_out

Caveats:
  * Runs jobs strictly SEQUENTIALLY (no parallel GPU work).
  * Gated bf16 bases (e.g. ``google/gemma-4-*``) need HF_TOKEN + accepted license;
    big models (12B/26B/31B) need ``streaming=True`` and a 24-96 GB GPU. Failures
    are recorded and the batch continues (best-effort).
  * On remote boxes, launch under ``nohup``; per-job live logs land in
    ``<dir>/quant.log`` (unbuffered) and a summary in ``<output-dir>/batch_index.json``.

HuggingFace upload (``--upload``): each model is uploaded **private first**, then a quick
vLLM **smoke test** runs (capital-of-France etc.); on pass the repo is flipped **public**
(``update_repo_settings``), on fail it is kept private for inspection. Progress is recorded
incrementally to ``<output-dir>/batch_progress.json``, and a model is treated as **done**
(skipped on a re-run) once its repo is **public** on HF — so a batch is resumable at
whole-model granularity after a spot-VM termination (HF is the durable record; mid-quant
resume is NOT supported — an interrupted quant is redone from scratch). ``--no-smoke`` flips
straight to public; ``--private``/per-job ``private:true`` keep a model private (and that
private repo then counts as its done state).
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
from dataclasses import field
from datetime import datetime, timezone
from pathlib import Path

HF_NAMESPACE = "xv0y5ncu"  # default upload namespace (override with --hf-namespace)

# Completion-style smoke prompts (greedy decode works for base AND instruct models).
# Each is (prompt, expected_substring); the "Balanced" gate requires a hit on >=1.
SMOKE_PROMPTS = [
    ("The capital of France is", "paris"),
    ("The opposite of hot is", "cold"),
    ("Q: What is 2+2?\nA:", "4"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# --------------------------------------------------------------------------- #
# Job list  ──  EDIT ME
# --------------------------------------------------------------------------- #

JOBS: dict[str, dict] = {
    # Gemma-4-E4B-it across the bpw range + a mixed-precision example. The 4bpw
    # quant is already published, so the on-HF skip auto-excludes it (and any
    # other already-published quant, e.g. sarvam-30b-GLQ-4bpw).
    #   key = a label (the same base appears at several bpws, so "model" is set);
    #   value = glq-quantize params (only "bpw" required), + optional "private".
    "gemma-4-e4b@2bpw":  {"model": "google/gemma-4-e4b-it", "bpw": 2},
    "gemma-4-e4b@3bpw":  {"model": "google/gemma-4-e4b-it", "bpw": 3},
    "gemma-4-e4b@4bpw":  {"model": "google/gemma-4-e4b-it", "bpw": 4},
    "gemma-4-e4b@mix5":  {"model": "google/gemma-4-e4b-it", "bpw": 5.0,
                          "min_bpw": 3, "max_bpw": 8},  # avg 5.0, per-layer 3..8
}


# --------------------------------------------------------------------------- #
# Normalized job
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class QuantJob:
    """One quantization job. Built from a JOBS entry; fields map 1:1 to
    ``glq-quantize`` flags (see ``glq/quantize_model.py:main``)."""
    model: str                        # HF id or local path (defaults to the dict key)
    bpw: float                        # avg target if mixed; uniform if integer & no min/max
    min_bpw: int | None = None        # mixed-precision floor (2..8)
    max_bpw: int | None = None        # mixed-precision ceiling (2..8)
    codebook_size: int | None = None  # E8 entries (default 65536; 4096 = Blackwell smem)
    nsamples: int = 128               # CLAUDE.md mandate
    seqlen: int = 2048
    tune_iters: int = 0
    streaming: bool = False           # big / FP8 models (~30B+)
    trust_remote_code: bool = False
    device: str = "cuda"
    output: str | None = None         # output dir name; auto-named from model+bpw if None
    private: bool | None = None        # per-job HF visibility (None -> batch default)
    extra_args: list = field(default_factory=list)  # escape hatch -> raw glq-quantize flags
    name: str = ""                    # label for --only / index (set during normalization)

    def __post_init__(self) -> None:
        for b in (self.min_bpw, self.max_bpw):
            if b is not None and b not in (2, 3, 4, 5, 6, 7, 8):
                raise ValueError(f"{self.model}: min/max bpw must be in 2..8, got {b}")
        if (self.min_bpw is not None and self.max_bpw is not None
                and self.min_bpw > self.max_bpw):
            raise ValueError(
                f"{self.model}: min_bpw {self.min_bpw} > max_bpw {self.max_bpw}")
        if self.bpw <= 0:
            raise ValueError(f"{self.model}: bpw must be > 0, got {self.bpw}")
        self.is_mixed = (
            self.min_bpw is not None or self.max_bpw is not None
            or (isinstance(self.bpw, float) and not float(self.bpw).is_integer()))
        if (self.is_mixed and self.min_bpw is not None and self.max_bpw is not None
                and not (self.min_bpw <= float(self.bpw) <= self.max_bpw)):
            raise ValueError(
                f"{self.model}: avg bpw {self.bpw} not within "
                f"[{self.min_bpw}, {self.max_bpw}]")


def jobs_from_dict(d: dict) -> list[QuantJob]:
    """Normalize the authored ``JOBS`` dict into validated ``QuantJob`` objects.

    A bad param key (e.g. ``min_bwp``) raises ``TypeError`` here rather than being
    silently ignored — that's the whole point of the dataclass layer.
    """
    out: list[QuantJob] = []
    for key, params in d.items():
        p = dict(params)
        model = p.pop("model", key)
        try:
            job = QuantJob(model=model, **p)
        except TypeError as e:
            raise SystemExit(
                f"JOBS['{key}']: bad parameter — {e}. "
                f"Valid keys: {[f.name for f in dataclasses.fields(QuantJob) if f.name != 'name']}")
        job.name = key
        out.append(job)
    return out


# --------------------------------------------------------------------------- #
# Output naming + command builders
# --------------------------------------------------------------------------- #


def output_name(job: QuantJob) -> str:
    """Auto dir name matching the published convention (Gemma-4-31B-it-GLQ-5.0bpw-mix3-8)."""
    if job.output:
        return job.output
    base = job.model.rstrip("/").split("/")[-1]
    if job.is_mixed:
        mn = job.min_bpw if job.min_bpw is not None else 2
        mx = job.max_bpw if job.max_bpw is not None else 4
        return f"{base}-GLQ-{float(job.bpw):.1f}bpw-mix{mn}-{mx}"
    return f"{base}-GLQ-{int(job.bpw)}bpw"


def _shared_flags(job: QuantJob, device: str) -> list[str]:
    f = ["--nsamples", str(job.nsamples),
         "--seqlen", str(job.seqlen),
         "--device", device]
    if job.tune_iters:
        f += ["--tune-iters", str(job.tune_iters)]
    if job.codebook_size is not None:
        f += ["--codebook-size", str(job.codebook_size)]
    if job.streaming:
        f += ["--streaming"]
    if job.trust_remote_code:
        f += ["--trust-remote-code"]
    f += list(job.extra_args)
    return f


def commands_for(job: QuantJob, out_dir: Path, quantize_cmd: str,
                 device: str) -> list[tuple[str, list[str]]]:
    """Return [(pass_label, argv), ...] — 1 cmd for uniform, 2 for mixed precision."""
    base = shlex.split(quantize_cmd) + ["--model", job.model, "--output", str(out_dir)]
    sh = _shared_flags(job, device)
    if not job.is_mixed:
        return [("quant", base + ["--bpw", str(int(job.bpw))] + sh)]
    # mixed precision -> profile then re-quantize with the produced allocation
    p1 = base + ["--bpw", str(job.bpw)]
    if job.min_bpw is not None:
        p1 += ["--min-bpw", str(job.min_bpw)]
    if job.max_bpw is not None:
        p1 += ["--max-bpw", str(job.max_bpw)]
    p1 += sh
    alloc = str(out_dir / "bpw_allocation.json")
    p2 = base + ["--bpw-map", alloc] + sh
    return [("profile", p1), ("quantize", p2)]


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def _tail(path: Path, n: int = 1500) -> str:
    try:
        return path.read_text(errors="replace")[-n:]
    except Exception:  # noqa: BLE001
        return ""


def run_pass(label: str, cmd: list[str], log_path: Path, args) -> dict:
    """Run one glq-quantize pass best-effort. Streams output live to log_path."""
    printable = " ".join(shlex.quote(c) for c in cmd)
    if args.dry_run:
        print(f"    DRY-RUN [{label}]: {printable}")
        return {"pass": label, "status": "dry-run", "cmd": printable}
    print(f"    RUN [{label}]: {printable}", flush=True)
    t0 = time.time()
    try:
        with open(log_path, "a", buffering=1) as logf:
            logf.write(f"\n===== {label}: {printable} =====\n")
            logf.flush()
            p = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                               timeout=args.timeout)
        st = "ok" if p.returncode == 0 else "failed"
        rec = {"pass": label, "status": st, "returncode": p.returncode,
               "elapsed_s": round(time.time() - t0, 1), "cmd": printable}
        if st == "failed":
            rec["log_tail"] = _tail(log_path)
            print(f"      -> FAILED (rc={p.returncode}); see {log_path}", flush=True)
        else:
            print(f"      -> ok ({rec['elapsed_s']}s)", flush=True)
        return rec
    except subprocess.TimeoutExpired:
        print(f"      -> TIMEOUT after {args.timeout}s; see {log_path}", flush=True)
        return {"pass": label, "status": "timeout",
                "elapsed_s": round(time.time() - t0, 1), "cmd": printable,
                "log_tail": _tail(log_path)}
    except Exception as e:  # noqa: BLE001
        return {"pass": label, "status": "error",
                "error": f"{type(e).__name__}: {e}", "cmd": printable}


def _matches(job: QuantJob, out_dir_name: str, tokens: set[str]) -> bool:
    hay = f"{job.name} {job.model} {out_dir_name}"
    return any(t in hay for t in tokens)


# --------------------------------------------------------------------------- #
# HuggingFace: skip already-published quants, generate card, upload
# --------------------------------------------------------------------------- #


def existing_hf_quants(namespace: str, token: str | None) -> dict[str, bool | None]:
    """Map ``lower(repo_name) -> private`` for every model under ``namespace`` on HF.
    ``private`` is True/False (populated by the authenticated list) or None if unknown.
    Returns {} on any error (then nothing is skipped on the HF side)."""
    try:
        from huggingface_hub import HfApi
        return {m.id.split("/")[-1].lower(): getattr(m, "private", None)
                for m in HfApi(token=token).list_models(author=namespace)}
    except Exception as e:  # noqa: BLE001
        print(f"  (could not list {namespace} on HF: {e}; not skipping on HF)", flush=True)
        return {}


def is_done(out_name: str, job_name: str, target_private: bool,
            published: dict, ledger: dict) -> bool:
    """A model is 'done' (skip it) once it has reached its TARGET visibility on HF.

    - public target: done iff the repo is **public** on HF.
    - private target: done iff the repo **exists** (private is the intended end state).
    A repo that exists but is private while we wanted public is NOT done (it was uploaded
    earlier but never validated/released → re-run, re-smoke, flip). HF is authoritative;
    the local ledger ('done') is a same-VM fast path that survives a script crash.
    """
    vis = published.get(out_name.lower())  # None=absent, True=private, False=public
    if vis is False:                       # public on HF -> done regardless of target
        return True
    if vis is True and target_private:     # private on HF and we wanted private -> done
        return True
    return ledger.get(job_name, {}).get("status") == "done"


# --------------------------------------------------------------------------- #
# Smoke test: load the quantized checkpoint in vLLM and judge coherence.
# The judge is a PURE function (unit-tested); the vLLM load runs in a SUBPROCESS
# (--_smoke-dir mode) so its CUDA/vLLM state is torn down before any --bench run.
# --------------------------------------------------------------------------- #


def judge_smoke_output(samples) -> tuple[bool, str]:
    """``samples``: list of ``(prompt, expected_or_None, text)``. Balanced criterion:
    hard-fail on all-empty output or a degenerate repetition loop; pass requires a
    known-answer substring hit on >=1 factual prompt."""
    import re
    from collections import Counter
    any_text = False
    hit = False
    for _prompt, expected, text in samples:
        t = (text or "").strip()
        if t:
            any_text = True
        words = re.findall(r"\w+", t.lower())
        if len(words) >= 8:  # only judge repetition on long-enough output
            top = Counter(words).most_common(1)[0][1]
            if top / len(words) > 0.5:
                return False, f"degenerate repetition: {t[:80]!r}"
        if expected and expected.lower() in t.lower():
            hit = True
    if not any_text:
        return False, "all prompts produced empty output"
    if not hit:
        return False, "no known-answer matched (possible garbage/incoherent output)"
    return True, "ok"


def _run_smoke_worker(out_dir: Path, max_tokens: int) -> int:
    """In-subprocess: load the GLQ checkpoint in vLLM, generate, print a JSON verdict.
    Exit 0=pass, 1=judged-fail, 2=unavailable/error."""
    try:
        import glq_vllm  # noqa: F401  registers the "glq" quant method
        from vllm import LLM, SamplingParams
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": f"vllm/glq_vllm unavailable: {e}"}), flush=True)
        return 2
    try:
        llm = LLM(model=str(out_dir), quantization="glq", dtype="bfloat16",
                  trust_remote_code=True, max_model_len=2048,
                  gpu_memory_utilization=0.7, enforce_eager=True)
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        outs = llm.generate([p for p, _ in SMOKE_PROMPTS], sp)
        samples = [(SMOKE_PROMPTS[i][0], SMOKE_PROMPTS[i][1], outs[i].outputs[0].text)
                   for i in range(len(SMOKE_PROMPTS))]
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": f"generate failed: {type(e).__name__}: {e}"}), flush=True)
        return 2
    ok, reason = judge_smoke_output(samples)
    print(json.dumps({"ok": ok, "reason": reason,
                      "samples": [{"prompt": p, "text": t} for p, _, t in samples]}),
          flush=True)
    return 0 if ok else 1


def run_smoke(out_dir: Path, max_tokens: int, timeout: int = 3600) -> dict:
    """Parent-side: run the smoke worker as a subprocess (isolated vLLM), parse verdict.
    status in {passed, failed, unavailable, timeout}."""
    cmd = [sys.executable, os.path.abspath(__file__),
           "--_smoke-dir", str(out_dir), "--smoke-max-tokens", str(max_tokens)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"passed": False, "status": "timeout", "reason": f"smoke timed out ({timeout}s)"}
    info: dict = {}
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                info = json.loads(line)
                break
            except Exception:  # noqa: BLE001
                pass
    if proc.returncode == 0 and info.get("ok"):
        return {"passed": True, "status": "passed", "reason": info.get("reason", "ok"),
                "samples": info.get("samples")}
    status = "unavailable" if proc.returncode == 2 else "failed"
    reason = info.get("error") or info.get("reason") or f"rc={proc.returncode}"
    return {"passed": False, "status": status, "reason": reason,
            "samples": info.get("samples"), "stderr_tail": proc.stderr[-800:]}


# --------------------------------------------------------------------------- #
# Progress ledger: model-granular, resumable across spot-VM restarts.
# (HF-public is the authoritative durable record; this file is a same-VM convenience.)
# --------------------------------------------------------------------------- #


def load_ledger(path: Path) -> dict:
    try:
        return (json.loads(path.read_text()) or {}).get("jobs", {})
    except Exception:  # noqa: BLE001 — absent / corrupt -> start fresh
        return {}


def record_progress(path: Path, namespace: str, ledger: dict, name: str, **fields) -> None:
    """Update one job's record and atomically rewrite the whole ledger (temp + replace)."""
    job = ledger.setdefault(name, {})
    job.update(fields)
    job["updated"] = _now()
    doc = {"updated": _now(), "namespace": namespace, "jobs": ledger}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, path)


def run_benchmarks(out_dir: Path, limit, token: str | None) -> list | None:
    """Quick MMLU-Pro + Winogrande via lm-eval on vLLM over the local dir. Opt-in
    (--bench). Writes benchmarks.json and returns the rows, or None if unavailable."""
    try:
        import glq_vllm  # noqa: F401  registers the "glq" quant method
        import lm_eval
    except Exception as e:  # noqa: BLE001
        print(f"    --bench: lm_eval/vllm unavailable ({e}); skipping benchmarks", flush=True)
        return None
    margs = (f"pretrained={out_dir},quantization=glq,dtype=bfloat16,"
             f"trust_remote_code=True,max_model_len=8192,"
             f"gpu_memory_utilization=0.7,enforce_eager=True")
    try:
        res = lm_eval.simple_evaluate(
            model="vllm", model_args=margs, tasks=["mmlu_pro", "winogrande"],
            limit=limit, apply_chat_template=True, fewshot_as_multiturn=True,
            batch_size="auto")
    except Exception as e:  # noqa: BLE001
        print(f"    --bench: eval failed ({e}); skipping benchmarks", flush=True)
        return None
    r = (res or {}).get("results", {})
    rows: list = []

    def _row(task, key, alts, label, metric):
        m = r.get(task) or {}
        val = m.get(key)
        for a in alts:
            if val is None:
                val = m.get(a)
        if val is not None:
            rows.append({"task": label, "metric": metric,
                         "n": m.get("sample_len"), "value": f"{float(val) * 100:.1f}%"})

    _row("mmlu_pro", "exact_match,custom-extract", [], "MMLU-Pro", "exact_match")
    _row("winogrande", "acc,none", ["acc"], "Winogrande", "acc")
    (out_dir / "benchmarks.json").write_text(json.dumps(rows, indent=2))
    return rows


def upload_private(out_dir: Path, repo_id: str, base_model: str,
                   token: str | None, benchmarks) -> None:
    """Generate the GLQ card (README.md) then create a PRIVATE repo + upload the folder.
    Always private first — a model is only flipped public after it passes the smoke test."""
    from glq.model_card import build_card
    from huggingface_hub import HfApi
    build_card(out_dir, base_model, repo_id=repo_id, benchmarks=benchmarks, hf_token=token)
    api = HfApi(token=token)
    api.create_repo(repo_id, private=True, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="model",
                      commit_message="GLQ quantized model + auto-generated card")


def set_repo_public(repo_id: str, token: str | None) -> None:
    """Flip an existing repo from private to public (after a passing smoke test)."""
    from huggingface_hub import HfApi
    HfApi(token=token).update_repo_settings(repo_id=repo_id, private=False,
                                            repo_type="model")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default="./glq_out",
                   help="root dir for all quantized model dirs (default: %(default)s)")
    p.add_argument("--config", default=None,
                   help="JSON file with a jobs dict (same shape as JOBS); "
                        "overrides the inline JOBS when given")
    p.add_argument("--only", default="",
                   help="comma list of substrings; keep jobs whose name/model/dir match")
    p.add_argument("--device", default=None,
                   help="override the device for ALL jobs (default: per-job, usually cuda)")
    p.add_argument("--quantize-cmd", default="glq-quantize",
                   help="the glq-quantize entry point (default: %(default)s; e.g. "
                        "'python -m glq.quantize_model' or a venv path)")
    p.add_argument("--timeout", type=int, default=24 * 3600,
                   help="per-pass timeout in seconds (default: %(default)s)")
    p.add_argument("--force", action="store_true",
                   help="re-run jobs/passes even if their outputs already exist")
    p.add_argument("--dry-run", action="store_true",
                   help="print the full command matrix and exit (no execution)")
    # --- HuggingFace upload + skip-existing + benchmarks ------------------- #
    p.add_argument("--hf-namespace", default=HF_NAMESPACE,
                   help="HF namespace for upload + skip-existing (default: %(default)s)")
    p.add_argument("--upload", action="store_true",
                   help="after a successful job, generate a GLQ card and push to HF")
    p.add_argument("--private", dest="private", action="store_true", default=None,
                   help="batch default: upload repos PRIVATE (per-job 'private' overrides)")
    p.add_argument("--public", dest="private", action="store_false",
                   help="batch default: upload repos PUBLIC (the default if neither given)")
    p.add_argument("--no-skip-hf", action="store_true",
                   help="do NOT skip jobs whose quant is already published under the namespace")
    p.add_argument("--bench", action="store_true",
                   help="run a quick MMLU-Pro + Winogrande eval (vLLM) before upload (~6-15 min/model)")
    p.add_argument("--bench-limit", type=float, default=0.02,
                   help="lm-eval per-task fraction/count for --bench (default: %(default)s)")
    p.add_argument("--no-smoke", action="store_true",
                   help="flip uploads to public WITHOUT a smoke test (default: smoke-gate the flip)")
    p.add_argument("--smoke-max-tokens", type=int, default=24,
                   help="tokens to generate per smoke prompt (default: %(default)s)")
    p.add_argument("--progress-file", default=None,
                   help="progress ledger path (default: <output-dir>/batch_progress.json)")
    # hidden: in-subprocess smoke worker (set by run_smoke; loads vLLM and exits 0/1/2)
    p.add_argument("--_smoke-dir", dest="_smoke_dir", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args._smoke_dir:  # subprocess smoke worker: load + judge one checkpoint, exit
        return _run_smoke_worker(Path(args._smoke_dir), args.smoke_max_tokens)

    jobs_src = JOBS
    if args.config:
        jobs_src = json.load(open(args.config))
    jobs = jobs_from_dict(jobs_src)
    if args.only:
        tokens = {t.strip() for t in args.only.split(",") if t.strip()}
        jobs = [j for j in jobs if _matches(j, output_name(j), tokens)]
        if not jobs:
            print(f"No jobs match --only={args.only}. Available: "
                  f"{[j.name for j in jobs_from_dict(jobs_src)]}", file=sys.stderr)
            return 1

    out_root = Path(args.output_dir)
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    n_mixed = sum(1 for j in jobs if j.is_mixed)
    print(f"== batch quantize: {len(jobs)} jobs ({n_mixed} mixed -> 2 passes each) "
          f"{'[DRY-RUN]' if args.dry_run else '[LIVE]'} ==")

    # Resume/skip: HF-public is the DURABLE record (spot NVMe is wiped on termination);
    # the local ledger is a same-VM fast path. List the namespace once for the batch.
    hf_token = os.environ.get("HF_TOKEN")
    published: dict[str, bool | None] = {}
    if not args.no_skip_hf and not args.force:
        published = existing_hf_quants(args.hf_namespace, hf_token)
        if published:
            n_pub = sum(1 for v in published.values() if v is False)
            print(f"  ({len(published)} repos under {args.hf_namespace}/, {n_pub} public — "
                  f"done jobs will be skipped)")

    prog_path = (Path(args.progress_file) if args.progress_file
                 else out_root / "batch_progress.json")
    ledger = load_ledger(prog_path)

    def _target_private(job: QuantJob) -> bool:
        return job.private if job.private is not None else bool(args.private)

    index: list[dict] = []
    for job in jobs:
        oname = output_name(job)
        out_dir = out_root / oname
        repo_id = f"{args.hf_namespace}/{oname}"
        tgt_priv = _target_private(job)
        kind = "mixed" if job.is_mixed else "uniform"
        print(f"\n# {job.name}  ({job.model})  [{kind}]  -> {out_dir}")

        if is_done(oname, job.name, tgt_priv, published, ledger) and not args.force:
            print(f"  SKIP — already done ({repo_id} at target visibility)")
            index.append({"name": job.name, "model": job.model, "dir": str(out_dir),
                          "repo_id": repo_id, "status": "skipped-done"})
            continue
        if args.upload and published.get(oname.lower()) is True and not tgt_priv:
            print(f"  NOTE — {repo_id} exists PRIVATE; will re-run + smoke + publish")

        # Quantization present locally? Skip the passes but still (re)attempt upload.
        already_local = (not args.dry_run and not args.force
                         and ((out_dir / "model.safetensors").exists()
                              or (out_dir / "config.json").exists()))
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "quant.log"

        if already_local:
            print("  quantization present locally — skipping passes (use --force to redo)")
            passes: list[dict] = [{"pass": "all", "status": "skipped-existing"}]
        else:
            passes = []
            for label, cmd in commands_for(job, out_dir, args.quantize_cmd,
                                           args.device or job.device):
                # resume: skip a finished profile pass (allocation already on disk)
                if (label == "profile" and not args.force and not args.dry_run
                        and (out_dir / "bpw_allocation.json").exists()):
                    print("    SKIP [profile] — bpw_allocation.json exists")
                    passes.append({"pass": label, "status": "skipped-existing"})
                    continue
                rec = run_pass(label, cmd, log_path, args)
                passes.append(rec)
                if rec["status"] not in ("ok", "dry-run", "skipped-existing"):
                    print("    -> aborting remaining passes for this job")
                    break

        entry = {"name": job.name, "model": job.model, "bpw": job.bpw,
                 "min_bpw": job.min_bpw, "max_bpw": job.max_bpw,
                 "mixed": job.is_mixed, "dir": str(out_dir), "repo_id": repo_id,
                 "log": str(log_path), "passes": passes}
        passed = bool(passes) and all(
            x["status"] in ("ok", "skipped-existing") for x in passes)

        if args.dry_run:
            index.append(entry)
            continue

        def _rec(**fields) -> None:  # ledger writer bound to this job
            record_progress(prog_path, args.hf_namespace, ledger, job.name,
                            model=job.model, output_name=oname, repo_id=repo_id, **fields)

        if not passed:
            _rec(quantized=False, status="failed", error="quantization pass failed")
            index.append(entry)
            continue
        _rec(quantized=True, status="quantized")

        if not args.upload:
            index.append(entry)
            continue

        # Upload PRIVATE -> smoke -> flip public (unless target private / --no-smoke).
        try:
            bench = None
            if args.bench:
                print("  benchmarking (MMLU-Pro + Winogrande)...", flush=True)
                bench = run_benchmarks(out_dir, args.bench_limit, hf_token)
            elif (out_dir / "benchmarks.json").exists():
                bench = json.loads((out_dir / "benchmarks.json").read_text())

            print(f"  uploading PRIVATE -> {repo_id} ...", flush=True)
            upload_private(out_dir, repo_id, job.model, hf_token, bench)
            _rec(uploaded=True, visibility="private", status="uploaded-private")

            if tgt_priv:
                print("  target visibility = private; leaving private")
                up = {"repo_id": repo_id, "visibility": "private", "smoke": "skipped"}
                _rec(smoke="skipped", status="done")
            elif args.no_smoke:
                set_repo_public(repo_id, hf_token)
                print(f"  -> PUBLIC (no smoke) {repo_id}")
                up = {"repo_id": repo_id, "visibility": "public", "smoke": "skipped"}
                _rec(smoke="skipped", visibility="public", status="done")
            else:
                print("  smoke testing (vLLM subprocess)...", flush=True)
                sm = run_smoke(out_dir, args.smoke_max_tokens)
                print(f"    smoke: {sm['status']} — {sm.get('reason', '')}")
                if sm["passed"]:
                    set_repo_public(repo_id, hf_token)
                    print(f"  -> PUBLIC {repo_id}")
                    up = {"repo_id": repo_id, "visibility": "public", "smoke": "passed"}
                    _rec(smoke="passed", visibility="public", status="done")
                else:
                    print(f"  -> kept PRIVATE (smoke {sm['status']}); retry on a later run")
                    up = {"repo_id": repo_id, "visibility": "private",
                          "smoke": sm["status"], "smoke_reason": sm.get("reason")}
                    _rec(smoke=sm["status"], visibility="private", status="uploaded-private")
            entry["upload"] = up
        except Exception as e:  # noqa: BLE001 — upload is best-effort, never kill the batch
            print(f"  -> UPLOAD FAILED: {type(e).__name__}: {e}", flush=True)
            entry["upload"] = {"repo_id": repo_id, "status": "failed",
                               "error": f"{type(e).__name__}: {e}"}
            _rec(status="failed", error=f"{type(e).__name__}: {e}")

        index.append(entry)

    if args.dry_run:
        print(f"\n== dry-run: {len(jobs)} jobs above ==")
        return 0

    idx_path = out_root / "batch_index.json"
    idx_path.write_text(json.dumps({"args": vars(args), "jobs": index}, indent=2))

    def _job_ok(j: dict) -> bool:
        if j.get("status") == "skipped-done":
            return True
        ps = j.get("passes", [])
        return bool(ps) and all(x["status"] in ("ok", "skipped-existing") for x in ps)

    ok = sum(1 for j in index if _job_ok(j))
    if args.upload:  # progress breakdown from the ledger (resumable record)
        counts: dict[str, int] = {}
        for rec in ledger.values():
            counts[rec.get("status", "?")] = counts.get(rec.get("status", "?"), 0) + 1
        if counts:
            print("  progress: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                  + f"  (ledger -> {prog_path})")
    print(f"\n== done: {ok}/{len(index)} jobs ok; index -> {idx_path} ==")
    return 0 if ok == len(index) else 1


if __name__ == "__main__":
    raise SystemExit(main())
