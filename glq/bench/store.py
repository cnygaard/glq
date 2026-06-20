"""Centralized results repo (standalone git) sync.

Records live in a dedicated git repo (durable across spot-instance churn) as
``records/<base_model>/<model>.jsonl``. ``results_repo`` may be a git URL (cloned
into ``results_dir``/cache) or an existing local checkout. Uses plain ``git`` via
subprocess (no GitPython dep).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from .record import BenchRecord, read_jsonl, write_jsonl

RECORDS_SUBDIR = "records"


def _git(repo_dir: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo_dir), *args],
                          capture_output=True, text=True, check=check)


def _default_checkout(results_repo: str) -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "glq-bench"
    name = results_repo.rstrip("/").split("/")[-1].removesuffix(".git") or "results"
    h = hashlib.sha1(results_repo.encode()).hexdigest()[:8]
    return cache / f"{name}-{h}"


def repo_dir(results_repo: str | None, results_dir: str | None = None) -> Path:
    """Resolve to a local working copy, cloning the URL if needed."""
    if not results_repo:
        raise ValueError("no results repo configured (pass --results-repo or set "
                         "$GLQ_BENCH_RESULTS_REPO).")
    # Local existing checkout?
    p = Path(results_repo)
    if p.exists() and (p / ".git").exists():
        return p
    dest = Path(results_dir) if results_dir else _default_checkout(results_repo)
    if (dest / ".git").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "50", results_repo, str(dest)],
                   capture_output=True, text=True, check=True)
    return dest


def pull(results_repo: str | None, results_dir: str | None = None) -> Path:
    d = repo_dir(results_repo, results_dir)
    _git(d, "pull", "--ff-only", check=False)
    return d


def _safe_name(model_id: str) -> str:
    return model_id.strip("/").replace("/", "__")


def _record_path(d: Path, rec: BenchRecord) -> Path:
    base = (rec.model.base_model or rec.model.id or "unparented").replace("/", "__")
    sub = d / RECORDS_SUBDIR / base
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{_safe_name(rec.model.id)}.jsonl"


def append(records: list[BenchRecord], results_repo: str | None,
           results_dir: str | None = None) -> Path:
    """Append records to the local checkout (no commit).

    Records are grouped by destination file (``records/<base>/<model>.jsonl``) and
    *merged* with any existing history there — so multiple tasks for one model, and
    re-runs over time, accumulate instead of overwriting. Exact-duplicate records
    (same serialized JSON) are dropped, making re-appends idempotent.
    """
    d = repo_dir(results_repo, results_dir)
    by_path: dict[Path, list[BenchRecord]] = {}
    for r in records:
        by_path.setdefault(_record_path(d, r), []).append(r)
    for path, new in by_path.items():
        existing = read_jsonl(path) if path.exists() else []
        merged: list[BenchRecord] = []
        seen: set[str] = set()
        for r in existing + new:
            key = r.to_json()
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)
        write_jsonl(merged, path)
    return d


def push(results_repo: str | None, results_dir: str | None = None,
         message: str = "glq-bench: add records") -> None:
    d = repo_dir(results_repo, results_dir)
    _git(d, "add", "-A")
    status = _git(d, "status", "--porcelain", check=False).stdout.strip()
    if not status:
        return                                # nothing to commit
    _git(d, "commit", "-m", message, check=False)
    _git(d, "push", check=False)


def append_and_push(records: list[BenchRecord], results_repo: str | None,
                    results_dir: str | None = None) -> Path:
    pull(results_repo, results_dir)
    d = append(records, results_repo, results_dir)
    models = sorted({r.model.id for r in records})
    push(results_repo, results_dir,
         message=f"glq-bench: {len(records)} record(s) for {', '.join(models)}")
    return d


def load_all(results_repo: str | None, results_dir: str | None = None,
             sync: bool = True) -> list[BenchRecord]:
    """All records in the repo (pull first unless sync=False)."""
    d = pull(results_repo, results_dir) if sync else repo_dir(results_repo, results_dir)
    out: list[BenchRecord] = []
    for jf in sorted((d / RECORDS_SUBDIR).rglob("*.jsonl")):
        out.extend(read_jsonl(jf))
    return out
