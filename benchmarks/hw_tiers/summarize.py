"""One markdown row per tier run, from the logs `tier_run.sh` writes.

    python benchmarks/hw_tiers/summarize.py /opt/dlami/nvme/tiers > results.md

Reads `<tag>.log` (the `RUN_START` line and the `TIER {json}` phases) and `<tag>.vllm.log`
(vLLM's own "Model loading took" and "Available KV cache memory"). Decode is the mean of the
probe's repeats with their range; a failed row carries vLLM's root-cause line instead.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


#: vLLM's own wrappers, printed after the error that caused them (as in `tier_launch.py`).
_WRAPPERS = ("See root cause above", "exited with status", "Engine core initialization failed")


def _last(pattern, text):
    found = re.findall(pattern, text)
    return found[-1] if found else None


def _root_cause(vtext):
    """The last `...Error: ...` line that is not one of vLLM's wrappers. Rows run before
    `tier_launch._vllm_error` existed recorded only the wrapper."""
    errs = [f"{m.group(1)}: {m.group(2)}" for line in vtext.splitlines()
            for m in [re.search(r"\b(\w*(?:Error|Exception)): (.*)", line)]
            if m and not line.lstrip().startswith("File ") and not any(w in line for w in _WRAPPERS)]
    return errs[-1] if errs else None


def row(log: Path) -> dict | None:
    text = log.read_text(errors="replace")
    start = re.search(r"^RUN_START (\S+) gpu=\S+ vram=(\S+) ram=(\S+) model=(\S+) flags: ?(.*?) \d{4}-",
                      text, re.M)
    if not start:
        return None
    tag, vram, ram, model, flags = start.groups()
    phases = {}
    for line in text.splitlines():
        if line.startswith("TIER "):
            d = json.loads(line[5:])
            phases[d["phase"]] = d
    vlog = log.with_name(log.stem + ".vllm.log")
    vtext = vlog.read_text(errors="replace") if vlog.exists() else ""
    plan = phases.get("plan", {})
    out = dict(tag=tag, vram=vram, ram=ram, model=model.split("/")[-1], forced="--force" in flags,
               window=plan.get("window"), offload=plan.get("offload_gib"),
               arena=plan.get("arena_gib"), k=plan.get("spec_tokens"),
               loaded=_last(r"Model loading took ([0-9.]+) GiB", vtext),
               kv=_last(r"Available KV cache memory: ([0-9.]+) GiB", vtext))
    runs = [r for r in phases.get("probe", {}).get("runs", []) if r.get("decode_tps")]
    if runs:
        tps = [r["decode_tps"] for r in runs]
        out["result"] = (f"{sum(tps) / len(tps):.1f} tok/s ({min(tps):.1f}-{max(tps):.1f}), "
                         f"TTFT {1000 * sum(r['ttft_s'] for r in runs) / len(runs):.0f} ms")
    elif "failed" in phases:
        why = phases["failed"].get("vllm")
        if not why or any(w in why for w in _WRAPPERS):
            why = _root_cause(vtext) or why or phases["failed"]["error"]
        out["result"] = "failed: " + why
    elif "plan" in phases and "ready" not in phases:
        out["result"] = f"refused by the planner (rc {phases.get('exit', {}).get('rc')})"
    else:
        out["result"] = "no probe"
    return out


def main(argv):
    root = Path(argv[1] if len(argv) > 1 else ".")
    rows = [r for p in sorted(root.glob("*.log"))
            if not p.name.endswith(".vllm.log") and not p.name.startswith("queue")
            for r in [row(p)] if r]
    rows.sort(key=lambda r: (float(r["vram"]), float(r["ram"]), r["tag"]))
    print("| tag | VRAM / RAM GiB | model | forced | window | offload GiB | arena GiB | k "
          "| loaded GiB | KV GiB | result |")
    print("|---|---|---|---|--:|--:|--:|--:|--:|--:|---|")
    for r in rows:
        res = r["result"].replace("|", "/")
        res = res if len(res) <= 160 else res[:157] + "..."
        print(f"| {r['tag']} | {r['vram']} / {r['ram']} | {r['model']} | {'yes' if r['forced'] else ''} "
              f"| {r['window'] or ''} | {r['offload'] or 0} | {r['arena'] or 0:g} | {r['k'] or 0} "
              f"| {r['loaded'] or '-'} | {r['kv'] or '-'} | {res} |")


if __name__ == "__main__":
    main(sys.argv)
