"""Diff two FULL-vs-PIECEWISE result JSONs (Phase 5.7v gate analysis).

Usage: python _ab_diff.py <a.json> <b.json> [labelA labelB]
Auto-detects MMLU (rows w/ pred/ok), NIAH (rows w/ pred/gold), or
varlen/mixed (runs[].rows w/ out). Reports per-question equality,
accuracy deltas, and text divergence.
"""
import json
import os
import sys


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def diff_mmlu(a, b, la, lb):
    ra, rb = a["rows"], b["rows"]
    n = min(len(ra), len(rb))
    pred_diff = sum(1 for x, y in zip(ra, rb) if x.get("pred") != y.get("pred"))
    nout_diff = sum(1 for x, y in zip(ra, rb) if x.get("n_out") != y.get("n_out"))
    oka = sum(x.get("ok") for x in ra)
    okb = sum(y.get("ok") for y in rb)
    lost = sum(1 for x, y in zip(ra, rb) if x.get("ok") and not y.get("ok"))
    gained = sum(1 for x, y in zip(ra, rb) if not x.get("ok") and y.get("ok"))
    print(f"  {la} correct={oka}  {lb} correct={okb}  (delta={okb - oka})")
    print(f"  pred differs:   {pred_diff}/{n}")
    print(f"  n_out differs:  {nout_diff}/{n}  (=> identical-gen on {n - nout_diff})")
    print(f"  {la}-right->{lb}-wrong (lost): {lost}   {la}-wrong->{lb}-right (gained): {gained}")


def diff_niah(a, b, la, lb):
    ra, rb = a["rows"], b["rows"]
    oka = sum(1 for x in ra if x.get("ok"))
    okb = sum(1 for y in rb if y.get("ok"))
    pred_diff = sum(1 for x, y in zip(ra, rb) if x.get("pred") != y.get("pred"))
    print(f"  {la} pass={oka}/{len(ra)}  {lb} pass={okb}/{len(rb)}  pred differs {pred_diff}")


def _varlen_rows(d, mode):
    for r in d.get("runs", []):
        if r.get("mode") == mode:
            return r.get("rows", [])
    return []


def diff_varlen(a, b, la, lb):
    for mode in ("varlen", "mixed"):
        ra, rb = _varlen_rows(a, mode), _varlen_rows(b, mode)
        if not ra and not rb:
            continue
        print(f"  [{mode}] {la}={len(ra)} {lb}={len(rb)} rows")
        # mixed rows keyed by rid; varlen by ctx — align by index
        for x, y in zip(ra, rb):
            tag = x.get("ctx", x.get("rid"))
            same = x.get("out") == y.get("out")
            print(f"    key={tag!s:>7} same_text={same}")
            if not same:
                print(f"       {la}: {x.get('out')!r}")
                print(f"       {lb}: {y.get('out')!r}")


def main():
    pa, pb = sys.argv[1], sys.argv[2]
    la = sys.argv[3] if len(sys.argv) > 3 else "A"
    lb = sys.argv[4] if len(sys.argv) > 4 else "B"
    a, b = load(pa), load(pb)
    if a is None or b is None:
        print(f"  MISSING: {pa} exists={a is not None}; {pb} exists={b is not None}")
        return
    print(f"=== {pa} ({la}) vs {pb} ({lb}) ===")
    if "runs" in a:
        diff_varlen(a, b, la, lb)
    elif a.get("rows") and "pred" in a["rows"][0] and "ok" in a["rows"][0] \
            and "n_out" in a["rows"][0]:
        diff_mmlu(a, b, la, lb)
    elif a.get("rows") and "gold" in a["rows"][0]:
        diff_niah(a, b, la, lb)
    else:
        print("  (unrecognized JSON shape)")


if __name__ == "__main__":
    main()
