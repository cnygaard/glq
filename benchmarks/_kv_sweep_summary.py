"""Summarise a KV-quantization sweep: capacity and speed, one arm per row.

Reads what `_kv_sweep_e8_vs_fp8.sh` leaves in the output directory — the capacity
line scraped from each server log, and vLLM's own `bench serve` result JSONs — and
prints the comparison the decision actually needs.

Capacity is reported as a ratio to the bf16 arm, because that is the claim: E8-KV
buys context per GiB. Speed is reported per request rate, because that is where
vLLM's TurboQuant blog rejected the 3-bit variants — a method can win on capacity
and still lose on TTFT under load.

    python benchmarks/_kv_sweep_summary.py --out /opt/dlami/nvme/kvsweep \
        --arms "bf16 fp8 turboquant_4bit_nc e8_relaxed2"
"""
from __future__ import annotations

import argparse
import json
import os


def _capacity(out_dir: str, arm: str) -> int | None:
    """Tokens the KV pool held, from `KV cache size: <N> tokens` in the server log."""
    path = os.path.join(out_dir, f"{arm}.capacity.txt")
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError:
        return None
    digits = "".join(c for c in text if c.isdigit())
    return int(digits) if digits else None


def _bench(out_dir: str, arm: str, rate: str) -> dict | None:
    path = os.path.join(out_dir, f"{arm}.rate{rate}.json")
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    # vLLM writes either a dict or a single-element list depending on version.
    if isinstance(data, list):
        data = data[0] if data else {}
    return data or None


def _fmt(value, spec=".1f"):
    return format(value, spec) if isinstance(value, (int, float)) else "—"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", required=True, help="space-separated arm names")
    ap.add_argument("--rates", default="2 8 inf")
    ap.add_argument("--baseline", default="bf16",
                    help="arm the capacity ratio is taken against")
    args = ap.parse_args()

    arms = args.arms.split()
    rates = args.rates.split()
    base_cap = _capacity(args.out, args.baseline)

    print(f"\nKV capacity (tokens in the pool; ratio vs {args.baseline})")
    print(f"  {'arm':22s} {'tokens':>12s}  {'ratio':>6s}")
    for arm in arms:
        cap = _capacity(args.out, arm)
        ratio = f"{cap / base_cap:.2f}x" if cap and base_cap else "—"
        print(f"  {arm:22s} {cap if cap else '—':>12}  {ratio:>6s}")

    for rate in rates:
        print(f"\nServing at request rate = {rate}")
        print(f"  {'arm':22s} {'out tok/s':>10s} {'TTFT ms':>10s} "
              f"{'TPOT ms':>10s} {'P99 TTFT':>10s}")
        for arm in arms:
            b = _bench(args.out, arm, rate)
            if b is None:
                print(f"  {arm:22s} {'— (no result)':>10s}")
                continue
            print(f"  {arm:22s} {_fmt(b.get('output_throughput')):>10s} "
                  f"{_fmt(b.get('mean_ttft_ms')):>10s} "
                  f"{_fmt(b.get('mean_tpot_ms')):>10s} "
                  f"{_fmt(b.get('p99_ttft_ms')):>10s}")

    failures = os.path.join(args.out, "failures.txt")
    if os.path.exists(failures):
        print("\nArms that did not run:")
        with open(failures) as fh:
            for line in fh:
                print(f"  {line.rstrip()}")

    print("\nEvery number above is one model on one GPU at a fixed "
          f"gpu-memory-utilization; capacity is comparable only because that "
          f"fraction and max-model-len were held equal across arms.")


if __name__ == "__main__":
    main()
