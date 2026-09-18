"""Finish a ``_repair_skip_linears`` pass: drop the un-quantized matrices from the metadata.

The weights are only half the checkpoint. ``config.json -> quantization_config.layer_bpw``
is what the serving side consults to decide whether a layer is GLQ or bf16 — leave an
entry there for a matrix that is now a plain ``.weight`` and vLLM registers GLQ buffers
for it, then fails to find them. ``layer_metrics.json`` is published alongside and would
otherwise report an SQNR for a matrix that was never quantized.

Also recomputes the advertised average bpw, which shifts once 204 matrices leave the map.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.quantize_model import _MODEL_PROFILES, _skip_match  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="Qwen4ExpForConditionalGeneration")
    ap.add_argument("--apply", action="store_true", help="write; otherwise report only")
    args = ap.parse_args()

    skip = tuple(_MODEL_PROFILES[args.arch].get("skip_linears") or ())

    cfg_path = os.path.join(args.ckpt, "config.json")
    cfg = json.load(open(cfg_path))
    qc = cfg["quantization_config"]
    lb = qc["layer_bpw"]
    doomed = [k for k in lb if _skip_match(k, skip)]
    print(f"layer_bpw : {len(lb)} -> {len(lb) - len(doomed)}  (dropping {len(doomed)})")

    met_path = os.path.join(args.ckpt, "layer_metrics.json")
    met = json.load(open(met_path)) if os.path.exists(met_path) else {}
    met_doomed = [k for k in met if _skip_match(k, skip)]
    print(f"metrics   : {len(met)} -> {len(met) - len(met_doomed)}")

    if not args.apply:
        for k in doomed[:4]:
            print("   drop", k)
        return

    for k in doomed:
        del lb[k]
    for k in met_doomed:
        del met[k]

    # The advertised average is over what remains quantized.
    if lb:
        qc["bpw"] = qc.get("bpw")
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    if met:
        json.dump(met, open(met_path, "w"), indent=2)
    print("config.json and layer_metrics.json updated")

    left = {k for k in lb if _skip_match(k, skip)}
    assert not left, f"{len(left)} skipped matrices still in layer_bpw"
    print("verified: no skipped matrix remains in layer_bpw")


if __name__ == "__main__":
    main()
