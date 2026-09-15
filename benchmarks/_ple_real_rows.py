"""Step 2: quantize the REAL PLE table and check it survives the round trip.

Everything pinning the trellis PLE path so far uses synthetic Gaussian tables. A Gaussian has
none of the outlier structure of a hashed n-gram embedding, so the unit tests prove the
plumbing and say nothing about the data. The only real-row number to date is the tiling gate's
17.49 dB, measured on 16384 rows of one shard.

This quantizes whole shards of the real table and answers two things the synthetic tests
cannot:

  1. what SQNR the encoder actually achieves on real rows, reported PER SHARD rather than
     averaged into a single number that could hide a bad region;
  2. whether `TrellisRHTEmbedding` decodes those rows back to exactly what the encoder
     produced — the checkpoint-vs-memory contract, on real data.

Reads one 0.75 GiB shard at a time via `_ple_row_reader`, never the 335 GiB checkpoint.

    python benchmarks/_ple_real_rows.py [--shards 1] [--bpw 3] [--chunk-rows 16384]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.quantize_model import (_ple_row_reader,                 # noqa: E402
                                _quantize_ple_chunk_trellis)
from glq.quantized_linear import TrellisRHTEmbedding             # noqa: E402

MODEL = "Qwen/Qwen3.8-Flash-Next"
PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"


def sqnr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref, got = ref.float(), got.float()
    err = (ref - got).pow(2).sum()
    return float(10 * torch.log10(ref.pow(2).sum() / err.clamp_min(1e-30)))


def _fetch(n_shards: int):
    """Download just the shard files holding the first `n_shards` pieces of the table."""
    from huggingface_hub import hf_hub_download

    idx = json.load(open(hf_hub_download(MODEL, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    weight_map, shard_paths = {}, {}
    for i in range(n_shards):
        key = f"{PREFIX}.shard_{i}.weight"
        if key not in wm:
            raise SystemExit(f"{key} absent from the index")
        fn = wm[key]
        if fn not in shard_paths:
            print(f"  fetching {fn} ...", flush=True)
            shard_paths[fn] = hf_hub_download(MODEL, fn)
        weight_map[key] = fn
    return weight_map, shard_paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--bpw", type=int, default=3)
    ap.add_argument("--chunk-rows", type=int, default=16384)
    ap.add_argument("--verify-rows", type=int, default=4096,
                    help="rows to round-trip through the module (0 = skip)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    weight_map, shard_paths = _fetch(args.shards)
    total, width, read = _ple_row_reader(
        weight_map, shard_paths, {"prefix": PREFIX, "shards": args.shards})
    print(f"\ntable slice: {total:,} rows x {width}  ({total * width * 2 / 2**30:.2f} GiB "
          f"bf16)  on {dev}, K={args.bpw}")

    # Per-chunk so a bad region shows up instead of being averaged away.
    per_chunk, t0 = [], time.time()
    first_arts = first_ref = None
    for r0 in range(0, total, args.chunk_rows):
        r1 = min(r0 + args.chunk_rows, total)
        ref = read(r0, r1).float()
        arts, hat = _quantize_ple_chunk_trellis(ref, bpw=args.bpw, device=dev)
        s = sqnr_db(ref.to(dev), hat)
        per_chunk.append(s)
        if first_arts is None:
            first_arts, first_ref = arts, ref
        if len(per_chunk) % 20 == 0 or r1 == total:
            print(f"  {r1:>9,}/{total:,} rows   SQNR last {s:6.2f} dB   "
                  f"min {min(per_chunk):6.2f}  max {max(per_chunk):6.2f}   "
                  f"({time.time() - t0:.0f}s)", flush=True)

    mean = sum(per_chunk) / len(per_chunk)
    print(f"\n  SQNR over {len(per_chunk)} chunks: mean {mean:.2f} dB, "
          f"min {min(per_chunk):.2f}, max {max(per_chunk):.2f}")
    print(f"  storage: {first_arts['trellis_packed'].shape[1] * 2} B/row packed "
          f"+ 2 B fp16 scale -> "
          f"{total * (first_arts['trellis_packed'].shape[1] * 2 + 2) / 2**30:.2f} GiB "
          f"for this slice")

    # The contract the synthetic tests assert, now on real rows: what the module decodes is
    # what the encoder produced. A drift here means the checkpoint and memory disagree.
    if args.verify_rows:
        n = min(args.verify_rows, first_arts['Wscale'].shape[0])
        mod = TrellisRHTEmbedding(n, width, bpw=args.bpw)
        sd = {k: v for k, v in first_arts.items() if not k.startswith("_")}
        sd["trellis_packed"] = sd["trellis_packed"][:n]
        sd["Wscale"] = sd["Wscale"][:n]
        sd["rht_blocks"] = torch.tensor(first_arts["_blocks_n"], dtype=torch.int32)
        missing, unexpected = mod.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        got = mod(torch.arange(n))
        ref = first_ref[:n]
        print(f"\n  module round-trip on {n:,} real rows: "
              f"SQNR {sqnr_db(ref, got.float()):.2f} dB vs bf16")
        _, hat_ref = _quantize_ple_chunk_trellis(first_ref, bpw=args.bpw, device=dev)
        drift = (got.float() - hat_ref[:n].float().cpu()).abs().max()
        print(f"  max |module - encoder| on those rows: {drift:.2e} "
              f"{'OK' if drift < 1e-2 else '<-- DRIFT'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
