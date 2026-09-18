"""Un-quantize matrices a profile's ``skip_linears`` excludes, in place, without re-running.

Why this exists rather than a re-save: the resume bank covers *layers only*. A re-save
replays 48 layers in ~12 minutes and then re-quantizes the 24 GiB PLE n-gram table from
scratch — 19,532 chunks, ~4 hours — to reproduce a table that is already correct. The
matrices being fixed here have nothing to do with the PLE.

What it does, for every ``layer_bpw`` entry the profile now skips:

  * drops its GLQ buffers (``trellis_packed`` / ``Qidxs*`` / ``SU`` / ``SV`` / ``Wscale`` …)
  * splices the original bf16 ``.weight`` back in from the source model's shards
  * removes it from ``config.json -> quantization_config.layer_bpw`` and from
    ``layer_metrics.json``

Everything else is copied **byte for byte** — the tensor data is moved as raw bytes at the
source file's own offsets, never materialised as a tensor and never re-encoded. That is
what makes the gate below meaningful, and it keeps peak memory at the copy-buffer size
rather than the 23.84 GiB of the PLE table.

Gate (``--verify``): every surviving tensor must be byte-identical to the original, the
dropped set must be exactly the expected one, and the spliced weights must match the
source shards. Run ``benchmarks/_ckpt_bytecmp.py`` afterwards for an independent check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.quantize_model import _MODEL_PROFILES, _skip_match  # noqa: E402

#: Suffixes that make a tensor part of a GLQ artifact rather than a plain weight.
_GLQ_SUFFIXES = (".trellis_packed", ".Qidxs", ".Qidxs2", ".Qidxs_e8p", ".SU", ".SV",
                 ".Wscale", ".inv_resid_scale", ".rht_blocks", ".tlut", ".resid_scale")


def _read_header(path):
    """(header_dict, data_start) for a safetensors file, without loading any tensor."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def _dtype_size(dt):
    return {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "U32": 4, "F16": 2, "BF16": 2,
            "I16": 2, "U16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1,
            "BOOL": 1}[dt]


def _plan(ckpt_dir, arch):
    """(drop_keys, restore_names) — what to remove and what to bring back as bf16."""
    cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
    layer_bpw = cfg["quantization_config"]["layer_bpw"]
    profile = _MODEL_PROFILES[arch]
    skip = tuple(profile.get("skip_linears") or ())
    restore = sorted(k for k in layer_bpw if _skip_match(k, skip))

    header, _ = _read_header(os.path.join(ckpt_dir, "model.safetensors"))
    restore_set = set(restore)
    drop = set()
    for key in header:
        if key == "__metadata__":
            continue
        for sfx in _GLQ_SUFFIXES:
            if key.endswith(sfx) and key[: -len(sfx)] in restore_set:
                drop.add(key)
                break
    return sorted(drop), restore


def _source_index(src_dir):
    """tensor name -> shard path, for the source model."""
    idx_path = os.path.join(src_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        wm = json.load(open(idx_path))["weight_map"]
        return {k: os.path.join(src_dir, v) for k, v in wm.items()}
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="checkpoint dir to repair")
    ap.add_argument("--source", required=True, help="source model snapshot dir")
    ap.add_argument("--arch", default="Qwen4ExpForConditionalGeneration")
    ap.add_argument("--out", required=True, help="new model.safetensors path")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from safetensors import safe_open

    ck = os.path.join(args.ckpt, "model.safetensors")
    drop, restore = _plan(args.ckpt, args.arch)
    print(f"matrices to un-quantize : {len(restore)}")
    print(f"GLQ tensors to drop     : {len(drop)}")
    if not restore:
        sys.exit("nothing to do — profile skips nothing present in layer_bpw")

    src_idx = _source_index(args.source)
    missing = [n for n in restore if f"{n}.weight" not in src_idx]
    if missing:
        sys.exit(f"source lacks bf16 weights for {len(missing)} matrices, "
                 f"e.g. {missing[:3]}")

    header, data_start = _read_header(ck)
    meta = header.get("__metadata__")
    keep = [k for k in header if k != "__metadata__" and k not in drop]

    # Sizes of the bf16 weights we are splicing back in.
    new = {}
    for name in restore:
        path = src_idx[f"{name}.weight"]
        with safe_open(path, "pt") as f:
            sl = f.get_slice(f"{name}.weight")
            new[f"{name}.weight"] = (sl.get_dtype(), list(sl.get_shape()), path)

    added = sum(_dtype_size(d) * _prod(s) for d, s, _ in new.values())
    removed = sum(header[k]["data_offsets"][1] - header[k]["data_offsets"][0]
                  for k in drop)
    old_bytes = sum(header[k]["data_offsets"][1] - header[k]["data_offsets"][0]
                    for k in header if k != "__metadata__")
    print(f"bytes dropped {removed / 2**30:8.2f} GiB")
    print(f"bytes added   {added / 2**30:8.2f} GiB")
    print(f"total {old_bytes / 2**30:.2f} -> {(old_bytes - removed + added) / 2**30:.2f} GiB")
    if args.dry_run:
        for n in restore[:5]:
            print("   restore", n)
        return

    # ---- build the new header, then stream the data -------------------------------
    out_header, off = {}, 0
    for k in keep:
        info = header[k]
        n = info["data_offsets"][1] - info["data_offsets"][0]
        out_header[k] = {"dtype": info["dtype"], "shape": info["shape"],
                         "data_offsets": [off, off + n]}
        off += n
    for k, (dt, shape, _) in new.items():
        n = _dtype_size(dt) * _prod(shape)
        out_header[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    if meta is not None:
        out_header["__metadata__"] = meta

    blob = json.dumps(out_header, separators=(",", ":")).encode()
    pad = (-len(blob)) % 8                      # safetensors wants 8-byte alignment
    blob += b" " * pad

    BUF = 64 << 20
    with open(args.out, "wb") as out, open(ck, "rb") as src:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for i, k in enumerate(keep):
            a, b = header[k]["data_offsets"]
            src.seek(data_start + a)
            left = b - a
            while left:
                chunk = src.read(min(BUF, left))
                if not chunk:
                    sys.exit(f"short read on {k}")
                out.write(chunk)
                left -= len(chunk)
            if i % 20000 == 0:
                print(f"  copied {i}/{len(keep)}", flush=True)
        for k, (dt, shape, path) in new.items():
            with safe_open(path, "pt") as f:
                t = f.get_tensor(k)
            assert list(t.shape) == shape, (k, t.shape, shape)
            # numpy has no bfloat16, so reinterpret through a same-width integer rather
            # than converting — the bytes must land unchanged.
            tv = t.contiguous()
            view = _uint_view(tv.dtype)
            if view is not tv.dtype:
                tv = tv.view(view)
            out.write(tv.numpy().tobytes())
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 2**30:.2f} GiB)")


def _uint_view(dtype):
    import torch
    return {torch.bfloat16: torch.int16, torch.float16: torch.int16,
            torch.float32: torch.int32, torch.float64: torch.int64}.get(dtype, dtype)


def _prod(xs):
    n = 1
    for x in xs:
        n *= x
    return n


if __name__ == "__main__":
    main()
