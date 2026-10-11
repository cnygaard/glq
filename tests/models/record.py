"""Record a model's tensor SHAPES — no weight data — as a test fixture.

Why this exists: fixtures written by hand encode the author's model of an architecture, and
that model is wrong in ways the architecture will not tell you about. Three measured examples
from one session:

* an MTP-head fixture at 16x8, which the production filter rejects outright for failing the
  trellis out%32 serving gate — so the tests passed against a fixture that could never occur;
* a `layer_bpw` fixture keyed on a bare ``...mlp.experts`` container, which no checkpoint
  contains (real ones enumerate one entry per expert per projection) — this hid a regression
  that declined every MoE layer and OOM'd a 96 GB card;
* norms in the real Qwen4Exp MTP head are spelled ``hc_norm``, ``k_norm``, ``q_norm``,
  ``pre_fc_norm_embedding``, ``pre_fc_norm_hidden`` and ``q_layernorm`` — a name rule catches
  two of nine, which is why **shape is the authority for norms, not spelling**.

A recorded manifest removes the guesswork: the test asks the real architecture.

## Format

Numeric path segments are collapsed to ``{i}`` and the index ranges recorded, so a 295,000-key
checkpoint becomes ~100 patterns and a few KB. Expansion is exact: the recorder asserts each
index set is contiguous ``0..max`` and stores the explicit set when it is not, rather than
quietly misrepresenting it.

## Usage

    # from a local checkpoint directory (reads headers only)
    python tests/models/record.py --checkpoint /path/to/ckpt --out tests/models/<slug>.json

    # or straight from the Hub, without downloading the weights
    python tests/models/record.py --repo google/gemma-4-26B-A4B-it --out tests/models/<slug>.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys

_NUM = re.compile(r"(?<=\.)\d+(?=\.|$)")


def collapse(key: str) -> tuple[str, tuple[int, ...]]:
    """``a.0.b.7.c`` -> ``("a.{i}.b.{i}.c", (0, 7))``."""
    idx = tuple(int(m.group()) for m in _NUM.finditer(key))
    return _NUM.sub("{i}", key), idx


def _header_shapes_local(checkpoint_dir):
    """``{key: (dtype, shape)}`` from local safetensors headers — no tensor data read."""
    from safetensors import safe_open

    out = {}
    files = sorted(f for f in os.listdir(checkpoint_dir) if f.endswith(".safetensors"))
    if not files:
        raise SystemExit(f"{checkpoint_dir}: no .safetensors files")
    for fn in files:
        with safe_open(os.path.join(checkpoint_dir, fn), framework="pt") as fh:
            for k in fh.keys():
                sl = fh.get_slice(k)
                out[k] = (sl.get_dtype(), tuple(sl.get_shape()))
    return out


def _header_shapes_hub(repo):
    """Same, straight from the Hub. The header alone is fetched, not the weights.

    Note the cap: `get_safetensors_metadata` refuses a header above 25 MB, and
    Qwen3.8-Flash-Next's is 39.45 MB — so that model must be recorded from a local copy.
    """
    from huggingface_hub import get_safetensors_metadata

    tok = os.environ.get("HF_TOKEN") or os.environ.get("HF_READ_TOKEN")
    meta = get_safetensors_metadata(repo, token=tok)
    out = {}
    # Per-tensor info lives one level down, in each file's own metadata — the repo-level
    # object carries only `weight_map` (tensor -> filename).
    # `files_metadata`, not `files` — the repo object exposes only `files_metadata`,
    # `weight_map`, `metadata`, `parameter_count` and `sharded`.
    for fmeta in meta.files_metadata.values():
        for k, t in getattr(fmeta, "tensors", {}).items():
            out[k] = (str(t.dtype), tuple(t.shape))
    if not out:
        raise SystemExit(
            f"{repo}: no tensor metadata returned. Note the 25 MB header cap — "
            "Qwen3.8-Flash-Next's header is 39.45 MB and must be recorded from a local "
            "copy with --checkpoint.")
    return out


def build_manifest(shapes, *, model, source, base_model=None, license_id=None,
                   license_url=None):
    patterns = {}
    for key, (dtype, shape) in shapes.items():
        pat, idx = collapse(key)
        rec = patterns.setdefault(pat, {"dtype": dtype, "shape": list(shape),
                                        "indices": set(), "n": 0})
        rec["n"] += 1
        rec["indices"].add(idx)
        if list(shape) != rec["shape"] or dtype != rec["dtype"]:
            # Two keys sharing a pattern but not a shape: keep them distinguishable rather
            # than silently recording one of them.
            rec.setdefault("conflicts", []).append({"key": key, "dtype": dtype,
                                                    "shape": list(shape)})
    out = {}
    for pat, rec in sorted(patterns.items()):
        idxs = sorted(rec.pop("indices"))
        entry = {"dtype": rec["dtype"], "shape": rec["shape"], "n": rec["n"]}
        if "conflicts" in rec:
            entry["conflicts"] = rec["conflicts"]
        if idxs and idxs != [()]:
            n_pos = len(idxs[0])
            ranges, explicit = [], False
            for pos in range(n_pos):
                vals = sorted({t[pos] for t in idxs})
                if vals != list(range(len(vals))):
                    explicit = True          # not contiguous 0..max; do not fake it
                ranges.append([vals[0], vals[-1]])
            entry["ranges"] = ranges
            if explicit or len(idxs) != _product(ranges):
                entry["index_tuples"] = [list(t) for t in idxs]
        out[pat] = entry
    return {
        "model": model,
        "source": source,
        "recorded": _dt.date.today().isoformat(),
        "note": "safetensors header only - shapes and dtypes, no weight data",
        # Attribution travels INSIDE the manifest, not in a sibling README, because the
        # manifest is what gets copied around and read programmatically. A GLQ checkpoint is
        # a derivative of somebody else's model and the base model's terms govern it; the
        # licenses genuinely differ (gemma-4 is Apache 2.0, Qwen3.8-Flash-Next is the Qwen
        # Community License 1.0), so this is recorded per model and never defaulted.
        "base_model": base_model or "",
        "license": license_id or "",
        "license_url": license_url or "",
        "n_keys": len(shapes),
        "shapes": out,
    }


def _product(ranges):
    n = 1
    for lo, hi in ranges:
        n *= (hi - lo + 1)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="local checkpoint directory")
    src.add_argument("--repo", help="HF repo id (header fetched, weights are not)")
    ap.add_argument("--model", default=None, help="name to record (default: the source)")
    ap.add_argument("--out", required=True)
    # Required, not optional with a default: a GLQ checkpoint is a derivative of somebody
    # else's model, the manifest goes into a public repo, and the licenses differ per model
    # (gemma-4 Apache 2.0, Qwen3.8-Flash-Next the Qwen Community License 1.0). Defaulting
    # any of these would put a guessed license on a real model.
    ap.add_argument("--base-model", required=True,
                    help="the upstream model this checkpoint derives from, e.g. "
                         "Qwen/Qwen3.8-Flash-Next")
    ap.add_argument("--license", required=True, dest="license_id",
                    help="the BASE model's license id, e.g. apache-2.0")
    ap.add_argument("--license-url", required=True,
                    help="link to the base model's license text")
    args = ap.parse_args(argv)

    if args.checkpoint:
        shapes = _header_shapes_local(args.checkpoint)
        source = f"local:{args.checkpoint}"
    else:
        shapes = _header_shapes_hub(args.repo)
        source = f"hub:{args.repo}"

    manifest = build_manifest(shapes, model=args.model or (args.repo or args.checkpoint),
                              source=source, base_model=args.base_model,
                              license_id=args.license_id, license_url=args.license_url)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    size = os.path.getsize(args.out)
    print(f"wrote {args.out}: {len(manifest['shapes'])} patterns covering "
          f"{manifest['n_keys']} keys, {size/1024:.1f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
