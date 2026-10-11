"""Load recorded model shape manifests, so tests can ask the real architecture.

A manifest in ``tests/models/`` holds a checkpoint's tensor keys, shapes and dtypes with **no
weight data** — a safetensors header, collapsed over repeated indices. See
``tests/models/record.py`` for why: hand-written fixtures encode the author's model of an
architecture, and that model is wrong in ways the architecture will not volunteer.

The interface is deliberately the one production already takes. ``shape_of`` matches the
``shape_of=`` callable that `glq.quantize_model.mtp_sublayer_weights` accepts, so a test can
run the real filter over a real key set:

    shapes = load("qwen3_8_flash_next_glq_3bpw")
    groups = mtp_sublayer_weights(shapes.keys(), shape_of=shapes.shape_of, skip=SKIP)
"""
from __future__ import annotations

import functools
import itertools
import json
import os
import re

_NUM = re.compile(r"(?<=\.)\d+(?=\.|$)")
_DIR = os.path.join(os.path.dirname(__file__), "models")

#: safetensors dtype name -> bytes per element, for footprint assertions.
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1,
                "I16": 2, "I32": 4, "I64": 8, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def collapse(key: str) -> str:
    """``a.0.b.7.c`` -> ``a.{i}.b.{i}.c`` — the manifest's pattern form."""
    return _NUM.sub("{i}", key)


class ModelShapes:
    """Shapes and dtypes for one recorded checkpoint. Read-only."""

    def __init__(self, data: dict):
        self._d = data
        self._shapes = data["shapes"]

    # ---- metadata ----------------------------------------------------------

    @property
    def model(self) -> str:
        return self._d["model"]

    @property
    def n_keys(self) -> int:
        """Concrete tensor count in the source checkpoint."""
        return self._d["n_keys"]

    @property
    def meta(self) -> dict:
        """Everything recorded except the shapes — provenance and licensing.

        Carries ``base_model`` / ``license`` / ``license_url``: a manifest holds no weights,
        but it is derived from somebody else's released model and lives in a public repo, so
        the base model's terms travel with it.
        """
        return {k: v for k, v in self._d.items() if k != "shapes"}

    def __repr__(self):
        return (f"<ModelShapes {self.model!r} {self.n_keys} keys "
                f"{len(self._shapes)} patterns recorded {self._d['recorded']}>")

    # ---- lookup ------------------------------------------------------------

    def shape_of(self, key: str):
        """Shape tuple for a CONCRETE key, or None if the model has no such tensor.

        Signature-compatible with `mtp_sublayer_weights(shape_of=...)`, which is the point:
        the test exercises the production filter rather than a paraphrase of it.
        """
        rec = self._shapes.get(collapse(key))
        if rec is None:
            return None
        for c in rec.get("conflicts", ()):      # same pattern, different shape
            if c["key"] == key:
                return tuple(c["shape"])
        return tuple(rec["shape"])

    def dtype_of(self, key: str):
        rec = self._shapes.get(collapse(key))
        return None if rec is None else rec["dtype"]

    def nbytes_of(self, key: str):
        """Bytes this tensor occupies, or None when the dtype is unknown."""
        rec = self._shapes.get(collapse(key))
        if rec is None or rec["dtype"] not in _DTYPE_BYTES:
            return None
        n = 1
        for d in self.shape_of(key):
            n *= d
        return n * _DTYPE_BYTES[rec["dtype"]]

    # ---- enumeration -------------------------------------------------------

    @functools.cached_property
    def _expanded(self) -> tuple[str, ...]:
        out = []
        for pat, rec in self._shapes.items():
            tuples = rec.get("index_tuples")
            if tuples is None:
                ranges = rec.get("ranges")
                if not ranges:
                    out.append(pat)
                    continue
                tuples = itertools.product(*(range(lo, hi + 1) for lo, hi in ranges))
            for idx in tuples:
                parts = pat.split("{i}")
                out.append("".join(p + (str(i) if k < len(idx) else "")
                                   for k, (p, i) in enumerate(
                                       itertools.zip_longest(parts, idx, fillvalue=""))))
        return tuple(out)

    def keys(self):
        """Every concrete tensor key, expanded from the recorded index ranges."""
        return self._expanded

    def keys_under(self, prefix: str):
        """Concrete keys beneath a dotted prefix, matched on SEGMENTS not characters."""
        pre = prefix.rstrip(".") + "."
        return tuple(k for k in self._expanded if k.startswith(pre))

    def patterns(self):
        """The collapsed patterns, for a test that wants kinds rather than instances."""
        return dict(self._shapes)


def available() -> tuple[str, ...]:
    if not os.path.isdir(_DIR):
        return ()
    return tuple(sorted(f[:-5] for f in os.listdir(_DIR) if f.endswith(".json")))


def load(slug: str) -> ModelShapes:
    path = os.path.join(_DIR, f"{slug}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no recorded shapes for {slug!r}. Available: {available() or '(none)'}. "
            f"Record one with: python tests/models/record.py --repo <id> --out {path}")
    with open(path) as f:
        return ModelShapes(json.load(f))
