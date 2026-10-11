"""Record MoE routing so an expert cache can be SIZED before it is built (S0).

Strata sized its cache this way rather than by reasoning — it scored a real run's routing
offline (`Memory/cache_allocation.py`) and found its cache was *"never short of capacity"*, it
was handing all of it to two layers. That diagnosis came from a trace, not from a plan.

This module records `(layer, topk_ids)` per MoE forward and nothing else: no compute change,
no allocation, and **completely inert unless `GLQ_MOE_TRACE_ROUTING` names an output path**.
Its output feeds `glq_vllm.expert_cache.score_trace` directly, so the two halves of S0 cannot
drift apart through a conversion step.

Usage::

    GLQ_MOE_TRACE_ROUTING=/opt/dlami/nvme/routing.npz python benchmarks/run_model.py ...
    python benchmarks/_expert_cache_score.py /opt/dlami/nvme/routing.npz
"""
from __future__ import annotations

import atexit
import json
import os

#: Set to a path to record. Unset (the default) means this module does nothing at all.
TRACE_ENV = "GLQ_MOE_TRACE_ROUTING"

#: Phases a trace can be sliced into, by the width of the forward each record came from.
#: ``decode`` is width 1 and is **the phase that decides decode tok/s** — see
#: `RoutingTrace.records` for why scoring ``all`` is actively misleading.
PHASES = ("all", "decode", "prefill")

_TRACE: RoutingTrace | None = None


def layer_key(prefix: str | None) -> str:
    """A stable, COLLISION-FREE key for one MoE layer.

    The full prefix, deliberately, not a parsed layer index. vLLM renumbers the MTP block to
    ``mtp.layers.48`` while the checkpoint calls it ``mtp.layers.0`` — and decoder layer 0
    also exists — so keying on the bare index would put two unrelated expert sets in one
    residency row, where they would evict each other.

    A caller with no prefix gets a single lumped key rather than an exception: a degraded
    trace is more useful than a crashed run, and the lumping is visible in the layer count.
    """
    return prefix if prefix else "<unknown-layer>"


class RoutingTrace:
    """Append-only `(layer_key, expert_ids)` records, saved as one ``.npz``."""

    def __init__(self, meta: dict | None = None, path: str | None = None,
                 flush_every: int = 0):
        #: Where to flush, and how often. Flushing PERIODICALLY rather than only at exit is
        #: not belt-and-braces: `apply` runs in vLLM's EngineCore **subprocess**, whose
        #: shutdown path force-kills it ("[shutdown] Process manager: force killing remaining
        #: process EngineCore"), and `atexit` does not survive SIGKILL. Measured: a full
        #: RUN_OK trace run wrote nothing at all.
        self._path = path
        self._flush_every = max(0, int(flush_every))
        self._since_flush = 0
        self._keys: list[str] = []
        self._index: dict[str, int] = {}
        self._layer: list[int] = []
        self._ids: list[list[int]] = []
        #: Tokens in the forward each record came from. Recorded, not inferred, because it is
        #: the ONLY thing separating a decode step from a prefill and the hit rate differs by
        #: ~2x between them.
        self._width: list[int] = []
        self._meta = dict(meta or {})
        self.meta_json = "{}"
        #: True when `_width` was reconstructed from contiguous same-layer runs rather than
        #: read from the file. An inference, so it is reported as one.
        self.widths_were_derived = False

    # ---- recording -------------------------------------------------------

    def record(self, prefix: str | None, topk_ids) -> None:
        """Append one forward's routed expert ids.

        ``topk_ids`` may be a torch tensor ``(tokens, top_k)``, a nested sequence, or a flat
        sequence. A 2-D input becomes ONE RECORD PER TOKEN, because the cache is asked about
        each token's routed set separately and a flattened batch would misstate the lookups.
        """
        key = layer_key(prefix)
        li = self._index.get(key)
        if li is None:
            li = len(self._keys)
            self._index[key] = li
            self._keys.append(key)

        rows = _as_rows(topk_ids)
        w = len(rows)
        for row in rows:
            self._layer.append(li)
            self._ids.append(row)
            self._width.append(w)
            self._since_flush += 1

        if self._path and self._flush_every and self._since_flush >= self._flush_every:
            self._since_flush = 0
            try:
                self.save(self._path)
            except Exception as e:                        # noqa: BLE001 - never break a run
                print(f"glq: routing-trace flush to {self._path} failed: "
                      f"{type(e).__name__}: {e}", flush=True)

    # ---- reading ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def n_layers(self) -> int:
        return len(self._keys)

    @property
    def n_experts(self) -> int:
        """One past the largest expert id seen. The true expert count may be higher; the
        scorer only needs an upper bound for its tables, and a trace cannot know what was
        never routed."""
        return (max((max(r) for r in self._ids if r), default=-1) + 1) if self._ids else 0

    @property
    def layer_keys(self) -> list[str]:
        return list(self._keys)

    @property
    def widths(self) -> list[int]:
        """Tokens in the forward each record came from."""
        return list(self._width)

    def phase_breakdown(self) -> dict[int, int]:
        """`forward width -> records`, which is how a trace's composition is read at a
        glance. On the real L4 trace this is what exposed the profiling run."""
        out: dict[int, int] = {}
        for w in self._width:
            out[w] = out.get(w, 0) + 1
        return dict(sorted(out.items()))

    def records(self, phase: str = "all") -> list[tuple[int, list[int]]]:
        """`(layer_index, ids)` for one phase.

        **★ `phase="decode"` is the number that decides the cache, and `"all"` is actively
        misleading.** vLLM runs a synthetic max-model-len forward at startup to size its KV
        pool. Measured on Flash-Next at `--max-model-len 4096`: that single profiling pass
        contributed **196,608 of 220,211 records — 89.3% of the trace** — and because its
        input is dummy data it routed to just **495** distinct (layer, expert) pairs out of
        24,576, against 14,109 for real decode. A cache nails a 495-pair working set, so
        including it reported **0.9400** where decode's truth was **0.5263**.

        So the split is not a convenience. A scorer whose default is `"all"` reports a
        number that is ~80% relative too high and looks like a decided question.
        """
        if phase not in PHASES:
            raise ValueError(
                f"phase={phase!r} is not one of {PHASES}. Refused rather than treated as "
                "'all', because 'all' is the misleading answer here: it is dominated by "
                "vLLM's dummy profiling forward.")
        pairs = zip(self._layer, self._ids, self._width)
        if phase == "decode":
            return [(li, ids) for li, ids, w in pairs if w == 1]
        if phase == "prefill":
            return [(li, ids) for li, ids, w in pairs if w > 1]
        return [(li, ids) for li, ids, _ in pairs]

    def scored_records(self, phase: str = "all"):
        """Exactly what `expert_cache.score_trace` consumes — `(layer_index, ids)`."""
        return self.records(phase)

    # ---- persistence -----------------------------------------------------

    def save(self, path: str, compress: bool = False) -> None:
        """Write atomically: a reader (the scorer) never sees half a file.

        `compress` defaults off because the periodic flush re-saves the whole file inside
        vLLM's EngineCore process, where CPU time competes with decode. For an artifact kept
        in the repo it is worth turning on: measured 9.70 MB -> 0.35 MB on the L4 trace, with
        the arrays byte-equal.
        """
        import numpy as np

        meta = dict(self._meta)
        meta.update(n_records=len(self._ids), n_layers=self.n_layers,
                    n_experts=self.n_experts, layer_keys=self._keys,
                    # Carried in meta, not inferred from the file's shape: once derived
                    # widths are written out they are indistinguishable from recorded ones,
                    # and a resave must not launder an inference into a fact.
                    widths_derived=bool(self.widths_were_derived))
        width = max((len(r) for r in self._ids), default=0)
        # Ragged top_k would mean a routing change mid-run; pad with -1 so it is visible
        # rather than silently reshaped.
        ids = np.full((len(self._ids), width), -1, dtype=np.int32)
        for i, row in enumerate(self._ids):
            ids[i, :len(row)] = row

        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        # np.savez APPENDS .npz unless the name already ends in it, so name the temp
        # explicitly; otherwise os.replace moves a file that does not exist and a .tmp.npz
        # is left behind for the scorer to trip over.
        tmp = path + ".tmp.npz"
        writer = np.savez_compressed if compress else np.savez
        writer(tmp, layer=np.asarray(self._layer, dtype=np.int32), ids=ids,
               width=np.asarray(self._width, dtype=np.int32), meta=json.dumps(meta))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> RoutingTrace:
        import numpy as np

        with np.load(path, allow_pickle=False) as z:
            layer = z["layer"]
            ids = z["ids"]
            width = z["width"] if "width" in z.files else None
            meta_json = str(z["meta"])
        meta = json.loads(meta_json)
        t = cls(meta={k: v for k, v in meta.items() if k != "layer_keys"})
        t._keys = list(meta.get("layer_keys", []))
        t._index = {k: i for i, k in enumerate(t._keys)}
        t._layer = [int(x) for x in layer]
        t._ids = [[int(v) for v in row if v >= 0] for row in ids]
        if width is None:
            t._width = _derive_widths(t._layer)
            t.widths_were_derived = True
        else:
            t._width = [int(x) for x in width]
            # A file written by a resave of a legacy trace carries recorded-looking widths
            # that were in fact derived; only meta knows.
            t.widths_were_derived = bool(meta.get("widths_derived", False))
        t.meta_json = meta_json
        return t


def _derive_widths(layer: list[int]) -> list[int]:
    """Reconstruct forward widths for a trace recorded before `width` existed.

    One forward appends a contiguous run of records all carrying the same layer, so a run's
    length is that forward's token count. Exact for any run where no forward carries a single
    token *and* is adjacent to another forward of the same layer — true of every benchmark
    driver here, and false in general for a 1-token prompt, which is why the caller flags the
    result as derived rather than recorded.
    """
    out: list[int] = []
    i, n = 0, len(layer)
    while i < n:
        j = i + 1
        while j < n and layer[j] == layer[i]:
            j += 1
        out.extend([j - i] * (j - i))
        i = j
    return out


def _as_rows(topk_ids) -> list[list[int]]:
    """`(tokens, top_k)` -> one list per token; a flat sequence -> a single row."""
    if hasattr(topk_ids, "detach"):                       # torch tensor
        t = topk_ids.detach().to("cpu")
        if t.dim() == 1:
            return [[int(v) for v in t.tolist()]]
        return [[int(v) for v in row] for row in t.tolist()]
    if not len(topk_ids):
        return []
    first = topk_ids[0]
    if isinstance(first, (list, tuple)):
        return [[int(v) for v in row] for row in topk_ids]
    return [[int(v) for v in topk_ids]]


def active_trace() -> RoutingTrace | None:
    """The one shared trace, or None when `GLQ_MOE_TRACE_ROUTING` is unset.

    Shared on purpose: every MoE layer appends to the same object, because the hit rate is a
    property of the whole run's routing. A trace per layer would have to be stitched back
    together, and would lose the interleaving that makes a global slot counter fail.
    """
    global _TRACE
    path = os.environ.get(TRACE_ENV)
    if not path:
        return None
    if _TRACE is None:
        every = int(os.environ.get("GLQ_MOE_TRACE_FLUSH_EVERY", "2000"))
        _TRACE = RoutingTrace(meta={"path": path}, path=path, flush_every=every)
        atexit.register(_flush, path)      # the tail, for a process that exits cleanly
    return _TRACE


def _flush(path: str) -> None:
    """Write at interpreter exit. Best-effort and loud on failure: a trace silently lost
    after a multi-minute run is worse than a message."""
    global _TRACE
    if _TRACE is None or not len(_TRACE):
        return
    try:
        _TRACE.save(path)
        print(f"glq: wrote routing trace {path} "
              f"({len(_TRACE)} records, {_TRACE.n_layers} layers)", flush=True)
    except Exception as e:                                # noqa: BLE001 - exit-time
        print(f"glq: FAILED to write routing trace {path}: {type(e).__name__}: {e}",
              flush=True)
