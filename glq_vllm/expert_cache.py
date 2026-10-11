"""Residency policy for a VRAM-resident GLQ expert cache, and the scorer that sizes it.

**Why this exists.** On a 24 GB card the Flash-Next 3bpw checkpoint needs 40 GiB offloaded to
fit, and vLLM's `--cpu-offload-gb` is UVA — charged per byte *actually accessed*, so every
routed expert streams over PCIe every token. Measured on an L4: 6.517 tok/s, with a fitted
marginal cost of 55.4 ms per extra token position. Keeping the hottest experts resident is the
lever; this module is its policy half.

**The policy lives here, once, so the simulator and the live cache cannot disagree.**
`score_trace` replays a recorded routing trace through exactly the `ResidencyPolicy` the cache
will use, which is the only way a hit rate measured offline means anything about the cache that
gets built. The storage half (the device arena, slot fills, byte verification) is deliberately
separate: Strata splits them the same way, and the split is what let it score policies without
a GPU.

**★ Per-layer admission: Strata's headline finding does NOT transfer, and the reason is
structural.** Strata measured a globally-shared slot counter at **2.97%** hit rate (1781 of
60000) where the same routing scored **21.4%** at 8 slots/layer — because a position looks at
top_k experts in *every* layer, so with 256 slots against 48x10=480 lookups per position
*"the cache fills entirely inside the first position it ever sees and never changes again."*

Measured on Flash-Next's own decode routing (`benchmarks/expert_cache_s0.md`), the gap is
**0.5236 GLOBAL vs 0.5300 per-layer** — nothing. The pathology needs
`slots < n_layers * top_k`; a 10 GiB arena affords 5565 slots against 480, i.e. 11.6
positions, so the first position cannot monopolise it. `per_layer=True` stays the default
because it costs nothing and makes a layer's quota predictable — but it is **not** the lever
here, and anyone sizing an arena below ~1 GiB should re-check that claim rather than inherit
it.

**★ Eviction is the lever instead — and is mandatory for GLQ, unlike Strata.** Strata's
`admit` never evicts and sends misses to its CPU expert pool. GLQ cannot: its CPU MoE computes
in fp32 against CUDA's fp16 storage, and — the binding problem — reduces in a different order,
so a GPU/CPU split would make the output a function of cache warmth. Bit-exactness needs ONE
kernel call over ONE weight source, which means filling on miss, which means evicting.

That constraint turns out to be a gift: on real decode routing LRU moved the same 10 GiB arena
from **0.5300 to 0.8243**, and beat LFU at every size from 16 slots/layer up. On the synthetic
fixture in `tests/test_expert_cache_policy.py` LRU was **12x worse** than no eviction. Both are
real; neither was guessable, which is why `eviction=None` is kept as the baseline a policy has
to beat rather than removed.

**★ Scoring the wrong phase doubles the answer.** Any trace scored here must be decode-only.
vLLM's startup KV-sizing pass is a single synthetic max-model-len forward that was **89.3% of
the recorded trace** and routed to 495 of 24,576 (layer, expert) pairs; including it reported
0.9400 where decode's truth was 0.5263. `RoutingTrace.records(phase="decode")` is the filter,
and the scorer defaults to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Not resident. An int, not a bool, because the slot index is what the caller needs and a
#: table that must be re-derived from a bitmask gets re-derived differently in two places.
NOT_RESIDENT = -1

_EVICTION_POLICIES = (None, "lru", "lfu")


class ResidencyPolicy:
    """`(layer, expert) -> slot | NOT_RESIDENT`, plus admission and eviction.

    Pure Python: no torch, no device, so it runs in a CPU-only test and inside the offline
    scorer. The live cache wraps this and adds the device arena.
    """

    def __init__(self, n_layers: int, n_experts: int, slots: int,
                 per_layer: bool = True, eviction: str | None = None):
        if eviction not in _EVICTION_POLICIES:
            raise ValueError(
                f"eviction={eviction!r} is not one of {_EVICTION_POLICIES}. Refused rather "
                "than ignored: a silently-dropped policy would be scored as if it had run, "
                "and the hit rate is what everything downstream gets sized against.")
        self.n_layers = int(n_layers)
        self.n_experts = int(n_experts)
        self.slots = max(0, int(slots))
        self.per_layer = bool(per_layer)
        self.eviction = eviction

        self._residency: dict[tuple[int, int], int] = {}
        self._slot_owner: dict[int, tuple[int, int]] = {}
        self._next_free = 0                                  # global-counter mode
        self._layer_next = [self.layer_slot_range(li)[0] for li in range(self.n_layers)]
        self._clock = 0
        self._used: dict[tuple[int, int], int] = {}           # lru: last touch
        self._count: dict[tuple[int, int], int] = {}          # lfu: routing count

        self.hits = 0
        self.misses = 0
        self.admissions = 0
        self.evictions = 0

    # ---- ranges ----------------------------------------------------------

    def layer_slot_range(self, layer: int) -> tuple[int, int]:
        """Slots layer `layer` may admit into. Ported from Strata's `layer_slot_range`: the
        last layer absorbs the remainder, so no slot is unreachable."""
        if self.n_layers <= 0 or self.slots <= 0 or not 0 <= layer < self.n_layers:
            return (0, 0)
        if not self.per_layer:
            return (0, self.slots)
        q = self.slots // self.n_layers
        lo = layer * q
        hi = self.slots if layer == self.n_layers - 1 else (layer + 1) * q
        return (lo, hi)

    # ---- lookup ----------------------------------------------------------

    def slot_of(self, layer: int, expert: int) -> int:
        """Bounds-checked, like Strata's: a bad index returns NOT_RESIDENT rather than
        reading whatever is adjacent in the table."""
        if not (0 <= layer < self.n_layers and 0 <= expert < self.n_experts):
            return NOT_RESIDENT
        return self._residency.get((layer, expert), NOT_RESIDENT)

    def touch(self, layer: int, expert: int) -> None:
        """Record a use, for the eviction policy's recency/frequency signal."""
        key = (layer, expert)
        self._clock += 1
        self._used[key] = self._clock
        self._count[key] = self._count.get(key, 0) + 1

    # ---- admission -------------------------------------------------------

    def admit(self, layer: int, expert: int) -> int | None:
        """Claim a slot for `(layer, expert)`. Returns the slot, or None when the layer's
        quota is full and no eviction policy is set."""
        if not (0 <= layer < self.n_layers and 0 <= expert < self.n_experts):
            return None
        key = (layer, expert)
        if key in self._residency:
            return self._residency[key]

        lo, hi = self.layer_slot_range(layer)
        if hi <= lo:
            return None

        if self.per_layer:
            nxt = self._layer_next[layer]
            if nxt < hi:
                self._layer_next[layer] = nxt + 1
                return self._claim(key, nxt)
        else:
            # The pathology, kept faithfully: ONE counter for every layer.
            if self._next_free < self.slots:
                slot = self._next_free
                self._next_free += 1
                return self._claim(key, slot)

        if self.eviction is None:
            return None
        candidate = self._pick_eviction_candidate(lo, hi)
        if candidate is None:
            return None
        self._evict(candidate)
        return self._claim(key, candidate)

    def _claim(self, key: tuple[int, int], slot: int) -> int:
        self._residency[key] = slot
        self._slot_owner[slot] = key
        self.admissions += 1
        self.touch(*key)
        return slot

    def _evict(self, slot: int) -> None:
        owner = self._slot_owner.pop(slot, None)
        if owner is not None:
            self._residency.pop(owner, None)
            self.evictions += 1

    def _pick_eviction_candidate(self, lo: int, hi: int) -> int | None:
        """Least-recently-used or least-frequently-used slot **within this layer's range
        only** — evicting across a layer boundary would corrupt another layer's residency."""
        best, best_key = None, None
        for slot in range(lo, hi):
            owner = self._slot_owner.get(slot)
            if owner is None:
                return slot                              # an unclaimed slot needs no candidate
            rank = (self._used.get(owner, 0) if self.eviction == "lru"
                    else self._count.get(owner, 0))
            if best_key is None or rank < best_key:
                best, best_key = slot, rank
        return best

    # ---- the one call the cache and the scorer share ---------------------

    def lookup_or_admit(self, layer: int, expert: int) -> tuple[int, bool]:
        """`(slot, was_hit)`. The single entry point, so hit accounting cannot drift between
        the simulator and the live cache."""
        slot = self.slot_of(layer, expert)
        if slot != NOT_RESIDENT:
            self.hits += 1
            self.touch(layer, expert)
            return slot, True
        self.misses += 1
        admitted = self.admit(layer, expert)
        return (admitted if admitted is not None else NOT_RESIDENT), False

    # ---- step semantics, which is what the live arena must use ----------

    def _slot_last_use(self, slot: int) -> int:
        """Use stamp of whatever occupies `slot`; 0 for never-claimed.

        0 sorts first, so an empty slot is always preferred over an occupied one — the same
        preference `_pick_eviction_candidate` expresses by returning an unclaimed slot immediately.
        """
        owner = self._slot_owner.get(slot)
        return self._used.get(owner, 0) if owner is not None else 0

    def _slot_count(self, slot: int) -> int:
        """Routing count of whatever occupies `slot`; 0 for never-claimed (so it sorts
        first, same preference as `_slot_last_use`)."""
        owner = self._slot_owner.get(slot)
        return self._count.get(owner, 0) if owner is not None else 0

    def plan_step(self, layer: int, expert_ids) -> tuple[list[int], list[int]]:
        """Resolve a whole step's routings at once: `(slot_of_routing, experts_to_fill)`.

        **Why this exists next to `lookup_or_admit`, rather than replacing it.** The two are
        genuinely different policies, and only this one is safe for the arena:

        `lookup_or_admit` admits one routing at a time, and can hand out as a candidate a slot
        that a *later routing of the same step* is about to read. Offline that merely costs a
        hit (measured: slots=2, warm {A,B}, step `[C, A]` scores **0 hits / 2 misses** where
        this method scores 1/1). **Live it would be a correctness bug**: the fused op reads the
        arena after the fill, so the matmul would read a slot holding a different expert —
        plausible output from the wrong weights.

        So a step's hit slots are protected from candidate selection, misses are admitted in
        **first-appearance order** (the order `lookup_or_admit` would have used), and one slot
        serves every routing of a repeated expert. Because protection strictly avoids evicting
        a would-be hit, this policy's hit rate is **>= the per-routing one** — S0's 0.8233 is
        therefore a conservative floor for what the arena achieves.
        """
        ids = [int(e) for e in expert_ids]
        lo, hi = self.layer_slot_range(layer)
        if hi - lo < len(ids):
            raise ValueError(
                f"{len(ids)} routings in one step cannot be served by {hi - lo} slots for "
                f"layer {layer}: every expert routed in a step must be resident "
                f"simultaneously, so a larger step would evict a routing before it is read.")

        slots = [self.slot_of(layer, e) for e in ids]
        protected = {s for s in slots if s != NOT_RESIDENT}

        wanted: dict[int, int] = {}                 # expert -> first routing position
        for pos, (e, s) in enumerate(zip(ids, slots)):
            if s == NOT_RESIDENT and e not in wanted:
                wanted[e] = pos

        if wanted:
            free = [sl for sl in range(lo, hi)
                    if sl not in protected and sl not in self._slot_owner]
            if self.eviction is None:
                # Scoring-only configuration: the baseline a real policy has to beat. The
                # live arena cannot run this — exactness needs every routed expert in the
                # arena, so a routing that gets no slot has nowhere to read from.
                cands = [(0, sl) for sl in free]
            else:
                rank = (self._slot_last_use if self.eviction == "lru"
                        else self._slot_count)
                cands = sorted((rank(sl), sl) for sl in range(lo, hi)
                               if sl not in protected)
            for (e, _pos), (_u, candidate) in zip(wanted.items(), cands):
                owner = self._slot_owner.pop(candidate, None)
                if owner is not None:
                    self._residency.pop(owner, None)
                    self.evictions += 1
                key = (layer, e)
                self._residency[key] = candidate
                self._slot_owner[candidate] = key
                self.admissions += 1

        slots = [self.slot_of(layer, e) for e in ids]
        # Stamp in routing order so intra-step recency is defined, and a repeated expert's
        # stamp is its LAST use.
        for e in ids:
            self.touch(layer, e)
        missed = sum(1 for e in ids if e in wanted)
        self.hits += len(ids) - missed
        self.misses += missed
        return slots, list(wanted)

    @property
    def admitted_per_layer(self) -> list[int]:
        out = [0] * self.n_layers
        for (layer, _), _slot in self._residency.items():
            out[layer] += 1
        return out


@dataclass
class ScoreResult:
    """What a scored trace says. `hit_rate` is over expert LOOKUPS, matching Strata's
    denominator ("1781 of 60000" is 125 tokens x 48 layers x 10 experts) — a per-token
    denominator would rescale every number we compare against."""
    slots: int
    per_layer: bool
    eviction: str | None
    lookups: int
    hits: int
    misses: int
    admissions: int
    evictions: int
    admitted_per_layer: list[int] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return (self.hits / self.lookups) if self.lookups else 0.0

    def __str__(self) -> str:
        pol = f"{'per-layer' if self.per_layer else 'GLOBAL  '} evict={self.eviction or 'none'}"
        return (f"slots={self.slots:<6} {pol}  hit_rate={self.hit_rate:7.4f}  "
                f"hits={self.hits:<8} misses={self.misses:<8} "
                f"evictions={self.evictions:<8} "
                f"min/max admitted per layer={min(self.admitted_per_layer, default=0)}/"
                f"{max(self.admitted_per_layer, default=0)}")


def score_trace(trace, n_layers: int, n_experts: int, slots: int,
                per_layer: bool = True, eviction: str | None = None,
                step_mode: bool = False) -> ScoreResult:
    """Replay `trace` — an iterable of `(layer, [expert ids])` — through `ResidencyPolicy`.

    This is S0: it answers "what hit rate would a cache of this size reach on GLQ's actual
    routing?" before any cache exists, which is how Strata sized its own design
    (`Memory/cache_allocation.py` scored a run's routing offline). Running the real policy
    rather than a model of it is the point.
    """
    pol = ResidencyPolicy(n_layers=n_layers, n_experts=n_experts, slots=slots,
                          per_layer=per_layer, eviction=eviction)
    lookups = 0
    if step_mode:
        # What the LIVE arena does: resolve a whole record's routings at once, protecting the
        # slots this step reads. Strictly >= the per-routing rate, because the per-routing
        # form can evict a slot a later routing of the same step needs. Measured on real
        # decode routing: +0.0335 at 16 slots/layer, +0.0014 at 120 — the gap closes as the
        # arena grows, since the pathology needs a tight cache to bite.
        for layer, ids in trace:
            lookups += len(ids)
            pol.plan_step(int(layer), ids)
        return ScoreResult(slots=slots, per_layer=per_layer, eviction=eviction,
                           lookups=lookups, hits=pol.hits, misses=pol.misses,
                           admissions=pol.admissions, evictions=pol.evictions,
                           admitted_per_layer=pol.admitted_per_layer)
    for layer, ids in trace:
        for e in ids:
            lookups += 1
            pol.lookup_or_admit(int(layer), int(e))
    return ScoreResult(slots=slots, per_layer=per_layer, eviction=eviction,
                       lookups=lookups, hits=pol.hits, misses=pol.misses,
                       admissions=pol.admissions, evictions=pol.evictions,
                       admitted_per_layer=pol.admitted_per_layer)
