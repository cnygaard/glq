"""VRAM expert arena: keep the hottest MoE experts resident instead of streaming every
routed expert over PCIe each token (S1).

**Why.** vLLM's `--cpu-offload-gb` is UVA, charged per byte *accessed*, and it chooses what to
offload with no knowledge of routing — measured on an L4, its effective hit rate is **0.055**,
resident purely by luck. A routing-aware arena of the same size reaches **0.8233**
(`benchmarks/expert_cache_s0.md`), and offload is **45.1%** of that card's 153.35 ms decode
step (`benchmarks/l4_offload_slope/`), so the predicted gain is **1.58x** against a hard
ceiling of 1.82x.

**Why no CUDA kernel change.** `glq/csrc/glq_cuda.cu` reaches a per-expert weight as
``qidxs_base + (long)eidx * qidxs_estride``. Hand it the arena as the base and a *slot* id as
``eidx`` and the already-compiled kernel reads the right bytes. The arena is layout-identical
to a slice of the source (`moe_shapes.arena_shapes`), so only the address changes.

**Why that is bit-exact.** The final reduce,
``glq_moe_weighted_scatter_reduce_kernel``, is a *"fixed-order k loop per output element ->
deterministic (no atomicAdd)"*: ``acc += topk_weights[r] * expert_out[inv_perm[r]]`` for k
ascending. Substituting slots changes *where* a routing's weights are read from, never the
accumulation order, and the arena holds a verbatim byte copy. One kernel call, one weight
source.

**Why eviction is mandatory, and a static hot set is not an option.** Because exactness needs
one weight source, every routed expert must be in the arena at call time — so a miss must be
*filled*, which means evicting. A per-routing fallback would need two sources in one call; a
whole-layer fallback would fire whenever any of top_k=10 missed, i.e. almost every step. S0
measured the policy rather than guessing it: LRU takes a 10 GiB arena from 0.5263 to 0.8233
and beats LFU above 16 slots/layer, while on synthetic routing LRU was *12x worse* than no
eviction. Different regimes; neither was guessable.
"""
from __future__ import annotations

from dataclasses import dataclass

from .moe_shapes import arena_shapes, bytes_per_slot, slots_for_budget

#: Env var holding the arena budget in GiB. Unset = off, and the existing path is untouched.
BUDGET_ENV = "GLQ_MOE_EXPERT_CACHE"
#: Token cap for the arena path. Static, so each captured CUDA graph takes one branch.
MAX_TOKENS_ENV = "GLQ_MOE_ARENA_MAX_TOKENS"
DEFAULT_MAX_TOKENS = 8

#: Below this, offload is rounding rather than a workload and an arena cannot pay for itself.
_MIN_OFFLOAD_BYTES = 1024**3


@dataclass(frozen=True)
class ArenaDecision:
    """Whether to build an arena for one layer, and — when not — why not.

    The reason is carried rather than logged-and-dropped because a cache that is silently off
    returns perfectly correct tokens, just slowly. That is how Strata's global-counter bug
    survived at a 2.97% hit rate, and it is why `fused_moe_method` prints this.
    """
    enabled: bool
    slots: int = 0
    bytes_per_slot: int = 0
    n_layers: int = 0
    reason: str = ""

    @property
    def total_bytes(self) -> int:
        return self.slots * self.bytes_per_slot * self.n_layers


def _parse_budget(budget: str) -> float | None:
    """GiB from the env string. Accepts ``10``, ``10GiB``, ``10 GiB``, ``10.0``.

    A value that is set but unparseable raises rather than defaulting to off: an arm that
    measured the baseline while reporting that it tested the cache is worse than a crash.
    """
    s = str(budget).strip()
    if not s:
        return None
    s = s.removesuffix("GiB").removesuffix("gib").removesuffix("GB").removesuffix("G").strip()
    try:
        return float(s)
    except ValueError:
        raise ValueError(
            f"{BUDGET_ENV}={budget!r} is not a size in GiB. Use e.g. "
            f"{BUDGET_ENV}=10, or unset it to disable the expert arena. Refused rather than "
            f"treated as 'off', because a disabled cache still produces correct output and "
            f"the run would silently measure the baseline.") from None


def arena_decision(*, budget: str, offloaded_bytes: int, n_layers: int, n_experts: int,
                   top_k: int, hidden_size: int, inter: int, w13_out: int, bpw: int,
                   max_tokens: int = DEFAULT_MAX_TOKENS) -> ArenaDecision:
    """Decide whether this layer gets an arena, and how many slots.

    Refuses — with a reason — when:

    * the budget is unset (the default);
    * little or nothing is offloaded, so there is no PCIe traffic to remove and the arena
      would be a second copy of weights already on the card;
    * the budget cannot hold ``max_tokens * top_k`` slots, which is a **correctness**
      precondition rather than a tuning one (see `slots_for_budget`).
    """
    gib = _parse_budget(budget)
    if gib is None or gib <= 0:
        return ArenaDecision(False, reason=f"{BUDGET_ENV} unset: expert arena disabled")

    if int(offloaded_bytes) < _MIN_OFFLOAD_BYTES:
        return ArenaDecision(
            False, reason=(
                f"only {offloaded_bytes / 1024**3:.2f} GiB of experts are offloaded, so "
                f"there is no PCIe traffic for an arena to remove; it would just be a "
                f"second copy of resident weights. Expert arena disabled."))

    per = bytes_per_slot(hidden_size=hidden_size, inter=inter, w13_out=w13_out, bpw=bpw)
    try:
        slots = slots_for_budget(int(gib * 1024**3), n_layers=n_layers,
                                 hidden_size=hidden_size, inter=inter, w13_out=w13_out,
                                 bpw=bpw, n_experts=n_experts, max_tokens=max_tokens,
                                 top_k=top_k)
    except ValueError as e:
        return ArenaDecision(False, reason=f"expert arena disabled: {e}")

    return ArenaDecision(True, slots=slots, bytes_per_slot=per, n_layers=n_layers,
                         reason=(f"expert arena: {slots} slots/layer x {n_layers} layers = "
                                 f"{slots * per * n_layers / 1024**3:.2f} GiB, "
                                 f"{per / 1024**2:.3f} MiB/slot"))


def arena_decision_from_env(env, **kw) -> ArenaDecision:
    """`arena_decision` with the two env vars read from a mapping (``os.environ``)."""
    return arena_decision(
        budget=env.get(BUDGET_ENV, ""),
        max_tokens=int(env.get(MAX_TOKENS_ENV, DEFAULT_MAX_TOKENS)),
        **kw)


def arena_buffer_shapes(decision: ArenaDecision, *, hidden_size: int, inter: int,
                        w13_out: int, bpw: int) -> dict[str, tuple[int, ...]]:
    """Device buffers to allocate for one layer's arena. Empty when disabled."""
    if not decision.enabled:
        return {}
    return arena_shapes(decision.slots, hidden_size=hidden_size, inter=inter,
                        w13_out=w13_out, bpw=bpw)


# ---------------------------------------------------------------------------
# The fill kernel
# ---------------------------------------------------------------------------
#
# **Why a kernel and not `index_copy_`.** A fixed-shape
# ``arena.index_copy_(0, slots, src.index_select(0, experts))`` is device-side and
# graph-capturable, but its row count is static, so it would copy *every* routing's blob every
# step whether that routing hit or missed — the same PCIe traffic as today, and zero saving.
# Skipping the hits needs a data-dependent row count, which needs a host sync, which breaks
# the graph capture the fused MoE op exists to preserve.
#
# A kernel resolves that: the **launch** is fixed-size (graph-safe) while the **work** is
# dynamic. A program whose routing hit returns before issuing any load, so no UVA page fault
# is taken and no bytes cross the link. That early return *is* the optimisation.

def _fill_kernel():
    """The Triton row-gather, compiled on first use.

    Deliberately no `@triton.autotune`: CLAUDE.md forbids capture-time autotune under CUDA
    graphs, and an autotuner would also make the first captured replay measure a different
    kernel from the one it recorded.
    """
    global _FILL
    if _FILL is not None:
        return _FILL
    import triton
    import triton.language as tl

    @triton.jit
    def _gather_rows(src_ptr, dst_ptr, fill_src_ptr, slot_ptr, n_elems,
                     BLOCK: tl.constexpr):
        r = tl.program_id(0)
        e = tl.load(fill_src_ptr + r)
        # ★ The saving. `fill_src[r] < 0` means routing r is already resident, and returning
        # here issues no load from `src_ptr` — which is host memory over PCIe when the
        # experts are offloaded.
        if e >= 0:
            s = tl.load(slot_ptr + r)
            off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
            mask = off < n_elems
            # int64 offsets: a layer's packed tensor is ~314M elements at 512 experts, and
            # the product with a slot index must not wrap.
            src = src_ptr + e.to(tl.int64) * n_elems + off
            dst = dst_ptr + s.to(tl.int64) * n_elems + off
            tl.store(dst, tl.load(src, mask=mask), mask=mask)

    _FILL = _gather_rows
    return _FILL


_FILL = None
#: Elements per program. Fixed, for the reasons in `_fill_kernel`.
FILL_BLOCK = 1024


# ---------------------------------------------------------------------------
# Residency state and the per-step plan
# ---------------------------------------------------------------------------

#: Mirrors `expert_cache.NOT_RESIDENT`; re-stated so the device side has no import-time
#: dependency on the policy module during a forward.
_NOT_RESIDENT = -1


class ArenaState:
    """Device-side residency for one layer's arena: `(layer, expert) -> slot`, plus LRU.

    Deliberately small and all on device, because `apply` must not host-sync — the fused MoE
    op exists precisely so the decode step stays capturable under FULL cudagraph, and the
    fallback loop's `topk_ids.unique()` is what forces `--enforce-eager`.
    """

    def __init__(self, n_experts: int, n_slots: int, device="cuda"):
        import torch

        self.n_experts = int(n_experts)
        self.n_slots = int(n_slots)
        self.device = device
        i32 = torch.int32
        self.residency = torch.full((self.n_experts,), _NOT_RESIDENT, dtype=i32,
                                    device=device)
        self.slot_owner = torch.full((self.n_slots,), _NOT_RESIDENT, dtype=i32,
                                     device=device)
        #: Monotone use stamp per slot. int64 so the clock cannot wrap inside a run: the
        #: tie-break below folds the slot index into this key.
        self.slot_last_use = torch.zeros((self.n_slots,), dtype=torch.int64, device=device)
        self.clock = torch.zeros((), dtype=torch.int64, device=device)
        #: hits / misses / fills, reported so a run can prove the cache engaged. A cache at a
        #: 0% hit rate still emits correct tokens, so the counters are the only evidence.
        self.counters = torch.zeros((3,), dtype=torch.int64, device=device)

    # counters as scalars, for reporting and tests
    @property
    def hits(self):
        return int(self.counters[0])

    @property
    def misses(self):
        return int(self.counters[1])

    @property
    def fills(self):
        return int(self.counters[2])

    def hit_rate(self) -> float:
        h, m = self.hits, self.misses
        return h / (h + m) if (h + m) else 0.0


class StepPlan:
    """`slot_of[r]` for every routing, and `fill_src[r]` = expert to copy (or -1 for a hit)."""

    __slots__ = ("slot_of", "fill_src")

    def __init__(self, slot_of, fill_src):
        self.slot_of = slot_of
        self.fill_src = fill_src


def plan_step(state: ArenaState, expert_ids) -> StepPlan:
    """Resolve one step's routings against the arena, admitting misses by LRU.

    **Vectorized, and provably the same selection as the serial `ResidencyPolicy`.** A serial
    admit evicts the least-recently-used slot and then *touches* it, so it is no longer least
    recent and the next admit takes the second-least: serial admission of M misses selects
    exactly the M least-recently-used slots. This takes those M in one shot, assigning them to
    misses in routing order — the same pairing. `tests/test_expert_arena_plan.py` asserts the
    agreement rather than trusting this paragraph.

    Two invariants the fused op depends on, because it reads the arena *after* the fill:

    * distinct experts in a step get distinct slots, and
    * nothing admitted in a step may evict anything else from that same step.

    Both follow from excluding this step's hit slots from candidacy outright, together with
    the `n_slots >= max_tokens * top_k` precondition. Relying on recency alone would not be
    enough: a slot last used long ago but read by *this* step is exactly the one an LRU scan
    would pick.
    """
    import torch

    ids = expert_ids.to(torch.int64).reshape(-1)
    n_routings = ids.numel()
    if n_routings > state.n_slots:
        raise ValueError(
            f"{n_routings} routings in one step cannot be served by {state.n_slots} slots: "
            f"every expert routed in a step must be resident SIMULTANEOUSLY, so a step "
            f"larger than the arena would evict a routing before the matmul reads it. Raise "
            f"{BUDGET_ENV} or lower {MAX_TOKENS_ENV}.")

    slot = state.residency[ids].to(torch.int64)                      # -1 where missing
    is_hit = slot >= 0

    # ★ Protect this step's hit slots from candidate selection *explicitly*, rather than by
    # stamping them first. A slot read by any routing of this step must survive the step: the
    # fused op reads the arena after the fill, so evicting one would leave the matmul reading
    # a slot that holds a different expert. `ResidencyPolicy.plan_step` protects the same set,
    # which is why the two agree; `lookup_or_admit` does NOT, and must not be used here.
    protected = torch.zeros(state.n_slots, dtype=torch.bool, device=ids.device)
    if bool(is_hit.any()):
        protected[slot[is_hit]] = True

    # One fill per DISTINCT missing expert: two tokens routing to the same cold expert must
    # share a slot. Separate slots would still be bit-exact (verbatim copies) but would pay
    # PCIe twice and burn a slot.
    miss_ids = ids[~is_hit]
    uniq, inverse = torch.unique(miss_ids, sorted=True, return_inverse=True)
    n_new = uniq.numel()

    # Admit in FIRST-APPEARANCE order, not sorted-expert order. The serial policy admits
    # routing by routing, so routing 0's expert takes the first candidate; `torch.unique` hands
    # back sorted ids, which would pair candidates with experts by id instead and disagree with
    # the policy on every step whose routings are not already ascending.
    if n_new:
        first_pos = torch.zeros(n_new, dtype=torch.int64, device=ids.device)
        appear = torch.arange(inverse.numel(), device=ids.device, dtype=torch.int64)
        first_pos.scatter_reduce_(0, inverse, appear, reduce="amin", include_self=False)
        order = torch.argsort(first_pos)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(n_new, device=ids.device, dtype=torch.int64)
        uniq = uniq[order]
        inverse = rank[inverse]
        first_pos = first_pos[order]

    fill_src = torch.full((n_routings,), _NOT_RESIDENT, dtype=torch.int32,
                          device=ids.device)
    if n_new:
        # Eviction candidates: the n_new least-recently-used UNPROTECTED slots. The key folds the slot
        # index in so ties break by ascending index, matching ResidencyPolicy's lo..hi scan
        # (which hands out never-claimed slots in index order). Protected slots are pushed
        # past every candidate rather than removed, so the shape stays static for Triton.
        idx = torch.arange(state.n_slots, device=ids.device, dtype=torch.int64)
        key = state.slot_last_use * state.n_slots + idx
        key = torch.where(protected, key + (1 << 62), key)
        candidates = torch.argsort(key)[:n_new]

        evicted = state.slot_owner[candidates].to(torch.int64)
        still_owned = evicted >= 0
        if bool(still_owned.any()):
            state.residency[evicted[still_owned]] = _NOT_RESIDENT

        state.residency[uniq] = candidates.to(torch.int32)
        state.slot_owner[candidates] = uniq.to(torch.int32)

        new_slots = candidates[inverse]
        slot = slot.clone()
        slot[~is_hit] = new_slots
        # Mark the fill on the FIRST routing that needs each expert; duplicates read the same
        # slot and must not trigger a second copy of identical bytes.
        miss_pos = torch.nonzero(~is_hit, as_tuple=True)[0]
        fill_src[miss_pos[first_pos]] = uniq.to(torch.int32)

    # Stamp in ROUTING order, so intra-step recency is defined the same way the reference
    # defines it and a repeated expert's stamp is its LAST use. `amax` makes the duplicate
    # case deterministic (highest routing position wins) rather than scatter-order dependent.
    base = state.clock + 1
    stamps = base + torch.arange(n_routings, device=ids.device, dtype=torch.int64)
    state.slot_last_use.scatter_reduce_(0, slot, stamps, reduce="amax",
                                        include_self=True)
    state.clock = base + n_routings - 1

    state.counters[0] += int(is_hit.sum())
    state.counters[1] += int((~is_hit).sum())
    state.counters[2] += n_new
    return StepPlan(slot.to(torch.int32), fill_src)


# ---------------------------------------------------------------------------
# The arena itself
# ---------------------------------------------------------------------------

class ExpertArena:
    """Device-resident slots for one MoE layer's hottest experts.

    Holds one buffer per per-expert source tensor, each with `n_slots` rows instead of
    `n_experts`, so a buffer is **layout-identical to a slice of its source**. That identity
    is the whole design: `glq_cuda.cu` reaches a per-expert weight as
    ``base + eidx * estride``, so the fused op reads the arena correctly when handed the
    arena as the base and a slot id as ``eidx`` — with the already-compiled kernel, no edit to
    `glq/csrc/*`, no JIT rebuild, no sm_86/89/120 portability question.
    """

    def __init__(self, decision: ArenaDecision, sources: dict, n_experts: int):
        import torch

        if not decision.enabled:
            raise ValueError(f"refusing to build a disabled arena: {decision.reason}")
        self.decision = decision
        self.n_slots = decision.slots
        self.state = ArenaState(n_experts=n_experts, n_slots=self.n_slots,
                                device=next(iter(sources.values())).device)
        self.sources = sources
        self.buffers = {}
        #: Elements per expert per tensor, which is also the row stride the kernel walks.
        self.row_elems = {}
        for name, src in sources.items():
            if src.shape[0] != n_experts:
                raise ValueError(
                    f"{name} has leading dim {src.shape[0]}, expected {n_experts}. Only "
                    f"per-expert tensors belong in the arena — a per-layer tensor (SV) read "
                    f"at a slot index would silently be the wrong expert's data.")
            self.buffers[name] = torch.empty((self.n_slots,) + tuple(src.shape[1:]),
                                             dtype=src.dtype, device=src.device)
            self.row_elems[name] = src[0].numel()

    # ---- the per-step path ------------------------------------------------

    def fill(self, plan: StepPlan) -> None:
        """Copy every missing routing's blob into its slot. One launch per tensor.

        Launch count is `len(sources)` per layer per step; the two packed tensors carry
        >99% of the bytes and the rest are a few KiB, so batching the small ones is an
        obvious later win — measure before bothering.
        """
        kern = _fill_kernel()
        n_routings = plan.fill_src.numel()
        for name, src in self.sources.items():
            elems = self.row_elems[name]
            grid = (n_routings, (elems + FILL_BLOCK - 1) // FILL_BLOCK)
            kern[grid](src, self.buffers[name], plan.fill_src, plan.slot_of, elems,
                       BLOCK=FILL_BLOCK)

    def warmup(self) -> None:
        """Compile the fill kernel and touch every buffer **before** graph capture.

        Triton JITs on first launch. Letting that happen inside `cudaStreamCapture` is at
        best a recorded compile and at worst the capture failure the `trellis_capture_sm120`
        note could not isolate — so pay it during weight processing instead, where it is
        just load time.
        """
        import torch

        zeros = torch.zeros(1, dtype=torch.int32, device=self.state.residency.device)
        self.fill(StepPlan(zeros, torch.full_like(zeros, _NOT_RESIDENT)))
        torch.cuda.synchronize(self.state.residency.device)

    def report(self) -> str:
        st = self.state
        return (f"arena {self.n_slots} slots: hit_rate={st.hit_rate():.4f} "
                f"hits={st.hits} misses={st.misses} fills={st.fills}")


class ArenaWeights:
    """Reads an arena buffer where one exists, else the layer's own tensor.

    Exists so the `fused_moe_trellis_3inst` call site is written once. The alternative — two
    near-identical 20-argument calls — is exactly how a divergence hides: one of them gets a
    fix and the other does not.
    """

    __slots__ = ("_buffers", "_layer")

    def __init__(self, buffers, layer):
        self._buffers = buffers
        self._layer = layer

    def __getattr__(self, name):
        buf = self._buffers.get(name)
        return buf if buf is not None else getattr(self._layer, name)
