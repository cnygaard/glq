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
#: Narrowest step the arena must serve to be worth allocating. 1 without speculation — every
#: decode step is one token wide, so any cap serves them and a small arena merely degrades.
#: MTP makes every step `1 + num_spec` wide, and a cap below that serves NONE of them, so the
#: arena would hold VRAM that KV needs and never be used. Set by `glq.supervisor.child_env`.
MIN_TOKENS_ENV = "GLQ_MOE_ARENA_MIN_TOKENS"

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
    #: Token count the arena path serves, DERIVED from what the budget affords rather than
    #: taken from the caller — see `arena_decision`. `apply` gates on this.
    max_tokens: int = 0
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
                   max_tokens: int = DEFAULT_MAX_TOKENS,
                   min_tokens: int = 1) -> ArenaDecision:
    """Decide whether this layer gets an arena, and how many slots.

    Refuses — with a reason — when:

    * the budget is unset (the default);
    * little or nothing is offloaded, so there is no PCIe traffic to remove and the arena
      would be a second copy of weights already on the card;
    * the budget cannot hold even ``top_k`` slots — one token's routings — which is a
      **correctness** precondition rather than a tuning one (see `slots_for_budget`).

    ``max_tokens`` is a **ceiling, not a demand.** The cap the arena actually serves is
    ``min(max_tokens, slots // top_k)``, because lowering it is always safe: fewer token
    counts take the arena path and the rest take today's exact path. Refusing instead cost
    three measured arms — 6 GiB over 49 containers is 71 slots/layer, a cap of 8 demands
    8 x top_k 10 = 80, so the arena declined and three runs silently measured no arena while
    looking entirely healthy.
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
    # Size first with no cap demanded, then derive the cap from what the budget bought. Asking
    # `slots_for_budget` to enforce `max_tokens` here would make a too-small budget a refusal
    # instead of a narrower arena.
    slots = slots_for_budget(int(gib * 1024**3), n_layers=n_layers, hidden_size=hidden_size,
                             inter=inter, w13_out=w13_out, bpw=bpw, n_experts=n_experts)
    cap = min(int(max_tokens), slots // max(int(top_k), 1))
    floor = max(1, int(min_tokens))
    if cap < floor:
        # `floor` is 1 without speculation, so this is the original "cannot hold even one
        # token's routings" refusal. With MTP on the caller raises it to `1 + num_spec`,
        # because every decode step is then at least that wide: a cap below it would serve NO
        # decode step, making the arena VRAM that KV needs and never gets. An 8 GiB arena
        # plus MTP once left `Available KV cache memory: -0.31 GiB` exactly this way.
        need = "ONE token's" if floor == 1 else f"a {floor}-token verify step's"
        extra = "" if floor == 1 else (
            f" Every MTP step carries 1 + num_spec = {floor} tokens, so a narrower cap would "
            f"serve none of them.")
        return ArenaDecision(
            False, reason=(
                f"expert arena disabled: {gib:g} GiB buys {slots} slots/layer across "
                f"{n_layers} layers, which cannot hold {need} top_k {top_k} routings "
                f"({floor} x {top_k} = {floor * int(top_k)} > {slots}). Every expert routed "
                f"in a step must be resident at once, so a narrower arena is not an option "
                f"here — raise {BUDGET_ENV}.{extra}"))

    note = "" if cap >= int(max_tokens) else \
        f" (cap lowered from {max_tokens}: {slots} slots / top_k {top_k})"
    return ArenaDecision(True, slots=slots, bytes_per_slot=per, n_layers=n_layers,
                         max_tokens=cap,
                         reason=(f"expert arena: {slots} slots/layer x {n_layers} layers = "
                                 f"{slots * per * n_layers / 1024**3:.2f} GiB, "
                                 f"{per / 1024**2:.3f} MiB/slot, serving steps up to "
                                 f"{cap} token(s){note}"))


def arena_decision_from_env(env, **kw) -> ArenaDecision:
    """`arena_decision` with the env vars read from a mapping (``os.environ``).

    `MIN_TOKENS_ENV` defaults to 1, i.e. today's behaviour; `glq.supervisor.child_env` raises
    it to `1 + num_spec` when it enables MTP, because that is the narrowest step speculation
    produces and a cap below it would serve none of them.
    """
    return arena_decision(
        budget=env.get(BUDGET_ENV, ""),
        max_tokens=int(env.get(MAX_TOKENS_ENV, DEFAULT_MAX_TOKENS)),
        min_tokens=int(env.get(MIN_TOKENS_ENV, 1)),
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
        # ★ One SCRATCH element past the end of each table. Under CUDA-graph capture a write
        # cannot be skipped — there is no data-dependent control flow — so a routing that
        # must not update residency writes to the scratch element instead. That is what
        # replaces the `if` an eager implementation would use, and it is why these are
        # n+1 long. Nothing ever reads the scratch row.
        self.residency = torch.full((self.n_experts + 1,), _NOT_RESIDENT, dtype=i32,
                                    device=device)
        self.slot_owner = torch.full((self.n_slots + 1,), _NOT_RESIDENT, dtype=i32,
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

    **★ Every operation here is fixed-shape and in-place, because this runs inside a CUDA
    graph.** Measured on an L4, each of `torch.unique`, `bool(t.any())`, `int(t.sum())` and
    `torch.nonzero` invalidates capture with `cudaErrorStreamCaptureInvalidated`, while
    `argsort` and `index_copy_` are fine. An earlier version of this function used the first
    four, worked perfectly eagerly, and would have silently forced `--enforce-eager` — trading
    the trellis MoE graph (**91.6 vs 8.5 tok/s**) for the arena's 1.6x. So:

    * dedup comes from an `(R, R)` comparison instead of `torch.unique` (R <= 80, so 6,400
      element ops — free at this scale);
    * a write that must not happen is redirected to a **scratch element** past the end of the
      table, since capture has no data-dependent control flow to skip it with;
    * state is mutated through `scatter_`/`add_`/advanced-index assignment only. Rebinding
      `state.x = <new tensor>` would record the graph against an address the next replay never
      writes.

    **Provably the same selection as `ResidencyPolicy.plan_step`**, which the tests assert
    rather than assume: a step's hit slots are protected from candidacy, and misses are
    admitted in first-appearance order taking the least-recently-used unprotected slots.
    """
    import torch

    # ★ Clamp into range before any gather. vLLM captures CUDA graphs with DUMMY inputs, so
    # the router can emit ids outside [0, n_experts). `residency[ids]` is a gather, and both
    # directions are hazards: an id past the end is an out-of-bounds read (device-side assert
    # under Triton, which is what took down the first graphed arm), and a NEGATIVE id silently
    # reaches the scratch element through Python negative indexing, is admitted as "expert
    # -1", and emits fill_src = -1 — which the kernel reads as a hit and skips, leaving the op
    # to read a slot nothing ever wrote.
    #
    # Not in place: `expert_ids` is vLLM's own topk_ids, which the fused op reads for its own
    # routing, and `.to(int64).reshape(-1)` can be a view when it is already int64.
    ids = expert_ids.to(torch.int64).reshape(-1).clamp(0, max(state.n_experts - 1, 0))
    n_routings = ids.numel()
    S, E = state.n_slots, state.n_experts
    if n_routings > S:
        # A shape, so this is a host-side check and safe under capture.
        raise ValueError(
            f"{n_routings} routings in one step cannot be served by {S} slots: every expert "
            f"routed in a step must be resident SIMULTANEOUSLY, so a step larger than the "
            f"arena would evict a routing before the matmul reads it. Raise {BUDGET_ENV} or "
            f"lower {MAX_TOKENS_ENV}.")

    dev = ids.device
    pos = torch.arange(n_routings, device=dev, dtype=torch.int64)
    slot0 = state.residency[ids].to(torch.int64)
    is_hit = slot0 >= 0

    # Dedup, fixed shape. `same[j, r]` is "routing j wants the same expert as r".
    same = ids[:, None] == ids[None, :]
    is_dup = (same & (pos[:, None] < pos[None, :])).any(dim=0)
    # First routing position wanting each expert, so duplicates can share its slot.
    first_idx = torch.where(same, pos[:, None],
                            torch.full_like(same, n_routings, dtype=torch.int64)
                            ).min(dim=0).values

    # One admission per distinct missing expert, in first-appearance order. The rank is a
    # running count over the same (R, R) comparison used for dedup above — `pos[None, :] <=
    # pos[:, None]` masks to positions at or before r — rather than a separate running-total
    # call. Same result, one fewer primitive, and it reuses a matrix already materialised.
    need = (~is_hit) & (~is_dup)
    at_or_before = pos[None, :] <= pos[:, None]
    rank = (need[None, :] & at_or_before).sum(dim=1) - 1

    # ★ Protect this step's hit slots from candidacy. Not by recency — a slot last used long
    # ago but read by THIS step is exactly what an LRU scan would pick, and evicting it would
    # leave the matmul reading a different expert.
    # `prot[idx] = True` would copy a Python scalar from CPU, which capture refuses
    # ("Cannot copy between CPU and CUDA tensors during CUDA graph capture"). Scatter a
    # device-side source instead. Non-hits address the scratch element and write False there,
    # and cannot collide with a hit because a hit's slot is always < S.
    prot = torch.zeros(S + 1, dtype=torch.bool, device=dev)
    prot.scatter_(0, torch.where(is_hit, slot0, torch.full_like(slot0, S)), is_hit)

    # Candidates ascending by (last use, slot index). The index tie-break matches
    # ResidencyPolicy's lo..hi scan, which hands out never-claimed slots in index order.
    sidx = torch.arange(S, device=dev, dtype=torch.int64)
    key = state.slot_last_use * S + sidx
    key = torch.where(prot[:S], key + (1 << 62), key)
    order = torch.argsort(key)

    chosen = order[rank.clamp(0, S - 1)]
    slot = torch.where(need, chosen, slot0)
    slot = slot.gather(0, first_idx.clamp(max=max(n_routings - 1, 0)))

    # Evict the previous owner of each chosen slot, then publish — in that order, so a slot
    # reused this step is not cleared after being claimed. Non-admissions address the scratch
    # element, which nothing reads.
    s_at = torch.where(need, slot, torch.full_like(slot, S))
    prev = state.slot_owner[s_at].to(torch.int64)
    clr = torch.where(need & (prev >= 0), prev, torch.full_like(prev, E))
    # Same reason as `prot` above: the right-hand side must live on the device, so build the
    # sentinel as a tensor rather than assigning the Python int.
    state.residency[clr] = torch.full_like(clr, _NOT_RESIDENT, dtype=torch.int32)
    e_at = torch.where(need, ids, torch.full_like(ids, E))
    state.residency[e_at] = s_at.to(torch.int32)
    state.slot_owner[s_at] = e_at.to(torch.int32)

    # Stamp in routing order so intra-step recency is defined as the reference defines it,
    # and a repeated expert's stamp is its LAST use. `amax` keeps duplicates deterministic.
    state.clock.add_(1)
    state.slot_last_use.scatter_reduce_(0, slot, state.clock + pos, reduce="amax",
                                        include_self=True)
    state.clock.add_(n_routings)

    state.counters[0] += is_hit.sum()
    state.counters[1] += (~is_hit).sum()
    state.counters[2] += need.sum()

    fill_src = torch.where(need, ids.to(torch.int32),
                           torch.full_like(ids, _NOT_RESIDENT, dtype=torch.int32))
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

    def rebind_sources(self, sources: dict) -> None:
        """Point the fill at the tensors the op will actually read, without reallocating.

        **Allocation and source binding have opposite timing constraints, so they are
        separate calls.**

        The buffers must be allocated *before* vLLM's memory profiling and graph capture.
        Allocating them at first apply put a 9.97 GiB arena inside `cudaStreamCapture`:
        vLLM reported *"Graph capturing finished in 3 secs, took 10.06 GiB"* against the usual
        0.3, subtracted that as graph memory, and refused to start with *"Available KV cache
        memory: -0.82 GiB"*. A persistent arena also has no business living in the graph's
        private pool.

        The sources can only be read *after* vLLM's UVA offload, which runs later than
        `process_weights_after_loading` — read too early they are the still-resident tensors,
        and the op would be fed an arena filled from storage it no longer uses.

        Shapes and dtypes are identical either way, so allocating from the early tensors and
        filling from the late ones is sound; this asserts that rather than trusting it.
        """
        for name, src in sources.items():
            buf = self.buffers.get(name)
            if buf is None:
                raise KeyError(
                    f"{name} has no arena buffer; the source set changed between allocation "
                    f"and binding, so the fill would read a tensor the op does not.")
            if tuple(src.shape[1:]) != tuple(buf.shape[1:]) or src.dtype != buf.dtype:
                raise ValueError(
                    f"{name} changed shape/dtype between allocation and binding: "
                    f"{tuple(src.shape)}/{src.dtype} vs arena {tuple(buf.shape)}/{buf.dtype}. "
                    f"The arena would not be layout-identical to its source.")
        self.sources = sources

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
        """Compile the fill kernel **before** graph capture.

        Triton JITs on first launch. Letting that happen inside `cudaStreamCapture` is at
        best a recorded compile and at worst the capture failure the `trellis_capture_sm120`
        note could not isolate — so pay it during weight processing instead, where it is
        just load time.
        """
        import torch

        zeros = torch.zeros(1, dtype=torch.int32, device=self.state.residency.device)
        self.fill(StepPlan(zeros, torch.full_like(zeros, _NOT_RESIDENT)))
        torch.cuda.synchronize(self.state.residency.device)

    def maybe_report(self, every: int = 0) -> None:
        """Print the hit accounting every `every` eager calls. Never during capture.

        Reading a counter needs a host sync, which capture forbids — and under FULL cudagraph
        `apply`'s Python runs only at capture time, so a graphed decode produces no reports at
        all. That is why the mechanism evidence for a graphed arm comes from a matched EAGER
        arm (identical routing, so an identical hit rate) rather than from the graphed run.
        """
        import torch

        if not every or torch.cuda.is_current_stream_capturing():
            return
        self._calls = getattr(self, "_calls", 0) + 1
        if self._calls % int(every) == 0:
            print(f"glq: {self.report()}", flush=True)

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
