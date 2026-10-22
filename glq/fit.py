"""Can this checkpoint run on this machine? Asked before the weight load, not during it.

`glq-code --model <id>` and `glq-chat --model <id>` accept any checkpoint and hand it to
vLLM. When it does not fit, the user learns so from a CUDA out-of-memory several minutes into
a multi-GiB download, with nothing in the message pointing at the cause. This module answers
the question first, cheaply, and says which resource binds.

**Admission, not planning.** `glq.supervisor` decides *what settings to use*; this decides
*whether to proceed at all*. It does not re-derive the plan — `check_fit` takes an
already-constructed `VllmSupervisor`, which is pure planning (no process, no files), and reads
the plan off it. A verdict computed from the same object that will be served cannot disagree
with what gets served, which a parallel re-derivation eventually would.

**Why not `installer.recommend`.** `supervisor` imports `glq.kv_compression`, which executes
`glq/__init__.py` and therefore imports torch, while `glq/installer/__init__.py` commits the
installer to stdlib only; and `supervisor.plan_expert_offload_gib` already imports *from*
`recommend`. The edge runs supervisor → recommend, so hosting this in `recommend` would invert
it and drag torch into the installer menu. The installer's own 0.75-of-VRAM heuristic also
answers a different question — whether to *download* — and is allowed to disagree with this.

## Two axes, and they are not interchangeable

**The card.** Gated on the FLOOR window plus `_MIN_KV_BYTES`, deliberately the weakest
defensible threshold: that the card can hold the weights, the runtime overhead and a 2 GiB
cache. Not the chosen window — `_KV_BYTES_PER_TOKEN` is an envelope over the worst pool ever
observed, `_say_window_shortfall` documents at length why an estimate-based clamp shortens
windows that would have served, and disk bytes over-state resident bytes (`recommend`'s
docstring records the 31B at 22.4 GiB on disk against ~16.5 GiB resident). A tight gate here
refuses working configurations, which is worse than not checking.

**Host RAM.** The axis nothing checked before, and the one that produced the incident this
module was written for. See `HOST_UNPINNED_FLOOR_BYTES`.

The remedies are **opposite**, so the verdict must name the binding resource: deeper offload
relieves a VRAM shortfall and makes a host one worse, because pinned pages cannot be swapped.

## Never refuse on an unknown

`fits` is `None` — and the caller proceeds exactly as before — when the checkpoint size, the
VRAM or the host RAM could not be determined, or on the CPU backend, which budgets weights and
KV from one pool with different arithmetic and already warns in `supervisor.start()`. Offline
must not become a refusal.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass

from glq.supervisor import _MIN_KV_BYTES, kv_headroom_bytes, window_kv_bytes

GIB = 1024 ** 3

#: Terminals are 80 columns until proven otherwise, and this text is read in one. Prose is
#: wrapped to it rather than hand-broken, because hand-broken lines go ragged the moment a
#: number in the middle of them changes width.
_WIDTH = 78

#: Host bytes that must stay UNPINNED for a serve to survive.
#:
#: Derived from the comment that justifies `supervisor._ARENA_PINNED_RAM_FRACTION`, which
#: reasons in absolute remainders even though it is expressed as a share: "that run pinned
#: 65.3 GiB on a box with no swap and still had ~13 GiB available at peak … 0.6 of 121 GiB
#: leaves ~48 GiB unpinned for the OS, the page cache and the serving processes."
#:
#: Both data points are swapless, and one of them PASSED:
#:
#:     pinned 65.3 of 121 GiB  ->  55.7 unpinned  ->  served
#:     pinned 18   of  30 GiB  ->  12.0 unpinned  ->  CUDA OOM during the weight load
#:
#: A *share* calls those two the same configuration and they are not, which is precisely how
#: 0.6 passed the small box. 16 GiB separates them with margin on both sides.
#:
#: One pass and one failure is an envelope, not a model. Widen it when a third beats it, and
#: note that host accounting under offload has separately been measured at 1.57x the planned
#: figure (102.8 GiB held against 65.3 planned — see `chat.resolve_mtp_tokens`), which is why
#: the floor is generous rather than tight.
HOST_UNPINNED_FLOOR_BYTES = 16 * GIB


@dataclass(frozen=True)
class FitReport:
    """A verdict, plus every number the message needs, so rendering reads no globals."""

    model: str
    #: True / False when there was enough information to judge; **None when there was not**.
    fits: bool | None
    #: "" when fitting or unknown, else which resource binds: "vram", "host", or "both".
    reason: str

    # the plan this verdict judged, read off the supervisor
    window: int
    offload_gib: int
    arena_gib: int
    weights_bytes: int | None
    resident_bytes: int

    # the card
    vram_bytes: int | None
    kv_floor_bytes: int
    headroom_bytes: int
    vram_short_bytes: int

    # the host
    ram_bytes: int | None
    swap_bytes: int | None
    available_ram_bytes: int | None
    pinned_bytes: int
    unpinned_bytes: int
    host_short_bytes: int

    @property
    def gib(self) -> dict:
        """Every byte field in GiB, for formatting."""
        return {k: (v / GIB if isinstance(v, int) else v)
                for k, v in self.__dict__.items() if k.endswith("_bytes")}


def check_fit(sup, *, ram_bytes=None, swap_bytes=None, available_ram_bytes=None,
              floor=None) -> FitReport:
    """Judge an already-planned `VllmSupervisor` against the machine it will run on.

    `floor` is the caller's own window floor — `glq-code` carries 16384 because a coding agent
    needs a real context, `glq-chat` 8192 — and defaults to the supervisor's chosen window
    only when the caller has nothing better, which is the permissive reading.
    """
    vram = sup.vram_bytes
    weights = getattr(sup, "weights_bytes", None)
    window = int(getattr(sup, "max_model_len", 0) or 0)
    conc = max(1, int(getattr(sup, "window_concurrency", 1) or 1))
    offload_gib = int(getattr(sup, "expert_offload_gib", 0) or 0)
    arena_gib = int(getattr(sup, "expert_arena_gib", 0) or 0)
    resident = int(getattr(sup, "resident_bytes", 0) or 0)
    pinned = int(getattr(sup, "pinned_host_bytes", 0) or 0)

    # The floor window is what the card is judged against; `_MIN_KV_BYTES` is the floor under
    # that, since a smaller pool serves no useful context and is therefore part of what fits.
    kv_floor = max(_MIN_KV_BYTES, window_kv_bytes(int(floor or window or 0), conc))
    headroom = kv_headroom_bytes(resident, vram) if vram else 0
    unpinned = max(0, int(ram_bytes) - pinned) if ram_bytes else 0

    blank = dict(
        model=sup.model, window=window, offload_gib=offload_gib, arena_gib=arena_gib,
        weights_bytes=weights, resident_bytes=resident, vram_bytes=vram,
        kv_floor_bytes=kv_floor, headroom_bytes=headroom, vram_short_bytes=0,
        ram_bytes=ram_bytes, swap_bytes=swap_bytes,
        available_ram_bytes=available_ram_bytes, pinned_bytes=pinned,
        unpinned_bytes=unpinned, host_short_bytes=0)

    # ---- the unknowns, which must never refuse
    if getattr(sup, "device", None) == "cpu" or not vram or not weights or not ram_bytes:
        return FitReport(fits=None, reason="", **blank)

    vram_short = max(0, kv_floor - headroom)

    # `pinned == 0` short-circuits: a checkpoint that offloads nothing cannot be refused for
    # pinning too much, and without this every dense model on a small-RAM box would be.
    # Swap present downgrades to a pass — it absorbs the overshoot around the pinned pages,
    # which stay capped by the planner's own fraction regardless.
    host_short = 0
    if pinned and not swap_bytes:
        host_short = max(0, HOST_UNPINNED_FLOOR_BYTES - unpinned)

    reason = {(True, True): "both", (True, False): "vram",
              (False, True): "host"}.get((bool(vram_short), bool(host_short)), "")
    return FitReport(fits=not reason, reason=reason,
                     **{**blank, "vram_short_bytes": vram_short,
                        "host_short_bytes": host_short})


def fit_ranked(checkpoints, *, vram_bytes, ram_bytes, swap_bytes=None,
               prefer_family=None, concurrency=1, floor=None, window_extra_gib=0,
               device=None):
    """`recommend.Ranked` for each checkpoint, judged with the serve-time math.

    Mirrors `recommend.rank`'s ordering and recommendation policy exactly — largest first,
    trellis preferred over size, a family preference honoured only when something in that
    family fits — so the picker and `glq-setup --list` cannot disagree about rank order. What
    differs is `fits`: `rank` asks the conservative *download* question (0.75 of VRAM against
    the resident floor, no KV, no arena, no host RAM) and this asks whether the planned serve
    actually fits. They are allowed to disagree, and on the incident they do.

    One supervisor per checkpoint, which is affordable because construction is pure planning.
    `device` is resolved ONCE by the caller or here, never per checkpoint: `detect_device`
    reads wheel metadata and can shell out to nvidia-smi, and paying that N times is a slow
    start nobody would attribute to this function.
    """
    from glq.installer.recommend import Ranked
    from glq.supervisor import VllmSupervisor, detect_device

    ordered = sorted(checkpoints, key=lambda c: c.size_bytes, reverse=True)
    if vram_bytes is None:
        # Same contract as `rank`: list everything, recommend nothing, rather than bluff a
        # recommendation that cannot be justified.
        return [Ranked(c, None, False) for c in ordered]

    device = device or detect_device()
    verdicts = {}
    for c in ordered:
        if not c.size_bytes:
            # 0 means the tree API gave nothing usable. Never "fits anywhere", or a broken
            # repo sorts to the top of the recommendation.
            verdicts[c.repo_id] = (False, "")
            continue
        sup = VllmSupervisor(
            model=c.repo_id, device=device, vram_bytes=vram_bytes, ram_bytes=ram_bytes,
            weights_bytes=c.size_bytes, ple_offload_bytes=c.ple_offload_bytes,
            expert_offload_bytes=c.expert_offload_bytes, nontext_bytes=c.nontext_bytes,
            # No model_max_len: knowing it costs another HTTP call per repo, and the window
            # tiers plus the floor are enough to price a menu entry.
            max_model_len_floor=floor or 8192, window_concurrency=concurrency,
            max_num_seqs=concurrency, window_offload_extra_gib=window_extra_gib)
        r = check_fit(sup, ram_bytes=ram_bytes, swap_bytes=swap_bytes, floor=floor)
        verdicts[c.repo_id] = (r.fits, r.reason)

    fitting = [c for c in ordered if verdicts[c.repo_id][0] is True]
    if prefer_family:
        fam = [c for c in fitting if prefer_family.lower() in c.repo_id.lower()]
        if fam:
            fitting = fam
    # Trellis first, then size — `rank`'s reasoning, not a new one: trellis decodes
    # single-stream at bf16 parity while shell/e8p are materially slower, so a smaller
    # trellis checkpoint beats a larger slow one. `is True`, so an unknown never outranks a
    # confirmed one.
    best = ([c for c in fitting if c.trellis is True] or fitting or [None])[0]
    best_id = best.repo_id if best is not None else None

    return [Ranked(c, verdicts[c.repo_id][0], c.repo_id == best_id,
                   verdicts[c.repo_id][1]) for c in ordered]


def refusal_lines(report: FitReport, *, alternatives=(), command="glq-code",
                  gpu_name=None) -> list[str]:
    """The refusal, as lines. Shared by both front-ends so they cannot word it differently.

    Raises on a passing report rather than returning nothing: rendering a refusal for
    something that fits is a caller bug, and an empty string would hide it.
    """
    if report.fits is not False:
        raise ValueError(f"refusal_lines called on a {report.fits!r} report for "
                         f"{report.model} — only a refusal has anything to render")

    def para(text):
        return textwrap.wrap(" ".join(text.split()), _WIDTH,
                             initial_indent="  ", subsequent_indent="  ")

    g = report.gib
    short = report.model.rsplit("/", 1)[-1]
    out = textwrap.wrap(
        f"error: {short} does not fit this machine. Refusing now, because vLLM would "
        f"otherwise fail several minutes into the weight load.",
        _WIDTH, subsequent_indent="       ") + [""]

    plan = [f"weights {g['weights_bytes']:.2f} GiB"]
    if report.offload_gib:
        plan.append(f"{report.offload_gib} GiB pinned in host memory")
        plan.append(f"{report.resident_bytes / GIB:.2f} GiB resident")
    if report.arena_gib:
        plan.append(f"{report.arena_gib} GiB expert cache")
    # Comma-separated, not bullet-separated: a wrapped bullet list leaves a stray "·"
    # starting a line, which reads as a new item rather than a continuation.
    out += para("plan: " + ", ".join(plan)) + [""]

    card = f"{gpu_name or 'this card'}"
    out.append(f"  {card:<26} {g['vram_bytes']:>6.2f} GiB"
               + (f"   SHORT by {g['vram_short_bytes']:.2f} GiB" if report.vram_short_bytes
                  else "   fits"))
    out.append(f"  {'host RAM':<26} {g['ram_bytes']:>6.2f} GiB"
               + (f"   SHORT by {g['host_short_bytes']:.2f} GiB" if report.host_short_bytes
                  else "   fits"))
    out.append("")

    if report.reason in ("host", "both"):
        # Deliberately does NOT say "raise --cpu-offload-gb". The pinnable cap is already
        # holding it shut, and pinned pages cannot be swapped out, so deeper offload makes
        # this strictly worse. `_say_window_shortfall` reasons the same way.
        out += para(
            "Host RAM is the binding constraint"
            + (", as well as the card. " if report.reason == "both" else ", not the card. ")
            + f"Serving this needs {report.offload_gib} GiB of MoE experts pinned in host "
            f"memory, leaving {g['unpinned_bytes']:.2f} GiB for the OS, the page cache and "
            f"vLLM itself. {HOST_UNPINNED_FLOOR_BYTES / GIB:.0f} GiB is the floor measured "
            "on hosts where this works, and this machine has no swap to absorb the "
            "difference. Pinned pages cannot be swapped out, so offloading more cannot "
            "help; host accounting under offload has been measured at 1.57x the plan."
        ) + [""] + para("Fixes: add swap, add RAM, or pick a smaller checkpoint.") + [""]
    elif report.reason == "vram":
        out += para(
            "The card is the binding constraint. "
            + ("This checkpoint declares no offloadable experts, so there is nothing to "
               "move into host memory." if not report.offload_gib else
               f"The deepest offload this host can pin still leaves it "
               f"{g['vram_short_bytes']:.2f} GiB short.")
        ) + [""]

    if alternatives:
        out.append("  these fit:")
        out += [f"    {a}" for a in alternatives]
        out.append("")

    out += para(f"or: --force to serve it anyway ({command} will probably fail), or "
                f"choose another with  {command} --model <repo-id>")
    return out
