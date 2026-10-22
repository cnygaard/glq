"""Own a vLLM server's lifetime, so stopping the chat frees the GPU.

`glq-chat` used to be a pure client: the user started `vllm serve` in one terminal and the
UI in another, and if they ran the UI first they got a chat window with an empty model
dropdown. Worse, the documented serve command passed no `--gpu-memory-utilization`, so vLLM
took its default 0.9 — on a 24 GB card serving a 1.8 GB model that reserves ~21.6 GB of KV
pool and holds it until the process dies. On a machine that also plays games, that is not a
background service, it is the whole GPU.

vLLM has neither lazy load nor idle unload (Ollama has both), so "just leave it running" is
not a kindness here. One process therefore owns the whole lifetime: start the server, use
it, and on exit — including the exception path — take it down.

Lives apart from `glq/chat.py` so it can be tested without gradio and without a GPU: the
process factory, the health probe and the clock are all injected.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

from glq import kv_compression, tooling

#: vLLM's own default is 0.9 of *total* VRAM, which is the right call for a dedicated
#: inference box and the wrong one for a desktop. A GLQ checkpoint is small by construction —
#: that is the point of the project — so a modest pool still leaves a usable context while
#: keeping most of the card free for whatever else the user is doing.
DEFAULT_GPU_MEMORY_UTILIZATION = 0.45

#: Weight load plus CUDA-graph capture. Minutes, not seconds, and not something a wheel
#: fixes — the prebuilt kernels remove the *compile*, not the load.
DEFAULT_READY_TIMEOUT = 900.0

#: How often to say something while waiting. Measured on an L4: `glq-chat` printed three
#: lines and then nothing for nine minutes, which is indistinguishable from a hang.
DEFAULT_REPORT_EVERY = 5.0


#: Floor for what vLLM needs on top of the weights before a single KV block exists:
#: activations, workspace, CUDA-graph capture, non-torch allocations. Budgeting 2 GiB was
#: measured wrong on an L4 — a 13.9 GiB checkpoint in an 18.0 GiB budget left 0.42 GiB of KV,
#: so the real overhead was ~3.7 GiB. 4 GiB, from that measurement rather than from taste.
#:
#: Kept as the FLOOR rather than the whole answer; see `runtime_overhead_bytes`.
_RUNTIME_OVERHEAD_BYTES = 4 * 1024 ** 3

#: Share of the resident weights added to the floor above. Non-torch allocations and
#: CUDA-graph memory scale with layer count and model size, so one constant cannot serve both
#: models that have been measured:
#:
#:     gemma-4 26B, 13.9 GiB resident, L4          ~3.70 GiB   (the measurement above)
#:     Qwen3.8-Flash-Next, 48.65 GiB, RTX PRO 6000  5.44 GiB
#:
#: The second came from a user-reported startup failure, derived from that run's own log:
#: `0.628 x 95.6 - 48.65 (Model loading took) - 5.95 (Available KV cache memory) = 5.44`.
#: A flat 4 GiB under-reserved by 1.44 GiB there; a flat 6 GiB covers it and makes a 23 GiB
#: L4 refuse to start, because 14.4 + 6 leaves 0.76 GiB under `_MAX_UTILIZATION` where 1.72
#: is needed. Hence proportional.
#:
#: 3% is calibrated so the one measured shortfall is covered (4 + 0.03x48.65 = 5.46 against
#: 5.44) while staying conservative on the smaller model (4.42 against 3.70). **Two data
#: points is an envelope, not a model** — widen it when a third beats it, and do not fit it
#: more tightly on the strength of these two.
_RUNTIME_OVERHEAD_WEIGHT_FRACTION = 0.03


def runtime_overhead_bytes(weights_bytes) -> int:
    """Non-weight, non-KV VRAM to reserve for a model of this resident size.

    Everything the pool must hold besides weights and KV blocks. See
    `_RUNTIME_OVERHEAD_WEIGHT_FRACTION` for why this is not a constant and what the two
    measurements behind it are.
    """
    if not weights_bytes:
        return _RUNTIME_OVERHEAD_BYTES
    return _RUNTIME_OVERHEAD_BYTES + int(
        int(weights_bytes) * _RUNTIME_OVERHEAD_WEIGHT_FRACTION)

#: A KV pool below this serves no useful context, so it is part of what must fit rather than
#: something to leave to chance.
_MIN_KV_BYTES = 2 * 1024 ** 3

#: vLLM defaults `max_model_len` to whatever the model declares, and gemma-4 declares
#: 262144 — which needs 6.15 GiB of KV for one request, measured. A chat does not need a
#: quarter-million-token window, and asking for one means no card of this size can serve a
#: model of this size. 8192 is a conversation; raise it with --max-model-len.
DEFAULT_MAX_MODEL_LEN = 8192

#: vLLM defaults max_num_seqs to 1024 — sized for a batch server, absurd for a single-user
#: chat — and on hybrid-GDN models every decode sequence reserves its own Mamba cache
#: block before a single request exists. Measured with a 27B GDN hybrid on a 96 GB card:
#: "max_num_seqs (1024) exceeds available Mamba cache blocks (399)", startup refused, with
#: a pool that held ~24 GiB of cache. 16 covers regenerations and a couple of parallel
#: tabs, and shrinks CUDA-graph capture and KV pressure for every model, not just hybrids.
DEFAULT_MAX_NUM_SEQS = 16

#: The CPU-backend default. At single-digit tok/s TOTAL, 16 concurrent decodes is thrash;
#: 4 covers a regeneration plus one parallel tab. Explicit --max-num-seqs always wins.
DEFAULT_CPU_MAX_NUM_SEQS = 4

#: Leave a slice of the card for the desktop. Above this, vLLM competes with the compositor
#: and a display server can fail to allocate.
_MAX_UTILIZATION = 0.92


def kv_headroom_bytes(resident_bytes, vram_bytes) -> int:
    """KV bytes the card can still hold once this much resident weight, plus its overhead, is in.

    One affordability question asked from two ends, which is why it lives in one place:
    `plan_max_model_len` asks *which window tier fits this footprint*, and
    `plan_expert_offload_gib` asks *how much weight must leave for this window to fit*. The
    same inequality. Inverting `runtime_overhead_bytes` algebraically to answer the second
    would be a second copy of that formula to keep in step with the next measurement that
    widens it — and the comment on `_RUNTIME_OVERHEAD_WEIGHT_FRACTION` promises there will be
    one.

    Measured against `_MAX_UTILIZATION` — what the card *could* give — not against the minimal
    pool. That distinction is not cosmetic: sizing against the minimal pool leaves 2 GiB of
    headroom on a 96 GB card and on a 23 GiB one alike, which is what made the window planner
    inert for exactly the checkpoints it was written for.
    """
    if not vram_bytes:
        return 0
    resident = max(0, int(resident_bytes or 0))
    return int(_MAX_UTILIZATION * float(vram_bytes) - resident
               - runtime_overhead_bytes(resident))


def plan_gpu_memory_utilization(*, weights_bytes, vram_bytes, kv_bytes=None):
    """How much of the card vLLM may reserve, sized from the checkpoint.

    A fixed fraction cannot be right for every model: measured on a 23 GB L4, the 0.45 that
    leaves a 1.8 GB SmolLM3 politely sharing the card gives a 15 GB MoE a 10.4 GB budget, so
    the weights alone overflow it and vLLM dies with "No available memory for the cache
    blocks". Ask instead for weights + runtime overhead + a usable cache, which is small when
    the model is small — the point of not simply taking vLLM's 0.9 — and large when it must
    be.

    `kv_bytes` is what the *served window* needs (see `window_kv_bytes`), and asking for it
    here is what makes a long window real instead of merely announced: a pool holding only
    `_MIN_KV_BYTES` is 2 GiB whatever the card, so a 96 GB Blackwell serving a 49.53 GiB
    checkpoint reserved 2 GiB and could not afford any window above 8192. It is a floor, not
    a replacement — a 4096-token window needs 0.1 GiB of KV, and reserving only that would
    starve activations.

    Returns the documented default when either input is unknown: `--model` can point at a
    repo whose size we failed to look up, and guessing large would seize the whole GPU.
    """
    if not weights_bytes or not vram_bytes:
        return DEFAULT_GPU_MEMORY_UTILIZATION
    kv = max(_MIN_KV_BYTES, int(kv_bytes or 0))
    needed = weights_bytes + runtime_overhead_bytes(weights_bytes) + kv
    return min(max(needed / vram_bytes, DEFAULT_GPU_MEMORY_UTILIZATION), _MAX_UTILIZATION)


#: Extra GiB left free beyond the weight fraction when sizing expert offload. Small on
#: purpose: `WEIGHT_FRACTION` already holds back 25% of the card for KV, activations and
#: recurrent state, which is the bulk of the protection. The one measured failure here — a
#: 12 GiB expert budget on a 45 GiB card loading at 36.67 GiB and then refusing to serve with
#: 0.14 GiB of KV against 0.44 needed — happened at `--gpu-memory-utilization 0.95`, where
#: only 5.5 GiB was left for everything non-weight. Under 0.75 that same card caps weights at
#: 33.3 GiB and leaves ~11 GiB, which would have served. So this is a cushion, not the
#: mechanism; sizing it large would offload far more than necessary and cost decode speed.
_EXPERT_OFFLOAD_MARGIN_BYTES = 2 * 1024**3

#: Share of host RAM that pinned offloaded weights may occupy. Pinned pages cannot be
#: swapped, so overshooting here is not slow — it fails to allocate, or starves the OS. Mirrors
#: the reasoning behind `recommend.CPU_WEIGHT_FRACTION`.
_PINNED_RAM_FRACTION = 0.5


def plan_expert_offload_gib(*, weights_bytes, ple_offload_bytes, expert_offload_bytes,
                            vram_bytes, ram_bytes=None, nontext_bytes=0,
                            kv_bytes=0) -> int:
    """GiB of MoE expert weights to hold in host RAM so the remainder fits the card.

    Returns 0 — no offload flags at all — whenever the model already fits, so a box that does
    not need this never pays the PCIe cost. Also 0 when any input is unknown or the checkpoint
    declares no offloadable experts: guessing would be worse than serving as before.

    `kv_bytes` is what the served window needs (see `window_kv_bytes`), and passing it is what
    makes a long window real rather than merely announced. Fitting `WEIGHT_FRACTION` of the
    card is only the *first* requirement; on a 23 GiB L4 serving Flash-Next it is satisfied at
    14.86 GiB resident, which leaves 1.85 GiB of KV headroom against the 7.92 GiB a 262144
    window needs — so vLLM loaded the weights and then refused to start. With the window
    priced in, the two requirements are taken together and the tighter one wins.

    Measured anchors on Qwen3.8-Flash-Next (72.2 GiB of weights, 23.84 GiB of PLE):
    a 24 GiB budget left 24.31 GiB resident and served; 36 GiB left 12.15 GiB.
    """
    if not weights_bytes or not vram_bytes or not expert_offload_bytes:
        return 0
    from glq.installer.recommend import WEIGHT_FRACTION   # one source for the fraction

    # Non-text bytes come off first: they were never going to occupy the card, so counting
    # them would offload experts to make room for weights nothing loads — and every offloaded
    # expert is paid for again on every token that routes to it.
    after_ple = (int(weights_bytes) - int(ple_offload_bytes or 0)
                 - int(nontext_bytes or 0))
    need = after_ple - WEIGHT_FRACTION * float(vram_bytes)
    if need > 0:
        # The margin belongs to THIS branch only. It is a cushion on a proxy: `WEIGHT_FRACTION`
        # holds back 25% of the card as a stand-in for KV and activations, so a little more is
        # prudent. Where `kv_bytes` states the KV outright, `runtime_overhead_bytes` covers the
        # activations and `_KV_SAFETY_FACTOR` already pads the KV — adding this on top would
        # double-count a cushion and buy PCIe cost with nothing to show for it.
        need += _EXPERT_OFFLOAD_MARGIN_BYTES
    if kv_bytes:
        # Shed whole GiB until the window fits, asking `kv_headroom_bytes` rather than
        # inverting the overhead formula; see that function. ~50 iterations at worst.
        step = 1024**3
        shed, limit = 0, max(0, math.ceil(after_ple / step))
        while shed < limit and kv_headroom_bytes(after_ple - shed * step,
                                                 vram_bytes) < int(kv_bytes):
            shed += 1
        need = max(need, shed * step)
    if need <= 0:
        return 0
    gib = math.ceil(need / 1024**3)
    # Never claim more than the checkpoint actually has, or vLLM silently offloads less than
    # we planned for and the footprint prediction is wrong in the dangerous direction.
    gib = min(gib, int(int(expert_offload_bytes) // 1024**3))
    if ram_bytes:
        host_cap = int(float(ram_bytes) * _PINNED_RAM_FRACTION / 1024**3) \
            - math.ceil(int(ple_offload_bytes or 0) / 1024**3)
        gib = min(gib, max(0, host_cap))
    return max(0, gib)


#: Floor the arena must leave for KV beyond what the chosen window needs. The window figure is
#: already padded by `_KV_SAFETY_FACTOR`, but KV block granularity and a second request's
#: partial blocks are not in it, and coming up short is not graceful — vLLM refuses at startup.
#: Measured either side of this on an L4: an 8 GiB arena left 1.16 GiB of pool and served; 10
#: GiB left -0.82 GiB and the engine declined to start.
_ARENA_KV_SLACK_BYTES = 1024**3


def plan_expert_arena_gib(*, weights_bytes, ple_offload_bytes, expert_offload_bytes,
                          vram_bytes, offload_gib, nontext_bytes=0, kv_bytes=0) -> int:
    """GiB of VRAM to give the routing-aware expert arena (`GLQ_MOE_EXPERT_CACHE`).

    Measured on an L4 (`benchmarks/expert_arena_s1/RESULT.md`): a routing-aware arena takes
    decode from 6.436 to **17.211 tok/s**, byte-identical, and **2.441x of that survives an
    iso-VRAM control** at a matched 14.79 GiB footprint — so it is routing-awareness, not
    simply more resident weight. `--cpu-offload-gb` chooses what to keep with no knowledge of
    routing; its effective hit rate is the 2.7% that happens to be resident.

    Returns **0 unless experts are actually offloaded**. With the model resident there is no
    PCIe traffic to remove and the arena would be a second copy of weights the card already
    holds, so a box that does not need this never pays for it. Also 0 when any input is
    unknown, matching `plan_expert_offload_gib`: guessing is worse than serving as before.

    **The window keeps priority.** `kv_bytes` — what the chosen window needs, via
    `window_kv_bytes` — comes off first, so an arena is never bought with context the caller
    asked for. On a 22.49 GiB L4 the tested 8 GiB arena plus a 262,144 window needs 23.07 GiB
    against a 20.69 GiB budget, so the two genuinely do not fit; the window wins and the arena
    takes ~5.6 GiB, which is ~66 slots/layer. That costs little, because the hit-rate curve has
    flattened by there: 71 slots/layer measured 16.816 tok/s against 94 at 17.211, 2.4% apart.

    The arena sizes its own slot count and derives its token cap from this budget
    (`glq_vllm.expert_arena.arena_decision`), so this function answers only "how many GiB are
    genuinely spare".
    """
    if not weights_bytes or not vram_bytes or not expert_offload_bytes:
        return 0
    if int(offload_gib or 0) <= 0:
        return 0

    resident = (int(weights_bytes) - int(ple_offload_bytes or 0) - int(nontext_bytes or 0)
                - int(offload_gib) * 1024**3)
    spare = (kv_headroom_bytes(max(0, resident), vram_bytes)
             - int(kv_bytes or 0) - _ARENA_KV_SLACK_BYTES)
    if spare < 1024**3:
        return 0
    # Whole GiB, matching the env var's units and `plan_expert_offload_gib`'s granularity.
    gib = int(spare // 1024**3)
    # Never more than the experts themselves: past full residency the slots cannot be filled.
    return max(0, min(gib, int(int(expert_offload_bytes) // 1024**3)))


#: Pinned-RAM share permitted when deepening offload specifically to buy an expert arena.
#: Higher than `_PINNED_RAM_FRACTION`, and used ONLY by `plan_arena_and_offload`, so no
#: existing plan changes: `plan_expert_offload_gib` keeps the 0.5 share for every card and
#: checkpoint.
#:
#: Why it has to be higher at all. 0.5 caps expert offload at ~35 GiB on a 121 GiB host, which
#: leaves ~14 GiB resident on a 22 GiB card — no VRAM for an arena at any window. The measured
#: 2.674x configuration pins 42 GiB of experts beside the 23.84 GiB PLE table, i.e. 66.5 GiB,
#: which is **54.9%** of that host. At 0.5 the default could never reach the thing that was
#: measured.
#:
#: Why 0.6 and not more. That run pinned 65.3 GiB on a box with **no swap** and still had
#: ~13 GiB available at peak, so it is demonstrated rather than assumed; 0.6 of 121 GiB leaves
#: ~48 GiB unpinned for the OS, the page cache and the serving processes. Pinned pages cannot
#: be swapped, so overshooting does not run slowly — it fails to allocate or starves the
#: machine, and with no swap there is nothing to absorb it.
_ARENA_PINNED_RAM_FRACTION = 0.6


def plan_arena_and_offload(*, weights_bytes, ple_offload_bytes, expert_offload_bytes,
                           vram_bytes, base_offload_gib, ram_bytes=None, nontext_bytes=0,
                           kv_bytes=0, forced: bool = False) -> tuple[int, int]:
    """``(expert_offload_gib, expert_arena_gib)`` — deepen offload when that buys an arena.

    **Why deepening is the right trade, measured.** `plan_expert_offload_gib` offloads the
    *minimum* that fits the card and the window, which leaves no VRAM spare — so on the 22.49
    GiB L4 this feature exists for, the arena would come out at 0 GiB and ship inert. Moving
    the rest of the experts to host frees exactly the VRAM the arena needs, and the arena is
    what makes that extra offload cheap:

        offload 36, resident 7.67, no arena        ~6.8 tok/s   (the old plan)
        offload 41.48 + 8 GiB arena, resident 14.79  17.211     (measured)

    **It does not cost context.** Deeper offload *increases* KV headroom, so the window
    chosen by `plan_max_model_len` is still affordable — `kv_bytes` is subtracted before the
    arena is offered anything, and a test asserts the window still fits afterwards.

    Returns the base plan untouched when:

    * nothing was being offloaded — the model fits, there is no PCIe traffic for an arena to
      remove, and creating some would be strictly worse than serving as before. This is the
      whole meaning of "when offload is needed";
    * the caller pinned `--cpu-offload-gb`, which is an instruction rather than a starting
      point: moving more weight to host than they asked for changes the footprint they chose;
    * deepening does not actually buy a usable arena, so the PCIe cost would be paid for
      nothing.
    """
    if forced or int(base_offload_gib or 0) <= 0:
        return int(base_offload_gib or 0), 0
    if not weights_bytes or not vram_bytes or not expert_offload_bytes:
        return int(base_offload_gib or 0), 0

    full = int(int(expert_offload_bytes) // 1024**3)
    if ram_bytes:
        # The arena's own share, not the base plan's — see `_ARENA_PINNED_RAM_FRACTION`.
        # Still a share: pinned pages cannot be swapped, so a host without the room is not
        # asked to hold them.
        host_cap = (int(float(ram_bytes) * _ARENA_PINNED_RAM_FRACTION / 1024**3)
                    - math.ceil(int(ple_offload_bytes or 0) / 1024**3))
        full = min(full, max(0, host_cap))
    full = max(full, int(base_offload_gib))        # never shallower than the base plan

    arena = plan_expert_arena_gib(
        weights_bytes=weights_bytes, ple_offload_bytes=ple_offload_bytes,
        expert_offload_bytes=expert_offload_bytes, vram_bytes=vram_bytes,
        offload_gib=full, nontext_bytes=nontext_bytes, kv_bytes=kv_bytes)
    if arena <= 0:
        # Deepening bought nothing, so do not pay PCIe for it.
        return int(base_offload_gib), 0
    return full, arena


#: Worst-case KV cost per token among the served families, used to price a window before
#: vLLM exists to ask. Taken from **measured pools**, not from arithmetic: two
#: Qwen3.8-Flash-Next serves on an RTX PRO 6000 (vLLM 0.30.0, PLE offload) report
#: `Available KV cache memory` and `KV cache size` together, which divide out to
#:
#:     72.07 GiB / 2,744,055 tokens = 28,201 B   (131072 window, 38 GiB expert offload)
#:     11.17 GiB /   445,792 tokens = 26,903 B   (262144 window, PLE offload only)
#:
#: The two differ by 4.8%, and that is structural rather than noise: a GDN hybrid's
#: linear-attention layers hold constant-size recurrent state **per sequence**, so total pool
#: is not a per-token quantity at all and any single B/token figure depends on the
#: configuration it was measured in. Hence a conservative envelope over the worst observed
#: value, never a fit. Exact per-model KV math is vLLM's job; duplicating it client-side
#: breaks on every backend change.
#:
#: History, because the direction of error is what matters here. This started as gemma-4's
#: 6.15 GiB-per-262144-request (23,459... i.e. 25,190 B/token) with a comment claiming it was
#: the worst case among the served families *because* hybrid-GDN models keep constant-size
#: state and sliding-window layers page smaller. Flash-Next then beat it twice over: vLLM's
#: own refusal quoted 6.55 GiB for one request (26,829 B/token), and the pools above are
#: higher again. Each time the old value was low, which is the direction that promises a
#: window and then cannot serve one request in it.
_KV_BYTES_PER_TOKEN = 28_201

#: 131072 and 262144 are here because both published long-context families declare
#: `max_position_embeddings = 262144` (gemma-4 and Qwen3.8-Flash-Next, verified live), so the
#: old 65536 ceiling was the tier list stopping short, not a limit any card imposed.
_WINDOW_TIERS = (8192, 16384, 32768, 65536, 131072, 262144)

#: Default concurrency a window must be affordable at — half of DEFAULT_MAX_NUM_SEQS, which
#: deliberately over-subscribes: beyond the pool vLLM queues rather than failing. This is the
#: constant that keeps a 23 GiB L4 serving a 14.4 GiB 26B at 8192 (headroom ≈ 2.8 GiB;
#: 16384×8×26.2 KiB ≈ 3.3 GiB does not fit). `glq-code` passes 1 instead: pi issues one
#: request at a time, and one stream is the difference between 131072 and the full 262144.
_WINDOW_CONCURRENCY = 8


#: Cushion over the measured per-token cost, applied when a window is both CHOSEN and
#: reserved for. Without it the plan lands on vLLM's stated minimum exactly — a 95 GiB card
#: planned a 262144 window against a pool of precisely 6.55 GiB, which is the figure vLLM
#: quotes as what one such request *needs*. There is then nothing left for KV block
#: granularity, for a second request's partial blocks, or for a family whose per-token cost
#: beats the anchor (one already did, by 6.5%). Coming up short is not graceful: vLLM refuses
#: at startup.
#:
#: 15% is invisible on the cards that cannot afford slack — 8192×8 is 1.64 GiB against a
#: 2 GiB `_MIN_KV_BYTES` floor, so a 23 GiB L4's plan is unchanged to the digit — and free on
#: the cards that can.
_KV_SAFETY_FACTOR = 1.15


#: Offload a caller may spend, beyond what fitting the card requires, to reach a longer window.
#: `0` — the default, and what `glq-chat` uses — means never trade: a chat session at 8k tokens
#: must not pay PCIe on every token for context it will not fill. `glq-code` opts in, because
#: for a coding agent the window IS the product.
#:
#: The cost is measured rather than assumed — Flash-Next-3bpw-ple4, fixed concurrency B=1,
#: 3 repeats per arm, two cards:
#:
#:     RTX PRO 4500, 31.9 GiB, window 8192     27 GiB  14.496 tok/s
#:                                             34      13.877   (-4.3%)
#:                                             42      13.310   (-8.2%)   -> 0.55%/GiB
#:     RTX PRO 6000, 96 GiB, window 262144      0 GiB  31.949 tok/s
#:                                              8      28.379  (-11.2%)
#:                                             24      23.472  (-26.5%)   -> 1.10%/GiB
#:
#: **The percentage is not portable and must not be quoted as the cost.** It doubles on the
#: faster card for the reason a fixed cost always does: the same transfer is a larger share of
#: a shorter step. What IS stable is the absolute per-token cost — 0.409 ms/GiB on the 4500 and
#: 0.460-0.493 ms/GiB on the 6000 (three points, essentially linear) — and it is stable because
#: both boxes are **PCIe 5 x16**, not because it is a property of the GPU.
#:
#: The bottleneck is the link, not host memory: PCIe 5 x16 is ~55 GB/s effective against
#: ~358 GB/s for DDR5-5600 across the 8 channels of the Xeon in that box, a 6.5x margin. Per
#: token the GPU fetches only the ROUTED experts, which at 0.47 ms/GiB works out to ~26 MB of
#: traffic per GiB offloaded, i.e. ~2.5% of the offloaded bytes — the routing sparsity. Expect
#: roughly double on a PCIe 4 box, and more on a model with denser routing.
#:
#: So 8 GiB costs ~4.4% where this policy actually fires and ~9% where it does not, and that
#: asymmetry is structural rather than lucky: the trade only fires on a card too small to hold
#: the model, which is the slower card with the gentler percentage. A 96 GiB card plans zero
#: offload and never pays the steep slope that card would charge.
#:
#: Still a rail, not a fitted value. On every card checked the declared-expert and pinnable-RAM
#: caps bind first — a 23 GiB L4 spends 3 GiB, a 32 GiB g7 spends 5 — which is where a
#: constant resting on two boxes belongs: bounding the unmeasured tail, not deciding the
#: common case.
WINDOW_OFFLOAD_MAX_EXTRA_GIB = 8


def _fraction_arg(util: float) -> str:
    """A VRAM fraction as it should appear on a command line: three decimals, rounded UP.

    The plan is a raw division, so the serve command carried
    `--gpu-memory-utilization 0.6723162690529302` — 17 digits of noise in a line users read
    in the log and paste into a shell. Up rather than nearest, because rounding down hands
    back a few MB of the pool that was just sized to hold a specific window, and the cap is
    re-applied so this can never ask for more of the card than `_MAX_UTILIZATION` allows.

    Trailing zeros are trimmed so a value the user supplied comes back as they wrote it:
    echoing `--gpu-memory-utilization 0.35` as `0.350` is the same number and a worse answer.
    """
    capped = min(math.ceil(util * 1000) / 1000, _MAX_UTILIZATION)
    return f"{capped:.3f}".rstrip("0").rstrip(".")


def window_kv_bytes(max_model_len, concurrency=_WINDOW_CONCURRENCY) -> int:
    """KV bytes to RESERVE for a `max_model_len` window at `concurrency` requests.

    The measured cost (`_KV_BYTES_PER_TOKEN`) plus `_KV_SAFETY_FACTOR`, not the bare cost —
    see that constant for why a plan that exactly meets the requirement is a plan that fails
    to start.

    One function so the window planner and the pool planner cannot drift apart: pricing the
    window differently in the two places is exactly how a 262144 window got paired with a
    2 GiB pool.
    """
    cost = int(max_model_len) * int(concurrency) * _KV_BYTES_PER_TOKEN
    return int(cost * _KV_SAFETY_FACTOR)


#: Largest window the AUTO planner will choose, by card size. `--max-model-len` overrides it.
#:
#: Needed because the window is chosen before the expert arena exists, so left alone the
#: planner tiers up to whatever the card can hold and the arena gets the remainder. Measured
#: on a 24 GiB card: 262,144 leaves a 1 GiB arena, 11 slots/layer, decode hit rate ~0.33 —
#: where 131,072 leaves 5 GiB, 59 slots, ~0.68. PCIe traffic scales as (1 - hit), so that is
#: roughly 4x the traffic on a term measured at 45% of the un-arena'd step.
#:
#: The thresholds are deliberately generous about context rather than set where the arena
#: saturates (~94 slots/layer): these are the windows a user of each card should get. Cards
#: report a little under nominal — an L4 reads 22.49, a 24 GB 3090 ~23.6, a 32 GB 5090 ~31.4 —
#: so the bands are set between the classes, not at them.
_WINDOW_CEILING_BY_VRAM_GIB = ((20, 32_768), (28, 131_072))


def window_ceiling_for_vram(vram_bytes) -> int:
    """Largest window the auto planner may choose on a card this size.

    Returns the model's own maximum when the card is unknown: the same stance as every other
    planner here, which is to not invent a limit from an input it does not have.
    """
    if not vram_bytes:
        return _WINDOW_TIERS[-1]
    gib = float(vram_bytes) / 1024**3
    for ceiling_gib, window in _WINDOW_CEILING_BY_VRAM_GIB:
        if gib <= ceiling_gib:
            return window
    return _WINDOW_TIERS[-1]


def plan_max_model_len(*, weights_bytes, vram_bytes, model_max_len,
                       floor=DEFAULT_MAX_MODEL_LEN,
                       concurrency=_WINDOW_CONCURRENCY):
    """The served context window: the largest tier the card could afford to hold.

    A fixed 8192 was designed for 24 GiB desktops and wastes a 96 GiB card; the model's
    declared maximum (262,144) drowns a small one. Pick the largest tier whose full window,
    at `concurrency` requests and the worst-case per-token anchor, fits the headroom left at
    `_MAX_UTILIZATION` — what the card *could* give, not what the minimal pool reserves.
    `plan_gpu_memory_utilization` is then told the answer so the pool grows to match.

    Measuring against the minimal pool instead is what made this feature inert for exactly
    the models that needed it: that pool is weights + overhead + `_MIN_KV_BYTES`, so the
    headroom was 2 GiB on a 96 GB card and on a 23 GiB one alike, and no tier above 8192
    could ever be chosen for a large checkpoint.

    Any unknown input returns the floor: tiering up blind is strictly worse than a
    small window, because vLLM refuses a --max-model-len above the model's declared
    maximum (SmolLM2-class models declare 8192) and then nothing serves at all.
    """
    if not weights_bytes or not vram_bytes or not model_max_len:
        return floor
    headroom = kv_headroom_bytes(weights_bytes, vram_bytes)
    chosen = floor
    for tier in _WINDOW_TIERS:
        if window_kv_bytes(tier, concurrency) <= headroom:
            chosen = max(chosen, tier)
    # Card-indexed ceiling, so the window does not consume the VRAM the expert arena needs.
    # Never below the floor: a small card still serves its minimum window.
    return min(chosen, int(model_max_len), max(floor, window_ceiling_for_vram(vram_bytes)))


def default_log_path():
    return Path(os.environ.get("GLQ_HOME", Path.home() / ".glq")) / "vllm.log"


def server_up(base_url: str, timeout: float = 1.5) -> bool:
    """Is something already answering the OpenAI API at `base_url`?

    Deliberately urllib rather than the `openai` client: this runs before the UI is built and
    must not depend on it, and a bare GET is all the question needs.
    """
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:      # noqa: S310
            return 200 <= r.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _ninja_present(which=shutil.which) -> bool:
    """Is ninja reachable? FlashInfer's JIT shells out to it **by name**, with no override.

    `install.sh` installs it into the venv, but running a venv binary by absolute path does
    not put the venv's bin/ on PATH — which is why `child_env` prepends it.
    """
    return bool(which("ninja") or which(
        "ninja", path=os.path.dirname(sys.executable)))


def _nvcc_present(which=shutil.which) -> bool:
    """Is the NVIDIA CUDA Toolkit's compiler on PATH (or under CUDA_HOME)?

    `nvcc` is the component that decides this: FlashInfer shells out to it. The toolkit is
    what the user installs; nvcc is how we detect it.
    """
    if which("nvcc"):
        return True
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    return bool(home) and os.path.exists(os.path.join(home, "bin", "nvcc"))


def _compute_cap(run=subprocess.run):
    """The GPU's compute capability as reported by nvidia-smi, e.g. "12.0", or None."""
    try:
        out = run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                  capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    line = (out.stdout or "").strip().splitlines()
    return line[0].strip() if line else None


#: Below this, FlashInfer ships prebuilt kernels and needs no toolkit.
_FLASHINFER_PREBUILT_BELOW = 12.0


def flashinfer_env(compute_cap=None, have_nvcc=None, have_ninja=None) -> dict:
    """Environment overrides needed for vLLM to start on this machine.

    Measured on an RTX PRO 6000 (sm_120), vLLM 0.27.1, no CUDA Toolkit: GLQ's own prebuilt
    kernel loads and decodes fine, but EngineCore dies before generating a token because
    vLLM's sampler backend JIT-compiles for `120f` and cannot find nvcc. The same container
    with VLLM_USE_FLASHINFER_SAMPLER=0 answered "Paris".

    Scoped deliberately to new architectures: FlashInfer ships prebuilt kernels for sm_86
    and sm_89, so switching samplers there would trade speed away to fix nothing. Unknown
    capability changes nothing either — a silent sampler swap on a guess is worse than the
    error, which at least names its own cause.
    """
    if compute_cap is None:
        compute_cap = _compute_cap()
    if have_nvcc is None:
        have_nvcc = _nvcc_present()
    if have_ninja is None:
        have_ninja = _ninja_present()
    # nvcc alone is not enough. Measured on an RTX PRO 6000 with a CUDA 13.2 toolkit on
    # PATH: FlashInfer still died with `FileNotFoundError: 'ninja'`, because the JIT shells
    # out to ninja and the venv's bin/ was not on the child's PATH. A toolchain missing
    # either half cannot build, so either half is grounds to fall back.
    if (have_nvcc and have_ninja) or not compute_cap:
        return {}
    try:
        cap = float(compute_cap)
    except (TypeError, ValueError):
        return {}
    return {"VLLM_USE_FLASHINFER_SAMPLER": "0"} if cap >= _FLASHINFER_PREBUILT_BELOW else {}


def _installed_vllm_version() -> str:
    from importlib.metadata import version
    return version("vllm")


def detect_device(vllm_version=_installed_vllm_version, vram=None) -> str:
    """"cpu" or "cuda" — the backend this venv can actually serve with.

    The INSTALLED WHEEL wins over the live GPU: a `+cpu` vLLM cannot use a GPU however
    the flags read (real case: a GPU box where the user installed the CPU stack), and
    reading the wheel's local version is a metadata lookup, not an import. Only when the
    version is unreadable does the GPU probe decide.
    """
    try:
        if "+cpu" in vllm_version():
            return "cpu"
    except Exception:                                    # noqa: BLE001 - fall to the probe
        pass
    if vram is None:
        from glq.installer.hardware import vram_bytes as vram_probe
        vram = vram_probe
    return "cuda" if vram() is not None else "cpu"


#: VLLM_CPU_KVCACHE_SPACE bounds, GiB. 8 is the value the CPU serving path was validated
#: with; at the ~24.6 KiB/token worst-case anchor it already holds ~340k tokens, far past
#: any served window at CPU concurrency — more would only crowd out the weights.
_CPU_KV_MIN_GIB, _CPU_KV_MAX_GIB = 2, 8


#: What the CPU serving stack costs beyond the weights and the KV pool. Measured at steady
#: state with `glq-chat` serving the 13.9 GiB 26B-A4B on an 8-vCPU box (per-process PSS from
#: smaps_rollup, summed against /proc/meminfo AnonPages, which agreed to 0.1 GiB):
#:
#:     VLLM::Worker      22.56 GiB  = 13.9 weights + 5.0 pool + 3.7 activations
#:     chat/gradio UI     2.09 GiB  — same machine, and glq-chat starts it
#:     EngineCore + API   1.06 GiB
#:     ------------------------------------------------------------------
#:     beyond weights + pool: 6.8 GiB, rounded up
#:
#: An earlier 2 GiB here came from reading one worker's RSS and forgetting that the UI and
#: the engine processes live in the same RAM. It under-counted by 4.7 GiB.
_CPU_RUNTIME_OVERHEAD_BYTES = 7 * 1024 ** 3

#: Share of RAM the *anonymous* demand — weights + pool + runtime — may reach. The rest is
#: for the page cache, and on CPU that is not a luxury: the loader streams the whole
#: checkpoint through it, and a box with no swap configured (the cloud default) can reclaim
#: nothing else. Measured on a 30.8 GiB box holding the 13.9 GiB 26B-A4B, with the real
#: overhead above:
#:
#:     pool 7 GiB -> 27.7 GiB anonymous = 90%  kswapd0 at 100%, 85% iowait,
#:                                             buff/cache 100 MiB, ssh unreachable
#:     pool 5 GiB -> 25.7 GiB anonymous = 83%  served fine, 4.42 GiB still available
#:
#: 0.85 admits the configuration that worked and refuses the one that did not.
_CPU_ANON_FRACTION = 0.85

#: What to leave free on a machine that is already busy, beyond the weights, the pool and
#: the runtime. The healthy measured run finished with 4.42 GiB still available, so this is
#: that observation rounded down: enough page cache to stream a checkpoint without evicting
#: it in a loop, and enough slack that the rest of the desktop is not pushed into reclaim.
_CPU_HEADROOM_BYTES = 4 * 1024 ** 3

#: The ceiling when the checkpoint size could not be looked up, so the arithmetic above is
#: unavailable. 4 GiB is the pool that served the 13.9 GiB 26B-A4B across every run of this
#: work without pressure; 8 would be betting that the model is small.
_CPU_KV_UNKNOWN_MAX_GIB = 4


def plan_cpu_kvcache_gib(ram_bytes, weights_bytes=None, available_bytes=None) -> int:
    """The CPU KV pool in GiB, clamped to [2, 8].

    On a GPU the weights and the KV cache come out of VRAM while the loader, the runtime
    and the page cache come out of RAM, so sizing the pool against one number works. On CPU
    they all come out of the *same* RAM, which is why this needs `weights_bytes`: a quarter
    of RAM is 7 GiB on a 32 GiB box whether the checkpoint is 1.8 GiB or 13.9, and in the
    second case that overcommits the machine.

    Without a checkpoint size — an unknown repo, a local path we could not measure — it
    keeps the old fraction-of-RAM answer. That is the input we have, and a pool that is
    too small costs context rather than the machine.
    """
    if not ram_bytes:
        return _CPU_KV_MAX_GIB
    if not weights_bytes:
        # Size unknown — an offline box, a local path, a 404. A quarter of RAM was the old
        # rule and it is a gamble here: on a 32 GiB machine it hands out 7 GiB whether the
        # checkpoint is 1.8 GiB or 13.9. Cap the guess at the largest pool observed serving
        # comfortably, because the two errors are not symmetric — too small costs context,
        # too large hangs a machine with no swap.
        quarter = int(ram_bytes / 2**30) // 4
        return max(_CPU_KV_MIN_GIB, min(_CPU_KV_UNKNOWN_MAX_GIB, quarter))
    # Two bounds, and the tighter one wins.
    #
    #   * a share of what is INSTALLED — the measured ceiling, which keeps a slice of RAM
    #     for the page cache the loader streams the checkpoint through;
    #   * what is actually FREE right now, less the headroom the healthy run was measured
    #     to leave. Sizing against MemTotal assumes an idle machine, and a desktop with a
    #     browser open can be 10 GiB down before this starts. On CPU that 10 GiB comes out
    #     of the same pool as the weights, so planning as if it were not there is how a
    #     plan that looks fine on paper takes the machine down.
    budget = _CPU_ANON_FRACTION * ram_bytes
    if available_bytes:
        budget = min(budget, available_bytes - _CPU_HEADROOM_BYTES)
    return max(_CPU_KV_MIN_GIB,
               min(_CPU_KV_MAX_GIB,
                   int((budget - weights_bytes - _CPU_RUNTIME_OVERHEAD_BYTES) / 2**30)))


def child_env(device: str = "cuda", ram_bytes=None, weights_bytes=None,
              available_bytes=None, ple_offload: bool = False,
              expert_arena_gib: int = 0, max_num_seqs: int = 0,
              min_step_tokens: int = 0) -> dict:
    """The environment `vllm serve` is started with.

    PYTHONUNBUFFERED: without it the child block-buffers into the log file, so the lines
    explaining a failure arrive long after we have given up — or never, if it dies with them
    unflushed.

    PATH: the venv's own bin/ goes first. We launch vLLM by absolute path, which does NOT
    put that directory on PATH the way `source activate` would — so a child that shells out
    to a sibling tool cannot find it. Measured: FlashInfer JIT-compiles its sampler on
    sm_120, runs `ninja` by name, and dies with FileNotFoundError while
    ~/.glq/venv/bin/ninja sits there unused, installed by install.sh for this exact purpose.
    """
    bindir = os.path.dirname(sys.executable)
    path = os.environ.get("PATH", "")
    parts = [p for p in path.split(os.pathsep) if p and p != bindir]
    env = {**os.environ, "PYTHONUNBUFFERED": "1",
           "PATH": os.pathsep.join([bindir, *parts])}
    if device == "cpu":
        # The CPU backend takes its KV pool from this env var (GiB). A user-set value
        # always wins; the flashinfer probe is skipped — it answers a GPU question with
        # an nvidia-smi subprocess this path has no use for.
        env.setdefault("VLLM_CPU_KVCACHE_SPACE",
                       str(plan_cpu_kvcache_gib(ram_bytes, weights_bytes, available_bytes)))
        # vLLM's auto CPU binding takes one logical CPU per physical core and then holds
        # one back for itself, so a 4-core machine runs GLQ's kernels on 3. Decode is
        # memory-bound and scales with cores until the bandwidth saturates: serving the
        # 26B-A4B on an 8-vCPU Sapphire Rapids measured 3.0 tok/s on 3 cores and 3.3 on 4
        # (binding all 8 logical CPUs added nothing — SMT siblings share the same
        # bandwidth). Nothing is competing for that core during single-stream chat, which
        # is what this supervisor serves, so claim it. setdefault: a busy box can hold
        # cores back by setting this itself.
        env.setdefault("VLLM_CPU_NUM_OF_RESERVED_CPU", "0")
        return env
    if ple_offload:
        # vLLM already defaults this ON, but a checkpoint whose n-gram table does not fit
        # beside its decoder CANNOT SERVE without it -- at a 262k context the resident
        # configuration stops at startup needing 6.55 GiB of KV it does not have. Depending on
        # an upstream default for that is the kind of thing that breaks on a vLLM bump, and
        # GLQ's own refusal message already documents the setting being flipped. setdefault, so
        # a user who deliberately exports 0 still gets their answer (and GLQ's clear refusal).
        env.setdefault("VLLM_PLE_CPU_OFFLOAD", "1")
    if expert_arena_gib and int(expert_arena_gib) > 0:
        # The routing-aware expert arena. Measured on an L4: decode 6.436 -> 17.211 tok/s,
        # byte-identical, and 2.441x of that survives an iso-VRAM control — so it is
        # routing-awareness rather than simply more resident weight. Planned only when experts
        # are offloaded (`plan_expert_arena_gib`), so a card that fits the model is untouched.
        #
        # setdefault, following VLLM_PLE_CPU_OFFLOAD above: a user who exports a value keeps
        # it, and `GLQ_MOE_EXPERT_CACHE=0` is how they opt out.
        env.setdefault("GLQ_MOE_EXPERT_CACHE", str(int(expert_arena_gib)))
        if max_num_seqs and int(max_num_seqs) > 0:
            # The arena path's token cap, which is the widest decode step it may be asked to
            # serve: one token per concurrent stream, times `1 + num_spec` when MTP is on,
            # because a verify step carries the draft tokens alongside the real one. Callers
            # pass `VllmSupervisor.step_tokens`, which is that product.
            #
            # A ceiling, not a demand: the arena lowers it when its slots cannot cover it.
            env.setdefault("GLQ_MOE_ARENA_MAX_TOKENS", str(int(max_num_seqs)))
        if min_step_tokens and int(min_step_tokens) > 1:
            # The floor below which lowering the cap stops being safe. Without speculation
            # every decode step is one token wide, so any cap >= 1 serves them and a narrow
            # arena just degrades. With MTP on, every step is at least `1 + num_spec` wide, so
            # a cap under that serves NO decode step and the arena is pure wasted VRAM —
            # VRAM that KV needs, which is how an 8 GiB arena plus MTP once produced
            # `Available KV cache memory: -0.31 GiB`. The arena refuses rather than allocate.
            env.setdefault("GLQ_MOE_ARENA_MIN_TOKENS", str(int(min_step_tokens)))
    # After PATH, so the probe sees the tools the child will actually have.
    env.update(flashinfer_env())
    return env


def _hf_download_bytes() -> int:
    """Bytes of in-flight HF downloads: the sizes of `*.incomplete` blobs in the hub cache.

    A checkpoint download writes NOTHING to vLLM's log, so this is the only sign of life
    during the phase that takes longest on a first run. Shallow glob on the hub layout
    (`models--*/blobs/*.incomplete`) — cheap enough to poll every 2 s even on a large cache.
    """
    root = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    total = 0
    for p in (root / "hub").glob("models--*/blobs/*.incomplete"):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


class VllmSupervisor:
    """Start vLLM if nothing is serving, wait for it, and stop it again on the way out.

    Use it as a context manager; `__exit__` is the VRAM-release guarantee and has to hold on
    the exception path, which is exactly where it matters.
    """

    def __init__(self, *, model, port=8000, base_url=None, gpu_memory_utilization=None,
                 vllm_bin=None, extra_args=(), serve=True,
                 spawn=subprocess.Popen, probe=server_up,
                 sleep=time.sleep, monotonic=time.monotonic,
                 download_bytes=_hf_download_bytes,
                 killpg=os.killpg, getpgid=os.getpgid,
                 timeout=DEFAULT_READY_TIMEOUT, out=None,
                 log_path=None, verbose=False,
                 report_every=DEFAULT_REPORT_EVERY,
                 weights_bytes=None, vram_bytes=None,
                 max_model_len=None, model_max_len=None,
                 max_model_len_floor=DEFAULT_MAX_MODEL_LEN, fp8_kv=False,
                 max_num_seqs=None, device=None, ram_bytes=None,
                 ple_offload_bytes=0, expert_offload_bytes=0, nontext_bytes=0,
                 window_concurrency=_WINDOW_CONCURRENCY, expert_offload_gib=None,
                 window_offload_extra_gib=0, mtp_tokens=0, decoder_layers=None,
                 mtp_layers=0):
        self.model = model
        self.port = int(port)
        self.base_url = base_url or f"http://127.0.0.1:{self.port}/v1"
        #: Draft tokens per step for MTP speculation, 0 = off. Set before anything it changes:
        #: it decides whether the MTP head is loaded, which moves both of the figures below.
        #: Measured on an L4 (`benchmarks/expert_arena_s1/RESULT.md`): with the arena,
        #: `num_spec=1` reaches 19.692 tok/s (3.060x) and beats `num_spec=3` at 18.569 —
        #: the step cost is steeply linear in drafted positions, so the extra acceptance 3
        #: buys does not cover the extra positions.
        self.mtp_tokens = max(0, int(mtp_tokens or 0))
        # Host offload is planned FIRST, because everything downstream is sized from what
        # actually lands in VRAM. Passing the file size to the pool and window planners is how
        # a PLE checkpoint ended up under-sized by ~24 GiB: the n-gram table was counted
        # against the card it never occupies.
        self.ple_offload_bytes = int(ple_offload_bytes or 0)
        self.expert_offload_bytes = int(expert_offload_bytes or 0)
        #: The MTP block's own 512-expert MoE layer, which `quantize_model._offloadable_bytes`
        #: drops from every total: it is "mutually exclusive, non-text first", so once the head
        #: is quantized `mtp.layers.0.mlp.experts.*` is claimed by the non-text branch and then
        #: discarded by `if not mtp_is_loaded`. The tell is that `expert_offload_bytes` is
        #: byte-identical (45,770,637,312) for the bf16-head and quantized-head builds of
        #: Flash-Next, though only one of them loads that layer.
        #:
        #: Only counted when speculation is on, because only then is the block loaded. Derived
        #: from the two declared numbers rather than estimated, and 0 when the layer count is
        #: unknown — the planner's standing rule that an unknown input means fall back.
        #:
        #: This omission is the measured shortfall that stopped arena+MTP before: an 8 GiB
        #: arena plus MTP reported `Available KV cache memory: -0.31 GiB`.
        self.mtp_expert_bytes = 0
        if self.mtp_tokens and decoder_layers and self.expert_offload_bytes:
            per_layer = self.expert_offload_bytes // int(decoder_layers)
            self.mtp_expert_bytes = per_layer * max(0, int(mtp_layers or 0))
            self.expert_offload_bytes += self.mtp_expert_bytes
        #: Bytes outside the text decoder (MTP head, vision tower). Not offloaded — never
        #: loaded, so they cost neither VRAM nor pinned host RAM. Subtracting them is what
        #: closed a measured 5.62 GiB over-estimate (15.72 predicted vs 10.1 reported), worth
        #: two window tiers on a 24 GB card.
        #:
        #: Sound only while speculation is OFF, which is the condition this line now enforces
        #: rather than assuming: MTP loads the head, so subtracting it would under-plan the
        #: footprint by exactly that much. The declared figure is kept so the plan can explain
        #: what turning MTP on cost.
        self._nontext_declared = int(nontext_bytes or 0)
        self.nontext_bytes = 0 if self.mtp_tokens else self._nontext_declared
        #: How many concurrent requests the window is priced for. 1 for an agent driving a
        #: single stream, `_WINDOW_CONCURRENCY` for a chat server. Set before the offload plan
        #: because that plan now prices a window, and a window priced at the wrong concurrency
        #: is the wrong window.
        self.window_concurrency = max(1, int(window_concurrency))
        # An explicit budget wins, exactly as `--gpu-memory-utilization` does.
        #
        # `None` means "plan it"; 0 is a real answer meaning "serve this resident", which is
        # why the two cannot be collapsed.
        self._expert_offload_forced = expert_offload_gib is not None
        self.expert_offload_gib = 0
        if self._expert_offload_forced:
            self.expert_offload_gib = max(0, int(expert_offload_gib))
        elif self.ple_offload_bytes or self.expert_offload_bytes:
            self.expert_offload_gib = self._plan_offload(
                weights_bytes=weights_bytes, vram_bytes=vram_bytes, ram_bytes=ram_bytes,
                max_model_len=max_model_len, model_max_len=model_max_len,
                floor=max_model_len_floor,
                extra_cap=max(0, int(window_offload_extra_gib or 0)))
        resident_bytes = weights_bytes
        if weights_bytes:
            resident_bytes = max(0, int(weights_bytes) - self.ple_offload_bytes
                                 - self.nontext_bytes
                                 - self.expert_offload_gib * 1024**3)
        self.vllm_bin = vllm_bin or os.path.join(os.path.dirname(sys.executable), "vllm")
        self.extra_args = list(extra_args)
        # The window is decided BEFORE the pool, and the pool is then sized to hold it. The
        # reverse order is what made the window planner inert on exactly the configurations
        # it was written for: the minimal pool leaves 2 GiB of headroom whatever the card, so
        # no tier above 8192 was reachable for a large checkpoint.
        #
        # max_model_len=None means "size it": the largest tier the card could afford,
        # clamped to the model's declared maximum. With any sizing input unknown this lands
        # exactly on the old fixed default, so callers that never pass the lookups keep
        # today's behavior.
        self._window_note = ""
        #: Whether the window was sized rather than pinned. The re-plan after deepening must
        #: touch only the automatic choice; a pinned `--max-model-len` is an instruction.
        max_model_len_was_auto = max_model_len is None
        if max_model_len is None:
            max_model_len = plan_max_model_len(
                weights_bytes=resident_bytes, vram_bytes=vram_bytes,
                model_max_len=model_max_len, floor=max_model_len_floor,
                concurrency=self.window_concurrency)
            if max_model_len > max_model_len_floor:
                self._window_note = (
                    f" (sized from KV headroom at {self.window_concurrency} concurrent "
                    f"request{'' if self.window_concurrency == 1 else 's'}; "
                    f"--max-model-len overrides)")
        self.max_model_len = int(max_model_len)
        self.max_model_len_base = self.max_model_len
        # An explicit flag always wins; otherwise size the pool from the checkpoint AND the
        # window just chosen, because a fixed fraction starves anything bigger than it and a
        # minimal KV reservation starves the window. A PINNED window is sized for too: that
        # is the case the user hit, `--max-model-len 262144` against a 2 GiB pool.
        self.gpu_memory_utilization = (
            float(gpu_memory_utilization) if gpu_memory_utilization is not None
            else plan_gpu_memory_utilization(
                weights_bytes=resident_bytes, vram_bytes=vram_bytes,
                kv_bytes=window_kv_bytes(self.max_model_len, self.window_concurrency)))
        # None = auto: follow the wheel/GPU detection. Explicit "cpu"/"cuda" is for tests
        # and for callers that already decided (the config records the installer's choice,
        # but the wheel in THIS venv is what actually serves — so live detection is the
        # default even then).
        self.device = device if device in ("cpu", "cuda") else detect_device()
        self._available_bytes = None
        if self.device == "cpu" and ram_bytes is None:
            try:
                from glq.installer.hardware import (available_ram_bytes as _avail,
                                                    ram_bytes as _ram)
                ram_bytes = _ram()
                # Read at start, because that is when the decision is made: what the
                # machine has free now is what this server can have. A browser opened
                # afterwards is not something any plan can anticipate.
                self._available_bytes = _avail()
            except Exception:                            # noqa: BLE001 - pool falls to 8 GiB
                ram_bytes = None
        self._ram_bytes = ram_bytes
        #: Kept so the shortfall note can re-ask the affordability question the plan asked.
        self._vram_bytes = vram_bytes
        # A --gpu-memory-utilization on the CPU backend would be silently meaningless;
        # remember it so start() can say so once instead of dropping it wordlessly.
        self._ignored_gpu_flag = (self.device == "cpu"
                                  and gpu_memory_utilization is not None)
        self.max_num_seqs = int(max_num_seqs) if max_num_seqs is not None else (
            DEFAULT_CPU_MAX_NUM_SEQS if self.device == "cpu" else DEFAULT_MAX_NUM_SEQS)
        #: Widest decode step the arena may be asked to serve: one token per stream without
        #: speculation, and `1 + mtp_tokens` per stream with it, because an MTP verify step
        #: carries the draft tokens alongside the real one.
        #:
        #: This is the quantity the measured arena+MTP arms needed: "an MTP verify step carries
        #: 1 + num_spec = 4 tokens, so GLQ_MOE_ARENA_MAX_TOKENS=4 brings the requirement to
        #: 40 <= 71 slots". With `max_num_seqs` there instead, a 6 GiB arena over 49 containers
        #: refused a cap of 8 (8 x top_k 10 = 80 > 71) and three arms silently measured no
        #: arena at all while looking entirely healthy.
        self.step_tokens = self.max_num_seqs * (1 + self.mtp_tokens)
        #: The NARROWEST step speculation produces — one stream's verify step. The arena's cap
        #: may safely land between this and `step_tokens`: wider steps simply take today's
        #: exact path. Below it, though, the arena serves NO decode step at all, because with
        #: MTP on every step is at least this wide — so a cap under this is not graceful
        #: degradation but wasted VRAM, and the arena is dropped instead.
        self.min_step_tokens = 1 + self.mtp_tokens
        #: GiB for the routing-aware expert arena, planned after the window and the offload
        #: budget so the window keeps priority — an arena is never bought with context the
        #: caller asked for. 0 on CPU and whenever nothing is offloaded, so a box that does
        #: not need it is untouched. See `plan_expert_arena_gib`.
        self.expert_arena_gib = 0
        self.expert_offload_base_gib = self.expert_offload_gib
        if self.device != "cpu":
            # One call, so the offload depth and the arena size cannot disagree: deepening
            # offload is what frees the VRAM the arena needs, and an arena that did not
            # materialise must not leave the deeper offload behind paying PCIe for nothing.
            #: What the window search chose on its own — the SMALLEST offload reaching the
            #: chosen tier. Kept separate from the final figure because deepening for the
            #: arena makes the final one deliberately larger, and "smallest that reaches the
            #: tier" is still a true and tested property of the search itself.
            self.expert_offload_base_gib = self.expert_offload_gib
            self.expert_offload_gib, self.expert_arena_gib = plan_arena_and_offload(
                weights_bytes=weights_bytes, ple_offload_bytes=self.ple_offload_bytes,
                expert_offload_bytes=self.expert_offload_bytes,
                vram_bytes=self._vram_bytes, base_offload_gib=self.expert_offload_gib,
                ram_bytes=ram_bytes, nontext_bytes=self.nontext_bytes,
                kv_bytes=window_kv_bytes(self.max_model_len, self.window_concurrency),
                forced=self._expert_offload_forced)
            if (self.expert_offload_gib > self.expert_offload_base_gib
                    and max_model_len_was_auto):
                # Deepening freed VRAM the window could not claim, because the window was
                # chosen from the BASE footprint. Measured across cards at 100-110 GiB of host
                # RAM, that left the window at 8192 where the deepened footprint carries
                # 65,536 to 262,144 — up to 16x the context, from VRAM already free. No extra
                # PCIe is paid: the offload depth is settled by this point.
                #
                # Only ever an increase, and `window_ceiling_for_vram` bounds it so the window
                # cannot take the arena's share. The arena is then re-sized against the window
                # actually chosen.
                deeper = plan_max_model_len(
                    weights_bytes=max(0, int(weights_bytes) - self.ple_offload_bytes
                                      - self.nontext_bytes
                                      - self.expert_offload_gib * 1024**3),
                    vram_bytes=self._vram_bytes, model_max_len=model_max_len,
                    floor=max_model_len_floor, concurrency=self.window_concurrency)
                if deeper > self.max_model_len:
                    #: The window the base footprint could afford, before deepening freed any
                    #: VRAM. Kept because it is the quantity that answers "did this context
                    #: come from counting right or from spending more PCIe" — the re-plan does
                    #: spend more, so a test asserting the former has to read this.
                    self.max_model_len_base = self.max_model_len
                    self.max_model_len = deeper
                    self.expert_arena_gib = plan_expert_arena_gib(
                        weights_bytes=weights_bytes,
                        ple_offload_bytes=self.ple_offload_bytes,
                        expert_offload_bytes=self.expert_offload_bytes,
                        vram_bytes=self._vram_bytes,
                        offload_gib=self.expert_offload_gib,
                        nontext_bytes=self.nontext_bytes,
                        kv_bytes=window_kv_bytes(self.max_model_len,
                                                 self.window_concurrency))
        self.fp8_kv = bool(fp8_kv)
        self.serve = serve
        self._spawn, self._probe = spawn, probe
        self._sleep, self._monotonic = sleep, monotonic
        self._download_bytes = download_bytes
        self._killpg, self._getpgid = killpg, getpgid
        self._dl_seen = 0                #: highest in-flight download byte count seen
        self._dl_at = None               #: when the download last grew
        #: expected .safetensors bytes — the download total, when the caller looked it up
        self.weights_bytes = weights_bytes
        self.timeout = float(timeout)
        self._out = out if out is not None else sys.stderr
        self.proc = None                 #: set only when *we* started it
        self.log_path = Path(log_path) if log_path else default_log_path()
        self.verbose = verbose
        self.report_every = float(report_every)
        self._log_fh = None              #: the child writes here
        self._reader = None              #: we read the same file to report progress
        self._tail = deque(maxlen=40)
        self._last_output_at = None      #: when the child last said anything

    # ------------------------------------------------------------------ what was planned

    @property
    def resident_bytes(self) -> int:
        """Weight bytes that will actually occupy VRAM, after everything that leaves.

        The same subtraction the planners do (`:924`, `:1039`), exposed because the pre-flight
        fit check in `glq.fit` has to judge the footprint that will be served and a third copy
        of this formula is a third thing to keep in step.

        Reads `self.expert_offload_gib` as it stands *after* `plan_arena_and_offload` may have
        deepened it, so this is the final figure rather than the base one the window was first
        chosen from — which is the number a fit verdict needs. Only meaningful once `__init__`
        has finished, since `self.weights_bytes` is assigned near the end of it.
        """
        if not self.weights_bytes:
            return 0
        return max(0, int(self.weights_bytes) - self.ple_offload_bytes - self.nontext_bytes
                   - self.expert_offload_gib * 1024 ** 3)

    @property
    def vram_bytes(self) -> int | None:
        """The card the plan was sized against, or None when it could not be detected.

        Public because `glq.fit` has to judge against the same figure the plan used, and
        reaching into `_vram_bytes` from another module would make that a private contract.
        None is meaningful rather than missing: no nvidia-smi, a container started without
        `--gpus`, or an explicit `--gpu-memory-utilization`, which deliberately passes None
        because that flag already IS the answer sizing would have produced.
        """
        return self._vram_bytes

    @property
    def pinned_host_bytes(self) -> int:
        """Host memory the plan will pin: the PLE table plus the offloaded experts.

        Pinned, so it cannot be swapped out and cannot be reclaimed under pressure — which is
        why it is budgeted against total RAM rather than against what is momentarily free.
        """
        return self.ple_offload_bytes + self.expert_offload_gib * 1024 ** 3

    # ------------------------------------------------------------------ lifecycle

    def argv(self) -> list[str]:
        # --dtype is passed explicitly rather than left to vLLM's `auto`, which reads the
        # checkpoint's declared bf16. On CUDA that is the wrong default: the kernels compute
        # fp16, so bf16 pays a conversion at every quantized layer for nothing — fp16 is
        # +12.8% on an RTX PRO 6000 at IDENTICAL footprint, and wikitext PPL is marginally
        # better. On CPU it stays bf16, for reasons that are about footprint and vLLM's own
        # CPU kernels rather than alignment; `preferred_dtype` documents both, and returns
        # bf16 for the architectures where fp16 NaNs or is refused outright.
        # An explicit --dtype in extra_args still wins: vLLM takes the last occurrence.
        dtype = tooling.preferred_dtype(self.model, self.device)
        self._say_window_shortfall()
        if self.device == "cpu":
            # --gpu-memory-utilization means nothing to the CPU backend, and its
            # fullgraph compile cannot trace the fused CPU decode path — --enforce-eager
            # is a requirement, not a preference. The KV pool travels in the child env
            # (VLLM_CPU_KVCACHE_SPACE), not a flag.
            return [self.vllm_bin, "serve", self.model,
                    "--quantization", "glq",
                    "--dtype", dtype,
                    "--port", str(self.port),
                    "--enforce-eager",
                    "--max-model-len", str(self.max_model_len),
                    "--max-num-seqs", str(self.max_num_seqs),
                    *self.extra_args]
        return [self.vllm_bin, "serve", self.model,
                "--quantization", "glq",
                "--dtype", dtype,
                "--port", str(self.port),
                "--gpu-memory-utilization", _fraction_arg(self.gpu_memory_utilization),
                "--max-model-len", str(self.max_model_len),
                "--max-num-seqs", str(self.max_num_seqs),
                *self._offload_args(),
                *kv_compression.serve_args(self.fp8_kv),
                *self._speculative_args(),
                *self.extra_args]

    def _speculative_args(self) -> list[str]:
        """`--speculative-config` for MTP, or nothing.

        Before `extra_args` deliberately: vLLM takes the LAST occurrence, the same convention
        `--dtype` relies on above, so a caller passing their own `--speculative-config` still
        wins.

        Nothing on the CPU backend. Speculation is unmeasured there, and that path decodes in
        single digits where a wrong draft budget costs proportionally more; announcing a
        configuration nobody has run is how a plausible-looking number gets published.
        """
        if not self.mtp_tokens or self.device == "cpu":
            return []
        return ["--speculative-config",
                json.dumps({"method": "mtp", "num_speculative_tokens": self.mtp_tokens})]

    def _plan_offload(self, *, weights_bytes, vram_bytes, ram_bytes, max_model_len,
                      model_max_len, floor, extra_cap) -> int:
        """Expert offload sized for the served window as well as for the card.

        Two cases, and only the second is a policy question:

        * **A pinned `max_model_len`** is an instruction, not a trade — the user named the
          window, so it is priced in with no cap and no opt-in. This is the path that was
          broken: the budget ignored the pinned window, the pool was then sized to hold it and
          clamped at `_MAX_UTILIZATION`, and vLLM refused at startup.
        * **An auto window** means spending decode time on context nobody asked for, so it is
          bounded by `extra_cap` and off unless the caller opts in. See
          `WINDOW_OFFLOAD_MAX_EXTRA_GIB` for the measured cost.
        """
        def budget(kv_bytes):
            return plan_expert_offload_gib(
                weights_bytes=weights_bytes, ple_offload_bytes=self.ple_offload_bytes,
                expert_offload_bytes=self.expert_offload_bytes, vram_bytes=vram_bytes,
                ram_bytes=ram_bytes, nontext_bytes=self.nontext_bytes, kv_bytes=kv_bytes)

        base = budget(0)
        if max_model_len is not None:
            return max(base, budget(window_kv_bytes(max_model_len,
                                                    self.window_concurrency)))
        # `model_max_len` unknown means `plan_max_model_len` will return the floor whatever we
        # do here, so buying a tier would pay PCIe for a window that is never requested.
        if extra_cap <= 0 or not weights_bytes or not vram_bytes or not model_max_len:
            return base

        def resident_at(gib):
            return max(0, int(weights_bytes) - self.ple_offload_bytes
                       - self.nontext_bytes - gib * 1024**3)

        ceiling = min(int(model_max_len), _WINDOW_TIERS[-1])
        # Downward, so the answer is the SMALLEST offload reaching the BEST tier. Upward would
        # land on the declared maximum and pay for GiB that buy no further context: on the g7,
        # 32 and 38 both serve 262144, and the difference is ~3% of decode for nothing.
        for tier in sorted((t for t in _WINDOW_TIERS if t <= ceiling), reverse=True):
            if tier <= floor:
                break               # at or below the floor the window is already free
            kv = window_kv_bytes(tier, self.window_concurrency)
            cand = budget(kv)
            if cand > base + extra_cap:
                continue
            # Not redundant with the search inside `budget`: the declared-expert and
            # pinnable-RAM caps can clip `cand` BELOW what this tier needs, and accepting a
            # clipped value would announce a window the card cannot hold.
            if kv_headroom_bytes(resident_at(cand), vram_bytes) >= kv:
                return cand
        return base

    def _window_shortfall_bytes(self) -> int:
        """How far the plan falls short of holding the window it is about to serve, or 0.

        Only ever non-zero for a window that was PINNED — the auto path refuses a tier it
        cannot afford, rather than announcing one.
        """
        if self.device == "cpu" or not self._vram_bytes or not self.weights_bytes:
            return 0
        resident = max(0, int(self.weights_bytes) - self.ple_offload_bytes
                       - self.nontext_bytes - self.expert_offload_gib * 1024**3)
        need = window_kv_bytes(self.max_model_len, self.window_concurrency)
        return max(0, need - kv_headroom_bytes(resident, self._vram_bytes))

    def _say_window_shortfall(self) -> None:
        """Report a window the plan cannot pay for — without changing it.

        Deliberately a note and not a clamp. `_KV_BYTES_PER_TOKEN` is a conservative envelope
        over the worst pool ever observed, not a per-model figure, so a window this arithmetic
        calls unaffordable may well serve; vLLM's own startup check is the authoritative one.
        Clamping on our estimate would shorten windows that would have worked. Same contract
        as `--cpu-offload-gb 999`: honored verbatim, reported rather than silently adjusted.
        """
        short = self._window_shortfall_bytes()
        if not short:
            return
        # Is host RAM the binding constraint rather than the card? If so, "raise
        # --cpu-offload-gb" is advice the user cannot take: the pinnable cap already holds it
        # shut. Measured on a 23 GiB card, this checkpoint cannot serve at all on 60, 70, 80
        # or 90 GiB of host RAM, and serves a 131072 window from 100 GiB up — so the figure
        # that decides it is the host's, and naming the card instead sends the user nowhere.
        want = self._uncapped_offload_gib()
        if want > self.expert_offload_gib and self._ram_bytes:
            ram_gib = self._ram_bytes / 1024**3
            need_ram = ((want + math.ceil(self.ple_offload_bytes / 1024**3))
                        / _ARENA_PINNED_RAM_FRACTION)
            self._say(
                f"  note: this checkpoint needs about {want} GiB of experts in host RAM to "
                f"fit this card, but only {self.expert_offload_gib} GiB may be pinned on a "
                f"{ram_gib:.0f} GiB host (pinned pages cannot be swapped out, so the share is "
                f"capped). vLLM will likely refuse to start. Roughly {need_ram:.0f} GiB of "
                f"RAM would carry it; adding swap makes a larger pinned set safe; otherwise "
                f"serve a smaller checkpoint")
            return
        self._say(f"  note: --max-model-len {self.max_model_len} needs roughly "
                  f"{short / 1024**3:.2f} GiB more KV than this card can hold alongside the "
                  f"weights, so vLLM may refuse to start. Raise --cpu-offload-gb (now "
                  f"{self.expert_offload_gib}) to move more experts to host RAM, or ask for a "
                  f"shorter window")

    def _uncapped_offload_gib(self) -> int:
        """Expert offload the plan would choose if host RAM were unlimited.

        The difference against `expert_offload_gib` is exactly how far the pinnable cap is
        holding the plan back, which is what decides whether a shortfall is the card's fault
        or the host's.
        """
        if self._expert_offload_forced or not self.weights_bytes or not self._vram_bytes:
            return self.expert_offload_gib
        return plan_expert_offload_gib(
            weights_bytes=self.weights_bytes, ple_offload_bytes=self.ple_offload_bytes,
            expert_offload_bytes=self.expert_offload_bytes, vram_bytes=self._vram_bytes,
            ram_bytes=None, nontext_bytes=self.nontext_bytes,
            kv_bytes=window_kv_bytes(self.max_model_len, self.window_concurrency))

    def _offload_args(self) -> list[str]:
        """Expert-offload flags, or nothing at all when the model already fits.

        `uva`, never `prefetch`: prefetch bulk-copies whole layer groups, so it transfers
        experts that were never routed -- measured 2.4x slower at a SMALLER offload volume, and
        it fails outright on GLQ (`CPU storage for ...trellis_packed is not pinned!`).

        The filter is `--cpu-offload-params` because that is the UVA backend's field;
        `--offload-params` belongs to prefetch and would be silently ignored here.
        """
        if self.expert_offload_gib <= 0:
            return []
        declared = self.expert_offload_bytes // 1024**3
        if self._expert_offload_forced and declared and self.expert_offload_gib > declared:
            # The flag still wins, but vLLM can only offload what exists, so the resident
            # prediction the window was sized from is wrong in the dangerous direction.
            self._say(f"  note: asked to offload {self.expert_offload_gib} GiB of experts "
                      f"but this checkpoint declares {declared} GiB; vLLM will offload less "
                      f"than that and the footprint will exceed the plan")
        return ["--offload-backend", "uva",
                "--cpu-offload-gb", str(self.expert_offload_gib),
                "--cpu-offload-params", "experts"]

    def start(self) -> bool:
        """True if we started a server, False if we attached to a running one.

        Attaching matters twice over: a second vLLM on the same port would fail anyway, and
        reusing a warm server is the only thing that makes a repeat `glq-chat` instant.
        """
        if self._probe(self.base_url):
            self._say(f"attaching to the server already on {self.base_url}")
            return False
        if not self.serve:
            raise RuntimeError(
                f"no server at {self.base_url} and --no-serve was given. Start one with:\n"
                f"  {' '.join(self.argv())}")

        self._say(f"starting vLLM for {self.model}")
        if self.device == "cpu":
            self._say(f"  serving on the CPU backend — expect single-digit tokens/s; "
                      f"{self.max_model_len} token context")
            if self._ignored_gpu_flag:
                self._say("  (--gpu-memory-utilization has no effect on the CPU backend "
                          "— ignoring it)")
        else:
            self._say(f"  reserving {self.gpu_memory_utilization:.0%} of VRAM for the weights "
                      f"and KV cache, with a {self.max_model_len} token context"
                      f"{self._window_note}")
        if self.expert_arena_gib > 0:
            # Say it out loud. A cache that is silently on or off produces correct output
            # either way, so the only way a user can tell is if we tell them — the same
            # reason `arena_decision` carries its reason instead of logging and dropping it.
            self._say(f"  keeping the hottest experts in {self.expert_arena_gib} GiB of VRAM "
                      f"(routing-aware cache; measured 2.7x decode on an L4). "
                      f"GLQ_MOE_EXPERT_CACHE=0 turns it off")
        elif self.expert_offload_gib > 0 and self.device != "cpu":
            self._say(f"  no VRAM left for the expert cache after a {self.max_model_len} "
                      f"token context — a shorter window or more host RAM would buy one "
                      f"(GLQ_MOE_EXPERT_CACHE=<GiB> forces it)")
        if self.mtp_tokens and self.device != "cpu":
            # Say it out loud, for the same reason as the arena line above — but with one
            # extra thing to disclose. Speculative decoding is often described as lossless at
            # temperature 0, and here it is not: three L4 speculative arms produced three
            # distinct digests while EVERY non-speculative arm agreed, so this is MTP's
            # property rather than the arena's. A user comparing output between runs should
            # know that before concluding something broke.
            self._say(f"  speculating {self.mtp_tokens} draft token"
                      f"{'' if self.mtp_tokens == 1 else 's'} per step with the MTP head "
                      f"(--mtp 0 turns it off). Output is not byte-identical with "
                      f"speculation on, even at temperature 0")
            if self._nontext_declared:
                self._say(f"  the draft head costs "
                          f"{self._nontext_declared / 2**30:.2f} GiB of VRAM that a "
                          f"text-only serve would not load")
        # Say how long this takes *before* going quiet. Minutes of silence you were warned
        # about is patience; the same silence unannounced is indistinguishable from a hang,
        # and that is what a first run looks like today.
        size = (f" (~{self.weights_bytes / 2**30:.1f} GiB)"
                if self.weights_bytes else "")
        self._say(f"  the first run downloads the weights{size} and loads the model — "
                  f"expect a few minutes")

        # A file, not a pipe. A pipe holds ~64 KiB and then *blocks the writer*, so a child
        # this chatty could wedge on its own logging with nobody reading. It also gives the
        # user something to tail, and us something to quote when startup fails.
        log = self._open_log()
        self._say(f"  vLLM log: {self.log_path}")

        # PYTHONUNBUFFERED: without it the child block-buffers into the file, so the lines
        # explaining a failure arrive long after we have given up — or never, if it dies
        # with them unflushed.
        if self.fp8_kv:
            self._say("  KV cache in fp8 (vLLM's own) — about twice the context per GiB")
        env = child_env(device=self.device, ram_bytes=self._ram_bytes,
                        ple_offload=bool(self.ple_offload_bytes),
                        weights_bytes=self.weights_bytes,
                        available_bytes=self._available_bytes,
                        expert_arena_gib=self.expert_arena_gib,
                        max_num_seqs=self.step_tokens,
                        min_step_tokens=self.min_step_tokens)
        if self.device == "cpu" and self._ram_bytes and self.weights_bytes:
            # Say the arithmetic out loud. On CPU the weights, the pool, the runtime and the
            # page cache share one pool of RAM, and when it does not fit there is no swap to
            # absorb it on a default cloud box — the machine stops answering rather than
            # OOM-killing one process, which is a much harder failure to read after the fact.
            pool = int(env["VLLM_CPU_KVCACHE_SPACE"])
            anon = self.weights_bytes + pool * 2**30 + _CPU_RUNTIME_OVERHEAD_BYTES
            budget = (f"{self._available_bytes / 2**30:.1f} GiB free"
                      if self._available_bytes
                      else f"{self._ram_bytes / 2**30:.1f} GiB total")
            self._say(f"  RAM plan: {self.weights_bytes / 2**30:.1f} GiB weights + "
                      f"{pool} GiB KV pool + ~{_CPU_RUNTIME_OVERHEAD_BYTES / 2**30:.0f} GiB "
                      f"runtime = {anon / 2**30:.1f} GiB, against {budget}")
            ceiling = _CPU_ANON_FRACTION * self._ram_bytes
            if self._available_bytes:
                ceiling = min(ceiling, self._available_bytes - _CPU_HEADROOM_BYTES)
            if anon > ceiling:
                self._say(f"  warning: that is {(anon - ceiling) / 2**30:.1f} GiB more than "
                          f"this machine has room for. Close what else is running, serve a "
                          f"smaller checkpoint, or set VLLM_CPU_KVCACHE_SPACE lower — on CPU "
                          f"the weights, the pool and the page cache share one pool of RAM, "
                          f"and a machine with no swap thrashes rather than failing cleanly.")
        if "VLLM_USE_FLASHINFER_SAMPLER" in env:
            self._say("  the NVIDIA CUDA Toolkit is not installed, and FlashInfer ships no "
                      "prebuilt sampler for this GPU, so it")
            self._say("  cannot build one — falling back to vLLM's built-in sampler. "
                      "Please install the CUDA Toolkit for the faster path:")
            self._say("  https://developer.nvidia.com/cuda-downloads")
        # Its own session: the engine core is a spawned grandchild, and the group is what
        # holds the VRAM — stop() signals the group, which only works if we lead one.
        self.proc = self._spawn(self.argv(), stdout=log,
                                stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        try:
            self._wait_until_ready()
        except BaseException:
            self.stop()                  # never leave a half-started server holding the card
            raise
        return True

    def _open_log(self):
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self.log_path, "w", buffering=1)
        except OSError:                  # unwritable GLQ_HOME: a log is not worth failing over
            import tempfile
            self.log_path = Path(tempfile.mkstemp(prefix="glq-vllm-", suffix=".log")[1])
            self._log_fh = open(self.log_path, "w", buffering=1)
        return self._log_fh

    def stop(self) -> None:
        """End the child's whole process group, escalating to SIGKILL. A server we merely
        attached to is left alone — we did not start it, so it is not ours to end.

        The group, not just the child: vLLM's engine core is a spawned grandchild, and a
        SIGTERM to the API server alone left an EngineCore parked in a weight download
        holding 17 GiB — while this method printed "the GPU is free again". The group is
        the unit that holds VRAM, so the group is the unit stop() ends.
        """
        proc, self.proc = self.proc, None
        if proc is None:
            return
        self._say("stopping vLLM — the GPU is free again")
        self._signal_tree(proc, hard=False)
        try:
            proc.wait(timeout=20)
        except Exception:                # noqa: BLE001 - TimeoutExpired, or a fake's stand-in
            self._signal_tree(proc, hard=True)
            try:
                proc.wait(timeout=10)
            except Exception:            # noqa: BLE001 - nothing further we can do
                pass

    def _signal_tree(self, proc, *, hard: bool) -> None:
        """Signal the child's process group if it leads one; fall back to the child alone.

        The leadership check keeps this safe for anything not spawned by start() with its
        own session — never signal a group we do not own.
        """
        sig = signal.SIGKILL if hard else signal.SIGTERM
        try:
            if self._getpgid(proc.pid) == proc.pid:
                self._killpg(proc.pid, sig)
                return
        except Exception:                # noqa: BLE001 - no such process, or a fake's stand-in
            pass
        proc.kill() if hard else proc.terminate()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_exc):
        self.stop()
        return False

    # ------------------------------------------------------------------ internals

    def _say(self, msg: str) -> None:
        print(f"  {msg}", file=self._out, flush=True)

    def _wait_until_ready(self) -> None:
        """Poll until the endpoint answers, the child dies, or we run out of patience.

        Reports progress in vLLM's own words while it waits. Its log says what it is doing —
        downloading shards, loading weights, capturing CUDA graphs — which is both more
        accurate and more reassuring than anything we could invent, and the difference
        between a terminal that looks busy and one that looks hung.
        """
        started = self._monotonic()
        last_progress = started
        last_report = started
        dl_baseline = self._safe_download_bytes()
        while True:
            rc = self.proc.poll()
            self._collect()
            if rc is not None:
                raise RuntimeError(
                    f"vLLM exited with status {rc} before it was ready.\n" + self._drain())
            if self._probe(self.base_url):
                self._say(f"ready in {self._monotonic() - started:.0f}s — {self.base_url}")
                return
            now = self._monotonic()
            # The timeout is a NO-PROGRESS window, not a stopwatch. A first run downloads
            # the weights, and a download writes nothing to the log — measured on a 26B:
            # 15 minutes parked in snapshot_download, perfectly healthy, and a fixed
            # deadline shot it. Log output and download growth both count as progress; a
            # true hang still ends after `timeout` seconds of neither.
            if self._last_output_at is not None:
                last_progress = max(last_progress, self._last_output_at)
            dl = self._safe_download_bytes()
            if dl > dl_baseline:
                dl_baseline = dl
                self._dl_seen = dl
                self._dl_at = now
                last_progress = now
            if now - last_progress >= self.timeout:
                hint = ""
                if any("unauthenticated requests" in ln for ln in self._tail):
                    hint = ("\nThe weight download is running unauthenticated, which HF "
                            "rate-limits hard — set HF_TOKEN in the environment and retry.")
                raise RuntimeError(
                    f"vLLM made no progress for {self.timeout:.0f}s — no log output and "
                    f"no weight-download movement.\n" + self._drain() + hint)
            if now - last_report >= self.report_every:
                self._report(now - started)
                last_report = now
            self._sleep(2.0)

    def _safe_download_bytes(self) -> int:
        try:
            return int(self._download_bytes())
        except Exception:                # noqa: BLE001 - a broken probe must not kill the wait
            return 0

    @staticmethod
    def _informative(line: str) -> bool:
        """Is this line worth showing as "what vLLM is doing"?

        Python prints warnings over two lines — the message, then the offending source line —
        and the source line lands last, so a naive "latest line" reports
        `warnings.warn('resource_tracker: ...'` and keeps reporting it while the real work
        goes on silently. Measured on an L4: ten identical progress lines over a minute.
        """
        s = line.strip()
        return bool(s) and not s.startswith(("warnings.warn", "warnings.simplefilter",
                                             "self._", "return ", "File \""))

    def _collect(self) -> None:
        """Pull whatever the child has written since last time into the tail buffer."""
        if self._reader is None:
            try:
                self._reader = open(self.log_path, "r", errors="replace")
            except OSError:
                return
        try:
            fresh = self._reader.read()
        except OSError:                  # noqa: BLE001 - nothing to add
            return
        for line in fresh.splitlines():
            if line.strip():
                self._tail.append(line)
                self._last_output_at = self._monotonic()
                if self.verbose:
                    print(f"  | {line}", file=self._out, flush=True)

    def _report(self, elapsed: float) -> None:
        """One line: how long we have been waiting, and the child's latest word on it."""
        latest = next((ln for ln in reversed(self._tail) if self._informative(ln)), None)
        latest = (latest or "waiting for vLLM to come up").strip()
        if len(latest) > 96:
            latest = latest[:93] + "..."

        # Distinguish "still working" from "has said nothing for a while". Without this the
        # same line reappears on a timer and the two look identical.
        quiet = ""
        if self._last_output_at is not None:
            since = self._monotonic() - self._last_output_at
            if since >= 30:
                quiet = f"  (no new output for {since:.0f}s)"
        # A silent log with a growing cache is the most common healthy state of a first
        # run — say what is actually happening instead of quoting a stale line. But the
        # download owns the line only while it is the FRESHEST signal: measured, the old
        # unconditional label froze at "17.8 GiB so far" through the whole load phase.
        if self._dl_seen and (self._last_output_at is None
                              or (self._dl_at or 0) >= self._last_output_at):
            if self.weights_bytes and self._dl_seen >= 0.98 * self.weights_bytes:
                latest = (f"weights downloaded "
                          f"({self.weights_bytes / 2**30:.1f} GiB) — the engine is "
                          f"loading them")
            elif self.weights_bytes:
                latest = (f"downloading weights: {self._dl_seen / 2**30:.1f} / "
                          f"{self.weights_bytes / 2**30:.1f} GiB")
            else:
                latest = f"downloading weights: {self._dl_seen / 2**30:.1f} GiB so far"
            quiet = ""
        self._say(f"[{elapsed:4.0f}s] {latest}{quiet}")

    def _drain(self, keep: int = 25) -> str:
        """The child's last lines, plus where to read the rest — the reason a startup failed
        is often further up than any tail we would print."""
        self._collect()
        tail = "\n".join(list(self._tail)[-keep:]) or "(vLLM produced no output)"

        # vLLM names the flag but not a number, and the user cannot see what we computed.
        # This is the one startup failure with a one-line fix, so spell it out.
        hint = ""
        if "No available memory for the cache blocks" in tail:
            if self.device == "cpu":
                hint = ("\n\nThe model plus its KV pool did not fit in RAM. Retry with a "
                        "smaller pool, e.g.\n"
                        "  VLLM_CPU_KVCACHE_SPACE=2 glq-chat\n"
                        "or pick a smaller checkpoint.")
            else:
                hint = (f"\n\nThe model did not fit in {self.gpu_memory_utilization:.0%} of this "
                        f"GPU. Retry with a larger share, e.g.\n"
                        f"  glq-chat --gpu-memory-utilization "
                        f"{min(self.gpu_memory_utilization + 0.2, 0.95):.2f}")
        return f"{tail}{hint}\n\nFull log: {self.log_path}"
