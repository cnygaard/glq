"""The pre-flight fit check: can this checkpoint actually run on this machine?

Written against a real incident. `glq-code --model xv0y5ncu/Kolibri-1-GLQ-trellis-3inst-3bpw`
on an RTX PRO 4500 Blackwell (31.86 GiB VRAM, 30 GiB host RAM, **no swap**) failed with a CUDA
out-of-memory several minutes into the weight load, with nothing pointing at the cause.

Measured with the real planners, the surprise is that **VRAM is not the binding constraint**:

    weights                     28.70 GiB   (correctly detected)
    offloadable experts         26.77 GiB   (correctly detected)
    kv_headroom fully resident  -4.25 GiB   impossible
    after the planned offload   +14.29 GiB  fits comfortably
    pinned host RAM             18 GiB of 30, leaving 12 unpinned, with no swap

So the check has to judge **host RAM as a first-class axis**, not only the card — and it has
to say which one binds, because the remedies are opposite: more offload relieves a VRAM
shortfall and *worsens* a host one, since pinned pages cannot be swapped out.

The gate is deliberately the WEAKEST defensible one — the floor window plus `_MIN_KV_BYTES`,
never the chosen window. Disk bytes over-state resident bytes (`recommend`'s own docstring
records the 31B at 22.4 GiB on disk against ~16.5 GiB resident), and
`supervisor._say_window_shortfall` already documents at length why an estimate-based *clamp*
would shorten windows that would have served. A refusal is a statement that the card cannot
hold the weights, the runtime overhead and a 2 GiB cache — not a prediction about a window.

The two host data points are both swapless, and one of them is a **pass**: a run that pinned
65.3 GiB on a 121 GiB box kept ~13 GiB available at peak and served. That run is the failing
control here. Without it a host gate tuned to the failure alone would refuse working setups.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq import fit  # noqa: E402
from glq.supervisor import VllmSupervisor, kv_headroom_bytes  # noqa: E402

GIB = 2 ** 30
CODE_FLOOR = 16384          # glq.code.DEFAULT_CODE_MAX_MODEL_LEN

#: Kolibri-1 at 3 bpw, as the quantizer declares it. 96.7% of its params are experts, so
#: almost the whole checkpoint is offloadable — which is what makes the VRAM side pass.
KOLIBRI = dict(weights_bytes=int(28.70 * GIB), ple_offload_bytes=0,
               expert_offload_bytes=int(26.77 * GIB), model_max_len=262144)

#: A dense checkpoint declaring nothing offloadable — the case where the card really is the
#: problem and no amount of host memory can help.
DENSE = dict(weights_bytes=int(28.70 * GIB), ple_offload_bytes=0,
             expert_offload_bytes=0, model_max_len=262144)


def _sup(spec=None, *, vram_gib=31.86, ram_gib=30.0, **kw):
    """A supervisor is pure planning — no process, no files — so it can be built in a unit
    test and the plan read straight off it. That is the point: a verdict derived from the
    same object that will be served cannot disagree with what gets served."""
    base = dict(model="org/ckpt", device="cuda", max_model_len_floor=CODE_FLOOR,
                window_concurrency=1, vram_bytes=int(vram_gib * GIB),
                ram_bytes=int(ram_gib * GIB))
    # `kw` last, so a test can override any of it — including to None, which is how the
    # "never refuse on an unknown" cases are expressed.
    return VllmSupervisor(**{**base, **(spec if spec is not None else KOLIBRI), **kw})


def _check(sup, *, ram_gib=30.0, swap_gib=0.0, **kw):
    return fit.check_fit(sup, ram_bytes=int(ram_gib * GIB),
                         swap_bytes=int(swap_gib * GIB), floor=CODE_FLOOR, **kw)


# ---- the incident, and its failing control ---------------------------------------------

def test_the_incident_is_refused():
    """The whole reason this module exists: 18 GiB pinned on a 30 GiB swapless box."""
    r = _check(_sup())
    assert r.fits is False
    assert r.reason == "host", f"host RAM binds here, not the card — got {r.reason!r}"


def test_the_incident_names_host_ram_not_the_card():
    """Blaming the card would send the user shopping for a GPU that fixes nothing."""
    r = _check(_sup())
    assert r.vram_short_bytes == 0, "the card is NOT short once the offload is planned"
    assert r.host_short_bytes > 0


def test_vram_is_genuinely_fine_after_the_planned_offload():
    """Pins the measured +14.29 GiB, so a future change that makes this refuse on VRAM
    grounds has to confront the measurement rather than the test."""
    s = _sup()
    assert s.expert_offload_gib == 18
    head = kv_headroom_bytes(s.resident_bytes, 31.86 * GIB)
    assert head > 14 * GIB


def test_a_121gib_host_is_not_refused():
    """THE FAILING CONTROL. A run that pinned 65.3 GiB of 121 served fine and kept ~13 GiB
    available at peak. A host gate tuned only to the failure would refuse it, and refusing a
    configuration that demonstrably works is worse than not checking at all."""
    r = _check(_sup(vram_gib=95.0, ram_gib=121.0), ram_gib=121.0)
    assert r.fits is not False, f"refused a known-good configuration: {r.reason!r}"


def test_swap_downgrades_a_host_refusal():
    """Swap absorbs the non-pinned overshoot, and pinned pages are still capped by the
    planner's own fraction, so the failure mode this gate predicts cannot occur."""
    assert _check(_sup(), swap_gib=64.0).fits is not False


# ---- the card, when the card really is the problem -------------------------------------

def test_a_dense_checkpoint_too_big_for_the_card_is_refused_on_vram():
    """28.70 GiB resident on a 31.86 GiB card leaves -4.25 GiB of headroom. Nothing is
    offloadable, so this is the card's fault and the message must say so."""
    r = _check(_sup(DENSE))
    assert r.fits is False
    assert r.reason == "vram"
    assert r.vram_short_bytes > 0


def test_host_is_not_blamed_when_nothing_is_pinned():
    """`pinned == 0` must short-circuit the host gate, or every dense model on every box
    with less than the unpinned floor would be refused for a reason that cannot apply."""
    r = _check(_sup(DENSE, vram_gib=95.0), ram_gib=8.0)
    assert r.pinned_bytes == 0
    assert r.reason != "host"


def test_a_comfortable_box_is_not_refused():
    r = _check(_sup(KOLIBRI, vram_gib=95.0, ram_gib=256.0), ram_gib=256.0)
    assert r.fits is True


# ---- never refuse on an unknown -------------------------------------------------------

def test_unknown_weights_never_refuse():
    """`_checkpoint_bytes` returns None offline. The check must never be the thing that
    stops a setup that works today."""
    assert _check(_sup(KOLIBRI, weights_bytes=None)).fits is None


def test_unknown_vram_never_refuses():
    """No nvidia-smi, or a container started without --gpus. Also what
    `--gpu-memory-utilization` deliberately produces, since that flag IS the answer."""
    assert _check(_sup(KOLIBRI, vram_bytes=None)).fits is None


def test_unknown_host_ram_never_refuses():
    assert fit.check_fit(_sup(), ram_bytes=None, swap_bytes=None,
                         floor=CODE_FLOOR).fits is None


def test_the_cpu_backend_is_not_judged_here():
    """CPU serving budgets weights and KV from one pool of RAM; `supervisor.start()` already
    warns there, with different arithmetic. Judging it with VRAM rules would be nonsense."""
    s = VllmSupervisor(model="org/ckpt", device="cpu", vram_bytes=None,
                       ram_bytes=30 * GIB, **KOLIBRI)
    assert _check(s).fits is None


# ---- the message ----------------------------------------------------------------------

def test_a_host_refusal_never_advises_raising_cpu_offload_gb():
    """The advice a user cannot take. The pinnable cap is already holding it shut, and
    pinned pages cannot be swapped out, so more offload makes a host shortfall worse."""
    lines = "\n".join(fit.refusal_lines(_check(_sup()), command="glq-code"))
    assert "--cpu-offload-gb" not in lines


def test_a_refusal_states_the_shortfall_and_the_override():
    lines = "\n".join(fit.refusal_lines(_check(_sup()), command="glq-code"))
    assert "--force" in lines
    assert "30" in lines and "18" in lines, "the pinned and total host figures"


def test_a_refusal_offers_the_alternatives_it_was_given():
    r = _check(_sup())
    lines = "\n".join(fit.refusal_lines(r, alternatives=["org/small-one"],
                                        command="glq-code"))
    assert "org/small-one" in lines


def test_no_alternatives_still_produces_a_usable_message():
    """An empty list must not render an empty 'these fit:' heading."""
    lines = "\n".join(fit.refusal_lines(_check(_sup()), alternatives=(),
                                        command="glq-code"))
    assert lines.strip()
    assert "these fit" not in lines.lower()


def test_refusal_lines_refuses_to_render_a_passing_report():
    """Rendering a refusal for something that fits would be a caller bug, and a silent
    empty string would hide it."""
    with pytest.raises(ValueError):
        fit.refusal_lines(_check(_sup(KOLIBRI, vram_gib=95.0, ram_gib=256.0),
                                 ram_gib=256.0))


# ---- the host probe ------------------------------------------------------------------

def test_swap_bytes_reads_swaptotal():
    from glq.installer import hardware
    meminfo = "MemTotal:       31430000 kB\nSwapTotal:      67108864 kB\n"
    assert hardware.swap_bytes(read=lambda: meminfo) == 67108864 * 1024


def test_swap_bytes_is_zero_when_there_is_no_swap():
    """A swapless box reports SwapTotal: 0 — that is a known 0, not an unknown."""
    from glq.installer import hardware
    assert hardware.swap_bytes(read=lambda: "SwapTotal:             0 kB\n") == 0


def test_swap_bytes_is_none_when_unreadable():
    from glq.installer import hardware

    def boom():
        raise OSError("no /proc")

    assert hardware.swap_bytes(read=boom) is None
    assert hardware.swap_bytes(read=lambda: "MemTotal: 1 kB\n") is None


# ---- ranking a collection with the real math -------------------------------------------
#
# `recommend.rank` gates on `resident_floor <= 0.75 * vram` — a deliberately conservative
# DOWNLOAD decision, with no KV, no arena and no host RAM in it. That is the right question
# for the installer and the wrong one for a picker about to serve, which is why these are two
# functions rather than one. `fit_ranked` mirrors rank's *ordering and recommendation policy*
# exactly (largest first, trellis preferred, family preference honoured only when something
# in that family fits) so the menu and `glq-setup --list` cannot disagree about rank order
# while disagreeing about what fits.

def _ck(repo, gib, *, experts_gib=0.0, trellis=True):
    from glq.installer.discovery import Checkpoint
    return Checkpoint(repo, int(gib * GIB), trellis, experts_gib > 0,
                      0, int(experts_gib * GIB), 0)


def test_fit_ranked_is_largest_first():
    got = fit.fit_ranked([_ck("org/small", 2), _ck("org/big", 20), _ck("org/mid", 9)],
                         vram_bytes=int(31.86 * GIB), ram_bytes=256 * GIB, device="cuda")
    assert [r.checkpoint.short_name for r in got] == ["big", "mid", "small"]


def test_fit_ranked_recommends_the_largest_that_fits():
    got = fit.fit_ranked([_ck("org/small", 2), _ck("org/huge", 400), _ck("org/mid", 9)],
                         vram_bytes=int(31.86 * GIB), ram_bytes=256 * GIB, device="cuda")
    rec = [r.checkpoint.short_name for r in got if r.recommended]
    assert rec == ["mid"], got


def test_fit_ranked_carries_the_host_reason_into_the_label():
    """The whole point of threading `reason` through: a checkpoint the CARD could hold but
    the HOST cannot must not be rendered as too large for the GPU."""
    [r] = fit.fit_ranked([_ck("org/kolibri", 28.70, experts_gib=26.77)],
                         vram_bytes=int(31.86 * GIB), ram_bytes=30 * GIB, swap_bytes=0, device="cuda")
    assert r.fits is False
    assert r.reason == "host", r


def test_fit_ranked_disagrees_with_recommend_rank_on_the_incident():
    """Pins that the two gates are allowed to differ, so nobody collapses one into the
    other: rank's 0.75-of-VRAM heuristic says this checkpoint is downloadable (its resident
    floor after offload is tiny), while the serve-time check refuses it on host memory."""
    from glq.installer.recommend import rank
    ck = _ck("org/kolibri", 28.70, experts_gib=26.77)
    [plain] = rank([ck], int(31.86 * GIB), ram_bytes=30 * GIB)
    [strict] = fit.fit_ranked([ck], vram_bytes=int(31.86 * GIB), ram_bytes=30 * GIB,
                              swap_bytes=0, device="cuda")
    assert plain.fits is True and strict.fits is False


def test_fit_ranked_honours_a_family_preference_only_when_it_fits():
    cks = [_ck("org/qwen-small", 2), _ck("org/gemma-mid", 9)]
    rec = [r.checkpoint.short_name for r in
           fit.fit_ranked(cks, vram_bytes=int(31.86 * GIB), ram_bytes=256 * GIB,
                          prefer_family="qwen", device="cuda") if r.recommended]
    assert rec == ["qwen-small"]
    # nothing in the family fits -> the preference is ignored rather than recommending
    # something that cannot serve
    rec2 = [r.checkpoint.short_name for r in
            fit.fit_ranked([_ck("org/qwen-huge", 400), _ck("org/gemma-mid", 9)],
                           vram_bytes=int(31.86 * GIB), ram_bytes=256 * GIB,
                           prefer_family="qwen", device="cuda") if r.recommended]
    assert rec2 == ["gemma-mid"]


def test_fit_ranked_with_no_detectable_gpu_recommends_nothing():
    got = fit.fit_ranked([_ck("org/a", 2)], vram_bytes=None, ram_bytes=256 * GIB, device="cuda")
    assert [r.fits for r in got] == [None]
    assert not any(r.recommended for r in got)


def test_fit_ranked_resolves_the_device_once_not_once_per_checkpoint(monkeypatch):
    """Each supervisor construction would otherwise call `detect_device`, which reads wheel
    metadata and can shell out to nvidia-smi — N times, on every bare `glq-code`. Invisible
    when wrong except as a slow start, so it gets asserted."""
    from glq import supervisor as S
    calls = []
    monkeypatch.setattr(S, "detect_device", lambda *a, **k: calls.append(1) or "cuda")
    # NO device= here, deliberately. Passing it would skip the call this test exists to
    # count, and the assertion would then hold for a function that probed six times.
    got = fit.fit_ranked([_ck(f"org/c{i}", 2 + i) for i in range(6)],
                         vram_bytes=int(31.86 * GIB), ram_bytes=256 * GIB)
    assert len(got) == 6, "the ranking must still have happened"
    assert calls, "the probe must have been reached at all — else this proves nothing"
    assert len(calls) == 1, f"detect_device called {len(calls)}x for 6 checkpoints"


# ---- a forced offload budget is not the same as pinned bytes ---------------------------
#
# Found from a real report: a user on a 94.27 GiB host served Flash-Next with
# `--cpu-offload-gb 60`. The checkpoint declares 42.63 GiB of offloadable experts, so vLLM
# cannot pin 60 — there is not that much to pin. Counting the REQUEST rather than what is
# achievable over-stated pinned memory by 17.37 GiB and flipped the verdict to a refusal, and
# over-counting is the dangerous direction: it refuses a configuration that works.

def test_a_forced_budget_is_capped_by_the_declared_expert_bytes():
    s = _sup(KOLIBRI, ram_gib=120.0, expert_offload_gib=80)
    # 80 GiB requested, 26.77 GiB of experts exist -> at most 26.77 can be pinned
    assert s.pinned_host_bytes <= KOLIBRI["expert_offload_bytes"] + 0


def test_an_over_large_budget_does_not_manufacture_a_host_refusal():
    """The real-report case, scaled down: the request exceeds what the checkpoint has, so
    the host gate must judge the achievable figure."""
    r = _check(_sup(KOLIBRI, ram_gib=60.0, expert_offload_gib=200), ram_gib=60.0)
    assert r.pinned_bytes <= KOLIBRI["expert_offload_bytes"]
    assert r.fits is not False, f"refused on a budget that cannot be spent: {r.reason}"


def test_an_undeclared_offload_still_trusts_the_request():
    """A checkpoint declaring no offloadable experts has no figure to clamp to, so a forced
    budget must be taken at face value rather than silently becoming zero."""
    s = _sup(DENSE, ram_gib=120.0, expert_offload_gib=20)
    assert s.pinned_host_bytes == 20 * GIB


# ---- a family's MEASURED runtime overhead: Arete-27B 4 bpw on a 24 GB card -------------
#
# Measured 2026-11-19 on vLLM 0.33.0, an RTX PRO 4500 (31.86 GiB) and the same card capped to
# 24,564 MiB, an RTX 4090's nvidia-smi total; `benchmarks/arete_24gb/README.md`. On 0.8.37 both
# glq-code and glq-chat refused Arete 4 bpw there, "SHORT by 1.42 GiB"; forced, both served,
# glq-code at 16,384 with 2.60 GiB of KV against the ~1.47 that window needs, 28.0 tok/s.
#
# vLLM's breakdown, 4 bpw: weights 16.58 GiB, activation peak 2.50 (2.62 with 16 sequences),
# non-torch 0.10, CUDA graphs 0.04-0.08. The envelope reserves 4 GiB + 3% of the resident for
# all of that, 4.51 GiB, beside a resident counted at the file size, 16.98.

ARETE_ID = "xv0y5ncu/Arete-27B-GLQ-trellis-3inst-4bpw"
#: The Hub's safetensors bytes, which is what glq-code and glq-chat count as weights.
ARETE_FILE = {"4": 18_236_907_824, "5": 21_277_885_240, "6": 24_318_755_704}
ARETE4 = dict(weights_bytes=ARETE_FILE["4"], ple_offload_bytes=0, expert_offload_bytes=0,
              model_max_len=262144)
CHAT_FLOOR = 8192           # glq.supervisor.DEFAULT_MAX_MODEL_LEN

#: nvidia-smi `memory.total` of the 24 GB desktop cards: RTX 4090 / RTX A5000, RTX 3090.
_24GB_CARDS_MIB = (24564, 24576)

#: Every served run's overhead AS THE PLANNER COUNTS IT: the budget planned (utilization x the
#: card's nvidia-smi GiB), less the resident figure (the file), less the KV pool vLLM reported.
#: That one number absorbs everything the planner does not model separately — activations,
#: graphs, non-torch allocations, the gap between nvidia-smi's total and the CUDA total vLLM
#: budgets against (31.86 vs 31.43 GiB here), and the vision tower that is on disk but not
#: loaded under `--language-model-only`.
#:
#:     (launcher, card GiB, bpw, window): (planned utilization, KV GiB vLLM reported)
_ARETE_RUNS = {
    ("chat", 23.99, "4", 8192): (0.92, 2.47),        # 0.8.37 --force
    ("code", 23.99, "4", 16384): (0.92, 2.60),       # 0.8.37 --force
    ("code", 31.86, "4", 65536): (0.8374287, 7.15),  # 0.8.37
    ("chat", 23.99, "4", 16384): (0.9164003, 2.38),  # this change, from here down
    ("code", 23.99, "4", 16384): (0.9164003, 2.52),
    ("chat", 31.86, "5", 65536): (0.8789415, 5.18),
    ("code", 31.86, "5", 65536): (0.8789415, 5.31),
    ("chat", 31.86, "6", 32768): (0.8956405, 2.95),
    ("code", 31.86, "6", 32768): (0.8956405, 3.08),
}


def _measured_overhead_gib(run):
    (_, vram, bpw, _), (util, kv) = run
    return util * vram - ARETE_FILE[bpw] / GIB - kv


def _arete(vram_gib, **kw):
    return _sup(ARETE4, model=ARETE_ID, vram_gib=vram_gib, ram_gib=32.0, **kw)


@pytest.mark.parametrize("mib", _24GB_CARDS_MIB)
def test_arete_4bpw_fits_a_24gb_card_for_glq_code(mib):
    s = _arete(mib / 1024)
    # The mechanism first: without it a pass could come from anything that loosened the gate.
    assert s.runtime_overhead is not None, "the family's measured overhead did not engage"
    r = _check(s, ram_gib=32.0)
    assert r.fits is True, f"SHORT by {r.vram_short_bytes / GIB:.2f} GiB"


@pytest.mark.parametrize("mib", _24GB_CARDS_MIB)
def test_arete_4bpw_fits_a_24gb_card_for_glq_chat(mib):
    s = _arete(mib / 1024, max_model_len_floor=CHAT_FLOOR)
    r = fit.check_fit(s, ram_bytes=32 * GIB, swap_bytes=0, floor=CHAT_FLOOR)
    assert r.fits is True, f"SHORT by {r.vram_short_bytes / GIB:.2f} GiB"


def test_arete_4bpw_is_still_refused_on_an_l4():
    """22.49 GiB. vLLM itself would refuse 16,384 there — about 20.3 GiB of budget, less
    16.68 of weights and 2.54 of activations and graphs, leaves ~1.1 GiB of KV against ~1.47 —
    so the gate must still say no (measured on the box: "SHORT by 1.29 GiB"). A measured
    overhead narrows the gate; it does not open it."""
    r = _check(_arete(22.49), ram_gib=32.0)
    assert r.fits is False
    assert r.reason == "vram"


def test_an_unmeasured_family_of_the_same_size_keeps_the_envelope():
    """Same bytes, unknown family: nothing about this model was measured, so nothing moves —
    neither the overhead nor the 2 GiB KV floor."""
    from glq.supervisor import _MIN_KV_BYTES
    s = _sup(ARETE4, model="org/ckpt", vram_gib=24564 / 1024, ram_gib=32.0)
    assert s.runtime_overhead is None
    r = _check(s, ram_gib=32.0)
    assert r.fits is False
    assert r.kv_floor_bytes == _MIN_KV_BYTES


def test_the_measured_overhead_stays_off_where_vllm_prefills_8192_tokens():
    """From 70 GiB vLLM's API server takes 8,192 tokens a step, not 2,048, and the same
    checkpoints' activation peak measured 5.08 GiB there (RTX PRO 6000) against 2.50-2.63. The
    small-card figure would under-reserve by ~2.5 GiB, so big cards keep the envelope."""
    assert _arete(95.59).runtime_overhead is None


def test_the_family_overhead_covers_every_measured_run_with_a_cushion():
    """At least the worst run — 5 bpw under glq-chat — plus the spread in non-torch memory
    the SAME checkpoints showed between two cards (4 bpw 0.10 here against 0.34 on an RTX PRO
    6000, 5 bpw 0.37 against 0.59, 6 bpw 0.13 against 0.35). One card cannot show what a
    3090's driver and attention backend allocate."""
    from glq import tooling
    p = tooling.profile_for(ARETE_ID, "template")
    worst = max(_measured_overhead_gib(r) for r in _ARETE_RUNS.items())
    assert worst > 3.0, "the table must include the 5 bpw glq-chat run that set this"
    assert p.runtime_overhead >= (worst + (0.34 - 0.10)) * GIB


def test_the_kv_floor_of_a_measured_family_is_its_floor_windows_priced_kv():
    """`_MIN_KV_BYTES` stands in for "a useful context" when KV is priced at the envelope.
    A family whose KV cost and overhead were both measured is judged on its floor window's
    own priced KV instead — 16,384 tokens of Arete is 1.74 GiB with the 15% cushion, against
    the ~1.47 vLLM 0.33.0 asks for and the 2.38-2.60 GiB it served with on 24 GB."""
    from glq.supervisor import _MIN_KV_BYTES, window_kv_bytes
    s = _arete(24564 / 1024)
    r = _check(s, ram_gib=32.0)
    assert r.kv_floor_bytes == window_kv_bytes(CODE_FLOOR, 1, s.kv_cost)
    assert r.kv_floor_bytes < _MIN_KV_BYTES


def test_every_measured_overhead_comes_with_a_measured_kv_cost_and_no_drafter():
    """The measured KV floor above needs both measurements, and the runs behind every
    measured overhead had no draft model: a family that gains one must be measured again with
    it, not inherit a figure that never saw its activations."""
    from glq import tooling
    for p in tooling.PROFILES:
        if p.runtime_overhead is not None:
            assert p.kv_cost is not None, p.family
            assert p.dspark_drafter is None, p.family


def test_the_planner_and_the_gate_use_the_same_overhead():
    """The window and the pool are sized with the figure the gate judges with — two
    overheads in one plan is how a plan passes its check and then fails to start."""
    from glq.supervisor import (_MAX_UTILIZATION, _MIN_KV_BYTES, runtime_overhead_bytes,
                                window_kv_bytes)
    vram = 24564 / 1024 * GIB
    s = _arete(24564 / 1024)
    assert runtime_overhead_bytes(s.resident_bytes, s.runtime_overhead) == s.runtime_overhead
    assert s.max_model_len == CODE_FLOOR
    kv = max(_MIN_KV_BYTES, window_kv_bytes(CODE_FLOOR, 1, s.kv_cost))
    want = min(_MAX_UTILIZATION, (s.resident_bytes + s.runtime_overhead + kv) / vram)
    assert s.gpu_memory_utilization == pytest.approx(want, abs=1e-6)
