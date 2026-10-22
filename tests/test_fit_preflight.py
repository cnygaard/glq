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
