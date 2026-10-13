"""Host-memory offload in the serve plan: budget, flags and the env var.

Qwen3.8-Flash-Next is 72.2 GiB of weights that does not need 72.2 GiB of VRAM: GLQ can hold
its n-gram (PLE) table in pinned host memory and vLLM can hold the MoE experts. Measured on
real hardware (RTX PRO 6000 / L40S, vLLM 0.30.0):

    no offload                                  73.3 GiB resident
    PLE offload                                 49.53
    PLE + --cpu-offload-gb 24 ... experts       24.31
    PLE + --cpu-offload-gb 36 ... experts       12.15

These pin that the plan follows from the declared bytes, that a checkpoint declaring nothing
is untouched, and that the flags are the ones the UVA backend actually reads.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.supervisor import (WINDOW_OFFLOAD_MAX_EXTRA_GIB,  # noqa: E402
                            VllmSupervisor, child_env, kv_headroom_bytes,
                            plan_expert_offload_gib, window_kv_bytes)

GIB = 2 ** 30
#: Flash-Next as the quantizer declares it.
FN = dict(weights_bytes=int(72.2 * GIB), ple_offload_bytes=int(23.842 * GIB),
          expert_offload_bytes=int(36.18 * GIB), model_max_len=262144)


def _sup(vram_gib, ram_gib=124.0, **kw):
    return VllmSupervisor(model="org/flash-next", device="cuda",
                          vram_bytes=int(vram_gib * GIB), ram_bytes=int(ram_gib * GIB),
                          **{**FN, **kw})


# ---- the budget ------------------------------------------------------------------------

def test_no_offload_when_the_model_already_fits():
    """A 96 GB card holds the 48.4 GiB that remains after the PLE table leaves, so the
    experts stay resident. A box that does not need offload must not pay the PCIe cost."""
    assert _sup(95.0).expert_offload_gib == 0


def test_smaller_cards_get_a_bigger_budget():
    assert _sup(45.0).expert_offload_base_gib < _sup(23.0).expert_offload_base_gib


def test_the_budget_never_exceeds_the_declared_expert_bytes():
    """Claiming more than the checkpoint has means vLLM offloads less than planned, and the
    footprint prediction is then wrong in the dangerous direction."""
    s = _sup(8.0)
    assert 0 < s.expert_offload_gib <= FN["expert_offload_bytes"] // GIB


def test_the_budget_is_capped_by_pinnable_host_ram():
    """Pinned pages cannot be swapped: overshooting does not run slowly, it fails to
    allocate. 32 GiB of RAM cannot hold the 23.84 GiB PLE table plus 34 GiB of experts."""
    assert _sup(23.0, ram_gib=32.0).expert_offload_gib == 0
    assert _sup(23.0, ram_gib=128.0).expert_offload_gib > 0


def test_a_checkpoint_declaring_nothing_gets_no_budget():
    assert plan_expert_offload_gib(weights_bytes=int(72 * GIB),
                                   ple_offload_bytes=int(23 * GIB),
                                   expert_offload_bytes=0,
                                   vram_bytes=int(23 * GIB),
                                   ram_bytes=int(128 * GIB)) == 0


def test_unknown_vram_plans_no_offload():
    """Guessing with no card reading is worse than serving as before."""
    assert plan_expert_offload_gib(weights_bytes=int(72 * GIB),
                                   ple_offload_bytes=int(23 * GIB),
                                   expert_offload_bytes=int(36 * GIB),
                                   vram_bytes=None, ram_bytes=int(128 * GIB)) == 0


# ---- the flags -------------------------------------------------------------------------

def test_the_serve_command_uses_uva_and_the_uva_filter_flag():
    """`uva`, never `prefetch`: prefetch bulk-copies whole layer groups so it transfers
    experts that were never routed -- measured 2.4x slower at a SMALLER offload volume, and
    it fails outright on GLQ ("CPU storage for ...trellis_packed is not pinned!").

    And the filter is `--cpu-offload-params`, which is the UVA backend's field.
    `--offload-params` belongs to prefetch and would be accepted but ignored here."""
    argv = _sup(23.0).argv()
    assert "--offload-backend" in argv and argv[argv.index("--offload-backend") + 1] == "uva"
    assert "--cpu-offload-params" in argv
    assert argv[argv.index("--cpu-offload-params") + 1] == "experts"
    assert "--offload-params" not in argv
    assert "prefetch" not in argv


def test_the_budget_reaches_the_command_line():
    s = _sup(23.0)
    argv = s.argv()
    assert argv[argv.index("--cpu-offload-gb") + 1] == str(s.expert_offload_gib)


def test_no_offload_flags_at_all_when_the_budget_is_zero():
    assert not [a for a in _sup(95.0).argv() if "offload" in a]


def test_a_dense_checkpoint_is_untouched():
    """The planners are on every install path; a change here must not reach checkpoints that
    have nothing to do with offload."""
    s = VllmSupervisor(model="org/smollm3", device="cuda", weights_bytes=int(1.9 * GIB),
                       vram_bytes=int(23 * GIB), model_max_len=65536)
    assert s.expert_offload_gib == 0
    assert not [a for a in s.argv() if "offload" in a]


# ---- the env var -----------------------------------------------------------------------

def test_ple_offload_is_turned_on_explicitly(monkeypatch):
    """vLLM defaults this ON, but a checkpoint whose n-gram table does not fit beside its
    decoder CANNOT serve without it -- at a 262k context the resident configuration stops at
    startup needing 6.55 GiB of KV it does not have. Relying on an upstream default for that
    is what breaks on a vLLM bump."""
    monkeypatch.delenv("VLLM_PLE_CPU_OFFLOAD", raising=False)
    assert child_env(device="cuda", ple_offload=True)["VLLM_PLE_CPU_OFFLOAD"] == "1"


def test_it_is_not_set_for_a_checkpoint_without_a_ple_table(monkeypatch):
    monkeypatch.delenv("VLLM_PLE_CPU_OFFLOAD", raising=False)
    assert "VLLM_PLE_CPU_OFFLOAD" not in child_env(device="cuda", ple_offload=False)


def test_a_user_set_value_still_wins(monkeypatch):
    """setdefault, so someone who deliberately exports 0 gets their answer -- and GLQ's own
    refusal message naming the setting -- rather than being silently overridden."""
    monkeypatch.setenv("VLLM_PLE_CPU_OFFLOAD", "0")
    assert child_env(device="cuda", ple_offload=True)["VLLM_PLE_CPU_OFFLOAD"] == "0"


# ---- sizing from the resident footprint ------------------------------------------------

def test_the_pool_is_sized_from_the_resident_footprint_not_the_file_size():
    """Counting the n-gram table against the card it never occupies reserved 78.2 GiB of a
    96 GB card for a model that needs 48.4 -- seizing VRAM for nothing.

    Asserted against a control rather than a fixed threshold, and on the WEIGHTS term rather
    than the pool total. `util < 0.70` was a proxy for "the offload was credited", and the
    proxy stopped working once the pool began holding the served window: the freed VRAM is now
    spent on context (131072 instead of 32768 on this card) rather than handed back, so the
    two arms' totals are close while the weights accounting differs by the whole table."""
    from glq.supervisor import _RUNTIME_OVERHEAD_BYTES, window_kv_bytes

    def weights_term(sup):
        """What the plan set aside for weights + overhead, with the window's KV removed."""
        return (sup.gpu_memory_utilization * 95 * GIB
                - window_kv_bytes(sup.max_model_len, sup.window_concurrency))

    declared = _sup(95.0)
    undeclared = _sup(95.0, ple_offload_bytes=0, expert_offload_bytes=0)
    credited = weights_term(undeclared) - weights_term(declared)
    assert abs(credited - FN["ple_offload_bytes"]) < GIB, (
        f"credited {credited / GIB:.2f} GiB for a "
        f"{FN['ple_offload_bytes'] / GIB:.2f} GiB table")
    # The user-visible payoff: the same card affords a longer context for it.
    assert declared.max_model_len > undeclared.max_model_len
    assert weights_term(declared) - _RUNTIME_OVERHEAD_BYTES < 50 * GIB


# ------------------------------------- non-text bytes reach the pool and window plans

# The last term of the resident footprint. `size - ple - experts` predicted 15.72 GiB at a
# 38 GiB expert budget; vLLM reported `Model loading took 10.1 GiB`. The 5.62 GiB difference
# is the MTP head (4.856 GiB) and vision tower (0.836 GiB), counted from the checkpoint's own
# safetensors headers, and on a 24 GB card it is worth two window tiers.

FN_NT = dict(FN, nontext_bytes=int(5.69 * GIB))


#: Flash-Next exactly as published, and the corrected counts measured from its safetensors
#: headers. The published `expert_offload_bytes` double-counts the two MTP expert tensors,
#: which is why the corrected expert figure is 4.68 GiB SMALLER while nontext appears.
FN_PUBLISHED = dict(weights_bytes=int(77.56 * GIB), ple_offload_bytes=25_600_122_880,
                    expert_offload_bytes=50_803_802_112, model_max_len=262144)
FN_CORRECTED = dict(FN_PUBLISHED, expert_offload_bytes=45_770_637_312,
                    nontext_bytes=5_214_301_696)   # MTP head only; see quantize tests


def _code_sup(vram_gib, ram_gib=124.0, **kw):
    """A glq-code-shaped plan: one stream, the coding floor, and the window-for-offload trade.

    `window_offload_extra_gib` is what `glq-code` passes and `glq-chat` does not; defaulting it
    here rather than in each test keeps "the code shape" one thing.
    """
    kw.setdefault("window_offload_extra_gib", WINDOW_OFFLOAD_MAX_EXTRA_GIB)
    return VllmSupervisor(model="org/flash-next", device="cuda", window_concurrency=1,
                          max_num_seqs=1, vram_bytes=int(vram_gib * GIB),
                          ram_bytes=int(ram_gib * GIB), max_model_len_floor=16384, **kw)


def test_the_window_grows_on_a_24gb_card_once_the_counts_are_right():
    """The payoff, on the card that motivated it and with the published numbers rather than a
    fixture: 16384 -> 131072 for the same checkpoint and the same policy.

    The history matters, because this pair has moved twice and each move was the arithmetic
    getting more honest rather than the feature getting better. It was 32768 -> 65536 while the
    overhead allowance was a flat 4 GiB -- measurably 1.44 GiB short on this checkpoint, so the
    old pair was reachable only by under-reserving what vLLM then needed; proportional overhead
    cost this card a tier, correctly, leaving 16384 -> 32768. The window-aware budget then
    raised the second arm again, and this time nothing was borrowed to pay for it.

    `before` is unimproved ON PURPOSE and is the control: the published counts over-state what
    must leave the card, so even at the pinnable-RAM cap (38 GiB) the 15.72 GiB that stays
    resident leaves under 1 GiB of KV headroom and no tier above the floor is affordable. The
    corrected counts reach 131072 while offloading one GiB LESS, which is the whole claim --
    the context came from counting right, not from spending more PCIe."""
    # `max_model_len_base` rather than `max_model_len`: the window is now re-planned once
    # deepening for the expert arena frees VRAM, and that re-plan DOES spend more PCIe, so it
    # would rescue `before` too and dissolve the control. The base figure is the one this
    # claim is about — the context the corrected counts afford without deepening.
    before = _code_sup(23.0, **FN_PUBLISHED)
    after = _code_sup(23.0, **FN_CORRECTED)
    assert before.max_model_len_base == 16384
    assert after.max_model_len_base == 131072
    assert after.expert_offload_base_gib < before.expert_offload_base_gib


def test_the_same_fix_also_hands_back_utilization():
    """The third side of the saving. Counting the MTP head correctly buys context (the test
    above) and costs one GiB less offload, and what is left over is given back to the card
    rather than reserved: util 0.920 -> 0.877 on the 23 GiB arm, 0.703 -> 0.651 on the 96 GiB
    one. The 0.920 is `_MAX_UTILIZATION`, i.e. the published counts saturate the card AND get
    the shortest window -- the worst corner of both trades."""
    for vram in (23.0, 95.6):
        before = _code_sup(vram, **FN_PUBLISHED)
        after = _code_sup(vram, **FN_CORRECTED)
        assert after.gpu_memory_utilization < before.gpu_memory_utilization


def test_a_96gb_card_keeps_its_window_and_reserves_less_for_it():
    """No regression where the window was already maximal — the gain there is VRAM handed
    back rather than context."""
    before = _code_sup(95.6, **FN_PUBLISHED)
    after = _code_sup(95.6, **FN_CORRECTED)
    assert after.max_model_len == before.max_model_len == 262144
    assert after.gpu_memory_utilization < before.gpu_memory_utilization


def test_the_expert_budget_shrinks_because_less_has_to_leave():
    """Same mechanism from the other side: if 5.69 GiB was never resident, less of the
    experts need to go to host RAM, and every offloaded expert costs PCIe on every token."""
    assert _sup(23.0, **FN_NT).expert_offload_gib <= _sup(23.0, **FN).expert_offload_gib


def test_an_undeclared_checkpoint_plans_exactly_as_before():
    """Every published checkpoint except the patched one has no such field, and must size
    identically to today -- this is the guard that keeps the change inert for them."""
    a = _sup(45.0, **FN)
    b = _sup(45.0, **dict(FN, nontext_bytes=0))
    assert (a.expert_offload_gib, a.max_model_len, a.gpu_memory_utilization) == \
           (b.expert_offload_gib, b.max_model_len, b.gpu_memory_utilization)


def test_speculative_decoding_is_never_enabled_so_the_mtp_subtraction_holds():
    """The one soundness condition. MTP weights DO load under speculative decoding, so
    subtracting them is only valid because glq-chat and glq-code never ask for it. Asserted
    on the generated argv rather than trusted: if a future change adds the flag, the resident
    estimate silently becomes wrong by ~4.9 GiB."""
    argv = " ".join(str(a) for a in _sup(95.0, **FN_NT).argv())
    for flag in ("--speculative", "--num-speculative-tokens", "speculative_config"):
        assert flag not in argv, f"{flag} would load the MTP head this plan excludes"


# ------------------------------------------- the explicit offload budget escape hatch

# The budget is computed inside the supervisor, and hand-running `vllm serve` to override it
# loses `flashinfer_env()` (on sm_120 it then stops on ninja) -- so without a flag there is no
# way to disagree with the plan at all. That mattered most when the plan stopped at
# `WEIGHT_FRACTION`; the window-aware budget below now reaches 131072 on a 23 GiB L4 unaided,
# and this flag has become what it should be: the override for the cases the policy cannot see,
# including `0` for "serve it resident and keep every token fast".

def test_an_explicit_budget_is_used_verbatim():
    """Same contract as --gpu-memory-utilization: an explicit flag wins over the plan."""
    sup = _code_sup(23.0, expert_offload_gib=42, **FN_CORRECTED)
    assert sup.expert_offload_gib == 42
    argv = " ".join(str(a) for a in sup.argv())
    assert "--cpu-offload-gb 42" in argv


def test_the_explicit_budget_lengthens_the_window_it_was_asked_for():
    """The point of the flag -- the window is sized from the resident footprint the budget
    produces, so raising the budget is what buys the context."""
    # At 70 GiB of host RAM the pinnable cap holds the planned offload down, so the window
    # the budget buys is visible. On a 124 GiB host both arms reach this card's 131072
    # ceiling (`window_ceiling_for_vram`) and the property is masked by saturation rather
    # than absent.
    planned = _code_sup(23.0, ram_gib=70.0, **FN_CORRECTED)
    forced = _code_sup(23.0, ram_gib=70.0, expert_offload_gib=42, **FN_CORRECTED)
    assert forced.max_model_len > planned.max_model_len


def test_zero_turns_offload_off_rather_than_meaning_unset():
    """`--cpu-offload-gb 0` has to be distinguishable from not passing it, or there is no way
    to say "serve this resident" on a card where the planner would offload."""
    sup = _code_sup(23.0, expert_offload_gib=0, **FN_CORRECTED)
    assert sup.expert_offload_gib == 0
    assert "--cpu-offload-gb" not in " ".join(str(a) for a in sup.argv())


def test_an_absent_flag_still_plans():
    planned = _code_sup(23.0, **FN_CORRECTED)
    assert planned.expert_offload_gib > 0        # the 34 GiB the policy chooses


def test_a_budget_beyond_the_declared_experts_is_reported_not_silently_clipped():
    """Asking for more than the checkpoint has means vLLM offloads less than requested, so the
    footprint prediction is wrong in the dangerous direction. The flag still wins -- it is an
    escape hatch -- but it must say so rather than letting the user believe the number."""
    import io
    out = io.StringIO()
    sup = _code_sup(23.0, expert_offload_gib=999, out=out, **FN_CORRECTED)
    sup.argv()
    assert sup.expert_offload_gib == 999
    assert "declared" in out.getvalue().lower() or "999" in out.getvalue()


# ------------------------------------------------------- the window-aware offload budget

# The budget used to stop as soon as resident fit `WEIGHT_FRACTION` of VRAM, without asking
# whether the headroom it left affords a useful window. Two failures came out of that, and the
# first is not an optimization:
#
#   * a PINNED `--max-model-len` could not drive the budget. Offload was planned first and the
#     pool sized for the window afterwards, clamped at `_MAX_UTILIZATION` -- so on a 23 GiB L4
#     the plan offloaded 34 GiB, leaving 1.85 GiB of KV headroom against the 7.92 GiB a 262144
#     window needs, and vLLM refused at startup. That is the 0.8.24 report (`6.55 GiB KV cache
#     is needed ... larger than the available 5.91 GiB`); 0.8.25 made the overhead allowance
#     honest without connecting the window to the budget.
#   * the auto window stopped short for the agent that wants context: 32768 on that L4, where
#     +3 GiB of offload reaches 131072.
#
# The trade is measured rather than assumed, on two cards -- see `WINDOW_OFFLOAD_MAX_EXTRA_GIB`
# for the arms. ~0.55%/GiB of decode on an RTX PRO 4500, ~1.10%/GiB on an RTX PRO 6000; the
# percentage is NOT portable (a fixed PCIe cost is a larger share of a faster step) while
# ~0.47 ms/token/GiB is, across two PCIe 5 x16 boxes.
#
# The pricing chain was then gated against vLLM itself rather than inferred. Serving Flash-Next
# on a 96 GiB RTX PRO 6000 at the plan's own choice (util 0.652, window 262144, offload 0,
# vLLM 0.31.0): `Model loading took 48.65 GiB` against a predicted 48.86 resident, and
# `Available KV cache memory: 8.24 GiB` / `GPU KV cache size: 328,790 tokens` against a
# reservation of 7.92 GiB -- hence `Maximum concurrency for 262,144 tokens per request: 1.25x`,
# and a real request completed. vLLM's own per-token cost there is 26,912 B, which makes
# `_KV_BYTES_PER_TOKEN = 28_201` conservative by 4.8%: the safe direction, and the first check
# of that anchor AT a 262144 window rather than extrapolated from pools measured at 8192.


def test_a_pinned_window_drives_the_budget():
    """The failure this fixes. A user asking for 262144 on a 23 GiB card used to get the
    fits-the-card budget and a pool that cannot hold one request in that window.

    Asserted on HEADROOM rather than on a budget number, because the headroom is the thing
    vLLM then checks."""
    sup = _code_sup(23.0, ram_gib=192.0, max_model_len=262144, **FN_CORRECTED)
    resident = (FN_CORRECTED["weights_bytes"] - FN_CORRECTED["ple_offload_bytes"]
                - FN_CORRECTED["nontext_bytes"] - sup.expert_offload_gib * GIB)
    assert kv_headroom_bytes(resident, int(23.0 * GIB)) >= window_kv_bytes(262144, 1)


def test_a_pinned_window_is_not_subject_to_the_auto_cap():
    """`glq-chat` does not trade decode for a window it was not asked for, but a window the
    user NAMED is not a trade -- it is an instruction. So the pinned path must work with the
    cap at its chat default of 0."""
    chat_shaped = VllmSupervisor(
        model="org/flash-next", device="cuda", window_concurrency=1, max_num_seqs=1,
        vram_bytes=int(23.0 * GIB), ram_bytes=int(192.0 * GIB),
        max_model_len=262144, window_offload_extra_gib=0, **FN_CORRECTED)
    baseline = _code_sup(23.0, ram_gib=192.0, window_offload_extra_gib=0,
                         **FN_CORRECTED).expert_offload_base_gib
    assert chat_shaped.expert_offload_base_gib > baseline


def test_an_unaffordable_pinned_window_is_served_anyway_and_reported():
    """`_KV_BYTES_PER_TOKEN` is a conservative envelope over the worst pool ever observed, so
    a window this arithmetic calls unaffordable may still serve -- and vLLM's own startup check
    is the authoritative one. Clamping would shorten windows that would have worked, so the
    pinned value is passed through untouched and the shortfall is REPORTED.

    Same contract as `--cpu-offload-gb 999`: honored verbatim, never silently adjusted."""
    import io
    out = io.StringIO()
    # A 16 GiB card with ample (192 GiB) host RAM: every expert is already offloaded, so
    # host RAM is NOT the binding constraint and the card genuinely cannot hold this window's
    # KV. That separation matters — the same shortfall on a RAM-starved host is reported
    # differently, by `_say_window_shortfall`'s host-RAM branch, because "raise
    # --cpu-offload-gb" is not advice a capped host can act on.
    sup = _code_sup(16.0, ram_gib=192.0, max_model_len=262144, out=out, **FN_CORRECTED)
    sup.argv()
    assert sup.max_model_len == 262144           # not clamped
    said = out.getvalue().lower()
    assert "262144" in said and "cpu-offload-gb" in said


def test_the_auto_window_tiers_up_within_the_budget():
    """The measured payoff on the card that motivated this: 32768 -> 131072 for +3 GiB of
    offload, about 1.7% of decode at the slope above.

    262144 is NOT reached here and that is correct -- it needs ~40 GiB, which 124 GiB of host
    RAM cannot pin beside the 23.84 GiB PLE table."""
    sup = _code_sup(23.0, **FN_CORRECTED)
    assert sup.expert_offload_base_gib == 37
    assert sup.max_model_len == 131072


def test_the_auto_window_reaches_the_full_context_on_a_32gb_card():
    """The g7, where the trade is bracketed by measurement rather than extrapolated: the
    chosen 32 GiB sits between the 27 GiB and 34 GiB arms that were benched."""
    sup = _code_sup(31.9, **FN_CORRECTED)
    assert sup.expert_offload_base_gib == 32
    assert sup.max_model_len == 262144


def test_it_picks_the_smallest_offload_that_reaches_the_tier():
    """Walking the tiers downward is what makes this land on the sweet spot instead of the
    declared maximum. On the g7, 32 GiB and 38 GiB both serve 262144; every GiB beyond the
    first one that reaches the tier is decode speed spent for nothing."""
    sup = _code_sup(31.9, **FN_CORRECTED)
    lower = _code_sup(31.9, expert_offload_gib=sup.expert_offload_base_gib - 1, **FN_CORRECTED)
    assert lower.max_model_len < sup.max_model_len


def test_the_window_the_search_accepted_is_the_window_that_gets_served():
    """The search and `plan_max_model_len` must agree: the budget is chosen by asking which
    tier a resident footprint affords, and the window is then planned from that same
    footprint. If those two ever disagree the plan promises a window it did not buy."""
    for vram in (23.0, 31.9, 95.6):
        sup = _code_sup(vram, **FN_CORRECTED)
        resident = (FN_CORRECTED["weights_bytes"] - FN_CORRECTED["ple_offload_bytes"]
                    - FN_CORRECTED["nontext_bytes"] - sup.expert_offload_gib * GIB)
        assert kv_headroom_bytes(resident, int(vram * GIB)) >= window_kv_bytes(
            sup.max_model_len, 1), f"{vram} GiB promised {sup.max_model_len}"


def test_a_tier_the_caps_cannot_reach_is_refused_not_promised():
    """The cap interaction that would otherwise produce a window the card cannot hold: the
    pinnable-RAM and declared-expert caps can clip the budget BELOW what the tier needs, and
    accepting the clipped value would announce a context that does not fit.

    Same card, same checkpoint, more host RAM -- and only the one with RAM to pin reaches the
    tier. The pair is 90 vs 124 GiB rather than 124 vs 192: this card's auto window is now
    capped at 131072 by `window_ceiling_for_vram`, so 124 and 192 both land there and the
    distinction has to be drawn below the ceiling, where it is still real."""
    from glq.supervisor import window_ceiling_for_vram

    lean = _code_sup(23.0, ram_gib=90.0, **FN_CORRECTED).max_model_len
    ample = _code_sup(23.0, ram_gib=124.0, **FN_CORRECTED).max_model_len
    assert lean < ample, f"more host RAM should reach a higher tier: {lean} vs {ample}"
    assert ample == window_ceiling_for_vram(int(23.0 * 1024**3))


def test_the_extra_never_exceeds_the_declared_budget():
    """The one weakly-justified constant here, so it is asserted rather than trusted. On every
    card measured the caps bind first and this never does, which is where a constant backed by
    three points on one box belongs."""
    base = plan_expert_offload_gib(
        weights_bytes=FN_CORRECTED["weights_bytes"],
        ple_offload_bytes=FN_CORRECTED["ple_offload_bytes"],
        expert_offload_bytes=FN_CORRECTED["expert_offload_bytes"],
        nontext_bytes=FN_CORRECTED["nontext_bytes"],
        vram_bytes=int(23.0 * GIB), ram_bytes=int(192.0 * GIB))
    sup = _code_sup(23.0, ram_gib=192.0, **FN_CORRECTED)
    assert sup.expert_offload_gib <= base + WINDOW_OFFLOAD_MAX_EXTRA_GIB


def test_a_big_card_keeps_a_zero_budget():
    """No offload where none is needed. A 96 GiB card already affords 262144 resident, so
    there is no tier to buy and nothing to pay for it with."""
    sup = _code_sup(95.6, **FN_CORRECTED)
    assert sup.expert_offload_gib == 0
    assert sup.max_model_len == 262144


# ---- what must NOT move ----------------------------------------------------------------

def test_the_chat_default_plans_exactly_as_before():
    """`glq-chat` opts out: a session at 8k tokens must not pay PCIe on every token for a
    window it never fills. Asserted to the digit on all three planned values, because this is
    the guard that keeps the change inert for every checkpoint and card glq-chat serves."""
    for vram in (23.0, 31.9, 45.0, 95.6):
        before = _sup(vram, **FN)
        after = _sup(vram, window_offload_extra_gib=0, **FN)
        assert (after.expert_offload_gib, after.max_model_len,
                after.gpu_memory_utilization) == (
            before.expert_offload_gib, before.max_model_len, before.gpu_memory_utilization)


def test_a_dense_checkpoint_is_still_untouched():
    """The planners are on every install path, and the trade must not reach a checkpoint with
    no experts to offload."""
    s = VllmSupervisor(model="org/smollm3", device="cuda", weights_bytes=int(1.9 * GIB),
                       vram_bytes=int(23 * GIB), model_max_len=65536,
                       window_offload_extra_gib=WINDOW_OFFLOAD_MAX_EXTRA_GIB)
    assert s.expert_offload_gib == 0
    assert not [a for a in s.argv() if "offload" in a]


def test_an_explicit_budget_still_beats_the_window_aware_plan():
    """The escape hatch outranks the richer policy exactly as it outranked the poorer one --
    including 0, which on this card now means giving up three window tiers on purpose."""
    assert _code_sup(23.0, expert_offload_gib=20, **FN_CORRECTED).expert_offload_gib == 20
    zero = _code_sup(23.0, expert_offload_gib=0, **FN_CORRECTED)
    assert zero.expert_offload_gib == 0
    assert "--cpu-offload-gb" not in " ".join(str(a) for a in zero.argv())


# ---------------------------------------------------------------------------
# ★ The expert arena: default ON when experts are offloaded, window keeps priority
# ---------------------------------------------------------------------------
#
# Measured on an L4 (benchmarks/expert_arena_s1/RESULT.md): a routing-aware arena takes decode
# from 6.436 to 17.211 tok/s, byte-identical, and 2.441x of that survives an iso-VRAM control
# -- so it is not just "more weights resident". It is worth defaulting on, and it costs VRAM
# the window also wants, so the two have to be planned together.

_L4 = int(23034 * 1024**2)                 # as nvidia-smi reports the card
_FLASHNEXT = int(72.2 * 1024**3)
_PLE = int(23.84 * 1024**3)
_EXPERTS = int(42.62 * 1024**3)
_NONTEXT = int(4.69 * 1024**3)             # the bf16 MTP head, never loaded


from glq.supervisor import (_WINDOW_TIERS, plan_expert_arena_gib,  # noqa: E402
                            plan_max_model_len, window_ceiling_for_vram)


def _arena(**kw):
    base = dict(weights_bytes=_FLASHNEXT, ple_offload_bytes=_PLE,
                expert_offload_bytes=_EXPERTS, nontext_bytes=_NONTEXT,
                vram_bytes=_L4, offload_gib=42, kv_bytes=0)
    base.update(kw)
    return plan_expert_arena_gib(**base)


def test_no_arena_when_nothing_is_offloaded():
    """★ The arena's whole job is to stop re-reading experts over PCIe. With the model
    resident there is no traffic to remove, and the arena would be a second copy of weights
    the card already holds — it would push the model back into offload to pay for itself.

    `arena_decision` refuses this independently at load time; this is the earlier, cheaper
    check, and it keeps a big card's plan byte-for-byte what it is today.
    """
    assert _arena(offload_gib=0) == 0


def test_no_arena_when_an_input_is_unknown():
    """Matches `plan_expert_offload_gib`: guessing is worse than serving as before."""
    assert _arena(vram_bytes=None) == 0
    assert _arena(weights_bytes=None) == 0


def test_the_262144_window_keeps_its_KV_and_still_gets_an_arena():
    """★ The case the whole policy is built on, and the regression test for it.

    On a 22.49 GiB L4 the *tested* 8 GiB arena plus a 262,144 window needs 23.07 GiB against a
    20.69 GiB budget — 2.37 GiB short — so both are not available and the window wins. What is
    left is still a useful arena: ~5.6 GiB is ~66 slots/layer, and the hit-rate curve has
    flattened by then (measured 71 slots -> 16.816 tok/s against 94 -> 17.211, 2.4% apart).

    So the window is never shortened to buy decode, and the arena still arrives.
    """
    kv = window_kv_bytes(262144, 1)
    gib = _arena(kv_bytes=kv)
    assert gib > 0, "a 262k window should still leave room for an arena"
    assert 4 <= gib <= 7, f"expected ~5-6 GiB on this card, got {gib}"


def test_the_arena_never_eats_the_window_KV_at_any_tier():
    """At every tier the plan must still afford the window it chose. An arena that left the
    window short would turn a served window into a startup refusal, which is the failure
    `plan_expert_offload_gib` was already hardened against."""
    from glq.supervisor import _MIN_KV_BYTES, kv_headroom_bytes

    for tier in _WINDOW_TIERS:
        kv = window_kv_bytes(tier, 1)
        gib = _arena(kv_bytes=kv)
        resident = (_FLASHNEXT - _PLE - _NONTEXT - 42 * 1024**3) + gib * 1024**3
        left = kv_headroom_bytes(max(0, resident), _L4)
        assert left >= kv or gib == 0, (
            f"tier {tier}: arena {gib} GiB leaves {left / 1024**3:.2f} GiB for a window "
            f"needing {kv / 1024**3:.2f}")
        assert left >= min(kv, _MIN_KV_BYTES) or gib == 0


def test_a_bigger_card_gets_a_bigger_arena():
    """Sizing follows genuinely spare VRAM, so a card with more of it gets more arena. Pins
    that the planner is not returning a constant."""
    small = _arena(kv_bytes=window_kv_bytes(262144, 1))
    big = plan_expert_arena_gib(
        weights_bytes=_FLASHNEXT, ple_offload_bytes=_PLE, expert_offload_bytes=_EXPERTS,
        nontext_bytes=_NONTEXT, vram_bytes=int(48 * 1024**3), offload_gib=42,
        kv_bytes=window_kv_bytes(262144, 1))
    assert big > small


# ---------------------------------------------------------------------------
# child_env: the arena vars, and the user's last word
# ---------------------------------------------------------------------------

def test_child_env_sets_the_arena_vars_when_offload_is_planned():
    from glq.supervisor import child_env

    env = child_env(device="cuda", expert_arena_gib=6, max_num_seqs=1)
    assert env["GLQ_MOE_EXPERT_CACHE"] == "6"
    assert env["GLQ_MOE_ARENA_MAX_TOKENS"] == "1"


def test_child_env_sets_NOTHING_when_no_arena_was_planned():
    """A card that fits the model must be byte-for-byte unchanged, env included."""
    from glq.supervisor import child_env

    env = child_env(device="cuda", expert_arena_gib=0, max_num_seqs=8)
    assert "GLQ_MOE_EXPERT_CACHE" not in env
    assert "GLQ_MOE_ARENA_MAX_TOKENS" not in env


def test_an_exported_arena_budget_WINS_including_zero(monkeypatch):
    """★ `GLQ_MOE_EXPERT_CACHE=0` is how a user opts out, so it must survive the plan.
    Follows the `VLLM_PLE_CPU_OFFLOAD` / `VLLM_CPU_KVCACHE_SPACE` precedent in `child_env`:
    every value there is a setdefault precisely so an explicit export is the last word."""
    from glq.supervisor import child_env

    monkeypatch.setenv("GLQ_MOE_EXPERT_CACHE", "0")
    monkeypatch.setenv("GLQ_MOE_ARENA_MAX_TOKENS", "3")
    env = child_env(device="cuda", expert_arena_gib=6, max_num_seqs=1)
    assert env["GLQ_MOE_EXPERT_CACHE"] == "0"
    assert env["GLQ_MOE_ARENA_MAX_TOKENS"] == "3"


def test_the_cpu_path_never_plans_an_arena():
    """The arena is a VRAM structure served by a Triton kernel; on the CPU backend there is
    no card to hold it and no PCIe hop to remove."""
    from glq.supervisor import child_env

    env = child_env(device="cpu", ram_bytes=64 * 1024**3, weights_bytes=10 * 1024**3,
                    expert_arena_gib=6, max_num_seqs=1)
    assert "GLQ_MOE_EXPERT_CACHE" not in env


# ---------------------------------------------------------------------------
# ★ MTP stays OFF, and the offload plan depends on it
# ---------------------------------------------------------------------------

def test_speculation_is_OFF_unless_asked_for_and_the_subtraction_holds():
    """★ A conditional that another plan depends on, so it is asserted rather than assumed.

    `supervisor.py` subtracts `nontext_bytes` — the MTP head and vision tower — from the
    weights it plans against, which closed a measured 5.62 GiB over-estimate (15.72 predicted
    vs 10.1 reported), worth two window tiers on a 24 GB card. That is sound only while
    speculation is off, because MTP loads the head.

    This used to assert the supervisor could NEVER speculate, which is how the condition was
    kept true. `mtp_tokens` now makes it a conditional instead: the subtraction is withheld
    exactly when the head is loaded (see `tests/test_mtp_planning.py`). What this test keeps
    is the other half — that nothing speculates unless a caller asked, so the default plan is
    unchanged.
    """
    sup = VllmSupervisor(model="m", device="cuda", vram_bytes=_L4,
                         weights_bytes=_FLASHNEXT, ple_offload_bytes=_PLE,
                         expert_offload_bytes=_EXPERTS, nontext_bytes=_NONTEXT,
                         serve=False)
    argv = " ".join(sup.argv())
    assert "--speculative-config" not in argv
    assert "speculative" not in argv
    # and the subtraction that depends on it is in force
    assert sup.nontext_bytes == _NONTEXT
    assert sup.step_tokens == sup.max_num_seqs, "a non-speculative step is one token/stream"
    assert sup.min_step_tokens == 1


# ---------------------------------------------------------------------------
# ★ Deepening offload to buy the arena
# ---------------------------------------------------------------------------

def _pair(**kw):
    from glq.supervisor import plan_arena_and_offload
    base = dict(weights_bytes=_FLASHNEXT, ple_offload_bytes=_PLE,
                expert_offload_bytes=_EXPERTS, nontext_bytes=_NONTEXT, vram_bytes=_L4,
                ram_bytes=int(120 * 1024**3), kv_bytes=window_kv_bytes(262144, 1),
                base_offload_gib=36)
    base.update(kw)
    return plan_arena_and_offload(**base)


def test_a_model_that_FITS_is_left_completely_alone():
    """★ "when offload is needed" is the whole condition. With nothing offloaded the model is
    resident, there is no PCIe traffic for an arena to remove, and deepening offload to create
    some would be strictly worse than serving as before."""
    assert _pair(base_offload_gib=0) == (0, 0)


def test_offload_is_deepened_so_the_arena_has_room():
    """★ The minimum-offload plan leaves nothing spare, so the arena would be 0 GiB on the
    card it was built for. Measured on an L4: all experts offloaded plus an 8 GiB arena is
    17.211 tok/s; the shallower plan with no arena is ~6.8.

    Deepening does not cost context — more offload means *more* KV headroom, not less.

    The chat window (8192 at concurrency 8) is the case where it clearly pays: offload
    deepens 29 -> 36 and a 5 GiB arena appears. 36 rather than the full 42 because the host
    RAM cap binds first, which the test below pins.
    """
    off, arena = _pair(base_offload_gib=29, kv_bytes=window_kv_bytes(8192, 8))
    assert off > 29, f"offload should deepen past the minimum, got {off}"
    assert arena >= 4, f"expected a usable arena, got {arena} GiB"


def test_a_host_with_RAM_to_spare_offloads_everything():
    """With the pinned-RAM cap out of the way, deepening reaches the full expert set — which
    is the configuration the 17.211 tok/s arm actually ran."""
    off, arena = _pair(ram_bytes=int(192 * 1024**3))
    assert off >= int(_EXPERTS // 1024**3) - 1, f"expected full offload, got {off}"
    assert arena >= 5


def test_the_window_still_wins_at_262k_but_every_shorter_one_gets_an_arena():
    """★ Where the window/arena boundary actually falls, with the recorded 77.52 GiB
    checkpoint size rather than a docstring figure.

    On a 22.49 GiB card a 262,144 window needs 7.92 GiB of KV against ~7.4 GiB of headroom
    even with every expert offloaded, so the two genuinely do not fit and the window wins —
    it is not shortened to buy decode, and the arena declines. One card size up (23.0 GiB) it
    just fits, at 1 GiB.

    Every shorter window does get one, and that is what deepening plus
    `_ARENA_PINNED_RAM_FRACTION` bought: without them all four rows would be 0.
    """
    kw = dict(weights_bytes=_FN_TRUE, base_offload_gib=36, ram_bytes=int(121 * 1024**3))
    off, arena = _pair(kv_bytes=window_kv_bytes(262144, 1), **kw)
    assert (off, arena) == (36, 0), "262k on this card leaves nothing; the window wins"

    for window, least in ((131072, 3), (65536, 5), (8192, 7)):
        off, arena = _pair(kv_bytes=window_kv_bytes(window, 1), **kw)
        assert off > 36, f"window {window}: should deepen, got {off}"
        assert arena >= least, f"window {window}: expected >= {least} GiB, got {arena}"


def test_deepening_leaves_the_window_its_KV():
    """Deeper offload frees VRAM, so this should hold by construction — assert it anyway,
    because a window that is planned and then refused at startup is the failure mode
    `plan_expert_offload_gib` was already hardened against."""
    from glq.supervisor import kv_headroom_bytes

    kv = window_kv_bytes(262144, 1)
    off, arena = _pair(kv_bytes=kv)
    resident = (_FLASHNEXT - _PLE - _NONTEXT - off * 1024**3) + arena * 1024**3
    assert kv_headroom_bytes(max(0, resident), _L4) >= kv


def test_a_host_without_RAM_for_the_experts_is_not_asked_to_hold_them():
    """Pinned pages cannot be swapped, so overshooting here does not run slowly — it fails to
    allocate or starves the OS. `_PINNED_RAM_FRACTION` already caps `plan_expert_offload_gib`;
    deepening respects the same cap, and on a host with no room it does not deepen at all.

    This cap is why the measured 17.211 arm is not simply the default: that run passed
    `--cpu-offload-gb 44` explicitly, pinning 65.3 GiB against a 121 GiB host — 54%, past this
    policy's 50% — and `free` showed about 1 GiB spare at peak.
    """
    off, _arena = _pair(ram_bytes=int(40 * 1024**3))
    assert off == 36, "a host that cannot hold more must not be asked to"


def test_a_user_forced_offload_is_never_deepened():
    """An explicit --cpu-offload-gb is an instruction, not a starting point. Quietly moving
    more weight to host than the user asked for would change the footprint they chose."""
    off, _arena = _pair(base_offload_gib=20, forced=True)
    assert off == 20


# ---------------------------------------------------------------------------
# ★ Pinning: a higher share, but only when it buys an arena
# ---------------------------------------------------------------------------

_FN_TRUE = int(77.52 * 1024**3)      # the recorded checkpoint total, not the 72.2 docstring


def test_deepening_may_pin_more_than_the_base_plan_would():
    """★ `_PINNED_RAM_FRACTION` 0.5 caps expert offload at ~35 GiB on a 121 GiB host, which
    leaves ~14 GiB resident on a 22 GiB card and no VRAM for an arena at any window. The
    measured 2.674x configuration pins 66.5 GiB — **54.9%** — so the default could never reach
    it.

    `_ARENA_PINNED_RAM_FRACTION` permits that, and only here: the arena is what makes the
    deeper offload worth paying for. Evidence it is safe on the box it was measured on: that
    run pinned 65.3 GiB with **no swap** and still had ~13 GiB available at peak, and 0.6 of a
    121 GiB host leaves ~48 GiB unpinned.
    """
    from glq.supervisor import plan_arena_and_offload

    off, arena = plan_arena_and_offload(
        weights_bytes=_FN_TRUE, ple_offload_bytes=_PLE, expert_offload_bytes=_EXPERTS,
        vram_bytes=_L4, base_offload_gib=35, ram_bytes=int(121 * 1024**3),
        nontext_bytes=_NONTEXT, kv_bytes=window_kv_bytes(8192, 8))
    assert off >= 41, f"should reach (nearly) full offload, got {off}"
    assert arena > 0, "the whole point of deepening is that an arena results"


def test_the_BASE_offload_planner_is_untouched_by_the_arena_fraction():
    """★ The blast radius is the new path only. `plan_expert_offload_gib` keeps the 0.5 share,
    so every existing plan — every card, every checkpoint, arena or not — is unchanged."""
    from glq.supervisor import _PINNED_RAM_FRACTION, plan_expert_offload_gib

    assert _PINNED_RAM_FRACTION == 0.5
    gib = plan_expert_offload_gib(
        weights_bytes=_FN_TRUE, ple_offload_bytes=_PLE, expert_offload_bytes=_EXPERTS,
        vram_bytes=_L4, ram_bytes=int(121 * 1024**3), nontext_bytes=_NONTEXT,
        kv_bytes=window_kv_bytes(8192, 8))
    host_cap = int(121 * _PINNED_RAM_FRACTION) - 24       # minus the PLE table
    assert gib <= host_cap, f"base plan pinned past its own share: {gib} > {host_cap}"


def test_the_arena_fraction_still_scales_with_the_host():
    """It is a share, not a licence. A host with little RAM is still not asked to hold the
    experts — pinned pages cannot be swapped, and there is no swap on these boxes."""
    from glq.supervisor import plan_arena_and_offload

    off, _ = plan_arena_and_offload(
        weights_bytes=_FN_TRUE, ple_offload_bytes=_PLE, expert_offload_bytes=_EXPERTS,
        vram_bytes=_L4, base_offload_gib=35, ram_bytes=int(64 * 1024**3),
        nontext_bytes=_NONTEXT, kv_bytes=window_kv_bytes(8192, 8))
    assert off == 35, "a 64 GiB host cannot hold 66 GiB of pinned weights"


# ---------------------------------------------------------------------------
# ★ The auto window is capped by card size, so the arena stays useful
# ---------------------------------------------------------------------------

def test_the_window_ceiling_follows_the_card():
    """★ The auto window is chosen BEFORE the arena exists, so left alone it tiers up to the
    largest window the card can hold and leaves the arena whatever is over. Measured on a
    24 GiB card: 262,144 leaves a 1 GiB arena at ~11 slots/layer (hit ~0.33) where 131,072
    leaves 5 GiB at ~59 slots (~0.68). Traffic scales as (1 - hit), so that is roughly 4x the
    PCIe traffic, against a term that was 45% of the un-arena'd step.

    So the ceiling is indexed on VRAM. The thresholds are generous about context on purpose —
    these are the windows a user of each card should get, not the largest the arena would
    prefer:

        <= 20 GiB ->  32,768     (16 GiB cards, which report ~15.6)
        <= 28 GiB -> 131,072     (22.49 GiB L4 and 23.6 GiB 3090/4090-class)
        >  28 GiB -> 262,144     (31.4 GiB 5090-class and up, where no trade is needed)

    `--max-model-len` overrides it; this bounds only the automatic choice.
    """
    from glq.supervisor import window_ceiling_for_vram

    G = 1024**3
    assert window_ceiling_for_vram(int(15.6 * G)) == 32768
    assert window_ceiling_for_vram(int(22.49 * G)) == 131072     # the L4
    assert window_ceiling_for_vram(int(23.6 * G)) == 131072      # a 24 GB 3090/4090
    assert window_ceiling_for_vram(int(31.4 * G)) == 262144      # a 32 GB 5090
    assert window_ceiling_for_vram(int(95 * G)) == 262144


def test_an_unknown_card_is_not_capped():
    """Same stance as every other planner here: with the input unknown, do not invent a
    limit — `plan_max_model_len` already returns the floor when it cannot size anything."""
    from glq.supervisor import window_ceiling_for_vram

    assert window_ceiling_for_vram(None) >= 262144
    assert window_ceiling_for_vram(0) >= 262144


def test_the_ceiling_leaves_a_useful_arena_on_each_target_card():
    """★ The point of the ceiling, asserted as an outcome rather than as a threshold: on each
    card the capped window must leave an arena that can actually serve a token."""
    from glq.supervisor import plan_arena_and_offload, window_ceiling_for_vram
    from glq_vllm.moe_shapes import bytes_per_slot

    per = bytes_per_slot(hidden_size=2560, inter=640, w13_out=1280, bpw=3)
    G = 1024**3
    for vram, least_slots in ((22.49, 40), (24.0, 55), (31.4, 90)):
        win = window_ceiling_for_vram(int(vram * G))
        _off, arena = plan_arena_and_offload(
            weights_bytes=_FN_TRUE, ple_offload_bytes=_PLE, expert_offload_bytes=_EXPERTS,
            vram_bytes=int(vram * G), base_offload_gib=36, ram_bytes=int(121 * G),
            nontext_bytes=_NONTEXT, kv_bytes=window_kv_bytes(win, 1))
        slots = int(arena * G // (per * 49))
        assert slots >= least_slots, (
            f"{vram} GiB at window {win}: {arena} GiB is {slots} slots/layer, "
            f"expected >= {least_slots}")


def test_a_pinned_window_still_beats_the_ceiling():
    """`--max-model-len` is an instruction. Same contract as `--cpu-offload-gb`: honoured
    verbatim, never silently adjusted, and the shortfall reported if it does not fit."""
    sup = VllmSupervisor(model="m", device="cuda", vram_bytes=int(22.49 * 1024**3),
                         ram_bytes=int(121 * 1024**3), weights_bytes=_FN_TRUE,
                         ple_offload_bytes=_PLE, expert_offload_bytes=_EXPERTS,
                         nontext_bytes=_NONTEXT, max_model_len=262144,
                         window_concurrency=1, max_num_seqs=1, serve=False)
    assert sup.max_model_len == 262144


# ---------------------------------------------------------------------------
# ★ When host RAM is the binding constraint, say so
# ---------------------------------------------------------------------------

def test_a_host_too_small_to_pin_the_experts_is_diagnosed_as_such():
    """★ The advice has to be followable. On a 23 GiB card with 90 GiB of host RAM this
    checkpoint does not fit at all — the pinnable cap allows 21 GiB of experts, so ~28 GiB of
    weights stay resident on a 23 GiB card — and the old note blamed KV and said "raise
    --cpu-offload-gb", which host RAM makes impossible.

    Measured across hosts on a 23 GiB card: 60, 70, 80 and 90 GiB of RAM all collapse to the
    8192 floor and none of them can serve; 100 GiB and up serve a 131072 window. So the
    message must name **host RAM** and the remedies that exist — more RAM, swap so a larger
    pinned set is safe, or a smaller checkpoint — rather than a knob that is already at its
    ceiling.
    """
    import io
    out = io.StringIO()
    sup = _code_sup(23.0, ram_gib=90.0, **FN_CORRECTED)
    sup._out = out
    sup.argv()
    said = out.getvalue().lower()
    assert "may be pinned on a" in said, f"should name the pinnable cap: {said!r}"
    assert "90 gib host" in said, f"should name this host's size: {said!r}"
    assert "swap" in said, f"should offer swap as a remedy: {said!r}"
    # and it must NOT send the user at a knob the host cap has already pinned shut
    assert "raise --cpu-offload-gb" not in said


def test_a_host_with_room_says_nothing_about_RAM():
    """The diagnosis is for the case that cannot serve. A 124 GiB host serves a 131072 window
    on the same card, and must not be warned about anything."""
    import io
    out = io.StringIO()
    sup = _code_sup(23.0, ram_gib=124.0, **FN_CORRECTED)
    sup._out = out
    sup.argv()
    said = out.getvalue().lower()
    assert "swap" not in said and "host ram" not in said


# ---------------------------------------------------------------------------
# ★ The window is re-planned after deepening, so freed VRAM is not left idle
# ---------------------------------------------------------------------------

def test_the_window_claims_the_VRAM_that_deepening_freed():
    """★ An ordering gap. The window is chosen from the BASE offload's resident figure, then
    deepening moves more experts to host and frees VRAM the window can no longer claim.

    Measured across cards at 100-110 GiB of host RAM, the window came out at 8192 where the
    deepened footprint could carry 65,536 to 262,144 — up to 16x the context, from VRAM that
    was already free and otherwise sat idle. No extra PCIe is paid for it: the offload depth
    is already decided by then.

    `window_ceiling_for_vram` is what keeps this from eating the arena: the window rises only
    to the card's ceiling, and the arena takes the remainder.
    """
    sup = _code_sup(23.0, ram_gib=110.0, **FN_CORRECTED)
    from glq.supervisor import window_ceiling_for_vram

    assert sup.expert_offload_gib > sup.expert_offload_base_gib, "this case should deepen"
    assert sup.max_model_len == window_ceiling_for_vram(int(23.0 * 1024**3))
    assert sup.expert_arena_gib > 0, "the ceiling should leave the arena its share"


def test_replanning_never_SHORTENS_the_window():
    """It may only claim idle VRAM. A window already chosen is a promise, and the re-plan is
    an increase or nothing — otherwise a card would announce less context after an upgrade."""
    for ram in (100.0, 110.0, 124.0, 192.0):
        sup = _code_sup(23.0, ram_gib=ram, **FN_CORRECTED)
        base_window = plan_max_model_len(
            weights_bytes=int((77.52 - 23.84 - 4.69) * 1024**3)
            - sup.expert_offload_base_gib * 1024**3,
            vram_bytes=int(23.0 * 1024**3), model_max_len=262144, concurrency=1)
        assert sup.max_model_len >= min(base_window,
                                        window_ceiling_for_vram(int(23.0 * 1024**3)))


def test_a_pinned_window_is_not_replanned():
    """`--max-model-len` is an instruction; the re-plan touches only the automatic choice."""
    sup = _code_sup(23.0, ram_gib=110.0, max_model_len=16384, **FN_CORRECTED)
    assert sup.max_model_len == 16384
