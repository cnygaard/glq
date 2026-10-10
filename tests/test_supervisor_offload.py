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
    assert _sup(45.0).expert_offload_gib < _sup(23.0).expert_offload_gib


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
    before = _code_sup(23.0, **FN_PUBLISHED)
    after = _code_sup(23.0, **FN_CORRECTED)
    assert before.max_model_len == 16384
    assert after.max_model_len == 131072
    assert after.expert_offload_gib < before.expert_offload_gib


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
    planned = _code_sup(23.0, **FN_CORRECTED)
    forced = _code_sup(23.0, expert_offload_gib=42, **FN_CORRECTED)
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
                         **FN_CORRECTED).expert_offload_gib
    assert chat_shaped.expert_offload_gib > baseline


def test_an_unaffordable_pinned_window_is_served_anyway_and_reported():
    """`_KV_BYTES_PER_TOKEN` is a conservative envelope over the worst pool ever observed, so
    a window this arithmetic calls unaffordable may still serve -- and vLLM's own startup check
    is the authoritative one. Clamping would shorten windows that would have worked, so the
    pinned value is passed through untouched and the shortfall is REPORTED.

    Same contract as `--cpu-offload-gb 999`: honored verbatim, never silently adjusted."""
    import io
    out = io.StringIO()
    sup = _code_sup(23.0, ram_gib=124.0, max_model_len=262144, out=out, **FN_CORRECTED)
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
    assert sup.expert_offload_gib == 37
    assert sup.max_model_len == 131072


def test_the_auto_window_reaches_the_full_context_on_a_32gb_card():
    """The g7, where the trade is bracketed by measurement rather than extrapolated: the
    chosen 32 GiB sits between the 27 GiB and 34 GiB arms that were benched."""
    sup = _code_sup(31.9, **FN_CORRECTED)
    assert sup.expert_offload_gib == 32
    assert sup.max_model_len == 262144


def test_it_picks_the_smallest_offload_that_reaches_the_tier():
    """Walking the tiers downward is what makes this land on the sweet spot instead of the
    declared maximum. On the g7, 32 GiB and 38 GiB both serve 262144; every GiB beyond the
    first one that reaches the tier is decode speed spent for nothing."""
    sup = _code_sup(31.9, **FN_CORRECTED)
    lower = _code_sup(31.9, expert_offload_gib=sup.expert_offload_gib - 1, **FN_CORRECTED)
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

    Same card, same checkpoint, more host RAM -- and only the one with RAM to pin reaches
    262144."""
    assert _code_sup(23.0, ram_gib=124.0, **FN_CORRECTED).max_model_len == 131072
    assert _code_sup(23.0, ram_gib=192.0, **FN_CORRECTED).max_model_len == 262144


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
