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

from glq.supervisor import (VllmSupervisor, child_env,  # noqa: E402
                            plan_expert_offload_gib)

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


def _code_sup(vram_gib, **kw):
    """A glq-code-shaped plan: one stream, the coding floor."""
    return VllmSupervisor(model="org/flash-next", device="cuda", window_concurrency=1,
                          max_num_seqs=1, vram_bytes=int(vram_gib * GIB),
                          ram_bytes=int(124 * GIB), max_model_len_floor=16384, **kw)


def test_the_window_doubles_on_a_24gb_card_once_the_counts_are_right():
    """The payoff, on the card that motivated it and with the published numbers rather than a
    fixture: 16384 -> 32768 for the same checkpoint and the same policy.

    These were 32768 -> 65536 while the overhead allowance was a flat 4 GiB. That allowance was
    measurably 1.44 GiB short on this checkpoint and is now proportional, which costs this card
    one tier -- correctly, because the old pair was reachable only by under-reserving what vLLM
    then needed. A doubling either way; the base moved because the arithmetic got honest.

    Not 131072, which the raw resident figure suggests. `WEIGHT_FRACTION = 0.75` is satisfied
    at ~15 GiB resident on a 23 GiB card, so the planner stops offloading there; reaching the
    ~10 GiB that affords a longer window would mean deliberately offloading more experts, a
    PCIe-decode-speed trade this policy does not make on its own."""
    before = _code_sup(23.0, **FN_PUBLISHED)
    after = _code_sup(23.0, **FN_CORRECTED)
    assert before.max_model_len == 16384
    assert after.max_model_len == 32768


def test_the_same_fix_also_offloads_fewer_experts():
    """Both sides of the saving, and the other one is decode speed: 38 GiB of experts in host
    memory becomes 33 GiB, and every offloaded expert is paid for again on each token that
    routes to it."""
    before = _code_sup(23.0, **FN_PUBLISHED)
    after = _code_sup(23.0, **FN_CORRECTED)
    assert after.expert_offload_gib < before.expert_offload_gib


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
