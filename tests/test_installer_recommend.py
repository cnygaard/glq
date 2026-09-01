"""VRAM-aware checkpoint recommendation (glq/installer/recommend.py).

The installer prompts the user, but a bare list of nine repo ids is not a choice anyone can
make — the whole value is marking which ones the card in front of them can actually run.

Two sizing facts drive the arithmetic, and both push the same way:

  * the Hub tree API reports **on-disk** bytes (the 31B is 22.4 GiB) while vLLM's resident
    footprint for the same checkpoint is ~16.5 GiB, so disk over-states what VRAM needs;
  * weights are not the only resident tensor — the KV cache and activations come out of the
    same card, and vLLM defaults to `gpu_memory_utilization=0.9`.

So the rule reserves a fraction of VRAM for non-weight memory and compares disk bytes
against what's left. That is deliberately conservative in both directions: it may pass over
a checkpoint that would in fact have fit, which is the safe way to be wrong — the opposite
error is an OOM several GiB into a download.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.installer import recommend as R  # noqa: E402
from glq.installer.discovery import Checkpoint  # noqa: E402

GIB = 1024 ** 3

# The real collection, sized 2026-08-15.
FLEET = [
    Checkpoint("xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel", int(1.8 * GIB)),
    Checkpoint("xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw", int(15.0 * GIB)),
    Checkpoint("xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8", int(22.4 * GIB)),
    Checkpoint("xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw", int(5.8 * GIB)),
    Checkpoint("xv0y5ncu/SmolLM2-360M-Instruct-GLQ-block-diagonal-4bpw", int(0.24 * GIB)),
]


def _rec(vram_gib):
    return R.rank(FLEET, vram_bytes=int(vram_gib * GIB))


def test_biggest_that_fits_is_recommended_on_a_96gb_card():
    """The Blackwell box this was built on. Nothing here strains it, so the largest wins."""
    ranked = _rec(95.6)
    picked = [r for r in ranked if r.recommended]
    assert len(picked) == 1
    assert picked[0].checkpoint.repo_id == "xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8"


def test_a_24gb_card_does_not_get_the_31b():
    """22.4 GiB of weights on a 24 GiB card leaves nothing for the KV cache. Recommending
    it would OOM after a 22 GiB download — the worst failure this module can produce."""
    ranked = _rec(24.0)
    picked = [r for r in ranked if r.recommended][0]
    assert picked.checkpoint.repo_id != "xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8"
    assert picked.checkpoint.size_bytes <= 15.0 * GIB


def test_an_8gb_card_gets_something_small_enough_for_a_kv_cache():
    ranked = _rec(8.0)
    picked = [r for r in ranked if r.recommended][0]
    assert picked.checkpoint.size_bytes < 8.0 * GIB
    fits = {r.checkpoint.repo_id for r in ranked if r.fits}
    assert "xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8" not in fits


def test_a_tiny_card_still_gets_one_option():
    """A 4 GiB laptop GPU is exactly who the 360M checkpoint is published for."""
    ranked = _rec(4.0)
    assert [r for r in ranked if r.fits], "something must fit a 4 GiB card"
    picked = [r for r in ranked if r.recommended][0]
    assert picked.checkpoint.size_bytes <= 1.8 * GIB


def test_every_checkpoint_is_listed_even_when_it_does_not_fit():
    """The prompt shows the whole collection; not-fitting ones are marked, never hidden —
    a user with a second card may still want one."""
    ranked = _rec(4.0)
    assert len(ranked) == len(FLEET)
    assert any(not r.fits for r in ranked)


def test_ordering_is_largest_first_so_the_capable_models_read_first():
    ranked = _rec(95.6)
    sizes = [r.checkpoint.size_bytes for r in ranked]
    assert sizes == sorted(sizes, reverse=True)


def test_unknown_vram_recommends_nothing_but_still_lists():
    """No nvidia-smi (CPU box, or a container without the device) must not crash the
    installer, and must not bluff a recommendation it cannot justify."""
    ranked = R.rank(FLEET, vram_bytes=None)
    assert len(ranked) == len(FLEET)
    assert not any(r.recommended for r in ranked)
    assert all(r.fits is None for r in ranked)


def test_zero_size_checkpoints_are_never_recommended():
    """size 0 means the tree API gave us nothing usable; treating that as 'fits anywhere'
    would put a broken repo at the top of the list."""
    fleet = [Checkpoint("xv0y5ncu/mystery", 0)]
    ranked = R.rank(fleet, vram_bytes=8 * GIB)
    assert not ranked[0].recommended


def test_nothing_fits_is_reported_rather_than_guessed():
    fleet = [Checkpoint("xv0y5ncu/huge", 400 * GIB)]
    ranked = R.rank(fleet, vram_bytes=8 * GIB)
    assert not any(r.recommended for r in ranked)
    assert ranked[0].fits is False


def test_headroom_is_a_named_constant_not_a_magic_number():
    """This fraction is the single tuning knob between 'OOM on load' and 'needlessly
    conservative'; it must be inspectable and documented."""
    assert 0.5 < R.WEIGHT_FRACTION < 1.0


# --------------------------------------------------------- trellis preference
# "Biggest that fits" is the wrong objective on its own. A trellis checkpoint decodes
# single-stream at bf16 parity while the shell/e8p formats are materially slower, so a
# 13.9 GiB trellis model is a better recommendation than a 22.4 GiB non-trellis one even
# though both fit. Size only breaks ties *within* a format.

TRELLIS_FLEET = [
    Checkpoint("xv0y5ncu/gemma-4-26B-trellis-3inst-4bpw", int(13.9 * GIB), trellis=True),
    Checkpoint("xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8", int(22.4 * GIB), trellis=False),
    Checkpoint("xv0y5ncu/Devstral-Small-2-24B-GLQ-4bpw", int(19.1 * GIB), trellis=False),
    Checkpoint("xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw", int(1.8 * GIB), trellis=True),
]


def test_a_trellis_checkpoint_wins_over_a_larger_non_trellis_one():
    """The case that prompted this: on a 96 GiB card the old rule recommended the 22.4 GiB
    non-trellis 31B over the faster-decoding 13.9 GiB trellis MoE."""
    picked = [r for r in R.rank(TRELLIS_FLEET, int(95.6 * GIB)) if r.recommended]
    assert len(picked) == 1
    assert picked[0].checkpoint.repo_id == "xv0y5ncu/gemma-4-26B-trellis-3inst-4bpw"


def test_largest_trellis_that_fits_wins_among_trellis():
    """Within the preferred format, size still decides — a bigger model is more capable."""
    picked = [r for r in R.rank(TRELLIS_FLEET, int(95.6 * GIB)) if r.recommended][0]
    assert picked.checkpoint.size_bytes == int(13.9 * GIB)


def test_non_trellis_is_recommended_only_when_no_trellis_fits():
    """A 3 GiB card cannot hold the 13.9 GiB trellis MoE; the 1.8 GiB trellis one still
    fits, so it should still win. Sanity that the fallback is not reached too eagerly."""
    picked = [r for r in R.rank(TRELLIS_FLEET, int(4.0 * GIB)) if r.recommended][0]
    assert picked.checkpoint.repo_id == "xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw"


def test_falls_back_to_non_trellis_when_nothing_trellis_fits():
    fleet = [
        Checkpoint("xv0y5ncu/huge-trellis", int(80 * GIB), trellis=True),
        Checkpoint("xv0y5ncu/small-shell", int(2 * GIB), trellis=False),
    ]
    picked = [r for r in R.rank(fleet, int(8 * GIB)) if r.recommended][0]
    assert picked.checkpoint.repo_id == "xv0y5ncu/small-shell"


def test_unknown_trellis_status_does_not_beat_a_confirmed_one():
    """trellis=None means the config could not be read. It must not outrank a checkpoint
    we positively know is trellis, or a network blip changes the recommendation."""
    fleet = [
        Checkpoint("xv0y5ncu/unknown-big", int(20 * GIB), trellis=None),
        Checkpoint("xv0y5ncu/known-trellis", int(10 * GIB), trellis=True),
    ]
    picked = [r for r in R.rank(fleet, int(95.6 * GIB)) if r.recommended][0]
    assert picked.checkpoint.repo_id == "xv0y5ncu/known-trellis"


def test_trellis_flag_is_exposed_for_the_menu():
    """The prompt labels it so the user can see why it was preferred."""
    ranked = R.rank(TRELLIS_FLEET, int(95.6 * GIB))
    by_id = {r.checkpoint.repo_id: r for r in ranked}
    assert by_id["xv0y5ncu/gemma-4-26B-trellis-3inst-4bpw"].checkpoint.trellis is True
    assert by_id["xv0y5ncu/Devstral-Small-2-24B-GLQ-4bpw"].checkpoint.trellis is False


def test_ordering_still_lists_largest_first():
    """Preference changes what is *recommended*, not the reading order of the list."""
    sizes = [r.checkpoint.size_bytes for r in R.rank(TRELLIS_FLEET, int(95.6 * GIB))]
    assert sizes == sorted(sizes, reverse=True)


def test_old_style_checkpoints_without_the_flag_still_work():
    """Checkpoint(repo, size) with no trellis argument must keep working — the field is
    additive, and discovery may not have been able to determine it."""
    c = Checkpoint("xv0y5ncu/legacy", int(2 * GIB))
    assert c.trellis is None
    assert len(R.rank([c], int(95.6 * GIB))) == 1


# ---- per-command family preference: glq-code wants Qwen, glq-chat wants gemma-4 --------
# Rationale (measured, 2026-08): Qwen3.8's tool calling is native hermes markup — no
# external template, no thought-markup leak risk — and its GLQ-4bpw AIME ties bf16, which
# is what a coding agent needs. gemma-4's 26B-A4B MoE decodes fastest for interactive chat.

QWEN27 = "xv0y5ncu/Qwen3.8-27B-GLQ-trellis-3inst-4bpw"
GEMMA26 = "xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw"

FAMILY_FLEET = [
    Checkpoint("xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8", int(22.4 * GIB), trellis=True),
    Checkpoint(QWEN27, int(16.7 * GIB), trellis=True),
    Checkpoint(GEMMA26, int(15.0 * GIB), trellis=True),
    Checkpoint("xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel", int(1.8 * GIB), trellis=True),
]


def test_prefer_family_qwen_beats_a_larger_gemma():
    """On a big card the 31B would win on size; prefer_family='qwen' must pick the Qwen."""
    ranked = R.rank(FAMILY_FLEET, int(95.6 * GIB), prefer_family="qwen")
    picked = [r for r in ranked if r.recommended][0]
    assert picked.checkpoint.repo_id == QWEN27


def test_prefer_family_gemma4_picks_the_moe_not_the_qwen():
    ranked = R.rank(FAMILY_FLEET, int(95.6 * GIB), prefer_family="gemma-4")
    picked = [r for r in ranked if r.recommended][0]
    # Largest fitting gemma-4 wins within the family.
    assert picked.checkpoint.repo_id == "xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8"


def test_family_preference_never_recommends_what_does_not_fit():
    """A 16 GiB card (12 GiB weight budget) cannot run the 16.7 GiB Qwen; the preference
    must fall back to the plain recommendation, not force an OOM download."""
    plain = [r for r in R.rank(FAMILY_FLEET, int(16 * GIB)) if r.recommended][0]
    fam = [r for r in R.rank(FAMILY_FLEET, int(16 * GIB), prefer_family="qwen")
           if r.recommended][0]
    assert fam.checkpoint.repo_id == plain.checkpoint.repo_id


def test_no_family_argument_is_byte_identical_to_before():
    assert R.rank(FAMILY_FLEET, int(24 * GIB)) == R.rank(FAMILY_FLEET, int(24 * GIB),
                                                         prefer_family=None)


def test_per_command_picks_assigns_qwen_to_code_and_gemma_to_chat():
    picks = R.per_command_picks(FAMILY_FLEET, int(24 * GIB), fallback=GEMMA26)
    assert picks["code_model"] == QWEN27
    assert picks["chat_model"] == GEMMA26   # 31B does not fit 24 GiB; 26B does


def test_per_command_picks_falls_back_when_vram_is_unknown():
    """No nvidia-smi → nothing is recommended → both commands serve the generic choice."""
    picks = R.per_command_picks(FAMILY_FLEET, None, fallback=GEMMA26)
    assert picks == {"code_model": GEMMA26, "chat_model": GEMMA26}


# ---- CPU gating: RAM budget fraction + trellis-only + no MoE -----------------------------

CPU_FLEET = [
    Checkpoint("xv0y5ncu/gemma-4-26B-A4B-moe-4bpw", int(15.0 * GIB), trellis=True,
               moe=True),
    Checkpoint("xv0y5ncu/dense-9B-trellis-4bpw", int(5.6 * GIB), trellis=True, moe=False),
    Checkpoint("xv0y5ncu/dense-3B-trellis-4bpw", int(1.8 * GIB), trellis=True, moe=False),
    Checkpoint("xv0y5ncu/old-e8p-4bpw", int(1.7 * GIB), trellis=False, moe=False),
    Checkpoint("xv0y5ncu/unknown-traits", int(1.6 * GIB)),
]


def test_weight_fraction_changes_the_budget():
    """16 GiB at the CPU fraction (0.5) affords 8 GiB of weights; at the GPU 0.75 it
    would afford 12. The 9B (5.6 GiB) fits either; a 10 GiB entry only the latter."""
    fleet = [Checkpoint("xv0y5ncu/ten-gib", int(10 * GIB), trellis=True, moe=False)]
    gpu = R.rank(fleet, int(16 * GIB))
    cpu = R.rank(fleet, int(16 * GIB), weight_fraction=R.CPU_WEIGHT_FRACTION)
    assert gpu[0].fits is True
    assert cpu[0].fits is False


def test_cpu_gate_excludes_moe_and_non_trellis_and_unknown():
    ranked = R.rank(CPU_FLEET, int(32 * GIB), weight_fraction=R.CPU_WEIGHT_FRACTION,
                    require_trellis=True)
    picked = [r for r in ranked if r.recommended]
    assert len(picked) == 1
    assert picked[0].checkpoint.repo_id == "xv0y5ncu/dense-9B-trellis-4bpw"


def test_cpu_gate_recommends_nothing_rather_than_an_unservable_model():
    """A fleet of only MoE/e8p/unknown entries must yield NO recommendation on CPU —
    the menu still lists them, but recommending one would download gigabytes into a
    hard refusal at serve time."""
    fleet = [c for c in CPU_FLEET if c.repo_id != "xv0y5ncu/dense-9B-trellis-4bpw"
             and c.repo_id != "xv0y5ncu/dense-3B-trellis-4bpw"]
    ranked = R.rank(fleet, int(32 * GIB), weight_fraction=R.CPU_WEIGHT_FRACTION,
                    require_trellis=True)
    assert not any(r.recommended for r in ranked)


def test_per_command_picks_under_cpu_gating_never_return_moe():
    picks = R.per_command_picks(CPU_FLEET, int(32 * GIB),
                                fallback="xv0y5ncu/dense-3B-trellis-4bpw",
                                weight_fraction=R.CPU_WEIGHT_FRACTION,
                                require_trellis=True)
    for model in picks.values():
        assert "moe" not in model
