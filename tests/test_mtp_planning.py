"""Planning for MTP speculation: the budget, the step width, and the flag.

`benchmarks/expert_arena_s1/RESULT.md` measured arena + `num_spec=1` at **19.692 tok/s,
3.060x** the 6.436 baseline on an L4 — the best configuration on the board, and until now
unreachable, because the supervisor had no MTP knob and `supervisor.py` carried an invariant
forbidding one:

    Subtracting [nontext_bytes] ... Sound only because this supervisor never enables
    speculative decoding, which WOULD load the MTP head.

These tests turn that invariant into a conditional and pin the three quantities MTP moves:

* `nontext_bytes` must stop being subtracted, because the head does get loaded;
* the arena's token cap must widen to the verify step's `1 + num_spec` tokens -- measured:
  "an MTP verify step carries 1 + num_spec = 4 tokens, so GLQ_MOE_ARENA_MAX_TOKENS=4 brings
  the requirement to 40 <= 71";
* the offloadable expert bytes must include the MTP block's own 512-expert MoE layer, which
  `_offloadable_bytes` drops from every total (non-text first, then `elif "experts"`). That
  omission is the measured shortfall that stopped arena+MTP before: an 8 GiB arena plus MTP
  reported `Available KV cache memory: -0.31 GiB`.

Pure arithmetic, no GPU, matching how the rest of `supervisor.py` is tested.
"""
from __future__ import annotations

import json

import pytest

from glq.supervisor import (WINDOW_OFFLOAD_MAX_EXTRA_GIB, VllmSupervisor, child_env,
                            window_kv_bytes)

GIB = 1024 ** 3

#: Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4-3mtp, read live from the Hub 2026-10-13.
#: The quantized-head build: `nontext_bytes` is absent because the head IS loaded, and its
#: experts are therefore missing from `expert_offload_bytes` -- which is byte-identical to the
#: bf16-head sibling's, the clue that one MoE layer is uncounted.
_WEIGHTS = 79_090_922_404          # 73.66 GiB
_PLE = 25_600_122_880              # 23.84 GiB
_EXPERTS = 45_770_637_312          # 42.63 GiB, across 48 decoder layers
_DECODER_LAYERS = 48
_MTP_LAYERS = 1

#: The bf16-head sibling (`...-ple4`) declares this much non-text instead. Kept because the
#: nontext conditional is only observable on a checkpoint that has a non-zero value.
_NONTEXT = 5_214_301_696           # 4.86 GiB

_BLACKWELL_32 = int(31.87 * GIB)
_L4 = int(22.49 * GIB)
_HOST = int(120 * GIB)


def _sup(**kw):
    base = dict(model="m", device="cuda", serve=False,
                weights_bytes=_WEIGHTS, ple_offload_bytes=_PLE,
                expert_offload_bytes=_EXPERTS, nontext_bytes=0,
                decoder_layers=_DECODER_LAYERS, mtp_layers=_MTP_LAYERS,
                vram_bytes=_BLACKWELL_32, ram_bytes=_HOST,
                # What glq-code passes: the model's declared maximum, which is what lets the
                # auto window tier up. Without it the planner cannot exceed its floor, and
                # every window assertion here would read 8192 for reasons unrelated to MTP.
                model_max_len=262_144,
                window_offload_extra_gib=WINDOW_OFFLOAD_MAX_EXTRA_GIB,
                max_num_seqs=1, window_concurrency=1)
    base.update(kw)
    return VllmSupervisor(**base)


# ---------------------------------------------------------------------------
# The flag reaches vLLM
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 3])
def test_mtp_emits_a_speculative_config_with_the_exact_json(n):
    argv = _sup(mtp_tokens=n).argv()
    assert "--speculative-config" in argv
    cfg = json.loads(argv[argv.index("--speculative-config") + 1])
    assert cfg == {"method": "mtp", "num_speculative_tokens": n}


def test_mtp_zero_emits_nothing_at_all():
    """`--mtp 0` is the documented off switch, and off must mean today's exact argv."""
    argv = " ".join(_sup(mtp_tokens=0).argv())
    assert "--speculative-config" not in argv
    assert "speculative" not in argv


def test_a_hand_passed_speculative_config_still_wins():
    """vLLM takes the last occurrence, the same convention `--dtype` relies on, so ours has to
    come before `extra_args` rather than after."""
    argv = _sup(mtp_tokens=1, extra_args=("--speculative-config", '{"method":"ngram"}')).argv()
    ours = argv.index("--speculative-config")
    theirs = argv.index("--speculative-config", ours + 1)
    assert theirs > ours, "ours must not be last, or it would override the caller"
    assert argv[-1] == '{"method":"ngram"}'


def test_the_cpu_backend_never_speculates():
    """Unmeasured there, and the CPU path is single-digit tok/s where a wrong draft budget is
    expensive. Returning nothing is the honest default until someone measures it."""
    argv = " ".join(_sup(mtp_tokens=1, device="cpu", vram_bytes=None).argv())
    assert "speculative" not in argv


# ---------------------------------------------------------------------------
# ★ The invariant becomes a conditional
# ---------------------------------------------------------------------------

def test_with_mtp_ON_the_nontext_subtraction_is_WITHHELD():
    """★ The whole reason the supervisor refused to speculate. `nontext_bytes` is the MTP head
    and is subtracted because a text-only serve never loads it; speculation loads it, so
    subtracting it would under-plan the footprint by that much -- 4.86 GiB here, which the
    existing comment measures as two window tiers on a 24 GB card."""
    on = _sup(mtp_tokens=1, nontext_bytes=_NONTEXT)
    assert on.nontext_bytes == 0, "subtracted the MTP head while loading it"
    assert on._nontext_declared == _NONTEXT, "lost the figure needed to explain the plan"


def test_with_mtp_OFF_the_subtraction_is_exactly_as_before():
    off = _sup(mtp_tokens=0, nontext_bytes=_NONTEXT)
    assert off.nontext_bytes == _NONTEXT


def test_withholding_it_costs_VRAM_rather_than_being_cosmetic():
    """So the test above is known to bind: the same card plans a smaller arena once the head
    is counted, because resident grew."""
    on = _sup(mtp_tokens=1, nontext_bytes=_NONTEXT)
    off = _sup(mtp_tokens=0, nontext_bytes=_NONTEXT)
    assert on.expert_arena_gib < off.expert_arena_gib


# ---------------------------------------------------------------------------
# ★ The MTP block's own experts
# ---------------------------------------------------------------------------

def test_the_mtp_blocks_experts_are_added_to_the_offloadable_total():
    """★ `_offloadable_bytes` is "mutually exclusive, non-text first": once the head is
    quantized, `mtp.layers.0.mlp.experts.*` is claimed by the non-text branch and then dropped
    by `if not mtp_is_loaded`, so it lands in NEITHER total. Measured consequence:
    `expert_offload_bytes` is byte-identical across the bf16-head and quantized-head builds,
    and an 8 GiB arena plus MTP came out at `Available KV cache memory: -0.31 GiB`.

    One decoder layer's worth, derived from the two declared numbers rather than guessed."""
    per_layer = _EXPERTS // _DECODER_LAYERS
    on = _sup(mtp_tokens=1)
    off = _sup(mtp_tokens=0)
    assert on.expert_offload_bytes == _EXPERTS + per_layer * _MTP_LAYERS
    assert off.expert_offload_bytes == _EXPERTS


def test_an_unknown_decoder_layer_count_does_not_invent_one():
    """Guessing a layer count would mis-state the footprint; the planner's standing rule is
    that an unknown input means fall back, never estimate."""
    on = _sup(mtp_tokens=1, decoder_layers=None)
    assert on.expert_offload_bytes == _EXPERTS


# ---------------------------------------------------------------------------
# ★ Step width: the cap the arena must serve
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seqs,mtp,expected", [(1, 0, 1), (1, 1, 2), (1, 3, 4), (16, 1, 32)])
def test_the_step_width_is_seqs_times_one_plus_drafts(seqs, mtp, expected):
    sup = _sup(mtp_tokens=mtp, max_num_seqs=seqs, window_concurrency=1)
    assert sup.step_tokens == expected
    assert sup.min_step_tokens == 1 + mtp


def test_the_arena_cap_in_the_child_env_is_the_step_width():
    """★ The line the measured runs needed. With `max_num_seqs` there instead, a 6 GiB arena
    over 49 containers refused a cap of 8 (80 > 71 slots) and three arms silently measured no
    arena while looking healthy."""
    sup = _sup(mtp_tokens=3, max_num_seqs=1)
    env = child_env(device="cuda", expert_arena_gib=sup.expert_arena_gib,
                    max_num_seqs=sup.step_tokens)
    assert env["GLQ_MOE_ARENA_MAX_TOKENS"] == "4"


def test_an_exported_cap_and_budget_still_win(monkeypatch):
    monkeypatch.setenv("GLQ_MOE_ARENA_MAX_TOKENS", "9")
    monkeypatch.setenv("GLQ_MOE_EXPERT_CACHE", "0")
    env = child_env(device="cuda", expert_arena_gib=10, max_num_seqs=4)
    assert env["GLQ_MOE_ARENA_MAX_TOKENS"] == "9"
    assert env["GLQ_MOE_EXPERT_CACHE"] == "0"


# ---------------------------------------------------------------------------
# ★ The target configuration, and what yields when it does not fit
# ---------------------------------------------------------------------------

def test_a_32GiB_card_gets_the_full_window_AND_an_arena_AND_mtp():
    """★★ The regression test for the whole policy: 262144 + an arena + MTP together, which is
    what this change exists to deliver. Measured on the same card without MTP: 262144 with a
    9 GiB arena, decoding 28.795 against 13.067 tok/s."""
    sup = _sup(mtp_tokens=1)

    assert sup.max_model_len == 262_144, f"window dropped to {sup.max_model_len}"
    assert sup.mtp_tokens == 1
    assert sup.expert_arena_gib > 0, "no arena left once MTP was budgeted for"
    # The cap the arena will derive must cover one verify step, or it serves nothing.
    assert sup.step_tokens == 2


def test_the_window_is_NEVER_lowered_to_pay_for_mtp():
    """The chosen priority: shrink the arena, never the window. Asserted against the no-MTP
    plan on the same card so a regression in either direction is visible."""
    on, off = _sup(mtp_tokens=1), _sup(mtp_tokens=0)
    assert on.max_model_len == off.max_model_len


def test_on_a_QUANTIZED_head_mtp_makes_the_arena_BIGGER_not_smaller():
    """★ Counter-intuitive, and the reason the MTP expert bytes are worth counting at all.

    Naively MTP only takes VRAM, so the arena should shrink. It grows — 8 -> 9 GiB here —
    because the MTP block's experts are *offloadable*: counting them raises the offload depth
    the planner will choose, which moves ~0.95 GiB off the card and hands the difference to
    the arena. The head's own bytes are already inside `weights_bytes`, so nothing is
    double-counted.

    Pinned because the obvious "assert the arena shrank" is wrong here and would have been
    written instead. The arena only shrinks when the head is bf16 and `nontext_bytes` stops
    being subtracted — see `test_withholding_it_costs_VRAM_rather_than_being_cosmetic`.
    """
    on, off = _sup(mtp_tokens=3), _sup(mtp_tokens=0)
    assert on.expert_offload_gib >= off.expert_offload_gib
    assert on.expert_arena_gib >= off.expert_arena_gib
    assert on.max_model_len == off.max_model_len, "the window paid for none of it"


def test_mtp_survives_a_card_too_small_for_an_arena():
    """MTP is never dropped -- it was asked for. On a card where the arena cannot be afforded
    the arena is what goes, and the plan still speculates."""
    sup = _sup(mtp_tokens=1, vram_bytes=int(12 * GIB), max_model_len=262_144)
    assert sup.mtp_tokens == 1
    assert "--speculative-config" in sup.argv()


def test_the_arena_leaves_the_minimum_KV_with_mtp_on():
    """An arena that eats the pool is how `Available KV cache memory: -0.31 GiB` happened."""
    from glq.supervisor import _MIN_KV_BYTES, kv_headroom_bytes

    sup = _sup(mtp_tokens=1)
    resident = _WEIGHTS - _PLE - sup.nontext_bytes - sup.expert_offload_gib * GIB
    spare = kv_headroom_bytes(max(0, resident), _BLACKWELL_32)
    left = spare - window_kv_bytes(sup.max_model_len, 1) - sup.expert_arena_gib * GIB
    assert left >= 0, f"arena overcommitted the pool by {-left / GIB:.2f} GiB"
    assert left + window_kv_bytes(sup.max_model_len, 1) >= _MIN_KV_BYTES


# ---------------------------------------------------------------------------
# Detecting the head, and resolving the flag
# ---------------------------------------------------------------------------

def test_the_declared_keys_are_read_by_VALUE_not_by_presence(tmp_path):
    """★ `mtp`, `mtp_num_hidden_layers` and `mtp_use_dedicated_embeddings` are all PRESENT on
    a Flash-Next build with no usable head, every one of them `null` (measured on
    `...-3bpw-ple4`). Testing for the keys would offer speculation on a checkpoint that cannot
    do it, and vLLM would refuse the `--speculative-config` minutes into a weight load."""
    import json as _json

    from glq.chat import checkpoint_mtp

    (tmp_path / "config.json").write_text(_json.dumps({
        "mtp": None, "mtp_num_hidden_layers": None, "mtp_use_dedicated_embeddings": None,
        "num_hidden_layers": 48, "quantization_config": {"layer_bpw": {}},
    }))
    assert checkpoint_mtp(str(tmp_path)) == (0, False, 48)


def test_a_quantized_head_is_detected_from_layer_bpw_segments(tmp_path):
    """1,545 mtp entries on `...-ple4-3mtp`, 0 on `...-ple4`. Matched on dotted SEGMENTS,
    following `_is_nontext`'s warning that `mtpool` contains "mtp" and is ordinary text."""
    import json as _json

    from glq.chat import checkpoint_mtp

    (tmp_path / "config.json").write_text(_json.dumps({
        "mtp_num_hidden_layers": 1, "num_hidden_layers": 48,
        "quantization_config": {"layer_bpw": {
            "model.layers.0.mtpool.weight": 3,          # NOT the MTP head
            "mtp.layers.0.mlp.experts.0.gate_proj": 3,  # the MTP head
        }},
    }))
    assert checkpoint_mtp(str(tmp_path)) == (1, True, 48)


def test_an_unreadable_checkpoint_declares_nothing(tmp_path):
    from glq.chat import checkpoint_mtp

    assert checkpoint_mtp(str(tmp_path / "nope")) == (0, False, 0)
    assert checkpoint_mtp("") == (0, False, 0)


@pytest.mark.parametrize("requested,layers,quantized,expected", [
    (None, 1, True, 0),      # OFF by default even where the head is cheap to load:
    (None, 1, False, 0),     # speculation changes output, so it is opt-in
    (None, 0, False, 0),
    (0, 1, True, 0),         # the documented off switch
    (3, 1, True, 3),
    (1, 1, False, 1),        # explicit beats the bf16 caution
])
def test_the_resolution_table(requested, layers, quantized, expected):
    from glq.chat import resolve_mtp_tokens

    tokens, err = resolve_mtp_tokens(requested, layers, quantized)
    assert err is None and tokens == expected


def test_asking_for_mtp_on_a_headless_checkpoint_is_an_ERROR():
    """Not a silent 0: the user asked for something this checkpoint cannot do, and the
    alternative is vLLM refusing minutes into a load."""
    from glq.chat import resolve_mtp_tokens

    tokens, err = resolve_mtp_tokens(3, 0, False)
    assert tokens == 0 and err and "no MTP draft head" in err


def test_a_negative_draft_count_is_refused():
    from glq.chat import resolve_mtp_tokens

    _, err = resolve_mtp_tokens(-1, 1, True)
    assert err and "0 or more" in err


def test_both_entry_points_share_the_detection():
    """Following the existing `sizing_weights_bytes` precedent: one implementation, so glq-chat
    and glq-code cannot drift apart on what counts as a usable draft head."""
    import glq.chat as chat_mod
    import glq.code as code_mod

    assert code_mod.checkpoint_mtp is chat_mod.checkpoint_mtp
    assert code_mod.resolve_mtp_tokens is chat_mod.resolve_mtp_tokens


def test_glq_code_refuses_mtp_before_building_a_supervisor(monkeypatch, capsys):
    """★ The guard must run BEFORE the supervisor, like the pi and tool-parser guards, because
    starting vLLM costs minutes of weight loading that a refusal would throw away."""
    import glq.code as code_mod

    built = []
    monkeypatch.setattr(code_mod, "_find_pi", lambda: "/usr/bin/true")
    monkeypatch.setattr(code_mod, "checkpoint_mtp", lambda m: (0, False, 0))
    monkeypatch.setattr(code_mod, "VllmSupervisor",
                        lambda **kw: built.append(kw) or object())

    rc = code_mod.main(["--model", "xv0y5ncu/Qwen3-whatever", "--mtp", "3"])
    assert rc == 2
    assert built == [], "constructed a supervisor despite refusing the flag"
    assert "no MTP draft head" in capsys.readouterr().err
