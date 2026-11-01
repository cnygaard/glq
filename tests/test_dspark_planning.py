"""Planning for DSpark speculation: the drafter, the verify step's arena, and the flag.

DSpark is a separate small draft model (`"method": "dspark"` in vLLM), unlike MTP, whose head
lives inside the checkpoint. Measured with GLQ's Nemotron-3.5-Super-VL 3 bpw build on a
21.96 GiB budget (RTX PRO 4500 capped, PCIe 5, vLLM 0.32.0, `benchmarks/supervl3_spec_24gb/`):
31.50 tok/s per request at 4 draft tokens against 20.15 without, with the expert arena on.

These tests pin what the planner must get right for that to start at all:

* the drafter is a second model on the card, so its bytes are resident;
* a verify step routes `(1 + k) x top_k` experts per MoE layer at once, and the arena refuses
  a step it cannot hold, so the depth is chosen against the arena (`arena_floor_gib`);
* `--dspark` alone (auto) never costs the window: planned as if DSpark were off, and kept
  only if the drafter fits in what is left. `--dspark N` (pinned) is an instruction.

Pure arithmetic, no GPU, matching `test_mtp_planning.py`.
"""
from __future__ import annotations

import json

import pytest

from glq.supervisor import (WINDOW_OFFLOAD_MAX_EXTRA_GIB, VllmSupervisor, child_env,
                            kv_cost, runtime_overhead_bytes, window_ceiling_for_vram,
                            window_kv_bytes)

GIB = 1024 ** 3

SV = "xv0y5ncu/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-3bpw"
DRAFTER = "RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B.dspark"
#: Read live from the Hub 2026-11-01, the way the launchers size them (`repo_size_bytes`,
#: `offload_bytes`). The offload figures are the corrected ones uploaded that day, with the
#: 5.48 GiB MTP head declared non-text; see `test_auto_uses_the_checkpoints_declared_figures`.
_WEIGHTS = 55_038_723_416          # 51.26 GiB
_EXPERTS = 43_244_953_920          # 40.28 GiB
_NONTEXT = 5_884_651_520           # 5.48 GiB
_DRAFT = 2_420_229_602             # 2.25 GiB
#: What config.json declared before that upload: no non-text bytes, and the MTP head's bf16
#: experts (and its shared expert, 84 MiB) counted as offloadable.
_EXPERTS_BEFORE = 48_970_178_880   # 45.61 GiB
#: From config.json: num_experts_per_tok, the "moe" count in layers_block_type, and
#: `moe_shapes.bytes_per_slot` for the 1024-wide latent, inter 2688, relu2 (not gated), 3 bpw.
_TOPK, _MOE_LAYERS, _SLOT = 22, 40, 2_071_828


def _sup(vram_gib, ram_gib=128.0, **kw):
    base = dict(model=SV, device="cuda", serve=False, weights_bytes=_WEIGHTS,
                expert_offload_bytes=_EXPERTS, nontext_bytes=_NONTEXT,
                vram_bytes=int(vram_gib * GIB),
                ram_bytes=int(ram_gib * GIB), model_max_len=1_048_576,
                max_model_len_floor=16384, max_num_seqs=1, window_concurrency=1,
                window_offload_extra_gib=WINDOW_OFFLOAD_MAX_EXTRA_GIB,
                moe_top_k=_TOPK, moe_layers=_MOE_LAYERS, arena_slot_bytes=_SLOT)
    base.update(kw)
    return VllmSupervisor(**base)


def _pinned(k, vram_gib=24.0, **kw):
    return _sup(vram_gib, draft_model=DRAFTER, draft_tokens=k, draft_bytes=_DRAFT, **kw)


def _auto(vram_gib, **kw):
    return _sup(vram_gib, draft_model=DRAFTER, draft_tokens=4, draft_auto=True,
                draft_bytes=_DRAFT, **kw)


def _plan(sup):
    return (sup.max_model_len, sup.expert_offload_gib, float(sup.expert_arena_gib),
            sup.gpu_memory_utilization)


def _spec(sup):
    argv = sup.argv()
    if "--speculative-config" not in argv:
        return None
    return json.loads(argv[argv.index("--speculative-config") + 1])


# ---------------------------------------------------------------------------
# The flag reaches vLLM
# ---------------------------------------------------------------------------

def test_dspark_emits_the_exact_speculative_config():
    assert _spec(_pinned(3)) == {"method": "dspark", "model": DRAFTER,
                                 "num_speculative_tokens": 3}


def test_the_drafter_never_gets_remote_code():
    """The drafter repo carries a `config.py`; vLLM's own `dspark` method loads it without
    one, and `--trust-remote-code` runs Python from the Hub on the user's machine."""
    assert "--trust-remote-code" not in _pinned(3).argv()


def test_a_hand_passed_speculative_config_still_wins():
    """vLLM takes the last occurrence, so ours must come before `extra_args`."""
    argv = _pinned(3, extra_args=["--speculative-config", "{}"]).argv()
    idx = [i for i, a in enumerate(argv) if a == "--speculative-config"]
    assert len(idx) == 2 and argv[idx[-1] + 1] == "{}"


def test_mtp_and_dspark_together_are_refused():
    with pytest.raises(ValueError, match="exclusive"):
        _pinned(3, mtp_tokens=1)


def test_the_cpu_backend_never_speculates():
    sup = _pinned(3, device="cpu")
    assert sup.spec_tokens == 0 and _spec(sup) is None


# ---------------------------------------------------------------------------
# The drafter is resident, and a verify step needs its arena
# ---------------------------------------------------------------------------

def test_the_drafter_counts_as_resident():
    """Same offload either way (pinned), so the difference is the drafter alone. `fit` reads
    `resident_bytes`, so the pre-flight check sees it too."""
    without = _sup(24.0, expert_offload_gib=45)
    with_ = _pinned(3, expert_offload_gib=45)
    assert with_.resident_bytes - without.resident_bytes == _DRAFT


def test_a_verify_step_floor_is_its_slots_across_every_moe_layer():
    """k=3: 4 tokens x top_k 22 = 88 slots per layer, x 40 layers x 1.976 MiB = 6.79 GiB."""
    sup = _sup(24.0)
    assert sup.arena_floor_gib(4) == pytest.approx(6.80, abs=0.005)
    assert sup.arena_floor_gib(2) == pytest.approx(3.40, abs=0.005)
    assert _sup(24.0, moe_top_k=0).arena_floor_gib(4) is None


def test_a_pinned_depth_gets_an_arena_that_holds_its_step():
    """★ The env the arena reads must buy at least 88 slots per layer, or it refuses the
    step and the run silently measures no arena. `GLQ_MOE_EXPERT_CACHE` is rounded UP for
    exactly this reason."""
    from glq_vllm.moe_shapes import slots_for_budget

    sup = _pinned(3)
    assert sup.spec_tokens == 3
    assert (sup.step_tokens, sup.min_step_tokens) == (4, 4)
    env = child_env(device="cuda", expert_arena_gib=sup.expert_arena_gib,
                    max_num_seqs=sup.step_tokens, min_step_tokens=sup.min_step_tokens)
    assert env["GLQ_MOE_ARENA_MAX_TOKENS"] == env["GLQ_MOE_ARENA_MIN_TOKENS"] == "4"
    slots = slots_for_budget(int(float(env["GLQ_MOE_EXPERT_CACHE"]) * GIB),
                             n_layers=_MOE_LAYERS, hidden_size=1024, inter=2688,
                             w13_out=2688, bpw=3)
    assert slots >= 4 * _TOPK


def _startup_lines(sup_kw, tmp_path):
    """What `start()` prints before it spawns vLLM — the spawn itself is stopped short."""
    import io

    class _Stop(Exception):
        pass

    def spawn(*_a, **_kw):
        raise _Stop

    out = io.StringIO()
    sup = _sup(24.0, serve=True, out=out, log_path=tmp_path / "vllm.log",
               spawn=spawn, probe=lambda _url: False, **sup_kw)
    with pytest.raises(_Stop):
        sup.start()
    return out.getvalue()


def test_a_pinned_depth_beyond_the_margins_says_so(tmp_path):
    """Pinned is honoured where the plan's margins say no, and the user is told by how much,
    what it costs, and that the output is not byte-identical."""
    text = _startup_lines(dict(draft_model=DRAFTER, draft_tokens=3, draft_bytes=_DRAFT),
                          tmp_path)
    assert "speculating 3 draft tokens per step with DSpark" in text
    assert DRAFTER in text and "not byte-identical" in text
    assert "--dspark 3 needs a 6.8 GiB expert cache" in text


def test_a_pinned_depth_on_a_hand_set_budget_claims_no_margins(tmp_path):
    """With `--gpu-memory-utilization` the planner never reads the card, so it has no margins
    to fall short of. Measured: `--dspark 3 --cpu-offload-gb 40 --gpu-memory-utilization
    0.6931` printed "6.80 GiB more than this plan's margins leave", then served 262,144 tokens
    with 5.68 GiB of KV. The arena must still hold the step."""
    kw = dict(draft_model=DRAFTER, draft_tokens=3, draft_bytes=_DRAFT, vram_bytes=None,
              gpu_memory_utilization=0.6931, expert_offload_gib=40, max_model_len=262144)
    text = _startup_lines(kw, tmp_path)
    assert "speculating 3 draft tokens per step with DSpark" in text
    assert "margins" not in text
    sup = _sup(24.0, **kw)
    assert float(sup.expert_arena_gib) == pytest.approx(6.80, abs=0.005)
    assert sup.min_step_tokens == 4


def test_auto_says_why_it_turned_dspark_off(tmp_path):
    text = _startup_lines(dict(draft_model=DRAFTER, draft_tokens=4, draft_auto=True,
                               draft_bytes=_DRAFT, vram_bytes=int(22.49 * GIB),
                               ram_bytes=64 * GIB), tmp_path)
    assert "DSpark off" in text and "--dspark N" in text
    assert "speculating" not in text


# ---------------------------------------------------------------------------
# Auto: the deepest k that fits, never at the window's expense
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vram,ram", [(22.49, 124), (24.0, 128), (24.0, 64), (31.86, 60),
                                      (31.86, 120), (95.6, 128)])
def test_auto_never_costs_the_window(vram, ram):
    """★ Planned as if DSpark were off. Before, the drafter was priced in first: on a 64 GiB
    host the window fell from 262,144 to 16,384, and DSpark then turned itself off anyway."""
    off, auto = _sup(vram, ram), _auto(vram, ram_gib=ram)
    assert (auto.max_model_len, auto.expert_offload_gib) == \
           (off.max_model_len, off.expert_offload_gib)


@pytest.mark.parametrize("vram,ram", [(22.49, 64), (24.0, 64)])
def test_auto_off_is_exactly_the_plan_without_dspark(vram, ram):
    """On a 64 GiB host the drafter (2.25 GiB) and a one-token arena (3.40 GiB) do not fit
    beside 262,144 tokens, so the drafter is not loaded and nothing about the serve changes."""
    off, auto = _sup(vram, ram), _auto(vram, ram_gib=ram)
    assert auto.spec_tokens == 0
    assert auto.draft_model is None and auto.draft_bytes == 0
    assert _plan(auto) == _plan(off)
    assert _spec(auto) is None
    assert "DSpark off" in auto._spec_note


def test_auto_is_monotone_in_card_size():
    ks = [_auto(v).spec_tokens for v in (22.49, 24.0, 31.86)]
    assert ks == sorted(ks), ks
    assert ks[-1] >= 1


def test_an_auto_depth_leaves_a_pool_that_holds_everything():
    """The fraction holds the weights, the drafter, the arena, the overhead and the window's
    KV — the utilization is planned last, from what was actually kept."""
    for vram in (24.0, 31.86):
        sup = _auto(vram)
        assert sup.spec_tokens > 0
        need = (sup.resident_bytes + runtime_overhead_bytes(sup.resident_bytes)
                + float(sup.expert_arena_gib) * GIB
                + window_kv_bytes(sup.max_model_len, 1, kv_cost(SV)))
        assert sup.gpu_memory_utilization * vram * GIB >= need


def test_an_auto_arena_holds_its_step():
    for vram in (24.0, 31.86):
        sup = _auto(vram)
        assert float(sup.expert_arena_gib) >= sup.arena_floor_gib(1 + sup.spec_tokens)


def test_a_resident_model_keeps_the_deepest_auto_depth():
    """Every expert on a 96 GB card: no arena to size, and room for the drafter."""
    sup = _auto(95.6)
    assert sup.expert_offload_gib == 0 and sup.spec_tokens == 4


def test_auto_with_an_unknown_moe_shape_is_off():
    sup = _auto(31.86, moe_top_k=0)
    assert sup.spec_tokens == 0 and "unknown" in sup._spec_note


def test_auto_uses_the_checkpoints_declared_figures():
    """Until 2026-11-01 config.json declared no non-text bytes and counted the bf16 MTP head's
    experts as offloadable, because the quantizer matched `mtp` only as the first dotted
    segment and Super-VL spells it `language_model.mtp`. The checkpoint looked 5.48 GiB heavier
    on a RAM-capped host than it is, and on 31.86 GiB / 60 GiB auto found no depth. With the
    corrected figures it finds 3 — the planner reading its inputs, not a margin."""
    before = dict(expert_offload_bytes=_EXPERTS_BEFORE, nontext_bytes=0)
    assert _auto(31.86, ram_gib=60, **before).spec_tokens == 0
    assert _auto(31.86, ram_gib=60).spec_tokens == 3


# ---------------------------------------------------------------------------
# Window pricing for this family, and nothing else moves
# ---------------------------------------------------------------------------

def test_the_supervl_window_covers_what_vllm_measured():
    """Per-request KV vLLM reported with the drafter loaded: 3.12 GiB at 262,144 and 10.43 at
    1,048,576. The priced figure must sit above both."""
    cost = kv_cost(SV)
    assert window_kv_bytes(262_144, 1, cost) >= 3.12 * GIB
    assert window_kv_bytes(1_048_576, 1, cost) >= 10.43 * GIB


def test_other_families_keep_the_envelope():
    assert kv_cost("xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw") == (0, 28_201)
    assert window_ceiling_for_vram(24 * GIB) == 131_072
    assert window_ceiling_for_vram(24 * GIB, kv_cost(SV)) >= 262_144


_FN = dict(model="xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4-3mtp",
           weights_bytes=79_090_922_404, ple_offload_bytes=25_600_122_880,
           expert_offload_bytes=45_770_637_312, model_max_len=262_144)
_G26 = dict(model="xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw",
            weights_bytes=int(14.97 * GIB), model_max_len=262_144)
_CHAT = dict(window_concurrency=1)
_CODE = dict(window_concurrency=1, max_num_seqs=1, max_model_len_floor=16384,
             window_offload_extra_gib=WINDOW_OFFLOAD_MAX_EXTRA_GIB)


@pytest.mark.parametrize("model,mode,vram,ram,expected", [
    # (window, expert offload GiB, arena GiB), computed with glq 0.8.33 (the release tree)
    # on these exact inputs. Only the utilization may differ: it now includes the arena.
    (_FN, _CODE, 22.49, 124, (131_072, 42, 3)),
    (_FN, _CODE, 31.87, 124, (262_144, 42, 8)),
    (_FN, _CHAT, 22.49, 124, (131_072, 42, 3)),
    (_FN, _CHAT, 31.87, 124, (262_144, 42, 8)),
    (_FN, _CODE, 95.6, 124, (262_144, 0, 0)),
    (_FN, _CODE, 22.49, 64, (16_384, 8, 0)),
    (_G26, _CHAT, 22.49, 64, (32_768, 0, 0)),
    (_G26, _CODE, 24.0, 64, (65_536, 0, 0)),
])
def test_plans_without_dspark_are_unchanged(model, mode, vram, ram, expected):
    sup = VllmSupervisor(device="cuda", serve=False, vram_bytes=int(vram * GIB),
                         ram_bytes=int(ram * GIB), **model, **mode)
    assert (sup.max_model_len, sup.expert_offload_gib, sup.expert_arena_gib) == expected
    if sup.expert_arena_gib:
        # The fix that moved the utilization: the pool now holds the arena it plans.
        assert sup.gpu_memory_utilization * vram * GIB > \
            sup.resident_bytes + sup.expert_arena_gib * GIB


# ---------------------------------------------------------------------------
# The launchers: drafter lookup, checkpoint facts, and the flag
# ---------------------------------------------------------------------------

def test_the_drafter_is_known_for_both_supervl_builds_only():
    from glq.tooling import dspark_drafter

    assert dspark_drafter(SV) == DRAFTER
    assert dspark_drafter(SV.replace("3bpw", "4bpw")) == DRAFTER
    assert dspark_drafter("xv0y5ncu/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GLQ-"
                          "trellis-3inst-4bpw") is None
    assert dspark_drafter("") is None


def _supervl_config(tmp_path, bpw=3, codebook="trellis"):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "nemotron_h_omni",
        "text_config": {
            "hidden_size": 4096, "moe_latent_size": 1024, "moe_intermediate_size": 2688,
            "num_experts_per_tok": 22, "mlp_hidden_act": "relu2",
            "layers_block_type": ["full_attention"] * 8 + ["linear_attention"] * 40
                                 + ["moe"] * 40},
        "quantization_config": {"quant_method": "glq", "codebook": codebook, "bpw": bpw},
    }))
    return str(tmp_path)


def test_checkpoint_moe_facts_reads_the_arena_shape(tmp_path):
    from glq.chat import checkpoint_moe_facts

    assert checkpoint_moe_facts(_supervl_config(tmp_path)) == (_TOPK, _MOE_LAYERS, _SLOT)


def test_checkpoint_moe_facts_without_a_trellis_moe_is_zeros(tmp_path):
    from glq.chat import checkpoint_moe_facts

    assert checkpoint_moe_facts(_supervl_config(tmp_path, codebook="e8p")) == (0, 0, 0)
    assert checkpoint_moe_facts(str(tmp_path / "nope")) == (0, 0, 0)
    assert checkpoint_moe_facts("") == (0, 0, 0)


@pytest.mark.parametrize("dspark,draft_model,expected", [
    (None, None, (None, 0, False, None)),                 # off by default
    ("auto", None, (DRAFTER, 4, True, None)),              # bare --dspark
    (3, None, (DRAFTER, 3, False, None)),                  # --dspark 3
    (None, "me/drafter", ("me/drafter", 4, True, None)),   # --draft-model implies auto
    (2, "me/drafter", ("me/drafter", 2, False, None)),
])
def test_the_resolution_table(dspark, draft_model, expected):
    from glq.chat import resolve_dspark

    assert resolve_dspark(dspark, draft_model, SV) == expected


def test_dspark_on_a_family_with_no_drafter_is_an_error():
    from glq.chat import resolve_dspark

    drafter, k, auto, err = resolve_dspark("auto", None, "xv0y5ncu/gemma-4-26B-A4B-it-GLQ")
    assert drafter is None and k == 0 and "--draft-model" in err


def test_both_entry_points_share_the_flags_and_the_resolution():
    """Following `sizing_weights_bytes`: one implementation, so glq-chat and glq-code cannot
    drift apart on what `--dspark` means."""
    import glq.chat as chat_mod
    import glq.code as code_mod

    assert code_mod.add_dspark_flags is chat_mod.add_dspark_flags
    assert code_mod.dspark_kwargs is chat_mod.dspark_kwargs


def _capture(module, argv, monkeypatch):
    seen = {}

    class _Spy:
        def __init__(self, **kw):
            seen.update(kw)
            raise SystemExit(0)

    monkeypatch.setattr(module, "VllmSupervisor", _Spy)
    try:
        module.main(argv)
    except SystemExit:
        pass
    return seen


@pytest.fixture
def launchers(monkeypatch):
    """glq-chat and glq-code with every Hub lookup answered locally."""
    import glq.chat as chat
    import glq.code as code

    monkeypatch.setattr(code, "_find_pi", lambda: "/usr/bin/true")
    monkeypatch.setattr(chat, "missing_chat_deps", lambda: [])
    # `dspark_kwargs` lives in glq.chat and reads these from there, for both commands.
    monkeypatch.setattr(chat, "checkpoint_moe_facts", lambda repo: (_TOPK, 40, _SLOT))
    monkeypatch.setattr(chat, "_draft_bytes", lambda repo: _DRAFT)
    monkeypatch.setattr(chat, "_checkpoint_bytes", lambda repo: _WEIGHTS)
    for mod in (chat, code):
        monkeypatch.setattr(mod, "checkpoint_offload_bytes",
                            lambda repo: (0, _EXPERTS, _NONTEXT))
        monkeypatch.setattr(mod, "checkpoint_mtp", lambda repo: (0, False, 0))
        monkeypatch.setattr(mod, "_model_max_len", lambda repo: 1_048_576)
    return chat, code


@pytest.mark.parametrize("flags,expected", [
    (["--dspark"], (DRAFTER, 4, True)),
    (["--dspark", "3"], (DRAFTER, 3, False)),
    (["--draft-model", "me/drafter"], ("me/drafter", 4, True)),
])
def test_both_launchers_pass_dspark_to_the_planner(launchers, monkeypatch, flags, expected):
    for mod in launchers:
        seen = _capture(mod, ["--no-serve", "--model", SV, *flags], monkeypatch)
        assert (seen["draft_model"], seen["draft_tokens"], seen["draft_auto"]) == expected
        assert seen["draft_bytes"] == _DRAFT
        assert (seen["moe_top_k"], seen["moe_layers"], seen["arena_slot_bytes"]) == \
               (_TOPK, 40, _SLOT)


def test_both_launchers_serve_super_vl_text_only(launchers, monkeypatch):
    """The first `glq-code --dspark 3` run at 262,144 refused to start: vLLM profiled the
    vision encoder with a maximum-size video and kept 1.41 GiB of KV against the 3.05 GiB
    the window needs. Neither command sends an image, so neither loads the tower."""
    for mod in launchers:
        seen = _capture(mod, ["--no-serve", "--model", SV, "--dspark", "3"], monkeypatch)
        assert seen["extra_args"].count("--language-model-only") == 1, mod.__name__


def test_without_the_flag_the_launchers_ask_for_no_drafter(launchers, monkeypatch):
    for mod in launchers:
        seen = _capture(mod, ["--no-serve", "--model", SV], monkeypatch)
        assert not seen.get("draft_model") and not seen.get("draft_tokens")


@pytest.mark.parametrize("flags", [["--dspark", "--mtp", "1"], ["--dspark", "5"],
                                   ["--dspark", "0"]])
def test_bad_dspark_flags_are_argparse_errors(launchers, monkeypatch, flags):
    for mod in launchers:
        with pytest.raises(SystemExit) as e:
            mod.main(["--no-serve", "--model", SV, *flags])
        assert e.value.code == 2


def test_an_unknown_family_exits_before_the_supervisor(launchers, monkeypatch, capsys):
    built = []
    for mod in launchers:
        monkeypatch.setattr(mod, "VllmSupervisor", lambda **kw: built.append(kw))
        rc = mod.main(["--no-serve", "--model", "xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-"
                       "3inst-4bpw", "--dspark"])
        assert rc == 2
    assert built == []
    assert "--draft-model" in capsys.readouterr().err
