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

KOLIBRI = "xv0y5ncu/Kolibri-1-GLQ-trellis-3inst-3bpw"
KOLIBRI_DRAFTER = "intuit/Kolibri-1-dspark"
_KOLIBRI_DRAFT = 825_838_209      # 0.77 GiB, read live from the Hub 2026-11-02
#: From its config.json: top-6 routing over 50 MoE layers, hidden 2560, inter 512, silu
#: (gated), 3 bpw.
_K_TOPK, _K_MOE_LAYERS, _K_SLOT = 6, 50, 1_481_748


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


@pytest.mark.parametrize("launcher", [_CODE, _CHAT])
def test_kolibri_on_a_32_gb_card_plans_what_served(launcher):
    """The plan both launchers made for Kolibri-1 3 bpw on an RTX PRO 4500 (32,623 MiB, 30.07
    GiB host RAM, vLLM 0.29.0), from these inputs, and that then served
    (`benchmarks/kolibri_dspark_rtxpro4500/`): the full window, 18 GiB of experts on host, and
    a 4 GiB arena holding the 5-token verify step on all 50 MoE layers. The drafter is paid
    for out of the arena (5 GiB without it), not out of the window or the offload."""
    facts = dict(model=KOLIBRI, device="cuda", serve=False, weights_bytes=30_819_045_448,
                 expert_offload_bytes=28_740_327_000, nontext_bytes=0,
                 vram_bytes=32623 * 1024 ** 2, ram_bytes=31_535_316 * 1024,
                 model_max_len=262_144, moe_top_k=_K_TOPK, moe_layers=_K_MOE_LAYERS,
                 arena_slot_bytes=_K_SLOT, **launcher)
    off = VllmSupervisor(**facts)
    on = VllmSupervisor(**facts, draft_model=KOLIBRI_DRAFTER, draft_tokens=4, draft_auto=True,
                        draft_bytes=_KOLIBRI_DRAFT)
    assert _plan(off)[:3] == (262_144, 18, 5.0)
    assert _plan(on)[:3] == (262_144, 18, 4.0)
    assert _spec(on) == {"method": "dspark", "model": KOLIBRI_DRAFTER,
                         "num_speculative_tokens": 4}


# ---------------------------------------------------------------------------
# The drafter's own KV
# ---------------------------------------------------------------------------
#
# A DSpark drafter is a small transformer with KV of its own. `glq-code --draft-model
# RedHatAI/Qwen3.8-27B-speculator.dspark --dspark 3` on Arete-27B 4 bpw planned the full 262,144
# window on one RTX PRO 6000 and vLLM 0.33.0 refused it at startup, with and without the pi
# adapter: "21.39 GiB KV cache is needed, which is larger than the available KV cache memory
# (19.78 GiB)". The family's KV cost was measured without a drafter, and nothing added its.
# (`benchmarks/arete_pi_adapter/README.md`, 2026-11-22.)

ARETE = "xv0y5ncu/Arete-27B-GLQ-trellis-3inst-4bpw"
ARETE_DRAFTER = "RedHatAI/Qwen3.8-27B-speculator.dspark"
#: Read live from the Hub 2026-11-22, the way the launchers size them.
_ARETE_WEIGHTS = 18_236_907_824    # 16.98 GiB
_ARETE_DRAFT = 3_976_869_890       # 3.70 GiB
_PI_ADAPTER = int(0.40 * GIB)      # tooling's PiAdapter.gpu_bytes
#: From the drafter's config.json (`transformer_layer_config`): 5 layers, 4 KV heads, head_dim
#: 256, bf16. Its layers declare a 2,048-token sliding window, but vLLM 0.33 charged them at the
#: full window: 21.39 GiB against Arete's own ~16.5 GiB at 262,144 is 5 x 4 KiB per token.
_ARETE_DRAFT_KV = 5 * 2 * 4 * 256 * 2

#: vLLM budgets against the CUDA total (95.03 GiB on that card); the planner reads nvidia-smi
#: (95.59).
_PRO6000_SMI, _PRO6000_CUDA = 95.59, 95.03
#: A card on which an auto drafter's bytes fit beside Arete's 262,144-token window and its KV
#: for that window does not. Found by sweeping this plan in 0.1 GiB steps: the band is 47.1 to
#: 53.3 GiB (with 37.1-40.2 at 131,072 and 32.1-33.0 at 65,536).
_AUTO_KV_BINDS_GIB = 48.0


def _arete(vram_gib=_PRO6000_SMI, adapter=True, **kw):
    base = dict(model=ARETE, device="cuda", serve=False,
                weights_bytes=_ARETE_WEIGHTS + (_PI_ADAPTER if adapter else 0),
                vram_bytes=int(vram_gib * GIB), ram_bytes=int(499 * GIB),
                model_max_len=262_144, **_CODE)
    base.update(kw)
    return VllmSupervisor(**base)


def _arete_dspark(k=3, kv=_ARETE_DRAFT_KV, **kw):
    return _arete(draft_model=ARETE_DRAFTER, draft_tokens=k, draft_bytes=_ARETE_DRAFT,
                  draft_kv_per_token=kv, **kw)


@pytest.mark.parametrize("adapter,util,loaded,kv_had", [
    # The two runs that stopped: the plan's utilization, vLLM's "Model loading took" and
    # "Available KV cache memory", GiB.
    (True, 0.4680, 21.19, 19.79),
    (False, 0.4634, 20.89, 19.63),
])
def test_a_pinned_drafter_leaves_the_kv_vllm_asked_for(adapter, util, loaded, kv_had):
    """Replays vLLM's own accounting of the stopped runs: what it kept outside weights and KV
    (activations, graphs) is the budget less those two. The plan without the drafter's KV
    reproduces the refusal; with it, the pool holds the 21.39 GiB one request needs."""
    non_kv = util * _PRO6000_CUDA - loaded - kv_had

    def kv_vllm_would_have(sup):
        return sup.gpu_memory_utilization * _PRO6000_CUDA - loaded - non_kv

    before = _arete_dspark(adapter=adapter, kv=0)
    assert before.max_model_len == 262_144
    assert round(before.gpu_memory_utilization, 3) == round(util, 3)   # the plan that ran
    assert kv_vllm_would_have(before) < 21.39

    after = _arete_dspark(adapter=adapter)
    assert after.max_model_len == 262_144
    assert after.kv_cost == (kv_cost(ARETE)[0], kv_cost(ARETE)[1] + _ARETE_DRAFT_KV)
    assert kv_vllm_would_have(after) >= 21.39


def test_the_drafter_s_kv_reaches_the_fit_check_floor():
    """`fit.check_fit` prices its floor window from `supervisor.kv_cost`, so a drafter that
    is loaded is in the floor too."""
    from glq.supervisor import kv_floor_bytes

    on, off = _arete_dspark(), _arete()
    assert kv_floor_bytes(16384, 1, on.kv_cost) > kv_floor_bytes(16384, 1, off.kv_cost)


def test_an_auto_drafter_is_kept_only_if_its_kv_fits_as_well():
    """Auto plans the window without DSpark and keeps the drafter only if it fits in what is
    left. On this card its bytes fit beside the window, its KV for that window does not."""
    vram = _AUTO_KV_BINDS_GIB
    bytes_only = _arete_dspark(k=4, kv=0, draft_auto=True, vram_gib=vram)
    with_kv = _arete_dspark(k=4, draft_auto=True, vram_gib=vram)
    assert bytes_only.max_model_len == with_kv.max_model_len == _arete(vram_gib=vram).max_model_len
    assert bytes_only.spec_tokens == 4
    assert with_kv.spec_tokens == 0 and with_kv.draft_model is None
    assert "DSpark off" in with_kv._spec_note and "KV" in with_kv._spec_note
    assert with_kv.kv_cost == kv_cost(ARETE)                # not paid for, not loaded


def test_an_auto_drafter_that_is_kept_is_in_the_pool():
    sup = _arete_dspark(k=4, draft_auto=True)
    assert sup.spec_tokens == 4 and sup.max_model_len == 262_144
    assert sup.kv_cost == (kv_cost(ARETE)[0], kv_cost(ARETE)[1] + _ARETE_DRAFT_KV)
    assert sup.gpu_memory_utilization * _PRO6000_SMI * GIB >= (
        sup.resident_bytes + window_kv_bytes(262_144, 1, sup.kv_cost))


def test_families_whose_kv_price_covers_the_drafter_do_not_move():
    """Super-VL's measured cost was taken with its drafter loaded, and Kolibri-1's envelope
    served its drafter at 262,144 with KV to spare (1.73x the window). Their plans are the
    released ones whatever the drafter's KV."""
    for sup_kw in (dict(), dict(draft_auto=True, draft_tokens=4)):
        a = _sup(24.0, draft_model=DRAFTER, draft_bytes=_DRAFT,
                 **{"draft_tokens": 3, **sup_kw})
        b = _sup(24.0, draft_model=DRAFTER, draft_bytes=_DRAFT, draft_kv_per_token=5_120,
                 **{"draft_tokens": 3, **sup_kw})
        assert _plan(a) == _plan(b) and b.kv_cost == kv_cost(SV)
    facts = dict(model=KOLIBRI, device="cuda", serve=False, weights_bytes=30_819_045_448,
                 expert_offload_bytes=28_740_327_000, nontext_bytes=0,
                 vram_bytes=32623 * 1024 ** 2, ram_bytes=31_535_316 * 1024,
                 model_max_len=262_144, moe_top_k=_K_TOPK, moe_layers=_K_MOE_LAYERS,
                 arena_slot_bytes=_K_SLOT, draft_model=KOLIBRI_DRAFTER, draft_tokens=4,
                 draft_auto=True, draft_bytes=_KOLIBRI_DRAFT, **_CODE)
    on = VllmSupervisor(**facts, draft_kv_per_token=10_240)
    assert _plan(on)[:3] == (262_144, 18, 4.0) and on.kv_cost == kv_cost(KOLIBRI)


def _drafter_config(tmp_path, **tlc):
    layer = dict(model_type="qwen3", num_hidden_layers=5, num_key_value_heads=4, head_dim=256,
                 hidden_size=5120, num_attention_heads=20, sliding_window=2048)
    layer.update(tlc)
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["DSparkDraftModel"], "speculators_model_type": "dspark",
        "dtype": "bfloat16", "block_size": 8, "transformer_layer_config": layer}))
    return str(tmp_path)


@pytest.mark.parametrize("tlc,expected", [
    (dict(), 20_480),                                              # RedHatAI Qwen3.8-27B
    (dict(num_key_value_heads=2, head_dim=128, hidden_size=4096,   # RedHatAI Super-VL
          num_attention_heads=32), 5_120),
    (dict(head_dim=128, hidden_size=2560), 10_240),                # intuit Kolibri-1
    (dict(head_dim=None, hidden_size=5120, num_attention_heads=20), 20_480),  # hidden / heads
])
def test_the_drafter_s_kv_per_token_comes_from_its_config(tmp_path, tlc, expected):
    from glq.chat import _draft_kv_per_token

    assert _draft_kv_per_token(_drafter_config(tmp_path, **tlc)) == expected


def test_an_unreadable_drafter_config_prices_nothing(tmp_path, monkeypatch):
    from glq.chat import _draft_kv_per_token
    from glq.installer import discovery

    def offline(url, *a, **kw):
        raise OSError(f"offline: {url}")

    monkeypatch.setattr(discovery, "_fetch_json", offline)    # a repo id that cannot be read
    assert _draft_kv_per_token("RedHatAI/not-a-drafter") == 0
    assert _draft_kv_per_token("") == 0
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["DSparkDraftModel"]}))
    assert _draft_kv_per_token(str(tmp_path)) == 0


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


def test_the_kolibri_1_drafter():
    """The GLQ build and both of Aleph Alpha's bases take it. It reads Kolibri-1's hidden
    states (layers 4, 14, 24, 34, 44), so a later Kolibri generation does not."""
    from glq.tooling import dspark_drafter

    for model in (KOLIBRI, "Aleph-Alpha/Kolibri-1", "Aleph-Alpha/Kolibri-1-BF16"):
        assert dspark_drafter(model) == KOLIBRI_DRAFTER, model
    assert dspark_drafter("Aleph-Alpha/Kolibri-2") is None
    assert dspark_drafter(SV) == DRAFTER


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


def test_checkpoint_moe_facts_reads_a_kolibri_config(tmp_path):
    """The other branch: no `layers_block_type`, so every decoder layer is MoE, no
    `text_config`, and silu is gated, so w13 holds gate and up (2 x 512)."""
    from glq.chat import checkpoint_moe_facts

    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "kolibri1", "hidden_size": 2560, "moe_intermediate_size": 512,
        "num_experts_per_tok": 6, "num_hidden_layers": 50, "hidden_act": "silu",
        "quantization_config": {"quant_method": "glq", "codebook": "trellis", "bpw": 3},
    }))
    assert checkpoint_moe_facts(str(tmp_path)) == (_K_TOPK, _K_MOE_LAYERS, _K_SLOT)


def test_checkpoint_moe_facts_reads_a_glm5_config(tmp_path):
    """GLM-5.3 marks its MLPs in `mlp_layer_types`: layers 0-2 are dense, so 42 of its 45
    decoder layers are MoE, not every one. Hidden 4096, inter 2048, silu (gated), 4 bpw."""
    from glq.chat import checkpoint_moe_facts

    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "glm5_next",
        "text_config": {
            "hidden_size": 4096, "moe_intermediate_size": 2048, "num_experts_per_tok": 8,
            "num_hidden_layers": 45, "hidden_act": "silu", "layers_block_type": None,
            "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42},
        "quantization_config": {"quant_method": "glq", "codebook": "trellis", "bpw": 4},
    }))
    assert checkpoint_moe_facts(str(tmp_path)) == (8, 42, 12_599_316)


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
    monkeypatch.setattr(chat, "_draft_kv_per_token", lambda repo: 5_120)
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
        assert seen["draft_kv_per_token"] == 5_120
        assert (seen["moe_top_k"], seen["moe_layers"], seen["arena_slot_bytes"]) == \
               (_TOPK, 40, _SLOT)


def test_both_launchers_pick_the_kolibri_drafter(launchers, monkeypatch):
    chat, code = launchers
    # glq-code refuses Kolibri before the supervisor when the aleph-alpha-inference plugin
    # is missing; that guard has its own tests, and this one is about the drafter.
    monkeypatch.setattr(code, "_plugin_installed", lambda module: True)
    monkeypatch.setattr(chat, "_draft_bytes",
                        lambda repo: _KOLIBRI_DRAFT if repo == KOLIBRI_DRAFTER else 0)
    for mod in launchers:
        seen = _capture(mod, ["--no-serve", "--model", KOLIBRI, "--dspark"], monkeypatch)
        assert (seen["draft_model"], seen["draft_tokens"], seen["draft_auto"]) == \
               (KOLIBRI_DRAFTER, 4, True), mod.__name__
        assert seen["draft_bytes"] == _KOLIBRI_DRAFT


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
