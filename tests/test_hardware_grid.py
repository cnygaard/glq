"""Every recommended checkpoint, planned on every card from 8 to 96 GiB and every host from 16
to 128 GiB, for both launchers.

`test_dspark_planning.py` sweeps four cards between 22.49 and 31.86 GiB. This widens the sweep
to the range glq-code and glq-chat are meant for, 8 GiB laptops through 96 GiB workstation
cards, and asks the same questions of every plan the launchers would make there:

* the pool vLLM is given holds what was planned (weights, overhead, arena, window KV);
* what is pinned stays inside the planner's own host caps;
* a bigger card or host never turns a fit into a refusal, nor shrinks the window;
* the picker recommends something that actually fits, or nothing, with a reason;
* DSpark auto on Super-VL never costs the window and only deepens with the machine.

Pure planning on frozen checkpoint facts, no GPU and no network, matching
`test_dspark_planning.py`. Measured on a GPU in `benchmarks/hw_tiers/`.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import pytest

from glq import fit
from glq.installer.discovery import Checkpoint
from glq.supervisor import (_MAX_UTILIZATION, WINDOW_OFFLOAD_MAX_EXTRA_GIB, VllmSupervisor,
                            kv_cost, runtime_overhead_bytes, window_kv_bytes)
from glq.tooling import required_plugin, tool_serve_args

GIB = 1024 ** 3
ORG = "xv0y5ncu/"


@dataclass(frozen=True)
class _Facts:
    name: str
    size: int
    trellis: bool
    moe: bool
    ple: int
    expert: int
    nontext: int
    maxlen: int

    @property
    def repo(self):
        return ORG + self.name

    def checkpoint(self):
        return Checkpoint(self.repo, self.size, self.trellis, self.moe, self.ple,
                          self.expert, self.nontext)


#: Read live 2026-11-01: sizes from the Hub tree API (`discovery.repo_size_bytes`), offload
#: figures from config.json's `quantization_config` (`discovery.offload_bytes`), the window
#: from `discovery.model_max_len`. The Super-VL figures are the corrected ones, uploaded that
#: day: the 5.48 GiB MTP head is declared non-text rather than counted as experts.
START_HERE = (
    _Facts("SmolLM3-3B-trellis-3inst-4bpw-kernel", 1935670232, True, False, 0, 0, 0, 65536),
    _Facts("gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw", 14976261824, True, True, 0, 0, 0,
           262144),
    _Facts("gemma-4-26B-A4B-it-GLQ-trellis-3inst-3bpw", 11915828384, True, True, 0, 0, 0,
           262144),
    _Facts("Gemma-4-31B-it-GLQ-5.0bpw-mix3-8", 24055729364, False, False, 0, 0, 0, 262144),
    _Facts("Gemma-4-12B-it-GLQ-5.0bpw", 9422291648, False, False, 0, 0, 0, 262144),
    _Facts("gemma-4-E4B-it-GLQ-trellis-3inst-4bpw", 6592155332, True, False, 0, 0, 0, 131072),
    _Facts("SmolLM2-360M-Instruct-GLQ-trellis-3inst-6bpw", 331663608, True, False, 0, 0, 0,
           8192),
    _Facts("SmolLM2-135M-Instruct-GLQ-block-diagonal-4bpw", 110525584, False, False, 0, 0, 0,
           8192),
)
#: Published but not in start-here: the MoE builds that need host offload below 24-96 GiB.
#: Kolibri needs a vLLM plugin, so neither picker offers it without one.
OFFLOAD = (
    _Facts("NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GLQ-trellis-3inst-3bpw", 15730075776, True,
           True, 0, 11241793720, 2670652160, 262144),
    _Facts("NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GLQ-trellis-3inst-4bpw", 19588131448, True,
           True, 0, 14971017400, 2670652160, 262144),
    _Facts("NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-3bpw", 55038723416, True,
           True, 0, 43244953920, 5884651520, 1048576),
    _Facts("NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-4bpw", 69977261400, True,
           True, 0, 57558016320, 5884651520, 1048576),
    _Facts("gemma-4-31B-it-GLQ-trellis-3inst-4bpw", 18631697016, True, False, 0, 0, 0, 262144),
    _Facts("Kolibri-1-GLQ-trellis-3inst-3bpw", 30819045448, True, True, 0, 28740327000, 0,
           262144),
)
ALL = START_HERE + OFFLOAD
SV3, SV4 = OFFLOAD[2], OFFLOAD[3]

#: 95.6 is the RTX PRO 6000 as nvidia-smi reports it (97,887 MiB).
VRAMS = (8, 12, 16, 24, 32, 48, 95.6)
RAMS = (16, 32, 64, 128)

#: The two launchers, as `glq.code.main` and `glq.chat.main` construct their supervisors.
#: glq-code: one stream, a 16,384 floor, and up to 8 GiB of extra offload for a longer window.
#: glq-chat: the default 16 sequences priced at one, an 8,192 floor, no window offload.
LAUNCHERS = {
    "code": dict(floor=16384, kw=dict(max_model_len_floor=16384, max_num_seqs=1,
                                      window_concurrency=1,
                                      window_offload_extra_gib=WINDOW_OFFLOAD_MAX_EXTRA_GIB)),
    "chat": dict(floor=8192, kw=dict(window_concurrency=1)),
}

#: The drafter both Super-VL builds take (`tooling.dspark_drafter`), and the MoE facts
#: `chat.dspark_kwargs` reads from config.json: num_experts_per_tok, the "moe" count in
#: layers_block_type, and the 1024-wide latent with inter 2688, relu2 (not gated).
DRAFTER = "RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B.dspark"
_DRAFT = 2_420_229_602
_TOPK, _MOE_LAYERS = 22, 40


def _slot(bpw):
    from glq_vllm.moe_shapes import bytes_per_slot
    return bytes_per_slot(hidden_size=1024, inter=2688, w13_out=2688, bpw=bpw)


def _sup(f, vram, ram, launcher, **extra):
    kw = dict(LAUNCHERS[launcher]["kw"])
    kw.update(extra)
    return VllmSupervisor(
        model=f.repo, device="cuda", serve=False, model_max_len=f.maxlen,
        weights_bytes=f.size, vram_bytes=int(vram * GIB), ram_bytes=int(ram * GIB),
        ple_offload_bytes=f.ple, expert_offload_bytes=f.expert, nontext_bytes=f.nontext,
        **kw)


def _auto(f, vram, ram, launcher="code"):
    bpw = 3 if f is SV3 else 4
    return _sup(f, vram, ram, launcher, draft_model=DRAFTER, draft_tokens=4, draft_auto=True,
                draft_bytes=_DRAFT, moe_top_k=_TOPK, moe_layers=_MOE_LAYERS,
                arena_slot_bytes=_slot(bpw))


def _verdict(sup, ram, launcher):
    return fit.check_fit(sup, ram_bytes=int(ram * GIB), swap_bytes=0,
                         available_ram_bytes=int(ram * GIB * 0.9),
                         floor=LAUNCHERS[launcher]["floor"])


@functools.lru_cache(maxsize=None)
def _cell(name, vram, ram, launcher):
    f = next(x for x in ALL if x.name == name)
    sup = _sup(f, vram, ram, launcher)
    return sup, _verdict(sup, ram, launcher)


def _need(sup):
    """Everything the pool must hold for this plan, in bytes."""
    cost = getattr(sup, "kv_cost", None) or kv_cost(sup.model)
    return (sup.resident_bytes + runtime_overhead_bytes(sup.resident_bytes)
            + float(sup.expert_arena_gib or 0) * GIB
            + window_kv_bytes(sup.max_model_len, sup.window_concurrency, cost))


_CELLS = [(f.name, v, r, ln) for f in ALL for v in VRAMS for r in RAMS for ln in LAUNCHERS]


def _id(cell):
    name, v, r, ln = cell
    return f"{ln}-{v}g-{r}r-{name}"


# ---------------------------------------------------------------------------
# One plan at a time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell", _CELLS, ids=_id)
def test_a_fitting_plan_gives_vllm_a_pool_that_holds_it(cell):
    """The utilization is the only memory figure vLLM sees. If it is short of the plan, the
    planned window or arena does not exist on the card, however the plan reads."""
    sup, r = _cell(*cell)
    if r.fits is not True:
        return
    vram = cell[1] * GIB
    assert sup.gpu_memory_utilization <= _MAX_UTILIZATION + 1e-9
    # `- 1`: the utilization is need / vram, and multiplying back can lose an ulp.
    assert sup.gpu_memory_utilization * vram >= _need(sup) - 1, (
        f"util {sup.gpu_memory_utilization} x {cell[1]} GiB = "
        f"{sup.gpu_memory_utilization * cell[1]:.2f} GiB < need {_need(sup) / GIB:.2f} GiB")


@pytest.mark.parametrize("cell", _CELLS, ids=_id)
def test_what_is_pinned_stays_inside_the_planners_host_caps(cell):
    """Half the host for the base offload, 0.6 when it was deepened for an arena
    (`supervisor._PINNED_RAM_FRACTION`, `_ARENA_PINNED_RAM_FRACTION`), and never more experts
    than the checkpoint declares.

    Keyed on the deepening, not on the arena that is left: the window re-plan after deepening
    can spend all of it. Super-VL 4 bpw on 32 GiB / 64 GiB deepens 32 -> 38 GiB for a 1 GiB
    arena, then 131,072 tokens take that GiB. Without the deeper offload it does not fit."""
    sup, _ = _cell(*cell)
    f = next(x for x in ALL if x.name == cell[0])
    share = 0.6 if sup.expert_offload_gib > sup.expert_offload_base_gib else 0.5
    assert sup.pinned_host_bytes <= f.ple + share * cell[2] * GIB
    assert sup.effective_expert_offload_gib * GIB <= f.expert


# ---------------------------------------------------------------------------
# Across the grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize("f", ALL, ids=lambda f: f.name)
def test_more_card_or_more_host_never_refuses_what_fit(f, launcher):
    bad = []
    for i, v in enumerate(VRAMS):
        for j, r in enumerate(RAMS):
            if _cell(f.name, v, r, launcher)[1].fits is not True:
                continue
            for v2 in VRAMS[i:]:
                for r2 in RAMS[j:]:
                    r2v = _cell(f.name, v2, r2, launcher)[1]
                    if r2v.fits is not True:
                        bad.append(f"fits at {v}/{r}, refused at {v2}/{r2} ({r2v.reason})")
    assert not bad, bad


@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize("f", ALL, ids=lambda f: f.name)
def test_a_bigger_card_never_shrinks_the_window(f, launcher):
    bad = []
    for r in RAMS:
        prev = None
        for v in VRAMS:
            sup, rep = _cell(f.name, v, r, launcher)
            if rep.fits is not True:
                continue
            if prev is not None and sup.max_model_len < prev[1]:
                bad.append(f"ram {r}: {prev[0]} GiB -> {prev[1]}, {v} GiB -> {sup.max_model_len}")
            prev = (v, sup.max_model_len)
    assert not bad, bad


def test_chat_offloads_what_the_kv_floor_needs_on_a_24_gib_card():
    """Lightning 4 bpw is 16.92 GiB resident. `WEIGHT_FRACTION` (0.75 of the card) says that
    fits 24 GiB, so the base offload was 0; the fit check then found 0.65 GiB of KV room
    against its 2 GiB floor and refused, on the card class this model is for, while 12 and
    16 GiB cards (which fail the 0.75 test, so offload) served it. glq-code on the same
    24 GiB / 64 GiB tier, on an RTX PRO 6000 capped to it: 13 GiB offloaded, a 10 GiB arena,
    a 131,072 window and 91.0-91.4 tok/s at B=1 (row t24_l4, `benchmarks/hw_tiers/README.md`)."""
    f = OFFLOAD[1]
    sup, r = _cell(f.name, 24, 64, "chat")
    assert r.fits is True, (r.reason, r.vram_short_bytes / GIB)
    assert sup.expert_offload_gib > 0


# ---------------------------------------------------------------------------
# The picker
# ---------------------------------------------------------------------------

def _offerable(launcher):
    cks = [f.checkpoint() for f in START_HERE]
    if launcher == "code":
        # `code._pick_model`'s filter: a known tool parser, and no plugin it would need.
        cks = [c for c in cks
               if tool_serve_args(c.repo_id) is not None and required_plugin(c.repo_id) is None]
    return cks


def _ranked(vram, ram, launcher):
    code = launcher == "code"
    return fit.fit_ranked(_offerable(launcher), vram_bytes=int(vram * GIB),
                          ram_bytes=int(ram * GIB), swap_bytes=0, concurrency=1,
                          floor=LAUNCHERS[launcher]["floor"],
                          window_extra_gib=WINDOW_OFFLOAD_MAX_EXTRA_GIB if code else 0,
                          device="cuda")


@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize("ram", RAMS)
@pytest.mark.parametrize("vram", VRAMS)
def test_the_pick_fits_or_nothing_is_picked_with_a_reason(vram, ram, launcher):
    ranked = _ranked(vram, ram, launcher)
    rec = [r for r in ranked if r.recommended]
    if rec:
        [r] = rec
        assert r.fits is True
        # The launcher plans with the model's declared window as well; the pick must still
        # pass its check, or the menu offers what the launcher then refuses.
        f = next(x for x in ALL if x.repo == r.checkpoint.repo_id)
        assert _cell(f.name, vram, ram, launcher)[1].fits is True
    else:
        assert all(r.fits is False and r.reason in ("vram", "host", "both") for r in ranked)


@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize("ram", RAMS)
def test_a_bigger_card_never_picks_a_smaller_model(ram, launcher):
    sizes = []
    for v in VRAMS:
        rec = [r.checkpoint for r in _ranked(v, ram, launcher) if r.recommended]
        sizes.append((v, rec[0].size_bytes if rec else 0, rec[0].short_name if rec else None))
    assert [s for _, s, _ in sizes] == sorted(s for _, s, _ in sizes), sizes


# ---------------------------------------------------------------------------
# DSpark auto on Super-VL, across the grid
# ---------------------------------------------------------------------------

_SV_CELLS = [(f, v, r) for f in (SV3, SV4) for v in VRAMS for r in RAMS]


def _sv_id(cell):
    f, v, r = cell
    return f"{'sv3' if f is SV3 else 'sv4'}-{v}g-{r}r"


@pytest.mark.parametrize("cell", _SV_CELLS, ids=_sv_id)
def test_auto_never_costs_the_window_anywhere(cell):
    f, v, r = cell
    off, auto = _sup(f, v, r, "code"), _auto(f, v, r)
    assert (auto.max_model_len, auto.expert_offload_gib) == \
           (off.max_model_len, off.expert_offload_gib)


@pytest.mark.parametrize("cell", _SV_CELLS, ids=_sv_id)
def test_auto_off_is_exactly_the_plan_without_dspark_anywhere(cell):
    f, v, r = cell
    off, auto = _sup(f, v, r, "code"), _auto(f, v, r)
    if auto.spec_tokens:
        return
    assert auto.draft_model is None and auto.draft_bytes == 0
    assert (auto.max_model_len, auto.expert_offload_gib, float(auto.expert_arena_gib),
            auto.gpu_memory_utilization) == \
           (off.max_model_len, off.expert_offload_gib, float(off.expert_arena_gib),
            off.gpu_memory_utilization)
    assert "--speculative-config" not in auto.argv()


@pytest.mark.parametrize("cell", _SV_CELLS, ids=_sv_id)
def test_an_auto_depth_holds_its_step_in_a_pool_that_holds_everything(cell):
    f, v, r = cell
    sup = _auto(f, v, r)
    if not sup.spec_tokens:
        return
    if sup.expert_offload_gib:
        # With every expert resident there is no arena to hold a step.
        assert float(sup.expert_arena_gib) >= sup.arena_floor_gib(1 + sup.spec_tokens) - 1e-9
    assert sup.gpu_memory_utilization <= _MAX_UTILIZATION + 1e-9
    assert sup.gpu_memory_utilization * v * GIB >= _need(sup) - 1


@pytest.mark.parametrize("f", (SV3, SV4), ids=("sv3", "sv4"))
def test_auto_depth_only_deepens_with_the_machine(f):
    ks = {(v, r): _auto(f, v, r).spec_tokens for v in VRAMS for r in RAMS}
    bad = [f"{v}/{r} k={ks[v, r]} > {v2}/{r2} k={ks[v2, r2]}"
           for (v, r) in ks for (v2, r2) in ks
           if v2 >= v and r2 >= r and ks[v2, r2] < ks[v, r]]
    assert not bad, bad
    assert ks[95.6, 128] == 4


def test_the_slot_size_matches_the_figure_the_dspark_tests_use():
    assert _slot(3) == 2_071_828
