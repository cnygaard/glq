"""Quantizing NON-GATED stacked routed experts — the NemotronH MoE.

Every stacked-expert site in `glq.quantize_model` was written against a GATED container: a 3-D
``gate_up_proj`` [E, 2I, H] that gets row-split into gate and up. `NemotronHExperts` instead
holds two 3-D parameters, ``up_proj`` [E, I, H] and ``down_proj`` [E, H, I], and runs
``down(relu(up(x))**2)`` — no gate anywhere. Neither weight walk sees it: the `nn.Linear` walk
because the parameters are bare 3-D tensors, and the stacked walk because it keys on
``gate_up_proj``.

On NVIDIA-Nemotron-3.5-Super-VL-120B-A12B that is **210.0 of 231.8 GiB, 90.6%** of the
checkpoint, so the failure mode is a "3 bpw" output heavier than its bf16 input, produced by a
run that reports success. #165 shipped a refusal; this is the support that replaces it.

The fixtures here are **real `NemotronHExperts` and real `NemotronHBlock`s**, sized down to CPU
toys, not hand-written stand-ins. `tests/models/README.md` explains why: a fixture encodes the
author's model of an architecture, and that model is wrong in ways the architecture will not
volunteer. One sizing rule is load-bearing — **`mamba_num_heads * mamba_head_dim` must equal
`expand * hidden_size`**, or the Mamba block raises `repeat_interleave: Invalid output_size`.
The real model satisfies it (128 * 64 == 2 * 4096).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

from glq import quantize_model as qm

sys.path.insert(0, os.path.dirname(__file__))

from model_shapes import load as load_shapes  # noqa: E402

SLUG = "nemotron_3_5_super_vl_120b_a12b"


# ---- real NemotronH fixtures -------------------------------------------------------


def _nemotron_cfg(*, hidden=64, n_exp=8, inter=32, latent=16, shared=48,
                  block_types=("mamba", "moe", "attention")):
    """A real `NemotronHConfig`, CPU-sized. `latent=None` gives the no-latent variant."""
    pytest.importorskip("transformers", minversion="5.18.0")
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    expand, mhd = 2, 8
    return NemotronHConfig(
        hidden_size=hidden, num_hidden_layers=len(block_types), vocab_size=128,
        layers_block_type=list(block_types),
        n_routed_experts=n_exp, num_experts_per_tok=2, moe_intermediate_size=inter,
        moe_latent_size=latent, moe_shared_expert_intermediate_size=shared,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        # mamba_num_heads * mamba_head_dim == expand * hidden_size, or the mixer raises.
        expand=expand, mamba_head_dim=mhd, mamba_num_heads=(hidden * expand) // mhd,
        ssm_state_size=16, conv_kernel=4, n_groups=1,
    )


def _moe_block(**kw):
    """A real MoE `NemotronHBlock` (layer index 1 of the default fixture).

    The expert parameters are filled explicitly. `NemotronHExperts` allocates them with
    `torch.empty`, so a freshly constructed block holds uninitialized memory — in practice
    including NaN, which makes every `torch.equal` comparison below false for a reason that
    has nothing to do with the code under test.
    """
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHBlock
    cfg = _nemotron_cfg(**kw)
    blk = NemotronHBlock(cfg, layer_idx=1).to(torch.float32).eval()
    gen = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for p in (blk.mixer.experts.up_proj, blk.mixer.experts.down_proj):
            p.copy_(torch.randn(p.shape, generator=gen) * 0.02)
    return blk, cfg


def _gated_container(n_exp=4, inter=8, hidden=6):
    """A gated stand-in — this one may be synthetic: it stands for gemma-4 / Qwen4Exp /
    Kolibri-1, each of which has its own real-module test already."""
    mod = nn.Module()
    mod.gate_up_proj = nn.Parameter(torch.randn(n_exp, 2 * inter, hidden))
    mod.down_proj = nn.Parameter(torch.randn(n_exp, hidden, inter))
    mod.act_fn = torch.nn.functional.silu
    layer = nn.Module()
    layer.mixer = nn.Module()
    layer.mixer.experts = mod
    return layer


def test_the_fixture_really_is_the_architecture_under_test():
    """Guards every test below. If transformers changes the container, these fixtures stop
    representing the model and the whole file goes quietly vacuous."""
    blk, cfg = _moe_block()
    e = blk.mixer.experts
    assert type(e).__name__ == "NemotronHExperts"
    assert isinstance(e.up_proj, nn.Parameter) and e.up_proj.dim() == 3
    assert isinstance(e.down_proj, nn.Parameter) and e.down_proj.dim() == 3
    assert not hasattr(e, "gate_up_proj"), "the container gained a gate; retire this file"
    assert cfg.mlp_hidden_act == "relu2"
    # The experts sit BEHIND fc1_latent_proj, so their input width is the latent, not hidden.
    assert e.up_proj.shape[2] == cfg.moe_latent_size == 16
    assert blk.mixer.fc1_latent_proj.weight.shape == (16, 64)


def test_the_detector_from_165_finds_the_real_container():
    """`_collect_nongated_stacked_experts` already shipped; this is the work's starting point,
    so pin that it fires on the real module rather than only on the synthetic one."""
    blk, _ = _moe_block()
    found = qm._collect_nongated_stacked_experts(blk)
    assert len(found) == 1
    mn, mod, pname, shape = found[0]
    assert mn == "mixer.experts" and pname == "up_proj"
    assert shape == (8, 32, 16)
    assert qm._collect_stacked_experts(blk) == [], "the gated collector must not claim it"


# ---- the routing report ------------------------------------------------------------


def test_the_routing_report_survives_a_container_with_no_gate():
    """`width = emod.gate_up_proj.shape[-1]` is an outright AttributeError here, and it runs on
    EVERY stacked-expert layer — so this crashes the run after calibration, with the layer's
    Hessians already computed and thrown away."""
    blk, _ = _moe_block()
    emod = blk.mixer.experts
    counts = {("mixer.experts", e): 100 * (e + 1) for e in range(8)}
    lines = []
    qm._report_expert_routing(counts, [("mixer.experts", emod)], emit=lines.append)

    assert lines, "reported nothing for a container that saw tokens"
    joined = "\n".join(lines)
    # The width must be the UP projection's input dim (the latent, 16) — the dimension the
    # up Hessian actually has.
    assert "Hessian is 16 wide" in joined, joined


def test_the_routing_report_is_unchanged_for_a_gated_container():
    lines = []
    emod = _gated_container(n_exp=4, inter=8, hidden=6).mixer.experts
    counts = {("mixer.experts", e): 50 for e in range(4)}
    qm._report_expert_routing(counts, [("mixer.experts", emod)], emit=lines.append)
    assert "Hessian is 6 wide" in "\n".join(lines)


# ---- the state converter -----------------------------------------------------------


def _per_expert_state(emod, prefix="mixer.experts"):
    """The CHECKPOINT form: one 2-D tensor per expert per projection."""
    out = {}
    for e in range(emod.up_proj.shape[0]):
        out[f"{prefix}.{e}.up_proj.weight"] = emod.up_proj.data[e].clone()
        out[f"{prefix}.{e}.down_proj.weight"] = emod.down_proj.data[e].clone()
    return out


def test_the_converter_reproduces_the_real_modules_state_dict_exactly():
    """The whole point of a converter. The streaming path does its own `load_state_dict`, so
    transformers' `MergeModulelist` conversion never runs and every per-expert key is
    "unexpected" — `_assert_load_coverage` then raises. This asserts the output is what the
    real module wants, key for key and value for value."""
    blk, _ = _moe_block()
    emod = blk.mixer.experts
    want = {k: v for k, v in blk.state_dict().items() if "experts" in k and "shared" not in k}

    state = _per_expert_state(emod)
    got = qm._convert_state_nemotron_h(state)

    assert set(got) == set(want), f"keys differ: {set(got) ^ set(want)}"
    for k in want:
        assert torch.equal(got[k], want[k]), f"{k} differs in value"


def test_the_converter_passes_everything_else_through_untouched():
    blk, _ = _moe_block()
    state = _per_expert_state(blk.mixer.experts)
    extra = {"mixer.gate.weight": torch.randn(8, 64),
             "mixer.fc1_latent_proj.weight": torch.randn(16, 64),
             "norm.weight": torch.randn(64)}
    got = qm._convert_state_nemotron_h({**state, **extra})
    for k, v in extra.items():
        assert k in got and torch.equal(got[k], v)


def test_the_converter_refuses_a_gap_in_the_expert_indices():
    """Stacking a non-contiguous set would place weights at the wrong expert index, which
    routing cannot detect: every expert still has *a* weight, just not its own."""
    blk, _ = _moe_block()
    state = _per_expert_state(blk.mixer.experts)
    del state["mixer.experts.3.up_proj.weight"]
    del state["mixer.experts.3.down_proj.weight"]
    with pytest.raises(ValueError, match="contiguous"):
        qm._convert_state_nemotron_h(state)


def test_the_converter_refuses_an_expert_missing_one_projection():
    blk, _ = _moe_block()
    state = _per_expert_state(blk.mixer.experts)
    del state["mixer.experts.5.down_proj.weight"]
    with pytest.raises(ValueError, match="up_proj/down_proj|down_proj"):
        qm._convert_state_nemotron_h(state)


def test_the_converter_rejects_a_gated_checkpoint_rather_than_half_converting_it():
    """A gate_proj in the input means this is not the architecture the converter is for.
    Silently ignoring it would stack up/down and drop the gate — a model that loads and is
    wrong."""
    blk, _ = _moe_block()
    state = _per_expert_state(blk.mixer.experts)
    state["mixer.experts.0.gate_proj.weight"] = torch.randn(32, 16)
    # Matched on the specific diagnosis, not just "gate_proj": the generic completeness
    # check below it also prints the leaf names, so a loose regex passes even with the
    # gated-rejection removed and proves nothing about WHY it refused.
    with pytest.raises(ValueError, match="these experts are GATED"):
        qm._convert_state_nemotron_h(state)


def test_the_converter_preserves_device_and_dtype():
    """Streaming loads onto meta before materializing; a converter that silently moved tensors
    would defeat that."""
    blk, _ = _moe_block()
    state = {k: v.to(torch.bfloat16) for k, v in _per_expert_state(blk.mixer.experts).items()}
    got = qm._convert_state_nemotron_h(state)
    assert all(v.dtype is torch.bfloat16 for v in got.values())


def test_the_converter_is_registered_under_a_name_profiles_can_select():
    assert "nemotron_h" in qm._STATE_CONVERTERS
    blk, _ = _moe_block()
    state = _per_expert_state(blk.mixer.experts)
    via_profile = qm._convert_layer_state(state, {"convert_state": "nemotron_h"})
    assert "mixer.experts.up_proj" in via_profile


def test_the_kolibri_converter_is_not_reused_for_this(monkeypatch):
    """Kolibri's regex also matches `experts.N.up_proj.weight`, but its completeness check
    demands a gate_proj — so routing Nemotron through it raises. Pinned so a later refactor
    does not "simplify" the two into one."""
    blk, _ = _moe_block()
    with pytest.raises(ValueError):
        qm._convert_state_kolibri1(_per_expert_state(blk.mixer.experts))


# ---- the virtual-Linear expansion --------------------------------------------------


def test_the_expansion_makes_one_virtual_linear_per_expert_per_projection():
    blk, _ = _moe_block()
    emod = blk.mixer.experts
    linears, writeback, dropped = {}, {}, set()

    qm._expand_nongated_experts([("mixer.experts", emod)], linears, writeback, dropped,
                                sd_prefix="backbone.layers", layer_idx=1)

    assert len(linears) == 2 * 8
    for e in range(8):
        up = linears[f"mixer.experts.{e}.up_proj"]
        dn = linears[f"mixer.experts.{e}.down_proj"]
        assert isinstance(up, nn.Linear) and isinstance(dn, nn.Linear)
        # [out, in]: up is latent->inter, down is inter->latent.
        assert tuple(up.weight.shape) == (32, 16)
        assert tuple(dn.weight.shape) == (16, 32)
        assert torch.equal(up.weight.data, emod.up_proj.data[e])
        assert torch.equal(dn.weight.data, emod.down_proj.data[e])


def test_the_expansion_names_match_the_checkpoint_keys():
    """The artifact prefix plus `.weight` must BE the source key, because that is what makes
    the save side drop the original without a special case — and it is what vLLM's
    `ckpt_names=("up_proj", "down_proj", "")` expects to read back."""
    blk, _ = _moe_block()
    ms = load_shapes(SLUG)
    linears = {}
    qm._expand_nongated_experts([("mixer.experts", blk.mixer.experts)], linears, {}, set(),
                                sd_prefix="language_model.backbone.layers", layer_idx=1)
    for name in linears:
        key = f"language_model.backbone.layers.1.{name}.weight"
        assert ms.shape_of(key) is not None, f"{key} is not a real checkpoint key"


def test_the_expansion_registers_writeback_into_the_3d_slice():
    """Without writeback the post-quant forward uses the ORIGINAL weights, so every later
    layer calibrates against a distribution the finished checkpoint never produces."""
    blk, _ = _moe_block()
    emod = blk.mixer.experts
    linears, writeback = {}, {}
    qm._expand_nongated_experts([("mixer.experts", emod)], linears, writeback, set(),
                                sd_prefix="backbone.layers", layer_idx=1)
    assert len(writeback) == 2 * 8
    param, idx = writeback["mixer.experts.3.up_proj"]
    assert param is emod.up_proj and idx == 3


def test_the_expansion_drops_the_stacked_originals_from_the_save():
    blk, _ = _moe_block()
    dropped = set()
    qm._expand_nongated_experts([("mixer.experts", blk.mixer.experts)], {}, {}, dropped,
                                sd_prefix="backbone.layers", layer_idx=7)
    assert dropped == {"backbone.layers.7.mixer.experts.up_proj",
                       "backbone.layers.7.mixer.experts.down_proj"}


def test_the_expansion_takes_a_copy_so_writeback_cannot_corrupt_the_source():
    """The quant loop reads `linears[name].weight.data` and writeback assigns into
    `param.data[e]`. If the virtual Linear were a view, quantizing expert 0 would change the
    input that expert 0's own Hessian was built from."""
    blk, _ = _moe_block()
    emod = blk.mixer.experts
    linears = {}
    qm._expand_nongated_experts([("mixer.experts", emod)], linears, {}, set(),
                                sd_prefix="backbone.layers", layer_idx=1)
    before = emod.up_proj.data[0].clone()
    linears["mixer.experts.0.up_proj"].weight.data.zero_()
    assert torch.equal(emod.up_proj.data[0], before), "the virtual Linear aliases the source"


# ---- replay / resume ---------------------------------------------------------------


class _FakeArts(dict):
    """Stands in for a stored artifact bundle; `_restore_replayed_weights` only passes it
    through to `dequantize_fn`, which the test supplies."""


def test_replay_restores_non_gated_experts():
    """Resume replays already-quantized layers to rebuild the activation distribution. The
    gated branch looks up `.gate_proj`/`.up_proj` pairs and finds nothing here, so without a
    non-gated branch a replayed MoE layer silently carries bf16 experts into calibration — and
    at 52-88 layers on a spot box, resume WILL run."""
    blk, _ = _moe_block()
    emod = blk.mixer.experts
    arts = {}
    for e in range(8):
        arts[f"L1.mixer.experts.{e}.up_proj"] = _FakeArts(tag=("up", e))
        arts[f"L1.mixer.experts.{e}.down_proj"] = _FakeArts(tag=("down", e))

    def fake_dequant(a, in_f, out_f, codebook_type, codebook=None):
        kind, e = a["tag"]
        return torch.full((out_f, in_f), float(e + 1) * (1 if kind == "up" else -1))

    n = qm._restore_replayed_weights(blk, arts, "L1", "trellis", None,
                                     dequantize_fn=fake_dequant)

    assert n == 2 * 8, f"restored {n}, expected 16"
    for e in range(8):
        assert torch.allclose(emod.up_proj.data[e], torch.full((32, 16), float(e + 1)))
        assert torch.allclose(emod.down_proj.data[e], torch.full((16, 32), -float(e + 1)))


def test_replay_still_restores_gated_experts_unchanged():
    """The regression that matters: gemma-4, Qwen4Exp and Kolibri-1 all go through the gated
    branch, and `tests/test_resume_replay_experts.py` pins its count."""
    layer = _gated_container(n_exp=4, inter=8, hidden=6)
    emod = layer.mixer.experts
    arts = {}
    for e in range(4):
        arts[f"L0.mixer.experts.{e}.gate_proj"] = _FakeArts(v=1.0)
        arts[f"L0.mixer.experts.{e}.up_proj"] = _FakeArts(v=2.0)
        arts[f"L0.mixer.experts.{e}.down_proj"] = _FakeArts(v=3.0)

    def fake_dequant(a, in_f, out_f, codebook_type, codebook=None):
        return torch.full((out_f, in_f), a["v"])

    n = qm._restore_replayed_weights(layer, arts, "L0", "trellis", None,
                                     dequantize_fn=fake_dequant)
    assert n == 2 * 4
    assert torch.allclose(emod.gate_up_proj.data[0][:8], torch.full((8, 6), 1.0))
    assert torch.allclose(emod.gate_up_proj.data[0][8:], torch.full((8, 6), 2.0))


# ---- the per-expert Hessian hook ---------------------------------------------------


def test_the_hook_keys_the_up_hessian_and_builds_the_down_input_through_the_activation():
    """The mechanism, not just the key names. For a gated container the down-projection's
    input is `act(gate) * up`; here it is `act(up(x))` with no multiply, and getting that
    wrong gives a plausible-looking Hessian for the wrong distribution — invisible to SQNR.
    """
    blk, _ = _moe_block()
    emod = blk.mixer.experts
    tokens, hess = {}, {}
    hook = qm._make_nongated_expert_hook("mixer.experts", emod, tokens, hess)

    torch.manual_seed(0)
    hs = torch.randn(5, 16)                      # 5 token rows in the LATENT width
    ti = torch.tensor([[0, 1], [0, 2], [1, 2], [0, 1], [3, 0]])
    hook(emod, (hs, ti), {}, None)

    assert hess["mixer.experts.0.up_proj"].shape == (16, 16)
    assert hess["mixer.experts.0.down_proj"].shape == (32, 32)
    assert tokens[("mixer.experts", 0)] == 4     # rows 0,1,3,4 selected expert 0
    assert tokens[("mixer.experts", 3)] == 1

    # Recompute expert 0's down-proj Gram by hand, through the real activation.
    sel = (ti == 0).any(dim=-1)
    xs = hs[sel]
    want_up = xs.float().t() @ xs.float()
    di = emod.act_fn(nn.functional.linear(xs, emod.up_proj[0])).float()
    want_down = di.t() @ di
    assert torch.allclose(hess["mixer.experts.0.up_proj"], want_up, atol=1e-5)
    assert torch.allclose(hess["mixer.experts.0.down_proj"], want_down, atol=1e-5)


def test_the_hook_counts_an_expert_that_saw_nothing():
    """A never-routed expert falls back to an identity Hessian — plain nearest-neighbour, no
    error feedback — and that is the single most useful thing to know about a calibration run.
    It is reported only if counted."""
    blk, _ = _moe_block()
    emod = blk.mixer.experts
    tokens, hess = {}, {}
    hook = qm._make_nongated_expert_hook("mixer.experts", emod, tokens, hess)
    hook(emod, (torch.randn(2, 16), torch.tensor([[0, 1], [0, 1]])), {}, None)

    assert tokens[("mixer.experts", 7)] == 0
    assert "mixer.experts.7.up_proj" not in hess, "built a Gram from no rows"


def test_the_hook_accumulates_across_calibration_samples():
    blk, _ = _moe_block()
    emod = blk.mixer.experts
    tokens, hess = {}, {}
    hook = qm._make_nongated_expert_hook("mixer.experts", emod, tokens, hess)
    hs, ti = torch.randn(3, 16), torch.tensor([[0, 1], [0, 1], [0, 1]])
    hook(emod, (hs, ti), {}, None)
    one = hess["mixer.experts.0.up_proj"].clone()
    hook(emod, (hs, ti), {}, None)
    assert torch.allclose(hess["mixer.experts.0.up_proj"], 2 * one, atol=1e-5)
    assert tokens[("mixer.experts", 0)] == 6


def test_the_hook_reads_keyword_args_too():
    """`NemotronHExperts.forward(hidden_states, top_k_index, top_k_weights)` is called
    positionally by `NemotronHMoE`, but a hook must not depend on that."""
    blk, _ = _moe_block()
    emod = blk.mixer.experts
    tokens, hess = {}, {}
    hook = qm._make_nongated_expert_hook("mixer.experts", emod, tokens, hess)
    hook(emod, (), {"hidden_states": torch.randn(2, 16),
                    "top_k_index": torch.tensor([[0, 1], [0, 1]])}, None)
    assert tokens[("mixer.experts", 0)] == 2


def test_the_hook_is_a_no_op_when_the_forward_shape_is_unexpected():
    blk, _ = _moe_block()
    tokens, hess = {}, {}
    hook = qm._make_nongated_expert_hook("mixer.experts", blk.mixer.experts, tokens, hess)
    hook(blk.mixer.experts, (), {}, None)
    assert tokens == {} and hess == {}


def test_the_hook_runs_under_a_real_forward_of_the_real_block():
    """End to end on the architecture: register it the way `quantize()` does and drive the
    block's own forward, so the arg-extraction contract is tested against the real caller
    rather than against this file's idea of it."""
    blk, cfg = _moe_block()
    emod = blk.mixer.experts
    tokens, hess = {}, {}
    h = emod.register_forward_hook(
        qm._make_nongated_expert_hook("mixer.experts", emod, tokens, hess), with_kwargs=True)
    try:
        with torch.no_grad():
            blk(torch.randn(1, 6, 64), position_ids=torch.arange(6).unsqueeze(0),
                cache_position=torch.arange(6), use_cache=False)
    finally:
        h.remove()

    assert sum(tokens.values()) == 6 * cfg.num_experts_per_tok, tokens
    routed = [k for k in hess if k.endswith(".up_proj")]
    assert routed, "no expert Hessian captured from a real forward"
    for k in routed:
        assert hess[k].shape == (16, 16)
        assert hess[k.replace(".up_proj", ".down_proj")].shape == (32, 32)
