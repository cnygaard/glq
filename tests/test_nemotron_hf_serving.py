"""Serving a GLQ **trellis** Nemotron checkpoint under HF transformers.

0.8.31 made these models quantizable. Serving them split by runtime, and only vLLM worked:
`glq_vllm/fused_moe_method.py` has a `_create_weights_trellis` path that takes `is_gated`, and
vLLM's own `nemotron_h.py` renames `backbone.` -> `model.` for us. The HF path did not, and
`glq/fused_experts.py` said so in a comment rather than in code:

    codebook_type is threaded through deliberately: `_ExpertPair` drops it and always
    builds e8_shell buffers, so a trellis checkpoint would fail on shapes inside HF's
    loader with nothing naming GLQ.

`_ExpertPair` called `E8RHTLinear` without `codebook_type`, so it registered `Qidxs*`; the same
class built as trellis registers `tlut / trellis_packed / trellis_packed2`. Since trellis is
the default codebook, that is **every** checkpoint the quantizer produces.

The buffer names here are not invented. They are the ones in the live
Nemotron-3.5-Lightning-30B run: layer 6 holds 1,032 tensors named
``backbone.layers.6.mixer.experts.{e}.{up,down}_proj.{SU,SV,Wscale,trellis_packed}`` — 256
expert projections x 4 buffers, plus the shared expert's 8.

Scope: **correct, not fast.** The fused MoE kernel stays unreachable from HF for trellis, so
the forward falls back to the per-expert Python loop. HF is the parity oracle for validating
vLLM's numbers, not a performance path.
"""
from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from glq.fused_experts import (
    E8RHTFusedExperts,
    _replace_nemotron_h_experts,
)

#: The buffers a 3 bpw / 3inst trellis checkpoint actually carries, measured from the live run.
#: `tlut` is absent because 3INST is lookup-free; `trellis_packed2` and `inv_resid_scale2`
#: because 3 bpw is a native single-stage rate, not stacked RVQ.
CKPT_BUFFERS = ("SU", "SV", "Wscale", "trellis_packed")


def _cfg(*, n_exp=4, hidden=64, inter=32, latent=None, act="relu2"):
    """A flat config shaped like NemotronH's text config."""
    return types.SimpleNamespace(
        n_routed_experts=n_exp, hidden_size=hidden, moe_intermediate_size=inter,
        moe_latent_size=latent, mlp_hidden_act=act, model_type="nemotron_h")


# ---- the actual bug: codebook_type never reached the experts -----------------------


def test_a_trellis_container_registers_trellis_buffers_not_e8_ones():
    """The whole defect in one assertion. Without `codebook_type` reaching `_ExpertPair`,
    every expert registers `Qidxs*` and a trellis checkpoint cannot load."""
    mod = E8RHTFusedExperts(_cfg(), codebook_type="trellis")
    bufs = {n for n, _ in mod[0].up_proj.named_buffers()}

    assert "trellis_packed" in bufs, sorted(bufs)
    assert not any(n.startswith("Qidxs") for n in bufs), sorted(bufs)


def test_the_e8_default_is_unchanged():
    """The regression guard: e8 non-gated is the only Nemotron path shipping today."""
    mod = E8RHTFusedExperts(_cfg())
    bufs = {n for n, _ in mod[0].up_proj.named_buffers()}
    assert "Qidxs" in bufs and "trellis_packed" not in bufs, sorted(bufs)


def test_both_projections_get_the_codebook_not_just_the_first():
    mod = E8RHTFusedExperts(_cfg(), codebook_type="trellis")
    for proj in ("up_proj", "down_proj"):
        bufs = {n for n, _ in getattr(mod[0], proj).named_buffers()}
        assert "trellis_packed" in bufs, (proj, sorted(bufs))


def test_the_latent_case_sizes_the_experts_from_the_latent_not_the_hidden():
    """The VL model's experts sit behind `fc1_latent_proj`, so their input width is
    `moe_latent_size` (1024 on the 120B), not `hidden_size`. Already correct; pinned because
    getting it wrong produces a plausible module that mismatches on load."""
    mod = E8RHTFusedExperts(_cfg(hidden=64, inter=32, latent=16), codebook_type="trellis")
    assert mod.input_dim == 16
    assert mod[0].up_proj.in_features == 16
    assert mod[0].up_proj.out_features == 32
    assert mod[0].down_proj.in_features == 32
    assert mod[0].down_proj.out_features == 16


# ---- the real checkpoint's keys load, key for key ---------------------------------


def _ckpt_state_for(mod, prefix="", n_exp=4):
    """A state dict with the REAL buffer names, sized to the module so only names are
    under test. Mirrors what the loader sees after `backbone.` -> `model.`."""
    sd = {}
    for e in range(n_exp):
        for proj in ("up_proj", "down_proj"):
            lin = getattr(mod[e], proj)
            for buf in CKPT_BUFFERS:
                key = f"{prefix}{e}.{proj}.{buf}"
                sd[key] = torch.zeros_like(getattr(lin, buf))
    return sd


def test_every_checkpoint_buffer_name_exists_on_the_module():
    """Zero *unexpected* keys is the half that proves serving. The three the module has and
    the checkpoint lacks — `tlut`, `trellis_packed2`, `inv_resid_scale2` — are expected
    absences for a lookup-free 3INST single-stage build, and `_load_from_state_dict` injects
    zero defaults for them (see the note in `hf_integration._process_model_before_weight_loading`).
    """
    mod = E8RHTFusedExperts(_cfg(), codebook_type="trellis")
    sd = _ckpt_state_for(mod)

    result = mod.load_state_dict(sd, strict=False)
    assert list(result.unexpected_keys) == [], result.unexpected_keys
    missing = {k.rsplit(".", 1)[-1] for k in result.missing_keys}
    assert missing <= {"tlut", "trellis_packed2", "inv_resid_scale2"}, sorted(missing)


def test_an_e8_checkpoint_would_NOT_load_into_a_trellis_container():
    """The failure this change prevents, stated as a test: the two buffer sets are disjoint
    where it counts, so a codebook mismatch is a load error rather than silent garbage."""
    trellis = E8RHTFusedExperts(_cfg(), codebook_type="trellis")
    e8 = E8RHTFusedExperts(_cfg())
    e8_names = {n for n, _ in e8[0].up_proj.named_buffers()}
    trellis_names = {n for n, _ in trellis[0].up_proj.named_buffers()}
    assert "Qidxs" in e8_names - trellis_names
    assert "trellis_packed" in trellis_names - e8_names


# ---- the fused kernel must decline, not raise -------------------------------------


def test_try_build_stacked_declines_for_trellis_instead_of_raising():
    """`_try_build_stacked` read `e0.up_proj.Qidxs` unconditionally. On a trellis container
    that attribute does not exist, so the old code raised `AttributeError` from inside a
    helper whose contract is "return False on any unsupported case". Declining sends the
    caller to the per-expert Python loop, which applies `act_fn` itself and is correct."""
    mod = E8RHTFusedExperts(_cfg(), codebook_type="trellis")
    # Satisfy the first gate so the old code would reach the Qidxs access.
    mod[0].up_proj.codebook = object()

    assert mod._try_build_stacked() is False


def test_try_build_stacked_still_reaches_the_device_check_for_e8():
    """The e8 path must be untouched: it declines here only because this is a CPU box, not
    because the new guard swallowed it."""
    mod = E8RHTFusedExperts(_cfg())
    assert mod._try_build_stacked() is False          # no codebook attached yet
    mod[0].up_proj.codebook = object()
    assert mod._try_build_stacked() is False          # Qidxs is not on CUDA here
    assert hasattr(mod[0].up_proj, "Qidxs"), "e8 container lost its Qidxs buffer"


# ---- buffer sizing, so device_map does not see an empty model ----------------------


def test_packed_shapes_size_the_buffers_from_the_checkpoint_header():
    """Without this the buffers register at shape [0], a 13 GiB checkpoint looks empty, and
    `device_map="auto"` budgets against nothing. Same reason the gated path threads `packed`."""
    from glq.fused_experts import _nongated_packed_shapes

    shapes = {}
    for e in range(4):
        for proj, shp in (("up_proj", [19488, 48]), ("down_proj", [19488, 48])):
            shapes[f"backbone.layers.6.mixer.experts.{e}.{proj}"] = {"trellis_packed": shp}

    packed = _nongated_packed_shapes("model.layers.6.mixer.experts", 4, shapes)
    assert set(packed) == {0, 1, 2, 3}, packed
    assert packed[0]["up_proj"] == (19488, 48)
    assert packed[0]["down_proj"] == (19488, 48)


def test_packed_shapes_resolve_the_checkpoint_spelling_not_the_module_one():
    """The subtlety that is easiest to get silently wrong. `_collect_quantized_shapes` keys on
    the CHECKPOINT key, which for this family is `backbone.*`, while `named_modules()` yields
    `model.*` — transformers renames the prefix on load. A lookup using the module spelling
    finds nothing, the packed map comes back empty, and the symptom is a bad device map rather
    than an error."""
    from glq.fused_experts import _nongated_packed_shapes

    ckpt_spelled = {"backbone.layers.6.mixer.experts.0.up_proj": {"trellis_packed": [8, 4]}}
    got = _nongated_packed_shapes("model.layers.6.mixer.experts", 1, ckpt_spelled)
    assert got.get(0, {}).get("up_proj") == (8, 4), got

    # ...and a checkpoint already written with native prefixes still resolves, since the
    # renamer's own docstring notes the rename is a no-op on those.
    native = {"model.layers.6.mixer.experts.0.up_proj": {"trellis_packed": [8, 4]}}
    got = _nongated_packed_shapes("model.layers.6.mixer.experts", 1, native)
    assert got.get(0, {}).get("up_proj") == (8, 4), got


def test_packed_shapes_is_empty_rather_than_wrong_when_shapes_are_absent():
    from glq.fused_experts import _nongated_packed_shapes
    assert _nongated_packed_shapes("model.layers.0.mixer.experts", 4, None) == {}
    assert _nongated_packed_shapes("model.layers.0.mixer.experts", 4, {}) == {}


# ---- the replacement helper --------------------------------------------------------


def _nemotron_moe_model(n_exp=4, hidden=64, inter=32):
    """A real `NemotronHExperts` inside a minimal model, so the walk under test is the
    production one rather than a name match."""
    pytest.importorskip("transformers", minversion="5.18.0")
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHExperts

    experts = NemotronHExperts.__new__(NemotronHExperts)
    nn.Module.__init__(experts)
    experts.up_proj = nn.Parameter(torch.zeros(n_exp, inter, hidden))
    experts.down_proj = nn.Parameter(torch.zeros(n_exp, hidden, inter))

    inner = _cfg(n_exp=n_exp, hidden=hidden, inter=inter)
    model = nn.Module()
    model.config = types.SimpleNamespace(model_type="nemotron_h", text_config=inner)
    model.config.get_text_config = lambda: inner
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].mixer = nn.Module()
    model.model.layers[0].mixer.experts = experts
    return model


@pytest.fixture
def _window_ok(monkeypatch):
    """Neutralise the transformers-window guard for tests about the *replacement*.

    Not incidental: this development machine runs transformers **5.20.1**, which the guard
    correctly refuses — the GPU box is pinned to 5.19.0. So without this the two tests below
    fail for a reason that has nothing to do with what they assert.
    `test_the_replacement_checks_the_transformers_window` covers the call itself, so stubbing
    it here cannot hide the guard being dropped from the call site.
    """
    import glq.fused_experts as fe
    monkeypatch.setattr(fe, "_require_tested_transformers", lambda *a, **k: None)


def test_the_replacement_checks_the_transformers_window(monkeypatch):
    """The guard must be *called* by the replacement, and told whether this is the VL
    wrapper — the omni floor applies only there."""
    import glq.fused_experts as fe
    calls = []
    monkeypatch.setattr(fe, "_require_tested_transformers",
                        lambda *a, **k: calls.append(k.get("omni")))

    fe._replace_nemotron_h_experts(_nemotron_moe_model(), codebook_type="trellis")
    assert calls == [False], calls

    model = _nemotron_moe_model()
    model.config.model_type = "nemotron_h_omni"
    calls.clear()
    fe._replace_nemotron_h_experts(model, codebook_type="trellis")
    assert calls == [True], calls


def test_the_replacement_builds_trellis_experts_when_told_to(_window_ok):
    model = _nemotron_moe_model()
    n = _replace_nemotron_h_experts(model, codebook_type="trellis")
    assert n == 1
    built = model.get_submodule("model.layers.0.mixer.experts")
    assert isinstance(built, E8RHTFusedExperts)
    bufs = {nm for nm, _ in built[0].up_proj.named_buffers()}
    assert "trellis_packed" in bufs and not any(b.startswith("Qidxs") for b in bufs)


def test_the_replacement_builds_on_meta_so_a_30b_does_not_materialize_first(_window_ok):
    """Lightning-30B is 23 MoE layers x 128 experts x 2 = 5,888 E8RHTLinears, and HF
    materializes them from the checkpoint immediately afterwards. The gated helper already
    builds on meta for exactly this reason."""
    model = _nemotron_moe_model()
    _replace_nemotron_h_experts(model, codebook_type="trellis")
    built = model.get_submodule("model.layers.0.mixer.experts")
    assert built[0].up_proj.SU.is_meta, "experts were materialized on CPU"


def test_the_replacement_still_no_ops_without_experts():
    """The CI regression from 0.8.31: this helper runs against every model GLQ loads, most of
    which are not NemotronH and have no `.config` at all."""
    assert _replace_nemotron_h_experts(nn.Sequential(nn.Linear(8, 4))) == 0


# ---- the transformers window -------------------------------------------------------


def test_the_tested_window_refuses_versions_that_moved_nemotron():
    """5.20+ breaks coexistence with vLLM 0.32.0 (`transformers<5.20.0`), and **5.21.0**
    specifically added a Nemotron tensor-parallel plan (#48780) and fixed two Nemotron
    integration tests failing on `output_mismatch` (#48945) — so the module layout and the
    numerics both moved. A model that loads with quietly wrong output is worse than one that
    refuses, so this raises."""
    from glq.fused_experts import _require_tested_transformers

    for bad in ("5.20.0", "5.20.2", "5.21.0"):
        with pytest.raises(RuntimeError) as exc:
            _require_tested_transformers(bad)
        assert "5.19" in str(exc.value), str(exc.value)

    for ok in ("5.18.0", "5.19.0"):
        _require_tested_transformers(ok)


def test_the_window_is_compared_numerically_not_lexically():
    """`"5.9.0" > "5.18.0"` as strings. A string comparison would refuse 5.9 and accept
    5.21 — exactly backwards."""
    from glq.fused_experts import _require_tested_transformers
    with pytest.raises(RuntimeError):
        _require_tested_transformers("5.21.0")
    # 5.9 is below the omni floor but must be rejected as too OLD, not accepted as "< 5.20".
    with pytest.raises(RuntimeError):
        _require_tested_transformers("5.9.0", omni=True)


def test_the_omni_floor_applies_only_to_the_vl_wrapper():
    """`nemotron_h_omni` landed in transformers 5.18.0. The text-only architecture has been
    native far longer, so it must not inherit that floor."""
    from glq.fused_experts import _require_tested_transformers
    _require_tested_transformers("5.17.0")                 # text-only: fine
    with pytest.raises(RuntimeError, match="5.18"):
        _require_tested_transformers("5.17.0", omni=True)  # the VL wrapper is not


# ======================================================================================
# The `backbone.` / `model.` rename, and the nn.Linear walk it was silently missing
# ======================================================================================
#
# MEASURED, not theorised. Arm 6 of the Lightning-30B debug run loaded 14.42 GiB (so the
# MoE experts WERE replaced) and then failed in the Mamba mixer:
#
#   File ".../models/nemotron_h/modeling_nemotron_h.py", line 440, in forward
#     projected_states = self.in_proj(hidden_states)
#   File ".../torch/nn/modules/linear.py", line 134, in forward
#     return F.linear(input, self.weight, self.bias)
#   RuntimeError: expected mat1 and mat2 to have the same dtype, but got:
#                 c10::BFloat16 != c10::Half
#
# A plain `nn.Linear`, in a layer the checkpoint quantizes and `layer_bpw` lists. The
# experts survived because `_replace_nemotron_h_experts` matches on module TYPE; the mamba
# projections are matched by NAME, against a whitelist built from checkpoint keys
# (`backbone.*`) while `named_modules()` yields `model.*`. Every mamba projection missed.
#
# It hid behind the footprint: the experts are 10.47 of 14.65 GiB, so missing 0.31 GiB of
# mamba projections keeps the total inside any tolerance. This is the same bug class as the
# vLLM whitelist alias in `_lookup_bpw` and the expert packed-shape lookup above - the third
# site, which is why the aliasing now has one origin in `glq.name_aliases`.


def _mamba_mixer_model(hidden=64, out=128):
    """A model shaped like NemotronH's AFTER transformers renames the prefix: modules are
    `model.layers.N.*`, so the whitelist's `backbone.*` spelling has to be translated."""
    mixer = nn.Module()
    mixer.in_proj = nn.Linear(hidden, out, bias=False)
    mixer.out_proj = nn.Linear(out, hidden, bias=False)
    layer = nn.Module()
    layer.mixer = mixer
    inner = nn.Module()
    inner.layers = nn.ModuleList([layer])
    top = nn.Module()
    top.model = inner
    return top


def test_the_linear_walk_resolves_the_checkpoint_spelling():
    """The regression gate for the arm-6 crash: a `backbone.*` whitelist must reach the
    `model.*` modules, or they stay plain nn.Linear with no weight ever loaded."""
    from glq.hf_integration import replace_with_glq_linear
    from glq.quantized_linear import E8RHTLinear

    m = _mamba_mixer_model()
    whitelist = {"backbone.layers.0.mixer.in_proj", "backbone.layers.0.mixer.out_proj"}
    assert replace_with_glq_linear(m, quantized_layers=whitelist,
                                   codebook_type="trellis") is True
    assert isinstance(m.model.layers[0].mixer.in_proj, E8RHTLinear), (
        "mamba in_proj left as a plain nn.Linear — its GLQ weight will never load")
    assert isinstance(m.model.layers[0].mixer.out_proj, E8RHTLinear)


def test_a_layer_outside_the_whitelist_is_still_left_alone():
    """The failing control. Without it the test above would pass just as well if the
    aliasing degenerated into "replace everything", which would convert a multimodal
    model's unquantized vision tower and break it."""
    from glq.hf_integration import replace_with_glq_linear

    m = _mamba_mixer_model()
    replace_with_glq_linear(m, quantized_layers={"backbone.layers.0.mixer.in_proj"},
                            codebook_type="trellis")
    assert isinstance(m.model.layers[0].mixer.out_proj, nn.Linear) and not hasattr(
        m.model.layers[0].mixer.out_proj, "trellis_packed"), (
        "out_proj was converted despite being absent from the whitelist")


def test_the_packed_shapes_also_resolve_for_the_linear_walk():
    """Shapes are keyed the same way, so a translated name must find them too. Without
    this the buffers register at size 0 and accelerate's device map sees an empty model."""
    from glq.hf_integration import replace_with_glq_linear

    m = _mamba_mixer_model(hidden=64, out=128)
    shapes = {"backbone.layers.0.mixer.in_proj": {"trellis_packed": [96, 48]}}
    replace_with_glq_linear(m, quantized_layers={"backbone.layers.0.mixer.in_proj"},
                            codebook_type="trellis", shapes=shapes)
    assert tuple(m.model.layers[0].mixer.in_proj.trellis_packed.shape) == (96, 48)


@pytest.mark.parametrize("given,expected", [
    ("model.layers.0.mixer.in_proj", "backbone.layers.0.mixer.in_proj"),
    ("backbone.layers.0.mixer.in_proj", "model.layers.0.mixer.in_proj"),
    ("language_model.model.layers.0.mixer.in_proj",
     "language_model.backbone.layers.0.mixer.in_proj"),
    ("language_model.backbone.layers.0.mixer.in_proj",
     "language_model.model.layers.0.mixer.in_proj"),
])
def test_the_alias_helper_covers_both_directions_and_the_vl_nesting(given, expected):
    """One origin for the rename, since three call sites need it and the two that existed
    had already drifted — the expert-shape copy lacked the `language_model.` forms the VL
    wrapper needs."""
    from glq.name_aliases import backbone_aliases
    assert expected in backbone_aliases(given)


def test_a_name_with_no_alias_yields_nothing_rather_than_itself():
    """An empty result is what lets callers do `candidates = [name, *aliases]` without
    de-duplicating, and keeps a non-NemotronH model free of pointless lookups."""
    from glq.name_aliases import backbone_aliases
    assert backbone_aliases("model.layers.0.self_attn.q_proj") == {
        "backbone.layers.0.self_attn.q_proj"}
    assert backbone_aliases("visual.blocks.0.attn.qkv") == set()
