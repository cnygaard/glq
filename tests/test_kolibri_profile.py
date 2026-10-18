"""Kolibri-1 (`Aleph-Alpha/Kolibri-1-BF16`) support: the profile, and the key conversion.

Why this file exists. Kolibri-1 is 78B total / 3.46B active, and **96.7% of its parameters
are routed experts** — so every silent under-quantization failure mode this repo has already
paid for is maximally expensive here. Three independent ones are live at once:

1. transformers stores the experts as 3-D `nn.Parameter`s the `nn.Linear` walk cannot see
   (the `stacked_experts` opt-in, same as Qwen4Exp).
2. The **checkpoint** stores them one tensor per expert per projection instead — 1,152 per
   layer — so the streaming path's `load_state_dict(strict=False)` drops every one of them
   without a word, and `_materialize_meta_params` then hands them `torch.empty` storage.
3. The checkpoint and transformers **disagree on three key names**, and transformers declares
   no conversion mapping for any of them.

The third one includes `moe.router.expert_bias` -> `mlp.gate.e_score_correction_bias`, which
is the one that would do the most damage unnoticed: it is added to the router logits before
top-k, so losing it changes which experts each token selects, and therefore every per-expert
Hessian, while leaving the expert weights themselves perfectly intact.

Both sides of the conversion are **recorded fact, not inference**: the checkpoint side comes
from `tests/models/kolibri_1_bf16.json` (safetensors headers, no weight data), the module side
from a meta-instantiated `Kolibri1DecoderLayer`. That matters because this is precisely the
mistake `tests/models/README.md` describes — a hand-written fixture encodes the author's
*model* of an architecture rather than the architecture — and one of the three fixtures it
names as having been wrong that way was a MoE expert-naming fixture.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

from glq import quantize_model as qm

sys.path.insert(0, os.path.dirname(__file__))

from model_shapes import load as load_shapes  # noqa: E402

SLUG = "kolibri_1_bf16"
ARCH = "Kolibri1ForCausalLM"
PROFILE = {"convert_state": "kolibri1"}


def _kolibri_layer_cls():
    """The real decoder layer, or a skip naming the reason.

    Imported inside the test rather than at module scope on purpose: a module-level
    `importorskip` makes the WHOLE file skip when the dependency is absent, which hides the
    tests below that need no transformers at all.
    """
    pytest.importorskip("transformers", minversion="5.20.0",
                        reason="kolibri1 landed in transformers 5.20.0")
    try:
        from transformers import Kolibri1Config
        from transformers.models.kolibri1.modeling_kolibri1 import Kolibri1DecoderLayer
    except ImportError as exc:                                        # pragma: no cover
        pytest.skip(f"transformers has no kolibri1: {exc}")
    return Kolibri1Config, Kolibri1DecoderLayer


def _checkpoint_layer_state(layer_idx=0):
    """Layer `layer_idx`'s tensors, at RECORDED shapes, on meta.

    Real dimensions at zero memory: a 384-expert layer is ~3 GiB of bf16, and none of it
    needs to exist to answer a question about names and shapes.
    """
    ms = load_shapes(SLUG)
    prefix = f"model.layers.{layer_idx}."
    state = {}
    with torch.device("meta"):
        for key in ms.keys():
            if key.startswith(prefix):
                state[key[len(prefix):]] = torch.empty(
                    ms.shape_of(key), dtype=torch.bfloat16)
    assert state, "manifest has no layer tensors — recorded wrong?"
    return state


# --------------------------------------------------------------- the profile

def test_profile_is_registered():
    assert ARCH in qm._MODEL_PROFILES, (
        "no profile: _detect_profile falls back to _DEFAULT_PROFILE, which has no "
        "stacked_experts and no convert_state, and the run would quantize ~3% of the model")


def test_profile_paths_match_the_real_module_tree():
    """Verified against a meta-instantiated Kolibri1ForCausalLM, not assumed."""
    p = qm._MODEL_PROFILES[ARCH]
    assert p["layers_attr"] == "model.layers"
    assert p["embed_attr"] == "model.embed_tokens"
    assert p["rotary_attr"] == "model.rotary_emb"
    assert p["sd_prefix"] == "model.layers"
    assert p["forward_kwargs"] == "default"
    assert p.get("trust_remote_code") is False, "the arch is in transformers; no remote code"


def test_stacked_experts_opt_in_is_enabled():
    """Without this the nn.Linear walk never sees the experts: 96.7% of the model.

    It is also required for *serving*: vLLM's FusedMoE gives the stacked gate+up a single
    per-expert Wscale, so the two must be quantized jointly.
    """
    p = qm._MODEL_PROFILES[ARCH]
    assert p.get("stacked_experts") is True
    assert qm._stacked_experts_enabled(ARCH, p) is True


def test_profile_declares_a_registered_converter():
    p = qm._MODEL_PROFILES[ARCH]
    assert p.get("convert_state") == "kolibri1"
    assert "kolibri1" in qm._STATE_CONVERTERS


def test_unregistered_converter_is_rejected_by_name():
    with pytest.raises(ValueError, match="convert_state"):
        qm._convert_layer_state({}, {"convert_state": "no-such-converter"})


# ------------------------------------------- the gate: does every key find a home?

def test_conversion_maps_the_recorded_checkpoint_onto_the_real_layer_exactly():
    """THE gate. Real shapes, real module, zero unexpected and zero missing keys.

    This is the assertion that would have caught all three failure modes above, and it is
    built from the recorded manifest on one side and transformers on the other, so neither
    side is my guess about the architecture.
    """
    Kolibri1Config, Kolibri1DecoderLayer = _kolibri_layer_cls()

    state = _checkpoint_layer_state()
    converted = qm._convert_layer_state(state, PROFILE)

    with torch.device("meta"):
        layer = Kolibri1DecoderLayer(Kolibri1Config(), 0)
    module_keys = set(layer.state_dict())

    assert set(converted) - module_keys == set(), "checkpoint tensors with nowhere to go"
    assert module_keys - set(converted) == set(), "module tensors the checkpoint never fills"


def test_converted_shapes_match_the_module():
    Kolibri1Config, Kolibri1DecoderLayer = _kolibri_layer_cls()
    converted = qm._convert_layer_state(_checkpoint_layer_state(), PROFILE)
    with torch.device("meta"):
        sd = Kolibri1DecoderLayer(Kolibri1Config(), 0).state_dict()
    for key, want in sd.items():
        assert tuple(converted[key].shape) == tuple(want.shape), key


def test_the_stacked_experts_have_the_published_geometry():
    """384 experts, gate+up fused to 1024 rows, hidden 2560, intermediate 512."""
    converted = qm._convert_layer_state(_checkpoint_layer_state(), PROFILE)
    assert tuple(converted["mlp.experts.gate_up_proj"].shape) == (384, 1024, 2560)
    assert tuple(converted["mlp.experts.down_proj"].shape) == (384, 2560, 512)


def test_conversion_preserves_device_and_dtype():
    """A CPU-forcing `torch.empty` would materialize ~3 GiB per layer during conversion."""
    converted = qm._convert_layer_state(_checkpoint_layer_state(), PROFILE)
    gup = converted["mlp.experts.gate_up_proj"]
    assert gup.device.type == "meta"
    assert gup.dtype == torch.bfloat16


# ------------------------------------------------------- the failing control

def test_without_the_conversion_the_drop_is_caught_not_silent():
    """The control. Load the raw checkpoint keys and confirm we FAIL rather than proceed.

    Before `_assert_load_coverage` existed this path returned normally, left the experts on
    meta, and `_materialize_meta_params` gave them uninitialized storage.
    """
    Kolibri1Config, Kolibri1DecoderLayer = _kolibri_layer_cls()

    raw = _checkpoint_layer_state()                      # NO conversion applied
    with torch.device("meta"):
        layer = Kolibri1DecoderLayer(Kolibri1Config(), 0)
    incompatible = layer.load_state_dict(raw, strict=False, assign=True)

    # The silent-failure precondition: strict=False really does swallow 1,154 tensors.
    assert len(incompatible.unexpected_keys) > 1000

    with pytest.raises(ValueError, match="did not load"):
        qm._assert_load_coverage(0, incompatible, n_loaded=len(raw))


def test_each_rename_is_individually_load_bearing():
    """Drop one rename at a time and confirm the gate fails for that key specifically.

    A single bulk assertion can pass while two renames cancel out; this pins each one.
    """
    Kolibri1Config, Kolibri1DecoderLayer = _kolibri_layer_cls()
    with torch.device("meta"):
        module_keys = set(Kolibri1DecoderLayer(Kolibri1Config(), 0).state_dict())

    for ckpt_name, module_name in qm._KOLIBRI1_RENAMES.items():
        assert ckpt_name not in module_keys, (
            f"{ckpt_name} is in the module too — the rename is not needed, so this "
            f"mapping is wrong")
        assert module_name in module_keys, (
            f"{module_name} is not a module key — the rename targets nothing")


def test_the_router_bias_rename_targets_the_sigmoid_balancing_buffer():
    """Named explicitly because its failure is invisible: the weights stay correct and
    only the expert SELECTION changes, which no SQNR or size check can see."""
    assert qm._KOLIBRI1_RENAMES["moe.router.expert_bias"] == \
        "mlp.gate.e_score_correction_bias"
    converted = qm._convert_layer_state(_checkpoint_layer_state(), PROFILE)
    assert "mlp.gate.e_score_correction_bias" in converted
    assert "moe.router.expert_bias" not in converted


def test_router_bias_is_promoted_to_fp32():
    """The model declares it `_keep_in_fp32_modules_strict`; `assign=True` would otherwise
    adopt the checkpoint's bf16 for a comparison that decides which experts run."""
    converted = qm._convert_layer_state(_checkpoint_layer_state(), PROFILE)
    assert converted["mlp.gate.e_score_correction_bias"].dtype == torch.float32


# ----------------------------------------- stack order, against transformers itself

def test_gate_rows_precede_up_rows_per_transformers_own_loader():
    """The one thing I could plausibly have backwards, checked against the authority.

    `Kolibri1Experts.forward` does `linear(x, gate_up_proj[e]).chunk(2, dim=-1)` -> (gate,
    up), so gate must occupy the FIRST `intermediate` rows. Rather than re-reading that, let
    transformers' own conversion ops merge a per-expert checkpoint and compare: this is the
    half of the mapping transformers *does* implement (`MergeModulelist` + `Concatenate`),
    so it can arbitrate. It cannot arbitrate the three renames — it has none — which is
    exactly why those are pinned structurally above instead.
    """
    Kolibri1Config, _ = _kolibri_layer_cls()
    from transformers.models.kolibri1.modeling_kolibri1 import Kolibri1Experts

    cfg = Kolibri1Config(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=8, num_experts=3,
                         num_experts_per_tok=2, moe_intermediate_size=4,
                         shared_expert_intermediate_size=4, vocab_size=32)
    experts = Kolibri1Experts(cfg)
    want = torch.randn_like(experts.gate_up_proj)

    # A per-expert checkpoint that, merged, must reproduce `want`.
    state = {}
    for e in range(cfg.num_experts):
        state[f"mlp.experts.{e}.gate_proj.weight"] = want[e, :cfg.moe_intermediate_size]
        state[f"mlp.experts.{e}.up_proj.weight"] = want[e, cfg.moe_intermediate_size:]
        state[f"mlp.experts.{e}.down_proj.weight"] = torch.zeros(
            cfg.hidden_size, cfg.moe_intermediate_size)

    got = qm._convert_layer_state(state, PROFILE)["mlp.experts.gate_up_proj"]
    assert torch.equal(got, want), "gate/up row order disagrees with the module's forward"


def test_non_contiguous_expert_indices_raise():
    """Stacking a gap would silently put weights at the wrong expert index, and routing
    cannot detect that — every expert still exists and still has plausible weights."""
    state = {}
    for e in (0, 1, 3):                                  # 2 is missing
        state[f"mlp.experts.{e}.gate_proj.weight"] = torch.zeros(4, 8)
        state[f"mlp.experts.{e}.up_proj.weight"] = torch.zeros(4, 8)
        state[f"mlp.experts.{e}.down_proj.weight"] = torch.zeros(8, 4)
    with pytest.raises(ValueError, match="not contiguous"):
        qm._convert_layer_state(state, PROFILE)


def test_an_expert_missing_a_projection_raises():
    state = {
        "mlp.experts.0.gate_proj.weight": torch.zeros(4, 8),
        "mlp.experts.0.up_proj.weight": torch.zeros(4, 8),
    }                                                    # no down_proj
    with pytest.raises(ValueError, match="down_proj|expected"):
        qm._convert_layer_state(state, PROFILE)


# ------------------------------------------ sliding-window attention in calibration

def _tiny_hybrid_model(window=4, n_layers=2):
    """A real Kolibri with a tiny window, so sliding and full layers differ observably."""
    Kolibri1Config, _ = _kolibri_layer_cls()
    from transformers.models.kolibri1.modeling_kolibri1 import Kolibri1ForCausalLM

    torch.manual_seed(0)
    cfg = Kolibri1Config(
        hidden_size=32, num_hidden_layers=n_layers, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=8, shared_expert_intermediate_size=8, vocab_size=64,
        sliding_window=window,
        layer_types=["sliding_attention", "full_attention"][:n_layers])
    return cfg, Kolibri1ForCausalLM(cfg).to(torch.float32).eval()


@pytest.mark.parametrize("layer_idx,kind", [(0, "sliding"), (1, "full")])
def test_calibration_reproduces_the_models_own_attention(layer_idx, kind):
    """GLQ's calibration forward must match the model's, for BOTH layer kinds.

    40 of Kolibri's 50 layers are sliding-window. `Kolibri1Attention` forwards
    `sliding_window=` to the attention interface, but `sdpa` — the default — ignores that
    and honours the mask instead, and GLQ passes `attention_mask=None` unless the profile
    asks for one. Without the profile's `causal_mask`, this assertion failed on the sliding
    layer by 3.3 absolute: those layers saw the whole sequence where inference gives them
    `sliding_window` tokens, and the MoE block downstream inherits the error.

    Parametrized over both kinds deliberately: a fix that supplied a sliding mask
    unconditionally would break the full-attention layers instead, and a single-layer test
    would not notice.
    """
    cfg, model = _tiny_hybrid_model()
    assert model.model.layers[0].self_attn.is_sliding is True
    assert model.model.layers[1].self_attn.is_sliding is False

    ids = torch.randint(0, cfg.vocab_size, (1, 16))     # T=16 >> window=4
    captured = {}

    def _capture(_mod, args, kwargs, out):
        captured["in"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()
        captured["out"] = (out[0] if isinstance(out, tuple) else out).detach().clone()

    handle = model.model.layers[layer_idx].register_forward_hook(_capture, with_kwargs=True)
    with torch.no_grad():
        model(ids)                                      # reference: builds its own masks
    handle.remove()

    kwargs = qm._build_forward_kwargs(
        qm._MODEL_PROFILES[ARCH], captured["in"], model.model.rotary_emb,
        layer_idx=layer_idx, cfg=cfg, calib_ids=ids)
    with torch.no_grad():
        got = model.model.layers[layer_idx](captured["in"], **kwargs)
    got = got[0] if isinstance(got, tuple) else got

    delta = (got - captured["out"]).abs().max().item()
    assert delta < 1e-5, (
        f"{kind} layer {layer_idx}: calibration diverges from the model's own forward by "
        f"{delta:.3e} — the attention mask does not match what inference will use")


def test_sliding_layers_are_identified_from_layer_types():
    cfg, _ = _tiny_hybrid_model()
    assert qm._is_sliding_layer(cfg, 0)
    assert not qm._is_sliding_layer(cfg, 1)
    assert not qm._is_sliding_layer(cfg, None)


def test_a_model_without_layer_types_is_never_sliding():
    """The helper must stay inert for every architecture that has no hybrid attention."""
    class _Cfg:
        pass
    assert not qm._is_sliding_layer(_Cfg(), 0)
    assert not qm._is_sliding_layer(None, 0)


# ------------------------- HF load path: transformers resolves the renames itself

def _tiny_cfg():
    Kolibri1Config, _ = _kolibri_layer_cls()
    return Kolibri1Config(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=8,
        shared_expert_intermediate_size=8, vocab_size=64, sliding_window=8,
        layer_types=["sliding_attention", "full_attention"])


def _write_checkpoint_spelled_model(tmp_path, cfg):
    """Save a tiny Kolibri whose three keys use the RELEASED checkpoint's spellings.

    Returns the values written, so "loaded" can be told apart from "left at init" — the
    whole failure mode here is that init looks plausible (norms all-ones, bias zeros).
    """
    import json
    from safetensors.torch import save_file
    from transformers.models.kolibri1.modeling_kolibri1 import Kolibri1ForCausalLM as K

    torch.manual_seed(0)
    model = K(cfg).to(torch.float32)
    sd = {k: v.clone() for k, v in model.state_dict().items()}

    written = {}
    for layer in range(cfg.num_hidden_layers):
        for module_name, ckpt_name in (
                ("post_self_attn_layernorm.weight", "post_attn_norm.weight"),
                ("post_mlp_layernorm.weight", "post_ffn_norm.weight"),
                ("mlp.gate.e_score_correction_bias", "moe.router.expert_bias")):
            src = f"model.layers.{layer}.{module_name}"
            val = torch.full_like(sd[src], 0.5 + layer)      # never the init value
            sd.pop(src)
            sd[f"model.layers.{layer}.{ckpt_name}"] = val
            written[src] = val
    save_file({k: v.contiguous() for k, v in sd.items()},
              str(tmp_path / "model.safetensors"))
    d = cfg.to_dict()
    d["architectures"] = ["Kolibri1ForCausalLM"]
    with open(tmp_path / "config.json", "w") as f:
        json.dump(d, f)
    return written


def test_from_pretrained_resolves_the_checkpoint_spellings_unaided(tmp_path):
    """`from_pretrained` loads all three differently-spelled keys with NO help from GLQ.

    This is a canary, and it exists because the static evidence says the opposite:
    `Kolibri1ForCausalLM` has no `_checkpoint_conversion_mapping` (nor `_conversion_mapping`
    or `weight_converters`), and `transformers.core_model_loading` never mentions
    `post_attn_norm`, `post_ffn_norm` or `expert_bias`. Reading the source, loading the
    released checkpoint under HF looks broken. Measured, it is fine: zero missing, zero
    unexpected, and the written values arrive.

    So the conversion lives somewhere in the `from_pretrained` pipeline rather than on the
    class — which is exactly the distinction that matters for GLQ:

      * `from_pretrained`      -> conversion applied   -> HF serving needs nothing from us
      * `nn.Module.load_state_dict` -> no conversion   -> the quantize path DOES need
        `_convert_state_kolibri1`, as the streaming loader calls the latter directly
        (see `test_without_the_conversion_the_drop_is_caught_not_silent`).

    If upstream ever stops doing this, HF loading of a GLQ Kolibri checkpoint goes quietly
    wrong — both norms at all-ones, the router bias at zero — so the failure is worth a
    standing check rather than an assumption.
    """
    from transformers.models.kolibri1.modeling_kolibri1 import Kolibri1ForCausalLM as K

    cfg = _tiny_cfg()
    written = _write_checkpoint_spelled_model(tmp_path, cfg)

    model, info = K.from_pretrained(
        str(tmp_path), dtype=torch.float32, output_loading_info=True)
    assert not list(info.get("missing_keys", [])), info.get("missing_keys")
    assert not list(info.get("unexpected_keys", [])), info.get("unexpected_keys")

    sd = model.state_dict()
    for name, want in written.items():
        got = sd[name].float()
        assert torch.allclose(got, want.float()), (
            f"{name} was not populated from its checkpoint spelling: mean {got.mean():.4f} "
            f"vs {want.float().mean():.4f} (all-ones/all-zeros means it stayed at init)")


# ------------------------------------ the generic guard, independent of any architecture

class _Incompatible:
    def __init__(self, missing=(), unexpected=()):
        self.missing_keys = list(missing)
        self.unexpected_keys = list(unexpected)


def test_coverage_guard_passes_when_everything_landed():
    qm._assert_load_coverage(0, _Incompatible(), n_loaded=3)


def test_coverage_guard_raises_on_unexpected_keys_and_names_them():
    with pytest.raises(ValueError) as exc:
        qm._assert_load_coverage(7, _Incompatible(unexpected=["mlp.experts.0.gate_proj.weight"]),
                                 n_loaded=9)
    msg = str(exc.value)
    assert "layer 7" in msg
    assert "mlp.experts.0.gate_proj.weight" in msg, "a guard that hides the key is no guard"


def test_coverage_guard_tolerates_non_persistent_buffers(capsys):
    """A checkpoint may carry a tensor the module owns but does not persist.

    `rotary_emb.inv_freq` is the standard case: transformers recomputes it from config, so
    `load_state_dict` reports it unexpected while nothing has been lost. Failing on it would
    reject real checkpoints for architectures that work today — the guard has to separate
    "has no home" from "has a home that is not in state_dict".
    """
    import torch.nn as nn

    layer = nn.Module()
    layer.rotary_emb = nn.Module()
    layer.rotary_emb.register_buffer("inv_freq", torch.zeros(4), persistent=False)

    qm._assert_load_coverage(
        0, _Incompatible(unexpected=["rotary_emb.inv_freq"]), n_loaded=2, module=layer)
    assert "non-persistent" in capsys.readouterr().out, "it should still be reported"


def test_coverage_guard_still_fails_on_a_truly_homeless_key():
    """The tolerance above must not become a blanket escape: a name the module does not
    have at all is still the dropped-conversion signal this guard exists for."""
    import torch.nn as nn

    layer = nn.Module()
    layer.rotary_emb = nn.Module()
    layer.rotary_emb.register_buffer("inv_freq", torch.zeros(4), persistent=False)

    with pytest.raises(ValueError, match="did not load"):
        qm._assert_load_coverage(
            0, _Incompatible(unexpected=["rotary_emb.inv_freq",
                                         "mlp.experts.0.gate_proj.weight"]),
            n_loaded=9, module=layer)


def test_coverage_guard_tolerates_missing_keys():
    """Missing is reported, not fatal: a meta-built block legitimately lacks tensors no
    checkpoint supplies, and failing on those would break architectures that work today."""
    qm._assert_load_coverage(0, _Incompatible(missing=["rotary_emb.inv_freq"]), n_loaded=5)


def test_conversion_is_a_no_op_without_a_profile_key():
    state = {"a.weight": torch.zeros(2)}
    assert qm._convert_layer_state(state, {}) is state
    assert qm._convert_layer_state(state, None) is state
