"""GLM-5.3-Flash (`zai-org/GLM-5.3-Flash-BF16`) support: profile, key conversion, calibration.

GLM-5.3-Flash is 314B total, and the routed experts are 304.4B of it (97%), so every silent
under-quantization failure this repo has paid for before is live and maximally expensive:

1. transformers stores the experts as 3-D `nn.Parameter`s the `nn.Linear` walk cannot see
   (the `stacked_experts` opt-in);
2. the **checkpoint** stores them one tensor per expert per projection, and spells the
   hyper-connection, forget-gate and short-conv tensors differently from the module. The
   streaming path's own `load_state_dict` applies none of transformers' conversions, so
   without `_convert_state_glm5_next` they would all be dropped;
3. the decoder takes a **4-D** input, `[B, S, hc_mult, D]`, and the sparse-attention indexer
   dereferences a 2-D boolean `attention_mask` that the default calibration kwargs omit;
4. the expert activation is a **clamped** SwiGLU (`swiglu_limit` 10) applied through
   `_apply_gate`; the container has no `act_fn`, which is what the expert Hessian hook used.

The authorities are recorded, not assumed. The checkpoint side is
`tests/models/glm_5_3_flash_bf16.json` (safetensors headers). The module side is a
meta-built `Glm5NextTextDecoderLayer`. The mapping between them is arbitrated by transformers
itself: `save_pretrained` writes the checkpoint spelling, so a tiny model round-trips through
it without anyone hand-writing the inverse.
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

SLUG = "glm_5_3_flash_bf16"
ARCH = "Glm5NextForConditionalGeneration"
SD_PREFIX = "model.language_model.layers"
#: The checkpoint's own `index_kpool`; transformers' default is 16, and it sets the shape of
#: `index_kpool_compress_ape`. Every other dimension matches the default config.
REAL_INDEX_KPOOL = 4


def _glm():
    """transformers' glm5_next pieces, or a skip naming the reason.

    Imported inside each test, not at module scope: a module-level skip would also hide the
    tests below that need no transformers at all.
    """
    pytest.importorskip("transformers", minversion="5.17.0",
                        reason="glm5_next landed in transformers 5.17")
    try:
        from transformers.models.glm5_next import modeling_glm5_next as mod
        from transformers.models.glm5_next.configuration_glm5_next import (
            Glm5NextConfig, Glm5NextTextConfig)
    except ImportError as exc:                                        # pragma: no cover
        pytest.skip(f"transformers has no glm5_next: {exc}")
    return mod, Glm5NextConfig, Glm5NextTextConfig


def _tiny_text_kwargs(**over):
    """A two-layer GLM: layer 0 is KDA + dense MLP, layer 1 is MLA/DSA + MoE.

    Small enough to run on a CPU in fp32, with every structural feature of the real model:
    hyper-connections at hc_mult 4, a k-pooled indexer that genuinely sparsifies (16 tokens,
    top 8 in pools of 4), shared + routed experts, and the clamped SwiGLU.
    """
    kw = dict(vocab_size=64, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
              num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
              n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
              kv_lora_rank=16, q_lora_rank=24, qk_nope_head_dim=8, v_head_dim=8,
              index_topk=8, index_head_dim=8, index_n_heads=2, index_kpool=4,
              linear_head_dim=8, linear_num_heads=4, hc_mult=4,
              layer_types=["linear_attention", "indexed_attention"],
              mlp_layer_types=["dense", "sparse"], indexer_types=["full", "full"],
              pad_token_id=0)
    kw.update(over)
    return kw


def _checkpoint_layer_state(layer_idx):
    """Layer `layer_idx`'s tensors at RECORDED shapes and dtypes, on meta (zero memory)."""
    ms = load_shapes(SLUG)
    prefix = f"{SD_PREFIX}.{layer_idx}."
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32}
    state = {}
    with torch.device("meta"):
        for key in ms.keys():
            if key.startswith(prefix):
                state[key[len(prefix):]] = torch.empty(
                    ms.shape_of(key), dtype=dtypes[ms.dtype_of(key)])
    assert state, "manifest has no tensors for this layer"
    return state


# --------------------------------------------------------------------------- the profile

def test_profile_is_registered():
    assert ARCH in qm._MODEL_PROFILES, (
        "no profile: the default has no stacked_experts and no convert_state, and the run "
        "would quantize the shared experts and dense MLPs only, about 0.5% of the model")


def test_profile_paths_and_switches():
    p = qm._MODEL_PROFILES[ARCH]
    assert p["layers_attr"] == "model.language_model.layers"
    assert p["embed_attr"] == "model.language_model.embed_tokens"
    assert p["rotary_attr"] is None, "GLM-5.3-Flash is NoPE: qk_rope_head_dim is 0"
    assert p["sd_prefix"] == SD_PREFIX
    assert p["embed_key"] == "model.language_model.embed_tokens.weight"
    assert p["trust_remote_code"] is False
    assert p["forward_kwargs"] == "default"
    assert p["multimodal_text"] is True
    assert p["stacked_experts"] is True
    assert qm._stacked_experts_enabled(ARCH, p)
    assert p["convert_state"] == "glm5_next" and "glm5_next" in qm._STATE_CONVERTERS
    assert p["bool_attention_mask"] is True


def test_profile_paths_resolve_on_the_real_module_tree():
    """Read off a meta-built model with the default (= real) dimensions, not assumed."""
    mod, Glm5NextConfig, _ = _glm()
    p = qm._MODEL_PROFILES[ARCH]
    with torch.device("meta"):
        model = qm._meta_model_from_config(Glm5NextConfig(), multimodal=True)
    assert type(model).__name__ == ARCH
    layers = qm._resolve_attr(model, p["layers_attr"])
    assert len(layers) == 45
    assert isinstance(layers[0], mod.Glm5NextTextDecoderLayer)
    assert isinstance(qm._resolve_attr(model, p["embed_attr"]), nn.Embedding)
    assert qm._discover_rotary_cls(model, p) is None
    qm._require_supported_expert_layout(layers, ARCH, p)          # must not raise


def test_streaming_finds_the_text_decoder_and_embedding_in_the_real_keys():
    weight_map = {k: "s" for k in load_shapes(SLUG).keys()}
    embed_key, sd_prefix = qm._streaming_wrapper_keys(weight_map)
    assert sd_prefix == SD_PREFIX, "the vision tower's blocks must not be picked"
    assert embed_key == "model.language_model.embed_tokens.weight"


def test_streaming_is_required():
    with pytest.raises(ValueError, match="--streaming"):
        qm._require_streaming_for_wrapper(ARCH, qm._MODEL_PROFILES[ARCH], streaming=False)


# ------------------------------------------------ which matrices get quantized, and which not

def _linear_names(layer_idx):
    mod, _, Glm5NextTextConfig = _glm()
    with torch.device("meta"):
        layer = mod.Glm5NextTextDecoderLayer(
            Glm5NextTextConfig(index_kpool=REAL_INDEX_KPOOL), layer_idx)
    return set(qm._collect_linears(layer, qm._MODEL_PROFILES[ARCH]))


@pytest.mark.parametrize("layer_idx,want", [
    (0, {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}),              # KDA + dense
    (3, {"mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj",
         "mlp.shared_experts.down_proj"}),                                # MLA/DSA + MoE
    (4, {"mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj",
         "mlp.shared_experts.down_proj"}),                                # KDA + MoE
])
def test_only_the_mlps_are_walked(layer_idx, want):
    """Attention stays bf16, for a reason that is not a preference.

    vLLM 0.32 builds every KDA, MLA and indexer projection with `quant_config=None`, so a
    quantized one would fail weight loading. The routed experts are not nn.Linears and come
    through the stacked path; the router is an nn.Parameter and is never walked.
    """
    assert _linear_names(layer_idx) == want


# ------------------------------------- the conversion: does every checkpoint key find a home?

@pytest.mark.parametrize("layer_idx", [0, 3, 4])
def test_conversion_maps_the_recorded_checkpoint_onto_the_real_layer_exactly(layer_idx):
    """THE gate: real shapes, real module, zero unexpected and zero missing keys.

    Three layer kinds, because each carries keys the others do not: KDA + dense (0),
    MLA/DSA + MoE (3), KDA + MoE (4).
    """
    mod, _, Glm5NextTextConfig = _glm()
    converted = qm._convert_layer_state(_checkpoint_layer_state(layer_idx),
                                        qm._MODEL_PROFILES[ARCH])
    with torch.device("meta"):
        sd = mod.Glm5NextTextDecoderLayer(
            Glm5NextTextConfig(index_kpool=REAL_INDEX_KPOOL), layer_idx).state_dict()

    assert set(converted) - set(sd) == set(), "checkpoint tensors with nowhere to go"
    assert set(sd) - set(converted) == set(), "module tensors the checkpoint never fills"
    for key, want in sd.items():
        assert tuple(converted[key].shape) == tuple(want.shape), key


def test_the_stacked_tensors_have_the_published_geometry():
    """288 experts, gate+up fused to 4096 rows, hidden 4096, intermediate 2048; the KDA
    short conv is q, k and v (8192 channels each) concatenated."""
    converted = qm._convert_layer_state(_checkpoint_layer_state(4), qm._MODEL_PROFILES[ARCH])
    assert tuple(converted["mlp.experts.gate_up_proj"].shape) == (288, 4096, 4096)
    assert tuple(converted["mlp.experts.down_proj"].shape) == (288, 4096, 2048)
    assert tuple(converted["self_attn.conv1d.weight"].shape) == (3 * 8192, 1, 4)


def test_conversion_preserves_device_and_dtype():
    """A CPU-forcing `torch.empty` would materialize ~7 GiB per layer during conversion, and
    the fp32 tensors must stay fp32."""
    converted = qm._convert_layer_state(_checkpoint_layer_state(4), qm._MODEL_PROFILES[ARCH])
    assert converted["mlp.experts.gate_up_proj"].device.type == "meta"
    assert converted["mlp.experts.gate_up_proj"].dtype == torch.bfloat16
    assert converted["self_attn.forget_gate.A_log"].dtype == torch.float32
    assert converted["attn_hc.base"].dtype == torch.float32


def test_without_the_conversion_the_drop_is_caught_not_silent():
    """The control: raw checkpoint keys must FAIL the coverage guard rather than proceed."""
    mod, _, Glm5NextTextConfig = _glm()
    raw = _checkpoint_layer_state(3)
    with torch.device("meta"):
        layer = mod.Glm5NextTextDecoderLayer(
            Glm5NextTextConfig(index_kpool=REAL_INDEX_KPOOL), 3)
    incompatible = layer.load_state_dict(raw, strict=False, assign=True)
    assert len(incompatible.unexpected_keys) > 800          # 864 per-expert + hc tensors
    with pytest.raises(ValueError, match="did not load"):
        qm._assert_load_coverage(3, incompatible, n_loaded=len(raw))


def _saved_tiny_model(tmp_path):
    """A tiny GLM saved by transformers itself, plus the module state it came from."""
    mod, Glm5NextConfig, _ = _glm()
    from safetensors import safe_open

    cfg = Glm5NextConfig(text_config=_tiny_text_kwargs(),
                         vision_config=dict(depth=1, hidden_size=16, intermediate_size=32,
                                            num_heads=2, out_hidden_size=32))
    torch.manual_seed(0)
    model = mod.Glm5NextForConditionalGeneration(cfg).float().eval()
    model.save_pretrained(str(tmp_path))
    saved = {}
    for fname in os.listdir(tmp_path):
        if fname.endswith(".safetensors"):
            with safe_open(str(tmp_path / fname), framework="pt") as f:
                for k in f.keys():
                    saved[k] = f.get_tensor(k)
    return model, saved


def test_conversion_agrees_with_transformers_own_round_trip(tmp_path):
    """The arbiter for ORDER, which a key-set check cannot see.

    `save_pretrained` reverses transformers' conversion and writes the checkpoint spelling
    (per-expert gate/up/down, `q/k/v_conv1d`, `hc_attn_fn`, `f_a_proj`). Converting that
    back with GLQ's converter must reproduce the live module's tensors bit for bit. A wrong
    gate/up row order or q/k/v concat order loads cleanly and fails only here.
    """
    model, saved = _saved_tiny_model(tmp_path)
    live = model.state_dict()
    for i in range(2):
        prefix = f"{SD_PREFIX}.{i}."
        layer_state = {k[len(prefix):]: v for k, v in saved.items() if k.startswith(prefix)}
        converted = qm._convert_layer_state(layer_state, qm._MODEL_PROFILES[ARCH])
        want = {k[len(prefix):]: v for k, v in live.items() if k.startswith(prefix)}
        assert set(converted) == set(want), (
            sorted(set(converted) ^ set(want)))
        for key in want:
            assert torch.equal(converted[key], want[key]), f"layer {i}: {key}"


def test_the_round_trip_spelling_is_the_real_checkpoints(tmp_path):
    """Ties the tiny round trip to the 598 GiB checkpoint: every layer key transformers
    writes has the same pattern as a key in the recorded manifest."""
    from model_shapes import collapse

    _, saved = _saved_tiny_model(tmp_path)
    recorded = set(load_shapes(SLUG).patterns())
    layer_patterns = {collapse(k) for k in saved if f"{SD_PREFIX}." in k}
    assert layer_patterns <= recorded, sorted(layer_patterns - recorded)


def test_conv1d_concatenation_refuses_a_partial_set():
    state = {"self_attn.q_conv1d.weight": torch.zeros(4, 1, 4),
             "self_attn.k_conv1d.weight": torch.zeros(4, 1, 4)}
    with pytest.raises(ValueError, match="conv1d"):
        qm._convert_state_glm5_next(state)


def test_non_contiguous_expert_indices_raise():
    state = {}
    for e in (0, 1, 3):
        state[f"mlp.experts.{e}.gate_proj.weight"] = torch.zeros(4, 8)
        state[f"mlp.experts.{e}.up_proj.weight"] = torch.zeros(4, 8)
        state[f"mlp.experts.{e}.down_proj.weight"] = torch.zeros(8, 4)
    with pytest.raises(ValueError, match="not contiguous"):
        qm._convert_state_glm5_next(state)


# --------------------------------------------------------- calibration forward fidelity

def _tiny_text_model():
    mod, _, Glm5NextTextConfig = _glm()
    torch.manual_seed(0)
    cfg = Glm5NextTextConfig(**_tiny_text_kwargs())
    return cfg, mod.Glm5NextTextModel(cfg).to(torch.float32).eval()


def _capture_layer(model, layer_idx, ids):
    captured = {}

    def _hook(_mod, args, kwargs, out):
        captured["in"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()
        captured["out"] = (out[0] if isinstance(out, tuple) else out).detach().clone()

    handle = model.layers[layer_idx].register_forward_hook(_hook, with_kwargs=True)
    with torch.no_grad():
        model(input_ids=ids)
    handle.remove()
    return captured


def test_hc_expansion_reproduces_the_models_first_layer_input():
    """The decoder consumes `hc_mult` parallel streams, `[B, S, 4, D]`, built by the text
    model as `unsqueeze(2).expand(...)` between the embedding and layer 0."""
    cfg, model = _tiny_text_model()
    ids = torch.randint(1, cfg.vocab_size, (1, 16))
    captured = _capture_layer(model, 0, ids)
    with torch.no_grad():
        got = qm._apply_hc_expansion(model.embed_tokens(ids), cfg)
    assert got.shape == (1, 16, 4, cfg.hidden_size)
    assert got.is_contiguous(), "HyperConnection views it as [B, S, 4*D]"
    assert torch.equal(got, captured["in"])


def test_hc_expansion_is_a_no_op_without_hc_mult():
    class _Cfg:
        pass
    h = torch.randn(1, 4, 8)
    assert qm._apply_hc_expansion(h, _Cfg()) is h


@pytest.mark.parametrize("layer_idx,kind", [(0, "KDA + dense"), (1, "MLA/DSA + MoE")])
def test_calibration_reproduces_the_models_own_layer_forward(layer_idx, kind):
    """GLQ's per-layer calibration call must match the model's own forward exactly.

    The MLA layer is the demanding one: its indexer selects 2 pools of 4 out of 16 tokens
    plus the tail, so a wrong mask is not a rounding difference.
    """
    cfg, model = _tiny_text_model()
    ids = torch.randint(1, cfg.vocab_size, (1, 16))
    captured = _capture_layer(model, layer_idx, ids)

    kwargs = qm._build_forward_kwargs(qm._MODEL_PROFILES[ARCH], captured["in"], None,
                                      layer_idx=layer_idx, cfg=cfg, calib_ids=ids)
    with torch.no_grad():
        got = model.layers[layer_idx](captured["in"], **kwargs)
    got = got[0] if isinstance(got, tuple) else got

    delta = (got - captured["out"]).abs().max().item()
    assert delta < 1e-6, f"{kind}: calibration diverges from the model by {delta:.3e}"


def test_without_the_mask_the_indexer_raises():
    """The control for `bool_attention_mask`: the default kwargs carry no mask, and the
    indexer dereferences it unconditionally."""
    cfg, model = _tiny_text_model()
    ids = torch.randint(1, cfg.vocab_size, (1, 16))
    captured = _capture_layer(model, 1, ids)
    kwargs = qm._build_forward_kwargs({"forward_kwargs": "default"}, captured["in"], None,
                                      layer_idx=1, cfg=cfg)
    with pytest.raises(AttributeError):
        with torch.no_grad():
            model.layers[1](captured["in"], **kwargs)


def test_the_mask_is_two_dimensional_and_all_visible():
    h = torch.zeros(1, 16, 4, 32)
    m = qm._build_forward_kwargs({"forward_kwargs": "default", "bool_attention_mask": True},
                                 h, None, layer_idx=0)["attention_mask"]
    assert m.dtype == torch.bool and m.shape == (1, 16) and bool(m.all())


# ------------------------------------------------------- the expert Hessian hook

def _experts_beyond_the_clamp():
    mod, _, Glm5NextTextConfig = _glm()
    cfg = Glm5NextTextConfig(**_tiny_text_kwargs())
    torch.manual_seed(0)
    experts = mod.Glm5NextTextExperts(cfg).float()
    with torch.no_grad():
        experts.gate_up_proj.normal_(std=3.0)       # pre-activations well past +-10
        experts.down_proj.normal_(std=0.1)
    return cfg, experts


def test_gated_hook_uses_the_modules_own_clamped_gate():
    """The down projection's Hessian is built from the activation the model actually feeds
    it. For GLM that is `_apply_gate`: gate clamped to <= 10, up to [-10, 10], then
    `silu(gate) * up`. There is no `act_fn` on this container."""
    cfg, experts = _experts_beyond_the_clamp()
    assert not hasattr(experts, "act_fn")

    x = torch.randn(6, cfg.hidden_size) * 4
    top_k = torch.tensor([[0, 1]] * 6)
    weights = torch.full((6, 2), 0.5)
    counts, hessians = {}, {}
    handle = experts.register_forward_hook(
        qm._make_gated_expert_hook("mlp.experts", experts, counts, hessians),
        with_kwargs=True)
    with torch.no_grad():
        experts(x, top_k, weights)
    handle.remove()

    go = nn.functional.linear(x, experts.gate_up_proj[0])
    want = experts._apply_gate(go).float()
    assert torch.equal(hessians["mlp.experts.0.down_proj"], want.t() @ want)
    assert torch.equal(hessians["mlp.experts.0.gate_up_proj"], x.t() @ x)
    assert counts[("mlp.experts", 0)] == 6 and counts[("mlp.experts", 2)] == 0

    g, u = go.chunk(2, dim=-1)
    unclamped = (nn.functional.silu(g) * u).float()
    assert not torch.equal(want, unclamped), "the inputs never reached the clamp"


def test_gated_hook_still_uses_act_fn_where_the_container_has_one():
    """Every gated container GLQ already supports (gemma-4, Qwen4Exp, Kolibri-1) carries
    `act_fn`, and their checkpoints must stay byte-identical."""
    class _Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(2, 8, 4))
            self.down_proj = nn.Parameter(torch.randn(2, 4, 4))
            self.act_fn = nn.GELU()

        def forward(self, hidden_states, top_k_index, top_k_weights):
            return hidden_states

    emod = _Experts()
    x = torch.randn(3, 4)
    counts, hessians = {}, {}
    hook = qm._make_gated_expert_hook("experts", emod, counts, hessians)
    hook(emod, (x, torch.tensor([[1]] * 3)), {}, None)
    g, u = nn.functional.linear(x, emod.gate_up_proj[1]).chunk(2, dim=-1)
    want = (emod.act_fn(g) * u).float()
    assert torch.equal(hessians["experts.1.down_proj"], want.t() @ want)


# ------------------------------------------------------- fp32-strict tensors in calibration

def test_fp32_strict_tensors_stay_fp32_through_the_device_move():
    """transformers keeps `e_score_correction_bias`, `conv1d`, `dt_bias` and `A_log` in fp32
    even in a bf16 model (`_keep_in_fp32_modules_strict`). The router bias in particular
    decides which experts each calibration token reaches, so rounding it to bf16 changes the
    per-expert Hessians."""
    mod, _, Glm5NextTextConfig = _glm()
    cfg = Glm5NextTextConfig(**_tiny_text_kwargs())
    p = qm._MODEL_PROFILES[ARCH]
    assert set(p["keep_fp32"]) == set(mod.Glm5NextPreTrainedModel._keep_in_fp32_modules_strict)

    torch.manual_seed(0)
    layers = [mod.Glm5NextTextDecoderLayer(cfg, i).float() for i in range(2)]
    with torch.no_grad():
        layers[1].mlp.gate.e_score_correction_bias.normal_()
    before = {i: {k: v.clone() for k, v in l.state_dict().items()} for i, l in enumerate(layers)}

    for i, layer in enumerate(layers):
        qm._move_layer_keeping_bulk_on_cpu(layer, "cpu", torch.bfloat16,
                                           keep_fp32=p["keep_fp32"])
        for k, v in layer.state_dict().items():
            if any(s in k for s in p["keep_fp32"]):
                assert v.dtype == torch.float32, k
                assert torch.equal(v, before[i][k]), f"{k} was rounded on the way"
            elif v.is_floating_point():
                assert v.dtype == torch.bfloat16, k


def test_without_keep_fp32_every_float_tensor_is_cast_as_before():
    """The default must not move: every published checkpoint was calibrated this way."""
    mod, _, Glm5NextTextConfig = _glm()
    layer = mod.Glm5NextTextDecoderLayer(Glm5NextTextConfig(**_tiny_text_kwargs()), 1).float()
    qm._move_layer_keeping_bulk_on_cpu(layer, "cpu", torch.bfloat16)
    assert all(v.dtype == torch.bfloat16 for v in layer.state_dict().values()
               if v.is_floating_point())


# ------------------------------------------------------- the MTP layer, `layers.45`

def test_the_mtp_layer_is_found_beyond_the_decoder():
    """GLM's MTP head is a 46th decoder-shaped layer, `layers.45`, not an `mtp.*` tree, so
    the segment rule in `_is_nontext` cannot see it."""
    weight_map = {k: "s" for k in load_shapes(SLUG).keys()}
    assert qm._layers_beyond_decoder(weight_map, SD_PREFIX, n_layers=45) == (
        f"{SD_PREFIX}.45.",)


def test_the_mtp_layer_counts_as_nontext_not_as_offloadable_experts():
    """vLLM does not load it without speculative decoding. Counting its 288 experts as
    offloadable would claim 14.5 GiB of host RAM that a serve never pins."""
    ms = load_shapes(SLUG)
    state = {}
    with torch.device("meta"):
        for key in ms.keys():
            if key.startswith((f"{SD_PREFIX}.44.", f"{SD_PREFIX}.45.")):
                state[key] = torch.empty(ms.shape_of(key), dtype=torch.bfloat16)
    mtp_bytes = sum(t.numel() * 2 for k, t in state.items() if k.startswith(f"{SD_PREFIX}.45."))
    expert44 = sum(t.numel() * 2 for k, t in state.items()
                   if k.startswith(f"{SD_PREFIX}.44.") and "experts" in k)

    _, expert, nontext = qm._offloadable_bytes(state, nontext_prefixes=(f"{SD_PREFIX}.45.",))
    assert nontext == mtp_bytes
    assert expert == expert44

    _, expert_old, nontext_old = qm._offloadable_bytes(state)
    assert nontext_old == 0 and expert_old > expert, "the control: the old accounting"
