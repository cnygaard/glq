"""LFM2-MoE (`LiquidAI/LFM2-8B-A1B`) support in the quantizer: profile, key conversion, names.

LFM2-MoE is a hybrid tower: per layer either a gated short convolution or full attention
(`layer_types`), then either a dense SwiGLU MLP (the first `num_dense_layers`) or a sparse MoE
block with sigmoid routing and an fp32 `expert_bias`. Three things make it more than a profile
entry, each of which fails silently without its own handling:

1. transformers stores the routed experts as two 3-D parameters (`experts.gate_up_proj`,
   `experts.down_proj`) the `nn.Linear` walk cannot see, while the **checkpoint** stores them
   one tensor per expert as `experts.{e}.w1|w3|w2`. The streaming loader's
   `load_state_dict(strict=False)` drops those unless they are stacked first.
2. The GLQ checkpoint must keep the checkpoint's own expert names. vLLM's LFM2-MoE loader
   looks experts up by `ckpt_names=("w1", "w2", "w3")` and skips any other spelling without a
   word, which serves uninitialised experts; the generic save rule also drops a bf16 original
   only when its key shares the artifact's prefix.
3. `expert_bias` is an fp32 buffer that transformers keeps fp32 in a bf16 model. It is added
   to the router's sigmoid scores before top-k, so rounding it to bf16 during calibration
   changes which experts a token reaches near a tie, and every per-expert Hessian with it.

Shapes come from the recorded manifest `tests/models/lfm2_8b_a1b.json` (safetensors headers,
no weights) and the module side from transformers itself, so neither is a guess.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch
import torch.nn as nn

from glq import quantize_model as qm

sys.path.insert(0, os.path.dirname(__file__))

from model_shapes import load_or_skip as load_shapes  # noqa: E402

SLUG = "lfm2_8b_a1b"
ARCH = "Lfm2MoeForCausalLM"
PROFILE_KEYS = {"convert_state": "lfm2_moe"}


def _lfm2():
    """(Lfm2MoeConfig, module) or a skip; imported per test so the pure tests still run."""
    try:
        from transformers import Lfm2MoeConfig
        from transformers.models.lfm2_moe import modeling_lfm2_moe as mod
    except ImportError as exc:                                        # pragma: no cover
        pytest.skip(f"transformers has no lfm2_moe: {exc}")
    return Lfm2MoeConfig, mod


def _manifest_layer_types(ms):
    """`layer_types` read off the manifest: a layer with `self_attn.*` tensors is attention."""
    n = 1 + max(int(k.split(".")[2]) for k in ms.keys() if k.startswith("model.layers."))
    attn = {int(k.split(".")[2]) for k in ms.keys() if ".self_attn." in k}
    return ["full_attention" if i in attn else "conv" for i in range(n)]


def _real_config():
    """LFM2-8B-A1B's decoder geometry. The defaults of `Lfm2MoeConfig` are this model's
    (hidden 2048, intermediate 7168, moe 1792, 32 experts, 2 dense layers); the layer count and
    types come from the manifest."""
    Lfm2MoeConfig, _ = _lfm2()
    types = _manifest_layer_types(load_shapes(SLUG))
    return Lfm2MoeConfig(num_hidden_layers=len(types), layer_types=types)


def _checkpoint_layer_state(layer_idx):
    """Layer `layer_idx`'s tensors at RECORDED shapes and dtypes, on meta."""
    ms = load_shapes(SLUG)
    prefix = f"model.layers.{layer_idx}."
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32}
    state = {}
    with torch.device("meta"):
        for key in ms.keys():
            if key.startswith(prefix):
                state[key[len(prefix):]] = torch.empty(
                    ms.shape_of(key), dtype=dtypes[ms.dtype_of(key)])
    assert state, "manifest has no layer tensors — recorded wrong?"
    return state


# --------------------------------------------------------------- the profile

def test_profile_is_registered():
    assert ARCH in qm._MODEL_PROFILES, (
        "no profile: the default one has no stacked_experts, so the run would refuse at meta "
        "discovery (or, without that guard, quantize the 8% of the model outside the experts)")


def test_profile_paths_match_the_real_module_tree():
    """Read off `Lfm2MoeForCausalLM`: the rotary is `model.pos_emb`, not `rotary_emb`, so the
    heuristic finds nothing and the attention layers would get no position embeddings."""
    _, mod = _lfm2()
    p = qm._MODEL_PROFILES[ARCH]
    assert p["layers_attr"] == "model.layers"
    assert p["embed_attr"] == "model.embed_tokens"
    assert p["rotary_attr"] == "model.pos_emb"
    assert p["sd_prefix"] == "model.layers"
    assert p["forward_kwargs"] == "default"
    assert p.get("trust_remote_code") is False
    with torch.device("meta"):
        model = mod.Lfm2MoeForCausalLM(_real_config())
    for attr in ("layers_attr", "embed_attr", "rotary_attr"):
        assert qm._resolve_attr(model, p[attr]) is not None, attr


def test_profile_passes_no_4d_causal_mask():
    """The short-conv layers multiply hidden states by the mask as a 2-D padding mask
    (`apply_mask_to_padding_states`); a 4-D causal mask there is the wrong tensor. sdpa with
    no mask is causal, which is what the attention layers need."""
    p = qm._MODEL_PROFILES[ARCH]
    assert not p.get("causal_mask")
    assert not p.get("bool_attention_mask")


def test_stacked_experts_opt_in_and_converter():
    p = qm._MODEL_PROFILES[ARCH]
    assert qm._stacked_experts_enabled(ARCH, p) is True
    assert p.get("convert_state") == "lfm2_moe"
    assert "lfm2_moe" in qm._STATE_CONVERTERS


def test_profile_keeps_the_checkpoint_expert_names():
    """gate, up, down, in that order: the save maps the jointly quantized gate+up halves and
    the down projection onto these leaves."""
    assert qm._MODEL_PROFILES[ARCH]["expert_leaf_names"] == ("w1", "w3", "w2")


def test_profile_keeps_the_router_bias_fp32():
    assert "expert_bias" in qm._MODEL_PROFILES[ARCH]["keep_fp32"]


def test_other_profiles_keep_the_default_expert_names():
    """The key is opt-in; every published architecture keeps gate_proj/up_proj/down_proj."""
    for arch, p in qm._MODEL_PROFILES.items():
        if arch != ARCH:
            assert "expert_leaf_names" not in p, arch
    assert qm._expert_leaf_names(qm._DEFAULT_PROFILE) == ("gate_proj", "up_proj", "down_proj")
    assert qm._expert_leaf_names(None) == ("gate_proj", "up_proj", "down_proj")


# ------------------------------------------- the gate: does every key find a home?

@pytest.mark.parametrize("layer_idx,kind", [
    (0, "conv + dense MLP"), (2, "attention + MoE"), (3, "conv + MoE")])
def test_conversion_maps_the_recorded_checkpoint_onto_the_real_layer_exactly(layer_idx, kind):
    """Real shapes on one side, the real module on the other: zero unexpected, zero missing,
    and every shape and dtype equal."""
    _, mod = _lfm2()
    converted = qm._convert_layer_state(_checkpoint_layer_state(layer_idx), PROFILE_KEYS)
    with torch.device("meta"):
        sd = mod.Lfm2MoeDecoderLayer(_real_config(), layer_idx).state_dict()

    assert set(converted) - set(sd) == set(), f"{kind}: checkpoint tensors with nowhere to go"
    assert set(sd) - set(converted) == set(), f"{kind}: module tensors the checkpoint misses"
    for key, want in sd.items():
        assert tuple(converted[key].shape) == tuple(want.shape), key


def test_the_stacked_experts_have_the_published_geometry():
    converted = qm._convert_layer_state(_checkpoint_layer_state(2), PROFILE_KEYS)
    assert tuple(converted["feed_forward.experts.gate_up_proj"].shape) == (32, 3584, 2048)
    assert tuple(converted["feed_forward.experts.down_proj"].shape) == (32, 2048, 1792)
    assert converted["feed_forward.experts.gate_up_proj"].device.type == "meta"


def test_the_dense_mlp_and_the_router_bias_pass_through_unchanged():
    """Layers 0-1 keep `feed_forward.w1/w3/w2` as plain nn.Linear weights, and the fp32 bias
    keeps its dtype (the converter must not cast it either way)."""
    dense = qm._convert_layer_state(_checkpoint_layer_state(0), PROFILE_KEYS)
    for leaf in ("w1", "w2", "w3"):
        assert f"feed_forward.{leaf}.weight" in dense
    moe = qm._convert_layer_state(_checkpoint_layer_state(2), PROFILE_KEYS)
    assert moe["feed_forward.expert_bias"].dtype == torch.float32


def test_without_the_conversion_the_drop_is_caught_not_silent():
    """The control: raw per-expert keys do not load, and the coverage guard says so."""
    _, mod = _lfm2()
    raw = _checkpoint_layer_state(2)
    with torch.device("meta"):
        layer = mod.Lfm2MoeDecoderLayer(_real_config(), 2)
    incompatible = layer.load_state_dict(raw, strict=False, assign=True)
    assert len(incompatible.unexpected_keys) == 96          # 32 experts x w1/w2/w3
    with pytest.raises(ValueError, match="did not load"):
        qm._assert_load_coverage(2, incompatible, n_loaded=len(raw))


# ----------------------------------------- stack order, against transformers itself

def _tiny_config(**kw):
    Lfm2MoeConfig, _ = _lfm2()
    args = dict(vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
                num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                num_dense_layers=1, num_experts=4, num_experts_per_tok=2,
                layer_types=["conv", "full_attention", "conv"])
    args.update(kw)
    return Lfm2MoeConfig(**args)


def test_w1_is_the_gate_half_per_transformers_own_save(tmp_path):
    """The one thing that could plausibly be backwards. transformers' `save_pretrained`
    writes the stacked experts back out per expert (`w1`, `w3`, `w2`) through its own
    conversion mapping; stacking what it wrote must rebuild the module's tensors exactly."""
    from safetensors.torch import load_file
    _, mod = _lfm2()
    torch.manual_seed(0)
    model = mod.Lfm2MoeForCausalLM(_tiny_config())
    model.save_pretrained(tmp_path)
    saved = load_file(str(tmp_path / "model.safetensors"))
    prefix = "model.layers.1."
    layer_state = {k[len(prefix):]: v for k, v in saved.items() if k.startswith(prefix)}
    assert "feed_forward.experts.0.w1.weight" in layer_state, sorted(layer_state)[:8]

    got = qm._convert_layer_state(layer_state, PROFILE_KEYS)
    experts = model.model.layers[1].feed_forward.experts
    assert torch.equal(got["feed_forward.experts.gate_up_proj"], experts.gate_up_proj.data)
    assert torch.equal(got["feed_forward.experts.down_proj"], experts.down_proj.data)


def test_non_contiguous_expert_indices_raise():
    state = {}
    for e in (0, 1, 3):
        state[f"feed_forward.experts.{e}.w1.weight"] = torch.zeros(4, 8)
        state[f"feed_forward.experts.{e}.w3.weight"] = torch.zeros(4, 8)
        state[f"feed_forward.experts.{e}.w2.weight"] = torch.zeros(8, 4)
    with pytest.raises(ValueError, match="not contiguous"):
        qm._convert_layer_state(state, PROFILE_KEYS)


def test_an_expert_missing_a_projection_raises():
    state = {"feed_forward.experts.0.w1.weight": torch.zeros(4, 8),
             "feed_forward.experts.0.w3.weight": torch.zeros(4, 8)}
    with pytest.raises(ValueError, match="w2|expected"):
        qm._convert_layer_state(state, PROFILE_KEYS)


# ------------------------------------------------- calibration matches the model

@pytest.mark.parametrize("layer_idx,kind", [
    (0, "conv + dense"), (1, "attention + MoE"), (2, "conv + MoE")])
def test_calibration_reproduces_the_models_own_forward(layer_idx, kind):
    """GLQ's per-layer forward (no mask, the profile's rotary) must equal the layer's output
    inside the model's own forward, for every block kind the tower has."""
    _, mod = _lfm2()
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = mod.Lfm2MoeForCausalLM(cfg).eval()
    with torch.no_grad():
        for blk in model.model.layers:
            ff = blk.feed_forward
            if hasattr(ff, "expert_bias"):
                ff.expert_bias.copy_(torch.randn_like(ff.expert_bias) * 0.1)
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    captured = {}

    def _capture(_mod, args, kwargs, out):
        captured["in"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()
        captured["out"] = (out[0] if isinstance(out, tuple) else out).detach().clone()

    handle = model.model.layers[layer_idx].register_forward_hook(_capture, with_kwargs=True)
    with torch.no_grad():
        model(ids)
    handle.remove()

    p = qm._MODEL_PROFILES[ARCH]
    kwargs = qm._build_forward_kwargs(p, captured["in"], qm.get_rotary_emb(model, p),
                                      layer_idx=layer_idx, cfg=cfg, calib_ids=ids)
    with torch.no_grad():
        got = model.model.layers[layer_idx](captured["in"], **kwargs)
    got = got[0] if isinstance(got, tuple) else got
    delta = (got - captured["out"]).abs().max().item()
    assert delta < 1e-5, f"{kind}: calibration differs from the model's forward by {delta:.3e}"


def test_the_device_move_keeps_the_router_bias_fp32():
    """The streaming path casts each layer to the calibration dtype; `keep_fp32` restores the
    named tensors at full precision, as transformers keeps them in a bf16 model."""
    _, mod = _lfm2()
    p = qm._MODEL_PROFILES[ARCH]
    model = mod.Lfm2MoeForCausalLM(_tiny_config())
    layer = model.model.layers[1]
    want = torch.tensor([0.1234567, -0.2, 0.3, 1e-4])
    with torch.no_grad():
        layer.feed_forward.expert_bias.copy_(want)
    keep = qm._keep_fp32_names(p, model)
    qm._move_layer_keeping_bulk_on_cpu(layer, "cpu", torch.bfloat16, keep_fp32=keep)
    bias = layer.feed_forward.expert_bias
    assert bias.dtype == torch.float32
    assert torch.equal(bias, want)
    assert layer.feed_forward.experts.gate_up_proj.dtype == torch.bfloat16


# --------------------------------------------------- resume replay under these names

def _stub_dequantize(arts, in_features, out_features, codebook_type, **kw):
    """Row r of W is SU[r], and the shape asked for must match it (see
    tests/test_resume_replay_experts.py): gate and up are decoded joined, as quantized."""
    su = arts["SU"].float()
    assert su.numel() == out_features, (su.numel(), out_features)
    return su[:, None].expand(out_features, in_features).clone()


def test_replay_restores_experts_stored_under_the_checkpoint_names():
    """A resumed run reads the artifacts back under the names they were saved with. Looking
    for gate_proj/up_proj/down_proj finds nothing here, and the replayed layer would carry
    its bf16 experts into calibration while reporting a plausible restore count."""
    _, mod = _lfm2()
    layer = mod.Lfm2MoeDecoderLayer(_tiny_config(), 1)
    experts = layer.feed_forward.experts
    with torch.no_grad():
        experts.gate_up_proj.zero_()
        experts.down_proj.zero_()
    inter = experts.down_proj.shape[2]
    arts = {}
    hidden = experts.down_proj.shape[1]
    for e in range(experts.gate_up_proj.shape[0]):
        arts[f"L.feed_forward.experts.{e}.w1"] = {"SU": torch.full((inter,), 2.0)}
        arts[f"L.feed_forward.experts.{e}.w3"] = {"SU": torch.full((inter,), 3.0)}
        arts[f"L.feed_forward.experts.{e}.w2"] = {"SU": torch.full((hidden,), 4.0)}
    n = qm._restore_replayed_weights(layer, arts, "L", "trellis", None,
                                     dequantize_fn=_stub_dequantize,
                                     expert_leaf_names=("w1", "w3", "w2"))
    assert n == 2 * experts.gate_up_proj.shape[0]
    assert torch.all(experts.gate_up_proj[:, :inter] == 2), "gate half"
    assert torch.all(experts.gate_up_proj[:, inter:] == 3), "up half"
    assert torch.all(experts.down_proj == 4)


# ----------------------------------- end to end: a tiny LFM2-MoE through quantize()

#: vocab matches the SmolLM2 tokenizer used for calibration, so ids stay in the embedding.
#: Layer 0: conv + dense MLP; 1: conv + MoE; 2: attention + MoE; 3: conv + MoE.
TINY = dict(vocab_size=49152, hidden_size=64, intermediate_size=128,
            moe_intermediate_size=32, num_hidden_layers=4, num_attention_heads=2,
            num_key_value_heads=1, num_dense_layers=1, num_experts=4,
            num_experts_per_tok=2, max_position_embeddings=256,
            layer_types=["conv", "conv", "full_attention", "conv"])


#: Calibration text for the tiny run. These checks are about names, coverage and loading, not
#: quality, so a fixed sentence stands in for WikiText-2: no `datasets`, no download.
_CALIB_TEXT = " ".join(
    ["The river runs past the old mill, and the town grows along its quiet banks."] * 40)


#: Stands in for LFM2's LICENSE (LFM Open License v1.0), which derivatives must carry.
_LICENSE_TEXT = b"LFM Open License v1.0 (stand-in for the test)\r\n"


@pytest.fixture(scope="module")
def quantized(tmp_path_factory):
    from transformers import AutoTokenizer
    Lfm2MoeConfig, mod = _lfm2()
    torch.manual_seed(0)
    cfg = Lfm2MoeConfig(**TINY)
    cfg.architectures = [ARCH]
    model = mod.Lfm2MoeForCausalLM(cfg).to(torch.bfloat16)
    with torch.no_grad():
        for blk in model.model.layers:
            if hasattr(blk.feed_forward, "expert_bias"):
                # `.to(bfloat16)` cast this buffer too; the real checkpoint stores it F32.
                blk.feed_forward.expert_bias = torch.linspace(-0.05, 0.05, 4,
                                                              dtype=torch.float32)
    src = str(tmp_path_factory.mktemp("lfm2_tiny_src"))
    model.save_pretrained(src, max_shard_size="2MB")
    assert _tensors(src)["model.layers.1.feed_forward.expert_bias"].dtype == torch.float32, \
        "the source must store the bias F32 like LFM2-8B-A1B, or the dtype check proves nothing"
    AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M").save_pretrained(src)
    with open(os.path.join(src, "LICENSE"), "wb") as f:
        f.write(_LICENSE_TEXT)

    out = str(tmp_path_factory.mktemp("lfm2_tiny_glq"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(qm._CALIB_CORPORA, "wikitext", lambda _n: [("fixed", _CALIB_TEXT)])
        mp.setenv("GLQ_TRELLIS_VARIANT", "3inst")
        avg = qm.quantize(model_name=src, output_dir=out, bpw=2, nsamples=1, seqlen=64,
                          device="cpu", workers=1, streaming=True, codebook_type="trellis")
    return src, out, avg


def _tensors(d):
    import glob
    from safetensors.torch import load_file
    found = {}
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        found.update(load_file(f))
    return found


@pytest.mark.slow
def test_experts_are_saved_under_the_checkpoint_names(quantized):
    _, out, avg = quantized
    assert avg > 0
    keys = set(_tensors(out))
    roots = {k[: -len(".trellis_packed")] for k in keys if k.endswith(".trellis_packed")}
    for layer in (1, 2, 3):
        for e in range(TINY["num_experts"]):
            for leaf in ("w1", "w3", "w2"):
                assert f"model.layers.{layer}.feed_forward.experts.{e}.{leaf}" in roots
    assert not any(r.endswith(("gate_proj", "up_proj", "down_proj")) for r in roots), (
        sorted(r for r in roots if r.endswith(("gate_proj", "up_proj", "down_proj")))[:4])


@pytest.mark.slow
def test_every_supported_linear_is_quantized_and_no_original_survives(quantized):
    _, out, _ = quantized
    keys = set(_tensors(out))
    roots = {k[: -len(".trellis_packed")] for k in keys if k.endswith(".trellis_packed")}
    want = {f"model.layers.0.feed_forward.{w}" for w in ("w1", "w2", "w3")}
    for layer in (0, 1, 3):
        want |= {f"model.layers.{layer}.conv.{p}" for p in ("in_proj", "out_proj")}
    want |= {f"model.layers.2.self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj", "out_proj")}
    assert want <= roots, sorted(want - roots)
    for r in roots:
        assert f"{r}.weight" not in keys, f"bf16 original written next to {r}"
    assert not any(k.endswith(("experts.gate_up_proj", "experts.down_proj")) for k in keys)


@pytest.mark.slow
def test_bf16_and_fp32_tensors_outside_the_linears_are_copied_verbatim(quantized):
    """Router weight, router bias (fp32), short-conv kernel, norms: byte for byte."""
    src, out, _ = quantized
    s, o = _tensors(src), _tensors(out)
    for key in ("model.layers.1.feed_forward.gate.weight",
                "model.layers.1.feed_forward.expert_bias",
                "model.layers.0.conv.conv.weight",
                "model.layers.2.self_attn.q_layernorm.weight",
                "model.embedding_norm.weight", "model.embed_tokens.weight"):
        assert key in o, key
        assert torch.equal(o[key], s[key]), key
    assert o["model.layers.1.feed_forward.expert_bias"].dtype == torch.float32


@pytest.mark.slow
def test_layer_bpw_lists_exactly_the_artifacts(quantized):
    _, out, _ = quantized
    cfg = json.load(open(os.path.join(out, "config.json")))
    keys = set(_tensors(out))
    roots = {k[: -len(".trellis_packed")] for k in keys if k.endswith(".trellis_packed")}
    assert cfg["architectures"] == [ARCH]
    assert set(cfg["quantization_config"]["layer_bpw"]) == roots


@pytest.mark.slow
def test_generation_config_and_licence_arrive_unchanged(quantized):
    """The stop tokens and the licence text the LFM Open License requires, byte for byte."""
    src, out, _ = quantized
    for name in ("generation_config.json", "LICENSE"):
        with open(os.path.join(src, name), "rb") as a, open(os.path.join(out, name), "rb") as b:
            assert b.read() == a.read(), name


# ------------------------------------------- the GLQ checkpoint through HF Transformers

def _artifacts_by_root(out):
    t = _tensors(out)
    roots = {k[: -len(".trellis_packed")] for k in t if k.endswith(".trellis_packed")}
    arts = {r: {} for r in roots}
    for k, v in t.items():
        root, _, leaf = k.rpartition(".")
        if root in arts:
            arts[root][leaf] = v
    return arts


def _dequantized_reference(src, out):
    """The source model with every quantized weight replaced by its dequantized value.

    Decoded straight from the checkpoint BY NAME, with LFM2's own meaning of the names
    (w1 = gate, rows first; w3 = up; w2 = down), so it is independent of the HF module
    wiring under test. gate and up were quantized as one [2I, H] matrix, so their halves
    are re-joined before decoding: decoding a half alone applies a different row Hadamard.
    """
    from transformers import AutoModelForCausalLM

    from glq.resume import dequantize_artifacts
    from glq.trellis import TrellisCodebook

    arts = _artifacts_by_root(out)
    K = next(iter(arts.values()))["trellis_packed"].shape[1] // 16
    cb = TrellisCodebook(variant="3inst", K=K, device="cpu")
    ref = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32)

    def deq(a, n_in, n_out):
        return dequantize_artifacts(a, n_in, n_out, "trellis", codebook=cb)

    for root, a in arts.items():
        pre, _, leaf = root.rpartition(".")
        if ".experts." not in root:
            lin = ref.get_submodule(root)
            lin.weight.data = deq(a, lin.in_features, lin.out_features)
            continue
        if leaf == "w3":
            continue                                    # joined with its w1 below
        container, _, e = pre.rpartition(".")
        experts, e = ref.get_submodule(container), int(e)
        if leaf == "w1":
            up = arts[f"{pre}.w3"]
            fused = {k: (torch.cat([v, up[k]]) if k in ("trellis_packed", "SU") else v)
                     for k, v in a.items()}
            _, two_i, h = experts.gate_up_proj.shape
            experts.gate_up_proj.data[e] = deq(fused, h, two_i)
        else:                                           # w2
            _, h, i = experts.down_proj.shape
            experts.down_proj.data[e] = deq(a, i, h)
    return ref


@pytest.mark.slow
def test_hf_loads_every_glq_tensor_and_matches_the_dequantized_reference(quantized):
    """The HF path end to end: no checkpoint tensor left unexpected, no expert pad left
    unloaded, and the logits those weights produce are the reference's."""
    import glq.hf_integration  # noqa: F401 — registers the quantizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src, out, _ = quantized
    model, info = AutoModelForCausalLM.from_pretrained(
        out, dtype=torch.float32, output_loading_info=True)
    assert not info["unexpected_keys"], sorted(info["unexpected_keys"])[:6]
    glq_bufs = ("trellis_packed", "SU", "SV", "Wscale")
    unloaded = [k for k in info["missing_keys"]
                if k.rpartition(".")[2] in glq_bufs and ".gate_up_proj." not in k]
    assert not unloaded, unloaded[:6]

    ids = AutoTokenizer.from_pretrained(src)(
        "The river runs past the old mill, and the town", return_tensors="pt").input_ids
    with torch.no_grad():
        got = model(ids).logits
        want = _dequantized_reference(src, out)(ids).logits
        unquantized = AutoModelForCausalLM.from_pretrained(
            src, dtype=torch.float32)(ids).logits
    err = (got - want).abs().max().item()
    gap = (unquantized - want).abs().max().item()
    print(f"\nHF GLQ vs dequantized reference: max |dlogit| {err:.3e}; "
          f"unquantized vs reference: {gap:.3e}")
    assert err < 0.02 * gap, (err, gap)
