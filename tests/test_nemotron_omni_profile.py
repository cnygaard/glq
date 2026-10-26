"""NVIDIA-Nemotron-3.5-Super-VL-120B-A12B: the layout, the profile, and the refusal.

Why this file exists. 90.6% of this model — **210.0 of 231.8 GiB, 40,960 tensors** — is routed
experts that neither of GLQ's two weight walks can see:

* the `nn.Linear` walk cannot, because transformers merges the checkpoint's per-expert tensors
  into two 3-D `nn.Parameter`s (`conversion_mapping.py`, `MergeModulelist`);
* the **stacked**-expert walk cannot either, because `NemotronHExperts` is **non-gated** — the
  forward is `up -> relu**2 -> down` and there is no `gate_up_proj`, which is the only shape
  `_collect_stacked_experts` and every site downstream of it recognise.

Both walks failing quietly is the worst case this repo knows: the run finishes, prints
per-sublayer metrics for the 7% it did quantize, and writes a "3 bpw" checkpoint weighing more
than the bf16 input. So the profile declares the architecture and
`_require_supported_expert_layout` refuses it, during meta discovery, **before** the 232 GiB
download. These tests pin the refusal, pin that it does not touch the four gated MoE models
already shipping, and pin the layout facts the eventual implementation will be written against.

Both sides are recorded fact rather than inference: the checkpoint side comes from
`tests/models/nemotron_3_5_super_vl_120b_a12b.json` (safetensors headers, no weight data), the
module side from a meta-instantiated model. That is the mistake `tests/models/README.md`
describes — a hand-written fixture encodes the author's *model* of an architecture rather than
the architecture — and the first draft of this very plan got three profile values wrong that
way (`layers_attr`, `forward_kwargs`, `trust_remote_code`).
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
ARCH = "NemotronH_Omni_Reasoning_V3"


# ---- the recorded checkpoint -------------------------------------------------------


def test_the_profile_is_registered_and_the_fallback_would_be_catastrophic():
    """`_detect_profile` returns `_DEFAULT_PROFILE` for an unknown architecture — there is no
    unsupported-arch error — and that default's `sd_prefix` is `model.layers`, which matches
    **zero** of this checkpoint's 43,079 keys. So the cost of a missing profile entry is not a
    crash, it is `_load_layer_state` finding nothing per layer, 88 times over. Asserted off the
    real key set so the number is fact rather than a warning in a comment.
    """
    assert ARCH in qm._MODEL_PROFILES, "profile not registered; see the module docstring"

    ms = load_shapes(SLUG)
    default_prefix = qm._DEFAULT_PROFILE["sd_prefix"]
    assert ms.keys_under(default_prefix) == (), (
        f"{default_prefix} now matches real keys, so the default profile would no longer be "
        f"silently empty — re-read this test's premise")


def test_sd_prefix_is_the_checkpoint_spelling_and_covers_the_whole_decoder():
    """By count, against the two wrong candidates.

    `sd_prefix` drives `_load_layer_state` over `weight_map` keys and names every saved
    artifact, so it must be the ON-DISK prefix — `language_model.backbone.layers` — even
    though the module path is `language_model.model.layers`. A typo here does not raise.
    """
    ms = load_shapes(SLUG)
    prof = qm._MODEL_PROFILES[ARCH]

    under = ms.keys_under(prof["sd_prefix"])
    assert len(under) == 41640, f"decoder key count moved: {len(under)}"

    layer_idxs = {int(k.split(".layers.")[1].split(".")[0]) for k in under}
    assert layer_idxs == set(range(88)), "not all 88 blocks are under sd_prefix"

    # The MTP head is a second `.layers.N.*` stack in the same namespace — 1,040 keys — and
    # the module path matches nothing at all.
    assert len(ms.keys_under("language_model.mtp.layers")) == 1040
    assert ms.keys_under("language_model.model.layers") == ()


def test_the_routed_experts_are_the_whole_model_by_bytes():
    """The number that makes the refusal worth having rather than an inconvenience."""
    ms = load_shapes(SLUG)
    decoder = ms.keys_under(qm._MODEL_PROFILES[ARCH]["sd_prefix"])
    routed = [k for k in decoder if ".mixer.experts." in k]

    assert len(routed) == 40960, f"{len(routed)} routed expert tensors, expected 40 x 512 x 2"
    routed_gib = sum(ms.nbytes_of(k) for k in routed) / 2**30
    total_gib = sum(ms.nbytes_of(k) for k in ms.keys()) / 2**30

    assert routed_gib == pytest.approx(210.0, abs=0.1)
    assert routed_gib / total_gib > 0.90, (
        f"routed experts are {routed_gib / total_gib:.1%} of the checkpoint")


def test_every_quantizable_matrix_clears_the_trellis_out_gate():
    """Why the profile carries no `skip_linears`, asserted rather than asserted-in-prose.

    Qwen3.5 and Qwen4Exp both need one because some projection has fewer than 32 output rows
    and so could never serve under the trellis kernel. This model has none — the narrowest is
    k/v_proj at 256 — but that is a property of NVIDIA's shapes, not a guarantee, so a reshape
    in a future release should fail here instead of at serve time.
    """
    ms = load_shapes(SLUG)
    checked = 0
    for k in ms.keys_under(qm._MODEL_PROFILES[ARCH]["sd_prefix"]):
        shape = ms.shape_of(k)
        if len(shape) != 2:
            continue                      # norms, A_log/D/dt_bias, the 3-D conv1d weight
        assert shape[0] % 32 == 0, f"{k} has {shape[0]} output rows, not 32-divisible"
        checked += 1
    # Without this the test passes by iterating nothing — which is exactly what a wrong
    # `sd_prefix` produces, and the failure mode the whole file is about.
    assert checked == 41272, f"checked {checked} matrices, expected 41,272"


# ---- the refusal ------------------------------------------------------------------


class _NonGatedExperts(nn.Module):
    """`NemotronHExperts` in miniature: 3-D up/down, no gate."""

    def __init__(self, n_exp=4, inter=8, hidden=6):
        super().__init__()
        self.up_proj = nn.Parameter(torch.zeros(n_exp, inter, hidden))
        self.down_proj = nn.Parameter(torch.zeros(n_exp, hidden, inter))


class _GatedExperts(nn.Module):
    """The supported layout — gemma-4 / Qwen4Exp / Kolibri-1 all look like this."""

    def __init__(self, n_exp=4, inter=8, hidden=6):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(n_exp, 2 * inter, hidden))
        self.down_proj = nn.Parameter(torch.zeros(n_exp, hidden, inter))


class _Block(nn.Module):
    def __init__(self, experts=None):
        super().__init__()
        self.norm = nn.LayerNorm(6)
        if experts is not None:
            self.mixer = nn.Module()
            self.mixer.experts = experts


#: A profile that opts into the stacked-expert walk, which is what makes a stacked container
#: safe. Both kinds of container are now supported; what is refused is one nothing will walk.
_OPTED_IN = {"stacked_experts": True}


def test_an_undeclared_stacked_container_is_refused_and_the_message_says_what_to_add():
    """Not merely "it raises": a test satisfied by any exception would pass on a typo in the
    function itself. The message must carry where the container is, what shape makes it
    unwalkable, and the one-line fix.

    Non-gated containers are *supported* now; the thing still worth refusing is a stacked
    container whose profile omits `stacked_experts`, because the ordinary walk cannot see a
    bare 3-D Parameter and the stacked walk is opt-in. Qwen4Exp lost 67% of its weights that
    way, and this model would lose 90.6%.
    """
    layers = [_Block(), _Block(_NonGatedExperts())]      # layer 0 mamba-like: no experts

    with pytest.raises(ValueError) as exc:
        qm._require_supported_expert_layout(layers, ARCH, profile=None)

    msg = str(exc.value)
    assert ARCH in msg
    assert "layers.1.mixer.experts" in msg, msg
    assert "up_proj" in msg and "[4, 8, 6]" in msg, msg
    assert "'stacked_experts': True" in msg, msg


def test_an_undeclared_GATED_container_is_refused_too():
    """The widening. A bare 3-D `gate_up_proj` is equally invisible to the nn.Linear walk, so
    the refusal is about the opt-in, not about the gate."""
    with pytest.raises(ValueError, match="'stacked_experts': True"):
        qm._require_supported_expert_layout([_Block(_GatedExperts())], "Whatever", profile=None)


def test_the_refusal_scans_past_the_first_layer():
    """A hybrid tower's layer 0 is a Mamba block with no experts in it at all, so a check that
    looked only at `layers[0]` would pass this model straight through."""
    with pytest.raises(ValueError, match="layers.39"):
        qm._require_supported_expert_layout(
            [_Block() for _ in range(39)] + [_Block(_NonGatedExperts())], ARCH, profile=None)


def test_both_kinds_of_container_pass_once_the_profile_opts_in():
    """The regression that matters most: this check sits in front of four shipping models,
    all of which declare the flag."""
    for experts in (_GatedExperts(), _NonGatedExperts()):
        qm._require_supported_expert_layout([_Block(), _Block(experts)], "Whatever",
                                            profile=_OPTED_IN)
    assert qm._collect_stacked_experts(_Block(_GatedExperts())), "gated detection broke"
    assert _nongated_names(_Block(_GatedExperts())) == []
    assert _nongated_names(_Block(_NonGatedExperts())) == ["mixer.experts"]


def test_the_real_profile_opts_in_so_the_real_model_is_accepted():
    """The two halves have to agree: the profile declares `stacked_experts` and the check
    reads the same flag. A profile that lost the flag would be refused — which is the point —
    and a check that stopped reading it would let an undeclared arch through."""
    assert qm._MODEL_PROFILES[ARCH].get("stacked_experts") is True
    assert qm._stacked_experts_enabled(ARCH, qm._MODEL_PROFILES[ARCH])
    qm._require_supported_expert_layout([_Block(_NonGatedExperts())], ARCH,
                                        profile=qm._MODEL_PROFILES[ARCH])


def test_a_3d_parameter_that_is_not_an_expert_container_is_ignored():
    """`mixer.conv1d.weight` is 3-D — [10240, 1, 4] in this model — and a 3-D-shape rule
    rather than an attribute-name rule would read it as a stack of 10,240 experts."""
    blk = _Block()
    blk.conv1d = nn.Conv1d(4, 4, 3)
    assert blk.conv1d.weight.dim() == 3
    qm._require_supported_expert_layout([blk], "Whatever", profile=None)


def test_a_container_missing_its_down_proj_is_not_called_an_expert_stack():
    mod = nn.Module()
    mod.up_proj = nn.Parameter(torch.zeros(4, 8, 6))
    blk = _Block()
    blk.mixer = mod
    qm._require_supported_expert_layout([blk], "Whatever", profile=None)


def _nongated_names(layer):
    return [mn for mn, _m, _p, _s in qm._collect_nongated_stacked_experts(layer)]


def test_both_load_paths_are_gated_not_just_streaming():
    """A gate on one of two paths is a gate the next person routes around by dropping a flag.

    `multimodal_text` already forces `--streaming` for *this* model, so its non-streaming
    branch is unreachable — but the text-only `NemotronHForCausalLM` profile has no such
    requirement and shares the same non-gated `NemotronHExperts`.

    Structural rather than behavioural, and deliberately so: driving `quantize()`'s
    non-streaming branch needs a real checkpoint on disk, and `inspect.getsource` catches the
    regression this guards — somebody deleting one of the two calls — which is the whole
    reason the second one exists. `getsource` resolves through the module's `__file__`, so it
    does not depend on the working directory.
    """
    import inspect

    src = inspect.getsource(qm.quantize)
    assert src.count("_require_supported_expert_layout(") == 2, (
        "expected the refusal on both the streaming and the non-streaming load path")


# ---- streaming key derivation -------------------------------------------------------


def _real_weight_map():
    """`{key: shard}` for the real checkpoint, from the recorded manifest.

    The derivation only reads keys, so the recorded key set is a faithful stand-in for the
    index — and it is the real key set, which is the whole point of the manifest.
    """
    return {k: "shard.safetensors" for k in load_shapes(SLUG).keys()}


def test_the_decoder_prefix_is_derived_as_the_backbone_not_the_mtp_head():
    wm = _real_weight_map()
    _embed, prefix = qm._streaming_wrapper_keys(wm)
    assert prefix == "language_model.backbone.layers"


def test_the_derivation_does_not_depend_on_which_stack_the_shards_list_first():
    """The defect this guards. Both `language_model.backbone.layers.0.*` (9 keys) and
    `language_model.mtp.layers.0.*` (8) satisfy a bare ".layers.0." + "language_model" match,
    so the old `next(...)` was decided by `weight_map` iteration order. It lands on the
    backbone in the published index; a re-shard or a resume reorders it, and quantizing a
    1-layer speculative head as an 88-layer decoder is a run that finishes.
    """
    real = _real_weight_map()
    mtp_first = {k: v for k, v in real.items() if k.startswith("language_model.mtp.")}
    mtp_first.update(real)
    assert next(iter(mtp_first)).startswith("language_model.mtp."), "fixture did not reorder"

    _embed, prefix = qm._streaming_wrapper_keys(mtp_first)
    assert prefix == "language_model.backbone.layers"


def test_the_embedding_key_is_found_despite_not_being_called_embed_tokens():
    """This model's embedding is `language_model.backbone.embeddings.weight`. A lookup for
    `embed_tokens.weight` alone finds nothing and the old fallback was a hardcoded guess that
    is not in the weight map, so streaming died in `_load_tensor_from_shards`.
    """
    wm = _real_weight_map()
    embed_key, _prefix = qm._streaming_wrapper_keys(wm)
    assert embed_key == "language_model.backbone.embeddings.weight"
    assert embed_key in wm, "derived a key the checkpoint does not contain"


def test_embed_tokens_still_wins_for_the_architectures_that_use_it():
    """Back-compatibility for gemma-4 / Qwen / Mistral3, whose text embedding is spelled
    `embed_tokens` and whose vision towers must stay excluded."""
    wm = {
        "model.vision_tower.embeddings.weight": "s",
        "model.language_model.embed_tokens.weight": "s",
        "model.language_model.layers.0.self_attn.q_proj.weight": "s",
    }
    embed_key, prefix = qm._streaming_wrapper_keys(wm)
    assert embed_key == "model.language_model.embed_tokens.weight"
    assert prefix == "model.language_model.layers"


def test_two_rival_decoder_stacks_raise_rather_than_picking_one():
    wm = {"a.language_model.layers.0.w": "s", "b.language_model.layers.0.w": "s"}
    with pytest.raises(ValueError, match="ambiguous"):
        qm._streaming_wrapper_keys(wm)


def test_mtp_is_matched_as_a_segment_not_a_prefix_or_a_substring():
    """`mtp` is spelled `language_model.mtp.*` here, so a `startswith("mtp")` test misses it;
    and `mtpool` contains it, so a substring test would drop a real text weight."""
    assert qm._has_segment("language_model.mtp.layers.0.norm.weight", "mtp")
    assert not qm._has_segment("model.layers.0.mtpool.weight", "mtp")
    assert not qm._has_segment("model.layers.0.mtp_head.weight", "mtp")


# ---- the real module tree ------------------------------------------------------------


def _text_config_from_manifest():
    """The text config THIS checkpoint describes, derived from the recorded shapes.

    Instantiating the architecture at the library's default sizes proves nothing about this
    model: a default NemotronH has 8 experts of intermediate 7688, so a test asserting 512
    would fail against a model that is structurally identical to the one shipping. The
    manifest is the authority, so the config comes out of it.

    Derived through the EXPANDED key set, not `shape_of`, because patterns collapse the layer
    index — `shape_of("…layers.0.mixer.A_log")` answers for the pattern and so would claim
    every layer is a Mamba block. Membership of `keys()` is per-layer exact, which is what the
    `values` index form in `record.py` is for.
    """
    ms = load_shapes(SLUG)
    keys = set(ms.keys())
    pre = "language_model.backbone.layers"
    idxs = sorted({int(k.split(".layers.")[1].split(".")[0])
                   for k in ms.keys_under(pre)})

    block_type = []
    for i in idxs:
        if f"{pre}.{i}.mixer.A_log" in keys:
            block_type.append("linear_attention")        # Mamba-2
        elif f"{pre}.{i}.mixer.gate.weight" in keys:
            block_type.append("moe")
        elif f"{pre}.{i}.mixer.q_proj.weight" in keys:
            block_type.append("full_attention")
        else:                                            # pragma: no cover
            raise AssertionError(f"layer {i} matches no known block type")

    moe_layer = idxs[block_type.index("moe")]
    emb = ms.shape_of(f"language_model.backbone.embeddings.weight")
    up = ms.shape_of(f"{pre}.{moe_layer}.mixer.experts.0.up_proj.weight")
    shared = ms.shape_of(f"{pre}.{moe_layer}.mixer.shared_experts.up_proj.weight")
    n_exp = 1 + max(int(k.split(".experts.")[1].split(".")[0])
                    for k in keys if f"{pre}.{moe_layer}.mixer.experts." in k)

    return dict(
        vocab_size=emb[0], hidden_size=emb[1],
        num_hidden_layers=len(idxs), layers_block_type=block_type,
        n_routed_experts=n_exp,
        moe_intermediate_size=up[0], moe_latent_size=up[1],
        moe_shared_expert_intermediate_size=shared[0],
    ), block_type


def _meta_model():
    """The real wrapper on `meta` at the recorded sizes. No weights, no download.

    Inside the test rather than at module scope so the manifest-only tests above still run
    where transformers is too old — the native port landed in 5.18.
    """
    pytest.importorskip("transformers", minversion="5.18.0",
                        reason="nemotron_h_omni is native from transformers 5.18")
    try:
        from transformers.models.nemotron_h_omni.configuration_nemotron_h_omni import (
            NemotronH_Omni_Reasoning_V3_Config)
    except ImportError as exc:                                        # pragma: no cover
        pytest.skip(f"transformers has no nemotron_h_omni: {exc}")

    text_cfg, _ = _text_config_from_manifest()
    cfg = NemotronH_Omni_Reasoning_V3_Config(text_config=text_cfg)
    with torch.device("meta"):
        return qm._meta_model_from_config(cfg, trust_remote_code=False,
                                          dtype=torch.bfloat16, multimodal=True)


def test_the_profile_paths_resolve_on_the_real_module_tree():
    """The one check that catches a path typo, and it needs no weights.

    `layers_attr` and `embed_attr` are MODULE paths while `sd_prefix` is the checkpoint
    prefix; this is the half that `test_sd_prefix_...` cannot see.
    """
    model = _meta_model()
    prof = qm._MODEL_PROFILES[ARCH]

    _text_cfg, block_type = _text_config_from_manifest()
    layers = qm._resolve_attr(model, prof["layers_attr"])
    assert isinstance(layers, nn.ModuleList)
    assert len(layers) == len(block_type) == 88

    # The 40 / 40 / 8 mix, read off the recorded tensors and reproduced by the real classes.
    from collections import Counter
    assert Counter(block_type) == {"linear_attention": 40, "moe": 40, "full_attention": 8}
    assert Counter(type(l.mixer).__name__ for l in layers) == {
        "NemotronHMamba2Mixer": 40, "NemotronHMoE": 40, "NemotronHAttention": 8}

    embed = qm._resolve_attr(model, prof["embed_attr"])
    assert isinstance(embed, nn.Embedding)

    assert prof["rotary_attr"] is None, "NemotronHAttention applies no rotary"


def test_the_real_experts_are_non_gated_and_the_real_profile_covers_them():
    """End to end on the architecture itself, not a stand-in. Without this, every assertion
    above could be true of a model whose experts the quantizer still walks past."""
    model = _meta_model()
    prof = qm._MODEL_PROFILES[ARCH]
    layers = qm._resolve_attr(model, prof["layers_attr"])

    found = [f for layer in layers for f in qm._collect_nongated_stacked_experts(layer)]
    assert found, "the real model's experts are no longer non-gated 3-D parameters"
    assert all(qm._collect_stacked_experts(layer) == [] for layer in layers), (
        "a gated container appeared — the gated path's assumptions may now apply")
    assert len(found) == 40, f"{len(found)} MoE containers, expected one per MoE layer"

    mn, _mod, pname, shape = found[0]
    assert mn.endswith("mixer.experts") and pname == "up_proj"
    assert len(shape) == 3 and shape[0] == 512, shape

    # ...and the profile declares the walk that covers them, so the run is not refused.
    qm._require_supported_expert_layout(layers, ARCH, profile=prof)
    assert prof.get("convert_state") == "nemotron_h", (
        "the checkpoint stores experts per-expert; without the converter the streaming load "
        "drops every one of them")

    # Drop the opt-in and the same real layers are refused — so the acceptance above is the
    # profile's doing, not the check having gone quiet.
    with pytest.raises(ValueError, match="'stacked_experts': True"):
        qm._require_supported_expert_layout(layers, ARCH, profile=None)


def test_the_decoder_layer_takes_the_default_forward_kwargs_not_the_nemotron_h_branch():
    """Why `forward_kwargs` is `'default'`. The `'nemotron_h'` branch passes
    `cache_params=None`, but `NemotronHBlock.forward` has no such parameter — it passes
    `cache_params=past_key_values` to the mixer itself and forwards `**kwargs`, so the two
    collide. Asserted against the real signature rather than by catching a TypeError, which
    would also pass if the call failed for any other reason.
    """
    import inspect

    model = _meta_model()
    layers = qm._resolve_attr(model, qm._MODEL_PROFILES[ARCH]["layers_attr"])
    params = inspect.signature(type(layers[0]).forward).parameters

    assert "cache_params" not in params, (
        "NemotronHBlock now accepts cache_params, so the 'nemotron_h' branch may be usable")
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()), (
        "no **kwargs, so an unexpected cache_params would raise rather than collide")
    assert qm._MODEL_PROFILES[ARCH]["forward_kwargs"] == "default"
    # ...and the branch really does supply the colliding keyword, so the reasoning above is
    # about this code and not about a branch that has since changed.
    h = torch.zeros(1, 4, 2)
    assert "cache_params" in qm._build_forward_kwargs(
        {"forward_kwargs": "nemotron_h"}, h, None)
    assert "cache_params" not in qm._build_forward_kwargs(
        qm._MODEL_PROFILES[ARCH], h, None)


def test_the_profile_does_not_ask_for_an_explicit_causal_mask():
    """`causal_mask` builds a 4-D mask, which this architecture's Mamba mixer rejects on the
    sequence dimension. Left unset deliberately — pinned so it is not added as
    belt-and-braces later."""
    assert not qm._MODEL_PROFILES[ARCH].get("causal_mask")


def test_streaming_is_mandatory_for_this_wrapper():
    """The wrapper builds no MTP module (`_keys_to_ignore_on_load_unexpected`), so a
    non-streaming save would drop the 5.40 GiB head without a word — the Qwen3.5 failure."""
    prof = qm._MODEL_PROFILES[ARCH]
    assert qm._is_multimodal_text(ARCH, prof)
    with pytest.raises(ValueError, match="--streaming"):
        qm._require_streaming_for_wrapper(ARCH, prof, streaming=False)
    qm._require_streaming_for_wrapper(ARCH, prof, streaming=True)
