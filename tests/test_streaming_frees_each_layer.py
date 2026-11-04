"""Nothing from a quantized layer is still alive during the next layer's calibration forward.

GLM-5.3-Flash's shake-out ran out of GPU memory in layer 4's calibration forward, right after
its first MoE layer, 3. Layer 4 itself fit. About 40 GiB of layer 3 was still allocated,
because `quantize()` keeps each layer's state in locals of one long function, and some are
only rebound after the next layer's forward has run:

* `futures`: the expert pool's completed futures, each holding its expert's fp32 `W_hat`
  (288 experts x 96 MiB = 27 GiB);
* `gup` / `dwn` and `moe_writeback`: layer 3's stacked expert parameters (13.5 GiB), alive
  after `del layer`.

This runs the real streaming `quantize()` on a tiny three-layer GLM on CPU: dense KDA, then
MoE with MLA and MoE with KDA. The references are the same on any device, and `workers=1`
takes the ThreadPool expert path the GPU run took. Each MoE layer's calibration forward checks,
through weakrefs, that every stacked expert parameter and every `W_hat` from earlier layers
has been freed.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import weakref

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_glm5_next_profile import _glm, _tiny_text_kwargs  # noqa: E402

VOCAB = 64
SD_PREFIX = "model.language_model.layers"


class Probe:
    """Weakrefs to each layer's big objects, checked from inside later layers' forwards.

    `gen` counts MoE layers as their expert hooks are created, so everything a layer
    produces is tagged with the generation it belongs to; generation 0 is the dense layer.
    """

    def __init__(self):
        self.gen = 0
        self.refs = []      # (gen, label, weakref)
        self.checks = 0
        self.stale = []     # (checking gen, owning gen, label)

    def track(self, label, obj):
        self.refs.append((self.gen, label, weakref.ref(obj)))

    def check(self, gen):
        gc.collect()
        self.checks += 1
        for owner, label, ref in self.refs:
            if owner < gen and ref() is not None:
                self.stale.append((gen, owner, label))


def _save_tokenizer(path):
    """A word-level tokenizer over the tiny vocab, so nothing is fetched."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<unk>": 1, **{f"t{i}": i for i in range(2, VOCAB)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>",
                            pad_token="<pad>").save_pretrained(path)


def build_tiny_glm(src):
    """Save a three-layer GLM-5.3 (dense KDA, MoE MLA, MoE KDA) and a tokenizer to `src`."""
    mod, Glm5NextConfig, _ = _glm()
    # The trellis kernel layout tiles in 32s, so the MLP widths are multiples of 32.
    text = _tiny_text_kwargs(
        intermediate_size=64, moe_intermediate_size=32, num_hidden_layers=3,
        layer_types=["linear_attention", "indexed_attention", "linear_attention"],
        mlp_layer_types=["dense", "sparse", "sparse"],
        indexer_types=["full"] * 3)
    cfg = Glm5NextConfig(text_config=text,
                         vision_config=dict(depth=1, hidden_size=16, intermediate_size=32,
                                            num_heads=2, out_hidden_size=32))
    cfg.architectures = ["Glm5NextForConditionalGeneration"]
    torch.manual_seed(0)
    mod.Glm5NextForConditionalGeneration(cfg).float().eval().save_pretrained(
        src, max_shard_size="200KB")
    _save_tokenizer(src)


def quantize_tiny_glm(src, out, probe=None, device="cpu", expert_devices=None):
    """The real streaming `quantize()`, with fixed random calibration ids and, if given, the
    probe wired into the expert hooks and the per-matrix quantizer. The box runs it on
    `cuda`, with and without `expert_devices`, to compare the checkpoints."""
    from glq import quantize_model as qm

    gen = torch.Generator().manual_seed(0)

    def _calib(corpus, tokenizer, nsamples, seqlen):
        return torch.randint(2, VOCAB, (nsamples, seqlen), generator=gen), {"random": nsamples}

    saved = {name: getattr(qm, name) for name in
             ("_load_calib_ids", "_make_gated_expert_hook", "quantize_layer_e8_shell_rht")}
    qm._load_calib_ids = _calib
    if probe is not None:
        def _hook_factory(mn, emod, token_counts, hessians):
            probe.gen += 1
            mine = probe.gen
            probe.track(f"gen {mine} gate_up_proj", emod.gate_up_proj)
            probe.track(f"gen {mine} down_proj", emod.down_proj)
            inner = saved["_make_gated_expert_hook"](mn, emod, token_counts, hessians)

            def _hook(module, args, kwargs, output):
                probe.check(mine)
                return inner(module, args, kwargs, output)
            return _hook

        def _quantize(*args, **kwargs):
            W_hat, artifacts, metrics = saved["quantize_layer_e8_shell_rht"](*args, **kwargs)
            probe.track(f"gen {probe.gen} W_hat {tuple(W_hat.shape)}", W_hat)
            return W_hat, artifacts, metrics

        qm._make_gated_expert_hook = _hook_factory
        qm.quantize_layer_e8_shell_rht = _quantize
    prev_variant = os.environ.get("GLQ_TRELLIS_VARIANT")
    os.environ["GLQ_TRELLIS_VARIANT"] = "3inst"
    try:
        return qm.quantize(model_name=src, output_dir=out, bpw=4, nsamples=2, seqlen=16,
                           device=device, workers=1, streaming=True, codebook_type="trellis",
                           expert_devices=expert_devices)
    finally:
        for name, fn in saved.items():
            setattr(qm, name, fn)
        if prev_variant is None:
            os.environ.pop("GLQ_TRELLIS_VARIANT", None)
        else:
            os.environ["GLQ_TRELLIS_VARIANT"] = prev_variant


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    _glm()
    src = str(tmp_path_factory.mktemp("glm_tiny_src"))
    out = str(tmp_path_factory.mktemp("glm_tiny_glq"))
    build_tiny_glm(src)
    probe = Probe()
    with pytest.warns(UserWarning, match="nsamples=2 is low"):    # deliberate: 2 x 16 tokens
        quantize_tiny_glm(src, out, probe)
    return out, probe


def test_the_probe_ran_inside_both_moe_layers(run):
    """Mechanism first: both MoE layers' forwards ran the check, and every generation
    produced a `W_hat` to track. Otherwise an empty `stale` list would prove nothing."""
    _, probe = run
    assert probe.gen == 2
    assert probe.checks > 0
    gens = {owner for owner, label, _ in probe.refs if "W_hat" in label}
    assert gens == {0, 1, 2}, gens


def test_no_earlier_layer_is_alive_in_a_later_layers_forward(run):
    _, probe = run
    assert probe.stale == [], sorted(set(probe.stale))


def test_only_the_mlps_are_quantized(run):
    """The GLM profile end to end: dense MLP and both MoE layers' shared and routed experts
    in `layer_bpw`, no attention."""
    out, _ = run
    layer_bpw = json.load(open(os.path.join(out, "config.json")))["quantization_config"]["layer_bpw"]
    assert not [k for k in layer_bpw if ".self_attn." in k]
    assert f"{SD_PREFIX}.0.mlp.down_proj" in layer_bpw
    for layer in (1, 2):
        assert f"{SD_PREFIX}.{layer}.mlp.shared_experts.down_proj" in layer_bpw
        assert f"{SD_PREFIX}.{layer}.mlp.experts.3.gate_proj" in layer_bpw
