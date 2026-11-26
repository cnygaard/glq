"""LFM2-MoE through HF Transformers: the checkpoint's expert names must land somewhere.

The GLQ checkpoint keeps LFM2's own expert spelling, `experts.{e}.w1|w3|w2` (gate, up,
down), because vLLM's LFM2-MoE loader reads experts by exactly those names and skips any
other spelling without an error. The HF path replaces transformers' stacked
`Lfm2MoeExperts` with `GLQStackedGatedExperts`, whose per-expert landing pads were always
called `gate_proj`/`up_proj`/`down_proj`. transformers 5.x assigns checkpoint tensors by
key, so with those pads every `w1`/`w3`/`w2` tensor is an unexpected key: the experts stay
at their initial values and the model generates text from them with nothing failing.

A key rename is not the fix. transformers 5.22 does not read a class-level
`_checkpoint_conversion_mapping`, and `HfQuantizer.update_weight_conversions` changed its
signature between 5.19 and 5.22. So the pads take the checkpoint's leaf names, detected
from the checkpoint header, and the post-load fuse turns them into the fused
`gate_up_proj` / `down_proj` the forward reads. Default checkpoints keep the old names.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("transformers")
import torch.nn as nn  # noqa: E402

from glq.fused_experts import (  # noqa: E402
    GLQStackedGatedExperts,
    _fused_packed_shapes,
    _replace_stacked_gated_experts,
)
from model_shapes import load_or_skip as load_shapes  # noqa: E402

SLUG = "lfm2_8b_a1b"
LFM2_LEAVES = ("w1", "w3", "w2")
GLQ_BUFFERS = ("trellis_packed", "SU", "SV", "Wscale")


def _lfm2():
    try:
        from transformers import Lfm2MoeConfig
        from transformers.models.lfm2_moe import modeling_lfm2_moe as mod
    except ImportError as exc:                                        # pragma: no cover
        pytest.skip(f"transformers has no lfm2_moe: {exc}")
    return Lfm2MoeConfig, mod


def _real_model_on_meta():
    """LFM2-8B-A1B's module tree at real geometry, on meta (no memory)."""
    Lfm2MoeConfig, mod = _lfm2()
    ms = load_shapes(SLUG)
    n = 1 + max(int(k.split(".")[2]) for k in ms.keys() if k.startswith("model.layers."))
    attn = {int(k.split(".")[2]) for k in ms.keys() if ".self_attn." in k}
    types = ["full_attention" if i in attn else "conv" for i in range(n)]
    cfg = Lfm2MoeConfig(num_hidden_layers=n, layer_types=types)
    with torch.device("meta"):
        return mod.Lfm2MoeForCausalLM(cfg)


def _glq_roots():
    """Every module the 4 bpw GLQ checkpoint carries a payload for, from the manifest: each
    2-D `.weight` under `model.layers` except the router (a bare Parameter)."""
    ms = load_shapes(SLUG)
    return sorted(k[: -len(".weight")] for k in ms.keys()
                  if k.startswith("model.layers.") and k.endswith(".weight")
                  and len(ms.shape_of(k)) == 2
                  and not k.endswith("feed_forward.gate.weight"))


def _header_shapes(roots, K=4):
    """What `_collect_quantized_shapes` reads off a trellis checkpoint's header."""
    ms = load_shapes(SLUG)
    out = {}
    for r in roots:
        m, n = ms.shape_of(r + ".weight")
        out[r] = {"trellis_packed": [(m // 16) * (n // 16), 16 * K],
                  "SU": [m], "SV": [n], "Wscale": []}
    return out


# ------------------------------------------------------------ the real module tree

def test_every_checkpoint_tensor_has_a_home_after_preprocessing(monkeypatch):
    """The entry point transformers calls, on LFM2-8B-A1B's real module tree: every GLQ
    tensor the checkpoint holds must name a buffer of the preprocessed model."""
    import glq.hf_integration as hfi

    roots = _glq_roots()
    shapes = _header_shapes(roots)
    monkeypatch.setattr(hfi, "_collect_quantized_layer_names", lambda p: set(roots))
    monkeypatch.setattr(hfi, "_collect_quantized_shapes", lambda p: shapes)
    monkeypatch.setattr(hfi, "_detect_block_diagonal", lambda p: True)
    monkeypatch.setattr(hfi, "_collect_trellis_embedding_names", lambda p: set())

    model = _real_model_on_meta()
    model.config._name_or_path = "lfm2-glq-under-test"
    q = hfi.GLQQuantizer(hfi.GLQConfig(codebook="trellis", variant="3inst", bpw=4,
                                       trellis_layout="kernel"))
    q._process_model_before_weight_loading(model)

    have = set(model.state_dict().keys())
    want = {f"{r}.{b}" for r in roots for b in GLQ_BUFFERS}
    homeless = sorted(want - have)
    assert not homeless, f"{len(homeless)} checkpoint tensors have no buffer: {homeless[:4]}"


def test_the_lfm2_pads_are_sized_from_the_header():
    """accelerate places modules by their registered size, so the w2 pad (the real down
    projection) and the fused gate_up must carry the header's packed shapes."""
    model = _real_model_on_meta()
    shapes = _header_shapes(_glq_roots())
    _replace_stacked_gated_experts(model, codebook_type="trellis", shapes=shapes)
    pair = model.model.layers[2].feed_forward.experts[0]
    g = shapes["model.layers.2.feed_forward.experts.0.w1"]["trellis_packed"]
    d = shapes["model.layers.2.feed_forward.experts.0.w2"]["trellis_packed"]
    assert tuple(pair.gate_up_proj.trellis_packed.shape) == (2 * g[0], g[1])
    assert tuple(pair.w2.trellis_packed.shape) == tuple(d)


def test_the_default_spelling_keeps_its_pads():
    """gemma-4 / Flash-Next / GLM checkpoints spell gate_proj/up_proj/down_proj; detecting
    LFM2's names must not move them."""
    E, I, H = 2, 32, 64
    m = nn.Module()
    m.experts = nn.Module()
    m.experts.gate_up_proj = nn.Parameter(torch.zeros(E, 2 * I, H))
    m.experts.down_proj = nn.Parameter(torch.zeros(E, H, I))
    shapes = {f"experts.{e}.{leaf}": {"trellis_packed": [8, 64]}
              for e in range(E) for leaf in ("gate_proj", "up_proj", "down_proj")}
    _replace_stacked_gated_experts(m, codebook_type="trellis", shapes=shapes)
    names = {n for n, _ in m.named_modules()}
    for leaf in ("gate_proj", "up_proj", "down_proj", "gate_up_proj"):
        assert f"experts.0.{leaf}" in names
    assert "experts.0.w1" not in names


def test_packed_shapes_are_read_under_the_checkpoint_leaf_names():
    shapes = {"x.0.w1": {"trellis_packed": [8, 64]}, "x.0.w2": {"trellis_packed": [6, 64]}}
    assert _fused_packed_shapes("x", 1, shapes, leaves=LFM2_LEAVES) == {
        0: {"gate_up_proj": (16, 64), "down_proj": (6, 64)}}
    # and the default reading is unchanged
    assert _fused_packed_shapes("x", 1, shapes) == {}


# ------------------------------------------------------------ load, then fuse

H, I = 64, 32


def _lfm2_state(n_exp=1):
    """One LFM2 expert's GLQ tensors, distinct per leaf so a mix-up is visible."""
    sd = {}
    for e in range(n_exp):
        gate = torch.arange(4 * 8, dtype=torch.int16).reshape(4, 8) + 10 * e
        sd |= {
            f"{e}.w1.trellis_packed": gate,
            f"{e}.w3.trellis_packed": (gate + 100).to(torch.int16),
            f"{e}.w1.SU": torch.arange(I, dtype=torch.float16),
            f"{e}.w3.SU": torch.arange(I, dtype=torch.float16) + 1000,
            f"{e}.w1.SV": torch.ones(H, dtype=torch.float16),
            f"{e}.w3.SV": torch.ones(H, dtype=torch.float16),
            f"{e}.w1.Wscale": torch.tensor(0.5),
            f"{e}.w3.Wscale": torch.tensor(0.5),
            f"{e}.w2.trellis_packed": (gate + 200).to(torch.int16),
            f"{e}.w2.SU": torch.ones(H, dtype=torch.float16) * 3,
            f"{e}.w2.SV": torch.ones(I, dtype=torch.float16),
            f"{e}.w2.Wscale": torch.tensor(0.25),
        }
    return sd


def _loaded(n_exp=1):
    experts = GLQStackedGatedExperts(n_exp, H, I, nn.SiLU(), codebook_type="trellis",
                                     leaves=LFM2_LEAVES)
    res = experts.load_state_dict(_lfm2_state(n_exp), strict=False)
    return experts, res


def test_every_lfm2_tensor_is_consumed():
    _, res = _loaded()
    assert not res.unexpected_keys, res.unexpected_keys


def test_fusing_puts_w1_rows_first_and_w2_becomes_down():
    """w1 is the gate (transformers' own loader concatenates w1 then w3 into gate_up_proj,
    and the forward chunks gate first); w2 is the down projection."""
    experts, _ = _loaded()
    sd = _lfm2_state()
    pair = experts[0]
    assert pair.fuse_gate_up() is True
    f = pair.gate_up_proj
    assert torch.equal(f.trellis_packed, torch.cat([sd["0.w1.trellis_packed"],
                                                    sd["0.w3.trellis_packed"]]))
    assert torch.equal(f.SU, torch.cat([sd["0.w1.SU"], sd["0.w3.SU"]]))
    assert torch.equal(pair.down_proj.trellis_packed, sd["0.w2.trellis_packed"])
    assert torch.equal(pair.down_proj.SU, sd["0.w2.SU"])


def test_after_fusing_the_pair_looks_like_any_other():
    """The forward, the stacked CPU build and the arena all read gate_up_proj/down_proj;
    the checkpoint-named pads must be gone, not left beside them holding a second copy."""
    experts, _ = _loaded()
    experts[0].fuse_gate_up()
    names = {n for n, _ in experts.named_modules()}
    assert {"0.gate_up_proj", "0.down_proj"} <= names
    assert not names & {"0.w1", "0.w3", "0.w2"}, sorted(names)


def test_fusing_into_stacked_reads_the_lfm2_pads():
    """The direct-to-stacked fuse (GLQ_HF_MOE_CPU_FUSED) reads the pads too."""
    experts, _ = _loaded(n_exp=2)
    sd = _lfm2_state(n_exp=2)
    assert experts.fuse_into_stacked() is None
    for e in range(2):
        assert torch.equal(experts._w13_packed[e], torch.cat(
            [sd[f"{e}.w1.trellis_packed"], sd[f"{e}.w3.trellis_packed"]]))
        assert torch.equal(experts._w2_packed[e], sd[f"{e}.w2.trellis_packed"])
        assert experts[e].down_proj.trellis_packed.data_ptr() == \
            experts._w2_packed[e].data_ptr()
