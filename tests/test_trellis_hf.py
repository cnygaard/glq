"""S0 HF-integration wiring test: GLQ trellis load factory.

Exercises the load-side glue added to hf_integration.py — GLQConfig round-tripping the
`variant`, and `_process_model_after_weight_loading` rebuilding ONE shared TrellisCodebook
(K from the packed shape, tlut from the buffer) and attaching it to every trellis layer so
they forward correctly. The full from_pretrained path is validated on the box (PPL gate).
CPU-only, seeded.
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("transformers")  # glq.hf_integration imports transformers at module top
import glq.trellis as gt  # noqa: E402
from glq.hf_integration import GLQConfig, GLQQuantizer  # noqa: E402
from glq.quantized_linear import E8RHTLinear  # noqa: E402


class _Cfg:
    _name_or_path = None
    torch_dtype = "float32"


class _Model(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.config = _Cfg()

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


# One tlut for the whole "model" — every trellis layer shares a single codebook,
# exactly as a real checkpoint does (the factory rebuilds one shared codebook).
torch.manual_seed(100)
_SHARED_TLUT = (torch.randn(2 ** 9, 2) * 0.9682458365518543).to(torch.float16)


def _make_trellis_layer(in_f, out_f, seed, K=2):
    torch.manual_seed(seed)
    W = (torch.randn(out_f, in_f) * 0.05).float()
    X = torch.randn(512, in_f)
    H = (X.T @ X) / 512
    cb = gt.TrellisCodebook(variant="hyb", K=K, tlut=_SHARED_TLUT.clone())
    W_hat, art = gt.quantize_layer_trellis_rht(W, H, cb)
    layer = E8RHTLinear(in_f, out_f, codebook_type="trellis")
    layer.load_state_dict({
        "trellis_packed": art["trellis_packed"], "tlut": art["tlut"],
        "SU": art["SU"], "SV": art["SV"],
        "Wscale": torch.tensor(art["Wscale"], dtype=torch.float32),
    }, strict=False)
    return layer, W_hat


def test_glqconfig_roundtrips_trellis_variant():
    cfg = GLQConfig(codebook="trellis", variant="hyb", codesz=16, bpw=2)
    d = cfg.to_dict()
    assert d["codebook"] == "trellis" and d["variant"] == "hyb"
    assert GLQConfig(**d).variant == "hyb"
    # non-trellis configs don't leak a variant key
    assert "variant" not in GLQConfig(codebook="e8_shell").to_dict()


@pytest.mark.parametrize("K", [2, 3, 4])
def test_trellis_factory_builds_one_shared_codebook_and_forwards(K):
    """The factory recovers K from ``trellis_packed.shape[1] // 16`` — the config's ``bpw``
    is NOT what sizes the codebook. Run every native rate: a K it can't recover would
    rebuild the wrong trellis and decode to garbage, silently."""
    l1, wh1 = _make_trellis_layer(128, 96, seed=0, K=K)
    l2, wh2 = _make_trellis_layer(96, 64, seed=1, K=K)
    model = _Model(l1, l2)
    q = GLQQuantizer(GLQConfig(codebook="trellis", variant="hyb", codesz=16, bpw=K))

    q._process_model_after_weight_loading(model)

    # exactly one shared codebook, attached to every trellis layer
    assert l1.codebook is not None
    assert l1.codebook is l2.codebook
    assert getattr(l1.codebook, "is_trellis", False)
    assert l1.codebook.K == K and l1.codebook.variant == "hyb"

    torch.manual_seed(7)
    x = torch.randn(3, 128)
    h = l1(x)
    assert torch.allclose(h, x @ wh1.float().T, atol=2e-3, rtol=2e-3)
    # l2 forwards correctly on its actual input (each layer matches its own dense dequant)
    assert torch.allclose(l2(h), h @ wh2.float().T, atol=2e-3, rtol=2e-3)
