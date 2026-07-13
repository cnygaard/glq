"""S0 inference-wiring test: E8RHTLinear trellis path.

Proves a *compressed* trellis layer (packed int16 + SU/SV/Wscale/tlut, loaded via
state_dict) forwards to the SAME output as the quantizer's dense W_hat — i.e. the stored
bytes + the layer's decode + inverse-RHT reproduce the quantized weight. This is the
per-layer half of the S0 round-trip gate (the model-level PPL gate runs on the box).
CPU-only, seeded.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import glq.trellis as gt  # noqa: E402
from glq.quantized_linear import E8RHTLinear  # noqa: E402


def _quantized_layer(in_f, out_f, seed=0):
    torch.manual_seed(seed)
    W = (torch.randn(out_f, in_f) * 0.05).float()
    X = torch.randn(512, in_f)
    H = (X.T @ X) / 512
    tlut = (torch.randn(2 ** 9, 2) * 0.9682458365518543).to(torch.float16)
    cb = gt.TrellisCodebook(variant="hyb", K=2, tlut=tlut, device="cpu")
    W_hat, art = gt.quantize_layer_trellis_rht(W, H, cb)
    return cb, W_hat, art


def _load_layer(in_f, out_f, cb, art):
    layer = E8RHTLinear(in_f, out_f, codebook_type="trellis")
    layer.load_state_dict({
        "trellis_packed": art["trellis_packed"],
        "tlut": art["tlut"],
        "SU": art["SU"],
        "SV": art["SV"],
        "Wscale": torch.tensor(art["Wscale"], dtype=torch.float32),
    }, strict=False)
    layer.set_codebook(cb)
    return layer


def test_trellis_layer_is_flagged_and_unpadded():
    layer = E8RHTLinear(128, 96, codebook_type="trellis")
    assert layer._is_trellis and not layer._is_e8p
    assert layer.m_pad == 96 and layer.n_pad == 128     # block-diag, no padding


def test_trellis_layer_forward_matches_dense_dequant():
    in_f, out_f = 128, 96
    cb, W_hat, art = _quantized_layer(in_f, out_f)
    layer = _load_layer(in_f, out_f, cb, art)

    torch.manual_seed(9)
    x = torch.randn(4, in_f)
    y = layer(x)
    y_ref = x @ W_hat.float().T
    assert torch.allclose(y, y_ref, atol=2e-3, rtol=2e-3), (y - y_ref).abs().max().item()


def test_trellis_layer_load_resizes_compressed_buffers():
    # 0-size trellis_packed/tlut placeholders must resize to the checkpoint shapes.
    in_f, out_f = 256, 128
    cb, _, art = _quantized_layer(in_f, out_f, seed=3)
    layer = _load_layer(in_f, out_f, cb, art)
    assert layer.trellis_packed.shape == art["trellis_packed"].shape
    assert layer.trellis_packed.dtype == torch.int16
    assert layer.tlut.shape == art["tlut"].shape


def test_trellis_layer_state_dict_roundtrips_through_save_load():
    # save_pretrained/from_pretrained-style: state_dict out → fresh layer in → same forward.
    in_f, out_f = 128, 96
    cb, W_hat, art = _quantized_layer(in_f, out_f, seed=5)
    layer = _load_layer(in_f, out_f, cb, art)
    sd = layer.state_dict()

    layer2 = E8RHTLinear(in_f, out_f, codebook_type="trellis")
    layer2.load_state_dict(sd, strict=False)
    layer2.set_codebook(cb)

    torch.manual_seed(11)
    x = torch.randn(3, in_f)
    assert torch.allclose(layer(x), layer2(x), atol=0, rtol=0)   # bit-exact reload
