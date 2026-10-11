"""Quantizing the MTP head: discovery, grouping, and the no-op guarantee.

The MTP head is a **vLLM-only** structure: HF's Qwen4Exp discards it outright
(`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`), so there is no HF block to construct and
no HF forward to hook. The quantizer therefore treats it tensor-wise — discover the `mtp.*`
weights, unstack the experts into virtual linears exactly as the main decoder's MoE path does,
and quantize those.

These tests are pure table-work: no GPU, no vLLM, no model.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.quantize_model import mtp_sublayer_weights  # noqa: E402


def test_a_checkpoint_without_an_mtp_head_is_untouched():
    """THE safety property. Every published checkpoint except Flash-Next has no `mtp.*` keys,
    and the quantize path must be bit-for-bit unchanged for them — this is what makes the
    byte-identical gate (`benchmarks/_ckpt_bytecmp.py`) pass. An empty result means the
    post-pass contributes no artifacts and supersedes no originals."""
    keys = ["model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.experts.down_proj",
            "model.embed_tokens.weight", "lm_head.weight"]
    assert mtp_sublayer_weights(keys) == {}


def test_the_mtp_segment_is_matched_on_a_dotted_segment_not_a_substring():
    """`model.layers.0.mtpool.weight` contains "mtp" and is an ordinary text weight. Matching it
    would quantize a decoder tensor as if it were the draft head — and `_is_nontext` already
    documents this exact trap for the byte accounting."""
    assert mtp_sublayer_weights(["model.layers.0.mtpool.weight"]) == {}
    assert mtp_sublayer_weights(["mtpool.weight"]) == {}


def test_plain_linears_are_grouped_by_their_sublayer_prefix():
    """The quantizer keys artifacts by sublayer prefix, so discovery must return the prefix the
    checkpoint uses, with `.weight` stripped."""
    got = mtp_sublayer_weights([
        "mtp.layers.48.self_attn.q_proj.weight",
        "mtp.layers.48.self_attn.o_proj.weight",
        "mtp.layers.48.input_layernorm.weight",      # 1-D, not quantizable
    ])
    assert set(got) == {"mtp.layers.48.self_attn.q_proj",
                        "mtp.layers.48.self_attn.o_proj"}
    assert got["mtp.layers.48.self_attn.q_proj"] == ["mtp.layers.48.self_attn.q_proj.weight"]


def test_stacked_experts_are_reported_whole_for_unstacking():
    """The MTP head's experts are stacked `[512, ...]` in one tensor, exactly like the main
    decoder's — measured from vLLM's own config: num_experts=512, hidden_dim=2560,
    intermediate_size=640. The caller unstacks them into per-expert virtual linears the way the
    `gemma4_experts` loop already does, so discovery must hand back the stacked keys rather
    than pretending they are ordinary 2-D linears."""
    got = mtp_sublayer_weights([
        "mtp.layers.48.mlp.experts.gate_up_proj",
        "mtp.layers.48.mlp.experts.down_proj",
    ])
    assert set(got) == {"mtp.layers.48.mlp.experts"}
    assert sorted(got["mtp.layers.48.mlp.experts"]) == [
        "mtp.layers.48.mlp.experts.down_proj",
        "mtp.layers.48.mlp.experts.gate_up_proj"]


def test_norms_and_biases_are_left_alone():
    """Only 2-D weight matrices are quantizable; norms, biases and the router gate stay as they
    are, which is what the main decoder path does too."""
    got = mtp_sublayer_weights([
        "mtp.layers.48.input_layernorm.weight",
        "mtp.layers.48.mlp.gate.weight",          # router: tiny, left bf16 like the decoder's
        "mtp.layers.48.self_attn.q_proj.bias",
        "mtp.norm.weight",
    ])
    assert "mtp.layers.48.input_layernorm" not in got
    assert "mtp.norm" not in got


# ---- the --mtp-bpw flag ------------------------------------------------------------------

def test_mtp_bpw_defaults_to_the_models_own_bpw():
    """One bpw for the whole checkpoint unless asked otherwise: it keeps `layer_bpw`, the
    config and the footprint accounting simple. The flag exists so acceptance-rate vs
    footprint can be swept — a more faithful drafter accepts more tokens, which is the entire
    point of MTP — without touching code."""
    from glq.quantize_model import resolve_mtp_bpw
    assert resolve_mtp_bpw(None, model_bpw=3) == 3
    assert resolve_mtp_bpw(None, model_bpw=4) == 4
    assert resolve_mtp_bpw(6, model_bpw=3) == 6


def test_a_fractional_model_bpw_still_yields_an_integer_mtp_bpw():
    """Mixed/fractional bpw is a per-layer *average* for the decoder, but the MTP head is one
    layer quantized at one rung — and trellis supports uniform integer bpw only (2-4 native,
    5-8 stacked RVQ). Rounding DOWN keeps it on a supported rung and errs toward the smaller
    draft, which is the safe direction on the card that motivated this."""
    from glq.quantize_model import resolve_mtp_bpw
    assert resolve_mtp_bpw(None, model_bpw=3.5) == 3
    assert resolve_mtp_bpw(None, model_bpw=2.5) == 2


def test_the_cli_exposes_mtp_bpw():
    """Asserted on the parser rather than trusted: a flag the parser does not carry is a flag
    nobody can use, and the sweep this exists for would silently run at the default."""
    from glq.quantize_model import build_parser
    opts = {a.dest for a in build_parser()._actions}
    assert "mtp_bpw" in opts


# ---------------------------------------------------------------------------
# The post-pass itself. A stub quantizer stands in for the real encode so these
# stay CPU-only table-work: what is under test is the plumbing — which prefixes
# the artifacts land under, which originals are superseded, and what is left alone.
# ---------------------------------------------------------------------------

def _fake_mtp_state_dict(n_experts=3, hidden=16, inter=8):
    """The shape Qwen3.8-Flash-Next's MTP head actually has, shrunk: one block with a
    plain attention projection, stacked routed experts, and norms/router alongside."""
    import torch
    return {
        "mtp.layers.0.input_layernorm.weight": torch.ones(hidden),
        "mtp.layers.0.self_attn.q_proj.weight": torch.randn(hidden, hidden),
        "mtp.layers.0.mlp.gate.weight": torch.randn(n_experts, hidden),
        "mtp.layers.0.mlp.experts.gate_up_proj": torch.randn(n_experts, 2 * inter, hidden),
        "mtp.layers.0.mlp.experts.down_proj": torch.randn(n_experts, hidden, inter),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(hidden, hidden),
    }


def _stub_quantize(W, H, codebook, bpw=2, **kw):
    """Stands in for `quantize_layer_e8_shell_rht`, same contract: (W_hat, arts, metrics).
    Artifacts carry one row-indexed and one shared key so the gate/up row-split is
    genuinely exercised rather than stubbed past."""
    import torch
    rows, cols = W.shape
    arts = {"Qidxs": torch.zeros(rows, 2, dtype=torch.int16),
            "SV": torch.ones(cols)}
    return W.clone(), arts, {"sqnr": 42.0, "bpw": bpw, "Wscale": 1.0, "proxy_loss": 0.0}


def test_expert_artifacts_use_the_published_per_expert_prefixes():
    """`mlp.experts.{e}.gate_proj` / `.up_proj` / `.down_proj` — read off a published
    checkpoint's layer_metrics.json, not guessed. Matching it exactly is what lets
    `GLQFusedMoEMethod` load the draft head with no new loader code: a different naming
    would need a second expert mapping for a structure identical to the decoder's."""
    from glq.quantize_model import quantize_mtp_head
    arts, _, _ = quantize_mtp_head(_fake_mtp_state_dict(), codebook=None, bpw=3,
                                   quantize_fn=_stub_quantize)
    for e in range(3):
        for leaf in ("gate_proj", "up_proj", "down_proj"):
            assert f"mtp.layers.0.mlp.experts.{e}.{leaf}" in arts


def test_the_stacked_originals_are_superseded():
    """The stacked `[E, ...]` tensors must not also be written to the checkpoint — they are
    replaced by per-expert artifacts, and shipping both would inflate the file by the full
    bf16 head (4.69 GiB on Flash-Next) while vLLM loaded the quantized copy."""
    from glq.quantize_model import quantize_mtp_head
    _, _, superseded = quantize_mtp_head(_fake_mtp_state_dict(), codebook=None, bpw=3,
                                         quantize_fn=_stub_quantize)
    assert superseded == {"mtp.layers.0.mlp.experts.gate_up_proj",
                          "mtp.layers.0.mlp.experts.down_proj"}


def test_a_plain_projection_is_quantized_under_its_own_prefix():
    from glq.quantize_model import quantize_mtp_head
    arts, _, _ = quantize_mtp_head(_fake_mtp_state_dict(), codebook=None, bpw=3,
                                   quantize_fn=_stub_quantize)
    assert "mtp.layers.0.self_attn.q_proj" in arts


def test_norms_the_router_and_the_decoder_are_left_alone():
    """Three separate ways this could corrupt a model: quantizing a 1-D norm, quantizing the
    router (whose argmax decides which expert runs), or straying out of the head into the
    decoder it was handed alongside."""
    from glq.quantize_model import quantize_mtp_head
    arts, _, superseded = quantize_mtp_head(_fake_mtp_state_dict(), codebook=None, bpw=3,
                                            quantize_fn=_stub_quantize)
    assert not [p for p in arts if "layernorm" in p or p.endswith("mlp.gate")]
    assert not [p for p in arts if p.startswith("model.")]
    assert not [k for k in superseded if k.startswith("model.")]


def test_every_quantized_prefix_reports_its_bpw():
    """`layer_bpw` is built from these metrics, and it is the one thing that makes the two
    halves of this work meet: a quantized MTP tensor absent from `layer_bpw` is declined by
    `glq_vllm`'s whitelist guard, and vLLM then looks for bf16 weights that are gone."""
    from glq.quantize_model import quantize_mtp_head
    arts, metrics, _ = quantize_mtp_head(_fake_mtp_state_dict(), codebook=None, bpw=3,
                                         quantize_fn=_stub_quantize)
    assert set(metrics) == set(arts)
    assert {m["bpw"] for m in metrics.values()} == {3}


def test_the_post_pass_never_calls_the_quantizer_without_an_mtp_head():
    """The no-op guarantee, asserted on the mechanism rather than the output: for every
    checkpoint but Flash-Next this pass must not run an encode at all, so it cannot perturb
    RNG, allocate, or change a byte."""
    from glq.quantize_model import quantize_mtp_head
    calls = []

    def _tripwire(*a, **kw):
        calls.append(a)
        raise AssertionError("quantizer invoked with no mtp.* weights present")

    arts, metrics, superseded = quantize_mtp_head(
        {"model.layers.0.self_attn.q_proj.weight": None}, codebook=None, bpw=3,
        quantize_fn=_tripwire)
    assert (arts, metrics, superseded) == ({}, {}, set())
    assert calls == []


def test_zero_means_match_the_model_rather_than_zero_bits():
    """`--mtp-bpw 0` is the "same rung as the decoder" sentinel. It exists because the flag is
    also the **opt-in**: omitting it leaves the head bf16, which is what keeps an existing
    Flash-Next resume store addressable (its run key has no mtp field) and what keeps every
    already-published checkpoint reproducible. So "quantize the head, I don't care which rung"
    needs a value to pass, and 0 bits is not a thing anyone could mean."""
    from glq.quantize_model import resolve_mtp_bpw
    assert resolve_mtp_bpw(0, model_bpw=3) == 3


def test_a_quantized_head_is_not_reported_as_never_loaded():
    """`nontext_bytes` means "a text-only serve does not load this", and the installer
    subtracts it to size a card. A head quantized *in order to enable speculation* is loaded,
    so counting it absent understates the resident footprint — the direction that makes a card
    look able to hold a model it cannot. Classify on whether the head was quantized, not on
    the name, which still begins with `mtp` either way."""
    import torch
    from glq.quantize_model import _offloadable_bytes
    sd = {"mtp.layers.0.self_attn.q_proj.Qidxs": torch.zeros(64, 8, dtype=torch.int16),
          "model.layers.0.self_attn.q_proj.Qidxs": torch.zeros(64, 8, dtype=torch.int16)}
    assert _offloadable_bytes(sd)[2] == 1024                   # bf16 head: not loaded
    assert _offloadable_bytes(sd, mtp_is_loaded=True)[2] == 0   # quantized: loaded


# ---------------------------------------------------------------------------
# The bpw a trellis encode actually uses comes from the CODEBOOK, not from the
# `bpw` argument: `quantize_layer_trellis_rht(W, H, cb, ...)` has no bpw parameter
# at all. So handing the post-pass the model's codebook and a different --mtp-bpw
# would encode the head at the model's rung while `layer_bpw` declared the other —
# and vLLM sizes `trellis_packed` at 16*layer_bpw[p], so that is a wrong buffer at
# load, not a quality question. These assert the mechanism, not the output.
# ---------------------------------------------------------------------------

class _FakeTrellisCodebook:
    is_trellis = True

    def __init__(self, K, stages=None):
        self.K = K
        if stages is not None:
            self.rvq_stages = stages


def test_the_codebooks_own_bpw_is_reported():
    """Stacked RVQ (5-8 bpw) is K=4 plus a residual stage, so the rate is the SUM over
    stages — reading `codebook.K` would call every stacked rate 4."""
    from glq.quantize_model import codebook_bpw
    assert codebook_bpw(_FakeTrellisCodebook(3)) == 3
    primary = _FakeTrellisCodebook(4)
    primary.rvq_stages = [_FakeTrellisCodebook(4), _FakeTrellisCodebook(2)]
    assert codebook_bpw(primary) == 6


def test_a_codebook_that_does_not_declare_a_rate_is_not_guessed():
    """Shell/e8p carry their rate elsewhere; returning a wrong number would be worse than
    returning none, because the guard below would then reject a correct run."""
    from glq.quantize_model import codebook_bpw
    assert codebook_bpw(object()) is None


def test_a_codebook_that_disagrees_with_the_requested_bpw_is_refused():
    """THE guard. Without it `--mtp-bpw 6` against a 3 bpw model's codebook produces a 3 bpw
    head declared as 6: it quantizes, saves, and uploads clean, then fails at serve time with
    a buffer-size error pointing nowhere near here."""
    import pytest
    from glq.quantize_model import quantize_mtp_head
    with pytest.raises(ValueError, match="codebook"):
        quantize_mtp_head(_fake_mtp_state_dict(), codebook=_FakeTrellisCodebook(3), bpw=6,
                          quantize_fn=_stub_quantize)


def test_a_matching_codebook_is_accepted():
    from glq.quantize_model import quantize_mtp_head
    arts, _, _ = quantize_mtp_head(_fake_mtp_state_dict(),
                                   codebook=_FakeTrellisCodebook(3), bpw=3,
                                   quantize_fn=_stub_quantize)
    assert arts
