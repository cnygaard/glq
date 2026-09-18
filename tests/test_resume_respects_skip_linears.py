"""A resumed run must honour the profile's skip list, not the bank's contents.

``--resume`` replays banked per-layer artifacts instead of re-quantizing::

    arts_map, metrics_map, losses_map = resume_store.load_layer(layer_idx)
    all_artifacts.update(arts_map)

and that ``update`` takes whatever is in the bank. ``_collect_linears`` — the only thing
that reads ``skip_linears`` — runs on the *other* branch, the one resume skips.

So a bank produced before a matrix was added to ``skip_linears`` keeps supplying it, and
the re-saved checkpoint is quantized exactly where the profile now says it must not be.
That is not hypothetical: Qwen4Exp's hyper-connection mixers were banked by an 11-hour
run and only afterwards found to be unservable (vLLM builds them with
``quant_config=None``) and the worst-reconstructing group in the file (15.33 dB against a
15.91 dB model mean). Without this filter the re-save would reproduce the same broken
checkpoint and the only symptom would be the same load failure, an hour later.

The profile is the authority; the bank is a cache.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.quantize_model import _MODEL_PROFILES, _drop_skipped_artifacts  # noqa: E402

QWEN4EXP = "Qwen4ExpForConditionalGeneration"
PROFILE = _MODEL_PROFILES[QWEN4EXP]
P = "model.language_model.layers.0"

BANKED = {
    f"{P}.attn_hyper_connection.input_mix_weight_down": "art",
    f"{P}.attn_hyper_connection.input_mix_weight_up": "art",
    f"{P}.mlp_hyper_connection.input_mix_weight_down": "art",
    f"{P}.mlp_hyper_connection.input_mix_weight_up": "art",
    f"{P}.self_attn.o_proj": "art",
    f"{P}.mlp.shared_expert.gate_proj": "art",
}


def test_banked_artifacts_for_skipped_matrices_are_dropped():
    kept = _drop_skipped_artifacts(BANKED, PROFILE)
    assert not [k for k in kept if "input_mix_weight" in k]


def test_everything_else_survives():
    """A filter that over-reaches would silently drop real weights and the checkpoint
    would come out smaller *and* broken."""
    kept = _drop_skipped_artifacts(BANKED, PROFILE)
    assert f"{P}.self_attn.o_proj" in kept
    assert f"{P}.mlp.shared_expert.gate_proj" in kept
    assert len(kept) == 2


def test_the_other_skip_entries_are_honoured_too():
    """The filter reads the whole skip list, not just the newest entry."""
    banked = {f"{P}.linear_attn.in_proj_b": "art",
              f"{P}.self_attn.indexer.index_qk_proj": "art",
              f"{P}.mlp.shared_expert_gate": "art",
              f"{P}.self_attn.o_proj": "art"}
    assert list(_drop_skipped_artifacts(banked, PROFILE)) == [f"{P}.self_attn.o_proj"]


def test_a_profile_without_a_skip_list_is_a_no_op():
    """Most architectures have none; resume must behave exactly as before for them."""
    assert _drop_skipped_artifacts(BANKED, {}) == BANKED


def test_the_input_is_not_mutated():
    """``load_layer`` returns three parallel maps; filtering one in place would leave
    metrics and losses describing matrices the checkpoint no longer contains."""
    before = dict(BANKED)
    _drop_skipped_artifacts(BANKED, PROFILE)
    assert BANKED == before


def test_metrics_and_losses_filter_by_the_same_rule():
    """layer_metrics.json is published alongside the weights; leaving an entry for a
    matrix that is bf16 in the checkpoint misreports what was quantized."""
    metrics = {k: {"sqnr": 15.0} for k in BANKED}
    kept = _drop_skipped_artifacts(metrics, PROFILE)
    assert set(kept) == {f"{P}.self_attn.o_proj", f"{P}.mlp.shared_expert.gate_proj"}


@pytest.mark.parametrize("arch", [a for a in _MODEL_PROFILES
                                  if _MODEL_PROFILES[a].get("skip_linears")])
def test_no_profile_filters_everything(arch):
    """A skip entry broad enough to match an ordinary projection would empty the
    checkpoint. Guard the shape of every profile's list, not just Qwen4Exp's."""
    ordinary = {f"{P}.self_attn.o_proj": "art", f"{P}.mlp.down_proj": "art"}
    assert _drop_skipped_artifacts(ordinary, _MODEL_PROFILES[arch]) == ordinary
