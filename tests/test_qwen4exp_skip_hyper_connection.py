"""Qwen4Exp's hyper-connection mixing matrices stay bf16.

``input_mix_weight_down`` and ``input_mix_weight_up`` decide how the residual streams
combine — the role a MoE router plays — and two independent lines of evidence say not to
quantize them:

**Quality.** On the 3 bpw Qwen3.8-Flash-Next run they are the worst-reconstructing group
in the checkpoint: mean SQNR **15.33 dB** against a 15.91 dB model mean, and the single
worst matrix at 12.82 dB (routed experts 15.90, shared expert 16.74, attention 16.4).
They are 0.35% of the model's parameters, so quantizing them buys 0.95 GiB of 77.5 (1.2%)
in exchange for the largest per-matrix error in the file, on its most sensitive weights.

**Servability.** vLLM will not accept them quantized under any quantizer. It builds them
with ``quant_config=None`` — the down projection merged with ``block_inject_weight`` into
one ``MergedColumnParallelLinear``, the up projection as a ``ReplicatedLinear`` — so
``get_quant_method`` is never called, and a GLQ checkpoint fails weight loading with
``ValueError: There is no module or parameter named ... input_mix_weight_down``. Upstream
FP8 checkpoints leave them bf16 for the same reason.

``.block_inject_weight``, the third shard of that merged linear, was already skipped. This
completes the set.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch", reason="torch not installed")
import torch.nn as nn  # noqa: E402

from glq.quantize_model import _MODEL_PROFILES, _collect_linears  # noqa: E402

QWEN4EXP = "Qwen4ExpForConditionalGeneration"
PROFILE = _MODEL_PROFILES[QWEN4EXP]


class _HyperConnection(nn.Module):
    """The real shapes, from layer_metrics.json: lora_rank 320, hc hidden 10240."""

    def __init__(self):
        super().__init__()
        self.input_mix_weight_down = nn.Linear(10240, 320, bias=False)
        self.input_mix_weight_up = nn.Linear(320, 10240, bias=False)
        self.block_inject_weight = nn.Linear(10240, 4, bias=False)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_hyper_connection = _HyperConnection()
        self.mlp_hyper_connection = _HyperConnection()
        # One ordinary projection, so a test that skips everything would fail.
        self.o_proj = nn.Linear(2560, 2560, bias=False)


@pytest.mark.parametrize("name", ["input_mix_weight_down", "input_mix_weight_up"])
@pytest.mark.parametrize("role", ["attn_hyper_connection", "mlp_hyper_connection"])
def test_the_mixing_matrices_are_not_collected(name, role):
    kept = _collect_linears(_Block(), PROFILE)
    assert f"{role}.{name}" not in kept


def test_the_injection_shard_stays_skipped():
    """It shares the merged linear with input_mix_weight_down; skipping one without the
    other would still leave that parameter unloadable."""
    kept = _collect_linears(_Block(), PROFILE)
    assert "attn_hyper_connection.block_inject_weight" not in kept


def test_every_shard_of_the_merged_linear_is_skipped_together():
    """vLLM packs ``[input_mix_weight_down, block_inject_weight, _input_mix_padding]``
    into one parameter. A mix of GLQ and bf16 shards cannot be loaded into it, so the
    set must be all-or-nothing."""
    kept = _collect_linears(_Block(), PROFILE)
    merged_shards = {"input_mix_weight_down", "block_inject_weight"}
    assert not {n.rsplit(".", 1)[-1] for n in kept} & merged_shards


def test_ordinary_projections_are_still_collected():
    """The guard must not become a blanket exclusion."""
    assert "o_proj" in _collect_linears(_Block(), PROFILE)


def test_the_skip_list_is_dot_anchored():
    """Entries are matched as a suffix, or as a namespace when they end in a dot. An
    un-anchored substring would silently skip unrelated layers on other architectures."""
    for entry in PROFILE["skip_linears"]:
        assert entry.startswith(".") or entry.endswith("."), (
            f"{entry!r} is neither dot-anchored nor a dotted namespace")


def test_a_namespace_entry_matches_beneath_itself():
    """``'.indexer.'`` means "everything under the indexer". Matched with a plain
    ``endswith`` it matched nothing — no module name ends in a dot — so Qwen4Exp's 12
    index_qk_proj matrices were quantized despite the profile declaring them skipped."""
    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.indexer = nn.Module()
            self.indexer.index_qk_proj = nn.Linear(2560, 640, bias=False)
            self.o_proj = nn.Linear(2560, 2560, bias=False)

    kept = _collect_linears(_Attn(), PROFILE)
    assert "indexer.index_qk_proj" not in kept
    assert "o_proj" in kept


def test_other_architectures_are_unaffected():
    """Profiles are per-architecture; adding these must not change anyone else's
    checkpoint bytes."""
    for arch, profile in _MODEL_PROFILES.items():
        if arch == QWEN4EXP:
            continue
        skip = profile.get("skip_linears") or ()
        assert not any("input_mix_weight" in s for s in skip)
