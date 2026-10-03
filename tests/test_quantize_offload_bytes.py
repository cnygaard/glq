"""What the quantizer declares as host-offloadable, and why the boundaries are where they are.

`quantization_config.ple_offload_bytes` / `expert_offload_bytes` are what let the installer
size a card from the RESIDENT footprint. Get them wrong upward and a model is promised onto a
card that cannot hold it -- it loads and then fails to serve, which is a much worse outcome
than not being offered.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _helper():
    """Import just the classifier: glq.quantize_model pulls the whole quantize stack."""
    src = open(os.path.join(os.path.dirname(__file__), "..",
                            "glq", "quantize_model.py")).read()
    # From the constants, not from the def: `_offloadable_bytes` calls `_is_nontext`, which
    # sits above it with the segment tables it reads.
    i = src.index("#: First path segment of a head")
    j = src.index("def _artifact_padded_weights")
    ns: dict = {}
    exec(compile(src[i:j], "quantize_model_fragment", "exec"), ns)
    return ns["_offloadable_bytes"]


class _T:
    def __init__(self, numel, elem=2):
        self._n, self._e = numel, elem

    def numel(self):
        return self._n

    def element_size(self):
        return self._e


PLE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"


def test_ple_counts_trellis_packed_and_reproduces_the_measured_size():
    """The real table is [320001536, 40] int16 = 23.842 GiB, and resident dropped
    73.3 -> 49.53 GiB when it was offloaded -- 23.77 GiB, which is this tensor."""
    ple, _, _ = _helper()({f"{PLE}.trellis_packed": _T(320001536 * 40)})
    assert ple == 320001536 * 40 * 2
    assert abs(ple / 2 ** 30 - 23.842) < 0.01


def test_wscale_is_excluded_because_it_stays_resident():
    """Summing the whole PLE group would over-promise by 0.596 GiB. The 23.77 GiB measured
    drop matches trellis_packed alone, which is how we know Wscale does not leave."""
    ple, _, _ = _helper()({f"{PLE}.trellis_packed": _T(100), f"{PLE}.Wscale": _T(50),
                           f"{PLE}.SV": _T(10)})
    assert ple == 200


def test_a_non_ple_trellis_packed_is_not_counted():
    """Every quantized linear has a trellis_packed; only the n-gram table can be offloaded
    by GLQ's embedding path."""
    ple, _, _ = _helper()({"model.layers.2.self_attn.qkv_proj.trellis_packed": _T(999)})
    assert ple == 0


def test_experts_count_every_tensor_under_an_expert_module():
    """vLLM's `--cpu-offload-params experts` filters by NAME, so whatever sits under those
    modules leaves -- packed weights and their scales alike."""
    _, exp, _ = _helper()({"model.layers.2.mlp.experts.trellis_packed": _T(1000),
                           "model.layers.2.mlp.experts.Wscale": _T(500),
                           "model.layers.2.self_attn.qkv_proj.trellis_packed": _T(7)})
    assert exp == 1500 * 2


def test_a_dense_checkpoint_declares_nothing():
    """Emitting zeros would add noise to every dense config.json; absent means resident."""
    assert _helper()({"model.layers.0.mlp.gate_proj.trellis_packed": _T(10)}) == (0, 0, 0)


def test_non_tensors_are_skipped_rather_than_crashing():
    assert _helper()({"junk": object(), PLE + ".trellis_packed": _T(4)}) == (8, 0, 0)


# ---------------------------------------- weights a text-only serve never loads

# `resident = size - ple - experts` still OVERSTATED what lands in VRAM, because vLLM loads
# only the text decoder. Measured on Qwen3.8-Flash-Next (RTX PRO 6000, vLLM 0.30.0): at a
# 38 GiB expert budget the arithmetic predicted 15.72 GiB resident and `Model loading took`
# reported **10.1 GiB**. The 5.62 GiB difference is the MTP head and the vision tower.
#
# Counted from the checkpoint's own safetensors headers (297,256 tensors):
#     experts      42.627 GiB      MTP     4.856 GiB
#     PLE ngram    24.438 GiB      vision  0.836 GiB
#     text-other    4.766 GiB      total  77.523 GiB
#
# On a 24 GB card the gap is two window tiers: 32768 against the 131072 the hardware affords.

MTP_EXPERT = "mtp.layers.0.mlp.experts.gate_up_proj"
VISION = "model.visual.blocks.0.attn.qkv.weight"


def test_the_mtp_head_and_vision_tower_are_declared_non_text():
    ple, exp, nontext = _helper()({"mtp.fc_embedding.weight": _T(100),
                                   VISION: _T(50),
                                   "model.layers.0.mlp.gate_proj.trellis_packed": _T(7)})
    assert nontext == 300
    assert (ple, exp) == (0, 0)


def test_mtp_experts_are_counted_once_not_twice():
    """The hazard this introduces. `mtp.layers.0.mlp.experts.gate_up_proj` matches BOTH
    "experts" and the MTP head, and a consumer computing
    `size - ple - experts - nontext` would subtract it twice -- under-stating resident,
    which is the direction that promises a card it cannot hold.

    Measured: the declared expert_offload_bytes was 47.31 GiB against 42.63 GiB of real
    expert tensors, and the 4.68 GiB difference is exactly these two MTP tensors."""
    ple, exp, nontext = _helper()({MTP_EXPERT: _T(1000),
                                   "model.layers.0.mlp.experts.gate_up_proj": _T(400)})
    assert nontext == 2000, "MTP experts must land in non-text"
    assert exp == 800, "MTP experts must NOT also be counted as offloadable experts"


def test_non_text_wins_over_every_other_category():
    """Precedence, stated once: these tensors are not loaded at all, so they cannot be
    'offloaded to host memory' -- that would reserve pinned RAM for weights nothing reads."""
    ple, exp, nontext = _helper()({
        "mtp.ple.ple_embedding.ngram_embedding.trellis_packed": _T(10)})
    assert (ple, exp, nontext) == (0, 0, 20)


def test_a_text_only_checkpoint_declares_no_non_text_bytes():
    """Most checkpoints have neither head, and absent must keep meaning 'all resident'."""
    assert _helper()({"model.layers.0.mlp.experts.trellis_packed": _T(5)})[2] == 0


def test_the_classifier_matches_path_segments_not_substrings():
    """`mtp` and `visual` are matched as dotted segments. A substring test would catch an
    unrelated module whose name merely contains them, and silently drop real weights from
    the resident estimate."""
    _, _, nontext = _helper()({"model.layers.0.mtpool.weight": _T(10),
                               "model.layers.0.visualizer_proj.weight": _T(10)})
    assert nontext == 0
