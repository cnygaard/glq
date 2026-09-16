"""The vLLM quant config must carry the PLE markers, not just the weight ones.

There are two GLQConfig classes — glq.hf_integration's (which writes config.json) and
glq_vllm.config's GLQvLLMConfig (which reads it). Adding a field to the writer does nothing
for the reader, and the failure is indirect: get_quant_method falls back to "shell",
create_weights registers Qidxs [vocab, n_pad/8], and the mismatch only surfaces much later
as a bare `assert param.data.shape == loaded_weight.shape` inside vLLM's embedding loader
with no mention of GLQ at all. Observed exactly that on an L40S.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

config = pytest.importorskip("glq_vllm.config",
                             reason="vLLM not installed (CPU dev box)")


def _cfg(**over):
    # trellis_layout="kernel" is required: a 3inst config without it is rejected as a
    # pre-kernel natural-layout checkpoint, which is a separate guard from anything here.
    base = {"bpw": 4, "codebook": "trellis", "variant": "3inst", "layer_bpw": {},
            "trellis_layout": "kernel"}
    base.update(over)
    return config.GLQvLLMConfig.from_config(base)


def test_the_ple_markers_round_trip_from_config_json():
    c = _cfg(ple_codebook="trellis", ple_bpw=4)
    assert getattr(c, "ple_codebook", None) == "trellis"
    assert getattr(c, "ple_bpw", None) == 4


def test_absent_markers_mean_shell():
    """Every published gemma-4 checkpoint predates these keys and must keep loading."""
    c = _cfg()
    assert getattr(c, "ple_codebook", None) in (None, "shell")


def test_the_ple_bpw_is_independent_of_the_weight_bpw():
    """A 3 bpw model can carry a 4 bpw table. Reusing the weight bpw would size the packed
    buffer wrongly, which surfaces as an opaque shape assertion in vLLM's loader."""
    c = _cfg(bpw=3, ple_codebook="trellis", ple_bpw=4)
    assert c.bpw == 3 and getattr(c, "ple_bpw", None) == 4
