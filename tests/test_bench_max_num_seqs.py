"""``max_num_seqs`` has to reach the engine, not just exist in a default.

``build_llm_kwargs`` already carries a hybrid-GDN-aware default and a comment explaining
why: on a model with linear-attention layers, **every decode slot reserves a Mamba cache
block before a single request arrives**, so vLLM refuses to start when max_num_seqs
exceeds the blocks that fit.

But ``load()`` never accepted the parameter, so it was pinned at 64 and no caller could
move it. That is not a cosmetic gap — on Qwen3.8-Flash-Next-GLQ (73.3 GiB of weights on a
96 GiB card) the two failures are a vice:

* ``--gpu-mem-util 0.9`` → ``max_num_seqs (64) exceeds available Mamba cache blocks (61)``
  and the engine refuses to start.
* ``--gpu-mem-util 0.95`` → starts, then the 248,320-token vocabulary makes one
  prompt_logprobs step ask for ~16 GB of logits against the ~4.7 GiB left outside the KV
  pool, and EngineCore dies mid-run.

Lowering concurrency relieves both at once, and it is free for perplexity — the task
prefills chunks and generates one throwaway token.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.bench.runtime import build_llm_kwargs, serving_command  # noqa: E402


def test_the_default_is_the_hybrid_gdn_safe_one():
    """Unchanged behaviour for every existing caller and recorded run."""
    assert build_llm_kwargs("m")["max_num_seqs"] == 64


def test_an_explicit_value_reaches_the_engine_kwargs():
    assert build_llm_kwargs("m", max_num_seqs=8)["max_num_seqs"] == 8


class _Stop(Exception):
    """Sentinel: LLM() reached, so every kwarg was already assembled."""


def test_load_forwards_it(monkeypatch):
    """The gap this closes: ``load()`` took gpu_mem_util and max_model_len but not this,
    so the only knob that relieves a Mamba-block shortage was unreachable from the CLI.

    vLLM is absent from CI and from the dev machine, and ``load()`` imports it on the
    first line — so a naive spy never reaches the code under test and passes vacuously.
    Stub the module instead and stop at ``LLM(**kw)``, which is the point by which the
    kwargs must be complete.
    """
    import types
    import glq.bench.runtime as rt

    seen = {}

    def _spy(model, **kw):
        seen.update(kw)
        return {"model": model, **kw}

    def _llm(**kw):
        raise _Stop

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=_llm))
    monkeypatch.setattr(rt, "build_llm_kwargs", _spy)
    with pytest.raises(_Stop):
        rt.load("m", max_num_seqs=8)
    assert seen.get("max_num_seqs") == 8, "load() dropped max_num_seqs"


def test_load_still_defaults_without_the_argument(monkeypatch):
    """Existing callers must be byte-for-byte unaffected."""
    import types
    import glq.bench.runtime as rt

    seen = {}
    monkeypatch.setitem(sys.modules, "vllm",
                        types.SimpleNamespace(LLM=lambda **kw: (_ for _ in ()).throw(_Stop())))
    monkeypatch.setattr(rt, "build_llm_kwargs",
                        lambda model, **kw: (seen.update(kw), {"model": model})[1])
    with pytest.raises(_Stop):
        rt.load("m")
    assert seen.get("max_num_seqs") == 64


def test_it_is_recorded_in_the_serving_command():
    """A PPL number measured at concurrency 8 is not the same run as one at 64; the
    reproduction command has to say so."""
    cmd = serving_command("m", build_llm_kwargs("m", max_num_seqs=8))
    assert "--max-num-seqs 8" in cmd
