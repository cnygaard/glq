"""CPU tests for the vLLM-side WikiText-2 PPL adapter.

The arithmetic is the whole adapter, and it is the kind that fails silently: an off-by-one
in the teacher-forcing shift, or averaging in the wrong space (mean of exp vs exp of mean),
produces a plausible number rather than an error. These pin it against hand-computed values
with a fake vLLM handle — no GPU, no model, no dataset download.
"""
from __future__ import annotations

import math
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.bench.tasks import perplexity_vllm as pv  # noqa: E402


class _LP:
    def __init__(self, logprob):
        self.logprob = logprob


class _Out:
    def __init__(self, prompt_logprobs):
        self.prompt_logprobs = prompt_logprobs


class _FakeLLM:
    """Returns a fixed per-position logprob for whichever token actually appears."""
    def __init__(self, logprob_per_pos):
        self.logprob_per_pos = logprob_per_pos
        self.calls = []

    def generate(self, prompts, sp, **kw):
        self.calls.append((prompts, sp))
        outs = []
        for p in prompts:
            ids = p["prompt_token_ids"]
            # position 0 has no conditional distribution — vLLM returns None there
            plp = [None] + [{ids[i]: _LP(self.logprob_per_pos)} for i in range(1, len(ids))]
            outs.append(_Out(plp))
        return outs


def _ctx(llm, ids):
    tok = types.SimpleNamespace(__call__=None)
    ctx = types.SimpleNamespace(
        model="fake/model",
        handle=types.SimpleNamespace(llm=llm, tokenizer=_FakeTok(ids)))
    return ctx


class _FakeTok:
    def __init__(self, ids):
        self._ids = ids

    def __call__(self, text):
        return types.SimpleNamespace(input_ids=self._ids)


@pytest.fixture
def _fake_dataset(monkeypatch):
    """Stub datasets.load_dataset so the test needs no network."""
    mod = types.ModuleType("datasets")
    mod.load_dataset = lambda *a, **k: {"text": ["irrelevant"]}
    monkeypatch.setitem(sys.modules, "datasets", mod)
    # SamplingParams is only constructed, never inspected by the fake LLM.
    vmod = types.ModuleType("vllm")
    vmod.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "vllm", vmod)


def test_ppl_equals_exp_of_mean_negative_logprob(_fake_dataset):
    """Every scored position has logprob -1.5 => CE 1.5 => PPL exp(1.5)."""
    seqlen, n_chunks = 8, 3
    ids = list(range(1, seqlen * n_chunks + 1))
    llm = _FakeLLM(-1.5)
    res, _ = pv.run(_ctx(llm, ids), {"seqlen": seqlen, "max_chunks": n_chunks})
    assert res.metric == "perplexity"
    assert res.value == pytest.approx(math.exp(1.5), rel=1e-9)


class _VaryingLLM(_FakeLLM):
    """Different logprob per chunk — required to tell exp(mean) from mean(exp)."""
    def __init__(self, per_chunk):
        super().__init__(None)
        self.per_chunk = per_chunk

    def generate(self, prompts, sp, **kw):
        self.calls.append((prompts, sp))
        outs = []
        for c, p in enumerate(prompts):
            ids = p["prompt_token_ids"]
            lp = self.per_chunk[c]
            outs.append(_Out([None] + [{ids[i]: _LP(lp)} for i in range(1, len(ids))]))
        return outs


def test_averages_in_log_space_not_ppl_space(_fake_dataset):
    """PPL must be exp(mean CE), NOT mean(exp CE).

    With a constant logprob the two agree, so a constant fixture cannot catch this — hence
    varying per-chunk values. Jensen makes mean(exp) strictly larger, so the wrong order
    yields a plausible but inflated PPL that no other assertion would flag.
    """
    seqlen = 8
    per_chunk = [-0.5, -1.5, -3.0]
    ids = list(range(1, seqlen * len(per_chunk) + 1))
    res, _ = pv.run(_ctx(_VaryingLLM(per_chunk), ids),
                    {"seqlen": seqlen, "max_chunks": len(per_chunk)})
    ces = [-lp for lp in per_chunk]
    exp_of_mean = math.exp(sum(ces) / len(ces))
    mean_of_exp = sum(math.exp(c) for c in ces) / len(ces)
    assert exp_of_mean != pytest.approx(mean_of_exp, rel=1e-6)   # fixture is discriminating
    assert res.value == pytest.approx(exp_of_mean, rel=1e-9)


def test_scores_seqlen_minus_one_positions_per_chunk(_fake_dataset):
    """Teacher forcing cannot score position 0, mirroring HF's logits[:, :-1] shift.
    Getting this wrong silently changes the denominator and hence the PPL."""
    seqlen, n_chunks = 8, 3
    ids = list(range(1, seqlen * n_chunks + 1))
    res, _ = pv.run(_ctx(_FakeLLM(-1.5), ids),
                    {"seqlen": seqlen, "max_chunks": n_chunks})
    assert res.extra["tokens_scored"] == n_chunks * (seqlen - 1)
    assert res.extra["n_chunks"] == n_chunks


def test_chunks_are_non_overlapping_and_capped(_fake_dataset):
    """max_chunks must cap, and windows must not overlap — an overlapping window would
    score some tokens twice and quietly lower PPL."""
    seqlen = 4
    ids = list(range(1, 41))                      # 10 chunks available
    llm = _FakeLLM(-1.0)
    pv.run(_ctx(llm, ids), {"seqlen": seqlen, "max_chunks": 3})
    prompts = llm.calls[0][0]
    assert len(prompts) == 3
    sent = [p["prompt_token_ids"] for p in prompts]
    assert sent == [ids[0:4], ids[4:8], ids[8:12]]


def test_raises_when_backend_returns_no_prompt_logprobs(_fake_dataset):
    """A backend without prompt_logprobs must raise, NOT fall back to another PPL
    definition — that would put two incompatible measures in one column."""
    class _NoLP(_FakeLLM):
        def generate(self, prompts, sp, **kw):
            return [_Out(None) for _ in prompts]

    with pytest.raises(RuntimeError, match="prompt_logprobs"):
        pv.run(_ctx(_NoLP(-1.0), list(range(1, 25))), {"seqlen": 8, "max_chunks": 2})


def test_requests_prompt_logprobs_and_greedy(_fake_dataset):
    """Mechanism: PPL is only teacher-forced if prompt_logprobs is actually requested."""
    llm = _FakeLLM(-1.0)
    pv.run(_ctx(llm, list(range(1, 25))), {"seqlen": 8, "max_chunks": 2})
    sp = llm.calls[0][1]
    assert sp.prompt_logprobs == 0
    assert sp.temperature == 0.0
    assert sp.max_tokens == 1
