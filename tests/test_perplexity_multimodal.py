"""wikitext2_ppl must judge a model by whether it HAS a causal text tower, not by
whether its architecture name contains "ConditionalGeneration".

The original guard refused any arch matching that substring. That is a name heuristic
standing in for a capability check, and it is wrong for vision-language models whose text
half is an ordinary dense causal transformer: teacher-forced PPL over a text-only batch is
perfectly well defined there, and it is the single cheapest quality signal we have for a
bpw ladder (~5 min/arm, resolves ~0.1 differences that no MMLU-Pro/AIME sample size can).
Refusing it costs the one measurement that can rank adjacent bpw rungs.

Muse-Glimmer-30B is the concrete case: MuseGlimmerForConditionalGeneration, rejected by
AutoModelForCausalLM but loaded fine by AutoModelForImageTextToText, and it generates
coherent text. These tests pin the capability probe with fake auto-classes — no GPU, no
model, no dataset download.

Also pinned: the dtype default. Every existing wikitext2_ppl record was measured in
float16, so the default MUST stay float16 or new numbers silently stop being comparable
with the stored ones. bf16-native models (logit softcapping, large outliers) opt in
explicitly via config.
"""
from __future__ import annotations

import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.bench.tasks import perplexity as P  # noqa: E402

# CI installs torch + glq[hub] only, so skip the module rather than fail collection.
transformers = pytest.importorskip("transformers")  # noqa: E402

# run() does `import glq.hf_integration`, which needs the REAL transformers package
# (transformers.quantizers.auto). Import it here, before any fixture swaps a fake
# module into sys.modules, so the later import is a cached no-op.
import glq.hf_integration  # noqa: E402,F401


class _CausalCfg:
    architectures = ["FakeForCausalLM"]
    model_type = "fake_causal"


class _MMCfg:
    """Multimodal wrapper whose text half is an ordinary causal transformer."""
    architectures = ["FakeForConditionalGeneration"]
    model_type = "fake_mm"


class _AudioOnlyCfg:
    architectures = ["FakeForAudioClassification"]
    model_type = "fake_audio"


VOCAB = 32


class _FakeModel:
    def __init__(self):
        self.dtype_used = None

    def train(self, mode):
        return self

    def __call__(self, chunk):
        b, t = chunk.shape
        return types.SimpleNamespace(logits=torch.zeros(b, t, VOCAB))


def _make_auto(mapping_types, recorder):
    class _Auto:
        _model_mapping = mapping_types

        @classmethod
        def from_pretrained(cls, path, **kw):
            recorder["cls"] = cls.__name__
            recorder["dtype"] = kw.get("dtype")
            recorder["trust_remote_code"] = kw.get("trust_remote_code")
            return _FakeModel()
    return _Auto


class _FakeTok:
    @staticmethod
    def from_pretrained(path, **kw):
        def _call(text, return_tensors=None):
            return types.SimpleNamespace(
                input_ids=torch.arange(1, 4097).unsqueeze(0) % VOCAB)
        return types.SimpleNamespace(__call__=_call, **{"__call__": _call})


@pytest.fixture
def env(monkeypatch):
    """Install a fake transformers + datasets so run() needs no network or GPU."""
    rec = {}

    def _install(cfg_obj, causal_types, itt_types):
        tmod = types.ModuleType("transformers")
        tmod.AutoConfig = types.SimpleNamespace(
            from_pretrained=staticmethod(lambda p, **kw: cfg_obj))
        tmod.AutoModelForCausalLM = _make_auto(causal_types, rec)
        tmod.AutoModelForCausalLM.__name__ = "AutoModelForCausalLM"
        tmod.AutoModelForImageTextToText = _make_auto(itt_types, rec)
        tmod.AutoModelForImageTextToText.__name__ = "AutoModelForImageTextToText"

        class _CpuIds:
            """run() calls .to("cuda"); this keeps the tensor on CPU so the test needs
            no GPU, while everything downstream sees an ordinary tensor."""
            def __init__(self, t):
                self._t = t

            def to(self, device):
                return self._t

        class _Tok:
            @staticmethod
            def from_pretrained(p, **kw):
                class T:
                    def __call__(self, text, return_tensors=None):
                        return types.SimpleNamespace(
                            input_ids=_CpuIds(torch.arange(4096).unsqueeze(0) % VOCAB))
                return T()
        tmod.AutoTokenizer = _Tok
        monkeypatch.setitem(sys.modules, "transformers", tmod)

        dmod = types.ModuleType("datasets")
        dmod.load_dataset = lambda *a, **k: {"text": ["x"]}
        monkeypatch.setitem(sys.modules, "datasets", dmod)

        monkeypatch.setattr(P, "_load_mem_gib", lambda: 0.0)
        return rec
    return _install


def _ctx():
    return types.SimpleNamespace(model="fake/model", standalone_serving=None)


def test_multimodal_with_text_tower_is_measured_not_skipped(env):
    """A ConditionalGeneration arch that AutoModelForImageTextToText can load must
    produce a PPL, not TaskUnsupported. This is the Muse-Glimmer case."""
    cfg = _MMCfg()
    rec = env(cfg, causal_types=[_CausalCfg], itt_types=[_MMCfg])
    res, _ = P.run(_ctx(), {"seqlen": 512, "max_chunks": 2})
    assert res.metric == "perplexity"
    assert res.value > 0
    assert rec["cls"] == "AutoModelForImageTextToText"


def test_plain_causal_still_uses_causal_class(env):
    """Regression: the ordinary path must not start routing through the multimodal class."""
    cfg = _CausalCfg()
    rec = env(cfg, causal_types=[_CausalCfg], itt_types=[_MMCfg])
    P.run(_ctx(), {"seqlen": 512, "max_chunks": 2})
    assert rec["cls"] == "AutoModelForCausalLM"


def test_arch_with_no_text_tower_is_still_refused(env):
    """The guard must survive for models with no causal LM head at all — otherwise the
    failure moves from a clean 'skipped' to an obscure load error."""
    cfg = _AudioOnlyCfg()
    env(cfg, causal_types=[_CausalCfg], itt_types=[_MMCfg])
    with pytest.raises(P.TaskUnsupported, match="no causal text"):
        P.run(_ctx(), {"seqlen": 512, "max_chunks": 2})


def test_dtype_defaults_to_float16_for_comparability(env):
    """Every stored wikitext2_ppl record was measured in float16. If the default moves,
    new numbers stop being comparable with the existing series."""
    cfg = _CausalCfg()
    rec = env(cfg, causal_types=[_CausalCfg], itt_types=[_MMCfg])
    res, _ = P.run(_ctx(), {"seqlen": 512, "max_chunks": 2})
    assert rec["dtype"] is torch.float16
    assert res.config["dtype"] == "float16"


def test_dtype_is_configurable_and_recorded(env):
    """bf16-native models (logit softcapping, big outliers) overflow in fp16, so the
    dtype must be selectable — and recorded, since it scopes the number."""
    cfg = _MMCfg()
    rec = env(cfg, causal_types=[_CausalCfg], itt_types=[_MMCfg])
    res, _ = P.run(_ctx(), {"seqlen": 512, "max_chunks": 2, "dtype": "bfloat16"})
    assert rec["dtype"] is torch.bfloat16
    assert res.config["dtype"] == "bfloat16"
