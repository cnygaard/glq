"""Routing GLQ into Qwen4Exp's per-layer-embedding table under vLLM.

Every other quantized layer reaches us through
``QuantizationConfig.get_quant_method``. Qwen4Exp's n-gram table does not: vLLM builds
it with an **explicitly passed** method,

    self.ngram_embedding = PLEVocabParallelEmbedding(
        padded_vocab_size, self.head_dim, ...,
        quant_method=_get_ple_embedding_quant_method(quant_config, f"{prefix}.ngram_embedding"))

and that helper accepts exactly one quantization::

    \"\"\"Select global-scale FP8 only for quantized PLE checkpoint shards.\"\"\"
    if not isinstance(quant_config, Fp8Config):
        return None

So GLQ's config is never consulted, ``VocabParallelEmbedding.__init__`` falls back to
``UnquantizedEmbeddingMethod``, and the table is built **dense in bf16**. For
Qwen3.8-Flash-Next that is a single ``torch.empty(320_001_536, 160)`` = **95.37 GiB**,
which OOMs a 94.97 GiB card at load with a traceback that never mentions GLQ.

These pin the wrapper that fixes it. They assert the *mechanism* — which method class
comes back — because "the model loaded" is exactly what a silent fall-through to dense
bf16 also looks like, right up until the allocator gives out.

vLLM 0.30.0 MOVED this dispatch point: the helper above was deleted and replaced by
``Qwen4ExpPLEEmbeddingMethod.from_quant_config`` in a new ``ngram_embedding`` module, and
the decline path changed from "return None so vLLM falls back" to "raise
NotImplementedError for any non-Fp8Config". So these tests resolve the dispatch point at
import time rather than naming one — hardcoding either version is the exact brittleness
that let the rename go unnoticed.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# vLLM is absent from the torch-only CI environment, and the Qwen4Exp PLE module only
# exists on builds new enough to ship that architecture.
pytest.importorskip("vllm", reason="vLLM not installed")
ple_layer = pytest.importorskip(
    "vllm.models.qwen4_exp.nvidia.ple_layer",
    reason="this vLLM build has no Qwen4Exp PLE layer")

from glq_vllm import _qwen4exp_ple  # noqa: E402
from glq_vllm.config import GLQvLLMConfig  # noqa: E402
from glq_vllm.embedding_method import GLQEmbeddingMethod  # noqa: E402

# ---- resolve whichever dispatch point this vLLM exposes ------------------------------
LEGACY = hasattr(ple_layer, "_get_ple_embedding_quant_method")
if LEGACY:
    API = "<=0.29.0 ple_layer._get_ple_embedding_quant_method"

    def _call(*a, **k):
        return ple_layer._get_ple_embedding_quant_method(*a, **k)

    def _save():
        return ple_layer._get_ple_embedding_quant_method

    def _restore(fn):
        ple_layer._get_ple_embedding_quant_method = fn
else:
    _ne = pytest.importorskip(
        "vllm.models.qwen4_exp.nvidia.ngram_embedding",
        reason="no legacy helper and no ngram_embedding module — unknown vLLM layout")
    _CLS = getattr(_ne, "Qwen4ExpPLEEmbeddingMethod", None)
    if _CLS is None or not hasattr(_CLS, "from_quant_config"):
        pytest.skip("neither PLE dispatch point present", allow_module_level=True)
    API = ">=0.30.0 Qwen4ExpPLEEmbeddingMethod.from_quant_config"

    def _call(*a, **k):
        return _CLS.from_quant_config(*a, **k)

    def _save():
        return _CLS.from_quant_config

    def _restore(fn):
        _CLS.from_quant_config = staticmethod(fn)

#: The prefix vLLM actually passes. The chain is
#: ``language_model`` -> ``.model`` -> ``.layers.{i}`` -> ``.ple`` -> ``.ple_embedding``
#: -> ``.ngram_embedding``, and ``i`` is the layer whose ``layer_idx + 1`` is in
#: ``ple_layer_ids`` (``[2]`` for Qwen3.8-Flash-Next, so layer 1). It is NOT the
#: checkpoint form: ``_lookup_bpw`` has to translate ``language_model.model.*`` back to
#: ``model.language_model.*`` for the lookup to hit.
VLLM_PREFIX = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
CKPT_KEY = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"


def _glq_config(**kw):
    """A config shaped like a real trellis-PLE checkpoint's quantization_config."""
    # trellis_layout="kernel" is not optional: GLQvLLMConfig refuses a 3inst checkpoint
    # without it (pre-kernel NATURAL layout would be silently scrambled).
    base = dict(bpw=3, codebook="trellis", variant="3inst", trellis_layout="kernel",
                layer_bpw={CKPT_KEY: 4}, ple_codebook="trellis", ple_bpw=4)
    base.update(kw)
    return GLQvLLMConfig(**base)


@pytest.fixture(autouse=True)
def _installed():
    """Install once per test and restore, so a failure cannot leak the patch into the
    rest of the suite (it mutates a third-party package)."""
    original = _save()
    _qwen4exp_ple.install()
    yield
    _restore(original)


def test_the_suite_is_pinned_to_a_known_dispatch_point():
    """Names which API is under test, so a green run on an unknown vLLM cannot be mistaken
    for coverage. If a future release moves it again, the module-level resolution skips and
    this never runs."""
    assert API.startswith(("<=0.29.0", ">=0.30.0")), API


# ---- the routing itself -------------------------------------------------------------

def test_a_glq_config_now_yields_the_glq_embedding_method():
    """The whole point. Without this the table is built dense and OOMs at 95.37 GiB."""
    assert isinstance(_call(_glq_config(), VLLM_PREFIX), GLQEmbeddingMethod)


def test_the_method_carries_the_tables_own_codebook_and_rate():
    """``ple_codebook``/``ple_bpw`` describe the TABLE, not the run. A 3 bpw trellis
    checkpoint carries a 4 bpw table, and create_weights registers buffers sized from
    these before any tensor key is visible — get them wrong and the only symptom is a
    shape assertion deep inside vLLM's loader."""
    method = _call(_glq_config(), VLLM_PREFIX)
    assert method.codebook == "trellis"
    assert method.bpw == 4, "took the run's 3 bpw instead of the table's 4"
    assert method.variant == "3inst"


def test_the_checkpoint_form_prefix_also_resolves():
    """Belt and braces: if a future vLLM names the module in checkpoint form, the
    lookup must still land rather than silently returning None."""
    assert isinstance(_call(_glq_config(), CKPT_KEY), GLQEmbeddingMethod)


def test_the_glq_method_satisfies_the_0_30_dequantize_contract():
    """0.30.0 SPLIT the lookup: ``embedding()`` returns raw rows and ``dequantize()``
    converts them. GLQ's ``embedding()`` already decodes to params_dtype, so dequantize is
    a pass-through — but it must EXIST, or the PLE forward dies with AttributeError on a
    path no unit test would otherwise reach."""
    import torch
    method = _call(_glq_config(), VLLM_PREFIX)
    rows = torch.zeros(4, 8, dtype=torch.bfloat16)
    assert method.dequantize(None, rows, torch.bfloat16) is rows, "should not copy"
    assert method.dequantize(None, rows, torch.float32).dtype == torch.float32


# ---- what it must NOT do ------------------------------------------------------------

def test_a_table_absent_from_layer_bpw_is_left_alone():
    """A GLQ checkpoint that left its PLE in bf16 must NOT acquire a GLQ method that would
    then find no buffers to load.

    The two vLLM versions express "not ours" differently — 0.29.0 returns None, 0.30.0
    returns its unquantized method — so assert on what actually matters: it is not a GLQ
    method, and it did not raise. On 0.30.0 delegating to the original here WOULD raise,
    which is why the hook answers with the unquantized method itself."""
    cfg = _glq_config(layer_bpw={"model.language_model.layers.0.mlp.gate_proj": 3})
    result = _call(cfg, VLLM_PREFIX)
    assert not isinstance(result, GLQEmbeddingMethod)
    if LEGACY:
        assert result is None


def test_a_non_glq_config_still_reaches_the_original():
    """We wrap, we do not replace. A config we do not recognise must get vLLM's answer —
    including, on 0.30.0, vLLM's own NotImplementedError rather than a GLQ method."""
    assert not isinstance(_call(None, VLLM_PREFIX), GLQEmbeddingMethod)
    if LEGACY:
        assert _call(object(), VLLM_PREFIX) is None
    else:
        with pytest.raises(NotImplementedError):
            _call(object(), VLLM_PREFIX)


def test_a_shell_ple_is_routed_as_shell():
    """Absent ``ple_codebook`` means shell — the gemma-4 default. Routing it as trellis
    would register the wrong buffers entirely."""
    cfg = _glq_config(ple_codebook=None, ple_bpw=None, codebook="e8_shell",
                      layer_bpw={CKPT_KEY: 4})
    method = _call(cfg, VLLM_PREFIX)
    assert isinstance(method, GLQEmbeddingMethod)
    assert method.codebook == "shell"


# ---- the patch mechanics ------------------------------------------------------------

def test_installing_twice_does_not_nest_the_wrapper():
    """``register()`` runs in every vLLM process and can be re-entered. A second wrap
    would still work but would make the delegation chain grow without bound."""
    first = _save()
    _qwen4exp_ple.install()
    assert _save() is first


def test_the_wrapper_is_identifiable_and_keeps_the_original():
    """A patch of a private third-party symbol has to be greppable when a future vLLM
    release renames or removes it."""
    hook = _save()
    assert getattr(hook, "_glq_wrapped", False) is True
    assert callable(getattr(hook, "_glq_original", None))


def test_install_is_a_noop_when_the_module_is_absent(monkeypatch):
    """vLLM builds without Qwen4Exp must import glq_vllm without raising — ``register()``
    calls this unconditionally in every process."""
    import builtins
    real_import = builtins.__import__

    def _no_qwen4exp(name, *args, **kw):
        if "qwen4_exp" in name:
            raise ImportError("no Qwen4Exp in this build")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_qwen4exp)
    _qwen4exp_ple.install()          # must not raise


def test_an_unknown_vllm_layout_warns_instead_of_silently_doing_nothing(monkeypatch):
    """The failure that motivated all of this: the 0.29.0 helper was renamed, install()
    returned silently, and the only symptom was an OOM (0.29.0) or a NotImplementedError
    (0.30.0) with nothing naming GLQ. A build with Qwen4Exp but no recognised dispatch
    point must now say so."""
    monkeypatch.setattr(_qwen4exp_ple, "_install_legacy", lambda _m: False)
    monkeypatch.setattr(_qwen4exp_ple, "_install_modern", lambda: False)
    with pytest.warns(RuntimeWarning, match="neither PLE quant-method dispatch point"):
        _qwen4exp_ple.install()
