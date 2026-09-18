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
    rest of the suite (it mutates a module-level function in a third-party package)."""
    original = ple_layer._get_ple_embedding_quant_method
    _qwen4exp_ple.install()
    yield
    ple_layer._get_ple_embedding_quant_method = original


# ---- the routing itself -------------------------------------------------------------

def test_a_glq_config_now_yields_the_glq_embedding_method():
    """The whole point. Without this the table is built dense and OOMs at 95.37 GiB."""
    method = ple_layer._get_ple_embedding_quant_method(_glq_config(), VLLM_PREFIX)
    assert isinstance(method, GLQEmbeddingMethod)


def test_the_method_carries_the_tables_own_codebook_and_rate():
    """``ple_codebook``/``ple_bpw`` describe the TABLE, not the run. A 3 bpw trellis
    checkpoint carries a 4 bpw table, and create_weights registers buffers sized from
    these before any tensor key is visible — get them wrong and the only symptom is a
    shape assertion deep inside vLLM's loader."""
    method = ple_layer._get_ple_embedding_quant_method(_glq_config(), VLLM_PREFIX)
    assert method.codebook == "trellis"
    assert method.bpw == 4, "took the run's 3 bpw instead of the table's 4"
    assert method.variant == "3inst"


def test_the_checkpoint_form_prefix_also_resolves():
    """Belt and braces: if a future vLLM names the module in checkpoint form, the
    lookup must still land rather than silently returning None."""
    method = ple_layer._get_ple_embedding_quant_method(_glq_config(), CKPT_KEY)
    assert isinstance(method, GLQEmbeddingMethod)


# ---- what it must NOT do ------------------------------------------------------------

def test_a_table_absent_from_layer_bpw_is_left_alone():
    """A GLQ checkpoint that left its PLE in bf16 must fall through to vLLM's own
    decision, not acquire a GLQ method that would then find no buffers to load."""
    cfg = _glq_config(layer_bpw={"model.language_model.layers.0.mlp.gate_proj": 3})
    assert ple_layer._get_ple_embedding_quant_method(cfg, VLLM_PREFIX) is None


def test_a_non_glq_config_still_reaches_the_original():
    """We wrap, we do not replace. An FP8 checkpoint must keep working, and a config we
    do not recognise must get vLLM's answer rather than ours."""
    assert ple_layer._get_ple_embedding_quant_method(None, VLLM_PREFIX) is None
    assert ple_layer._get_ple_embedding_quant_method(object(), VLLM_PREFIX) is None


def test_a_shell_ple_is_routed_as_shell():
    """Absent ``ple_codebook`` means shell — the gemma-4 default. Routing it as trellis
    would register the wrong buffers entirely."""
    cfg = _glq_config(ple_codebook=None, ple_bpw=None, codebook="e8_shell",
                      layer_bpw={CKPT_KEY: 4})
    method = ple_layer._get_ple_embedding_quant_method(cfg, VLLM_PREFIX)
    assert isinstance(method, GLQEmbeddingMethod)
    assert method.codebook == "shell"


# ---- the patch mechanics ------------------------------------------------------------

def test_installing_twice_does_not_nest_the_wrapper():
    """``register()`` runs in every vLLM process and can be re-entered. A second wrap
    would still work but would make the delegation chain grow without bound."""
    first = ple_layer._get_ple_embedding_quant_method
    _qwen4exp_ple.install()
    assert ple_layer._get_ple_embedding_quant_method is first


def test_the_wrapper_is_identifiable_and_keeps_the_original():
    """A patch of a private third-party function has to be greppable when a future vLLM
    release renames or removes it."""
    hook = ple_layer._get_ple_embedding_quant_method
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
