"""Pure-logic tests for the GLQ-MoE cudagraph capture-size cap
(glq_vllm/_cudagraph_cap.py). Loads the module directly from its file so the
test runs without vLLM/torch installed (the helper only imports `os`)."""
import importlib.util
import pathlib
import types

import pytest


def _load():
    p = pathlib.Path(__file__).resolve().parent.parent / "glq_vllm" / "_cudagraph_cap.py"
    spec = importlib.util.spec_from_file_location("_glq_cgcap", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CAP = _load()
NS = types.SimpleNamespace


def _vcfg(quant="glq", hf=None, sizes=None):
    cc = NS(cudagraph_capture_sizes=sizes,
            max_cudagraph_capture_size=(max(sizes) if sizes else None))
    return NS(model_config=NS(quantization=quant, hf_config=hf),
              compilation_config=cc)


def test_is_glq_moe_detects_expert_markers():
    assert CAP._is_glq_moe(_vcfg(hf=NS(num_experts=128)))
    assert CAP._is_glq_moe(_vcfg(hf=NS(num_experts_per_tok=8, num_experts=128)))
    # nested text_config (multimodal wrappers like Gemma4ForConditionalGeneration)
    assert CAP._is_glq_moe(_vcfg(hf=NS(text_config=NS(num_local_experts=8))))


def test_is_glq_moe_false_for_dense_and_non_glq():
    assert not CAP._is_glq_moe(_vcfg(hf=NS(hidden_size=2048)))        # dense GLQ
    assert not CAP._is_glq_moe(_vcfg(quant="awq", hf=NS(num_experts=128)))  # not glq
    assert not CAP._is_glq_moe(_vcfg(quant=None, hf=NS(num_experts=128)))


def test_cap_filters_sizes_above_256():
    c = _vcfg(hf=NS(num_experts=128),
              sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512])
    CAP._cap(c)
    assert c.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert c.compilation_config.max_cudagraph_capture_size == 256


def test_cap_noop_when_already_within_limit():
    c = _vcfg(hf=NS(num_experts=128), sizes=[1, 2, 4, 8, 16, 32])
    CAP._cap(c)
    assert c.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8, 16, 32]
    assert c.compilation_config.max_cudagraph_capture_size == 32


def test_cap_respects_env_override(monkeypatch):
    monkeypatch.setenv("GLQ_MOE_BD_MAX_TOKENS", "64")
    c = _vcfg(hf=NS(num_experts=128), sizes=[1, 32, 64, 128, 256])
    CAP._cap(c)
    assert c.compilation_config.cudagraph_capture_sizes == [1, 32, 64]
    assert c.compilation_config.max_cudagraph_capture_size == 64


def test_cap_handles_empty_or_eager():
    c = _vcfg(hf=NS(num_experts=128), sizes=None)   # enforce_eager → no sizes
    CAP._cap(c)  # must not raise
    assert c.compilation_config.cudagraph_capture_sizes is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
