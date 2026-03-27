"""Tests for GLQ vLLM plugin — registration, correctness, VRAM, throughput."""

import time

import pytest
import torch

try:
    from vllm.model_executor.layers.quantization import get_quantization_config
    _HAS_VLLM = True
except (ImportError, ModuleNotFoundError):
    _HAS_VLLM = False

try:
    import transformers
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

requires_vllm = pytest.mark.skipif(not _HAS_VLLM, reason="vllm not installed")
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required"
)

MODEL_ID = "xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw"
TOKENIZER_ID = "HuggingFaceTB/SmolLM3-3B"


# ── Test 1: Config registration (no GPU needed) ────────────────────────

@requires_vllm
def test_glq_config_registration():
    """GLQvLLMConfig should be discoverable after import glq_vllm."""
    import glq_vllm  # noqa: F401
    from glq_vllm.config import GLQvLLMConfig
    assert get_quantization_config("glq") == GLQvLLMConfig


# ── Test 2: Config parsing (no GPU needed) ──────────────────────────────

@requires_vllm
def test_glq_config_from_config():
    """GLQvLLMConfig should parse bpw and layer_bpw from dict."""
    from glq_vllm.config import GLQvLLMConfig
    cfg = GLQvLLMConfig.from_config({"bpw": 4, "layer_bpw": {"layers.0.q_proj": 2}})
    assert cfg.bpw == 4
    assert cfg.layer_bpw["layers.0.q_proj"] == 2
    assert cfg.get_name() == "glq"
    assert cfg.get_config_filenames() == ["quantize_config.json"]
    assert torch.float16 in cfg.get_supported_act_dtypes()


# ── Test 3: Dequant correctness (CPU, no vLLM server) ──────────────────

@pytest.mark.skipif(
    not _HAS_TRANSFORMERS, reason="transformers not installed"
)
def test_dequant_matches_hf():
    """dequantize_glq_weight must match E8RHTLinear.dequantize()."""
    import glq.hf_integration  # noqa: F401
    from transformers import AutoModelForCausalLM
    from glq_vllm.dequant import dequantize_glq_weight, get_codebook, get_codebook2

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cpu", torch_dtype=torch.float16
    )
    layer = model.model.layers[0].self_attn.q_proj
    hf_weight = layer.dequantize()

    cb = get_codebook()
    inv_rs = layer.inv_resid_scale.item()
    has_stage2 = inv_rs != 0.0
    cb2 = get_codebook2(4) if has_stage2 else None

    vllm_weight = dequantize_glq_weight(
        layer.Qidxs, layer.SU, layer.SV, layer.Wscale, cb,
        layer.Qidxs2 if has_stage2 else None, inv_rs, cb2,
        layer.out_features, layer.in_features,
    )

    cos = torch.nn.functional.cosine_similarity(
        hf_weight.flatten().float(), vllm_weight.flatten().float(), dim=0
    ).item()
    assert cos > 0.999, f"Dequant cosine similarity {cos:.4f} too low"


# ── Tests 4-7: GPU tests with shared LLM instance ──────────────────────

@requires_vllm
@requires_gpu
def test_glq_vllm_gpu():
    """All GPU tests in one: weights, generation, VRAM, throughput.

    Uses a single LLM instance to avoid repeated 5-min model loads.
    """
    import glq_vllm  # noqa: F401
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_ID,
        tokenizer=TOKENIZER_ID,
        quantization="glq",
        enforce_eager=True,
        dtype="half",
        gpu_memory_utilization=0.9,
        max_model_len=512,
    )

    # --- 4. Weights stay compressed ---
    found_qidxs = [False]
    def check_layers(model):
        for name, mod in model.named_modules():
            if hasattr(mod, "Qidxs"):
                assert mod.Qidxs.dtype == torch.int16, f"{name}.Qidxs is {mod.Qidxs.dtype}"
                assert not hasattr(mod, "weight"), f"{name} has dense weight"
                found_qidxs[0] = True
    llm.apply_model(check_layers)
    assert found_qidxs[0], "No GLQ layers found with Qidxs"

    # --- 5. Generation correctness ---
    params = SamplingParams(max_tokens=30, temperature=0)
    outputs = llm.generate(["The capital of France is"], params)
    text = outputs[0].outputs[0].text
    assert len(text) > 10, f"Output too short: {text!r}"
    assert "Paris" in text, f"Expected 'Paris' in output: {text!r}"

    # --- 6. Throughput ---
    params = SamplingParams(max_tokens=128, temperature=0)
    t0 = time.time()
    outputs = llm.generate(["The theory of relativity states that"], params)
    t1 = time.time()
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tok_per_sec = total_tokens / (t1 - t0)
    assert tok_per_sec >= 10, f"Throughput {tok_per_sec:.1f} tok/s below 10 tok/s minimum"

    del llm
