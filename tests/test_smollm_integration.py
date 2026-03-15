"""Integration test: quantize SmolLM2-135M (2 layers) on CPU and verify output."""

import json
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")
safetensors = pytest.importorskip("safetensors")

from glq.quantize_model import quantize, get_decoder_layers


@pytest.fixture(scope="class")
def _monkeypatch_class():
    """Class-scoped monkeypatch (pytest's builtin is function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="class")
def quantize_output(tmp_path_factory, _monkeypatch_class):
    """Quantize SmolLM2-135M (first 2 layers only) and return (output_dir, avg_sqnr)."""
    _orig = get_decoder_layers

    def _limited(text_model, profile=None):
        return _orig(text_model, profile)[:2]

    _monkeypatch_class.setattr("glq.quantize_model.get_decoder_layers", _limited)

    output_dir = str(tmp_path_factory.mktemp("smollm_glq"))
    avg_sqnr = quantize(
        model_name="HuggingFaceTB/SmolLM2-135M",
        output_dir=output_dir,
        bpw=2,
        nsamples=1,
        seqlen=128,
        device="cpu",
        workers=1,
    )
    return output_dir, avg_sqnr


@pytest.mark.slow
class TestSmolLM2Quantize:
    def test_returns_positive_sqnr(self, quantize_output):
        _, avg_sqnr = quantize_output
        assert avg_sqnr > 0, f"avg_sqnr should be positive, got {avg_sqnr}"

    def test_output_model_safetensors(self, quantize_output):
        output_dir, _ = quantize_output
        model_path = os.path.join(output_dir, "model.safetensors")
        assert os.path.exists(model_path)
        assert os.path.getsize(model_path) > 0

        from safetensors import safe_open
        with safe_open(model_path, framework="pt") as f:
            keys = list(f.keys())

        # Quantized artifacts for first 2 layers
        assert any("Qidxs" in k for k in keys)
        assert any("SU" in k for k in keys)
        assert any("SV" in k for k in keys)
        assert any("Wscale" in k for k in keys)

        # Non-quantized parameters (embedding, lm_head)
        assert any("embed_tokens" in k for k in keys)
        assert any("lm_head" in k or "norm" in k for k in keys)

    def test_output_config_json(self, quantize_output):
        output_dir, _ = quantize_output
        config_path = os.path.join(output_dir, "config.json")
        assert os.path.exists(config_path)

        with open(config_path) as f:
            config = json.load(f)

        qc = config["quantization_config"]
        assert qc["quant_method"] == "glq"
        assert qc["codebook"] == "e8_shell"
        assert qc["codesz"] == 8
        assert qc["bpw"] == 2

    def test_output_quantize_config_json(self, quantize_output):
        output_dir, avg_sqnr = quantize_output
        qc_path = os.path.join(output_dir, "quantize_config.json")
        assert os.path.exists(qc_path)

        with open(qc_path) as f:
            meta = json.load(f)

        assert meta["quant_method"] == "glq"
        assert meta["source_model"] == "HuggingFaceTB/SmolLM2-135M"
        assert meta["bpw"] == 2
        assert meta["avg_sqnr_db"] == round(avg_sqnr, 2)
        # 2 layers × 7 linear sublayers each = 14
        assert meta["n_quantized_layers"] == 14
        assert meta["nsamples"] == 1
        assert meta["seqlen"] == 128

    def test_output_codebook(self, quantize_output):
        output_dir, _ = quantize_output
        cb_path = os.path.join(output_dir, "e8_codebook.pt")
        assert os.path.exists(cb_path)

        from glq.codebook import E8ShellCodebook
        cb = E8ShellCodebook.load(cb_path, device="cpu")
        assert cb.codebook.shape == (65536, 8)

    def test_output_tokenizer(self, quantize_output):
        output_dir, _ = quantize_output
        tok_config = os.path.join(output_dir, "tokenizer_config.json")
        assert os.path.exists(tok_config)
