"""Tests for HuggingFace Transformers integration."""

import logging

import pytest
import torch
import torch.nn as nn

# Skip all tests if transformers is not installed
transformers = pytest.importorskip("transformers")

from glq.hf_integration import GLQConfig, GLQQuantizer, replace_with_glq_linear
from glq.quantized_linear import E8RHTLinear


class TestGLQConfig:
    def test_defaults(self):
        cfg = GLQConfig()
        assert cfg.quant_method == "glq"
        assert cfg.codebook == "e8_shell"
        assert cfg.codesz == 8
        assert cfg.bpw == 2

    def test_custom_bpw(self):
        cfg = GLQConfig(bpw=4)
        assert cfg.bpw == 4

    def test_to_dict(self):
        cfg = GLQConfig(bpw=3)
        d = cfg.to_dict()
        assert d == {
            "quant_method": "glq",
            "codebook": "e8_shell",
            "codesz": 8,
            "bpw": 3,
        }


class TestReplaceWithGLQLinear:
    def test_replaces_linear(self):
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        replaced = replace_with_glq_linear(model)
        assert replaced is True
        assert isinstance(model[0], E8RHTLinear)
        assert isinstance(model[2], E8RHTLinear)

    def test_preserves_non_linear(self):
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
        )
        replace_with_glq_linear(model)
        assert isinstance(model[1], nn.ReLU)
        assert isinstance(model[2], nn.LayerNorm)

    def test_skips_lm_head(self):
        """Modules with 'lm_head' in name should not be replaced."""
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden = nn.Linear(64, 32)
                self.lm_head = nn.Linear(32, 1000)

        model = ToyModel()
        replace_with_glq_linear(model)
        assert isinstance(model.hidden, E8RHTLinear)
        assert isinstance(model.lm_head, nn.Linear)  # NOT replaced

    def test_preserves_dimensions(self):
        model = nn.Sequential(nn.Linear(48, 33))
        replace_with_glq_linear(model)
        layer = model[0]
        assert isinstance(layer, E8RHTLinear)
        assert layer.in_features == 48
        assert layer.out_features == 33

    def test_returns_false_when_nothing_replaced(self):
        model = nn.Sequential(nn.ReLU(), nn.LayerNorm(32))
        replaced = replace_with_glq_linear(model)
        assert replaced is False

    def test_nested_modules(self):
        """Replaces linear layers inside nested submodules."""
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.Linear(64, 64)
                self.mlp = nn.Linear(64, 256)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = Block()
                self.lm_head = nn.Linear(64, 1000)

        model = Model()
        replaced = replace_with_glq_linear(model)
        assert replaced is True
        assert isinstance(model.block.attn, E8RHTLinear)
        assert isinstance(model.block.mlp, E8RHTLinear)
        assert isinstance(model.lm_head, nn.Linear)  # skipped


class TestGLQConfigTrustRemoteCode:
    def test_to_dict_includes_trust_remote_code(self):
        cfg = GLQConfig(trust_remote_code=True)
        d = cfg.to_dict()
        assert d["trust_remote_code"] is True

    def test_to_dict_omits_trust_remote_code_when_false(self):
        cfg = GLQConfig(trust_remote_code=False)
        d = cfg.to_dict()
        assert "trust_remote_code" not in d


class TestGLQQuantizer:
    def test_init(self):
        cfg = GLQConfig()
        q = GLQQuantizer(cfg)
        assert q.quantization_config is cfg

    def test_validate_environment(self):
        q = GLQQuantizer(GLQConfig())
        q.validate_environment()  # should not raise

    def test_is_trainable(self):
        q = GLQQuantizer(GLQConfig())
        assert q.is_trainable is False

    def test_is_serializable(self):
        q = GLQQuantizer(GLQConfig())
        assert q.is_serializable() is True

    def test_process_before_loading_replaces(self):
        """_process_model_before_weight_loading replaces nn.Linear with E8RHTLinear."""
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        q = GLQQuantizer(GLQConfig())
        result = q._process_model_before_weight_loading(model)
        assert result is model
        assert isinstance(model[0], E8RHTLinear)

    def test_process_before_loading_warns_no_replace(self, caplog):
        """Warns when no nn.Linear modules found."""
        model = nn.Sequential(nn.ReLU(), nn.LayerNorm(32))
        q = GLQQuantizer(GLQConfig())
        with caplog.at_level(logging.WARNING):
            q._process_model_before_weight_loading(model)
        assert "no nn.Linear" in caplog.text and "modules found" in caplog.text

    def test_process_after_loading_2bpw(self):
        """Attaches codebook to E8RHTLinear modules (2bpw, no codebook2)."""
        model = nn.Sequential(nn.Linear(64, 32))
        q = GLQQuantizer(GLQConfig(bpw=2))
        q._process_model_before_weight_loading(model)
        q._process_model_after_weight_loading(model)
        layer = model[0]
        assert isinstance(layer, E8RHTLinear)
        assert layer.codebook is not None
        assert layer.codebook2 is None

    def test_process_after_loading_3bpw(self):
        """3bpw attaches a small (256-entry) secondary codebook."""
        model = nn.Sequential(nn.Linear(64, 32))
        q = GLQQuantizer(GLQConfig(bpw=3))
        q._process_model_before_weight_loading(model)
        q._process_model_after_weight_loading(model)
        layer = model[0]
        assert layer.codebook is not None
        assert layer.codebook2 is not None
        assert layer.codebook2.CODEBOOK_SIZE == 256

    def test_process_after_loading_4bpw(self):
        """4bpw uses the full codebook as codebook2."""
        model = nn.Sequential(nn.Linear(64, 32))
        q = GLQQuantizer(GLQConfig(bpw=4))
        q._process_model_before_weight_loading(model)
        q._process_model_after_weight_loading(model)
        layer = model[0]
        assert layer.codebook is not None
        assert layer.codebook2 is layer.codebook
