"""Tests for HuggingFace Transformers integration."""

import pytest
import torch
import torch.nn as nn

# Skip all tests if transformers is not installed
transformers = pytest.importorskip("transformers")

from glq.hf_integration import GLQConfig, replace_with_glq_linear
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
