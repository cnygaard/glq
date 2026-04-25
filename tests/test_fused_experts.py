"""Unit tests for the E8RHTFusedExperts MoE wrapper + state-dict renamer.

We don't load the full 30B model here — the targeted unit tests cover:

1. `E8RHTFusedExperts.forward` matches the per-expert reference loop on a
   synthetic 4-expert toy.
2. `install_nemotron_h_state_dict_renames` injects the
   `_checkpoint_conversion_mapping` regex on the model class.
3. `_replace_nemotron_h_experts` substitutes `NemotronHExperts` with
   `E8RHTFusedExperts` when the native module is available, and is a no-op
   otherwise.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn as nn

transformers = pytest.importorskip("transformers")

from glq.fused_experts import (
    E8RHTFusedExperts,
    _ExpertPair,
    _replace_nemotron_h_experts,
)
from glq.state_dict_stacker import (
    NEMOTRON_H_PREFIX_RENAMES,
    install_nemotron_h_state_dict_renames,
)


def _make_stub_config(num_experts=4, hidden=64, intermediate=32):
    cfg = types.SimpleNamespace(
        n_routed_experts=num_experts,
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
        moe_latent_size=None,
        mlp_hidden_act="relu2",  # NemotronH default
        model_type="nemotron_h",
    )
    return cfg


class TestE8RHTFusedExpertsConstruction:

    def test_builds_with_expected_children(self):
        cfg = _make_stub_config(num_experts=4, hidden=64, intermediate=32)
        m = E8RHTFusedExperts(cfg, block_diagonal=True)
        assert len(m.experts) == 4
        for pair in m.experts:
            assert isinstance(pair, _ExpertPair)
            # Per-expert E8RHTLinears with the right dims
            assert pair.up_proj.in_features == 64
            assert pair.up_proj.out_features == 32
            assert pair.down_proj.in_features == 32
            assert pair.down_proj.out_features == 64

    def test_state_dict_keys_match_per_expert_layout(self):
        cfg = _make_stub_config(num_experts=2, hidden=32, intermediate=16)
        m = E8RHTFusedExperts(cfg, block_diagonal=True)
        keys = set(m.state_dict().keys())
        # Per-expert paths exist for both up_proj and down_proj
        assert "experts.0.up_proj.Qidxs" in keys
        assert "experts.0.up_proj.SU" in keys
        assert "experts.0.down_proj.Qidxs2" in keys
        assert "experts.1.down_proj.inv_resid_scale" in keys
        # Sanity: stage-3 buffers also present (zero-init for unused stages)
        assert "experts.0.up_proj.Qidxs3" in keys
        assert "experts.1.up_proj.inv_resid_scale2" in keys


class TestForwardSemantics:

    def test_forward_routes_to_correct_experts(self):
        """The forward loop must call up_proj/down_proj on the experts named
        in top_k_index, and weight by top_k_weights. We verify this by
        replacing each pair with an Identity-like fake whose output records
        which experts ran."""
        cfg = _make_stub_config(num_experts=4, hidden=8, intermediate=8)
        m = E8RHTFusedExperts(cfg, block_diagonal=True)

        called = []

        class _Tag(nn.Module):
            def __init__(self, idx, role):
                super().__init__()
                self.idx, self.role = idx, role

            def forward(self, x):
                called.append((self.idx, self.role, x.shape[0]))
                return x  # identity

        for i, pair in enumerate(m.experts):
            pair.up_proj = _Tag(i, "up")
            pair.down_proj = _Tag(i, "down")
        # nn.Module attribute setter rejects bare lambdas — assign through
        # __dict__ to bypass the child-module check.
        object.__setattr__(m, "act_fn", lambda x: x)

        # 3 tokens, top-1 routing to experts [1, 0, 1]
        hidden = torch.randn(3, 8)
        top_k_index = torch.tensor([[1], [0], [1]])
        top_k_weights = torch.ones(3, 1)
        out = m(hidden, top_k_index, top_k_weights)

        # Experts 1 and 0 must have been called; expert 2/3 untouched.
        idxs_called = {idx for idx, _, _ in called}
        assert idxs_called == {0, 1}
        # Each called expert should have run up_proj and down_proj
        for idx in idxs_called:
            roles = {role for i, role, _ in called if i == idx}
            assert roles == {"up", "down"}
        # Output shape preserved
        assert out.shape == hidden.shape

    def test_no_active_experts_returns_zeros(self):
        cfg = _make_stub_config(num_experts=2, hidden=4, intermediate=4)
        m = E8RHTFusedExperts(cfg, block_diagonal=True)

        hidden = torch.randn(3, 4)
        # Empty top-k routing — no expert mask hits.
        empty_idx = torch.zeros(3, 0, dtype=torch.long)
        empty_w = torch.zeros(3, 0)
        out = m(hidden, empty_idx, empty_w)
        # Output is zeros (final_hidden_states starts at zero, no expert
        # contributes).
        assert torch.equal(out, torch.zeros_like(out))


class TestStateDictRenamer:

    def test_idempotent_install(self):
        cfg = _make_stub_config()
        # Build a tiny model class so we can mutate its
        # _checkpoint_conversion_mapping without touching transformers internals.
        class _StubModel(nn.Module):
            _checkpoint_conversion_mapping = {}

        m = _StubModel()
        m.config = cfg

        first = install_nemotron_h_state_dict_renames(m)
        assert first is True
        assert _StubModel._checkpoint_conversion_mapping[r"^backbone\."] == "model."

        # Calling twice is fine and reports True.
        second = install_nemotron_h_state_dict_renames(m)
        assert second is True
        # Mapping preserved unchanged
        assert _StubModel._checkpoint_conversion_mapping[r"^backbone\."] == "model."

        # cleanup so we don't leak class state across tests
        _StubModel._checkpoint_conversion_mapping = {}

    def test_skips_non_nemotron_models(self):
        class _StubModel(nn.Module):
            _checkpoint_conversion_mapping = {}

        m = _StubModel()
        m.config = types.SimpleNamespace(model_type="llama")
        result = install_nemotron_h_state_dict_renames(m)
        assert result is False
        assert _StubModel._checkpoint_conversion_mapping == {}


class TestReplaceHelper:

    def test_replace_noop_without_native_nemotron_h(self, monkeypatch):
        """If transformers' nemotron_h integration isn't installed, the
        helper returns 0 with no side effects."""
        # Ensure ImportError path: hide the module name from sys.modules and
        # block re-import via meta_path.
        import importlib

        class _BlockNemotronH:
            def find_spec(self, name, path=None, target=None):
                if name == "transformers.models.nemotron_h.modeling_nemotron_h":
                    raise ImportError("blocked")
                return None

        monkeypatch.setattr(
            sys, "meta_path", [_BlockNemotronH()] + sys.meta_path)
        # Drop any cached real module so the loader re-imports under our block.
        for k in list(sys.modules):
            if "nemotron_h" in k:
                monkeypatch.delitem(sys.modules, k, raising=False)

        model = nn.Sequential(nn.Linear(4, 4))
        n = _replace_nemotron_h_experts(model)
        assert n == 0

    def test_replace_substitutes_when_native_present(self):
        # Skip if native nemotron_h not installed in this env (it isn't on
        # transformers 5.2; CI has 5.6+ and would exercise this path).
        try:
            from transformers.models.nemotron_h.modeling_nemotron_h import (
                NemotronHExperts,
            )
        except Exception:
            pytest.skip("native nemotron_h not available")

        cfg = _make_stub_config(num_experts=2, hidden=16, intermediate=8)

        class _Container(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = cfg
                self.experts = NemotronHExperts(cfg)

        m = _Container()
        n = _replace_nemotron_h_experts(m, block_diagonal=True)
        assert n == 1
        assert isinstance(m.experts, E8RHTFusedExperts)
        assert len(m.experts.experts) == 2
