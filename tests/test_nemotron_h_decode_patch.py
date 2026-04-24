"""Unit tests for the NemotronH decode-cache auto-patch.

We don't actually load NemotronH (it's huge and trust_remote_code), so we
build a synthetic stub module that mimics the surface of the real one and
verify _patch_nemotron_h_decode_cache rewires it correctly.
"""
import sys
import types

import pytest
import torch
import torch.nn as nn

transformers = pytest.importorskip("transformers")

from glq.hf_integration import _patch_nemotron_h_decode_cache


def _make_stub_nemotron_h_module():
    """Build a fake `transformers_modules.fake_nemotron_h.modeling_nemotron_h`
    module exposing the bare minimum surface: HybridMambaAttentionDynamicCache
    plus a tiny ForCausalLM that replicates the cache-routing bug."""
    mod = types.ModuleType("transformers_modules.fake_nemotron_h.modeling_nemotron_h")

    class HybridMambaAttentionDynamicCache:
        def __init__(self, config, batch_size, dtype=torch.float16, device=None):
            self.conv_states = [
                torch.zeros(batch_size, 4, 2, device=device, dtype=dtype)
                for _ in range(config.num_hidden_layers)
            ]
            self.ssm_states = [
                torch.zeros(batch_size, 4, 2, device=device, dtype=dtype)
                for _ in range(config.num_hidden_layers)
            ]

        def update_conv_state(self, layer_idx, new_conv_state, cache_init=False):
            # Original buggy version: dereferences .device on a list.
            if cache_init:
                self.conv_states[layer_idx] = new_conv_state.to(self.conv_states.device)
            else:
                self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(
                    self.conv_states.device
                )
            return self.conv_states[layer_idx]

        def update_ssm_state(self, layer_idx, new_ssm_state):
            self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states.device)
            return self.ssm_states[layer_idx]

    mod.HybridMambaAttentionDynamicCache = HybridMambaAttentionDynamicCache
    return mod


class _FakeConfig:
    def __init__(self):
        self.num_hidden_layers = 2
        self.conv_kernel = 4


class _FakeModel(nn.Module):
    """Stand-in for NemotronHForCausalLM exposing the buggy interface."""

    def __init__(self, mod_name):
        super().__init__()
        self._mod_name = mod_name
        self.linear = nn.Linear(4, 4)
        self.config = _FakeConfig()

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        return {"input_ids": input_ids, "past_key_values": past_key_values}

    def forward(self, *args, **kwargs):
        # Returns a ModelOutput-like with cache_params (no past_key_values).
        from transformers.modeling_outputs import ModelOutput

        class Out(ModelOutput):
            logits: torch.Tensor = None
            cache_params: object = None

        return Out(
            logits=self.linear(torch.zeros(1, 4)),
            cache_params=kwargs.get("cache_params"),
        )

    @property
    def __class__(self):
        # type(self).__module__ must point at the stub module so the patch
        # can find HybridMambaAttentionDynamicCache via sys.modules.
        return _registered_class


_registered_class = None


@pytest.fixture
def stub_module():
    mod = _make_stub_nemotron_h_module()
    sys.modules[mod.__name__] = mod

    global _registered_class

    class _StubModelCls(nn.Module):
        pass

    _StubModelCls.__module__ = mod.__name__
    _registered_class = _StubModelCls
    yield mod
    sys.modules.pop(mod.__name__, None)


def test_skips_non_nemotron_models():
    """Patch is a no-op on models that aren't NemotronH."""
    model = nn.Linear(2, 2)
    _patch_nemotron_h_decode_cache(model)
    assert not hasattr(type(model), "_glq_fwd_patched")


def test_cache_class_gains_conv_kernel_size_and_device(stub_module):
    """Bugs 3+4: cache.__init__ adds conv_kernel_size and a .device-aware list."""

    class _StubModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _FakeConfig()

            def _prep(*a, **k):
                return {"past_key_values": None, "input_ids": None}

            self.prepare_inputs_for_generation = _prep

        def forward(self, *args, **kwargs):
            return type("Out", (), {"cache_params": None, "past_key_values": None})()

    _StubModel.__module__ = stub_module.__name__
    model = _StubModel()
    _patch_nemotron_h_decode_cache(model)

    HybridCache = stub_module.HybridMambaAttentionDynamicCache
    assert HybridCache._glq_patched is True

    cache = HybridCache(_FakeConfig(), batch_size=1, dtype=torch.float32)
    assert hasattr(cache, "conv_kernel_size")
    assert cache.conv_kernel_size == 4
    # conv_states/ssm_states must answer .device without crashing
    assert cache.conv_states.device.type in ("cpu", "cuda", "meta")
    assert cache.ssm_states.device.type in ("cpu", "cuda", "meta")


def test_prepare_inputs_renames_past_key_values_to_cache_params(stub_module):
    """Bug 1: dict key gets renamed so the body's forward sees the cache."""

    class _StubModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _FakeConfig()

            def _prep(*a, **k):
                return {"past_key_values": "<cache>", "input_ids": [1, 2, 3]}

            self.prepare_inputs_for_generation = _prep

        def forward(self, *args, **kwargs):
            return None

    _StubModel.__module__ = stub_module.__name__
    model = _StubModel()
    _patch_nemotron_h_decode_cache(model)

    out = model.prepare_inputs_for_generation(input_ids=[1])
    assert "cache_params" in out
    assert "past_key_values" not in out
    assert out["cache_params"] == "<cache>"


def test_idempotent(stub_module):
    """Calling the patcher twice is safe — flags prevent double-wrapping."""

    class _StubModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _FakeConfig()

            def _prep(*a, **k):
                return {"past_key_values": None, "input_ids": None}

            self.prepare_inputs_for_generation = _prep

        def forward(self, *args, **kwargs):
            return None

    _StubModel.__module__ = stub_module.__name__
    model = _StubModel()
    _patch_nemotron_h_decode_cache(model)
    first_prep = model.prepare_inputs_for_generation
    _patch_nemotron_h_decode_cache(model)
    # Second call must not double-wrap; the bound method object stays the same.
    assert model.prepare_inputs_for_generation is first_prep
