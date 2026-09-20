"""The patch that routes GatedDeltaNet decode to the fused CPU kernel.

The seam is a MODULE GLOBAL, not an instance attribute: Qwen4Exp decorates
``torch_recurrent_gated_delta_rule`` with ``use_kernel_func_from_hub_with_fallback`` and its
forward calls it by global name. These tests stand up a fake modeling module with that
shape, so the install/delegate/restore contract is pinned without needing a 78 GiB
checkpoint or a transformers that carries qwen4_exp.

What matters most here is the *declining*. The decorator resolves its implementation at
import time with no device check, so whatever is bound may already be a CUDA kernel from
`fla` or `causal-conv1d`. The shim therefore wraps rather than replaces, and every path it
does not handle must reach the original untouched.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch", reason="torch not installed")
import torch.nn as nn  # noqa: E402

from glq import gdn_cpu  # noqa: E402

B, H, K, V = 1, 4, 64, 64


def _fake_modeling_module(name="fake_qwen_mod"):
    """A stand-in for `transformers.models.qwen4_exp.modeling_qwen4_exp`."""
    mod = types.ModuleType(name)
    calls = []

    def torch_recurrent_gated_delta_rule(query, key, value, g=None, beta=None,
                                         initial_state=None, output_final_state=False,
                                         use_qk_l2norm_in_kernel=False, **kwargs):
        calls.append({"seq": query.shape[1], "cuda": query.is_cuda,
                      "state": initial_state is not None})
        out = torch.zeros(query.shape[0], query.shape[1], value.shape[2], value.shape[3],
                          dtype=query.dtype)
        return out, initial_state

    mod.torch_recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
    mod.calls = calls
    sys.modules[name] = mod

    class FakeGatedDeltaNet(nn.Module):
        pass

    FakeGatedDeltaNet.__module__ = name
    mod.FakeGatedDeltaNet = FakeGatedDeltaNet

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = FakeGatedDeltaNet()
            self.b = FakeGatedDeltaNet()

    return mod, FakeModel()


@pytest.fixture
def fake(request):
    mod, model = _fake_modeling_module(f"fake_qwen_{request.node.name}")
    yield mod, model
    sys.modules.pop(mod.__name__, None)


def _args(seq=1, state=True, dtype=torch.float32):
    q = torch.randn(B, seq, H, K, dtype=dtype)
    k = torch.randn(B, seq, H, K, dtype=dtype)
    v = torch.randn(B, seq, H, V, dtype=dtype)
    g = -torch.rand(B, seq, H)
    beta = torch.rand(B, seq, H)
    st = torch.zeros(B, H, K, V) if state else None
    return q, k, v, g, beta, st


# ---- install / uninstall -------------------------------------------------------------

def test_install_wraps_the_module_global_once_per_module(fake):
    mod, model = fake
    orig = mod.torch_recurrent_gated_delta_rule
    assert gdn_cpu.install(model) == 1, "two GatedDeltaNets share one module -> one patch"
    assert mod.torch_recurrent_gated_delta_rule is not orig
    assert gdn_cpu.install(model) == 0, "second install must be a no-op"
    assert gdn_cpu.uninstall(model) >= 1
    assert mod.torch_recurrent_gated_delta_rule is orig


def test_install_leaves_unrelated_modules_alone(fake):
    mod, _ = fake

    class Plain(nn.Module):
        pass

    assert gdn_cpu.install(Plain()) == 0


def test_a_module_without_the_global_is_skipped(fake):
    mod, model = fake
    del mod.torch_recurrent_gated_delta_rule
    assert gdn_cpu.install(model) == 0


# ---- delegation: every ineligible path must reach the original -----------------------

@pytest.mark.parametrize("case", ["flag_off", "prefill", "no_state"])
def test_ineligible_calls_reach_the_original(fake, monkeypatch, case):
    mod, model = fake
    gdn_cpu.install(model)
    monkeypatch.setenv("GLQ_CPU_GDN", "0" if case == "flag_off" else "1")
    q, k, v, g, beta, st = _args(seq=4 if case == "prefill" else 1,
                                 state=case != "no_state")
    mod.torch_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=st,
                                         output_final_state=True,
                                         use_qk_l2norm_in_kernel=True)
    assert len(mod.calls) == 1, f"{case}: the original was not called"


def test_an_eligible_call_does_not_reach_the_original(fake, monkeypatch):
    if not _kernel_available():
        pytest.skip("CPU extension without the GDN entry")
    mod, model = fake
    gdn_cpu.install(model)
    monkeypatch.setenv("GLQ_CPU_GDN", "1")
    q, k, v, g, beta, st = _args()
    out, state = mod.torch_recurrent_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=st, output_final_state=True,
        use_qk_l2norm_in_kernel=True)
    assert not mod.calls, "the reference ran even though the call was eligible"
    assert out.shape == (B, 1, H, V)
    assert state is not None


def _kernel_available() -> bool:
    from glq.inference_kernel_cpu import _try_load_cpu_ext
    if not _try_load_cpu_ext():
        return False
    from glq import inference_kernel_cpu as ikc
    return hasattr(ikc._glq_cpu, "glq_gdn_recurrent_step_cpu")


# ---- the numbers, against the real reference -----------------------------------------

def test_the_shim_matches_the_reference_it_replaced(fake, monkeypatch):
    """The fake module's stub is not a real rule, so compare against a transcription of the
    reference instead -- the same one tests/test_gdn_cpu_kernel.py pins to upstream."""
    if not _kernel_available():
        pytest.skip("CPU extension without the GDN entry")
    sys.path.insert(0, os.path.dirname(__file__))
    from test_gdn_cpu_kernel import oracle_step

    mod, model = fake
    gdn_cpu.install(model)
    monkeypatch.setenv("GLQ_CPU_GDN", "1")
    q, k, v, g, beta, st = _args()
    st = torch.randn(B, H, K, V) * 0.1

    out, state = mod.torch_recurrent_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=st.clone(), output_final_state=True,
        use_qk_l2norm_in_kernel=True)

    # reproduce the shim's prep with the reference's own l2norm formula
    qq = gdn_cpu._l2norm(q.transpose(1, 2).float().squeeze(2)) / (K ** 0.5)
    kk = gdn_cpu._l2norm(k.transpose(1, 2).float().squeeze(2))
    vv = v.transpose(1, 2).float().squeeze(2)
    want_out, want_state = oracle_step(st.clone(), qq, kk, vv,
                                       g.transpose(1, 2).squeeze(-1),
                                       beta.transpose(1, 2).squeeze(-1))
    assert torch.allclose(out.squeeze(1), want_out, atol=2e-5, rtol=1e-5)
    assert torch.allclose(state, want_state, atol=2e-5, rtol=1e-5)


def test_l2norm_uses_the_references_formula_not_F_normalize():
    """`F.normalize` clamps the denominator; the reference adds eps under the sqrt. They
    differ, and the delta rule is sensitive exactly there."""
    x = torch.tensor([[[1e-4, 0.0, 0.0, 0.0]]])
    assert not torch.allclose(gdn_cpu._l2norm(x),
                              torch.nn.functional.normalize(x, dim=-1, eps=1e-6))
