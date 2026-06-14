"""Unit tests for the GLQ MoE grouped-GEMM dispatch policy (GLQ_MOE_GROUPED).

Pure-logic tests for ``glq_vllm._dispatch._grouped_enabled`` — the tri-state gate
that picks the Stage-3 grouped-GEMM path vs the bit-exact block-diag matvec. No
torch / vLLM / GPU needed, so this locks the default-on-for-batched policy in CI.
"""
import pytest

from glq_vllm._dispatch import _grouped_enabled


# --- default ("auto"): grouped for batched MoE, block-diag for b1 ------------

@pytest.mark.parametrize("env", [None, "", "auto", "AUTO", "  auto  ", "bogus"])
def test_auto_keeps_b1_on_blockdiag(env):
    # b1 (num_tokens < grouped_min) must NOT take the grouped path.
    assert _grouped_enabled(env, grouped_min=2, num_tokens=1) is False


@pytest.mark.parametrize("env", [None, "auto"])
@pytest.mark.parametrize("nt", [2, 4, 8, 16, 32, 64, 256])
def test_auto_uses_grouped_for_batched(env, nt):
    assert _grouped_enabled(env, grouped_min=2, num_tokens=nt) is True


def test_auto_respects_custom_min():
    # With a higher floor, mid batches fall back to block-diag until the floor.
    assert _grouped_enabled("auto", grouped_min=8, num_tokens=4) is False
    assert _grouped_enabled("auto", grouped_min=8, num_tokens=8) is True


# --- "1"/on: force grouped everywhere (incl. b1) -----------------------------

@pytest.mark.parametrize("env", ["1", "on", "true", "yes", "TRUE", " On "])
@pytest.mark.parametrize("nt", [1, 2, 32])
def test_force_on(env, nt):
    assert _grouped_enabled(env, grouped_min=2, num_tokens=nt) is True


# --- "0"/off: never grouped (force block-diag; A/B isolation) -----------------

@pytest.mark.parametrize("env", ["0", "off", "false", "no", "FALSE", " Off "])
@pytest.mark.parametrize("nt", [1, 2, 32, 1000])
def test_force_off(env, nt):
    assert _grouped_enabled(env, grouped_min=2, num_tokens=nt) is False


def test_b1_default_off_but_forceable():
    # The shipped default: b1 -> block-diag (bit-exact), batched -> grouped.
    assert _grouped_enabled(None, 2, 1) is False     # b1 decode
    assert _grouped_enabled(None, 2, 32) is True      # b32 decode
    # ...but a user can force grouped on b1 for testing.
    assert _grouped_enabled("1", 2, 1) is True
