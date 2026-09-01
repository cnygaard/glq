"""The self-check must actually run, and a failed one must not be dressed up as success.

`next_steps()` opens with "GLQ is installed." — printing that over a venv whose plugin
does not resolve is the failure mode this wiring exists to prevent. The user would copy the
`vllm serve` line and meet "Unknown quantization method: glq" with no hint that the
installer already knew.

So: run the checks after a real install; on a hard failure say so, suppress the cheerful
summary, and exit non-zero (install.sh runs under `set -e`, so the caller sees it too).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.installer import __main__ as M  # noqa: E402
from glq.installer import verify as V  # noqa: E402
from glq.installer.discovery import Checkpoint  # noqa: E402

GIB = 1024 ** 3
FLEET = [Checkpoint("xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel",
                    int(1.8 * GIB), trellis=True)]


@pytest.fixture
def offline(monkeypatch):
    """No network, no GPU, no pip — just the control flow."""
    monkeypatch.setattr(M.discovery, "discover", lambda *a, **k: FLEET)
    monkeypatch.setattr(M.hardware, "gpu_name", lambda *a, **k: "Fake GPU")
    monkeypatch.setattr(M.hardware, "vram_bytes", lambda *a, **k: int(24 * GIB))
    monkeypatch.setattr(M, "_install_python_extras", lambda *a, **k: None)
    monkeypatch.setattr(M.configure, "write_glq_config", lambda *a, **k: None)
    monkeypatch.setattr(M.configure, "write_pi_models", lambda *a, **k: None)


def _healthy(components, **kw):
    return [V.Check("glq importable", True, "glq 0.8.3")]


def _broken(components, **kw):
    return [V.Check("glq_vllm importable", False,
                    "glq_vllm is missing — vLLM will reject `--quantization glq`")]


def test_a_healthy_install_runs_the_check_and_reports_success(offline, monkeypatch, capsys):
    monkeypatch.setattr(M.verify, "run_checks", _healthy)
    rc = M.main(["--yes", "--components", "core,vllm"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Self-check" in out
    assert "GLQ is installed" in out


def test_a_failed_check_exits_non_zero(offline, monkeypatch, capsys):
    monkeypatch.setattr(M.verify, "run_checks", _broken)
    rc = M.main(["--yes", "--components", "core,vllm"])
    assert rc != 0, "a broken install must not report success"


def test_a_failed_check_suppresses_the_installed_banner(offline, monkeypatch, capsys):
    """The specific harm: 'GLQ is installed.' followed by a serve command that cannot work."""
    monkeypatch.setattr(M.verify, "run_checks", _broken)
    M.main(["--yes", "--components", "core,vllm"])
    out = capsys.readouterr().out
    assert "GLQ is installed" not in out
    assert "glq_vllm is missing" in out


def test_the_failure_output_tells_the_user_what_to_do(offline, monkeypatch, capsys):
    monkeypatch.setattr(M.verify, "run_checks", _broken)
    M.main(["--yes", "--components", "core,vllm"])
    out = capsys.readouterr().out.lower()
    assert "incomplete" in out or "not usable" in out


def test_dry_run_skips_the_check(offline, monkeypatch, capsys):
    """Nothing was installed, so probing the venv would fail for the wrong reason and
    scare someone who only asked what the installer would do."""
    called = []
    monkeypatch.setattr(M.verify, "run_checks",
                        lambda *a, **k: called.append(1) or _healthy(*a, **k))
    rc = M.main(["--yes", "--dry-run", "--components", "core,vllm"])
    assert rc == 0
    assert called == [], "dry-run must not run the self-check"


def test_verify_flag_checks_an_existing_install_without_reinstalling(monkeypatch, capsys):
    """`glq-setup --verify` is the thing to ask someone to paste when they report
    'vllm serve doesn't work'."""
    monkeypatch.setattr(M.verify, "run_checks", _broken)
    rc = M.main(["--verify"])
    out = capsys.readouterr().out
    assert rc != 0
    assert "glq_vllm" in out


def test_verify_flag_needs_no_network(monkeypatch, capsys):
    """It must work on the broken box it is meant to diagnose, which may be offline."""
    def boom(*a, **k):
        raise AssertionError("--verify must not hit the network")
    monkeypatch.setattr(M.discovery, "discover", boom)
    monkeypatch.setattr(M.verify, "run_checks", _healthy)
    assert M.main(["--verify"]) == 0


# ---- device resolution (GPU → cuda; none → cpu; forced by flag) --------------------------

def _capture_config(monkeypatch):
    written = {}
    monkeypatch.setattr(M.configure, "write_glq_config",
                        lambda path, **kw: written.update(kw))
    return written


def test_gpu_box_resolves_device_cuda(offline, monkeypatch):
    monkeypatch.setattr(M.verify, "run_checks", _healthy)
    written = _capture_config(monkeypatch)
    assert M.main(["--yes", "--components", "core,vllm"]) == 0
    assert written["device"] == "cuda"


def test_no_gpu_resolves_device_cpu(offline, monkeypatch):
    monkeypatch.setattr(M.hardware, "gpu_name", lambda *a, **k: None)
    monkeypatch.setattr(M.hardware, "vram_bytes", lambda *a, **k: None)
    monkeypatch.setattr(M.hardware, "ram_bytes", lambda *a, **k: int(32 * GIB))
    monkeypatch.setattr(M.verify, "run_checks", _healthy)
    written = _capture_config(monkeypatch)
    assert M.main(["--yes", "--components", "core,vllm"]) == 0
    assert written["device"] == "cpu"


def test_cpu_flag_forces_cpu_on_a_gpu_box(offline, monkeypatch):
    """The Xeon e2e case: a GPU is visible but the user wants the CPU stack."""
    monkeypatch.setattr(M.hardware, "ram_bytes", lambda *a, **k: int(32 * GIB))
    monkeypatch.setattr(M.verify, "run_checks", _healthy)
    written = _capture_config(monkeypatch)
    assert M.main(["--yes", "--components", "core,vllm", "--cpu"]) == 0
    assert written["device"] == "cpu"


def test_assume_no_gpu_is_a_back_compat_alias(offline, monkeypatch):
    monkeypatch.setattr(M.hardware, "ram_bytes", lambda *a, **k: int(32 * GIB))
    monkeypatch.setattr(M.verify, "run_checks", _healthy)
    written = _capture_config(monkeypatch)
    assert M.main(["--yes", "--components", "core,vllm", "--assume-no-gpu"]) == 0
    assert written["device"] == "cpu"
