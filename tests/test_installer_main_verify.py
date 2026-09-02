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
import subprocess
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


# The install path runs its self-check in a SUBPROCESS (see M._self_check), so these fake
# that seam: (rc, rendered output). The --verify path below still calls run_checks
# in-process — that IS the child process, and patching it there is the right seam.
def _healthy_check(venv, components, device=None, **kw):
    return 0, V.render(_healthy(components))


def _broken_check(venv, components, device=None, **kw):
    return 1, V.render(_broken(components))


def test_a_healthy_install_runs_the_check_and_reports_success(offline, monkeypatch, capsys):
    monkeypatch.setattr(M, "_self_check", _healthy_check)
    rc = M.main(["--yes", "--components", "core,vllm"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Self-check" in out
    assert "GLQ is installed" in out


def test_a_failed_check_exits_non_zero(offline, monkeypatch, capsys):
    monkeypatch.setattr(M, "_self_check", _broken_check)
    rc = M.main(["--yes", "--components", "core,vllm"])
    assert rc != 0, "a broken install must not report success"


def test_a_failed_check_suppresses_the_installed_banner(offline, monkeypatch, capsys):
    """The specific harm: 'GLQ is installed.' followed by a serve command that cannot work."""
    monkeypatch.setattr(M, "_self_check", _broken_check)
    M.main(["--yes", "--components", "core,vllm"])
    out = capsys.readouterr().out
    assert "GLQ is installed" not in out
    assert "glq_vllm is missing" in out


def test_the_failure_output_tells_the_user_what_to_do(offline, monkeypatch, capsys):
    monkeypatch.setattr(M, "_self_check", _broken_check)
    M.main(["--yes", "--components", "core,vllm"])
    out = capsys.readouterr().out.lower()
    assert "incomplete" in out or "not usable" in out


def test_dry_run_skips_the_check(offline, monkeypatch, capsys):
    """Nothing was installed, so probing the venv would fail for the wrong reason and
    scare someone who only asked what the installer would do."""
    called = []
    monkeypatch.setattr(M, "_self_check",
                        lambda *a, **k: called.append(1) or _healthy_check(*a, **k))
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
    monkeypatch.setattr(M, "_self_check", _healthy_check)
    written = _capture_config(monkeypatch)
    assert M.main(["--yes", "--components", "core,vllm"]) == 0
    assert written["device"] == "cuda"


def test_no_gpu_resolves_device_cpu(offline, monkeypatch):
    monkeypatch.setattr(M.hardware, "gpu_name", lambda *a, **k: None)
    monkeypatch.setattr(M.hardware, "vram_bytes", lambda *a, **k: None)
    monkeypatch.setattr(M.hardware, "ram_bytes", lambda *a, **k: int(32 * GIB))
    monkeypatch.setattr(M, "_self_check", _healthy_check)
    written = _capture_config(monkeypatch)
    assert M.main(["--yes", "--components", "core,vllm"]) == 0
    assert written["device"] == "cpu"


def test_cpu_flag_forces_cpu_on_a_gpu_box(offline, monkeypatch):
    """The Xeon e2e case: a GPU is visible but the user wants the CPU stack."""
    monkeypatch.setattr(M.hardware, "ram_bytes", lambda *a, **k: int(32 * GIB))
    monkeypatch.setattr(M, "_self_check", _healthy_check)
    written = _capture_config(monkeypatch)
    assert M.main(["--yes", "--components", "core,vllm", "--cpu"]) == 0
    assert written["device"] == "cpu"


def test_assume_no_gpu_is_a_back_compat_alias(offline, monkeypatch):
    monkeypatch.setattr(M.hardware, "ram_bytes", lambda *a, **k: int(32 * GIB))
    monkeypatch.setattr(M, "_self_check", _healthy_check)
    written = _capture_config(monkeypatch)
    assert M.main(["--yes", "--components", "core,vllm", "--assume-no-gpu"]) == 0
    assert written["device"] == "cpu"


# ---- the self-check must run in a FRESH interpreter ---------------------------------------
#
# Measured on a fresh L40S box (distro matrix, 2026-09-02): glq's own install pulls the
# newest torch (2.14.0), then `pip install vllm` DOWNGRADES it to 2.13.0 — vLLM pins torch.
# The installer process is still holding torch 2.14 in memory while 2.13 is on disk, so an
# in-process `import glq_vllm` (which imports vllm) dies with
#
#     AttributeError: '_OpNamespace' 'aten' object has no attribute 'cholesky'
#
# and the installer declares a perfectly good install "INCOMPLETE" with exit 1. A fresh
# interpreter sees one consistent torch and passes. Reproduced outside Docker; importlib
# cache invalidation does NOT help, because the files were never the problem.

def test_the_self_check_runs_in_a_subprocess_not_in_process(offline, monkeypatch, capsys):
    """The in-process probes cannot be trusted after pip has replaced torch under us."""
    def exploding(*a, **k):
        raise AssertionError("verify.run_checks was called IN-PROCESS after installing vLLM")
    monkeypatch.setattr(M.verify, "run_checks", exploding)

    seen = []

    def fake_run(cmd, **kw):
        seen.append([str(c) for c in cmd])
        return subprocess.CompletedProcess(cmd, 0, stdout="Self-check:\n  [ok  ] glq\n",
                                           stderr="")
    monkeypatch.setattr(M.subprocess, "run", fake_run)

    rc = M.main(["--yes", "--components", "core,vllm"])
    assert rc == 0
    check = [c for c in seen if "--verify" in c]
    assert check, f"no subprocess self-check was run: {seen}"
    assert check[0][0].endswith("python"), check[0]
    assert "core,vllm" in check[0]


def test_a_failing_subprocess_check_still_fails_the_install(offline, monkeypatch, capsys):
    monkeypatch.setattr(M.verify, "run_checks", lambda *a, **k: [])
    monkeypatch.setattr(M.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(
                            cmd, 1, stdout="Self-check:\n  [FAIL] glq_vllm importable\n",
                            stderr=""))
    rc = M.main(["--yes", "--components", "core,vllm"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "GLQ is installed" not in out


def test_the_device_travels_to_the_subprocess_check(offline, monkeypatch):
    """A cpu install must not have the CUDA rows probed in the child either."""
    monkeypatch.setattr(M.hardware, "ram_bytes", lambda *a, **k: int(32 * GIB))
    seen = []
    monkeypatch.setattr(M.subprocess, "run",
                        lambda cmd, **kw: seen.append([str(c) for c in cmd]) or
                        subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    M.main(["--yes", "--components", "core,vllm", "--cpu"])
    check = [c for c in seen if "--verify" in c]
    assert check and "cpu" in check[0], check
