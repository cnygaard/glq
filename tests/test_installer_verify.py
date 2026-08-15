"""Post-install self-check (glq/installer/verify.py).

Written after a real failure: a venv where `glq` imported fine but `glq_vllm` was absent, so
vLLM's plugin load failed and `--quantization glq` came back as *"Unknown quantization
method: glq"* — a message that points at the model or the flag, not at the broken install.

The installer is the last place that can catch that cheaply. It knows what it just
installed, so it can assert the pieces actually resolve before telling the user to run
`vllm serve` and letting them debug a confusing error minutes later.

Every check takes an injected probe, so the tests cover the broken combinations without
needing a broken venv.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.installer import verify as V  # noqa: E402


def _probes(glq=True, glq_vllm=True, plugin=True, cuda=True):
    return {
        "glq_importable": lambda: ("0.8.3" if glq else None),
        "glq_vllm_importable": lambda: glq_vllm,
        "plugin_registered": lambda: plugin,
        "cuda_available": lambda: cuda,
    }


def test_a_healthy_install_reports_all_ok():
    checks = V.run_checks(("core", "vllm"), **_probes())
    assert all(c.ok for c in checks), [c.name for c in checks if not c.ok]
    assert V.all_ok(checks) is True


def test_missing_glq_vllm_is_caught():
    """The exact failure this module exists for."""
    checks = V.run_checks(("core", "vllm"), **_probes(glq_vllm=False, plugin=False))
    assert V.all_ok(checks) is False
    bad = [c for c in checks if not c.ok]
    assert any("glq_vllm" in c.name for c in bad)


def test_the_failure_message_names_the_symptom_the_user_would_see():
    """Connecting cause to symptom is the whole value: without it, 'Unknown quantization
    method: glq' sends people to the model card or the CLI flag."""
    checks = V.run_checks(("core", "vllm"), **_probes(glq_vllm=False, plugin=False))
    text = " ".join(c.detail for c in checks if not c.ok)
    assert "quantization method" in text
    assert "pip install" in text          # and a way out


def test_plugin_check_is_skipped_without_the_vllm_component():
    """Someone who installed core only has no vLLM to register a plugin with; reporting a
    failure there would be noise."""
    names = [c.name for c in V.run_checks(("core",), **_probes(glq_vllm=False, plugin=False))]
    assert not any("plugin" in n for n in names)


def test_missing_glq_itself_is_caught():
    checks = V.run_checks(("core",), **_probes(glq=False))
    assert V.all_ok(checks) is False


def test_no_cuda_is_a_warning_not_a_failure():
    """CPU-only is a supported (slow) configuration — glq falls back to
    dequantize-then-matmul. Failing the install over it would be wrong."""
    checks = V.run_checks(("core", "vllm"), **_probes(cuda=False))
    cuda = [c for c in checks if "cuda" in c.name.lower()][0]
    assert cuda.ok is False
    assert cuda.warning_only is True
    assert V.all_ok(checks) is True       # warnings do not fail the install


def test_version_is_reported_so_a_stale_install_is_visible():
    checks = V.run_checks(("core",), **_probes())
    assert any("0.8.3" in c.detail for c in checks)


def test_a_probe_that_raises_is_a_failure_not_a_crash():
    """A self-check that takes down the installer is worse than no self-check."""
    def boom():
        raise RuntimeError("segfault in torch")
    probes = _probes()
    probes["glq_importable"] = boom
    checks = V.run_checks(("core",), **probes)
    assert V.all_ok(checks) is False
    assert "segfault" in " ".join(c.detail for c in checks)


def test_render_is_readable_and_marks_each_line():
    text = V.render(V.run_checks(("core", "vllm"), **_probes(glq_vllm=False, plugin=False)))
    assert "glq_vllm" in text
    assert "FAIL" in text and "ok" in text
