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


def _probes(glq=True, glq_vllm=True, plugin=True, cuda=True, kernels=(True, None)):
    return {
        "glq_importable": lambda: ("0.8.3" if glq else None),
        "glq_vllm_importable": lambda: glq_vllm,
        "plugin_registered": lambda: plugin,
        "cuda_available": lambda: cuda,
        "kernels_available": lambda: kernels,
    }


# --------------------------------------------------- the check that was missing
#
# Measured in an ubuntu:24.04 container (2026-08-15): `glq-setup --verify` reported
#
#     [ok  ] cuda available: GPU visible to torch
#
# and the very next command died in a forward pass, because the CUDA extension had never
# built. `torch.cuda.is_available()` answers "is there a GPU", which is not the question —
# GLQ compiles its kernels on the user's machine, so the question is "did that work".

BUILD_ERROR = "fatal error: nv/target: No such file or directory"


def test_a_gpu_with_unbuilt_kernels_is_a_failure_not_a_pass():
    checks = V.run_checks(("core", "vllm"),
                          **_probes(cuda=True, kernels=(False, BUILD_ERROR)))
    kernel = [c for c in checks if "kernel" in c.name.lower()]
    assert kernel, f"no kernel check at all: {[c.name for c in checks]}"
    kernel = kernel[0]

    assert kernel.ok is False
    assert kernel.warning_only is False, (
        "a GPU box whose kernels cannot build is broken for GLQ's purpose — reporting it as "
        "a warning is how this shipped green while the install was unusable")
    assert not V.all_ok(checks)


def test_the_kernel_failure_carries_the_build_reason():
    """The reason is the actionable part; recovering it by hand cost three container
    re-runs this morning."""
    checks = V.run_checks(("core",), **_probes(cuda=True, kernels=(False, BUILD_ERROR)))
    kernel = [c for c in checks if "kernel" in c.name.lower()][0]
    assert BUILD_ERROR in kernel.detail


def test_kernels_are_not_failed_on_a_cpu_only_box():
    """CPU-only is supported (dequantize-then-matmul). Without a GPU there is nothing to
    build against, so an unbuilt kernel is expected, not a fault."""
    checks = V.run_checks(("core",), **_probes(cuda=False, kernels=(False, BUILD_ERROR)))
    kernel = [c for c in checks if "kernel" in c.name.lower()][0]
    assert kernel.warning_only is True
    assert V.all_ok(checks)


def test_built_kernels_report_ok():
    checks = V.run_checks(("core",), **_probes(cuda=True, kernels=(True, None)))
    kernel = [c for c in checks if "kernel" in c.name.lower()][0]
    assert kernel.ok is True


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


def test_quantize_component_checks_its_deps_are_importable():
    """The binary exists in every install; only the deps distinguish a working quantize
    from a traceback on `from datasets import load_dataset` twenty minutes in."""
    checks = V.run_checks(("core", "quantize"), **_probes(),
                          quantize_deps_importable=lambda: False)
    c = [c for c in checks if "quantize" in c.name]
    assert c and not c[0].ok
    assert "glq[quantize]" in c[0].detail

    checks = V.run_checks(("core", "quantize"), **_probes(),
                          quantize_deps_importable=lambda: True)
    c = [c for c in checks if "quantize" in c.name]
    assert c and c[0].ok


def test_no_quantize_component_no_quantize_check():
    checks = V.run_checks(("core",), **_probes())
    assert not any("quantize" in c.name for c in checks)


def test_picode_component_checks_the_pi_binary_resolves():
    """The summary is about to print `glq-code`; if pi is not actually resolvable the
    user meets a runtime error far from the install that caused it."""
    checks = V.run_checks(("core", "picode"), **_probes(),
                          pi_resolvable=lambda: False)
    c = [c for c in checks if "pi " in c.name or "picode" in c.name]
    assert c and not c[0].ok
    assert "picode" in c[0].detail

    checks = V.run_checks(("core", "picode"), **_probes(),
                          pi_resolvable=lambda: True)
    c = [c for c in checks if "pi " in c.name or "picode" in c.name]
    assert c and c[0].ok


def test_no_picode_component_no_pi_check():
    checks = V.run_checks(("core",), **_probes())
    assert not any("pi " in c.name or "picode" in c.name for c in checks)


# ---- CPU serving checks (the CPU kernels + the installed-wheel/device match) -------------
# On a CPU-only box the vLLM component's serving path IS the CPU extension and the +cpu
# wheel; "green but silent about both" was the pre-CPU-serving behavior and it let a
# broken CPU install print "GLQ is installed."

def _cpu_probes(cuda=False, cpu_kernels=(True, "loaded (isa=avx2)"),
                vllm_version="0.28.0+cpu", **kw):
    probes = _probes(cuda=cuda, **kw)
    probes["cpu_kernels_available"] = lambda: cpu_kernels
    probes["vllm_version"] = lambda: vllm_version
    return probes


def test_missing_cpu_kernels_fail_on_a_cpu_box_with_vllm():
    checks = V.run_checks(("core", "vllm"),
                          **_cpu_probes(cpu_kernels=(False, "unavailable: no compiler")))
    row = next(c for c in checks if "cpu kernels" in c.name)
    assert not row.ok and not row.warning_only
    assert not V.all_ok(checks)


def test_missing_cpu_kernels_are_a_warning_on_a_gpu_box():
    checks = V.run_checks(("core", "vllm"),
                          **_cpu_probes(cuda=True,
                                        cpu_kernels=(False, "unavailable: no compiler")))
    row = next(c for c in checks if "cpu kernels" in c.name)
    assert row.warning_only


def test_missing_cpu_kernels_are_a_warning_without_the_vllm_component():
    checks = V.run_checks(("core",),
                          **_cpu_probes(cpu_kernels=(False, "unavailable: no compiler")))
    row = next(c for c in checks if "cpu kernels" in c.name)
    assert row.warning_only


def test_cuda_wheel_on_a_cpu_box_fails_and_names_the_fix():
    checks = V.run_checks(("core", "vllm"), **_cpu_probes(vllm_version="0.11.0"))
    row = next(c for c in checks if "vllm backend" in c.name)
    assert not row.ok and not row.warning_only
    assert "--device cpu" in row.detail


def test_cpu_wheel_on_a_cpu_box_is_ok():
    checks = V.run_checks(("core", "vllm"), **_cpu_probes())
    row = next(c for c in checks if "vllm backend" in c.name)
    assert row.ok


def test_cpu_wheel_on_a_gpu_box_is_a_warning():
    checks = V.run_checks(("core", "vllm"), **_cpu_probes(cuda=True))
    row = next(c for c in checks if "vllm backend" in c.name)
    assert not row.ok and row.warning_only


def test_unreadable_vllm_version_does_not_fail_the_check():
    """vllm may simply not be installed (core-only venv) — that is not a backend
    mismatch."""
    checks = V.run_checks(("core", "vllm"), **_cpu_probes(vllm_version=None))
    row = next(c for c in checks if "vllm backend" in c.name)
    assert row.ok or row.warning_only


# ---- device-aware verify: a --cpu install must not exercise (or fail on) CUDA ------------

def test_cpu_device_skips_the_cuda_jit_and_downgrades_cuda_rows():
    """Measured on a GPU box with --cpu: the installer's verify attempted the CUDA JIT
    (system nvcc + by-then +cpu torch headers = certain failure) and reported it at
    failure level — 'Install INCOMPLETE' on a healthy CPU install. With device=cpu the
    CUDA probes are not run at all."""
    probed = {"kernels": 0}

    def kernels():
        probed["kernels"] += 1
        return (False, "should never run")

    probes = _cpu_probes()
    probes["kernels_available"] = kernels
    checks = V.run_checks(("core", "vllm"), device="cpu", **probes)
    assert probed["kernels"] == 0, "the CUDA JIT was attempted on a cpu install"
    cuda_row = next(c for c in checks if c.name == "glq cuda kernels")
    assert cuda_row.warning_only
    assert V.all_ok(checks)


def test_cpu_device_accepts_the_cpu_wheel_without_a_gpu_warning():
    """On a deliberate cpu install, '+cpu wheel while a GPU is visible' is the CHOSEN
    state, not a surprise to warn about."""
    probes = _cpu_probes(cuda=True)
    checks = V.run_checks(("core", "vllm"), device="cpu", **probes)
    row = next(c for c in checks if "vllm backend" in c.name)
    assert row.ok and not row.warning_only


def test_default_device_keeps_live_probing():
    probes = _cpu_probes(cuda=True, cpu_kernels=(False, "x"))
    checks = V.run_checks(("core", "vllm"), **probes)
    row = next(c for c in checks if "vllm backend" in c.name)
    assert row.warning_only          # +cpu wheel on a GPU box, no declared device
