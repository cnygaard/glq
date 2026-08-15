"""Post-install self-check.

Written after a real failure on a test box: `glq` imported fine but `glq_vllm` was absent,
so vLLM's plugin load failed and `vllm serve --quantization glq` came back with

    Value error, Unknown quantization method: glq. Must be one of ['awq', ...]

which points the reader at the model or the CLI flag — anywhere but the actual cause. The
installer is the cheapest place to catch that: it knows what it just installed, so it can
assert the pieces resolve *before* telling the user to run a command that will fail
confusingly minutes later.

Probes are injected so the failure combinations can be tested without a broken venv.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    #: True for things that are worth saying but must not fail the install (e.g. no GPU).
    warning_only: bool = False


# ----------------------------------------------------------------- real probes

def _glq_importable():
    import glq
    return getattr(glq, "__version__", "unknown")


def _glq_vllm_importable() -> bool:
    try:
        import glq_vllm  # noqa: F401
        return True
    except Exception:                                             # noqa: BLE001
        return False


def _plugin_registered() -> bool:
    """Is the `glq` plugin visible in the entry-point group vLLM reads?

    This is what actually decides whether `--quantization glq` is accepted, so check the
    registration rather than merely that the module imports.
    """
    try:
        from importlib.metadata import entry_points
        return any(e.name == "glq"
                   for e in entry_points(group="vllm.general_plugins"))
    except Exception:                                             # noqa: BLE001
        return False


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:                                             # noqa: BLE001
        return False


def _safe(probe, on_error):
    """Run a probe; a self-check that crashes the installer is worse than none."""
    try:
        return probe(), None
    except Exception as exc:                                      # noqa: BLE001
        return on_error, f"{type(exc).__name__}: {exc}"


def run_checks(components, *, glq_importable=_glq_importable,
               glq_vllm_importable=_glq_vllm_importable,
               plugin_registered=_plugin_registered,
               cuda_available=_cuda_available) -> list[Check]:
    """Assert the install can actually do what the next-steps text is about to promise."""
    checks: list[Check] = []

    version, err = _safe(glq_importable, None)
    checks.append(Check(
        "glq importable", version is not None,
        f"glq {version}" if version else f"cannot import glq — {err or 'not installed'}"))

    if "vllm" in components:
        have, err = _safe(glq_vllm_importable, False)
        checks.append(Check(
            "glq_vllm importable", bool(have),
            "vLLM plugin package present" if have else
            "glq_vllm is missing — vLLM will reject `--quantization glq` with "
            "'Unknown quantization method: glq'. Fix: pip install --force-reinstall glq"
            + (f" ({err})" if err else "")))

        reg, err = _safe(plugin_registered, False)
        checks.append(Check(
            "vllm plugin registered", bool(reg),
            "entry point vllm.general_plugins:glq found" if reg else
            "the glq entry point is not registered, so vLLM will report "
            "'Unknown quantization method: glq'. Fix: pip install --force-reinstall glq"
            + (f" ({err})" if err else "")))

    cuda, err = _safe(cuda_available, False)
    checks.append(Check(
        "cuda available", bool(cuda),
        "GPU visible to torch" if cuda else
        "no CUDA GPU visible — GLQ falls back to dequantize-then-matmul on CPU, which "
        "works but is slow" + (f" ({err})" if err else ""),
        warning_only=True))

    return checks


def all_ok(checks) -> bool:
    """Warnings do not fail an install; CPU-only is a supported configuration."""
    return all(c.ok or c.warning_only for c in checks)


def render(checks) -> str:
    lines = ["", "Self-check:"]
    for c in checks:
        mark = "ok  " if c.ok else ("warn" if c.warning_only else "FAIL")
        lines.append(f"  [{mark}] {c.name}: {c.detail}")
    return "\n".join(lines)
