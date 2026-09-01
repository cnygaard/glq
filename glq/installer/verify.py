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


def _quantize_deps_importable() -> bool:
    """datasets is the quantize dep nothing else drags in — transformers, safetensors and
    accelerate all ride with vLLM, so its absence is what an incomplete quantize install
    actually looks like at runtime."""
    import importlib.util
    return importlib.util.find_spec("datasets") is not None


def _pi_resolvable() -> bool:
    """Can the pi binary the summary is about to promise actually be found?

    Lazy import: glq.code pulls the supervisor chain, which verify should not pay for
    unless the component was chosen.
    """
    try:
        from glq.code import _find_pi
        return _find_pi() is not None
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


def _kernels_available():
    """(built, reason) for GLQ's fused CUDA kernels.

    `torch.cuda.is_available()` answers "is there a GPU", which is not the question that
    decides whether GLQ works: the kernels are compiled on this machine, and that compile
    has its own prerequisites. Measured in a container — a box can report a healthy GPU and
    still have no usable kernels, which is how a broken install passed this self-check and
    then died in the first forward pass.
    """
    try:
        from glq import inference_kernel as ik
        return ik.cuda_ext_status()
    except Exception as exc:                                      # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _cpu_kernels_available():
    """(built, status) for the fused CPU kernels.

    Must actually attempt the load: cpu_ext_status() alone reads "not attempted" on a
    fresh process. On a CPU-only box with the vllm component this IS the serving path.
    """
    try:
        from glq import inference_kernel_cpu as ikc
        return bool(ikc._try_load_cpu_ext()), ikc.cpu_ext_status()
    except Exception as exc:                                      # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _vllm_version():
    """The installed vLLM's version string (metadata read, no import), or None."""
    try:
        from importlib.metadata import version
        return version("vllm")
    except Exception:                                             # noqa: BLE001
        return None


def _safe(probe, on_error):
    """Run a probe; a self-check that crashes the installer is worse than none."""
    try:
        return probe(), None
    except Exception as exc:                                      # noqa: BLE001
        return on_error, f"{type(exc).__name__}: {exc}"


def run_checks(components, *, glq_importable=_glq_importable,
               glq_vllm_importable=_glq_vllm_importable,
               plugin_registered=_plugin_registered,
               cuda_available=_cuda_available,
               kernels_available=_kernels_available,
               quantize_deps_importable=_quantize_deps_importable,
               pi_resolvable=_pi_resolvable,
               cpu_kernels_available=_cpu_kernels_available,
               vllm_version=_vllm_version) -> list[Check]:
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

    if "quantize" in components:
        have, err = _safe(quantize_deps_importable, False)
        checks.append(Check(
            "quantize deps importable", bool(have),
            "datasets present — glq-quantize can load calibration data" if have else
            "the quantize deps are missing — glq-quantize will fail at import. "
            "Fix: pip install 'glq[quantize]'" + (f" ({err})" if err else "")))

    if "picode" in components:
        have, err = _safe(pi_resolvable, False)
        checks.append(Check(
            "pi binary resolvable", bool(have),
            "the pi coding agent is installed — glq-code can run it" if have else
            "pi is not resolvable — glq-code cannot run. Fix: re-run install.sh with "
            "the picode component" + (f" ({err})" if err else "")))

    cuda, err = _safe(cuda_available, False)
    checks.append(Check(
        "cuda available", bool(cuda),
        "GPU visible to torch" if cuda else
        "no CUDA GPU visible — GLQ falls back to dequantize-then-matmul on CPU, which "
        "works but is slow" + (f" ({err})" if err else ""),
        warning_only=True))

    # The check that matters on a GPU box: the kernels are compiled here, so a healthy GPU
    # does not imply a working GLQ. Only a real failure when a GPU *is* present — with no
    # GPU there is nothing to build against and CPU-only is a supported configuration.
    (built, reason), err = _safe(kernels_available, (False, None))
    checks.append(Check(
        "glq cuda kernels", bool(built),
        "fused kernels ready" if built else
        "the CUDA kernels are not built, so GLQ cannot run its fast path"
        + (f":\n{reason}" if reason else "")
        + (f" ({err})" if err else ""),
        warning_only=not bool(cuda)))

    # The mirror check for CPU serving: on a CPU-only box with the vllm component, the
    # CPU extension IS the serving path — failure-level there, informational elsewhere.
    (cpu_built, cpu_status), err = _safe(cpu_kernels_available, (False, None))
    checks.append(Check(
        "glq cpu kernels", bool(cpu_built),
        f"fused CPU kernels ready ({cpu_status})" if cpu_built else
        "the CPU kernels are not available — CPU serving would fall back to the slow "
        "dense path" + (f": {cpu_status}" if cpu_status else "")
        + (f" ({err})" if err else ""),
        warning_only=bool(cuda) or "vllm" not in components))

    # Wheel/device match: a CUDA vLLM on a CPU-only box cannot serve at all, and the
    # failure it produces at start time (CUDA driver errors from a wheel import) does not
    # name the actual fix. A +cpu wheel on a GPU box serves — on the CPU — so that is a
    # warning: probably not what the user meant, but not broken.
    if "vllm" in components:
        ver, err = _safe(vllm_version, None)
        if ver is None:
            checks.append(Check(
                "vllm backend", True,
                "vllm not installed (or unreadable) — nothing to match against"
                + (f" ({err})" if err else ""), warning_only=True))
        elif "+cpu" in ver:
            checks.append(Check(
                "vllm backend", bool(cuda) is False,
                f"vLLM {ver} — the CPU backend" if not cuda else
                f"vLLM {ver} is the CPU backend but a GPU is visible — serving will run "
                f"on the CPU. Reinstall for the GPU: glq-setup --components vllm",
                warning_only=bool(cuda)))
        else:
            checks.append(Check(
                "vllm backend", bool(cuda),
                f"vLLM {ver} — the CUDA backend" if cuda else
                f"vLLM {ver} is the CUDA build but no GPU is visible — serving cannot "
                f"start. Fix: glq-setup --components vllm --device cpu"))

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
