"""Loader for the CPU fused-decode extension (glq._C_cpu), mirroring the CUDA ladder in
glq/inference_kernel.py but fully independent of it: the CUDA extension's absence means
"no fused GPU path", and overloading that state would make a CPU-only box look broken.

Ladder: (1) the prebuilt wheel extension ``glq._C_cpu``; (2) JIT via
``torch.utils.cpp_extension.load`` — plain C++ (g++/clang, no nvcc), so source installs
on machines with no CUDA toolchain gain a fused path for the first time. Both failures
are captured for ``cpu_ext_status()`` rather than raised: callers fall back to the dense
pure-torch path, exactly like the CUDA side's contract.
"""
from __future__ import annotations

import os

_glq_cpu = None
_cpu_ext_error: str | None = None
_tried = False

_SOURCES = ("glq_trellis_cpu_scalar.cpp", "glq_trellis_cpu_avx2.cpp",
            "glq_trellis_cpu_avx512.cpp", "glq_fht_cpu.cpp",
            "glq_cpu_dispatch.cpp", "glq_bindings_cpu.cpp")


def _csrc_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "cpu")


def _try_load_cpu_ext() -> bool:
    """Load the CPU extension if possible; idempotent; never raises."""
    global _glq_cpu, _cpu_ext_error, _tried
    if _glq_cpu is not None:
        return True
    if _tried:
        return False
    _tried = True

    prebuilt_err = None
    try:
        from glq import _C_cpu as _prebuilt  # type: ignore[attr-defined]
        _glq_cpu = _prebuilt
        return True
    except Exception as exc:  # noqa: BLE001 - diagnostic capture, not control flow
        prebuilt_err = f"prebuilt glq._C_cpu: {type(exc).__name__}: {exc}"

    try:
        import sys

        from torch.utils import cpp_extension

        # torch's JIT build shells out to `ninja` by name; in a non-interactive shell the
        # venv's bin/ is not on PATH even though ninja is installed there (the same field
        # failure decode_sweep hardens against). Prepend it so the probe finds the tool.
        bindir = os.path.dirname(sys.executable)
        path = os.environ.get("PATH", "")
        if bindir not in path.split(os.pathsep):
            os.environ["PATH"] = bindir + os.pathsep + path

        src_dir = _csrc_dir()
        _glq_cpu = cpp_extension.load(
            name="glq_cpu",
            sources=[os.path.join(src_dir, s) for s in _SOURCES],
            extra_cflags=["-O3", "-std=c++17", "-fopenmp"],
            extra_ldflags=["-fopenmp"],
            verbose=os.environ.get("GLQ_CPU_EXT_VERBOSE", "") == "1",
        )
        return True
    except Exception as exc:  # noqa: BLE001
        _cpu_ext_error = f"{prebuilt_err}; JIT build: {type(exc).__name__}: {exc}"
        return False


def cpu_ext_status() -> str:
    if _glq_cpu is not None:
        return f"loaded (isa={_glq_cpu.glq_cpu_active_isa()})"
    if _cpu_ext_error is not None:
        return f"unavailable: {_cpu_ext_error}"
    return "not attempted"


def require_cpu_ext(symbol: str | None = None):
    """The extension module, or a diagnostic RuntimeError naming what failed."""
    if not _try_load_cpu_ext():
        raise RuntimeError(f"glq CPU extension unavailable — {cpu_ext_status()}")
    if symbol is not None and not hasattr(_glq_cpu, symbol):
        raise RuntimeError(f"glq CPU extension loaded but lacks {symbol!r} "
                           f"(stale build? {cpu_ext_status()})")
    return _glq_cpu
