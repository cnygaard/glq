"""Capture reproducibility provenance: library versions + GPU identity.

Everything degrades gracefully to ``None`` so this imports and runs on a CPU box
with neither torch nor a GPU present (the unit tests rely on that). The only hard
dependency is the stdlib.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .record import EnvMeta, HardwareMeta


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:  # noqa: BLE001 — importlib.metadata always present on 3.8+, belt + braces
        return None


def _torch_cuda() -> str | None:
    try:
        import torch
        return getattr(torch.version, "cuda", None)
    except Exception:  # noqa: BLE001 — torch absent on a CPU planning box
        return None


def _nvidia_smi(query: str) -> list[str]:
    """Run ``nvidia-smi --query-gpu=<query> --format=csv,noheader`` -> stripped rows.

    Returns ``[]`` if nvidia-smi is missing or errors (CPU box / no driver)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False)
        if out.returncode != 0:
            return []
        return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception:  # noqa: BLE001
        return []


def _driver_version() -> str | None:
    rows = _nvidia_smi("driver_version")
    return rows[0] if rows else None


def _glq_git_sha() -> str | None:
    """Short git sha of the installed glq source tree, if it lives in a checkout."""
    try:
        import glq
        root = Path(glq.__file__).resolve().parent.parent
    except Exception:  # noqa: BLE001
        return None
    if not (root / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False)
        sha = out.stdout.strip()
        return sha or None
    except Exception:  # noqa: BLE001
        return None


def env_snapshot() -> EnvMeta:
    """Versions of the libraries that determine a run's behaviour."""
    import sys
    py = "%d.%d.%d" % sys.version_info[:3]
    return EnvMeta(
        python=py,
        torch=_pkg_version("torch"),
        vllm=_pkg_version("vllm"),
        transformers=_pkg_version("transformers"),
        sglang=_pkg_version("sglang"),
        triton=_pkg_version("triton"),
        cuda=_torch_cuda(),
        driver=_driver_version(),
        glq=_pkg_version("glq"),
        glq_git_sha=_glq_git_sha(),
    )


def hardware_snapshot() -> HardwareMeta:
    """GPU model + count + per-GPU total VRAM. torch first, nvidia-smi fallback."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return HardwareMeta(
                gpu_model=torch.cuda.get_device_name(0),
                gpu_count=torch.cuda.device_count(),
                gpu_total_vram_gib=round(props.total_memory / 2**30, 1),
            )
    except Exception:  # noqa: BLE001
        pass
    # nvidia-smi fallback (no torch / torch sees no CUDA)
    names = _nvidia_smi("name")
    mems = _nvidia_smi("memory.total")  # MiB
    if names:
        vram = None
        if mems:
            try:
                vram = round(float(mems[0]) / 1024, 1)
            except ValueError:
                vram = None
        return HardwareMeta(gpu_model=names[0], gpu_count=len(names),
                            gpu_total_vram_gib=vram)
    return HardwareMeta()
