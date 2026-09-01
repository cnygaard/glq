"""Detect the GPU without importing torch.

This runs seconds after `pip install glq`, which may have pulled a torch whose CUDA build
does not match the driver. `import torch` in that state does not always raise something
catchable — it can abort the interpreter — and it costs seconds even when it works.
`nvidia-smi` ships with the driver, answers in milliseconds, and fails cleanly.

Every failure mode here is a machine someone will actually run this on, so all of them
return None rather than raising: a CPU-only box and a container started without `--gpus`
must still get a working installer, just without a recommendation.
"""
from __future__ import annotations

import subprocess

MIB = 1024 ** 2
_TIMEOUT = 10


def _query(field: str, run=subprocess.run) -> list[str]:
    try:
        proc = run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
                   capture_output=True, text=True, timeout=_TIMEOUT, check=False)
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]


def vram_bytes(run=subprocess.run) -> int | None:
    """VRAM of the largest single GPU, or None if that can't be determined.

    The largest *single* card is the right number because vLLM serves on one GPU by
    default (tp=1); summing a multi-GPU host would recommend a checkpoint that no
    individual device can hold.
    """
    best = None
    for line in _query("memory.total", run=run):
        mib = line.split()[0]
        if not mib.isdigit():
            continue
        val = int(mib) * MIB
        best = val if best is None else max(best, val)
    return best


def gpu_name(run=subprocess.run) -> str | None:
    """Name of the first GPU, for the installer's banner — so a user can spot immediately
    that it detected the wrong card before agreeing to a multi-GiB download."""
    names = _query("name", run=run)
    return names[0] if names else None


def _read_meminfo() -> str:
    with open("/proc/meminfo") as fh:
        return fh.read()


def ram_bytes(read=_read_meminfo) -> int | None:
    """Total system RAM, or None if it can't be determined.

    The CPU-serving budget source: when there is no GPU, model recommendation sizes
    against RAM instead of VRAM. /proc/meminfo is Linux-only by design — GLQ's serving
    stack is Linux-only (see README system requirements) — and every failure returns
    None, same contract as vram_bytes: no recommendation beats a wrong one.
    """
    try:
        for line in read().splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1]) * 1024          # meminfo reports kB
                return None
    except Exception:                                    # noqa: BLE001 - any failure = unknown
        return None
    return None
