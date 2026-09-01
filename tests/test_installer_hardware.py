"""GPU detection for the installer (glq/installer/hardware.py).

Deliberately shells out to `nvidia-smi` rather than importing torch: this runs seconds after
`pip install glq`, potentially before a CUDA-matched torch is in place, and `import torch`
on a broken CUDA install does not fail cleanly — it can abort the process. `nvidia-smi` is
present on every box with a driver and answers in milliseconds.

Every branch here is a real machine someone will run the installer on: a CPU-only box, a
container with the driver but no device, and a multi-GPU host.
"""
from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.installer import hardware as H  # noqa: E402

MIB = 1024 ** 2


def _runner(stdout="", returncode=0, raises=None):
    def run(cmd, **kw):
        if raises:
            raise raises
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")
    return run


def test_parses_a_single_gpu():
    """Real output from the RTX PRO 6000 box this was developed against."""
    assert H.vram_bytes(run=_runner("97887 MiB\n")) == 97887 * MIB


def test_largest_gpu_wins_on_a_multi_gpu_host():
    """vLLM defaults to one GPU (tp=1), so the number that matters is the biggest single
    card, not the sum — summing would recommend a model no single device can hold."""
    assert H.vram_bytes(run=_runner("24576 MiB\n81920 MiB\n24576 MiB\n")) == 81920 * MIB


def test_no_nvidia_smi_returns_none():
    """CPU-only machine: the installer still runs, it just cannot recommend."""
    assert H.vram_bytes(run=_runner(raises=FileNotFoundError())) is None


def test_driver_present_but_no_device_returns_none():
    """Common in containers started without --gpus."""
    assert H.vram_bytes(run=_runner("", returncode=9)) is None


def test_unparseable_output_returns_none_rather_than_raising():
    assert H.vram_bytes(run=_runner("N/A\n")) is None


def test_timeout_is_survivable():
    """A wedged nvidia-smi must not hang the installer forever."""
    assert H.vram_bytes(
        run=_runner(raises=subprocess.TimeoutExpired("nvidia-smi", 5))) is None


def test_gpu_name_is_reported_for_the_banner():
    """The installer echoes what it detected so the user can catch a wrong-GPU decision."""
    assert H.gpu_name(run=_runner("NVIDIA RTX PRO 6000 Blackwell Server Edition\n")) \
        == "NVIDIA RTX PRO 6000 Blackwell Server Edition"


def test_gpu_name_is_none_without_a_gpu():
    assert H.gpu_name(run=_runner(raises=FileNotFoundError())) is None


# ---- system RAM (the CPU-serving budget source) ------------------------------------------
def _reader(text=None, raises=None):
    def read():
        if raises:
            raise raises
        return text
    return read


def test_ram_bytes_parses_meminfo():
    assert H.ram_bytes(read=_reader(
        "MemTotal:       32605616 kB\nMemFree:  1868688 kB\n")) == 32605616 * 1024


def test_ram_bytes_missing_file_returns_none():
    assert H.ram_bytes(read=_reader(raises=FileNotFoundError())) is None


def test_ram_bytes_garbage_returns_none():
    assert H.ram_bytes(read=_reader("not meminfo at all\n")) is None


def test_ram_bytes_missing_memtotal_returns_none():
    assert H.ram_bytes(read=_reader("MemFree: 123 kB\n")) is None
