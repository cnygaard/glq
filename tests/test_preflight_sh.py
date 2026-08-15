"""Pre-flight checks in install.sh, across distributions.

A `curl … | bash` installer that discovers a missing prerequisite *after* pulling ~3 GB of
PyTorch has already wasted the user's time. Pre-flight runs first, and — because the fix is
always "install a package" — it must name the *right* command for the distro in front of it,
not assume apt.

Distro coverage is driven from `/etc/os-release`, whose path is overridable via
`GLQ_OS_RELEASE` (the same configuration seam as `GLQ_HOME`). That is what lets these tests
exercise Fedora, Arch, RHEL, Amazon Linux, Azure Linux and SteamOS from one Ubuntu box —
with the honest caveat that this proves the *mapping*, not that GLQ runs on those distros.

SteamOS gets its own case for two reasons that would otherwise produce bad advice: its root
filesystem is read-only, so a bare `pacman -S` fails; and Steam Deck hardware is AMD, so
there is no CUDA and GLQ runs its CPU fallback.
"""
from __future__ import annotations

import os
import subprocess

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
INSTALL_SH = os.path.join(ROOT, "install.sh")

# Real /etc/os-release excerpts.
OS_RELEASE = {
    "ubuntu": 'ID=ubuntu\nID_LIKE=debian\nVERSION_ID="24.04"\nPRETTY_NAME="Ubuntu 24.04 LTS"\n',
    "debian": 'ID=debian\nVERSION_ID="12"\nPRETTY_NAME="Debian GNU/Linux 12"\n',
    "fedora": 'ID=fedora\nVERSION_ID=44\nPRETTY_NAME="Fedora Linux 44"\n',
    "rhel": 'ID="rhel"\nID_LIKE="fedora"\nVERSION_ID="9.4"\nPRETTY_NAME="Red Hat Enterprise Linux 9.4"\n',
    "rocky": 'ID="rocky"\nID_LIKE="rhel centos fedora"\nVERSION_ID="9.4"\n',
    "arch": 'ID=arch\nPRETTY_NAME="Arch Linux"\n',
    "manjaro": 'ID=manjaro\nID_LIKE=arch\n',
    "amzn": 'ID="amzn"\nVERSION_ID="2023"\nPRETTY_NAME="Amazon Linux 2023"\n',
    "azurelinux": 'ID=azurelinux\nVERSION_ID="3.0"\nPRETTY_NAME="Microsoft Azure Linux 3.0"\n',
    "mariner": 'ID=mariner\nVERSION_ID="2.0"\nPRETTY_NAME="CBL-Mariner"\n',
    "steamos": 'ID=steamos\nID_LIKE=arch\nVERSION_ID="3.5"\nPRETTY_NAME="SteamOS"\n',
    "opensuse": 'ID="opensuse-tumbleweed"\nID_LIKE="opensuse suse"\n',
    "weird": 'ID=plan9\nPRETTY_NAME="Definitely Not Linux"\n',
}


def _preflight(distro, tmp_path, extra=()):
    osr = tmp_path / f"os-release-{distro}"
    osr.write_text(OS_RELEASE[distro])
    return subprocess.run(
        ["bash", INSTALL_SH, "--preflight", *extra],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "GLQ_OS_RELEASE": str(osr),
             "GLQ_HOME": str(tmp_path / "glqhome")})


def _out(proc):
    return proc.stdout + proc.stderr


# ------------------------------------------------------------ package manager

@pytest.mark.parametrize("distro,expect", [
    ("ubuntu", "apt-get"), ("debian", "apt-get"),
    ("fedora", "dnf"), ("rhel", "dnf"), ("rocky", "dnf"), ("amzn", "dnf"),
    ("arch", "pacman"), ("manjaro", "pacman"),
    ("azurelinux", "tdnf"), ("mariner", "tdnf"),
    ("opensuse", "zypper"),
])
def test_the_remediation_command_matches_the_distro(distro, expect, tmp_path):
    """Telling a Fedora user to run apt-get is worse than saying nothing — it reads as
    'this tool was not written for you'."""
    out = _out(_preflight(distro, tmp_path))
    assert expect in out, f"{distro} should be advised to use {expect}"


def test_derivatives_resolve_via_id_like(tmp_path):
    """Rocky/Alma/CentOS set ID to themselves and ID_LIKE to rhel; matching only on ID
    would drop every derivative into the unknown bucket."""
    assert "dnf" in _out(_preflight("rocky", tmp_path))
    assert "pacman" in _out(_preflight("manjaro", tmp_path))


def test_an_unknown_distro_still_runs_and_says_so(tmp_path):
    """Never hard-fail on an unrecognised distro — the prerequisites may well be present.
    Say the distro is unknown and name the packages generically."""
    proc = _preflight("weird", tmp_path)
    out = _out(proc)
    assert "python" in out.lower()
    assert proc.returncode == 0 or "unknown" in out.lower()


# -------------------------------------------------------------------- SteamOS

def test_steamos_warns_about_the_read_only_root(tmp_path):
    """SteamOS ships an immutable rootfs: a bare `pacman -S` fails with a read-only error,
    which looks like a broken installer rather than a distro property."""
    out = _out(_preflight("steamos", tmp_path)).lower()
    assert "read-only" in out or "readonly" in out
    assert "steamos-readonly" in out or "distrobox" in out


def test_steamos_is_identified_by_name(tmp_path):
    assert "steamos" in _out(_preflight("steamos", tmp_path)).lower()


# ------------------------------------------------------------------- contents

def test_it_reports_python_and_its_version(tmp_path):
    out = _out(_preflight("ubuntu", tmp_path))
    assert "python" in out.lower()
    assert "3." in out


def test_it_reports_gpu_state(tmp_path):
    """Either the driver/CUDA it found, or that it found none — silence would leave the
    user guessing whether the fast path is available."""
    out = _out(_preflight("ubuntu", tmp_path)).lower()
    assert "gpu" in out or "nvidia" in out or "cuda" in out


def test_absent_gpu_is_a_warning_not_a_failure(tmp_path):
    """CPU-only is supported (dequantize-then-matmul), just slow. Pre-flight must not
    refuse to install on a laptop."""
    proc = _preflight("ubuntu", tmp_path, extra=["--assume-no-gpu"])
    assert proc.returncode == 0
    assert "cpu" in _out(proc).lower()


def test_it_reports_free_disk_because_the_stack_is_large(tmp_path):
    """torch + vLLM + one checkpoint is tens of GB; running out midway leaves a
    half-installed venv."""
    out = _out(_preflight("ubuntu", tmp_path)).lower()
    assert "disk" in out or "free" in out or "gb" in out


def test_preflight_changes_nothing(tmp_path):
    """It is a read-only inspection — safe to tell a nervous user to run it first."""
    home = tmp_path / "glqhome"
    _preflight("ubuntu", tmp_path)
    assert not home.exists()


def test_preflight_is_documented_in_help():
    proc = subprocess.run(["bash", INSTALL_SH, "--help"],
                          capture_output=True, text=True, timeout=30)
    assert "--preflight" in proc.stdout
