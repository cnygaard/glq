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
import re
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
    "almalinux": 'ID="almalinux"\nID_LIKE="rhel centos fedora"\nVERSION_ID="9.4"\n',
    "arch": 'ID=arch\nPRETTY_NAME="Arch Linux"\n',
    "manjaro": 'ID=manjaro\nID_LIKE=arch\n',
    "amzn": 'ID="amzn"\nVERSION_ID="2023"\nPRETTY_NAME="Amazon Linux 2023"\n',
    "azurelinux": 'ID=azurelinux\nVERSION_ID="3.0"\nPRETTY_NAME="Microsoft Azure Linux 3.0"\n',
    "mariner": 'ID=mariner\nVERSION_ID="2.0"\nPRETTY_NAME="CBL-Mariner"\n',
    "steamos": 'ID=steamos\nID_LIKE=arch\nVERSION_ID="3.5"\nPRETTY_NAME="SteamOS"\n',
    "opensuse": 'ID="opensuse-tumbleweed"\nID_LIKE="opensuse suse"\n',
    "weird": 'ID=plan9\nPRETTY_NAME="Definitely Not Linux"\n',
}


def _preflight(distro, tmp_path, extra=(), gcc_version=None, trap_awk=False):
    osr = tmp_path / f"os-release-{distro}"
    osr.write_text(OS_RELEASE[distro])
    env = {**os.environ, "GLQ_OS_RELEASE": str(osr),
           "GLQ_HOME": str(tmp_path / "glqhome")}
    if gcc_version is not None:
        # A PATH shim rather than an env override: the check reads `gcc -dumpversion`, and
        # faking the answer at that seam tests the real detection instead of a test-only
        # back door. Same spirit as fabricating /etc/os-release.
        shim = tmp_path / f"shim-gcc-{gcc_version}"
        shim.mkdir(exist_ok=True)
        (shim / "gcc").write_text(
            f'#!/bin/sh\n[ "$1" = "-dumpversion" ] && echo {gcc_version} && exit 0\nexit 0\n')
        (shim / "gcc").chmod(0o755)
        env["PATH"] = f"{shim}:{env['PATH']}"
    if trap_awk:
        # A shim that records the call and then fails exactly as a missing awk does.
        # openSUSE Tumbleweed and Photon base images ship no awk at all.
        shim = tmp_path / "shim-awk"
        shim.mkdir(exist_ok=True)
        (shim / "awk").write_text(
            f'#!/bin/sh\ntouch {tmp_path / "awk_was_called"}\n'
            f'echo "awk: command not found" >&2\nexit 127\n')
        (shim / "awk").chmod(0o755)
        env["PATH"] = f"{shim}:{env['PATH']}"
    return subprocess.run(
        ["bash", INSTALL_SH, "--preflight", *extra],
        capture_output=True, text=True, timeout=60, env=env)


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


def test_assume_no_gpu_alias_still_works(tmp_path):
    """--assume-no-gpu shipped before --cpu; scripts using it must keep working."""
    proc = _preflight("ubuntu", tmp_path, extra=["--assume-no-gpu"])
    assert proc.returncode == 0
    assert "cpu" in (proc.stdout + proc.stderr).lower()


def test_no_gpu_warning_names_the_cpu_backend(tmp_path):
    """CPU serving is real since the CPU decode work: the warning should say what the
    user GETS (the vLLM CPU backend, single-digit tok/s), not just what is missing."""
    proc = _preflight("ubuntu", tmp_path, extra=["--cpu"])
    out = proc.stdout + proc.stderr          # warn() writes to stderr
    assert "vLLM CPU backend" in out
    assert "single-digit" in out


def test_absent_gpu_is_a_warning_not_a_failure(tmp_path):
    """CPU-only is supported (dequantize-then-matmul), just slow. Pre-flight must not
    refuse to install on a laptop."""
    proc = _preflight("ubuntu", tmp_path, extra=["--cpu"])
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


# ------------------------------------------------- the python floor on RHEL 9

# Measured in containers on 2026-08-15, not assumed:
#
#   UBI9 / AlmaLinux 9 / Amazon Linux 2023   ->  default python3 is **3.9**
#   glq needs >= 3.10, so `dnf install python3-devel` installs 3.9 headers and
#   pre-flight still refuses afterwards — advice that leads nowhere.
#
#   `dnf install -y python3.12`  ->  exit 0, Python 3.12.13, venv available
#   (3.11 and 3.14 are in AppStream too; the hint just has to name >= 3.12.)
#
# The assertion is a floor rather than an exact string so that moving the hint to 3.14
# later stays green.

_PY_RE = re.compile(r"python3\.(\d+)")


def _hint_line(distro, tmp_path):
    out = _out(_preflight(distro, tmp_path))
    lines = [ln for ln in out.splitlines() if "packages on this distro" in ln]
    assert lines, f"no package hint printed for {distro}:\n{out}"
    return lines[0]


def _newest_python_named(hint):
    minors = [int(m) for m in _PY_RE.findall(hint)]
    return max(minors) if minors else None


@pytest.mark.parametrize("distro", ["rhel", "rocky", "almalinux", "amzn"])
def test_rhel_family_hint_names_python_312_or_newer(distro, tmp_path):
    """RHEL 9 and Amazon Linux 2023 default to python3.9. Unless the hint names an
    explicit >= 3.12 interpreter, following it leaves the user exactly where they started:
    pre-flight refuses again, having told them they were done."""
    hint = _hint_line(distro, tmp_path)
    newest = _newest_python_named(hint)
    assert newest is not None, f"{distro}: hint names no explicit python version: {hint}"
    assert newest >= 12, (
        f"{distro}: hint offers python3.{newest}, but the distro default is 3.9 and glq "
        f"needs >= 3.10 — name 3.12 or newer explicitly: {hint}")


@pytest.mark.parametrize("distro", ["rhel", "rocky", "almalinux", "amzn"])
def test_rhel_family_hint_does_not_ask_for_curl(distro, tmp_path):
    """These images preinstall `curl-minimal`, which *conflicts* with `curl`:

        package curl-minimal-7.76.1-40.el9 conflicts with curl provided by curl-7.76.1-40.el9

    so asking for it does not merely add nothing — it fails the whole dnf transaction and
    takes gcc down with it, leaving the user worse off than doing nothing. Measured on UBI9.
    """
    hint = _hint_line(distro, tmp_path)
    assert not re.search(r"\bcurl\b", hint), (
        f"{distro}: hint asks for curl, which conflicts with the preinstalled "
        f"curl-minimal and aborts the transaction: {hint}")


@pytest.mark.parametrize("distro", ["ubuntu", "debian", "arch", "opensuse"])
def test_other_families_still_install_curl(distro, tmp_path):
    """Regression: the curl removal is specific to the RHEL family, where curl-minimal is
    preinstalled. Everywhere else curl is genuinely needed and genuinely absent."""
    assert re.search(r"\bcurl\b", _hint_line(distro, tmp_path)), \
        f"{distro}: curl is not preinstalled here; the hint must still name it"


# --------------------------------------------------- the pacman database refresh

@pytest.mark.parametrize("distro", ["arch", "manjaro", "steamos"])
def test_arch_family_refreshes_the_package_database(distro, tmp_path):
    """`pacman -S` does not sync the package database; `-Sy` does.

    Measured in a container 2026-08-15: the archlinux image ships with no synced database,
    so a bare `pacman -S python gcc curl` cannot resolve any package name at all —

        warning: database file for 'core' does not exist (use '-Sy' to download)
        error: target not found: python

    which reads as "Arch has no python" rather than "pacman has no package list". `-Syu` is
    the form Arch documents; `-Sy` alone is a partial upgrade, which they discourage.

    The apt branch already pairs `apt-get update &&` with the install, and dnf/zypper/tdnf
    refresh implicitly — Arch is the only family where this must be explicit.
    """
    hint = _hint_line(distro, tmp_path)
    assert "-Syu" in hint, (
        f"{distro}: `pacman -S` without a database sync cannot resolve packages on a fresh "
        f"system: {hint}")


@pytest.mark.parametrize("distro", ["ubuntu", "debian", "fedora", "rhel", "almalinux",
                                    "amzn", "arch", "manjaro", "steamos", "azurelinux",
                                    "opensuse"])
def test_every_hint_is_non_interactive(distro, tmp_path):
    """Each hint must complete without a prompt.

    apt/dnf/tdnf/zypper already carry `-y`; pacman needs `--noconfirm` or it stops at

        :: Proceed with installation? [Y/n]

    which hangs anything scripted and leaves an interactive user staring at a half-finished
    install if they walked away. Measured in the arch container: the command resolved the
    packages and then stalled on the prompt.
    """
    hint = _hint_line(distro, tmp_path)
    assert (" -y" in hint or "--noconfirm" in hint), \
        f"{distro}: hint needs a non-interactive flag: {hint}"


# ----------------------------------------- the advice has to be enough to COMPILE, not just
#                                            enough to satisfy pre-flight's own checks
#
# GLQ's extension is C++ (glq_bindings.cpp) plus CUDA, and nvcc drives a C++ host compiler.
# On RPM distros the `gcc` package is the C compiler only — `cc1plus` lives in `gcc-c++`.
# Measured in fedora:43 after installing exactly what pkg_hint printed:
#
#     gcc: /usr/sbin/gcc        c++: MISSING        g++: command not found
#
# and the build died with `gcc: fatal error: cannot execute 'cc1plus'`. Debian-family hides
# this because build-essential pulls g++, which is why ubuntu never showed it.

RPM_FAMILY = ["fedora", "rhel", "almalinux", "amzn", "azurelinux", "suse"]


@pytest.mark.parametrize("distro", [d for d in RPM_FAMILY if d in OS_RELEASE])
def test_the_hint_installs_a_cxx_compiler_not_only_a_c_one(distro, tmp_path):
    """`gcc` alone cannot build glq on these distros, and pre-flight's `cc` check does not
    notice — so the user is told they are ready and then hits a compiler error."""
    hint = _hint_line(distro, tmp_path)
    assert any(pkg in hint for pkg in ("gcc-c++", "g++", "build-essential")), (
        f"{distro} hint installs no C++ compiler, so the CUDA extension cannot build: {hint!r}")


def test_preflight_checks_for_a_cxx_compiler(tmp_path):
    """A C-only toolchain satisfies `command -v gcc` and tells the user they are ready.
    The check has to ask for what the build actually needs."""
    src = open(INSTALL_SH).read()
    block = src.split("# A C compiler is needed", 1)[-1].split("# GPU", 1)[0]
    assert "c++" in block or "g++" in block, (
        "pre-flight's compiler check accepts a C-only toolchain; glq needs C++")


# ------------------------------------------- host compiler vs what CUDA will accept
#
# CUDA's crt/host_config.h refuses a host gcc newer than 15. fedora:44 ships gcc 16, so a
# source build there dies with "unsupported GNU version" on every .cu file. Measured against
# both the real toolkit (13.3.1) and the pip nvidia/cu13 headers — installing the full 4.1 GB
# toolkit does NOT raise the cap.
#
# The fix is a compat compiler plus NVCC_CCBIN, which was verified to remove the error
# entirely: 6 "unsupported GNU version" errors with the default gcc 16, 0 with
# NVCC_CCBIN=/usr/bin/g++-15.
#
# It must stay a NOTE. Since 0.8.6 the prebuilt wheels cover cp310-cp314, so a fedora:44 user
# compiles nothing and is unaffected; blocking them would refuse an install that works.

def test_a_too_new_gcc_is_reported(tmp_path):
    out = _out(_preflight("fedora", tmp_path, gcc_version=16))
    assert "16" in out
    assert "NVCC_CCBIN" in out, "the fix has to be named, not just the problem"


def test_the_compat_compiler_package_is_named_for_the_distro(tmp_path):
    """Measured on fedora:44: `dnf install gcc15 gcc15-c++` provides /usr/bin/g++-15."""
    out = _out(_preflight("fedora", tmp_path, gcc_version=16))
    assert "gcc15" in out


def test_a_supported_gcc_says_nothing(tmp_path):
    """No advice when there is nothing to fix — pre-flight output is read, so noise costs."""
    out = _out(_preflight("fedora", tmp_path, gcc_version=15))
    assert "NVCC_CCBIN" not in out


def test_it_never_blocks_the_install(tmp_path):
    """The prebuilt wheels need no compiler at all. A gcc too new for nvcc must not stop an
    install that will never invoke nvcc."""
    proc = _preflight("fedora", tmp_path, gcc_version=16)
    assert proc.returncode == 0, (
        "a too-new gcc is a note, not a blocker — the wheel path is unaffected\n"
        + _out(proc))


# ------------------------------------------------------- Azure Linux needs the libc headers
#
# Found by the distro matrix, and it is the failure mode this suite exists for: pre-flight
# followed its own advice, reported OK, and the JIT build then died with
#
#     nvidia/cu13/include/crt/host_config.h:208: fatal error: features.h: No such file
#
# 16 times in one run. `features.h` is glibc's, not CUDA's — on the minimal Azure Linux core
# image `gcc-c++` does not pull `glibc-devel`, whereas Fedora/RHEL's does and Debian's
# build-essential brings libc6-dev. Verified in the container: `tdnf install -y glibc-devel`
# succeeds and produces /usr/include/features.h.

def test_azure_linux_is_told_to_install_the_libc_headers(tmp_path):
    out = _out(_preflight("azurelinux", tmp_path))
    assert "glibc-devel" in out, (
        "azurelinux's gcc-c++ does not pull the libc headers, so the advice must name them; "
        "without it pre-flight passes and the kernel build fails later")


def test_mariner_gets_the_same_advice(tmp_path):
    """CBL-Mariner is Azure Linux's previous name and shares the package set."""
    assert "glibc-devel" in _out(_preflight("mariner", tmp_path))


def test_the_other_rpm_distros_are_not_changed(tmp_path):
    """Fedora and RHEL pull glibc-devel via gcc-c++, so adding it there would be noise in a
    command the user is asked to paste."""
    for distro in ("fedora", "rhel"):
        assert "glibc-devel" not in _out(_preflight(distro, tmp_path))


def test_preflight_computes_free_disk_without_awk(tmp_path):
    """Measured in the distro matrix (2026-09-02): openSUSE Tumbleweed and Photon ship no
    awk, so the disk check printed a raw `awk: command not found` into the user's terminal
    and then silently skipped its own gate — the free-space warning could never fire on
    exactly the minimal images most likely to be short of space."""
    proc = _preflight("ubuntu", tmp_path, trap_awk=True)
    assert not (tmp_path / "awk_was_called").exists(), \
        "pre-flight still shells out to awk; minimal images do not have it"
    assert "command not found" not in proc.stdout + proc.stderr
    assert re.search(r"disk:\s+\d+ GB free", proc.stdout), proc.stdout


# ---- `which` is a vLLM runtime dependency on RPM-family images ---------------------------
#
# Verified in vLLM's own source, not inferred: vllm/third_party/deep_gemm/__init__.py's
# _find_cuda_home() runs `subprocess.check_output(['which', 'nvcc'])`. A missing binary is
# caught there, but the fallback is /usr/local/cuda — which does not exist when the CUDA
# toolchain came from pip wheels, as it does for every prebuilt-wheel install — and the
# function then trips `assert cuda_home is not None`, taking the engine core with it.
#
# Nothing in install.sh or glq calls the binary (they use the `command -v` builtin). It is
# present on most full installs — Debian-family ships it in Essential debianutils — but
# minimal/container images omit it and Fedora dropped it from default installs, which is
# why the distro harness has to install it on every dnf/tdnf/zypper image and never on
# Debian-family.

@pytest.mark.parametrize("distro", ["fedora", "rhel", "rocky", "almalinux", "amzn",
                                    "azurelinux", "mariner", "opensuse"])
def test_rpm_family_hint_names_which(distro, tmp_path):
    proc = _preflight(distro, tmp_path)
    hint = [ln for ln in proc.stdout.splitlines() if "packages on this distro" in ln]
    assert hint, proc.stdout
    assert "which" in hint[0], (
        f"{distro}: the hint omits `which`, so vLLM's CUDA_HOME lookup fails and the "
        f"engine core asserts after an install that reported success:\n{hint[0]}")


def test_debian_family_hint_does_not_name_which(tmp_path):
    """debianutils is Essential there — asking for it would be noise in the one command
    a new user copy-pastes."""
    proc = _preflight("ubuntu", tmp_path)
    hint = [ln for ln in proc.stdout.splitlines() if "packages on this distro" in ln][0]
    assert "which" not in hint, hint
