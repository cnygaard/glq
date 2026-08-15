"""Safety properties of install.sh.

This script is published to be run as `curl … | bash`, which is the least forgiving way to
ship code: whatever bytes arrive get executed, by a user who has not read them. The tests
here pin the properties that make that survivable. They are cheap and they guard against a
future edit quietly removing one.

Behaviour is checked by running the script (with `--dry-run`, which touches nothing) rather
than by grepping for strings wherever possible — a grep passes just as happily on a
commented-out line.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
INSTALL_SH = os.path.join(ROOT, "install.sh")


def _run(args, env=None, timeout=60):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run(["bash", INSTALL_SH, *args], capture_output=True, text=True,
                          timeout=timeout, env=e)


def test_it_exists_and_is_executable():
    assert os.path.isfile(INSTALL_SH)
    assert os.access(INSTALL_SH, os.X_OK), "must be chmod +x for a git-cloned run"


def test_syntax_is_valid():
    """`bash -n` catches the class of typo that would otherwise only surface on a user's
    machine, halfway through an install."""
    proc = subprocess.run(["bash", "-n", INSTALL_SH], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_body_is_wrapped_in_main_invoked_on_the_last_line():
    """The property that makes `curl | bash` safe against a dropped connection: a truncated
    download defines functions and does nothing, instead of running half an install."""
    lines = [ln.strip() for ln in open(INSTALL_SH).read().splitlines() if ln.strip()]
    assert lines[-1] == 'main "$@"'


def test_strict_mode_is_on():
    src = open(INSTALL_SH).read()
    assert "set -euo pipefail" in src


def test_every_curl_invocation_pins_https_and_tls():
    """Any fetch the installer makes must refuse a cleartext downgrade.

    `curl -fsSL` follows a redirect to http:// silently; `--proto '=https'` makes that fail
    instead. Scans the whole shipped installer, because install.sh itself fetches nothing —
    pip does the downloading — and the only real fetch is the nvm bootstrap in
    glq/installer/__main__.py. An earlier version asserted these flags lived in install.sh,
    where they sat in an unused variable; shellcheck SC2034 caught that the test was pinning
    dead code.

    "Invocation" means `curl` followed by a flag. A bare `curl` is a package name in a
    pkg_hint line or `command -v curl`, neither of which fetches anything.
    """
    invoke = re.compile(r"\bcurl\s+-")
    loopback = re.compile(r"127\.0\.0\.1|localhost")
    checked = 0
    for path in (INSTALL_SH, os.path.join(ROOT, "glq", "installer", "__main__.py")):
        for line in open(path).read().splitlines():
            if line.strip().startswith("#") or not invoke.search(line):
                continue
            # Loopback is exempt, and deliberately: the printed "is it up?" check hits the
            # user's own vLLM on http://127.0.0.1, where plaintext is correct and https
            # would simply fail. The downgrade risk this guards only exists off-host.
            if loopback.search(line):
                continue
            checked += 1
            assert "--proto '=https'" in line and "--tlsv1.2" in line, (
                f"unpinned curl in {os.path.basename(path)}: {line.strip()}")
    assert checked, "expected at least one curl invocation (the nvm bootstrap)"


def test_umask_is_restrictive():
    assert "umask 077" in open(INSTALL_SH).read()


def test_help_exits_zero_and_documents_the_flags():
    proc = _run(["--help"])
    assert proc.returncode == 0
    for flag in ("--components", "--model", "--chat", "--dry-run", "--allow-root"):
        assert flag in proc.stdout


def test_dry_run_creates_nothing(tmp_path):
    """The flag has to be trustworthy — it is what a cautious user reaches for before
    letting the real thing touch their machine."""
    home = tmp_path / "glqhome"
    proc = _run(["--dry-run", "--yes"], env={"GLQ_HOME": str(home)})
    assert proc.returncode == 0, proc.stderr
    assert not home.exists(), "dry-run must not create the install directory"


def test_dry_run_prints_the_commands_it_would_have_run(tmp_path):
    proc = _run(["--dry-run", "--yes"], env={"GLQ_HOME": str(tmp_path / "h")})
    assert "[dry-run]" in proc.stdout
    assert "venv" in proc.stdout and "pip" in proc.stdout


def test_each_printed_command_is_on_one_line(tmp_path):
    """`--dry-run` exists so a cautious user can read what will happen. That only works if
    a command reads as a command.

    install.sh runs under the strict-mode idiom `IFS=$'\\n\\t'`, which makes `"$*"` join
    with newlines — so `run()` printed one argument per line:

          [dry-run] /home/u/.glq/venv/bin/pip
        install
        --upgrade
        glq

    Unreadable, and it silently defeats any test that greps for a command as a phrase.
    """
    proc = _run(["--dry-run", "--yes"], env={"GLQ_HOME": str(tmp_path / "h")})
    assert "pip install" in proc.stdout, (
        "arguments are being split across lines; commands are unreadable:\n"
        + proc.stdout)


@pytest.mark.skipif(shutil.which("unshare") is None, reason="unshare(1) not available")
def test_it_refuses_to_run_as_root(tmp_path):
    """Genuinely runs as uid 0 via a user namespace rather than faking $EUID (which bash
    makes readonly). The guard must fire and must name the escape hatch, or a user who
    needs root has no way forward."""
    probe = subprocess.run(["unshare", "-r", "id", "-u"], capture_output=True, text=True)
    if probe.returncode != 0 or probe.stdout.strip() != "0":
        pytest.skip("user namespaces unavailable in this environment")

    proc = subprocess.run(
        ["unshare", "-r", "bash", INSTALL_SH, "--yes"],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "GLQ_HOME": str(tmp_path / "h")})
    assert proc.returncode != 0, "running as root must fail"
    assert "root" in proc.stderr.lower()
    assert "--allow-root" in proc.stderr
    assert not (tmp_path / "h").exists(), "must refuse before creating anything"


@pytest.mark.skipif(shutil.which("unshare") is None, reason="unshare(1) not available")
def test_preflight_works_as_root(tmp_path):
    """Diagnosis must not be gated on the root check.

    Pre-flight is read-only, and the people most likely to run it as root are exactly the
    ones who need it: anyone inside a container (Docker defaults to uid 0). Refusing to even
    *report* what is missing, because of a guard that exists to protect the install, is
    unhelpful — and it makes the whole cross-distro matrix untestable, since every container
    runs as root."""
    probe = subprocess.run(["unshare", "-r", "id", "-u"], capture_output=True, text=True)
    if probe.returncode != 0 or probe.stdout.strip() != "0":
        pytest.skip("user namespaces unavailable in this environment")

    proc = subprocess.run(
        ["unshare", "-r", "bash", INSTALL_SH, "--preflight"],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "GLQ_HOME": str(tmp_path / "h")})
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"pre-flight should run as root, got {proc.returncode}: {out}"
    assert "Pre-flight checks" in out
    assert not (tmp_path / "h").exists(), "still read-only"


@pytest.mark.skipif(shutil.which("unshare") is None, reason="unshare(1) not available")
def test_allow_root_is_an_effective_escape_hatch(tmp_path):
    """The flag has to actually work — a guard with a broken override is just a wall."""
    probe = subprocess.run(["unshare", "-r", "id", "-u"], capture_output=True, text=True)
    if probe.returncode != 0 or probe.stdout.strip() != "0":
        pytest.skip("user namespaces unavailable in this environment")

    proc = subprocess.run(
        ["unshare", "-r", "bash", INSTALL_SH, "--yes", "--allow-root", "--dry-run"],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "GLQ_HOME": str(tmp_path / "h")})
    assert proc.returncode == 0, proc.stderr
    assert "refusing to run as root" not in proc.stderr


def test_no_sudo_is_ever_invoked(tmp_path):
    """The script prints the package command for your distro and stops; it never escalates
    privileges on your behalf.

    Checked by execution rather than by grep: pre-flight legitimately *prints* strings like
    `sudo pacman -S ...` as advice for the detected distro, so a textual search reports
    false positives. Here a tripwire `sudo` goes first on PATH — if the script ever really
    runs it, the marker file appears."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    marker = tmp_path / "sudo-was-called"
    sudo = bindir / "sudo"
    sudo.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 1\n')
    sudo.chmod(0o755)

    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}",
           "GLQ_HOME": str(tmp_path / "h")}
    for args in (["--preflight"], ["--dry-run", "--yes"]):
        subprocess.run(["bash", INSTALL_SH, *args], capture_output=True, text=True,
                       timeout=60, env=env)
        assert not marker.exists(), f"install.sh invoked sudo with args {args}"


def test_glq_version_flag_requires_a_value(tmp_path):
    proc = _run(["--glq-version"], env={"GLQ_HOME": str(tmp_path / "h")})
    assert proc.returncode != 0
    assert "needs a value" in proc.stderr


# ------------------------------------------- installing something other than a PyPI release

def test_glq_source_installs_the_given_spec_instead_of_pypi(tmp_path):
    """`--glq-source` hands pip an arbitrary spec: a wheel, a path, a VCS ref.

    install.sh is a *bootstrap* — it makes the venv, installs glq from PyPI, and hands over
    to `python -m glq.installer`. So an install can never be newer than the last release,
    and a fork, a release candidate or an unmerged branch cannot be installed by its own
    installer at all.

    That is not hypothetical. The cross-distro matrix mounts this branch and runs its
    install.sh, which dutifully fetched glq 0.8.3 from PyPI — a release predating
    `glq.installer` — and died with `No module named glq.installer`. The same wall blocks
    validating any installer fix in a container: the container would install the released
    glq, not the one under test.
    """
    proc = _run(["--dry-run", "--yes", "--glq-source", "/wheels/glq-9.9.9.whl"],
                env={"GLQ_HOME": str(tmp_path / "h")})
    assert proc.returncode == 0, proc.stderr

    # Match the *command*, not merely the string anywhere in the output — the spec also
    # appears in the passthrough line, and pytest's tmp dir is named after the test, so a
    # loose `"install" in line` check matches the venv path and passes for the wrong
    # reason. (It did, on the first run of this test.)
    installs = [ln for ln in proc.stdout.splitlines() if "pip install --upgrade" in ln]
    assert installs, f"no pip install command printed:\n{proc.stdout}"

    glq_installs = [ln for ln in installs if ln.rstrip().split()[-1] != "pip"]
    assert glq_installs, f"no glq install command printed:\n{proc.stdout}"
    assert all("/wheels/glq-9.9.9.whl" in ln for ln in glq_installs), (
        f"still installing glq from PyPI despite --glq-source: {glq_installs}")


def test_core_installs_the_build_toolchain_glq_compiles_with(tmp_path):
    """GLQ JIT-compiles its CUDA kernels on first use, so the toolchain is a hard runtime
    requirement — not an optional extra.

    Measured in an ubuntu:24.04 container with a GPU attached, each of these was discovered
    only by removing the previous one (2026-08-15):

        ninja                  RuntimeError: Ninja is required to load C++ extensions
        cuda-toolkit[nvcc]     no nvcc at all -> nothing to compile with
        cuda-toolkit[cccl]     fatal error: nv/target: No such file or directory

    Without them `curl … | bash` produces an install that looks healthy — glq imports, the
    plugin registers, torch sees the GPU — and dies on the first forward pass.

    They go in the *same* pip invocation as glq deliberately. torch already constrains
    `cuda-toolkit==<major.minor>`; resolving in one transaction lets pip apply the extras to
    that pin instead of us hardcoding a CUDA version that rots on every torch release.
    """
    proc = _run(["--dry-run", "--yes"], env={"GLQ_HOME": str(tmp_path / "h")})
    assert proc.returncode == 0, proc.stderr

    installs = [ln for ln in proc.stdout.splitlines()
                if "pip install --upgrade" in ln and ln.rstrip().split()[-1] != "pip"]
    assert installs, f"no glq install command printed:\n{proc.stdout}"
    line = installs[0]

    # ninja is safe to resolve alongside glq: it constrains nothing.
    assert "ninja" in line, f"ninja not installed; the JIT build cannot run: {line}"
    assert "glq" in line

    # The CUDA compiler/headers must still be installed, just not here — see below.
    assert "cuda-toolkit" in proc.stdout, (
        f"no CUDA compiler/headers installed at all; the kernels cannot build:\n{proc.stdout}")


def test_the_cuda_toolchain_is_resolved_apart_from_torch(tmp_path):
    """`cuda-toolkit[nvcc,cccl]` must NOT share a pip transaction with glq/torch.

    Measured in a container, 2026-08-15 — resolving them together does not, as one might
    hope, apply the extras to torch's own `cuda-toolkit==13.0.3` pin. pip instead picks
    cuda-toolkit 13.3.1, finds it incompatible, backtracks, and **silently downgrades torch
    2.13.0 -> 2.10.0**:

        glq-0.8.4  ninja-1.13.0  torch-2.10.0  cuda-toolkit-13.3.1  cuda-toolkit-13.0.3.0

    torch 2.10 then cannot resolve CUDA_HOME from the venv wheels, so the build fails
    earlier than before. A silent major-dependency downgrade is worse than the conflict it
    was meant to avoid; installing the extras separately, at the version already present,
    cannot move torch at all.
    """
    proc = _run(["--dry-run", "--yes"], env={"GLQ_HOME": str(tmp_path / "h")})
    for line in proc.stdout.splitlines():
        if "cuda-toolkit" in line and "glq" in line:
            raise AssertionError(
                f"cuda-toolkit shares a transaction with glq; pip will backtrack torch: {line}")


def test_no_cuda_version_is_hardcoded_in_the_installer():
    """The version comes from whatever torch pinned, read at install time.

    A literal would have to be updated in lockstep with every torch release, across every
    distro the installer supports — the maintenance burden this design exists to avoid.
    """
    src = open(INSTALL_SH).read()
    hardcoded = re.findall(r"cuda-toolkit\[[^\]]*\]==[0-9]", src)
    assert not hardcoded, f"hardcoded CUDA version in install.sh: {hardcoded}"


def test_glq_source_requires_a_value(tmp_path):
    proc = _run(["--glq-source"], env={"GLQ_HOME": str(tmp_path / "h")})
    assert proc.returncode != 0
    assert "needs a value" in proc.stderr


def test_glq_source_and_glq_version_are_mutually_exclusive(tmp_path):
    """Both decide which glq gets installed. Letting one silently win installs a version
    the user did not ask for — precisely the confusion this flag exists to remove."""
    proc = _run(["--dry-run", "--yes", "--glq-source", "/w/x.whl", "--glq-version", "0.8.4"],
                env={"GLQ_HOME": str(tmp_path / "h")})
    assert proc.returncode != 0
    assert "--glq-source" in proc.stderr and "--glq-version" in proc.stderr


def test_glq_source_is_documented_in_help():
    assert "--glq-source" in _run(["--help"]).stdout


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shellcheck_is_clean():
    proc = subprocess.run(["shellcheck", "-S", "warning", INSTALL_SH],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout


def test_readme_advertises_the_raw_url():
    """The one-liner users copy must point at a URL that actually serves the file —
    github.com/<org>/<repo>/install.sh is a 404."""
    readme = open(os.path.join(ROOT, "README.md")).read()
    assert "raw.githubusercontent.com/cnygaard/glq/main/install.sh" in readme
