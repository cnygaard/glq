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
