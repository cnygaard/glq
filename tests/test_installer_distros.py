"""install.sh across Linux distributions, in Docker, with a real GPU.

`tests/test_preflight_sh.py` proves the *mapping* — that a Fedora `/etc/os-release` produces
a `dnf` line rather than an `apt-get` one — using fabricated files. It cannot tell you
whether `python3-devel gcc curl` is the *correct* package set on Fedora, or whether the
installer works once you have run it. Only Ubuntu 24.04 has ever actually been executed.

That gap matters because pre-flight makes a promise: *run this command and you will be
ready*. This suite tests the promise, per distro:

    pre-flight on a pristine image  ->  run the command it printed  ->  pre-flight passes

then goes on to a real venv, a real `pip install glq`, and the self-check.

**Run with `--gpus all`**, deliberately. The NVIDIA container toolkit injects `nvidia-smi`
and the driver libraries into the image, and that depends on the image's loader and
ldconfig layout — so Arch and Azure Linux are genuine questions, not repeats of Ubuntu.
It also means pre-flight's GPU branch and its `CUDA Version:` parse get exercised on every
distro rather than on one.

Cost: a full run is hours and tens of GB. Marked `slow` (the marker already declared in
pyproject.toml), so ordinary runs deselect it with `-m "not slow"`.

**SteamOS is absent and cannot be added.** No image exists in either Docker Hub namespace,
and Steam Deck hardware is AMD, so neither its packages nor its GPU branch could be
validated on an NVIDIA host. It stays covered by the unit-test mapping only.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PIP_CACHE = os.environ.get("GLQ_DISTRO_PIP_CACHE", "/opt/dlami/nvme/pipcache")
HF_CACHE = os.environ.get("GLQ_DISTRO_HF_CACHE", "/opt/dlami/nvme/hf_cache_distro")

#: Smallest trellis checkpoint (1.8 GiB) — the format the recommender prefers, and quick
#: enough to pull once and reuse from the shared cache across all nine containers.
SMOKE_MODEL = "xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel"

pytestmark = pytest.mark.slow


# --------------------------------------------------------------------- gating

def _docker_ok() -> tuple[bool, str]:
    if shutil.which("docker") is None:
        return False, "docker not installed"
    if subprocess.run(["docker", "ps"], capture_output=True).returncode != 0:
        return False, "docker not usable by this user (group/permissions)"
    probe = subprocess.run(
        ["docker", "run", "--rm", "--gpus", "all", "ubuntu:24.04",
         "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=600)
    if probe.returncode != 0:
        return False, f"--gpus all unavailable: {probe.stderr.strip()[:120]}"
    return True, ""


_OK, _WHY = _docker_ok()
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not _OK, reason=_WHY or "docker/GPU")]


# --------------------------------------------------------------------- matrix

class Distro:
    def __init__(self, name, image, expect_pkg):
        self.name, self.image, self.expect_pkg = name, image, expect_pkg

    def __repr__(self):
        return self.name


DISTROS = [
    Distro("ubuntu",     "ubuntu:24.04",                                   "apt-get"),
    Distro("debian",     "debian:12",                                      "apt-get"),
    Distro("fedora",     "fedora:44",                                      "dnf"),
    Distro("ubi9",       "registry.access.redhat.com/ubi9/ubi:latest",     "dnf"),
    Distro("almalinux",  "almalinux:9",                                    "dnf"),
    Distro("amazon2023", "amazonlinux:2023",                               "dnf"),
    Distro("arch",       "archlinux:latest",                               "pacman"),
    Distro("azurelinux", "mcr.microsoft.com/azurelinux/base/core:3.0",     "tdnf"),
    Distro("opensuse",   "opensuse/tumbleweed",                            "zypper"),
]


def _sh(distro: Distro, script: str, timeout=3600) -> subprocess.CompletedProcess:
    """Run a shell script inside a fresh container for `distro`.

    The repo is mounted read-only at /glq so the container runs *this branch's* install.sh.
    The published `curl | bash` one-liner cannot be exercised until the script is on main.
    """
    for d in (PIP_CACHE, HF_CACHE):
        os.makedirs(d, exist_ok=True)
        os.chmod(d, 0o777)              # containers install as several different uids
    return subprocess.run(
        # --shm-size: vLLM needs far more than docker's 64 MB default and fails obscurely
        # without it. The torch_extensions cache is deliberately NOT shared — the CUDA
        # extension's JIT build against each distro's toolchain is part of what we test.
        ["docker", "run", "--rm", "--gpus", "all", "--shm-size=8g",
         "-v", f"{ROOT}:/glq:ro",
         "-v", f"{PIP_CACHE}:/pipcache",
         "-v", f"{HF_CACHE}:/hf",
         "-e", "PIP_CACHE_DIR=/pipcache",
         "-e", "HF_HOME=/hf",
         "-e", "GLQ_HOME=/root/.glq",
         distro.image, "bash", "-c", script],
        capture_output=True, text=True, timeout=timeout)


def _out(p) -> str:
    return p.stdout + p.stderr


# ------------------------------------------------------------- the advice loop

@pytest.mark.parametrize("distro", DISTROS, ids=lambda d: d.name)
def test_preflight_advice_is_correct_and_sufficient(distro):
    """The claim pre-flight makes, end to end, on a pristine image.

    Stage A records what it says is missing. Stage B runs the command it printed, verbatim
    and unedited — if that command is wrong for the distro, this is where it shows. Stage C
    demands pre-flight then passes: advice that does not lead to a working state is worse
    than no advice, because the user believes they are done.
    """
    script = r"""
set -u
echo "===A pristine pre-flight"
bash /glq/install.sh --preflight; echo "A_EXIT=$?"

echo "===HINT"
bash /glq/install.sh --preflight 2>&1 | sed -n 's/^  packages on this distro: //p' | head -1

echo "===B apply the advice"
CMD=$(bash /glq/install.sh --preflight 2>&1 | sed -n 's/^  packages on this distro: //p' | head -1)
# The hint is written for a normal user, so it says sudo; in a container we are already root
# and sudo usually is not installed. Stripping it changes nothing about which packages the
# advice names, which is what is under test here.
CMD=$(printf '%s' "$CMD" | sed 's/sudo //g')
echo "running: $CMD"
bash -c "$CMD" >/tmp/pkg.log 2>&1; echo "B_EXIT=$?"
tail -5 /tmp/pkg.log

echo "===C pre-flight after"
bash /glq/install.sh --preflight; echo "C_EXIT=$?"
"""
    proc = _sh(distro, script, timeout=2400)
    out = _out(proc)

    assert distro.expect_pkg in out, f"{distro.name}: wrong package manager advised\n{out[-2000:]}"
    assert "C_EXIT=0" in out, (
        f"{distro.name}: pre-flight still fails after following its own advice.\n"
        f"This is the bug this suite exists to find — the hint names the wrong packages, "
        f"or the distro's default python is older than 3.10.\n{out[-3000:]}")


@pytest.mark.parametrize("distro", DISTROS, ids=lambda d: d.name)
def test_gpu_is_visible_inside_the_container(distro):
    """The container toolkit injects nvidia-smi and the driver libs; whether that lands
    depends on the image's loader/ldconfig, so it is a per-distro question. Also exercises
    pre-flight's `CUDA Version:` parse against every base image."""
    proc = _sh(distro, "bash /glq/install.sh --preflight", timeout=900)
    out = _out(proc)
    assert "gpu:" in out
    assert "no NVIDIA GPU detected" not in out, f"{distro.name}: driver injection failed\n{out[-1500:]}"
    assert "cuda:" in out
    cuda_line = [ln for ln in out.splitlines() if "cuda:" in ln][0]
    assert any(ch.isdigit() for ch in cuda_line), f"CUDA version not parsed: {cuda_line!r}"


# ------------------------------------------------------------ the real install

@pytest.mark.parametrize("distro", DISTROS, ids=lambda d: d.name)
def test_full_install_and_coherent_generation(distro):
    """venv + glq + vLLM + the self-check + **a real token from a real checkpoint**.

    The generation stage is the one that can fail where everything else passes. GLQ's fused
    kernel is a CUDA extension **JIT-built on first use against whatever toolchain the
    distro ships** — so a container can install cleanly, register the plugin, report a GPU,
    and still produce noise if that build went wrong. Nothing short of decoding real tokens
    catches it, which is why the torch_extensions cache is deliberately not shared between
    containers: the per-distro build *is* the thing under test.

    "Coherent" has to be mechanical to be a test, so it is a factual completion at
    temperature 0: a working stack answers "The capital of France is" with Paris, and a
    miscompiled kernel does not.

    Non-root for the install, because that is the path a real user takes and it leaves the
    root guard free to be asserted separately.
    """
    script = r"""
set -u
CMD=$(bash /glq/install.sh --preflight 2>&1 | sed -n 's/^  packages on this distro: //p' | head -1)
bash -c "$(printf '%s' "$CMD" | sed 's/sudo //g')" >/tmp/pkg.log 2>&1

# `useradd` and `su` are the TEST's prerequisites, not glq's, so they are installed
# separately from the pre-flight set above — which must stay exactly what `pkg_hint` printed,
# or the claim that pre-flight's advice is sufficient stops being testable.
#
# Minimal RPM images ship neither. Measured on fedora:44: every stage returned 127 with
# `su: command not found`, and the `|| true` that used to sit on the useradd line swallowed
# the real cause, so a missing harness dependency read as a glq failure three stages later.
if ! command -v useradd >/dev/null 2>&1 || ! command -v su >/dev/null 2>&1; then
    {   if   command -v dnf     >/dev/null 2>&1; then dnf install -y shadow-utils util-linux
        elif command -v apt-get >/dev/null 2>&1; then apt-get install -y passwd login
        elif command -v pacman  >/dev/null 2>&1; then pacman -Sy --noconfirm shadow util-linux
        elif command -v tdnf    >/dev/null 2>&1; then tdnf install -y shadow-utils util-linux
        elif command -v zypper  >/dev/null 2>&1; then zypper -n install shadow util-linux
        fi
    } >/tmp/harness_prereq.log 2>&1 || true
fi
for tool in useradd su; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "HARNESS_PREREQ_MISSING=$tool"; tail -5 /tmp/harness_prereq.log; exit 90; }
done

# -s /bin/bash explicitly: useradd inherits the distro default, which is dash on
# Debian-family, and install.sh is a bash script (arrays, `local`). Everything below runs
# under bash, never sh. No `|| true`: a user we cannot create is a harness failure to report,
# not one to continue past.
id tester >/dev/null 2>&1 || useradd -m -s /bin/bash tester
chmod 0777 /pipcache /hf 2>/dev/null || true

echo "===D install as non-root (core + vllm)"
# Install the branch under test, not the last release. /glq is mounted read-only and pip
# builds a local directory IN-TREE (it writes glq.egg-info), so it has to be copied to a
# path `tester` owns first — a read-only source dies with `could not create glq.egg-info:
# Permission denied` before any of this is exercised.
#
# Without --glq-source install.sh pulls glq from PyPI, which makes every leg a test of the
# previous release: a fix cannot be validated before it ships, and a regression introduced
# on this branch passes. Measured — 0.8.5 shipped a broken kernel build precisely because
# the only end-to-end check ran against 0.8.4.
cp -a /glq /home/tester/src && chown -R tester:tester /home/tester/src
su tester -c 'GLQ_HOME=$HOME/.glq PIP_CACHE_DIR=/pipcache HF_HOME=/hf \
    bash /glq/install.sh --yes --components core,vllm --glq-source /home/tester/src' \
    >/tmp/install.log 2>&1
echo "D_EXIT=$?"
tail -15 /tmp/install.log

echo "===E self-check"
su tester -c '$HOME/.glq/venv/bin/glq-setup --verify'; echo "E_EXIT=$?"

echo "===F root is still refused"
bash /glq/install.sh --yes >/tmp/root.log 2>&1; echo "F_EXIT=$?"
grep -o 'refusing to run as root' /tmp/root.log | head -1
grep -o -- '--allow-root' /tmp/root.log | head -1

echo "===G generate on the GPU (JIT-builds the CUDA extension first)"
cat >/tmp/gen.py <<'PYEOF'
# The main guard is required, not stylistic: vLLM v1 loads the model in a spawned
# EngineCore subprocess, which re-imports this module. Without it the child re-runs
# construction and the failure surfaces as an unrelated spawn error.
if __name__ == "__main__":
    import glq.inference_kernel as ik
    ok, err = ik.cuda_ext_status()
    print("EXT_OK:%s" % ok)
    print("EXT_ERR:%s" % (err or "").replace("\n", " | "))

    from vllm import LLM, SamplingParams
    llm = LLM(model="__MODEL__", quantization="glq",
              gpu_memory_utilization=0.35, max_model_len=2048, enforce_eager=True)
    out = llm.generate(["The capital of France is"],
                       SamplingParams(temperature=0.0, max_tokens=8))
    print("GEN_TEXT:" + out[0].outputs[0].text.strip().replace("\n", " "))
PYEOF
sed -i "s|__MODEL__|$GLQ_SMOKE_MODEL|" /tmp/gen.py
chmod 0644 /tmp/gen.py
su tester -c "HF_HOME=/hf \$HOME/.glq/venv/bin/python /tmp/gen.py" >/tmp/gen.log 2>&1
echo "G_EXIT=$?"
grep -a "EXT_OK:\|EXT_ERR:\|GEN_TEXT:" /tmp/gen.log
# The whole log, never a tail. The engine-core traceback that explains a failure here is
# thousands of lines above the end, and truncating it discarded the root cause three
# separate times before this comment existed.
grep -aq "GEN_TEXT:" /tmp/gen.log || cat /tmp/gen.log
"""
    proc = _sh(distro, script.replace("$GLQ_SMOKE_MODEL", SMOKE_MODEL), timeout=9000)
    out = _out(proc)

    # Distinguish "this test cannot run here" from "glq is broken here". Without this the
    # two are indistinguishable in the output, and a missing `su` reads as an install
    # failure — which is exactly how the first fedora:44 run was misdiagnosed.
    assert "HARNESS_PREREQ_MISSING" not in out, (
        f"{distro.name}: the harness could not create or switch to a non-root user, so "
        f"nothing about glq was tested here.\n{out[-2000:]}")

    assert "D_EXIT=0" in out, f"{distro.name}: install failed\n{out[-4000:]}"
    assert "E_EXIT=0" in out, f"{distro.name}: self-check failed\n{out[-3000:]}"
    assert "F_EXIT=0" not in out.split("===F")[-1], f"{distro.name}: root guard did not fire"
    assert "refusing to run as root" in out
    assert "--allow-root" in out

    assert "G_EXIT=0" in out, f"{distro.name}: generation crashed\n{out[-4000:]}"

    # Assert the *mechanism*, not just the output. GLQ falls back to a pure-torch decode
    # when the extension is missing, which is correct but slow — so a green "Paris" alone
    # cannot distinguish "the fast path works here" from "the fast path is dead and nobody
    # noticed". Naming the reason matters too: `cuda_ext_status()` carries the compiler's
    # own words, and recovering them by hand cost three container re-runs.
    assert "EXT_OK:True" in out, (
        f"{distro.name}: the CUDA extension did not build, so this install has no fast "
        f"path.\n" + "\n".join(ln for ln in out.splitlines() if ln.startswith("EXT_ERR:")))

    gen = [ln for ln in out.splitlines() if ln.startswith("GEN_TEXT:")]
    assert gen, f"{distro.name}: no text generated\n{out[-3000:]}"
    text = gen[0][len("GEN_TEXT:"):].strip()

    # The coherence check. A working stack completes this with the city; a kernel that
    # JIT-built wrongly for this distro emits noise, repetition or empty output — all of
    # which fail here, which is the whole point of decoding real tokens per distro.
    assert "paris" in text.lower(), (
        f"{distro.name}: generation is not coherent — expected Paris, got {text!r}.\n"
        f"That points at the CUDA extension's JIT build against this distro's toolchain, "
        f"not at the packaging (install and self-check both passed).")

    # Print the evidence on SUCCESS too. `_sh` captures the container's output, and until
    # this existed it was only ever surfaced inside an assertion message — so a green run
    # left nothing to look at, and "the model generated Paris" was a claim backed by an
    # assertion nobody could see. Needs `-s`; the log is the artifact for this suite.
    print(f"\n[{distro.name}] kernels_built=True  generated={text!r}")
