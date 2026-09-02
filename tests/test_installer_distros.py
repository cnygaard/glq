"""install.sh across Linux distributions, in Docker, with a real GPU.

`tests/test_preflight_sh.py` proves the *mapping* — that a Fedora `/etc/os-release` produces
a `dnf` line rather than an `apt-get` one — using fabricated files. It cannot tell you
whether `python3-devel gcc curl` is the *correct* package set on Fedora, or whether the
installer works once you have run it. Only Ubuntu 24.04 has ever actually been executed.

That gap matters because pre-flight makes a promise: *run this command and you will be
ready*. This suite tests the promise, per distro:

    pre-flight on a pristine image  ->  run the command it printed  ->  pre-flight passes

then goes on to a real venv, a real `pip install glq`, the self-check, a real token from a
real checkpoint, and finally `glq-chat` — which starts vLLM itself and has to give the card
back when it stops.

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
import threading

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PIP_CACHE = os.environ.get("GLQ_DISTRO_PIP_CACHE", "/opt/dlami/nvme/pipcache")
HF_CACHE = os.environ.get("GLQ_DISTRO_HF_CACHE", "/opt/dlami/nvme/hf_cache_distro")

#: Fraction of the card each container's vLLM may reserve. 0.35 suits a 24 GB card running
#: one leg at a time; it is also what caps parallelism, because N legs need N × this.
#: The smoke model is 1.8 GiB, so on a large card a much smaller slice is ample and lets
#: `pytest -n` actually overlap — on a 96 GB Blackwell, 0.10 is ~9.8 GiB per leg and eight
#: legs still fit. Too small and vLLM refuses to start with "No available memory for the
#: cache blocks", which is a harness misconfiguration, not a glq bug.
GPU_UTIL = os.environ.get("GLQ_DISTRO_GPU_UTIL", "0.35")

#: Seconds to wait for vLLM to answer, passed to glq-chat AND used by this
#: suite's own readiness loop. 1200 rather than the supervisor's 900 default
#: because concurrent legs make startup much slower than a solo run.
READY_TIMEOUT = os.environ.get("GLQ_DISTRO_READY_TIMEOUT", "1200")

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
    # 26.04 LTS ships no python3 at all in the base image, so pre-flight's advice is the
    # only thing standing between a user and a dead install — a stronger test of it than
    # 24.04, which comes with a usable interpreter.
    Distro("ubuntu2604", "ubuntu:26.04",                                   "apt-get"),
    Distro("debian",     "debian:12",                                      "apt-get"),
    # Both current Fedoras, because they differ in the one way that decides whether GLQ can
    # compile at all: 43 ships gcc 15.3.1, 44 ships gcc 16.1.1, and CUDA 13.0's
    # crt/host_config.h refuses anything past 15. 44 is therefore the distro where the
    # prebuilt wheel is not a convenience but the only thing that works.
    Distro("fedora43",   "fedora:43",                                      "dnf"),
    Distro("fedora",     "fedora:44",                                      "dnf"),
    Distro("ubi9",       "registry.access.redhat.com/ubi9/ubi:latest",     "dnf"),
    Distro("almalinux",  "almalinux:9",                                    "dnf"),
    Distro("amazon2023", "amazonlinux:2023",                               "dnf"),
    Distro("arch",       "archlinux:latest",                               "pacman"),
    # Azure Linux 4.0 (beta channel — Microsoft has not published a 4.x tag under
    # azurelinux/base/core yet; the whole 4.x line lives in azurelinux-beta). It is the
    # rebase onto a Fedora upstream, and that is why it earns the slot over 3.0: the image
    # sets ID_LIKE=fedora, which routes it to a DIFFERENT pkg_hint branch than 3.0's
    # tdnf one — the *fedora* case matches first — and ships dnf alongside tdnf. Verified
    # in the image: ID=azurelinux, ID_LIKE=fedora, VERSION_ID=4.0, and no which/awk/
    # python3/gcc, so it exercises the advice path from a genuinely bare base.
    Distro("azurelinux", "mcr.microsoft.com/azurelinux-beta/base/core:4.0", "dnf"),
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
    cmd = [
        # --shm-size: vLLM needs far more than docker's 64 MB default and fails obscurely
        # without it. The torch_extensions cache is deliberately NOT shared — the CUDA
        # extension's JIT build against each distro's toolchain is part of what we test.
        "docker", "run", "--rm", "--gpus", "all", "--shm-size=8g",
        "-v", f"{ROOT}:/glq:ro",
        "-v", f"{PIP_CACHE}:/pipcache",
        "-v", f"{HF_CACHE}:/hf",
        "-e", "PIP_CACHE_DIR=/pipcache",
        "-e", "HF_HOME=/hf",
        "-e", "GLQ_HOME=/root/.glq",
        # Python block-buffers when stdout is a pipe, which every container process here
        # has. Without this its output arrives in 4-8 KB chunks *after* the shell `echo`s
        # that were meant to label it, so stages appear interleaved — and if a process dies
        # the unflushed tail, i.e. the part naming the failure, is simply gone.
        "-e", "PYTHONUNBUFFERED=1",
        distro.image, "bash", "-c", script,
    ]
    # Stream instead of `capture_output=True`. A leg runs for minutes; capture shows nothing
    # until the container exits, so a hang is indistinguishable from slow progress and an
    # interrupted pytest loses the entire log. Printing as lines arrive makes `-s` a live
    # view, still returns the whole text for the assertions, and means a tee'd run has the
    # output on disk as it happens rather than only at the end.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    killer = threading.Timer(timeout, proc.kill)
    killer.start()
    lines = []
    try:
        for line in proc.stdout:
            lines.append(line)
            print(line, end="", flush=True)
        proc.wait()
    finally:
        killer.cancel()
    # stderr is folded into stdout above so ordering is preserved; _out() still works.
    return subprocess.CompletedProcess(cmd, proc.returncode, "".join(lines), "")


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

#: Where glq comes from. Both matter, and they answer different questions.
#:
#:   wheel  — no --glq-source, so install.sh pip-installs from PyPI. Since 0.8.6 that is a
#:            prebuilt cp3XX manylinux wheel, which is what a real user gets and the only
#:            path that works on a distro whose compiler CUDA rejects (fedora:44, gcc 16).
#:            It cannot validate an unreleased fix — it tests the last release.
#:   source — --glq-source, installing the mounted tree, exercising the JIT build. This is
#:            what lets a fix be validated *before* it ships; running only the wheel arm is
#:            how 0.8.5 went out with kernels that could not build.
#:
#: Neither subsumes the other, so the leg runs both rather than picking one.
INSTALL_MODES = ("wheel", "source")


@pytest.mark.parametrize("install_from", INSTALL_MODES)
@pytest.mark.parametrize("distro", DISTROS, ids=lambda d: d.name)
def test_full_install_and_coherent_generation(distro, install_from):
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

# `useradd`, `su` and `which` are prerequisites of the TEST and of the stack it drives —
# not of glq — so they are installed separately from the pre-flight set above, which must
# stay exactly what `pkg_hint` printed or the claim that pre-flight's advice is sufficient
# stops being testable.
#
# Minimal RPM images ship none of them. Two separate measurements:
#
#   * fedora:44 — every stage returned 127 with `su: command not found`, and the `|| true`
#     that used to sit on the useradd line swallowed it, so a missing harness dependency
#     read as a glq failure three stages later.
#   * fedora:43 AND :44, in wheel mode — install clean (D_EXIT=0), then vLLM's engine core
#     died with `FileNotFoundError: [Errno 2] No such file or directory: 'which'`. Something
#     in vLLM's startup shells out to /usr/bin/which rather than using shutil.which, and
#     `which` is its own package on RPM distros. Debian-family images ship it, which is why
#     ubuntu never hit this. It fails before the gcc-version question is even reached, so
#     without this it masks the thing fedora:43 exists to test.
if ! command -v useradd >/dev/null 2>&1 || ! command -v su >/dev/null 2>&1 \
   || ! command -v which >/dev/null 2>&1; then
    {   if   command -v dnf     >/dev/null 2>&1; then dnf install -y shadow-utils util-linux which
        elif command -v apt-get >/dev/null 2>&1; then apt-get install -y passwd login debianutils
        elif command -v pacman  >/dev/null 2>&1; then pacman -Sy --noconfirm shadow util-linux which
        elif command -v tdnf    >/dev/null 2>&1; then tdnf install -y shadow-utils util-linux which
        elif command -v zypper  >/dev/null 2>&1; then zypper -n install shadow util-linux which
        fi
    } >/tmp/harness_prereq.log 2>&1 || true
fi
for tool in useradd su which; do
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
    bash /glq/install.sh --yes --components core,vllm __GLQ_SOURCE_FLAG__' \
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
              gpu_memory_utilization=__GPU_UTIL__, max_model_len=2048, enforce_eager=True)
    out = llm.generate(["The capital of France is"],
                       SamplingParams(temperature=0.0, max_tokens=8))
    print("GEN_TEXT:" + out[0].outputs[0].text.strip().replace("\n", " "))
PYEOF
sed -i "s|__MODEL__|$GLQ_SMOKE_MODEL|" /tmp/gen.py
chmod 0644 /tmp/gen.py
# VLLM_USE_FLASHINFER_SAMPLER=0 because this stage drives vLLM's Python API directly and
# therefore bypasses glq-chat, which sets it itself when the CUDA Toolkit is absent.
# Measured on sm_120 with no toolkit: FlashInfer ships no prebuilt sampler for that arch,
# JIT-compiles at engine start, cannot find nvcc, and EngineCore dies before a token —
# with GLQ's own kernel having loaded fine. Without this the leg re-measures FlashInfer
# rather than glq. It is a no-op on sm_86/sm_89, where FlashInfer ships prebuilt.
su tester -c "HF_HOME=/hf VLLM_USE_FLASHINFER_SAMPLER=0 \$HOME/.glq/venv/bin/python /tmp/gen.py" >/tmp/gen.log 2>&1
echo "G_EXIT=$?"
grep -a "EXT_OK:\|EXT_ERR:\|GEN_TEXT:" /tmp/gen.log
# The whole log, never a tail. The engine-core traceback that explains a failure here is
# thousands of lines above the end, and truncating it discarded the root cause three
# separate times before this comment existed.
grep -aq "GEN_TEXT:" /tmp/gen.log || cat /tmp/gen.log

echo "===H glq-chat owns the server (starts vLLM, serves the UI, frees the GPU on exit)"
# Capability probe, not a version comparison. In wheel mode install.sh takes glq from PyPI,
# so this stage runs against the LAST RELEASE — and the supervisor arrived after 0.8.6, whose
# glq-chat accepts only --base-url/--port/--share. Passing --model there is an argparse error
# before anything starts, which would fail this leg for a feature that is not published yet.
if ! su tester -c "\$HOME/.glq/venv/bin/glq-chat --help" 2>&1 | grep -q -- "--model"; then
    echo "H_SKIP: installed glq-chat has no supervisor (pre-0.8.7); stage H not applicable"
else
# The chat extra is installed here rather than folded into stage D so that a gradio problem
# is diagnosed as one — the stages above stay about the kernel.
su tester -c "PIP_CACHE_DIR=/pipcache \$HOME/.glq/venv/bin/pip install -q 'gradio>=6'" \
    >/tmp/gradio.log 2>&1; echo "H_PIP_EXIT=$?"
tail -5 /tmp/gradio.log

# --no-browser because there is no display; everything else is what a user gets. No
# `vllm serve` anywhere in this stage: starting it is the behaviour under test.
# --ready-timeout only exists from 0.8.8. In wheel mode this stage runs against whatever is
# published, so passing it unconditionally would be an argparse error on older releases —
# the same trap that made stage H fail for a feature that had not shipped yet. Probe, do not
# assume.
RT_FLAG=""
if su tester -c "\$HOME/.glq/venv/bin/glq-chat --help" 2>&1 | grep -q -- "--ready-timeout"; then
    RT_FLAG="--ready-timeout __READY_TIMEOUT__"
else
    echo "NOTE: installed glq-chat has no --ready-timeout; using its built-in default"
fi
su tester -c "HF_HOME=/hf nohup \$HOME/.glq/venv/bin/glq-chat --no-browser \
    --model $GLQ_SMOKE_MODEL --gpu-memory-utilization __GPU_UTIL__ --port 7861 $RT_FLAG \
    >/tmp/chat.log 2>&1 & echo \$! >/tmp/chat.pid"
CHAT_PID=$(cat /tmp/chat.pid 2>/dev/null || echo 0)
echo "CHAT_PID:$CHAT_PID"

READY=False
# 240 x 5s = 20 minutes, and glq-chat is given the same budget via --ready-timeout. Both
# halves matter: the harness used to wait 10 minutes while the chat gave up at its own
# hardcoded 15, so raising either alone just moves which one fires first. Measured at 5-way
# concurrency on 16 cores — 12 legs of 44 failed with "never got a server", not because
# anything was broken but because five simultaneous weight loads and CUDA-graph captures
# outran the window.
for _ in $(seq 1 240); do
    curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { READY=True; break; }
    kill -0 "$CHAT_PID" 2>/dev/null || break        # it died; no point waiting out the clock
    sleep 5
done
echo "CHAT_READY:$READY"

# Poll, do not curl once. The readiness loop above waits for vLLM's /v1/models, but
# glq-chat only starts gradio *after* the server answers — so there is a window where the
# API is up and the UI is not. A single curl wins or loses that race by luck: it passed on
# 2026-08-17 and failed on 2026-08-18 with the same code.
UI_OK=False
for _ in $(seq 1 30); do
    curl -sf http://127.0.0.1:7861/ >/dev/null 2>&1 && { UI_OK=True; break; }
    kill -0 "$CHAT_PID" 2>/dev/null || break
    sleep 2
done
echo "UI_OK:$UI_OK"

# vLLM says "Engine core initialization failed. See root cause above" — and "above" is its
# OWN log file, which dies with the container. Two separate investigations stalled here for
# exactly that reason, so dump it whenever the chat did not come up.
if [ "$READY" != "True" ]; then
    for f in /tmp/glq-vllm-*.log; do
        [ -f "$f" ] || continue
        echo "===VLLM_LOG $f"
        tail -200 "$f"
        echo "===VLLM_LOG_END"
    done
fi

curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\": \"$GLQ_SMOKE_MODEL\", \"prompt\": \"The capital of France is\", \
         \"max_tokens\": 8, \"temperature\": 0}" >/tmp/chat_gen.json 2>/dev/null
cat >/tmp/chat_text.py <<'PYEOF'
import json
d = json.load(open("/tmp/chat_gen.json"))
print("CHAT_TEXT:" + d["choices"][0]["text"].strip().replace("\n", " "))
PYEOF
chmod 0644 /tmp/chat_text.py
su tester -c "\$HOME/.glq/venv/bin/python /tmp/chat_text.py" 2>/dev/null || echo "CHAT_TEXT:"

# The point of the whole design: stopping the chat must free the card. vLLM has no idle
# unload, so a server that outlives its client holds its share of VRAM until something
# kills it. The assertion is that every vLLM process it started is gone afterwards, not
# merely that the chat exited.
# Exclude PID 1 and this shell. In a container PID 1 is the `bash -c <script>` running
# this very test, and the script text contains "vllm" (--components core,vllm, and this
# pattern itself), so `pgrep -f` matches it. PID 1 never exits while the script runs, which
# made VLLM_GONE:False a foregone conclusion — it reported a teardown failure on a run whose
# own log said "stopping vLLM — the GPU is free again" and whose nvidia-smi showed the card
# empty. Two releases' worth of teardown evidence was this artifact.
pgrep -f 'vllm|VLLM' 2>/dev/null | grep -vx -e 1 -e "$$" >/tmp/vllm.pids || true
echo "VLLM_PIDS:$(tr '\n' ' ' </tmp/vllm.pids)"
# SIGTERM, not SIGINT. A non-interactive shell sets SIGINT to SIG_IGN for background
# jobs and the child inherits that disposition — CPython preserves an inherited SIG_IGN
# rather than installing its own handler — so `kill -INT` on a nohup'd job is a no-op and
# this stage asserted nothing. Measured: vLLM survived, reported as VLLM_GONE:False, which
# read as a teardown defect. glq-chat handles SIGTERM and SIGHUP explicitly, and SIGTERM is
# also what a service manager sends, so it is both testable here and a real path.
kill -TERM "$CHAT_PID" 2>/dev/null || true
GONE=False
for _ in $(seq 1 45); do
    alive=0
    for p in $(cat /tmp/vllm.pids 2>/dev/null); do
        kill -0 "$p" 2>/dev/null && alive=$((alive+1))
    done
    kill -0 "$CHAT_PID" 2>/dev/null && alive=$((alive+1))
    [ "$alive" -eq 0 ] && { GONE=True; break; }
    sleep 2
done
echo "VLLM_GONE:$GONE"
echo "GPU_APPS_AFTER:$(nvidia-smi --query-compute-apps=pid,used_memory \
    --format=csv,noheader 2>/dev/null | tr '\n' ';')"
grep -a "attaching\|starting vLLM\|reserving\|ready —\|stopping vLLM" /tmp/chat.log || true
[ "$READY" = True ] && [ "$GONE" = True ] || tail -40 /tmp/chat.log
__GLQ_CHAT_STAGE_END__
"""
    script = script.replace(
        "__GLQ_SOURCE_FLAG__",
        "--glq-source /home/tester/src" if install_from == "source" else "")
    script = script.replace("__GLQ_CHAT_STAGE_END__", "fi")
    script = script.replace("__GPU_UTIL__", GPU_UTIL)
    script = script.replace("__READY_TIMEOUT__", READY_TIMEOUT)
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

    # ---- stage H: the one command a new user runs -----------------------------------
    #
    # `glq-chat` starts vLLM, waits for it, serves the UI and stops the server again on
    # Ctrl-C. Each half is asserted separately because they fail for different reasons: a
    # chat that never becomes ready is a startup bug, and a chat that leaves vLLM running
    # is a VRAM leak the user only discovers when their next program cannot allocate.
    if "H_SKIP:" in out:
        # Not a pass and not a failure: the published release under test predates the
        # supervisor. Printed rather than swallowed, so a green run cannot be mistaken for
        # evidence that glq-chat manages vLLM here.
        why = [ln for ln in out.splitlines() if "H_SKIP:" in ln][0].strip()
        print(f"\n[{distro.name}] stage H SKIPPED — {why}")
        return

    assert "H_PIP_EXIT=0" in out, f"{distro.name}: the chat extra would not install\n{out[-2000:]}"

    assert "CHAT_READY:True" in out, (
        f"{distro.name}: glq-chat never got a server answering on /v1/models, so it did "
        f"not start vLLM (or vLLM died starting).\n{out[-3000:]}")

    assert "UI_OK:True" in out, f"{distro.name}: the chat UI did not come up\n{out[-2000:]}"

    chat = [ln for ln in out.splitlines() if ln.startswith("CHAT_TEXT:")]
    chat_text = chat[0][len("CHAT_TEXT:"):].strip() if chat else ""
    assert "paris" in chat_text.lower(), (
        f"{distro.name}: the server glq-chat started does not decode coherently — "
        f"expected Paris, got {chat_text!r}\n{out[-2000:]}")

    # The mechanism, not the output: SIGINT must take vLLM down with the chat. A test that
    # only checked "the chat exited" would pass with the server still holding the card.
    # nvidia-smi is the ground truth for "the card is free": process bookkeeping can be
    # fooled (see the PID 1 comment in the script), an empty compute-apps list cannot.
    apps = [ln.split(":", 1)[1].strip() for ln in out.splitlines()
            if ln.startswith("GPU_APPS_AFTER:")]
    assert apps and apps[0] == "", (
        f"{distro.name}: the GPU still has compute apps after the chat stopped: "
        f"{apps[0] if apps else '(not reported)'}\n{out[-2000:]}")

    assert "VLLM_GONE:True" in out, (
        f"{distro.name}: vLLM survived Ctrl-C, so the GPU is still reserved with nothing "
        f"driving it.\n{out[-3000:]}")

    # Print the evidence on SUCCESS too. `_sh` captures the container's output, and until
    # this existed it was only ever surfaced inside an assertion message — so a green run
    # left nothing to look at, and "the model generated Paris" was a claim backed by an
    # assertion nobody could see. Needs `-s`; the log is the artifact for this suite.
    print(f"\n[{distro.name}] kernels_built=True  generated={text!r}"
          f"  chat_started_and_stopped_vllm=True")
