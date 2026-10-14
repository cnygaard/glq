# syntax=docker/dockerfile:1.6
#
# GLQ runtime + eval environment.
#
# Built by running the project's own installer, so this file cannot drift from what users
# actually execute. The previous version reimplemented the venv, the torch pin, the CUDA
# toolchain and the pip ordering by hand, and went stale: it shipped glq 0.5.3 / vLLM 0.20.2
# against a current 0.8.x, and nothing here would have noticed.
#
# Base: plain Ubuntu, NOT an nvidia/cuda image. install.sh installs nvcc and the CCCL headers
# from pip into the venv (`cuda-toolkit[nvcc,cccl]` → `<site-packages>/nvidia/cu13/bin/nvcc`),
# which is the path a user's machine takes, so the CUDA runtime and the compiler both arrive
# as wheels. At `docker run --gpus all` the NVIDIA Container Toolkit injects the *driver*
# (libcuda.so); the CUDA *runtime* (libcudart) comes from pip. Nothing here needs a system CUDA.
#
# This used to be `nvidia/cuda:12.9.1-runtime-ubuntu24.04`, and that pin broke the image for
# **six consecutive releases** (v0.8.6 through v0.8.26, measured) with the build failing at the
# kernel-assertion step below:
#
#   prebuilt glq._C did not load: ImportError: libcudart.so.13: cannot open shared object file
#   OSError: CUDA_HOME environment variable is not set.
#
# The wheels link against CUDA **13** while the base supplied libcudart.so.**12**, so the
# prebuilt extension could not load and the JIT fallback had no compiler to fall back to. Two
# separately-pinned CUDA versions is the defect; a plain base removes the class rather than
# re-pinning it, and `tests/test_installer_distros.py` already exercises `ubuntu:24.04`.
#
# 24.04 rather than 26.04 deliberately: nvcc caps the host gcc version, install.sh carries
# per-distro compat-compiler logic for that (`MAX_NVCC_GCC`), and 24.04's gcc 13 sits under
# CUDA 13's ceiling without needing it. Fedora 44 is blocked outright by gcc 16 vs CUDA 13 —
# see the distro-matrix notes. Move to 26.04 once the matrix has a green source-mode run on it.
#
# Build:   docker build -t ghcr.io/cnygaard/glq-env:0.8.27 .
#          docker build --build-arg GLQ_VERSION=0.8.27 .
# Run:     docker run --gpus all -it --rm \
#              -v $HOME/.cache/huggingface:/cache/hf \
#              -e HF_TOKEN=$HF_TOKEN \
#              ghcr.io/cnygaard/glq-env:0.8.27
ARG UBUNTU_VERSION=24.04

FROM ubuntu:${UBUNTU_VERSION}

# Which glq to install. Everything else — torch, vLLM, transformers, the CUDA build
# toolchain — is resolved by the installer, so there is exactly one pin to bump here.
ARG GLQ_VERSION=0.8.27

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Exactly the package set install.sh's own pre-flight prints for Debian-family, plus git.
# If this list and `pkg_hint()` disagree, the pre-flight is wrong for real users too — which
# is the point of installing what it asks for rather than what we think it needs.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-dev \
        build-essential \
        curl \
        ca-certificates \
        git

# A dedicated unprivileged user. install.sh refuses to run as root on purpose — nothing it
# does needs privileges, and a root-owned venv in /root is a habit worth not teaching — so
# running it under `--allow-root` here would be defeating our own guard rather than honouring
# it. This is also the path tests/test_installer_distros.py exercises.
#
# High UID/GID to avoid colliding with the `ubuntu` user (1000) that ships in the base image.
# Override to match the host when bind-mounting a cache you also write to from outside:
#   docker build --build-arg GLQ_UID=$(id -u) --build-arg GLQ_GID=$(id -g) .
ARG GLQ_UID=10001
ARG GLQ_GID=10001
RUN groupadd -g ${GLQ_GID} glq && \
    useradd -m -u ${GLQ_UID} -g ${GLQ_GID} -s /bin/bash glq && \
    mkdir -p /cache/hf /workspace && \
    chown -R glq:glq /cache/hf /workspace

USER glq

# The venv the installer creates, on PATH so `vllm`/`glq-*` resolve without activation.
# (That PATH entry is also what lets torch's `_find_cuda_home()` locate the venv's nvcc via
# `shutil.which` — see the bake step below.)
ENV GLQ_HOME=/home/glq/.glq \
    PATH=/home/glq/.glq/venv/bin:$PATH

# One source of truth for the install.
#
# COPY rather than `curl -fsSL …/main/install.sh | bash`: the published one-liner fetches
# whatever `main` says at build time, and Docker cannot see that the remote file moved — so
# the layer cache serves a stale image, and two builds of the same tag can differ with
# nothing recording why. Copying invalidates the layer exactly when the installer changes,
# and builds the installer from this commit rather than from a moving branch.
COPY --chown=glq:glq install.sh /tmp/install.sh
# No --mount=type=cache here: BuildKit's cache mounts are root-owned by default, and the
# uid=/gid= options do not take build args, so the one thing that would make them writable
# by this user cannot be expressed. Layer caching still covers the common rebuild.
# `picode` adds the pi coding agent, and node via nvm into /home/glq/.nvm, so `glq-code`
# runs in the container instead of exiting on a missing agent. It works here without nvm
# being sourced: `glq-code._find_pi()` globs `~/.nvm/versions/node/*/bin/pi` rather than
# trusting PATH, deliberately, because a bare `which pi` can resolve the unrelated
# Raspberry-Pi `pi` from apt. `chat` adds gradio for `glq-chat`.
#
# Both components were verified non-interactively under `--yes` on a fresh Ubuntu 24.04 box:
# pi landed at ~/.nvm/versions/node/v24.21.0/bin/pi and `glq-setup --verify` reported 7/7.
# `--device cuda` is REQUIRED and is the whole reason this build used to fail. `docker build`
# has no GPU, so install.sh's auto-detection is correct about what it can see and installs the
# CPU stack — measured in a failing build:
#
#   gpu:  no NVIDIA GPU detected — GLQ will serve on CPU (vLLM CPU backend)
#   == no NVIDIA GPU — skipping the CUDA build toolchain
#   [ ok ] vllm backend: vLLM 0.29.0+cpu — the CPU backend     (and torch-2.13.0+cpu)
#
# The image then asserted the CUDA extension below and could never pass: a +cpu torch ships
# no libcudart at any version, and the skipped toolchain left `_venv_cuda_home()` empty, so
# the JIT fallback died on `CUDA_HOME environment variable is not set`. The image is built on
# a GPU-less runner but RUN with `--gpus all`, so the build has to state the target rather
# than probe for it.
#
# GLQ_ASSUME_GPU=1 is needed alongside it. `--device cuda` is passed through to
# `python -m glq.installer` and picks the CUDA vLLM/torch wheels, but install.sh's OWN
# toolchain gate probes `nvidia-smi` independently and would still skip nvcc + the CCCL
# headers, leaving CUDA torch with no compiler and `_venv_cuda_home()` empty. That variable
# is documented in install.sh as the hook for asserting the GPU branch on a machine without
# one, which is precisely a Docker build.
RUN GLQ_ASSUME_GPU=1 bash /tmp/install.sh --yes \
        --components core,vllm,chat,picode \
        --device cuda \
        --glq-version "${GLQ_VERSION}"

# Bake the fused CUDA kernels for every architecture we support, so the first request in a
# container is not paying a ~1 min JIT compile. There is no GPU during `docker build`, so
# the arch list has to be explicit — torch cannot probe for it.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX"

# Assert, do not swallow. This used to end in `|| true`, which meant an image could ship
# with no kernels at all and say nothing — the same silent-downgrade failure the runtime
# diagnostics exist to prevent. `cuda_ext_status()` returns the compiler's own words.
#
# CUDA_HOME and the venv's lib dir are set here because there is no GPU during the build:
# torch only preloads its bundled nvidia libs when a device is present, so without this the
# prebuilt `_C` cannot resolve `libcudart.so.13` even though the wheel ships it at
# `<site-packages>/nvidia/cu13/lib/`. With the path set the PREBUILT extension loads, which
# is faster than a JIT build and a stricter check — it validates the wheel that ships rather
# than one compiled here. `CUDA_HOME` is the JIT fallback's requirement; the compiler lives
# in site-packages, not /usr/local/cuda, so torch cannot find it unaided.
#
# `echo` the value deliberately: when this was first attempted it printed `CUDA_HOME=` empty,
# which is what revealed that the real fault was further up — install.sh had taken the CPU
# path and never installed the toolchain at all. A blank here means `--device cuda` above did
# not take effect.
RUN export CUDA_HOME="$(python -c 'import glq.inference_kernel as ik; print(ik._venv_cuda_home() or "")')" && \
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}" && \
    echo "CUDA_HOME=${CUDA_HOME}" && \
    test -n "${CUDA_HOME}" && \
    python -c "\
import sys; import glq.inference_kernel as ik; \
ok, err = ik.cuda_ext_status(); \
print('glq CUDA ext:', ik._glq_cuda.__file__ if ok else 'FAILED'); \
sys.stderr.write((err or '') + '\n'); \
sys.exit(0 if ok else 1)"

# Eval + baseline-comparison tooling, into the same venv.
#
# CUDA_HOME comes from the venv's pip toolchain — these packages run their own setup.py
# builds and, unlike glq, have no idea the compiler lives in site-packages. gptqmodel is
# allowed to fail: it is a comparison baseline, not part of GLQ.
RUN export CUDA_HOME="$(python -c 'import glq.inference_kernel as ik; print(ik._venv_cuda_home() or "")')" && \
    pip install lm-eval langdetect immutabledict pypcre optimum && \
    { pip install gptqmodel --no-build-isolation \
        || echo "WARN: gptqmodel build failed — GPTQ baseline unavailable"; }

# Record what actually got installed. The versions are resolved by the installer, not pinned
# here, so the image must carry its own provenance or a result cannot be attributed.
RUN pip list --format=freeze \
      | grep -Ei '^(glq|torch|vllm|transformers|triton|cuda-toolkit|nvidia-cuda-nvcc)=' \
      | tee /home/glq/glq-image-versions

# HuggingFace cache on a writable volume so downloads survive container restarts.
ENV HF_HOME=/cache/hf
VOLUME ["/cache/hf"]

# NVIDIA Container Toolkit handoff — read at `docker run --gpus all` to expose host GPUs.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace
SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash"]
