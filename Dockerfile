# syntax=docker/dockerfile:1.6
#
# GLQ runtime + eval environment.
#
# Built by running the project's own installer, so this file cannot drift from what users
# actually execute. The previous version reimplemented the venv, the torch pin, the CUDA
# toolchain and the pip ordering by hand, and went stale: it shipped glq 0.5.3 / vLLM 0.20.2
# against a current 0.8.x, and nothing here would have noticed.
#
# Base: CUDA *runtime*, not devel. install.sh installs nvcc and the CCCL headers from pip
# into the venv, which is the path a user's machine takes, so a ~4 GB devel base buys
# nothing but a second, divergent toolchain. Note the runtime image still ships
# /usr/local/cuda (include/ + lib64/, no compiler) — torch's `_find_cuda_home()` resolves
# that directory on sight, so glq checks for `bin/nvcc` before believing it. Plain
# `ubuntu:24.04` also works and is what tests/test_installer_distros.py exercises; the CUDA
# runtime base is kept because the eval tooling below links against it.
#
# Build:   docker build -t ghcr.io/cnygaard/glq-env:0.8.6 .
#          docker build --build-arg GLQ_VERSION=0.8.6 .
# Run:     docker run --gpus all -it --rm \
#              -v $HOME/.cache/huggingface:/cache/hf \
#              -e HF_TOKEN=$HF_TOKEN \
#              ghcr.io/cnygaard/glq-env:0.8.6
ARG CUDA_VERSION=12.9.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# Which glq to install. Everything else — torch, vLLM, transformers, the CUDA build
# toolchain — is resolved by the installer, so there is exactly one pin to bump here.
ARG GLQ_VERSION=0.8.5

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
RUN bash /tmp/install.sh --yes \
        --components core,vllm \
        --glq-version "${GLQ_VERSION}"

# Bake the fused CUDA kernels for every architecture we support, so the first request in a
# container is not paying a ~1 min JIT compile. There is no GPU during `docker build`, so
# the arch list has to be explicit — torch cannot probe for it.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX"

# Assert, do not swallow. This used to end in `|| true`, which meant an image could ship
# with no kernels at all and say nothing — the same silent-downgrade failure the runtime
# diagnostics exist to prevent. `cuda_ext_status()` returns the compiler's own words.
RUN python -c "\
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
