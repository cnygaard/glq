# syntax=docker/dockerfile:1.6
#
# GLQ runtime + dev environment.
#
# Reproduces the cloud-init setup at infra/setup.sh.tftpl in a portable
# container so contributors can replicate eval results without AWS, and so
# our AMIs can be baked from the same source of truth.
#
# Base: NVIDIA CUDA 12.8 cuDNN devel on Ubuntu 24.04
# - CUDA 12.8 matches torch 2.11+cu128 binary wheels
# - Ubuntu 24.04 ships Python 3.12 — same minor we use in the venv
# - cudnn-devel includes nvcc, headers, and libs needed for JIT-compiling
#   our CUDA C extension and for building mamba-ssm / causal-conv1d
#
# Build:   docker build -t ghcr.io/cnygaard/glq-env:0.2.16 .
# Run:     docker run --gpus all -it --rm \
#              -v $HOME/.cache/huggingface:/cache/hf \
#              -e HF_TOKEN=$HF_TOKEN \
#              ghcr.io/cnygaard/glq-env:0.2.16
ARG CUDA_VERSION=12.8.0
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

# Build args (re-declared after FROM so they're in scope).
ARG GLQ_VERSION=0.2.16
ARG VLLM_VERSION=0.20.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

# Locale + non-interactive apt.
ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=0

# System packages — split into stable and rebuild-likely groups so layer
# caching is effective across small changes.
#
# - build-essential, ninja-build, cmake: needed for JIT cpp_extension loads
#   and for compiling causal-conv1d / mamba-ssm wheels
# - libnuma-dev, protobuf-compiler: dependencies of sgl-kernel (optional)
# - git: for fetching the GLQ source if the user mounts it / clones it
# - curl: for hf download fallback and general debugging
# - python3-venv: lets us create a clean isolated venv inside /opt/venv
# - python3-dev: headers required by some pip-installed C extensions
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        curl \
        ca-certificates \
        libnuma-dev \
        protobuf-compiler \
        python3 \
        python3-venv \
        python3-dev \
        python3-pip \
        && rm -rf /var/lib/apt/lists/*

# Python venv at /opt/venv. Activated by prepending to PATH.
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv

# Pin pip + base build tooling first (these rarely change).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel ninja

# Heavy CUDA stack: torch first so subsequent extensions build against it.
# Pinning torch 2.11+cu128 to match the rest of the production stack
# (vLLM 0.20.0 ABI, our compiled fast-hadamard-transform wheel).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        --extra-index-url ${TORCH_INDEX_URL} \
        "torch==2.11.0" "torchaudio==2.11.0" "torchvision==0.26.0"

# vLLM — pulls a long chain of deps including transformers, huggingface_hub,
# flashinfer, etc. Keep this layer separate so changes in glq version don't
# bust the vLLM cache.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "vllm==${VLLM_VERSION}"

# transformers/HF — pin the major to keep the chat-template thinking-mode
# behaviour predictable.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "transformers>=5.0,<6.0" "huggingface_hub>=0.30"

# CUDA_HOME is needed by mamba-ssm and causal-conv1d setup.py builds.
ENV CUDA_HOME=/usr/local/cuda

# Mamba/SSM and causal-conv1d are compiled against the installed torch.
# --no-build-isolation reuses the venv's torch instead of pulling a fresh
# copy into a sandboxed build env (which would mismatch ABI).
#
# These take ~10-15 min wall on a 8-core build. Allow it to fail silently
# (matching our setup.sh.tftpl behavior) so the rest of the image still
# builds; users who don't need Mamba/SSM models can ignore the warning.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install causal-conv1d --no-build-isolation \
        || echo "WARN: causal-conv1d build failed — Mamba models will be unavailable" && \
    pip install mamba-ssm --no-build-isolation \
        || echo "WARN: mamba-ssm build failed — Mamba models will be unavailable"

# Eval + auxiliary tooling.
# - lm-eval: gsm8k, mmlu, hellaswag, etc.
# - langdetect, immutabledict: ifeval task deps
# - pypcre + gptqmodel: GPTQ baseline comparison
# - optimum: HF transformers GPTQ loader
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install lm-eval langdetect immutabledict pypcre optimum && \
    pip install gptqmodel --no-build-isolation \
        || echo "WARN: gptqmodel build failed — GPTQ baseline unavailable"

# GLQ itself, last because it's the most-frequently-bumped package.
# `[quantize]` extra includes calibration-time deps (datasets, tqdm extras).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "glq[quantize]==${GLQ_VERSION}"

# Pre-warm the GLQ CUDA C extension JIT compile so the first inference call
# is fast. Build environment has nvcc + headers so this should succeed.
# Skipped when CUDA isn't available in the build runner; the kernel will
# still JIT at first import.
RUN python -c "from glq import inference_kernel as ik; ok = ik._try_load_cuda_ext(); print(f'glq CUDA ext: {\"baked\" if ok else \"will JIT at runtime\"}')" || true

# HuggingFace cache lives on a writable volume so model downloads survive
# container restarts. Mount via -v $HOME/.cache/huggingface:/cache/hf.
ENV HF_HOME=/cache/hf
VOLUME ["/cache/hf"]

# NVIDIA Container Toolkit handoff — these env vars are read by the runtime
# at `docker run --gpus all` to expose the host GPU(s).
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace
SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash"]
