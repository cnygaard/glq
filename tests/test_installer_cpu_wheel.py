"""Resolving the vLLM CPU wheel (glq/installer/cpu_wheel.py).

vLLM ships its CPU backend as GitHub-release wheels (`+cpu` local version), not on PyPI,
with an asset name that has already drifted once in the wild (docs said manylinux_2_35;
the real assets are manylinux_2_34 — a 404 at install time). So resolution scans the
latest release's ASSET LIST for a matching name — proving the file exists before pip
sees the URL — and any failure at all falls back to a pinned known-good wheel, loudly.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.installer import cpu_wheel as W  # noqa: E402

RELEASE = {
    "tag_name": "v0.29.0",
    "assets": [
        {"name": "vllm-0.29.0+cpu-cp312-cp312-macosx_11_0_arm64.whl",
         "browser_download_url": "https://example.invalid/mac.whl"},
        {"name": "vllm-0.29.0+cpu-cp38-abi3-manylinux_2_34_aarch64.whl",
         "browser_download_url": "https://example.invalid/arm.whl"},
        {"name": "vllm-0.29.0+cpu-cp38-abi3-manylinux_2_34_x86_64.whl",
         "browser_download_url": "https://example.invalid/x86.whl"},
        {"name": "vllm-0.29.0-cp38-abi3-manylinux1_x86_64.whl",
         "browser_download_url": "https://example.invalid/cuda.whl"},
    ],
}


def test_matches_the_x86_cpu_asset():
    assert W.latest_cpu_wheel_url("x86_64", fetch=lambda url: RELEASE) \
        == "https://example.invalid/x86.whl"


def test_matches_the_aarch64_cpu_asset():
    assert W.latest_cpu_wheel_url("aarch64", fetch=lambda url: RELEASE) \
        == "https://example.invalid/arm.whl"


def test_never_matches_the_cuda_wheel():
    """The CUDA asset lacks the +cpu local version; a substring slip here would install
    a CUDA vLLM on a CPU box and fail at import."""
    release = {"assets": [a for a in RELEASE["assets"] if "+cpu" not in a["name"]]}
    assert W.FALLBACK_X86 in W.latest_cpu_wheel_url("x86_64", fetch=lambda url: release)


def test_fetch_failure_falls_back_to_the_pinned_wheel():
    def boom(url):
        raise OSError("rate limited")
    url = W.latest_cpu_wheel_url("x86_64", fetch=boom)
    assert "v0.28.0" in url and "+cpu" in url and "x86_64" in url


def test_garbage_response_falls_back():
    assert "v0.28.0" in W.latest_cpu_wheel_url("x86_64", fetch=lambda url: {"weird": 1})


def test_percent_encoded_plus_is_normalized():
    """The live API returns %2B for the + in the local version; the printed pip command
    should carry the literal + (pip accepts both, humans grep for one)."""
    release = {"assets": [
        {"name": "vllm-0.29.0+cpu-cp38-abi3-manylinux_2_34_x86_64.whl",
         "browser_download_url": "https://example.invalid/vllm-0.29.0%2Bcpu-cp38-abi3-manylinux_2_34_x86_64.whl"}]}
    assert "%2B" not in W.latest_cpu_wheel_url("x86_64", fetch=lambda url: release)


def test_wheel_arch_maps_uname():
    assert W.wheel_arch(machine=lambda: "x86_64") == "x86_64"
    assert W.wheel_arch(machine=lambda: "aarch64") == "aarch64"
    assert W.wheel_arch(machine=lambda: "riscv64") is None


def test_install_args_carry_the_pytorch_cpu_index():
    args = W.cpu_install_args("https://example.invalid/x86.whl")
    assert args[0] == "https://example.invalid/x86.whl"
    assert "--extra-index-url" in args
    assert "download.pytorch.org/whl/cpu" in " ".join(args)
