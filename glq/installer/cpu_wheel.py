"""Resolve the vLLM CPU wheel for a CPU-only install.

vLLM publishes its CPU backend as GitHub-release wheels (a `+cpu` local version), not on
PyPI — `pip install vllm` always resolves the CUDA build. The asset name has already
drifted once in the wild (circulating docs said `manylinux_2_35`; the real assets are
`manylinux_2_34`, a 404 at install time), so this scans the latest release's **asset
list** for a matching name — proving the file exists before pip ever sees a URL — and
falls back to a pinned known-good wheel on any failure at all (network, the 60/hr
unauthenticated API rate limit, naming drift). The fallback announces itself: silently
installing an older vLLM than the user expects is a support ticket.

`fetch` is injected for tests, following discovery.py's pattern.
"""
from __future__ import annotations

import json
import platform
import urllib.request

_LATEST_API = "https://api.github.com/repos/vllm-project/vllm/releases/latest"
_ASSET_SUFFIX = "+cpu-cp38-abi3-manylinux_2_34_{arch}.whl"

#: Known-good pins — the exact wheels validated end to end (SmolLM3 + gemma-4-E4B served
#: on the CPU backend). Bump when a newer release is validated.
FALLBACK_X86 = ("https://github.com/vllm-project/vllm/releases/download/v0.28.0/"
                "vllm-0.28.0+cpu-cp38-abi3-manylinux_2_34_x86_64.whl")
FALLBACK_AARCH64 = ("https://github.com/vllm-project/vllm/releases/download/v0.28.0/"
                    "vllm-0.28.0+cpu-cp38-abi3-manylinux_2_34_aarch64.whl")

#: The torch index the +cpu wheel's dependencies resolve against.
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as resp:   # noqa: S310 - pinned https
        return json.loads(resp.read().decode())


def wheel_arch(machine=platform.machine) -> str | None:
    """The wheel arch tag for this machine, or None when vLLM ships no CPU wheel for it."""
    m = machine()
    if m in ("x86_64", "AMD64"):
        return "x86_64"
    if m in ("aarch64", "arm64"):
        return "aarch64"
    return None


def latest_cpu_wheel_url(arch: str, fetch=_fetch_json) -> str:
    """The newest release's +cpu wheel URL for `arch`, else the pinned fallback."""
    fallback = FALLBACK_X86 if arch == "x86_64" else FALLBACK_AARCH64
    suffix = _ASSET_SUFFIX.format(arch=arch)
    try:
        release = fetch(_LATEST_API)
        for asset in release.get("assets", []):
            if asset.get("name", "").endswith(suffix):
                # The API percent-encodes the `+` in the local version (%2B). pip accepts
                # either; decode so the printed command is greppable and human-readable.
                return asset["browser_download_url"].replace("%2B", "+")
    except Exception:                                       # noqa: BLE001 - fallback path
        pass
    print(f"  (could not resolve the latest vLLM CPU wheel — using the pinned "
          f"known-good {fallback.rsplit('/', 2)[-2]})")
    return fallback


def cpu_install_args(wheel_url: str) -> list[str]:
    """pip arguments installing the CPU wheel with its torch deps from the CPU index."""
    return [wheel_url, "--extra-index-url", PYTORCH_CPU_INDEX]
