"""Discover published GLQ checkpoints from the curated HF collection.

The installer reads the collection rather than a hardcoded list so that curating on the Hub
is enough to change what users are offered — no glq release required.

Sizing note, learned the hard way: `/api/models/{id}` exposes `usedStorage`, which looks
like the field to use and is *wrong* — it returned 0 for the 31B checkpoint (measured
2026-08-15). Sizing on it would mark a 22 GiB model as free and recommend it to a small
card. Sizes therefore come from the file tree, summing `.safetensors` entries.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

COLLECTION_SLUG = "xv0y5ncu/start-here-recommended-glq-checkpoints"
_API = "https://huggingface.co/api"
_TIMEOUT = 20
GIB = 1024 ** 3


class DiscoveryError(RuntimeError):
    """Network or API failure, with a message an installer can print verbatim."""


@dataclass(frozen=True)
class Checkpoint:
    repo_id: str
    size_bytes: int
    #: True/False from the checkpoint's own config, None when it could not be read.
    trellis: bool | None = None

    @property
    def size_gib(self) -> float:
        return self.size_bytes / GIB

    @property
    def short_name(self) -> str:
        return self.repo_id.split("/", 1)[-1]


def _fetch_json(url: str):
    """Default fetcher. Injected in tests so no test touches huggingface.co."""
    req = urllib.request.Request(url, headers={"User-Agent": "glq-installer"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:      # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def collection_repo_ids(fetch=_fetch_json) -> list[str]:
    """Model ids from the collection, in the curator's order.

    Non-model items (papers, datasets, Spaces) are dropped — feeding a paper id to
    `vllm serve` fails long after the user picked it. Private and gated repos are dropped
    too: a first-time user has no token and has accepted no EULA, so offering those turns a
    working install into a 401 mid-download.
    """
    try:
        data = fetch(f"{_API}/collections/{COLLECTION_SLUG}")
    except Exception as exc:                                          # noqa: BLE001
        raise DiscoveryError(
            f"could not read the GLQ collection ({COLLECTION_SLUG}): {exc}") from exc

    ids = []
    for item in data.get("items", []):
        if item.get("type") != "model":
            continue
        if item.get("private") or item.get("gated"):
            continue
        ids.append(item["id"])
    return ids


def repo_size_bytes(repo_id: str, fetch=_fetch_json) -> int:
    """Total `.safetensors` bytes for a repo, 0 if the tree exposes none.

    0 is returned rather than raised so one malformed repo cannot abort discovery of the
    others; `recommend.rank` refuses to recommend a zero-size entry.
    """
    try:
        tree = fetch(f"{_API}/models/{repo_id}/tree/main?recursive=true")
    except Exception:                                                 # noqa: BLE001
        return 0
    return sum(f.get("size", 0) for f in tree
               if str(f.get("path", "")).endswith(".safetensors"))


def repo_is_trellis(repo_id: str, fetch=_fetch_json) -> bool | None:
    """Whether a checkpoint uses the trellis codebook, per its own config.json.

    Read from `quantization_config` rather than inferred from the repo name. Names are a
    convention that can drift; the config is what the loader actually dispatches on, and
    this repo has already been bitten once by a name heuristic standing in for a capability
    check. The markers live in **config.json**, not quantize_config.json.

    None means "could not tell" (network failure, missing file) and is deliberately distinct
    from False, so a blip cannot silently demote a trellis checkpoint.
    """
    try:
        cfg = fetch(f"https://huggingface.co/{repo_id}/resolve/main/config.json")
    except Exception:                                                 # noqa: BLE001
        return None
    # Anything but a JSON object means the Hub handed back something we don't understand
    # (an error page, an LFS pointer, a redirect body) — unknown, not "not trellis".
    if not isinstance(cfg, dict):
        return None
    q = cfg.get("quantization_config")
    if not isinstance(q, dict):
        return False
    return bool(q.get("variant") or q.get("trellis_layout"))


def discover(fetch=_fetch_json) -> list[Checkpoint]:
    """Every offerable checkpoint, with its on-disk size and whether it is trellis."""
    return [Checkpoint(rid, repo_size_bytes(rid, fetch=fetch),
                       repo_is_trellis(rid, fetch=fetch))
            for rid in collection_repo_ids(fetch=fetch)]
