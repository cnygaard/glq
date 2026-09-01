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
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

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
    #: Mixture-of-experts, from the same config read (num_local_experts / num_experts /
    #: n_routed_experts > 1). None = could not tell. The CPU backend refuses MoE, so the
    #: CPU recommendation gate treats None as ineligible — conservative by design.
    moe: bool | None = None

    @property
    def size_gib(self) -> float:
        return self.size_bytes / GIB

    @property
    def short_name(self) -> str:
        return self.repo_id.split("/", 1)[-1]


def _auth_headers() -> dict:
    """Authorization for the Hub API when a token is around, else nothing.

    Anonymous is fine for the public collection, but glq-chat reuses this fetcher to size
    PRIVATE checkpoints for the KV-pool plan — and the anonymous tree call 401s there, so
    the plan silently degrades to the 0.45 default. Fine on a 96 GB card, fatal on a
    24 GB one serving a 17 GiB model. Plain stdlib on purpose: the installer's core
    profile has no huggingface_hub, so read the same places its client would — the env
    var first, then the token `hf auth login` stores.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        stored = Path(os.environ.get("HF_HOME")
                      or Path.home() / ".cache" / "huggingface") / "token"
        try:
            token = stored.read_text().strip()
        except OSError:
            token = None
    return {"Authorization": f"Bearer {token}"} if token else {}


def _fetch_json(url: str):
    """Default fetcher. Injected in tests so no test touches huggingface.co."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "glq-installer", **_auth_headers()})
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


def model_max_len(repo_id: str, fetch=_fetch_json):
    """The model's declared context maximum (max_position_embeddings), or None.

    This clamps the auto-sized serving window: vLLM refuses a --max-model-len above the
    declared maximum, so tiering up past it breaks serving outright — SmolLM2-class
    models declare 8192. Multimodal wrappers (gemma-4, Qwen3.5) declare the text limits
    on text_config. None on any failure, same contract as repo_size_bytes: the caller
    falls back to the conservative floor rather than aborting a start.
    """
    try:
        cfg = fetch(f"https://huggingface.co/{repo_id}/resolve/main/config.json")
        val = cfg.get("max_position_embeddings") or             (cfg.get("text_config") or {}).get("max_position_embeddings")
        return int(val) if val else None
    except Exception:                                                 # noqa: BLE001
        return None


_MOE_KEYS = ("num_local_experts", "num_experts", "n_routed_experts")


def repo_traits(repo_id: str, fetch=_fetch_json) -> tuple[bool | None, bool | None]:
    """(trellis, moe) for a checkpoint, from ONE read of its config.json.

    Both read from the config rather than inferred from the repo name. Names are a
    convention that can drift; the config is what the loader actually dispatches on, and
    this repo has already been bitten once by a name heuristic standing in for a capability
    check. Trellis markers live in **config.json → quantization_config**; MoE-ness is any
    expert-count key > 1 (checked on the top level and on text_config, where multimodal
    wrappers keep the LM config).

    None means "could not tell" (network failure, malformed body) and is deliberately
    distinct from False, so a blip cannot silently demote a checkpoint.
    """
    try:
        cfg = fetch(f"https://huggingface.co/{repo_id}/resolve/main/config.json")
    except Exception:                                                 # noqa: BLE001
        return None, None
    # Anything but a JSON object means the Hub handed back something we don't understand
    # (an error page, an LFS pointer, a redirect body) — unknown, not "not trellis".
    if not isinstance(cfg, dict):
        return None, None
    q = cfg.get("quantization_config")
    trellis = (bool(q.get("variant") or q.get("trellis_layout"))
               if isinstance(q, dict) else False)
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    moe = any(
        isinstance(src.get(key), int) and src.get(key) > 1
        for src in (cfg, text) for key in _MOE_KEYS)
    return trellis, moe


def repo_is_trellis(repo_id: str, fetch=_fetch_json) -> bool | None:
    """Back-compat single-trait read; new callers want repo_traits (same one fetch)."""
    return repo_traits(repo_id, fetch=fetch)[0]


def discover(fetch=_fetch_json) -> list[Checkpoint]:
    """Every offerable checkpoint: on-disk size, trellis-ness and MoE-ness (one config
    fetch per repo for both traits)."""
    out = []
    for rid in collection_repo_ids(fetch=fetch):
        trellis, moe = repo_traits(rid, fetch=fetch)
        out.append(Checkpoint(rid, repo_size_bytes(rid, fetch=fetch), trellis, moe))
    return out
