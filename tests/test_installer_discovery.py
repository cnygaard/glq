"""Checkpoint discovery from the published HF collection (glq/installer/discovery.py).

install.sh must not hardcode a model list: the collection is curated on the Hub and the
installer should follow it, so adding a checkpoint there reaches users without a release.

Two Hub endpoints, and the second one is not the obvious one. `/api/models/{id}` carries a
`usedStorage` field that looks like exactly what we want, but it reported **0** for
`Gemma-4-31B-it-GLQ-5.0bpw-mix3-8` (measured 2026-08-15) — sizing on it would have silently
called a 22 GiB checkpoint free and recommended it to a 4 GiB card. So size comes from
`/api/models/{id}/tree/main?recursive=true`, summing the `.safetensors` entries.

Both endpoints are stubbed here: the tests must not touch huggingface.co, or CI fails
whenever the Hub is slow.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.installer import discovery as D  # noqa: E402

# Shape mirrors the real responses (verified against the live API 2026-08-15).
_COLLECTION = {
    "title": "Start here - recommended GLQ checkpoints",
    "items": [
        {"type": "model", "id": "xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel",
         "gated": False, "private": False},
        {"type": "model", "id": "xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw",
         "gated": False, "private": False},
        {"type": "paper", "id": "2510.20984"},
        {"type": "model", "id": "xv0y5ncu/secret-wip", "gated": False, "private": True},
        {"type": "model", "id": "xv0y5ncu/needs-eula", "gated": "auto", "private": False},
    ],
}

_TREES = {
    "xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel": [
        {"type": "file", "path": "config.json", "size": 1_400},
        {"type": "file", "path": "model.safetensors", "size": 1_932_735_283},
        {"type": "file", "path": "README.md", "size": 8_000},
    ],
    "xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw": [
        {"type": "file", "path": "model-00001-of-00002.safetensors", "size": 3_221_225_472},
        {"type": "file", "path": "model-00002-of-00002.safetensors", "size": 3_006_477_107},
        {"type": "file", "path": "tokenizer.json", "size": 17_000_000},
    ],
}


def _fake_fetch(url: str):
    """Stand-in for the JSON fetcher, dispatching on the two endpoint shapes."""
    if "/api/collections/" in url:
        return _COLLECTION
    for repo, tree in _TREES.items():
        if f"/api/models/{repo}/tree/" in url:
            return tree
    raise AssertionError(f"unexpected URL: {url}")


def test_only_models_are_returned():
    """The collection can hold papers, datasets and Spaces; feeding a paper id to
    `vllm serve` would fail long after the user chose it."""
    ids = D.collection_repo_ids(fetch=_fake_fetch)
    assert "2510.20984" not in ids
    assert all(i.startswith("xv0y5ncu/") for i in ids)


def test_private_and_gated_are_skipped():
    """A first-time user has no token and has signed no EULA. Offering a repo they cannot
    download turns a working install into a 401 halfway through."""
    ids = D.collection_repo_ids(fetch=_fake_fetch)
    assert "xv0y5ncu/secret-wip" not in ids
    assert "xv0y5ncu/needs-eula" not in ids
    assert ids == [
        "xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel",
        "xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw",
    ]


def test_collection_order_is_preserved():
    """The collection is hand-ordered 'start here' first; that curation is the whole point
    of reading it, so discovery must not sort it away."""
    assert D.collection_repo_ids(fetch=_fake_fetch)[0].endswith("trellis-3inst-4bpw-kernel")


def test_size_sums_only_safetensors():
    """tokenizer.json is 17 MB and config.json is noise; counting them inflates the figure
    the fit decision is made from."""
    n = D.repo_size_bytes("xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw", fetch=_fake_fetch)
    assert n == 3_221_225_472 + 3_006_477_107


def test_size_handles_single_file_repo():
    n = D.repo_size_bytes("xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel", fetch=_fake_fetch)
    assert n == 1_932_735_283


def test_discover_pairs_each_repo_with_its_size():
    cks = D.discover(fetch=_fake_fetch)
    assert [c.repo_id for c in cks] == [
        "xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel",
        "xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw",
    ]
    assert cks[0].size_bytes == 1_932_735_283
    assert cks[1].size_bytes == 3_221_225_472 + 3_006_477_107


def test_size_is_reported_in_gib_for_display():
    cks = D.discover(fetch=_fake_fetch)
    assert cks[0].size_gib == pytest.approx(1.8, abs=0.05)


def test_a_repo_with_no_safetensors_is_size_zero_not_a_crash():
    """A malformed or LFS-pending repo must not abort discovery of the other eight."""
    def fetch(url):
        if "/api/collections/" in url:
            return {"items": [{"type": "model", "id": "xv0y5ncu/empty",
                               "gated": False, "private": False}]}
        return [{"type": "file", "path": "README.md", "size": 10}]
    cks = D.discover(fetch=fetch)
    assert len(cks) == 1 and cks[0].size_bytes == 0


def test_network_failure_raises_a_named_error_not_a_urlerror():
    """install.sh needs to print something a human can act on, not a raw traceback."""
    def boom(url):
        raise OSError("Network is unreachable")
    with pytest.raises(D.DiscoveryError, match="collection"):
        D.collection_repo_ids(fetch=boom)


def test_collection_slug_points_at_the_published_collection():
    """Guard against a typo silently yielding an empty picker."""
    assert D.COLLECTION_SLUG == "xv0y5ncu/start-here-recommended-glq-checkpoints"


# ------------------------------------------------------------------ trellis
# Trellis checkpoints decode much faster (single-stream at bf16 parity), so the recommender
# prefers them. Which repo *is* trellis is read from the checkpoint's own
# config.json -> quantization_config, never from the repo name: names are a convention, and
# this repo has already been bitten once by a name heuristic standing in for a capability
# check (the wikitext2_ppl "ConditionalGeneration" refusal). Marker values below are the
# real ones, read from the Hub 2026-08-15.

_CONFIGS = {
    "xv0y5ncu/trellis-one": {"quantization_config": {
        "quant_method": "glq", "variant": "3inst", "trellis_layout": "kernel"}},
    "xv0y5ncu/shell-one": {"quantization_config": {
        "quant_method": "glq", "bpw": 5.0}},
    "xv0y5ncu/no-quant-config": {"model_type": "llama"},
}


def _cfg_fetch(url: str):
    for repo, cfg in _CONFIGS.items():
        if f"/{repo}/resolve/" in url:
            return cfg
    raise AssertionError(f"unexpected URL: {url}")


def test_trellis_is_detected_from_the_quantization_config():
    assert D.repo_is_trellis("xv0y5ncu/trellis-one", fetch=_cfg_fetch) is True


def test_a_shell_checkpoint_is_not_trellis():
    assert D.repo_is_trellis("xv0y5ncu/shell-one", fetch=_cfg_fetch) is False


def test_a_config_without_quantization_config_is_not_trellis():
    assert D.repo_is_trellis("xv0y5ncu/no-quant-config", fetch=_cfg_fetch) is False


def test_an_unreadable_config_is_unknown_not_false():
    """None and False mean different things to the recommender: False is 'confirmed slow
    path', None is 'we could not tell'. Collapsing them would let a network blip silently
    demote a trellis checkpoint."""
    def boom(url):
        raise OSError("connection reset")
    assert D.repo_is_trellis("xv0y5ncu/whatever", fetch=boom) is None


def test_repo_name_alone_does_not_decide():
    """A repo called '...-trellis-...' whose config says otherwise must not be trusted;
    the config is the authority."""
    def fetch(url):
        return {"quantization_config": {"quant_method": "glq", "bpw": 4}}
    assert D.repo_is_trellis("xv0y5ncu/looks-like-trellis-3inst-4bpw", fetch=fetch) is False


# --------------------------------------------------------------- authenticated lookups

def test_auth_header_comes_from_the_env_token(monkeypatch):
    """Private checkpoints 401 on the anonymous tree call, which silently degrades
    glq-chat's pool sizing to the 0.45 default — fine on a 96 GB card, fatal on a 24 GB
    one serving a 17 GiB model."""
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    assert D._auth_headers() == {"Authorization": "Bearer hf_secret"}


def test_auth_header_falls_back_to_the_stored_login(monkeypatch, tmp_path):
    """`hf auth login` stores the token under $HF_HOME/token; read the same place the hub
    client does so a logged-in box needs no env plumbing."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    (tmp_path / "token").write_text("hf_stored\n")
    assert D._auth_headers() == {"Authorization": "Bearer hf_stored"}


def test_no_token_anywhere_means_no_header(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert D._auth_headers() == {}


# ---------------------------------------------------------------- declared context max

def test_model_max_len_reads_the_top_level_key():
    """The declared maximum clamps the auto-sized serving window: vLLM refuses a
    --max-model-len above it, so tiering up past it breaks serving outright."""
    fetch = lambda url: {"max_position_embeddings": 65536}  # noqa: E731
    assert D.model_max_len("org/m", fetch=fetch) == 65536


def test_model_max_len_descends_into_text_config():
    """gemma-4 and Qwen wrappers declare the text limits on text_config, not the top."""
    fetch = lambda url: {"text_config": {"max_position_embeddings": 262144}}  # noqa: E731
    assert D.model_max_len("org/m", fetch=fetch) == 262144


def test_model_max_len_is_none_when_absent_or_unreachable():
    assert D.model_max_len("org/m", fetch=lambda url: {"hidden_size": 64}) is None
    def boom(url):
        raise OSError("offline")
    assert D.model_max_len("org/m", fetch=boom) is None
