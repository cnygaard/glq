"""The on-disk checkpoint cache, so a bare `glq-code` is not 2N+1 HTTP calls.

`discover()` makes one call for the collection and two per repo (config.json and the file
tree), each with a 20 s timeout and no retry. That is the right cost for `glq-setup`, which
runs once and must always see the curator's latest. It is the wrong cost for `glq-code`,
which is about to show a picker and should start promptly every time.

Two properties carry most of the value, and neither is obvious:

**Stale-on-failure.** An expired cache that cannot be refreshed falls back to its stale
entries rather than to nothing. That is what makes an offline `glq-code` usable at all —
expiry means "try again", not "discard".

**A cache that cannot be written must never fail the command.** It is an optimisation; a
read-only or full `$GLQ_HOME` is not a reason to refuse to run.

`glq-setup` keeps calling `discover()` directly, so the installer menu never shows a stale
list. That is asserted here too, because it is the kind of thing a later refactor collapses.
"""
from __future__ import annotations

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.installer import discovery  # noqa: E402

COLL = f"https://huggingface.co/api/collections/{discovery.COLLECTION_SLUG}"


def _fetcher(calls, *, ids=("org/a",), size=1 << 30):
    """A fetch that records every URL it is asked for, so 'did not call the network' is
    asserted on the mechanism rather than on elapsed time."""
    def fetch(url):
        calls.append(url)
        if url.startswith(COLL):
            return {"items": [{"type": "model", "id": i} for i in ids]}
        if url.endswith("config.json"):
            return {"quantization_config": {"codebook": "trellis"}}
        if "/tree/" in url:
            return [{"type": "file", "path": "model.safetensors", "size": size}]
        raise AssertionError(f"unexpected url {url}")
    return fetch


def _exploding(url):
    raise AssertionError(f"the cache should have answered, but it fetched {url}")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("GLQ_HOME", str(tmp_path))
    return tmp_path


# ---- the point of it -------------------------------------------------------------------

def test_the_second_call_makes_no_requests(home):
    calls = []
    first = discovery.cached_discover(fetch=_fetcher(calls))
    assert calls, "the first call must actually fetch"
    second = discovery.cached_discover(fetch=_exploding)
    assert [c.repo_id for c in second] == [c.repo_id for c in first]


def test_every_checkpoint_field_survives_the_round_trip(home):
    """A cache that loses `expert_offload_bytes` would make the picker offer a checkpoint as
    unservable that is perfectly servable with offload — silently, and only on second run."""
    # Compared against what `discover()` itself produced, not against a hand-written
    # Checkpoint: hardcoding the expected traits here would assert my model of
    # `_traits_from_cfg` rather than the round-trip, and would fail whenever that parser
    # legitimately changed.
    live = discovery.discover(fetch=_fetcher([]))
    cached_path = discovery.cache_path()
    cached_path.unlink(missing_ok=True)
    discovery.cached_discover(fetch=_fetcher([]))
    from_cache = discovery.cached_discover(fetch=_exploding)
    assert from_cache == live, f"{from_cache} != {live}"


# ---- invalidation ----------------------------------------------------------------------

def test_an_expired_cache_refetches(home):
    discovery.cached_discover(fetch=_fetcher([]))
    calls = []
    discovery.cached_discover(fetch=_fetcher(calls), now=lambda: time.time() + 10 ** 6)
    assert calls, "past the TTL the list must be refreshed"


def test_a_corrupt_cache_refetches_rather_than_raising(home):
    discovery.cache_path().write_text("{not json")
    calls = []
    assert discovery.cached_discover(fetch=_fetcher(calls))
    assert calls


def test_a_different_collection_invalidates(home):
    """Retargeting COLLECTION_SLUG must not serve the previous collection's list."""
    discovery.cached_discover(fetch=_fetcher([]))
    doc = json.loads(discovery.cache_path().read_text())
    doc["collection"] = "xv0y5ncu/some-other-collection"
    discovery.cache_path().write_text(json.dumps(doc))
    calls = []
    discovery.cached_discover(fetch=_fetcher(calls))
    assert calls


def test_an_older_schema_invalidates(home):
    discovery.cached_discover(fetch=_fetcher([]))
    doc = json.loads(discovery.cache_path().read_text())
    doc["version"] = discovery.CACHE_VERSION - 1
    discovery.cache_path().write_text(json.dumps(doc))
    calls = []
    discovery.cached_discover(fetch=_fetcher(calls))
    assert calls


# ---- degrading ---------------------------------------------------------------------

def test_an_expired_cache_that_cannot_refresh_returns_the_STALE_list(home):
    """Expiry means "try again", not "discard". This is what an offline glq-code gets."""
    discovery.cached_discover(fetch=_fetcher([], ids=("org/cached",)))

    def offline(url):
        raise discovery.DiscoveryError("no network")

    got = discovery.cached_discover(fetch=offline, now=lambda: time.time() + 10 ** 6)
    assert [c.repo_id for c in got] == ["org/cached"]


def test_a_first_run_with_no_network_still_raises(home):
    """No cache and no network is a real failure; the caller needs to be able to say so."""
    def offline(url):
        raise discovery.DiscoveryError("no network")

    with pytest.raises(discovery.DiscoveryError):
        discovery.cached_discover(fetch=offline)


def test_an_unwritable_cache_does_not_fail_the_call(home, monkeypatch):
    """The cache is an optimisation. A read-only $GLQ_HOME is not a reason to refuse."""
    def no_write(*_a, **_kw):
        raise OSError("read-only file system")

    monkeypatch.setattr(discovery, "_write_cache", no_write)
    assert discovery.cached_discover(fetch=_fetcher([]))


# ---- the installer must not be affected -------------------------------------------------

def test_glq_setup_still_calls_discover_directly():
    """The installer menu must always show the curator's latest, so it keeps paying the
    full cost. Asserted because this is exactly the sort of thing a later tidy-up unifies."""
    import inspect

    from glq.installer import __main__ as installer_main
    src = inspect.getsource(installer_main)
    assert "discovery.discover(" in src
    assert "cached_discover" not in src
