"""CPU-only tests for the progress/heartbeat helpers."""
from __future__ import annotations

import io
from contextlib import redirect_stderr

import pytest

from glq.bench import progress


def test_fmt_durations():
    assert progress._fmt(5) == "5s"
    assert progress._fmt(65) == "1m05s"
    assert progress._fmt(3661) == "1h01m01s"


def test_log_ts_prefix_and_flush():
    buf = io.StringIO()
    old = progress._STREAM
    progress._STREAM = buf
    try:
        progress.log_ts("hello world")
    finally:
        progress._STREAM = old
    out = buf.getvalue()
    assert out.startswith("[glq-bench ") and "hello world" in out and out.endswith("\n")


def test_pbar_factory_tracks_and_logs():
    pytest.importorskip("tqdm")
    buf = io.StringIO()
    old = progress._STREAM
    progress._STREAM = buf
    try:
        # huge interval so the background heartbeat never fires during the test;
        # only update()/close() lines are emitted.
        factory = progress.pbar_factory("aime_2026", interval=10_000)
        pbar = factory(total=3, desc="Processed prompts")
        pbar.update(1)
        pbar.update(1)
        assert pbar.n == 2
        pbar.close()
    finally:
        progress._STREAM = old
    lines = [l for l in buf.getvalue().splitlines() if "aime_2026" in l]
    assert lines, "expected at least one heartbeat line"
    assert "1/3 done" in buf.getvalue()
    assert "3/3 done" in buf.getvalue() or "2/3 done" in buf.getvalue()
