"""The chat UI streams (glq/chat.py:make_responder).

A `glq-chat` that blocks for twenty seconds and then dumps a finished paragraph looks
broken, and on a 3B checkpoint at 32k tokens it looks *very* broken. Streaming is the
difference between "it's working" and "it's hung", so it deserves a test rather than an
assumption — and it had neither: `respond` used to be a closure inside `build_ui`, reachable
only by launching a browser, so nothing exercised it.

These tests use a fake OpenAI client, so they need no GPU, no server and no network. What
they pin is the property a user actually perceives: **partial text arrives before the reply
is complete**, and each yield extends the last rather than replacing it.

Gradio 6's ChatInterface consumes a generator and renders the newest yield, so cumulative
(not delta) yields are required — yielding deltas would make the UI show one token at a
time and lose the rest.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("openai")
from glq import chat as C  # noqa: E402


def _chunk(text):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=text))])


class _FakeClient:
    """Stands in for OpenAI(); records what the UI asked for and streams back deltas."""

    def __init__(self, deltas):
        self.deltas, self.seen = deltas, {}
        outer = self

        class _Completions:
            @staticmethod
            def create(**kw):
                outer.seen = kw
                return iter(_chunk(d) for d in outer.deltas)

        self.chat = types.SimpleNamespace(completions=_Completions())


def _drive(deltas=("Par", "is", " is", " the", " capital"), history=None,
           message="capital of France?"):
    client = _FakeClient(deltas)
    respond = C.make_responder(client)
    return client, list(respond(message, history, "xv0y5ncu/some-model", 0.7, 128))


def test_it_yields_before_the_reply_is_finished():
    """The actual user-visible property: text appears while generation is still running."""
    _, frames = _drive()
    assert len(frames) > 1, "a single yield means the UI blocks until the reply completes"


def test_each_yield_extends_the_previous_one():
    """Gradio renders the newest yield, so frames must be cumulative. Yielding bare deltas
    would leave the user staring at the last token alone."""
    _, frames = _drive()
    for earlier, later in zip(frames, frames[1:]):
        assert later.startswith(earlier), f"{later!r} does not extend {earlier!r}"


def test_the_final_frame_is_the_whole_reply():
    _, frames = _drive()
    assert frames[-1] == "Paris is the capital"


def test_streaming_is_actually_requested_from_the_server():
    """stream=True is the mechanism; without it the client blocks and no amount of
    generator plumbing in the UI would help."""
    client, _ = _drive()
    assert client.seen["stream"] is True


def test_sampling_controls_reach_the_server():
    """The UI exposes temperature and max-tokens sliders; they must not be decorative."""
    client, _ = _drive()
    assert client.seen["temperature"] == 0.7
    assert client.seen["max_tokens"] == 128
    assert client.seen["model"] == "xv0y5ncu/some-model"


def test_empty_deltas_do_not_break_the_stream():
    """vLLM sends role-only and finish-reason chunks whose delta.content is None; treating
    those as text would raise mid-reply."""
    client = _FakeClient([None, "Par", None, "is", None])
    frames = list(C.make_responder(client)("q", [], "m", 0.7, 32))
    assert frames[-1] == "Paris"


def test_prior_turns_are_sent_so_the_model_has_context():
    """Without history the assistant answers every message as if it were the first."""
    client, _ = _drive(history=[{"role": "user", "content": "hi"},
                                {"role": "assistant", "content": "hello"}])
    roles = [m["role"] for m in client.seen["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert client.seen["messages"][-1]["content"] == "capital of France?"


def test_no_history_is_fine():
    client, frames = _drive(history=None)
    assert len(client.seen["messages"]) == 1
    assert frames[-1] == "Paris is the capital"
