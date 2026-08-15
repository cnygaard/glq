"""The Gradio chat UI actually constructs (glq/chat.py).

Written after `glq-chat` crashed on launch with

    TypeError: ChatInterface.__init__() got an unexpected keyword argument 'type'

on gradio 6.24. `type="messages"` was *required* in Gradio 5 and *removed* in 6, where the
messages format became the only one. Nothing in the earlier tests touched Gradio, so the
break only surfaced when the binary was run on a box — the failure mode this file closes.

Skipped when gradio is absent (it lives in the `chat` extra, and CI installs only
torch + glq[hub]), so this guards anyone who has the extra, including the box.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

gr = pytest.importorskip("gradio")
pytest.importorskip("openai")

from glq import chat as C  # noqa: E402

MODELS = ["xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel", "xv0y5ncu/other"]


def test_build_ui_constructs_against_the_installed_gradio():
    """The regression itself: this raised TypeError on gradio 6 before the fix. It needs no
    server — construction is where the signature mismatch bites."""
    demo = C.build_ui("http://127.0.0.1:8000/v1", MODELS)
    assert demo is not None


def test_build_ui_survives_an_empty_model_list():
    """vLLM not up yet: the dropdown has nothing to show, and the UI must still render so
    the user can read the 'is vLLM running?' hint rather than meet a traceback."""
    assert C.build_ui("http://127.0.0.1:8000/v1", []) is not None


def test_gradio_major_version_is_the_one_the_code_targets():
    """Pinned as `gradio>=6` in pyproject. If a 5.x ever gets installed the history format
    silently becomes tuples and `respond` would read `turn["role"]` off a tuple."""
    major = int(gr.__version__.split(".")[0])
    assert major >= 6, f"glq/chat.py targets the gradio 6 messages format, got {gr.__version__}"


def test_chat_interface_no_longer_takes_the_removed_type_argument():
    """Pins the exact API fact behind the crash, so a future edit re-adding `type=` fails
    here instead of on a user's machine."""
    import inspect
    assert "type" not in inspect.signature(gr.ChatInterface.__init__).parameters


def test_served_models_reads_the_server_not_the_config(monkeypatch):
    """Preferred over ~/.glq/config.json because a restarted vLLM may be serving something
    else; the dropdown should reflect reality."""
    fake = types.SimpleNamespace(models=types.SimpleNamespace(
        list=lambda: types.SimpleNamespace(
            data=[types.SimpleNamespace(id="a/b"), types.SimpleNamespace(id="c/d")])))
    assert C._served_models(fake) == ["a/b", "c/d"]


def test_served_models_is_empty_when_the_server_is_down():
    """Must not raise — the UI still has to come up so the user can read why."""
    class Down:
        class models:
            @staticmethod
            def list():
                raise ConnectionError("connection refused")
    assert C._served_models(Down()) == []
