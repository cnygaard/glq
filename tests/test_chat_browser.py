"""The chat UI streams **in a real browser** (headless Chromium via Playwright).

`tests/test_chat_streaming.py` proves `make_responder` yields cumulative text. It cannot
prove the UI is wired to it: `gr.ChatInterface(respond, …)` could be handed the wrong
callable, the additional_inputs could be mis-ordered, or a Gradio upgrade could buffer the
generator and paint the reply in one go. All of those look identical to a passing unit test
and broken to a user.

So this drives the actual DOM and asserts the transcript **grows over time**.

No GPU and no model: a stub OpenAI-compatible server streams SSE deltas on a deliberate
cadence, standing in for vLLM. That keeps the test a few seconds long and runnable anywhere,
and makes the timing assertion robust rather than a race — real token cadence varies far
more than the stub's.

Selectors use the `elem_id`s set in `glq/chat.py` (`glq-input`, `glq-chatbot`). Gradio's own
class names are generated and change between releases; pinning to them produces a test that
breaks on every bump and gets ignored.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("gradio")
pytest.importorskip("openai")
sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

pytestmark = pytest.mark.slow

#: Deltas the stub streams, and the gap between them. The gap is what makes "did the text
#: grow between two observations" a reliable question instead of a coin flip.
DELTAS = ["Paris", " is", " the", " capital", " of", " France", "."]
DELTA_GAP_S = 0.30
MODEL = "xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel"


class _StubHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible endpoint: /v1/models and a streaming chat completion."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *a):        # keep pytest output readable
        pass

    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            body = json.dumps({"object": "list",
                               "data": [{"id": MODEL, "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def frame(delta=None, finish=None):
            payload = {"id": "stub", "object": "chat.completion.chunk", "model": MODEL,
                       "choices": [{"index": 0, "finish_reason": finish,
                                    "delta": ({"content": delta} if delta is not None
                                              else {"role": "assistant"})}]}
            return f"data: {json.dumps(payload)}\n\n".encode()

        try:
            self.wfile.write(frame())              # role-only chunk, like vLLM's first
            self.wfile.flush()
            for d in DELTAS:
                time.sleep(DELTA_GAP_S)
                self.wfile.write(frame(d))
                self.wfile.flush()
            self.wfile.write(frame(finish="stop"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass                                    # browser navigated away mid-stream


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def stub_server():
    port = _free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), _StubHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}/v1"
    srv.shutdown()


@pytest.fixture(scope="module")
def chat_url(stub_server):
    """Launch the real glq-chat UI against the stub, in-process."""
    from glq.chat import build_ui
    demo = build_ui(stub_server, [MODEL])
    demo.queue()                                    # streaming needs the queue
    _, local_url, _ = demo.launch(server_port=_free_port(), prevent_thread_lock=True,
                                  share=False, quiet=True)
    yield local_url
    demo.close()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as pw:
        try:
            b = pw.chromium.launch(headless=True)
        except Exception as exc:                     # noqa: BLE001
            pytest.skip(f"headless chromium unavailable: {exc}")
        yield b
        b.close()


def _transcript(page) -> str:
    node = page.locator("#glq-chatbot")
    return node.inner_text() if node.count() else ""


def test_the_page_loads_without_console_errors(browser, chat_url):
    """A JS error on load means the UI is dead however well the Python side behaves."""
    page = browser.new_page()
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(chat_url, wait_until="domcontentloaded")
    page.wait_for_selector("#glq-input", timeout=30_000)
    assert not errors, f"console errors on load: {errors[:3]}"
    page.close()


def test_the_expected_controls_are_present(browser, chat_url):
    """The dropdown and sliders are how a user picks a checkpoint and sampling; if they
    vanish behind a Gradio change, chat still 'works' but is no longer configurable."""
    page = browser.new_page()
    page.goto(chat_url, wait_until="domcontentloaded")
    page.wait_for_selector("#glq-input", timeout=30_000)
    for sel in ("#glq-input", "#glq-chatbot", "#glq-model",
                "#glq-temperature", "#glq-max-tokens"):
        assert page.locator(sel).count(), f"missing {sel}"
    page.close()


def test_tokens_appear_incrementally_in_the_browser(browser, chat_url):
    """The claim that only a browser can check: partial text is painted *while* the reply
    is still streaming, rather than appearing all at once at the end."""
    page = browser.new_page()
    page.goto(chat_url, wait_until="domcontentloaded")
    page.wait_for_selector("#glq-input", timeout=30_000)

    # A neutral prompt on purpose: the transcript contains the echoed *user* message too,
    # so asking "what is the capital of France?" makes every substring check match the
    # question rather than the answer — which is exactly how this test first fooled itself.
    page.locator("#glq-input").click()
    page.keyboard.type("hello")
    page.keyboard.press("Enter")

    # Sample far more often than the stub's 300 ms cadence, so a UI that paints once at the
    # end yields a single snapshot containing "Paris" and fails the assertion below.
    snapshots, deadline = [], time.time() + 40
    while time.time() < deadline:
        text = _transcript(page)
        if text and (not snapshots or text != snapshots[-1]):
            snapshots.append(text)
        if "France." in text:                      # only the completed reply has this
            break
        page.wait_for_timeout(100)

    page.close()

    assert snapshots, "nothing was ever rendered in the chatbot"
    partials = [s for s in snapshots if "Paris" in s]
    assert len(partials) > 1, (
        "the reply appeared in a single repaint — the UI is buffering the generator "
        f"instead of streaming it. snapshots={snapshots!r}")

    lengths = [len(s) for s in partials]
    assert lengths == sorted(lengths), f"transcript shrank mid-stream: {lengths}"
    assert "Paris is the capital of France." in snapshots[-1]


def test_the_final_transcript_holds_the_whole_reply(browser, chat_url):
    page = browser.new_page()
    page.goto(chat_url, wait_until="domcontentloaded")
    page.wait_for_selector("#glq-input", timeout=30_000)
    page.locator("#glq-input").click()
    page.keyboard.type("hello")
    page.keyboard.press("Enter")
    page.wait_for_function(
        "() => (document.querySelector('#glq-chatbot')?.innerText || '').includes('France.')",
        timeout=40_000)
    assert "Paris is the capital of France." in _transcript(page)
    page.close()
