"""glq-chat shows a thinking model's reasoning as a collapsible "Thinking" block.

Before this, glq-chat started vLLM with no reasoning parser and streamed only `content`. For a
model whose template thinks by default the whole trace landed in the chat bubble — for
Nemotron 3.5 followed by a literal `</think>`, which is not a special token there — and the
1,024-token default reply limit could run out before the answer began.

The display follows Gradio's agents-and-tool-usage guide: the reasoning is its own assistant
message with `metadata={"title": ...}` (rendered as an accordion), the answer a second message,
and the full list is yielded on every update. Gradio-free: these are the helpers `respond`
delegates to, so they run where the `chat` extra is not installed (CI).
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq import chat as C  # noqa: E402


def _chunk(content=None, reasoning=None, finish=None, field="reasoning"):
    delta = NS(content=content, **{field: reasoning})
    return NS(choices=[NS(delta=delta, finish_reason=finish)])


def _clock(*times):
    it = iter(times)
    return lambda: next(it)


def test_a_model_that_does_not_think_still_gets_a_plain_string():
    """Every family without a reasoning block must behave exactly as before."""
    out = list(C.stream_reply([_chunk("Hel"), _chunk("lo"), _chunk(finish="stop")]))
    assert out == ["Hel", "Hello", "Hello"]


def test_reasoning_and_answer_become_two_messages():
    stream = [_chunk(reasoning="Let me "), _chunk(reasoning="think."),
              _chunk(content="Oslo"), _chunk(content=" is cold.", finish="stop")]
    *_, last = C.stream_reply(stream, clock=_clock(10.0, 12.34))
    thought, answer = last
    assert thought["role"] == answer["role"] == "assistant"
    assert thought["content"] == "Let me think."
    assert thought["metadata"]["title"] == "Thinking"
    assert answer == {"role": "assistant", "content": "Oslo is cold."}
    assert "metadata" not in answer


def test_the_thought_is_pending_while_thinking_and_done_once_the_answer_starts():
    stream = [_chunk(reasoning="a"), _chunk(reasoning="b"), _chunk(content="x")]
    frames = list(C.stream_reply(stream, clock=_clock(1.0, 3.0)))
    assert [f[0]["metadata"]["status"] for f in frames[:2]] == ["pending", "pending"]
    assert len(frames[1]) == 1, "no answer message before the first answer token"
    assert frames[2][0]["metadata"]["status"] == "done"
    assert frames[2][0]["metadata"]["duration"] == 2.0


def test_the_thinking_time_is_rounded_to_a_tenth_of_a_second():
    *_, last = C.stream_reply([_chunk(reasoning="a"), _chunk(content="x")],
                              clock=_clock(0.0, 1.2345))
    assert last[0]["metadata"]["duration"] == 1.2


def test_older_servers_send_reasoning_content():
    """vLLM 0.32 calls the field `reasoning`; earlier releases called it
    `reasoning_content`. Both must be read, or an older server's reasoning vanishes."""
    stream = [_chunk(reasoning="hmm", field="reasoning_content"), _chunk(content="ok")]
    *_, last = C.stream_reply(stream, clock=_clock(0.0, 1.0))
    assert last[0]["content"] == "hmm" and last[1]["content"] == "ok"


def test_running_out_of_tokens_while_thinking_says_so():
    """The silent version of this is a chat that shows a thought and then nothing at all."""
    stream = [_chunk(reasoning="long..."), _chunk(reasoning="more", finish="length")]
    *_, last = C.stream_reply(stream, clock=_clock(0.0, 5.0))
    assert last[0]["metadata"]["status"] == "done"
    assert "max tokens" in last[1]["content"]


def test_a_finished_answer_is_not_mistaken_for_a_truncated_thought():
    stream = [_chunk(reasoning="a"), _chunk(content="done", finish="length")]
    *_, last = C.stream_reply(stream, clock=_clock(0.0, 1.0))
    assert last[1]["content"] == "done"


def test_chunks_without_choices_are_skipped():
    """A usage-only chunk at the end of a stream carries no choices."""
    out = list(C.stream_reply([_chunk("hi"), NS(choices=[])]))
    assert out[-1] == "hi"


def test_thoughts_are_never_sent_back_to_the_model():
    history = [
        {"role": "user", "content": "Weather in Oslo?"},
        {"role": "assistant", "content": "Let me think.",
         "metadata": {"title": "Thinking", "status": "done"}},
        {"role": "assistant", "content": "Cold.", "metadata": None},
    ]
    assert C.history_messages(history) == [
        {"role": "user", "content": "Weather in Oslo?"},
        {"role": "assistant", "content": "Cold."},
    ]


def test_history_content_in_parts_is_passed_through():
    """Gradio 6 normalizes message content to a list of parts; the OpenAI API accepts it."""
    parts = [{"type": "text", "text": "Hi"}]
    assert C.history_messages([{"role": "user", "content": parts, "metadata": {}}]) == [
        {"role": "user", "content": parts}]
