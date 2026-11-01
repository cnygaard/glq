"""Does a multi-turn session reuse its prefix? One shared ~20k-token prefix, four requests.

    python prefix_turns.py [prefix tokens]      # against the server on localhost:8000

A coding agent re-sends the whole conversation every turn. With prefill at ~140-200 tokens/s
under expert offload, a turn that misses the prefix cache re-prefills everything, so a 100k
session costs ~10 minutes per turn. This measures what the server actually reuses.

Requests, in order: turn 1; turn 1 again, byte-identical (the best case); turn 2 and turn 3,
each carrying the earlier turns as history, the way an agent does. Per request it prints TTFT
and the change in the server's prefix-cache counters, read from /metrics around it. NemotronH
caches Mamba state only at block boundaries ("align" mode, 2,192-token blocks here), so hits
come in whole blocks.

Thinking off and temperature 0, so the replies are short and repeatable.
"""
import glob
import sys
import time
import urllib.request

from openai import OpenAI
from transformers import AutoTokenizer

M = "xv0y5ncu/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-3bpw"
SNAP = glob.glob("/opt/dlami/nvme/hf_cache/hub/models--" + M.replace("/", "--") + "/snapshots/*")[0]
COUNTERS = ("vllm:prefix_cache_queries", "vllm:prefix_cache_hits")
QUESTIONS = (
    "Which function in this module computes a checksum? Answer in one sentence.",
    "Now name the function that parses a header line, and say what it returns. One sentence.",
    "Which constant sets the retry limit, and what is its value? One sentence.",
)

tok = AutoTokenizer.from_pretrained(SNAP)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none", timeout=3600)


def counters() -> dict[str, float]:
    text = urllib.request.urlopen("http://localhost:8000/metrics", timeout=30).read().decode()
    out = dict.fromkeys(COUNTERS, 0.0)
    for line in text.splitlines():
        for name in COUNTERS:
            if line.startswith(name + "_total"):
                out[name] += float(line.rsplit(" ", 1)[1])
    return out


def module(n_tokens: int) -> str:
    """A deterministic, code-shaped file of about `n_tokens` tokens."""
    parts = ["RETRY_LIMIT = 7\n\n"]
    i = 0
    while True:
        parts.append(
            f"def handler_{i}(record, offset={i % 13}):\n"
            f"    \"\"\"Shift field {i % 17} of the record by the offset and return it.\"\"\"\n"
            f"    value = record.get('field_{i % 17}', 0) + offset\n"
            f"    return {{'id': {i}, 'value': value}}\n\n")
        if i == 40:
            parts.append("def checksum(data):\n    return sum(data) % 65521\n\n")
        if i == 80:
            parts.append("def parse_header(line):\n    key, _, val = line.partition(':')\n"
                         "    return key.strip(), val.strip()\n\n")
        i += 1
        if i % 50 == 0 and len(tok("".join(parts))["input_ids"]) >= n_tokens:
            return "".join(parts)


def ask(messages: list[dict], label: str) -> str:
    before = counters()
    t0 = time.perf_counter()
    first, parts, usage = None, [], None
    stream = client.chat.completions.create(
        model=M, messages=messages, max_tokens=96, temperature=0, stream=True,
        stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    for ch in stream:
        if ch.usage:
            usage = ch.usage
        if ch.choices and ch.choices[0].delta.content:
            if first is None:
                first = time.perf_counter()
            parts.append(ch.choices[0].delta.content)
    after = counters()
    q, h = (after[k] - before[k] for k in COUNTERS)
    ttft = (first or time.perf_counter()) - t0
    print(f"TURN label={label} prompt_tokens={usage.prompt_tokens} ttft_s={ttft:.1f} "
          f"cache_queries={q:.0f} cache_hits={h:.0f} "
          f"hit_rate={h / q if q else float('nan'):.3f} "
          f"uncached_prefill_tokps={(usage.prompt_tokens - h) / ttft:.0f}", flush=True)
    text = "".join(parts)
    print("TEXT", repr(text[:200]), flush=True)
    return text


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    code = module(n)
    first_msg = {"role": "user", "content": "Here is a Python module:\n\n" + code + QUESTIONS[0]}
    ask([{"role": "user", "content": "Say hello."}], "warmup")
    history = [first_msg]
    reply = ask(history, "turn1")
    ask(history, "turn1_repeat")
    for i, question in enumerate(QUESTIONS[1:], start=2):
        history += [{"role": "assistant", "content": reply},
                    {"role": "user", "content": question}]
        reply = ask(history, f"turn{i}")
