"""Long-prompt check of a speculative-decoding serve: one needle at 60% depth.

    python needle_spec.py <prompt tokens>      # against the server on localhost:8000

Adapted from `benchmarks/supervl3_offload_24gb/needle_262k.py`, without its 30-minute skip:
the point here is to run the long prompt with the draft model loaded. A 16k probe runs first.
Thinking off and temperature 0 so the answer is short and deterministic; ignore_eos so the
decode rate covers a fixed 256 tokens. The server's speculative counters are read from
/metrics around each request, so acceptance at long context is reported per request.
"""
import glob
import random
import sys
import time
import urllib.request

from openai import OpenAI
from transformers import AutoTokenizer

M = "xv0y5ncu/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-3bpw"
SNAP = glob.glob("/opt/dlami/nvme/hf_cache/hub/models--" + M.replace("/", "--") + "/snapshots/*")[0]
NEEDLE = "The passphrase for the vault is amber-falcon-4127."
QUESTION = ("\n\nWhat is the passphrase for the vault? Give the passphrase first, then "
            "summarise the records above in about 150 words.")
ITEMS = ["copper wire", "rice", "olive oil", "machine parts", "wool", "coffee", "timber",
         "glassware", "paper", "salt", "tea", "cotton", "steel pipe", "medicine", "spices"]
PORTS = ["Lisbon", "Valparaiso", "Mombasa", "Busan", "Rotterdam", "Halifax", "Durban",
         "Chennai", "Genoa", "Auckland", "Antwerp", "Manila", "Recife", "Oslo", "Tunis"]
COUNTERS = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens",
            "vllm:spec_decode_num_accepted_tokens")

tok = AutoTokenizer.from_pretrained(SNAP)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none", timeout=7200)


def spec_counters() -> dict[str, float]:
    text = urllib.request.urlopen("http://localhost:8000/metrics", timeout=30).read().decode()
    out = dict.fromkeys(COUNTERS, 0.0)
    for line in text.splitlines():
        for name in COUNTERS:
            # Prometheus exports counters as <name>_total{labels} <value>.
            if line.startswith(name + "_total"):
                out[name] += float(line.rsplit(" ", 1)[1])
    return out


def document(n_tokens: int) -> str:
    rng = random.Random(0)
    lines = [f"Record {i}: a shipment of {rng.choice(ITEMS)} left {rng.choice(PORTS)} on day "
             f"{rng.randint(1, 365)} and arrived on schedule.\n" for i in range(n_tokens // 8)]
    while len(tok("".join(lines))["input_ids"]) > n_tokens:
        lines = lines[: int(len(lines) * 0.98)]
    lines.insert(int(len(lines) * 0.6), NEEDLE + "\n")
    return "".join(lines)


def ask(content: str, max_tokens: int, label: str) -> None:
    before = spec_counters()
    t0 = time.perf_counter()
    first, parts, usage = None, [], None
    stream = client.chat.completions.create(
        model=M, messages=[{"role": "user", "content": content}], max_tokens=max_tokens,
        temperature=0, stream=True, stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True})
    for ch in stream:
        if ch.usage:
            usage = ch.usage
        if ch.choices and ch.choices[0].delta.content:
            if first is None:
                first = time.perf_counter()
            parts.append(ch.choices[0].delta.content)
    end = time.perf_counter()
    after = spec_counters()
    d = {k: after[k] - before[k] for k in COUNTERS}
    drafts, drafted, accepted = (d[k] for k in COUNTERS)
    text = "".join(parts)
    ttft = first - t0
    rate = (usage.completion_tokens - 1) / (end - first)
    print(f"NEEDLE label={label} prompt_tokens={usage.prompt_tokens} "
          f"completion_tokens={usage.completion_tokens} ttft_s={ttft:.1f} "
          f"prefill_tokps={usage.prompt_tokens / ttft:.0f} decode_tokps={rate:.2f} "
          f"drafts={drafts:.0f} draft_tokens={drafted:.0f} accepted={accepted:.0f} "
          f"accept_rate={accepted / drafted if drafted else float('nan'):.3f} "
          f"tokens_per_step={1 + accepted / drafts if drafts else float('nan'):.2f} "
          f"found={'amber-falcon-4127' in text}", flush=True)
    print("TEXT", repr(text[:600]), flush=True)


if __name__ == "__main__":
    target = int(sys.argv[1])
    ask("Say hello.", 8, "warmup")
    ask(document(16000) + QUESTION, 256, "probe16k")
    ask(document(target) + QUESTION, 256, f"full{target // 1000}k")
