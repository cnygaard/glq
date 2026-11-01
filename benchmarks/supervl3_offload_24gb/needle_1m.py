"""Long-context check of the 1,048,576-token offload serve: one needle at 60% depth.

A 16k-token probe, then ~256k, then ~1.04M tokens if the 256k prefill projects the full
prompt under an hour. Thinking off and temperature 0 so the answer is short and
deterministic; ignore_eos so the decode rate covers a fixed 256 tokens.
"""
import glob
import random
import time

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

tok = AutoTokenizer.from_pretrained(SNAP)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none", timeout=7200)


def document(n_tokens: int) -> str:
    rng = random.Random(0)

    def line(i: int) -> str:
        return (f"Record {i}: a shipment of {rng.choice(ITEMS)} left {rng.choice(PORTS)} "
                f"on day {rng.randint(1, 365)} and arrived on schedule.\n")

    sample = "".join(line(i) for i in range(2000))
    per_line = len(tok(sample)["input_ids"]) / 2000
    rng = random.Random(0)
    lines = [line(i) for i in range(int(n_tokens / per_line * 0.99))]
    while len(tok("".join(lines))["input_ids"]) > n_tokens:
        lines = lines[: int(len(lines) * 0.995)]
    lines.insert(int(len(lines) * 0.6), NEEDLE + "\n")
    return "".join(lines)


def ask(content: str, max_tokens: int, label: str) -> float:
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
    text = "".join(parts)
    ttft = first - t0
    rate = (usage.completion_tokens - 1) / (end - first)
    print(f"NEEDLE label={label} prompt_tokens={usage.prompt_tokens} "
          f"completion_tokens={usage.completion_tokens} ttft_s={ttft:.1f} "
          f"prefill_tokps={usage.prompt_tokens / ttft:.0f} decode_tokps={rate:.2f} "
          f"found={'amber-falcon-4127' in text}", flush=True)
    print("TEXT", repr(text[:600]), flush=True)
    return ttft


ask("Say hello.", 8, "warmup")
ask(document(16000) + QUESTION, 64, "probe16k")
t256 = ask(document(256000) + QUESTION, 256, "prompt256k")
projected = t256 * 4 * 1.5  # 4x the tokens; 1.5 for attention growing faster than linear
print(f"projected ~1.04M-token TTFT {projected:.0f} s", flush=True)
if projected > 3600:
    print("SKIP 1M prompt: projected over an hour", flush=True)
else:
    ask(document(1040000) + QUESTION, 256, "prompt1040k")
