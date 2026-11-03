"""Three chat requests against the glq-chat --dspark server, plus vLLM's spec-decode counters.

A smoke of the launcher path (supervisor env, drafter, kolibri1 reasoning parser), not the
headline speed figure: sampling is the card's (temperature 1.0), and each request's rate
includes its prefill.
"""
import json
import time
import urllib.request

URL = "http://127.0.0.1:8000"
PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "A train travels 180 km in 2.5 hours. What is its average speed in m/s? Show the steps.",
    "Explain in three sentences why the sky is blue.",
]


def metrics():
    text = urllib.request.urlopen(f"{URL}/metrics", timeout=10).read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_") and not line.startswith("#"):
            name, val = line.rsplit(" ", 1)
            key = name.split("{")[0]
            if "per_pos" not in key:
                out[key] = out.get(key, 0.0) + float(val)
    return out


def main():
    model = json.load(urllib.request.urlopen(f"{URL}/v1/models", timeout=10))["data"][0]["id"]
    before = metrics()
    for p in PROMPTS:
        body = json.dumps({"model": model, "max_tokens": 1024,
                           "messages": [{"role": "user", "content": p}]}).encode()
        req = urllib.request.Request(f"{URL}/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        r = json.load(urllib.request.urlopen(req, timeout=600))
        dt = time.perf_counter() - t0
        msg = r["choices"][0]["message"]
        n = r["usage"]["completion_tokens"]
        print(f"REQ tokens={n} s={dt:.2f} tok_s_incl_prefill={n / dt:.1f} "
              f"finish={r['choices'][0]['finish_reason']} "
              f"reasoning_chars={len(msg.get('reasoning_content') or msg.get('reasoning') or '')} "
              f"content_chars={len(msg.get('content') or '')}")
        print("CONTENT", repr((msg.get("content") or "")[:300]))
    after = metrics()
    d = {k.replace("vllm:spec_decode_num_", ""): after[k] - before.get(k, 0.0) for k in after}
    if d.get("draft_tokens"):
        d["acceptance_rate"] = round(d["accepted_tokens"] / d["draft_tokens"], 4)
        d["accepted_per_draft"] = round(d["accepted_tokens"] / d["drafts"], 3)
    print("SPEC", json.dumps(d))


if __name__ == "__main__":
    main()
