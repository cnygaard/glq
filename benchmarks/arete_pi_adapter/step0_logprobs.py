"""Step 0b client: does the adapter change the model's logits at all?

Step 0 found the greedy output of the base model and of `qwen3.8-pi` byte-identical over 256
tokens, on a short coding prompt, although vLLM loaded adapter weights into all 160 wrapped
modules and adapter requests ran slower (the LoRA kernels ran). Identical argmax can happen with
a real but small delta; identical LOGPROBS cannot. So this asks both models for the top-5
logprobs at every position, greedy, thinking off, and reports the largest difference.

Two prompts: the short one from Step 0, and an agent turn with tools (the setting the adapter
was trained for). A client only, stdlib.
"""
import argparse
import json
import urllib.request

TOOLS = [{"type": "function", "function": {
    "name": n, "description": d,
    "parameters": {"type": "object", "properties": p, "required": list(p)}}}
    for n, d, p in [
        ("read", "Read a file.", {"path": {"type": "string"}}),
        ("bash", "Run a shell command.", {"command": {"type": "string"}}),
        ("edit", "Replace text in a file.", {"path": {"type": "string"},
                                              "oldText": {"type": "string"},
                                              "newText": {"type": "string"}}),
    ]]

PROMPTS = {
    "fib": [{"role": "user", "content":
             "Write a Python function `fib(n)` that returns the n-th Fibonacci number "
             "iteratively, then one sentence on its time complexity."}],
    "agent": [
        {"role": "system", "content":
         "You are an expert coding assistant. You help users by reading files, running "
         "commands, editing code, and writing new files. Use the tools to act; be concise."},
        {"role": "user", "content":
         "The tests in tests/test_parser.py fail with KeyError: 'name'. Find out why and fix "
         "the bug."},
    ],
}


def ask(base, model, messages):
    body = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 96,
            "logprobs": True, "top_logprobs": 5, "tools": TOOLS, "tool_choice": "auto",
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8011")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="qwen3.8-pi")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = {}
    for name, msgs in PROMPTS.items():
        b = ask(a.base_url, a.model, msgs)
        b2 = ask(a.base_url, a.model, msgs)
        d = ask(a.base_url, a.adapter, msgs)
        rec[name] = {"base": b, "base_repeat": b2, "adapter": d}
        lb = b["choices"][0]["logprobs"]["content"]
        lb2 = b2["choices"][0]["logprobs"]["content"]
        ld = d["choices"][0]["logprobs"]["content"]

        def maxdiff(x, y):
            n = min(len(x), len(y))
            first_tok = next((i for i in range(n) if x[i]["token"] != y[i]["token"]), None)
            m = first_tok if first_tok is not None else n
            diff = max((abs(x[i]["logprob"] - y[i]["logprob"]) for i in range(m)), default=0.0)
            return diff, first_tok, n

        rb, fb, nb = maxdiff(lb, lb2)
        ra, fa, na = maxdiff(lb, ld)
        print(f"PROMPT {name}: base vs base-repeat max |dlogprob| {rb:.6f} "
              f"(first token differing: {fb}, of {nb})")
        print(f"PROMPT {name}: base vs adapter     max |dlogprob| {ra:.6f} "
              f"(first token differing: {fa}, of {na})")
        print(f"PROMPT {name}: base    {json.dumps(b['choices'][0]['message'].get('content') or '')[:160]} "
              f"tools {json.dumps(b['choices'][0]['message'].get('tool_calls') or [])[:200]}")
        print(f"PROMPT {name}: adapter {json.dumps(d['choices'][0]['message'].get('content') or '')[:160]} "
              f"tools {json.dumps(d['choices'][0]['message'].get('tool_calls') or [])[:200]}")
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
