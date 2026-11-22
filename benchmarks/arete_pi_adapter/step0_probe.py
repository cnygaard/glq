"""Step 0 client: is the pi adapter really applied to GLQ Arete under vLLM?

A client only (stdlib, no vLLM import): it talks to a running `vllm serve` over HTTP. Run by
`step0.sh` once per server arm.

  --adapter NAME   also ask the adapter, and compare. Greedy (temperature 0), thinking off for
                   both, the same prompt: if the LoRA is applied the two outputs differ. A
                   tool-call request to the adapter checks the qwen3_xml parser still reads what
                   the adapted model emits.

Writes one JSON file with every response, so the judgement can be re-read later.
"""
import argparse
import json
import time
import urllib.request

PROMPT = ("Write a Python function `fib(n)` that returns the n-th Fibonacci number "
          "iteratively, then one sentence on its time complexity.")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read a file from the working directory.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string",
                                               "description": "Path of the file to read"}},
                       "required": ["path"]},
    },
}]


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    out["_elapsed_s"] = round(time.time() - t0, 3)
    return out


def chat(base, model, messages, **kw):
    body = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False}, **kw}
    return post(f"{base}/v1/chat/completions", body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8011")
    ap.add_argument("--model", required=True, help="the served base model name")
    ap.add_argument("--adapter", default=None, help="the served LoRA name")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    models = json.load(urllib.request.urlopen(f"{a.base_url}/v1/models", timeout=30))
    rec = {"served": [m["id"] for m in models["data"]], "prompt": PROMPT}
    msgs = [{"role": "user", "content": PROMPT}]
    rec["base"] = chat(a.base_url, a.model, msgs)
    base_text = rec["base"]["choices"][0]["message"]["content"]
    print("BASE_SAMPLE", json.dumps(base_text[:300]))

    if a.adapter:
        rec["adapter"] = chat(a.base_url, a.adapter, msgs)
        ad_text = rec["adapter"]["choices"][0]["message"]["content"]
        print("ADAPTER_SAMPLE", json.dumps(ad_text[:300]))
        rec["outputs_differ"] = ad_text != base_text
        print("OUTPUTS_DIFFER", rec["outputs_differ"])
        # Twice, so a difference cannot be greedy nondeterminism between two calls.
        rec["base_repeat"] = chat(a.base_url, a.model, msgs)
        rec["base_repeat_identical"] = (
            rec["base_repeat"]["choices"][0]["message"]["content"] == base_text)
        print("BASE_REPEAT_IDENTICAL", rec["base_repeat_identical"])

        tool_msgs = [{"role": "user", "content": "Read hello.py and tell me what it prints."}]
        rec["adapter_tool"] = chat(a.base_url, a.adapter, tool_msgs, tools=TOOLS,
                                   tool_choice="auto")
        m = rec["adapter_tool"]["choices"][0]["message"]
        calls = m.get("tool_calls") or []
        rec["tool_call_parsed"] = bool(calls) and calls[0]["function"]["name"] == "read"
        print("TOOL_CALL", json.dumps([c["function"] for c in calls])[:300],
              "PARSED", rec["tool_call_parsed"],
              "REASONING_EMPTY", not (m.get("reasoning") or m.get("reasoning_content")))

    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
