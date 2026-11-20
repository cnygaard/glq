"""How vLLM renders Arete's chat template for each set of chat_template_kwargs pi sends.

/tokenize only renders and tokenizes; nothing is generated. Stdlib only.
    python -I render_levels.py [base_url] [model]
"""
import json
import sys
import urllib.error
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
M = sys.argv[2] if len(sys.argv) > 2 else "xv0y5ncu/Arete-27B-GLQ-trellis-3inst-4bpw"
XH = "Reasoning effort is set to xhigh."
LO = "Reasoning effort is set to low."
TOOLS = [{"type": "function", "function": {
    "name": "read", "description": "Read a file.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
#: Exactly what pi 1.6.0 sent per level (stub capture, 2026-11-20), plus no kwargs at all
#: (glq-code before this change) and one value the template does not accept.
CASES = {
    "no kwargs (before)": None,
    "off": {"enable_thinking": False, "preserve_thinking": True},
    "low": {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "low"},
    "medium": {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "medium"},
    "xhigh": {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "xhigh"},
    "unaccepted 'high'": {"enable_thinking": True, "reasoning_effort": "high"},
}


def render(kw, tools):
    body = {"model": M, "add_generation_prompt": True, "return_token_strs": True,
            "messages": [{"role": "system", "content": "You are a coding agent."},
                         {"role": "user", "content": "hi"}]}
    if tools:
        body["tools"] = TOOLS
    if kw is not None:
        body["chat_template_kwargs"] = kw
    req = urllib.request.Request(URL + "/tokenize", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main():
    for tools in (False, True):
        for name, kw in CASES.items():
            try:
                r = render(kw, tools)
            except urllib.error.HTTPError as e:
                print(f"tools={tools!s:5} {name:20} HTTP {e.code}: {e.read().decode()[:150]}")
                continue
            text = "".join(r.get("token_strs") or []).replace("\u0120", " ").replace("\u010a", "\n")
            print(f"tools={tools!s:5} {name:20} tokens={r['count']:4} "
                  f"xhigh_instr={XH in text!s:5} low_instr={LO in text!s:5} "
                  f"ends={text[-28:]!r}")


if __name__ == "__main__":
    main()
