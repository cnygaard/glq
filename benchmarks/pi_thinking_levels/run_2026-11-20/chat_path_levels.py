"""The real /v1/chat/completions path, one output token per request: the prompt length vLLM
reports (`usage.prompt_tokens`) shows which template branch rendered. /tokenize builds its
chat params separately, so it cannot stand in for this path. Stdlib only.

    python -I chat_path_levels.py [base_url] [model]
"""
import json
import sys
import urllib.error
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
M = sys.argv[2] if len(sys.argv) > 2 else "xv0y5ncu/Arete-27B-GLQ-trellis-3inst-4bpw"

#: (label, extra request fields). The first group is what pi 0.79.9 and 1.6.0 send per level
#: (chat_template_kwargs only); the second is what pi 0.73.1-0.79.0 send for the same entry
#: (a top-level reasoning_effort, the map's value, "off" included).
CASES = [
    ("no kwargs (before)", {}),
    ("new pi off", {"chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True}}),
    ("new pi low", {"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True,
                                             "reasoning_effort": "low"}}),
    ("new pi medium", {"chat_template_kwargs": {"enable_thinking": True,
                                                "preserve_thinking": True,
                                                "reasoning_effort": "medium"}}),
    ("new pi xhigh", {"chat_template_kwargs": {"enable_thinking": True,
                                               "preserve_thinking": True,
                                               "reasoning_effort": "xhigh"}}),
    ("old pi off", {"reasoning_effort": "off"}),
    ("old pi none", {"reasoning_effort": "none"}),
    ("old pi low", {"reasoning_effort": "low"}),
    ("old pi medium", {"reasoning_effort": "medium"}),
    ("old pi xhigh", {"reasoning_effort": "xhigh"}),
]


def main():
    for label, extra in CASES:
        body = {"model": M, "max_tokens": 1, "temperature": 0,
                "messages": [{"role": "system", "content": "You are a coding agent."},
                             {"role": "user", "content": "hi"}], **extra}
        req = urllib.request.Request(URL + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)
            print(f"{label:20} prompt_tokens={out['usage']['prompt_tokens']}")
        except urllib.error.HTTPError as e:
            print(f"{label:20} HTTP {e.code}: {e.read().decode()[:140]}")


if __name__ == "__main__":
    main()
