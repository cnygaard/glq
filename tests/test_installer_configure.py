"""Config writing (glq/installer/configure.py).

Two things here are easy to get wrong and expensive when wrong:

  * **`~/.pi/agent/models.json` may already exist.** A user can have Anthropic, OpenAI or
    their own local providers configured. Overwriting the file to add ours would silently
    delete their setup — so we merge, replacing only the `glq` provider.
  * **The provider must be named `glq`.** pi's model syntax is `<provider>/<model>`, so the
    name is load-bearing, not cosmetic: `benchmarks/harbor_pi_glq.py` documents that
    `-m glq/<served-id>` is split into `--provider glq --model <served-id>`. Rename the
    provider and every documented invocation breaks.

Shape follows the committed `examples/pi/models.json`.
"""
from __future__ import annotations

import json
import os
import stat
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.installer import configure as C  # noqa: E402

MODEL = "xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel"


def test_provider_is_named_glq_so_pi_model_syntax_resolves():
    doc = C.pi_models_json("http://127.0.0.1:8000/v1", [MODEL])
    assert set(doc["providers"]) == {"glq"}


def test_api_dialect_is_the_one_vllm_speaks():
    doc = C.pi_models_json("http://127.0.0.1:8000/v1", [MODEL])
    assert doc["providers"]["glq"]["api"] == "openai-completions"


def test_base_url_carries_the_v1_suffix():
    """vLLM serves the OpenAI routes under /v1; without it every call 404s."""
    doc = C.pi_models_json("http://127.0.0.1:8000/v1", [MODEL])
    assert doc["providers"]["glq"]["baseUrl"].endswith("/v1")


def test_models_are_listed_by_served_id():
    doc = C.pi_models_json("http://127.0.0.1:8000/v1", [MODEL, "xv0y5ncu/other"])
    assert [m["id"] for m in doc["providers"]["glq"]["models"]] == [MODEL, "xv0y5ncu/other"]


def test_api_key_is_a_placeholder_not_a_secret():
    """vLLM does not check it, but the field must be present. It must never be filled from
    the environment — this file is written to disk and read by another process."""
    key = C.pi_models_json("http://127.0.0.1:8000/v1", [MODEL])["providers"]["glq"]["apiKey"]
    assert key == "glq"


def test_merge_preserves_other_providers(tmp_path):
    """The destructive case: a user with an existing pi setup must not lose it."""
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"providers": {
        "anthropic": {"apiKey": "sk-theirs", "models": [{"id": "claude"}]},
        "glq": {"baseUrl": "http://old:1/v1", "models": [{"id": "stale"}]},
    }}))

    C.write_pi_models(path, "http://127.0.0.1:8000/v1", [MODEL])

    doc = json.loads(path.read_text())
    assert doc["providers"]["anthropic"]["apiKey"] == "sk-theirs"      # untouched
    assert doc["providers"]["glq"]["baseUrl"] == "http://127.0.0.1:8000/v1"
    assert [m["id"] for m in doc["providers"]["glq"]["models"]] == [MODEL]


def test_merge_creates_the_file_and_parents_when_absent(tmp_path):
    path = tmp_path / "nested" / "agent" / "models.json"
    C.write_pi_models(path, "http://127.0.0.1:8000/v1", [MODEL])
    assert json.loads(path.read_text())["providers"]["glq"]["models"][0]["id"] == MODEL


def test_a_corrupt_existing_config_is_backed_up_not_silently_dropped(tmp_path):
    """Truncated JSON from an interrupted write: we cannot merge into it, but deleting a
    user's file without a copy is not ours to do."""
    path = tmp_path / "models.json"
    path.write_text('{"providers": {"anthropic":')          # truncated
    C.write_pi_models(path, "http://127.0.0.1:8000/v1", [MODEL])
    assert json.loads(path.read_text())["providers"]["glq"]["models"][0]["id"] == MODEL
    assert (tmp_path / "models.json.bak").exists()


def test_config_files_are_not_world_readable(tmp_path):
    """These sit next to files that hold real API keys for other providers."""
    path = tmp_path / "models.json"
    C.write_pi_models(path, "http://127.0.0.1:8000/v1", [MODEL])
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_glq_config_records_what_was_installed(tmp_path):
    """examples/chat reads this to populate its dropdown, and it is the only record of what
    the installer chose — worth having when someone reports 'it served the wrong model'."""
    path = tmp_path / "config.json"
    C.write_glq_config(path, model=MODEL, base_url="http://127.0.0.1:8000/v1",
                       components=("core", "vllm", "chat"), available=[MODEL, "x/y"])
    doc = json.loads(path.read_text())
    assert doc["model"] == MODEL
    assert doc["base_url"].endswith("/v1")
    assert "chat" in doc["components"]
    assert doc["available"] == [MODEL, "x/y"]


def test_model_entries_can_carry_the_output_budget():
    """Without contextWindow/maxTokens, pi requests the FULL window as output budget and
    every request fails: measured live, `max_tokens=16384` against a 16384 window left
    "0 input tokens" for a 5804-character prompt — 400 on the very first call, which
    pi's --print mode swallows silently."""
    doc = C.pi_models_json("http://127.0.0.1:8000/v1", ["org/m"],
                           context_window=16384, max_tokens=4096)
    entry = doc["providers"]["glq"]["models"][0]
    assert entry["contextWindow"] == 16384
    assert entry["maxTokens"] == 4096


def test_limits_are_omitted_when_not_given():
    """The plain shape stays byte-compatible for callers that do not know the window."""
    entry = C.pi_models_json("http://x/v1", ["org/m"])["providers"]["glq"]["models"][0]
    assert entry == {"id": "org/m"}


def test_write_pi_models_passes_the_limits_through(tmp_path):
    p = tmp_path / "models.json"
    C.write_pi_models(p, "http://127.0.0.1:8000/v1", ["org/m"],
                      context_window=8192, max_tokens=2048)
    import json
    entry = json.loads(p.read_text())["providers"]["glq"]["models"][0]
    assert entry["contextWindow"] == 8192 and entry["maxTokens"] == 2048


# ---- per-command model defaults (glq-code → Qwen, glq-chat → gemma-4) ------------------

def test_config_records_per_command_models(tmp_path):
    path = tmp_path / "config.json"
    C.write_glq_config(path, model="m", base_url="u", components=["core"], available=["m"],
                       code_model="xv0y5ncu/qwen", chat_model="xv0y5ncu/gemma")
    doc = json.loads(path.read_text())
    assert doc["code_model"] == "xv0y5ncu/qwen"
    assert doc["chat_model"] == "xv0y5ncu/gemma"


def test_per_command_models_are_omitted_when_not_chosen(tmp_path):
    """Old-style configs stay old-style: absent keys, not nulls — `cfg.get` fallbacks in
    glq-chat/glq-code rely on absence meaning 'use the generic model'."""
    path = tmp_path / "config.json"
    C.write_glq_config(path, model="m", base_url="u", components=["core"], available=["m"])
    doc = json.loads(path.read_text())
    assert "code_model" not in doc and "chat_model" not in doc
