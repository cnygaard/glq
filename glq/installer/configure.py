"""Write the config files that point other tools at the served GLQ model.

The provider name `glq` is load-bearing, not cosmetic. pi addresses models as
`<provider>/<model>`, so `-m glq/<served-id>` splits into `--provider glq --model
<served-id>` on its own — see `benchmarks/harbor_pi_glq.py`, which depends on exactly that.
Renaming the provider breaks every documented invocation.

`~/.pi/agent/models.json` is shared: a user may already have Anthropic, OpenAI or their own
local providers in it, and those entries can hold real API keys. So this merges into the
existing document and replaces only the `glq` provider — never rewrites the file wholesale.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

#: Placeholder. vLLM does not check the key, but the field must be present. Never fill this
#: from the environment: the file lands on disk and is read by a separate process.
API_KEY_PLACEHOLDER = "glq"

PROVIDER = "glq"


#: Divisor on the served window giving the per-turn OUTPUT budget pi may ask for. A HALF, which
#: on a 262144 window is 131072 — exactly the final-response length Qwen3.8-Flash-Next's card
#: asks for.
#:
#: Why not the whole window: `max_tokens` and the prompt are drawn from the same window, so
#: asking for all of it leaves nothing for input. Measured on vLLM 0.31.0 against a 262144
#: window — `max_tokens=262144` returns **HTTP 400** ("you requested 262144 output tokens and
#: your prompt contains 325 characters (more than 0 characters, which is the upper bound for 0
#: input tokens)") while 65536 returns 200. Qwen's "262144 reasoning tokens" is the window it
#: wants, not an achievable per-turn output ask; the two are easy to conflate.
#:
#: Why not a smaller share either — this was a quarter first, on the theory that the rest had
#: to be held back for a growing transcript. **That theory was wrong**: pi clamps the ask
#: itself, every turn,
#:
#:     available = contextWindow - estimateContextTokens(context) - CONTEXT_SAFETY_MARGIN
#:     ask       = min(maxTokens, max(MIN_MAX_TOKENS, available))      # MIN_MAX_TOKENS = 1
#:
#: so this value is a pure ceiling and costs the transcript nothing — a lower share only caps
#: reasoning for no gain. That ceiling is what truncated a TB-4.0 trial at exactly 16384 output
#: tokens (`finish_reason: length`, mid-`<think>`, so no tool call was ever emitted).
#:
#: A half rather than the whole window because the clamp's `available` rests on an *estimate*:
#: if it undershoots by more than `CONTEXT_SAFETY_MARGIN` the request exceeds the window and
#: 400s instead of degrading. Halving keeps every turn clear of that edge, which is the one
#: failure mode here that is loud rather than graceful.
_PI_OUTPUT_SHARE = 2

#: Smallest useful per-turn output budget, for windows too small for the share above to leave
#: anything to answer in.
_PI_MIN_MAX_TOKENS = 1024


def pi_max_tokens(context_window) -> int:
    """Per-turn output budget for pi, given the served window.

    One rule in one place: `glq-code` and the harbor bench agent both configure pi, and the
    harbor one shipped with **no** `maxTokens` at all — so pi fell back to its own 16384
    default and silently truncated every long reasoning turn mid-`<think>`, emitting no tool
    call. The agent then had nothing to execute and stopped having spent the tokens. See
    `_PI_OUTPUT_SHARE` for why this is not simply the window.
    """
    return max(_PI_MIN_MAX_TOKENS, int(context_window or 0) // _PI_OUTPUT_SHARE)


def pi_models_json(base_url: str, model_ids,
                   context_window=None, max_tokens=None,
                   provider: str = PROVIDER,
                   api_key: str = API_KEY_PLACEHOLDER) -> dict:
    """The provider block for pi, in the shape of `examples/pi/models.json`.

    `provider`/`api_key` are overridable so the harbor bench agent — which names its provider
    from the `-m glq/<id>` split — shares this builder instead of hand-rolling the dict. The
    output-budget reasoning below is the reason that matters.
    """
    return {"providers": {provider: {
        "baseUrl": base_url,
        "api": "openai-completions",       # the dialect vLLM's server speaks
        "apiKey": api_key,
        # Without these, pi asks for the FULL window as its output budget and vLLM
        # 400s every request — measured live: max_tokens=16384 against a 16384 window
        # left "0 input tokens" for the prompt, and pi's --print mode swallowed the
        # error into an empty assistant turn.
        "models": [{"id": m,
                    **({"contextWindow": int(context_window)} if context_window else {}),
                    **({"maxTokens": int(max_tokens)} if max_tokens else {})}
                   for m in model_ids],
    }}}


def _write_private_json(path: Path, doc: dict) -> None:
    """Write JSON at mode 0600, creating parents. These files sit alongside ones holding
    real provider keys, so they are never world-readable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def write_pi_models(path, base_url: str, model_ids,
                    context_window=None, max_tokens=None) -> None:
    """Merge the `glq` provider into an existing pi config, preserving the others.

    An unreadable existing file is copied to `<name>.bak` before being replaced: we cannot
    merge into truncated JSON, but deleting a user's config without a copy is not ours to do.
    """
    path = Path(path)
    doc = {"providers": {}}

    if path.exists():
        try:
            existing = json.loads(path.read_text())
            if isinstance(existing, dict):
                doc = existing
                doc.setdefault("providers", {})
        except (json.JSONDecodeError, OSError):
            backup = path.with_suffix(path.suffix + ".bak")
            backup.write_bytes(path.read_bytes())
            os.chmod(backup, 0o600)

    doc["providers"][PROVIDER] = pi_models_json(
        base_url, model_ids, context_window=context_window,
        max_tokens=max_tokens)["providers"][PROVIDER]
    _write_private_json(path, doc)


def write_glq_config(path, *, model: str, base_url: str, components, available,
                     fp8_kv: bool = False,
                     code_model: str | None = None,
                     chat_model: str | None = None,
                     device: str | None = None) -> None:
    """Record what the installer chose.

    `examples/chat/app.py` reads this to populate its model dropdown, and it is the only
    record of the installer's decisions — worth having when someone reports that it served
    a model they did not expect.

    `code_model`/`chat_model` are the per-command defaults (glq-code prefers Qwen, whose own
    chat template emits tool markup, glq-chat prefers gemma-4 for its MoE decode speed — see
    recommend.PREFERRED_FAMILIES). Written only when chosen: absence is what tells
    glq-code/glq-chat to fall back to the generic `model`, so old configs keep old behavior.
    """
    _write_private_json(Path(path), {
        "model": model,
        "base_url": base_url,
        "components": list(components),
        "available": list(available),
        # `glq-chat` reads this back as its default, so the KV question is asked once at
        # install time rather than on every start.
        "fp8_kv": bool(fp8_kv),
        **({"code_model": code_model} if code_model else {}),
        **({"chat_model": chat_model} if chat_model else {}),
        # The device the installer decided on ("cuda"/"cpu") — drives which vLLM wheel
        # was installed, model sizing, and messaging. Absent on pre-device configs; the
        # serving commands detect live regardless (the installed wheel wins).
        **({"device": device} if device else {}),
    })
