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


def pi_models_json(base_url: str, model_ids) -> dict:
    """The `glq` provider block for pi, in the shape of `examples/pi/models.json`."""
    return {"providers": {PROVIDER: {
        "baseUrl": base_url,
        "api": "openai-completions",       # the dialect vLLM's server speaks
        "apiKey": API_KEY_PLACEHOLDER,
        "models": [{"id": m} for m in model_ids],
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


def write_pi_models(path, base_url: str, model_ids) -> None:
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

    doc["providers"][PROVIDER] = pi_models_json(base_url, model_ids)["providers"][PROVIDER]
    _write_private_json(path, doc)


def write_glq_config(path, *, model: str, base_url: str, components, available,
                     fp8_kv: bool = False) -> None:
    """Record what the installer chose.

    `examples/chat/app.py` reads this to populate its model dropdown, and it is the only
    record of the installer's decisions — worth having when someone reports that it served
    a model they did not expect.
    """
    _write_private_json(Path(path), {
        "model": model,
        "base_url": base_url,
        "components": list(components),
        "available": list(available),
        # `glq-chat` reads this back as its default, so the KV question is asked once at
        # install time rather than on every start.
        "fp8_kv": bool(fp8_kv),
    })
