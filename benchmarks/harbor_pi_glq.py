"""Harbor agent: pi pointed at a locally-served GLQ checkpoint.

Harbor ships a working pi agent (`harbor.agents.installed.pi.Pi`) that installs the CLI and
drives it — none of that is reimplemented here. It has exactly one gap for our purpose: it
wires up **provider API keys** (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …) and has no way to
express a base URL, so it can only reach hosted providers. A GLQ checkpoint served by vLLM
on the host is unreachable through it.

pi's own answer is `~/.pi/agent/models.json`, which registers a custom provider given a
`baseUrl` and `api: "openai-completions"` — which vLLM's OpenAI-compatible server speaks.
So this subclass adds one thing: it writes that file into the container. Everything else —
node bootstrap, install, prompt handling, `run()`, trajectory parsing — is inherited.

The provider is named **glq**, which makes harbor's model syntax line up on its own:
`-m glq/<served-id>` is split by `Pi.run()` into `--provider glq --model <served-id>`, and
pi then resolves `glq` from the file we wrote. No patching of their argument handling.

Usage::

    GLQ_VLLM_BASE_URL=http://172.17.0.1:8000/v1 \\
    harbor run -d terminal-bench/terminal-bench-2 \\
      --agent-import-path benchmarks.harbor_pi_glq:PiGLQAgent \\
      -m glq/<served-id> --allow-agent-host 172.17.0.1 -k 1

``--allow-agent-host`` matters whenever a task declares ``network_mode = "allowlist"``:
harbor merges that value into the agent-phase allowlist. Tasks default to ``public`` and
need nothing; a task declaring ``no-network`` cannot reach any model server and will fail
for a hosted agent too.
"""
from __future__ import annotations

import json
import os
import shlex

from harbor.agents.installed.pi import Pi
from harbor.environments.base import BaseEnvironment

# pi's config path. Overridable in pi via PI_CODING_AGENT_DIR, but the default is what
# `exec_as_agent` lands in, so we use it and stay out of pi's way.
_MODELS_JSON = ".pi/agent/models.json"

# harbor 0.20.0's Pi.install() pins the LEGACY package (@mariozechner/pi-coding-agent).
# The models.json provider schema this class depends on is documented for the current
# package, so we overlay it — later npm -g install wins the `pi` binary. Drop this once
# harbor's own installer moves over.
_PI_PACKAGE = "@earendil-works/pi-coding-agent"


class PiGLQAgent(Pi):
    """pi, talking to a GLQ model served on the Docker host."""

    @staticmethod
    def name() -> str:
        return "pi-glq"

    async def install(self, environment: BaseEnvironment) -> None:
        # Their install first: apt/curl, the nvm node bootstrap, and a working `pi`.
        await super().install(environment)

        await self.exec_as_agent(
            environment,
            command=("set -euo pipefail; "
                     f"npm install -g --ignore-scripts {_PI_PACKAGE} && pi --version"),
        )

        base_url = os.environ.get("GLQ_VLLM_BASE_URL")
        if not base_url:
            raise RuntimeError(
                "GLQ_VLLM_BASE_URL is unset — the agent has no way to reach the model. "
                "Point it at the vLLM server as seen FROM INSIDE the container (the "
                "docker0 bridge address, typically http://172.17.0.1:8000/v1 — not "
                "localhost, which is the container itself).")

        if not self.model_name or "/" not in self.model_name:
            raise ValueError("model must be 'glq/<served-id>' so Pi.run() can split it")
        provider, served_id = self.model_name.split("/", 1)

        # apiKey is a literal dummy on purpose: pi hides models it considers unauthenticated
        # even when the server ignores auth, so a keyless local endpoint still needs a value.
        models = {"providers": {provider: {
            "baseUrl": base_url,
            "api": "openai-completions",
            "apiKey": os.environ.get("GLQ_VLLM_API_KEY", "dummy"),
            "models": [{"id": served_id}],
        }}}
        payload = shlex.quote(json.dumps(models, indent=2))
        await self.exec_as_agent(
            environment,
            command=(f"set -euo pipefail; mkdir -p $(dirname ~/{_MODELS_JSON}); "
                     f"printf '%s' {payload} > ~/{_MODELS_JSON}; "
                     # Read it back: a silently-unwritten config surfaces later as an
                     # opaque "model not found" from pi, long after the real failure.
                     f"test -s ~/{_MODELS_JSON} && head -c 200 ~/{_MODELS_JSON}"),
        )
