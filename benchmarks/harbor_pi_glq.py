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
    harbor run -d terminal-bench/terminal-bench@4.0.0 \\
      -a benchmarks.harbor_pi_glq:PiGLQAgent \\
      -m glq/<served-id> --allow-agent-host 172.17.0.1 -k 1

A custom agent travels on ``-a``; harbor's ``--agent-import-path`` was removed and is now
rejected at argument parsing.

``--allow-agent-host`` matters whenever a task declares ``network_mode = "allowlist"``:
harbor merges that value into the agent-phase allowlist. Tasks default to ``public`` and
need nothing; a task declaring ``no-network`` cannot reach any model server and will fail
for a hosted agent too.
"""
from __future__ import annotations

import json
import os
import shlex
import urllib.error
import urllib.request

from harbor.agents.installed.pi import Pi
from harbor.environments.base import BaseEnvironment

from glq.installer.configure import pi_max_tokens, pi_models_json

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

    @staticmethod
    def _served_window(base_url: str):
        """The served `max_model_len`, asked of the server rather than assumed.

        vLLM advertises it on `/v1/models`, so this works whether the run came through
        `glq-bench` or a hand-typed `harbor run` — no env plumbing to forget. The override
        exists for a server that does not report it; `None` means "leave pi's defaults alone"
        rather than guess, because a wrong window is worse than an absent one.
        """
        override = os.environ.get("GLQ_VLLM_MAX_MODEL_LEN")
        if override:
            try:
                return int(override)
            except ValueError:
                return None
        try:
            with urllib.request.urlopen(               # noqa: S310 - our own serve URL
                    base_url.rstrip("/") + "/models", timeout=10) as resp:
                doc = json.load(resp)
            for entry in doc.get("data") or []:
                window = entry.get("max_model_len")
                if isinstance(window, int) and window > 0:
                    return window
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            return None
        return None

    async def install(self, environment: BaseEnvironment) -> None:
        # Their install first: apt/curl, the nvm node bootstrap, and a working `pi`.
        await super().install(environment)

        # nvm is not on the PATH of a fresh non-login shell — harbor's own Pi sources it in
        # every command it sends (`get_version_command`, `run`), and so must we, or npm is
        # simply "command not found".
        # --force because the legacy package already claimed the `pi` bin: without it npm
        # aborts with EEXIST rather than overwriting, which is the whole point of the overlay.
        await self.exec_as_agent(
            environment,
            command=("set -euo pipefail; . ~/.nvm/nvm.sh; "
                     f"npm install -g --force --ignore-scripts {_PI_PACKAGE} && "
                     "pi --version"),
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

        # The output budget is NOT optional, and omitting it is not a small mistake. Without
        # `maxTokens` pi applies its own 16384 default, and a reasoning model emits its tool
        # call only AFTER `</think>` — so a long thinking turn is truncated mid-reasoning with
        # `finish_reason: length`, no tool call is ever produced, the agent has nothing to
        # execute, and the trial ends having spent the tokens and written nothing. Measured on
        # a TB-4.0 trial: one turn of exactly 16384 output tokens (2^14, stop reason `length`)
        # after two healthy 73- and 40-token tool-use turns, reward 0.0, no exception. It reads
        # as model incapability and is not.
        #
        # Shared builder rather than a second hand-rolled dict, because that duplication is
        # exactly how this field went missing here while `glq-code` had it all along.
        window = self._served_window(base_url)
        models = pi_models_json(
            base_url, [served_id], provider=provider,
            api_key=os.environ.get("GLQ_VLLM_API_KEY", "dummy"),
            context_window=window,
            max_tokens=pi_max_tokens(window) if window else None)
        payload = shlex.quote(json.dumps(models, indent=2))
        await self.exec_as_agent(
            environment,
            command=(f"set -euo pipefail; mkdir -p $(dirname ~/{_MODELS_JSON}); "
                     f"printf '%s' {payload} > ~/{_MODELS_JSON}; "
                     # Read the WHOLE file back, not a prefix. A silently-unwritten config
                     # surfaces later as an opaque "model not found" from pi, long after the
                     # real failure — and a 200-char prefix stops just short of the
                     # `maxTokens`/`contextWindow` entries, which is precisely how a missing
                     # output budget stayed invisible in the logs while truncating every long
                     # reasoning turn. The file holds a placeholder key, never a secret.
                     f"test -s ~/{_MODELS_JSON} && cat ~/{_MODELS_JSON}"),
        )
