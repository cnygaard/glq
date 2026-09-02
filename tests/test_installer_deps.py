"""What the installer pins into the serving venv, and why.

gemma-4 is the reason both ends of the transformers range exist:

* **floor** — `>=5.13.1`, or transformers has no gemma-4 at all. Already documented on
  `_install_open_webui`, which gets its own venv precisely because it pins 5.5.4.
* **ceiling** — `<5.15`, found by bisecting on an L4 against vLLM 0.27.1. 5.15.0 moved
  gemma-4 to a per-layer ("heterogeneous") config and made the global `config.head_dim`
  raise `AmbiguousGlobalPerLayerAttributeError`, which vLLM reads while building its
  ModelConfig. Measured, config-construction only:

      transformers 5.15.0  ->  FAIL  AmbiguousGlobalPerLayerAttributeError
      5.14.1 / 5.13.1 / 5.12.1 / 5.11.0 / 5.10.4  ->  OK, head_size=512

  This is not a GLQ bug: stock bf16 `google/gemma-4-E2B-it` fails identically on 5.15.0
  with no GLQ in the process. Nor is it fixable with the escape hatch transformers offers —
  setting `allow_global_per_layer_attribute_access` gets a ModelConfig with head_size=256
  for a model whose layers are 256 *and* 512, and the weight loader then dies on
  `assert param.size() == loaded_weight.size()`.

Drop the ceiling when vLLM can build a heterogeneous gemma-4 from per-layer configs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.installer import __main__ as M  # noqa: E402


def _pip_commands(components):
    seen = []
    M._install_python_extras(lambda cmd, **kw: seen.append([str(c) for c in cmd]),
                             Path("/home/u/.glq/venv"), components)
    return " ".join(" ".join(c) for c in seen)


def test_serving_pins_transformers_below_the_gemma4_break():
    """Without this, a fresh install resolves the newest transformers and every gemma-4
    checkpoint dies before a single weight is loaded."""
    flat = _pip_commands(("core", "vllm"))
    assert "transformers" in flat, f"transformers is left to pip's resolver:\n{flat}"
    assert "<5.15" in flat, f"no ceiling on transformers:\n{flat}"


def test_serving_keeps_the_floor_gemma4_needs():
    """5.13.1 is where transformers gained gemma-4. Below it the models do not exist."""
    assert ">=5.13.1" in _pip_commands(("core", "vllm"))


def test_the_pin_travels_with_vllm_not_with_the_chat_ui():
    """The clash is between vLLM and transformers. A chat-only install has no server in it,
    so pinning there would constrain a venv that never reads a model config."""
    assert "transformers" not in _pip_commands(("core", "chat"))


def test_nothing_is_installed_when_no_component_asks_for_it():
    assert _pip_commands(("core",)) == ""


def test_quantize_component_installs_the_extra_without_upgrading_glq():
    """`glq[quantize]` names glq itself, so `--upgrade` would replace a --glq-source dev
    install with the PyPI release — the extra must ride a plain install, where an
    already-satisfied glq is left alone and only the missing deps resolve."""
    seen = []
    M._install_python_extras(lambda cmd, **kw: seen.append([str(c) for c in cmd]),
                             Path("/home/u/.glq/venv"), ("core", "quantize"))
    quant = [c for c in seen if any("glq[quantize]" in a for a in c)]
    assert quant, f"no glq[quantize] install in: {seen}"
    assert "--upgrade" not in quant[0], quant[0]


def test_quantize_extra_is_not_installed_by_default():
    assert "glq[quantize]" not in _pip_commands(("core", "vllm", "chat"))


def test_picode_component_fetches_the_gemma4_tool_template():
    """gemma-4's tool template is not in the model checkpoint and not in the vLLM wheel —
    it lives in vLLM's repo examples. Measured on the box: the printed serve command
    referenced a template path that did not exist and vLLM refused to start. The picode
    component now fetches it, TLS-pinned like the installer's only other curl."""
    seen = []
    M._install_picode(lambda cmd, **kw: seen.append(" ".join(map(str, cmd))))
    fetch = [c for c in seen if "tool_chat_template_gemma4.jinja" in c]
    assert fetch, f"no template fetch in: {seen}"
    assert "--proto" in fetch[0] and "=https" in fetch[0] and "--tlsv1.2" in fetch[0]
    assert str(M.GLQ_HOME / "templates") in fetch[0]


def test_a_failed_template_fetch_does_not_abort_the_install():
    """Offline or a moved URL must not kill an install whose chosen model may not even be
    gemma-4 — warn and continue."""
    def run(cmd, **kw):
        if any("tool_chat_template" in str(c) for c in cmd):
            raise RuntimeError("curl: (6) could not resolve host")
    M._install_picode(run)


# ---- CPU-only machines get the vLLM +cpu wheel, not the CUDA one -------------------------

def _pip_commands_cpu(components, monkeypatch):
    """Hermetic: stub the release lookup — these tests assert argv shape, and the live
    GitHub API is rate-limited and returns %2B-encoded URLs."""
    from glq.installer import cpu_wheel
    monkeypatch.setattr(cpu_wheel, "latest_cpu_wheel_url",
                        lambda arch, fetch=None: cpu_wheel.FALLBACK_X86)
    seen = []
    M._install_python_extras(lambda cmd, **kw: seen.append([str(c) for c in cmd]),
                             Path("/home/u/.glq/venv"), components, device="cpu")
    return " ".join(" ".join(c) for c in seen)


def test_cpu_install_uses_the_cpu_wheel_and_pytorch_cpu_index(monkeypatch):
    flat = _pip_commands_cpu(("core", "vllm"), monkeypatch)
    assert "+cpu" in flat and "manylinux_2_34" in flat, f"no +cpu wheel URL:\n{flat}"
    assert "download.pytorch.org/whl/cpu" in flat


def test_cpu_install_never_names_bare_vllm(monkeypatch):
    """A bare `vllm` spec resolves the CUDA wheel from PyPI — the exact mistake this
    fork exists to prevent."""
    for cmd in _pip_commands_cpu(("core", "vllm"), monkeypatch).split(" "):
        assert cmd != "vllm"


def test_cpu_install_keeps_the_transformers_pin(monkeypatch):
    """The gemma-4 pin is model-bound, not device-bound — it applies equally on CPU."""
    flat = _pip_commands_cpu(("core", "vllm"), monkeypatch)
    assert ">=5.13.1" in flat and "<5.15" in flat


def test_default_device_argv_is_unchanged():
    """The GPU path must not drift: bare `vllm` from PyPI, no wheel URL, no extra index."""
    flat = _pip_commands(("core", "vllm"))
    assert " vllm " in f" {flat} "
    assert "+cpu" not in flat and "--extra-index-url" not in flat


def test_the_nvm_bootstrap_cannot_block_on_a_git_credential_prompt():
    """nvm's installer git-clones from github.com, and git asks for a username whenever
    GitHub refuses the request (403/404 — commonly unauthenticated rate-limiting from a
    shared cloud IP). Reported from a real install: the installer sat at
    `Username for 'https://github.com':` waiting for input a piped `curl | bash` may never
    be able to supply. No GitHub account is needed here, so failing fast with git's own
    error beats hanging forever."""
    seen = []
    M._install_picode(lambda cmd, **kw: seen.append(" ".join(map(str, cmd))))
    nvm = [c for c in seen if "nvm-sh/nvm" in c]
    assert nvm, f"no nvm bootstrap in: {seen}"
    assert "GIT_TERMINAL_PROMPT=0" in nvm[0], nvm[0]


def test_the_nvm_bootstrap_does_not_use_git():
    """Diagnosed on a cloud box: git sends `User-Agent: git/2.43.0` and GitHub replies
    `HTTP 401 www-authenticate: Basic realm="GitHub"`, while curl to the same endpoint
    gets 200 — GitHub challenges unauthenticated *git* operations from some IP ranges, so
    the clone hung on a username prompt. METHOD=script fetches nvm.sh over HTTPS instead;
    verified live, after which pi installed and `glq-setup --verify` went fully green."""
    seen = []
    M._install_picode(lambda cmd, **kw: seen.append(" ".join(map(str, cmd))))
    nvm = [c for c in seen if "nvm-sh/nvm" in c]
    assert nvm and "METHOD=script" in nvm[0], nvm


def test_pi_is_installed_without_running_dependency_scripts():
    """The package has no install hooks of its own (all its scripts are build-time) and a
    plain-JS bin, so --ignore-scripts costs nothing and keeps transitive postinstalls from
    running arbitrary code."""
    seen = []
    M._install_picode(lambda cmd, **kw: seen.append(" ".join(map(str, cmd))))
    npm = [c for c in seen if "pi-coding-agent" in c]
    assert npm and "--ignore-scripts" in npm[0], npm
