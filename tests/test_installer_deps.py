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
