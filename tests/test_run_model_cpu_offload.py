"""``--cpu-offload-gb`` has to reach the engine, and has to be recorded.

vLLM's `cpu_offload_gb` keeps part of the weights in pinned host memory and streams them
in during each forward pass, which is the only way a checkpoint larger than the card can
be loaded at all. For GLQ that is not a theoretical need: the Qwen3.8-Flash-Next 3 bpw
checkpoint loads 73.29 GiB of weights, so it cannot be examined on anything smaller than
a 96 GiB card without it.

Two properties, because getting either wrong makes the result a lie rather than an error:

* the value must reach ``LLM(**kw)`` — an offload flag that is parsed and dropped looks
  exactly like a card that was big enough all along;
* it must appear in the printed CONFIG, because a footprint or tok/s measured with 60 GiB
  of weights living in host RAM is not comparable to one measured on-device, and the only
  thing distinguishing the two records is that line.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(__file__)
DRIVER = os.path.join(HERE, "..", "benchmarks", "run_model.py")


def _help() -> str:
    r = subprocess.run([sys.executable, DRIVER, "--help"], capture_output=True, text=True)
    return r.stdout + r.stderr


def test_the_flag_exists():
    assert "--cpu-offload-gb" in _help()


def test_it_is_documented_as_host_memory():
    """A reader has to be able to tell this is not free VRAM."""
    h = _help()
    # rfind, not find: the flag appears first in the usage synopsis, where there is no
    # help text to inspect. The options section is the last occurrence.
    i = h.rfind("--cpu-offload-gb")
    assert i != -1
    assert any(w in h[i:i + 400].lower() for w in ("host", "cpu memory", "ram"))


@pytest.mark.parametrize("src,expect", [
    (["--cpu-offload-gb", "60"], 60.0),
    ([], 0.0),
])
def test_parsed_value(src, expect):
    """Default 0 means 'off', which is what every existing recorded run assumed."""
    sys.path.insert(0, os.path.join(HERE, ".."))
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_model", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.build_parser().parse_args(["--model", "m"] + src)
    assert float(args.cpu_offload_gb) == expect
# ---- HF class resolution -------------------------------------------------------------

def test_hf_class_resolves_from_config_architectures():
    """``AutoModelForCausalLM`` maps the *config class*, ignoring ``architectures``.

    For Qwen4Exp it returns the text-only ``Qwen4ExpForCausalLM`` (modules named
    ``model.layers.*``) while the checkpoint is ``Qwen4ExpForConditionalGeneration``
    (``model.language_model.layers.*``). Measured consequence: every GLQ tensor came back
    UNEXPECTED and ``model.visual.*`` was reported missing — a checkpoint that loads to
    garbage rather than an error.
    """
    sys.path.insert(0, os.path.join(HERE, ".."))
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_model_res", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "resolve_hf_class"), "run_model.py has no resolve_hf_class"

    class _Cfg:
        architectures = ["Qwen4ExpForConditionalGeneration"]

    class _FakeTF:
        class Qwen4ExpForConditionalGeneration:  # noqa: D401
            pass

    got = mod.resolve_hf_class(_Cfg(), transformers_mod=_FakeTF)
    assert got is _FakeTF.Qwen4ExpForConditionalGeneration


def test_hf_class_falls_back_when_architecture_is_unknown():
    """An architecture transformers does not export must not break the existing path."""
    sys.path.insert(0, os.path.join(HERE, ".."))
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_model_res2", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _Cfg:
        architectures = ["SomethingNotExported"]

    class _FakeTF:
        AutoModelForCausalLM = "fallback"

    assert mod.resolve_hf_class(_Cfg(), transformers_mod=_FakeTF) == "fallback"


def test_hf_class_falls_back_with_no_architectures():
    sys.path.insert(0, os.path.join(HERE, ".."))
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_model_res3", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _Cfg:
        pass

    class _FakeTF:
        AutoModelForCausalLM = "fallback"

    assert mod.resolve_hf_class(_Cfg(), transformers_mod=_FakeTF) == "fallback"
