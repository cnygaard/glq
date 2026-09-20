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


# ---- CPU device handling -------------------------------------------------------------

def _load_driver(tag):
    import importlib.util
    sys.path.insert(0, os.path.join(HERE, ".."))
    spec = importlib.util.spec_from_file_location(f"run_model_{tag}", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_footprint_uses_rss_on_cpu():
    """``torch.cuda.memory_allocated`` is meaningless without a GPU, and
    ``reset_peak_memory_stats`` raises outright — measured:

        RuntimeError: invalid argument to reset_peak_memory_stats

    A CPU run has to report host RSS instead, or it cannot report a footprint at all.
    """
    mod = _load_driver("cpu1")
    assert hasattr(mod, "hf_footprint_gib")
    got = mod.hf_footprint_gib("cpu")
    assert got > 0.0, "RSS should be non-zero for a live process"


def test_footprint_is_cuda_backed_on_gpu():
    mod = _load_driver("cpu2")
    calls = {}

    class _FakeCuda:
        @staticmethod
        def memory_allocated():
            calls["hit"] = True
            return 3 * 2 ** 30

    assert mod.hf_footprint_gib("cuda", cuda_mod=_FakeCuda) == 3.0
    assert calls.get("hit")


def test_cpu_device_map_is_not_a_cuda_string():
    """Guards the branch predicate itself: 'cpu' must not be treated as a CUDA device,
    and 'cuda:1' must be."""
    mod = _load_driver("cpu3")
    assert mod.is_cuda_device("cuda") is True
    assert mod.is_cuda_device("cuda:1") is True
    assert mod.is_cuda_device("cpu") is False
    assert mod.is_cuda_device("auto") is False


# ---- max_memory (accelerate device_map budgeting) ------------------------------------

def test_max_memory_parses_json():
    """`device_map="auto"` places whole MODULES. Qwen Next's PLE embedding is one module
    holding a 23.84 GiB buffer: it fits a 44.39 GiB card alone, but not after ~26 GiB of
    decoder has been placed, so auto OOMs. `max_memory` caps the GPU budget so the big
    table lands on CPU instead -- the split that actually wants to happen, since an
    embedding lookup is a gather and its output is kilobytes."""
    mod = _load_driver("mm1")
    assert hasattr(mod, "parse_max_memory")
    got = mod.parse_max_memory('{"0": "20GiB", "cpu": "200GiB"}')
    assert got == {0: "20GiB", "cpu": "200GiB"}, got


def test_max_memory_none_when_unset():
    mod = _load_driver("mm2")
    assert mod.parse_max_memory(None) is None
    assert mod.parse_max_memory("") is None


# ---- explicit device maps ------------------------------------------------------------

def test_device_map_accepts_json_for_pinning_a_module():
    """`device_map="auto"` places whole modules and cannot split one. When inference still
    puts a big module on the wrong device, transformers accepts a module-name-keyed dict --
    e.g. pin Qwen Next's 23.84 GiB PLE to cpu and leave the decoder on the GPU."""
    mod = _load_driver("dm1")
    got = mod.parse_device_map('{"model.language_model.layers.1.ple": "cpu", "": 0}')
    assert got == {"model.language_model.layers.1.ple": "cpu", "": 0}


def test_device_map_passes_plain_strings_through():
    """cuda / cpu / auto must keep working unchanged."""
    mod = _load_driver("dm2")
    for s in ("cuda", "cpu", "auto", "balanced"):
        assert mod.parse_device_map(s) == s


# ---- offload_buffers: GLQ stores weights as buffers ----------------------------------

def test_offload_buffers_is_opt_in():
    """accelerate's `offload_buffers=False` default keeps the BUFFERS of CPU-assigned
    modules on the execution device. Normal models barely notice -- buffers are small.
    GLQ stores its quantized weights with `register_buffer`, so on Qwen Next that is
    38.3 GiB that follows the model to the GPU even after accelerate correctly assigns
    45 of 53 modules to cpu. transformers warns and is ignored:

        Current model requires 41171526456 bytes of buffer for offloaded layers, which
        seems does not fit any GPU's remaining memory ... consider using offload_buffers=True

    Single-device loads must NOT set it -- there is nothing to offload.
    """
    mod = _load_driver("ob1")
    # Opt-in only: on Qwen Next this measured 0.2 tok/s against 1.5 for --device-map cpu,
    # because streaming GLQ's buffers per forward IS the cost module placement avoids.
    # Defaulting it on would turn an OOM into a silent 7.5x slowdown.
    assert mod.wants_offload_buffers("auto") is False
    assert mod.wants_offload_buffers({"": 0}) is False
    assert mod.wants_offload_buffers("auto", opt_in=True) is True
    assert mod.wants_offload_buffers({"": 0}, opt_in=True) is True
    # Single-device: nothing to offload, even when asked.
    assert mod.wants_offload_buffers("cuda", opt_in=True) is False
    assert mod.wants_offload_buffers("cpu", opt_in=True) is False
