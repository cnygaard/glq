"""Which dtype a model is served and measured with — one source, device-aware.

vLLM's `--dtype auto` reads the checkpoint's declared bf16, and bf16 is aligned with NEITHER
of GLQ's kernel paths: CUDA computes fp16 (`x2d.half()`), CPU computes fp32 (`x2d.float()`).
So bf16 pays conversions everywhere and measured slowest on both, decode tok/s, repeats 5,
swap 0, 2026-09-26:

    CUDA  gemma-4-26B-A4B 3bpw, RTX PRO 6000:  fp16 9.370 > fp32 9.224 > bf16 8.304
                                              fp16 and bf16 peak alloc IDENTICAL (16.46 GiB)
    CPU   Qwen3.8-Flash-Next 3bpw, Xeon 8559C: fp32 3.072 > fp16 3.017 > bf16 2.803

Quality does not pay for it: wikitext-2 PPL is marginally BETTER in fp16 (SmolLM3-3B trellis
6bpw: 9.1145 fp16 vs 9.1337 bf16), because fp16 has 10 mantissa bits against bf16's 7. The
only fp16 risk is exponent RANGE, i.e. overflow — never precision.

These tests exist because the default reaches users through `glq-chat`/`glq-code` via
`supervisor.argv()`, and a silent drift between what we benchmark and what we serve is how a
published tok/s stops describing the product.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.tooling import FP16_UNSAFE, preferred_dtype  # noqa: E402


# ---- the mapping --------------------------------------------------------------------------

@pytest.mark.parametrize("model", [
    "xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-3bpw",
    "xv0y5ncu/SmolLM3-3B-trellis-3inst-6bpw",
    "xv0y5ncu/gemma-4-12B-it-GLQ-5bpw",
])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_verified_families_get_fp16(model, device):
    """fp16 on both paths: it is what CUDA already computes in, and it still beats bf16 on
    CPU (+7.6%) even though fp32 is marginally faster there — fp32 costs a further 4.4 GiB
    and footprint is this project's point."""
    assert preferred_dtype(model, device) == "float16"


@pytest.mark.parametrize("model", [
    "mistralai/Ministral-3-3B-Reasoning-2512",
    "mistralai/Devstral-Small-2-24B-Instruct-2512",
    "xv0y5ncu/Ministral-3-3B-GLQ-trellis-3inst-4bpw",
])
def test_fp16_unsafe_models_get_bf16(model):
    """Activation outliers exceed fp16's 65504 range and produce a hard NaN. bf16 is slower
    on every path measured and correct here, which is the whole point of the exception."""
    assert preferred_dtype(model, "cuda") == "bfloat16"
    assert preferred_dtype(model, "cpu") == "bfloat16"


def test_qwen_is_excluded_provisionally_and_the_reason_is_written_down():
    """Qwen is on the list WITHOUT a measured fp16 failure: its fp16 arm raised a dtype
    mismatch but the bf16 control was also broken (PPL 2.9e6), so neither arm is evidence.
    If this ever gets removed, it should be because a Qwen checkpoint verified — not because
    someone assumed the entry was as well-founded as the Mistral one."""
    assert preferred_dtype("xv0y5ncu/Qwen3.8-27B-GLQ-trellis-3inst-4bpw", "cuda") == "bfloat16"
    src = open(os.path.join(os.path.dirname(__file__), "..", "glq", "tooling.py")).read()
    assert "PROVISIONAL" in src, "the weaker basis for the qwen entry must stay documented"


def test_matching_is_case_insensitive():
    """Repo ids are typed by hand; a capitalised family must not slip past the denylist."""
    assert preferred_dtype("MistralAI/Ministral-3-3B", "cuda") == "bfloat16"
    assert preferred_dtype("XV0Y5NCU/Gemma-4-26B-A4B-IT-GLQ-4BPW", "cuda") == "float16"


def test_an_unknown_family_gets_fp16_not_a_refusal():
    """Deliberately a DENYLIST, unlike `tool_serve_args` which refuses unknown families. That
    rule exists because a wrong tool parser "fails silently at the worst layer"; a wrong dtype
    does not — fp16 overflow is a NaN and visibly garbage on the first token. Loud failure
    justifies opting everything in; silent failure does not."""
    assert preferred_dtype("someorg/BrandNewArch-9B-GLQ-4bpw", "cuda") == "float16"


def test_an_unknown_device_falls_back_to_fp16():
    assert preferred_dtype("xv0y5ncu/gemma-4-12B-it-GLQ-5bpw", "rocm") == "float16"


# ---- it actually reaches the served command ----------------------------------------------

def _argv(model, device):
    from glq.supervisor import VllmSupervisor
    return VllmSupervisor(model=model, device=device, vllm_bin="/x/vllm").argv()


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_the_serve_command_carries_an_explicit_dtype(device):
    """Without an explicit --dtype, vLLM's `auto` reads the checkpoint's bf16 — which is how
    the slowest option became the effective default for every user."""
    argv = _argv("xv0y5ncu/gemma-4-26B-A4B-it-GLQ-4bpw", device)
    assert "--dtype" in argv, f"{device} serve branch lost --dtype"
    assert argv[argv.index("--dtype") + 1] == "float16"


def test_the_serve_command_honours_the_exception():
    argv = _argv("mistralai/Ministral-3-3B-Reasoning-2512", "cuda")
    assert argv[argv.index("--dtype") + 1] == "bfloat16"


def test_an_explicit_user_dtype_still_wins():
    """vLLM takes the last occurrence, so a user's --dtype in extra_args must override ours
    rather than collide with it."""
    from glq.supervisor import VllmSupervisor
    argv = VllmSupervisor(model="xv0y5ncu/gemma-4-12B-it-GLQ-5bpw", device="cuda",
                          vllm_bin="/x/vllm",
                          extra_args=["--dtype", "bfloat16"]).argv()
    assert argv[-2:] == ["--dtype", "bfloat16"], argv[-4:]


def test_the_denylist_is_not_empty():
    """A refactor that emptied it would silently make fp16 unconditional, and the models it
    protects fail with a NaN rather than a clear error."""
    assert FP16_UNSAFE and all(isinstance(t, str) and t.islower() for t in FP16_UNSAFE)
