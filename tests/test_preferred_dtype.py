"""Which dtype a model is served and measured with — one source, and the devices DIFFER.

vLLM's `--dtype auto` reads the checkpoint's declared bf16. That is the wrong default on
CUDA, and the right one on CPU, for two unrelated reasons — which is why this maps per
device and why the tests below are split by device rather than parametrized over both.

**CUDA — fp16, because that is genuinely what the kernels compute in.** Every GLQ CUDA
entry point casts (`glq/quantized_linear.py:772/795/917/925/934`, `x2d.half()`), so bf16
activations pay a conversion at every quantized layer boundary. Measured 2026-09-26,
decode tok/s, repeats 5, swap 0, gemma-4-26B-A4B 3bpw on an RTX PRO 6000:

    fp16 9.370 > fp32 9.224 > bf16 8.304        (+12.8% fp16 over bf16)
    fp16 and bf16 peak alloc IDENTICAL (16.46 GiB) — so on CUDA fp16 is free.

**CPU — bf16, and NOT because it is aligned.** GLQ's CPU kernels compute in fp32
(`glq/quantized_linear.py:888`, `:1082`), so no 16-bit dtype is aligned there and the
choice is a footprint/speed trade. Qwen3.8-Flash-Next 3bpw on a Xeon 8559C:

    fp32 3.072 tok/s @ ~93.5 GiB · fp16 3.017 @ 89.11 · bf16 2.803 @ 82.74

fp16 is a real Pareto middle, not a dominated option — it was chosen here originally on that
basis. It is still the wrong point on the frontier: +6.4 GiB buys +7.6% on the one platform
where exhausting RAM costs far more than 7.6%, and two findings on this project were
invalidated outright by swap contamination. bf16 is additionally the only 16-bit dtype
vLLM's own CPU kernels accept — `cpu_moe.py:466/502/535` (bf16 only, a SILENT
`return False`), `scaled_mm/cpu.py:257` (bf16/fp32 only), `mamba/ops/cpu/gdn_attention.py:54`
(hard assert), all vLLM 0.29.0.

Quality does not pay for the CUDA choice: wikitext-2 PPL is marginally BETTER in fp16
(SmolLM3-3B trellis 6bpw: 9.1145 fp16 vs 9.1337 bf16), because fp16 carries 10 mantissa bits
against bf16's 7. bf16 buys exponent range by spending precision, so where activations fit
fp16's range fp16 is the more accurate of the two and the only risk is overflow.

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

VERIFIED = [
    "xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-3bpw",
    "xv0y5ncu/SmolLM3-3B-trellis-3inst-6bpw",
    "xv0y5ncu/gemma-4-12B-it-GLQ-5bpw",
]


# ---- the mapping, per device ---------------------------------------------------------------

@pytest.mark.parametrize("model", VERIFIED)
def test_cuda_gets_fp16(model):
    """fp16 is what the CUDA kernels already compute in, +12.8% at identical footprint."""
    assert preferred_dtype(model, "cuda") == "float16"


@pytest.mark.parametrize("model", VERIFIED)
def test_cpu_gets_bf16_not_fp16(model):
    """The devices must NOT agree, and this is the direction that keeps getting unified.

    fp16 on CPU is not absurd — it is the middle of the measured Pareto frontier. It is
    wrong because it costs +6.4 GiB for +7.6% on the platform where swap is the dominant
    risk, and because it is the one 16-bit dtype vLLM's CPU kernels refuse: a silent
    `return False, "kernel requires bfloat16 activations"` at `cpu_moe.py:466/502/535`
    downgrades the fused CPU experts to the per-expert loop with no error at all.
    """
    assert preferred_dtype(model, "cpu") == "bfloat16"


def test_the_two_devices_disagree():
    """Pins the split itself. A refactor that collapses the device map to one value would
    keep every other test in this file green while silently reintroducing the bug."""
    m = VERIFIED[0]
    assert preferred_dtype(m, "cuda") != preferred_dtype(m, "cpu")


@pytest.mark.parametrize("model", [
    "mistralai/Ministral-3-3B-Reasoning-2512",
    "mistralai/Devstral-Small-2-24B-Instruct-2512",
    "xv0y5ncu/Ministral-3-3B-GLQ-trellis-3inst-4bpw",
])
def test_fp16_unsafe_models_get_bf16(model):
    """Activation outliers exceed fp16's 65504 range and produce a hard NaN — the loud
    failure mode, and the one the denylist was built for."""
    assert preferred_dtype(model, "cuda") == "bfloat16"
    assert preferred_dtype(model, "cpu") == "bfloat16"


def test_qwen_next_gets_bf16_because_vllm_refuses_fp16_for_it():
    """Qwen4Exp cannot run fp16 at all: `vllm/models/qwen4_exp/nvidia/qsa.py` and
    `indexer_qsa.py` raise NotImplementedError("Qwen4Exp QSA … requires BF16") in four
    places. Reproduced on vLLM 0.29.0 — the bf16 arm scored wikitext PPL 4.8917 against
    4.8898 on record (0.04%, so the harness was sound) while the fp16 arm exited on that
    raise. This is an architecture constraint upstream, not a GLQ bug and not an overflow.
    """
    assert preferred_dtype(
        "xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4", "cuda") == "bfloat16"
    src = open(os.path.join(os.path.dirname(__file__), "..", "glq", "tooling.py")).read()
    assert "qsa" in src.lower(), "the QSA refusal is the evidence for the qwen entry"


def test_the_whole_qwen_family_stays_on_bf16_because_fp16_buys_it_nothing():
    """Qwen3.8-27B is `qwen3_5` — a GDN hybrid but with no QSA, so fp16 is NOT refused for
    it and this entry is wider than strict safety requires. It stays wide on measured
    grounds rather than caution: a GDN model in fp16 silently forfeits the fused CUDA GDN
    decode kernel, which cancels the conversion saving. Qwen3.8-27B-GLQ-4bpw, sm_120,
    vLLM 0.29.0, B=1, decode 64, repeats 3 — bf16 39.509 (39.452-40.111) vs fp16 39.472
    (39.434-39.801), overlapping, both coherent. Narrowing would buy a tie.

    If someone later wants to narrow this, the thing to produce is a Qwen arm where fp16 is
    measurably FASTER — not merely one where it works.
    """
    assert preferred_dtype("xv0y5ncu/Qwen3.8-27B-GLQ-trellis-3inst-4bpw", "cuda") == "bfloat16"


def test_matching_is_case_insensitive():
    """Repo ids are typed by hand; a capitalised family must not slip past the denylist."""
    assert preferred_dtype("MistralAI/Ministral-3-3B", "cuda") == "bfloat16"
    assert preferred_dtype("XV0Y5NCU/Gemma-4-26B-A4B-IT-GLQ-4BPW", "cuda") == "float16"


def test_an_unknown_family_gets_the_device_default_not_a_refusal():
    """Deliberately a DENYLIST, unlike `tool_serve_args` which refuses unknown families.

    The original justification was that a wrong dtype cannot fail silently. That is false —
    see `FP16_UNSAFE`'s note on the third failure mode. It survives as a denylist because
    that silent mode was MEASURED and is small: forcing the Triton GDN decode path instead
    of the fused CUDA one cost -0.26% at B=1 on a GLQ checkpoint (32.035 vs 31.951 tok/s,
    overlapping ranges over 3 repeats), and the substitution was confirmed at the op level
    (1536 calls moving cleanly between the two custom ops) with the op measuring 0.70-0.82%
    of device time — so ~0.8% is the ceiling on the whole effect, not just what one run saw.
    """
    assert preferred_dtype("someorg/BrandNewArch-9B-GLQ-4bpw", "cuda") == "float16"
    assert preferred_dtype("someorg/BrandNewArch-9B-GLQ-4bpw", "cpu") == "bfloat16"


def test_an_unknown_device_falls_back_to_bf16():
    """The conservative end: an unrecognised accelerator has not been measured, and bf16 is
    what the checkpoint declares, so it is the choice that cannot be worse than today."""
    assert preferred_dtype("xv0y5ncu/gemma-4-12B-it-GLQ-5bpw", "rocm") == "bfloat16"


# ---- it actually reaches the served command ----------------------------------------------

def _argv(model, device):
    from glq.supervisor import VllmSupervisor
    return VllmSupervisor(model=model, device=device, vllm_bin="/x/vllm").argv()


@pytest.mark.parametrize("device,expect", [("cuda", "float16"), ("cpu", "bfloat16")])
def test_the_serve_command_carries_the_right_explicit_dtype(device, expect):
    """Without an explicit --dtype, vLLM's `auto` reads the checkpoint's bf16 — which is how
    the slowest option became the effective default for every CUDA user."""
    argv = _argv("xv0y5ncu/gemma-4-26B-A4B-it-GLQ-4bpw", device)
    assert "--dtype" in argv, f"{device} serve branch lost --dtype"
    assert argv[argv.index("--dtype") + 1] == expect


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
