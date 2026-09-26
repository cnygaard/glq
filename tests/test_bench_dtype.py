"""The bench's engine dtype: reachable at all, and never fp16 for a baseline arm.

Two separate defects, both in what gets *measured* rather than in what gets served.

**It was unreachable.** `build_llm_kwargs`/`load` took `dtype` from the start
(`runtime.py:41`, `:181`) but nothing plumbed it from the CLI, so the hardcoded
``dtype="bfloat16"`` stood for every run. Every vLLM-backed record in the results repo was
therefore measured in bf16 whatever the model wanted — including the ones used to compare
GLQ against bf16.

**A baseline arm must not be served fp16.** `preferred_dtype` keys on the repo id, which says
nothing about `--quant none`, so the bf16 reference arm would have inherited the GLQ
checkpoint's fp16. That is not symmetric: GLQ substitutes its own kernels for the quantized
layers, so the dtype-gated fast paths it skips are precisely the ones an UNQUANTIZED model
needs — `flashinfer_utils.py:120` ("Unquantized Moe Backend FlashInfer TRTLLM requires
bfloat16 weights") on CUDA MoE, `cpu_moe.py:466` on CPU. The bias only ever runs one way:
it slows the arm GLQ is being compared against, which reads as a GLQ win.

So each arm is served in the dtype that is actually best for it — fp16 for GLQ because its
kernels compute fp16, bf16 for the baseline because that is what its weights are stored in.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.bench.runtime import build_llm_kwargs, is_baseline_quant  # noqa: E402


# ---- the predicate, extracted so the two callers cannot drift -----------------------------

@pytest.mark.parametrize("quant,expect", [
    ("none", True), ("bf16", True), (None, True),
    ("glq", False), ("awq", False), ("nvfp4", False),
])
def test_is_baseline_quant(quant, expect):
    assert is_baseline_quant(quant) is expect


@pytest.mark.parametrize("quant", ["none", "bf16", None])
def test_a_baseline_arm_does_not_pass_quantization_to_the_engine(quant):
    """The behaviour the extraction must preserve: this predicate also decides whether
    `quantization` reaches vLLM at all, so getting it wrong would serve a GLQ checkpoint
    unquantized."""
    assert "quantization" not in build_llm_kwargs("m", quant=quant)


def test_a_glq_arm_does_pass_quantization_to_the_engine():
    assert build_llm_kwargs("m", quant="glq")["quantization"] == "glq"


# ---- what the runner resolves ------------------------------------------------------------

class _Stop(Exception):
    """Sentinel: `load` reached, so the dtype decision is already made."""


def _load_kwargs(monkeypatch, *, quant, model="xv0y5ncu/gemma-4-26B-A4B-it-GLQ-4bpw",
                 dtype=None):
    """Drive `runner.run` far enough to capture the kwargs it would hand the engine.

    `run` snapshots hardware and pulls HF metadata before loading, both of which reach out.
    It imports them inside the function body, so patch the source modules rather than
    `runner`'s namespace, and stop at `load` — the point by which the dtype must be decided.
    """
    import glq.bench.hfmeta as hfmeta
    import glq.bench.provenance as prov
    import glq.bench.runner as runner
    import glq.bench.runtime as rt

    seen = {}

    def _load(model, **kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(prov, "env_snapshot", lambda: object())
    monkeypatch.setattr(prov, "hardware_snapshot", lambda: object())
    monkeypatch.setattr(hfmeta, "model_meta", lambda m, **kw: type(
        "M", (), {"quant_method": quant, "architecture": "Gemma4ForCausalLM"})())
    monkeypatch.setattr(rt, "load", _load)
    with pytest.raises(_Stop):
        runner.run(model=model, tasks=["wikitext2_ppl_vllm"], quant=quant, dtype=dtype)
    return seen


def _eff_dtype(monkeypatch, **kw):
    return _load_kwargs(monkeypatch, **kw).get("dtype")


def test_a_glq_arm_is_measured_in_fp16(monkeypatch):
    assert _eff_dtype(monkeypatch, quant="glq") == "float16"


@pytest.mark.parametrize("quant", ["none", "bf16"])
def test_a_baseline_arm_is_measured_in_bf16(monkeypatch, quant):
    """The whole point: the reference arm keeps the dtype its weights are stored in, so it
    cannot lose a kernel that GLQ never needed."""
    assert _eff_dtype(monkeypatch, quant=quant) == "bfloat16"


def test_an_explicit_dtype_wins_over_both(monkeypatch):
    """A sweep that deliberately holds dtype constant across arms must be able to."""
    assert _eff_dtype(monkeypatch, quant="glq", dtype="bfloat16") == "bfloat16"
    assert _eff_dtype(monkeypatch, quant="none", dtype="float16") == "float16"


def test_the_denylist_still_applies_to_a_glq_arm(monkeypatch):
    """Qwen4Exp cannot run fp16 at all (vLLM's QSA raises), so the family exception has to
    survive the baseline logic rather than be bypassed by it."""
    assert _eff_dtype(
        monkeypatch, quant="glq",
        model="xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4") == "bfloat16"


# ---- it has to reach the adapters that start their OWN server ----------------------------

def test_the_context_carries_the_dtype(monkeypatch):
    """`kind="quality"` tasks share the engine `run` builds, but `kind="throughput"` ones
    start their own server, so the decision has to travel on the context or it does not
    reach them at all."""
    import glq.bench.runner as runner
    assert "dtype" in runner.RunContext.__dataclass_fields__


def test_decode_sweep_serves_in_the_chosen_dtype():
    """decode_sweep produces the README's decode tok/s. Without --dtype its server takes
    vLLM's `auto` (the checkpoint's bf16), so the published speed number would describe a
    dtype `glq-chat` no longer serves -- exactly the drift `preferred_dtype` exists to stop.
    """
    from glq.bench.tasks.decode_sweep import _serve_flags
    flags = _serve_flags(quant="glq", dtype="float16", max_model_len=4096, port=8123,
                         gpu_mem=0.9, serve_extra="")
    assert "--dtype float16" in flags


def test_decode_sweep_omits_dtype_when_it_was_not_chosen():
    """Absent a dtype the flag is omitted rather than passed as `auto`, which is not the
    same thing to every engine version -- the same rule kv_cache_dtype follows above."""
    from glq.bench.tasks.decode_sweep import _serve_flags
    assert "--dtype" not in _serve_flags(quant="glq", dtype=None, max_model_len=4096,
                                        port=8123, gpu_mem=0.9, serve_extra="")


def test_the_dtype_is_recorded_in_the_serving_command():
    """A PPL number in fp16 is not the same run as one in bf16, and the command string is
    what someone copies to reproduce it."""
    cmd = __import__("glq.bench.runtime", fromlist=["serving_command"]).serving_command(
        "m", build_llm_kwargs("m", quant="glq", dtype="float16"))
    assert "--dtype float16" in cmd
