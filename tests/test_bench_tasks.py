"""CPU-only unit tests for glq-bench task layer: parsers, registry, the vLLM
command builder, and the runner's skip-on-failure path. No GPU/vLLM needed."""
from __future__ import annotations

import pytest

from glq.bench import runner, runtime
from glq.bench.tasks import parse, registry


# ---- answer + log parsers ----------------------------------------------------
def test_extract_boxed_int():
    assert parse.extract_boxed_int(r"work ... \boxed{277} done") == 277
    assert parse.extract_boxed_int(r"a \boxed{1} then \boxed{42}.") == 42   # last wins
    assert parse.extract_boxed_int("the answer is 042") == 42               # fallback
    assert parse.extract_boxed_int("no digits at all here") is None


def test_extract_mmlu_letter():
    assert parse.extract_mmlu_letter("reasoning... The answer is (D).") == "D"
    assert parse.extract_mmlu_letter("so the answer is C") == "C"
    assert parse.extract_mmlu_letter("\\boxed{(B)}") == "B"
    assert parse.extract_mmlu_letter("nothing here") is None


def test_parse_load_mem_gib():
    log = "INFO ... Model loading took 16.51 GiB memory and 54.7 seconds"
    assert parse.parse_load_mem_gib(log) == 16.51
    assert parse.parse_load_mem_gib("no such line") is None


def test_parse_vllm_bench_throughput():
    p = parse.parse_vllm_bench_throughput("Output token throughput (tok/s): 430.56")
    assert p["output_tok_s"] == 430.56
    p2 = parse.parse_vllm_bench_throughput(
        "Throughput: 12.3 requests/s, 1234.5 total tokens/s, 430.6 output tokens/s")
    assert p2["output_tok_s"] == 430.6 and p2["total_tok_s"] == 1234.5
    assert parse.parse_vllm_bench_throughput("garbage")["output_tok_s"] is None


# ---- registry: every adapter imports + is callable ---------------------------
def test_registry_lists_and_loads_all_adapters():
    names = set(registry.list_tasks())
    assert {"mmlu_pro", "aime_2024", "aime_2025", "aime_2026",
            "wikitext2_ppl", "throughput"} <= names
    for name in names:
        spec = registry.get_task(name)
        assert callable(spec.load())            # imports the adapter module (CPU-safe)
    assert registry.get_task("mmlu_pro").standardized is True
    assert registry.get_task("throughput").standardized is False
    assert registry.get_task("throughput").kind == "throughput"
    assert registry.get_task("aime_2026").defaults["sets"] == ["2026"]
    with pytest.raises(KeyError):
        registry.get_task("does_not_exist")


# ---- vLLM serving command builder --------------------------------------------
def test_build_llm_kwargs_and_command():
    kw = runtime.build_llm_kwargs("xv/M-GLQ", quant="glq", max_model_len=20480,
                                  gpu_mem_util=0.9, multimodal=True)
    assert kw["quantization"] == "glq"
    assert kw["limit_mm_per_prompt"] == {"image": 0, "video": 0, "audio": 0}
    assert kw["max_model_len"] == 20480 and "compilation_config" in kw

    kw_bf16 = runtime.build_llm_kwargs("org/M", quant="none", multimodal=False)
    assert "quantization" not in kw_bf16 and "limit_mm_per_prompt" not in kw_bf16

    cmd = runtime.serving_command("org/M", kw)
    assert cmd.startswith("vllm serve xv/M-GLQ") is False  # uses passed model arg
    cmd2 = runtime.serving_command("xv/M-GLQ", kw)
    assert "vllm serve xv/M-GLQ" in cmd2
    assert "--quantization glq" in cmd2 and "--max-model-len 20480" in cmd2
    assert "--limit-mm-per-prompt" in cmd2


def test_is_multimodal():
    assert runtime.is_multimodal("Gemma4ForConditionalGeneration")
    assert not runtime.is_multimodal("LlamaForCausalLM")
    assert not runtime.is_multimodal(None)


# ---- runner skip-on-failure --------------------------------------------------
class _FakeSpec:
    name = "boomtask"
    metric = "accuracy"

    def load(self):
        def _f(ctx, cfg):
            raise ValueError("kaboom")
        return _f


def test_runner_safe_run_records_skip_not_raise():
    res, tp = runner._safe_run(_FakeSpec(), ctx=None, cfg={"x": 1})
    assert res.value is None
    assert res.standardized is False
    assert res.extra["status"] == "skipped"
    assert "kaboom" in res.extra["error"]
    assert tp is None


def test_runner_task_config_merges():
    spec = registry.get_task("mmlu_pro")
    cfg = runner._task_config(spec, n=20, budget=8192)
    assert cfg["task_name"] == "mmlu_pro" and cfg["standardized"] is True
    assert cfg["n"] == 20 and cfg["budget"] == 8192
