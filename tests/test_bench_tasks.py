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
    assert {"mmlu_pro", "aime_2024", "aime_2025", "aime_2026", "wikitext2_ppl",
            "throughput", "decode_sweep", "livecodebench"} <= names
    for name in names:
        spec = registry.get_task(name)
        assert callable(spec.load())            # imports the adapter module (CPU-safe)
    assert registry.get_task("mmlu_pro").standardized is True
    assert registry.get_task("throughput").standardized is False
    assert registry.get_task("throughput").kind == "throughput"
    # decode_sweep runs `vllm bench sweep serve`, which owns its own server, so it cannot
    # share the quality tasks' engine; and it is GPU-dependent, so it must never be folded
    # into the %-of-bf16 quality index.
    assert registry.get_task("decode_sweep").kind == "throughput"
    assert registry.get_task("decode_sweep").standardized is False
    assert registry.get_task("decode_sweep").defaults["concurrencies"] == [1, 32]
    # Perplexity loads its own HF model — as "quality" it joined the shared-engine group
    # and the runner started a vLLM engine nothing used.
    assert registry.get_task("wikitext2_ppl").kind == "hf"
    # The adapter reads max_chunks; a `nsamples` key here would be silently ignored.
    assert "max_chunks" in registry.get_task("wikitext2_ppl").defaults
    assert "nsamples" not in registry.get_task("wikitext2_ppl").defaults
    assert registry.get_task("aime_2026").defaults["sets"] == ["2026"]
    with pytest.raises(KeyError):
        registry.get_task("does_not_exist")


def test_livecodebench_is_reserved_not_silent():
    """The picker table has a coding column with no harness behind it. The task must fail
    loudly — a silently-absent task reads as 'this model scores nothing at coding'."""
    run = registry.get_task("livecodebench").load()
    with pytest.raises(NotImplementedError, match="no harness"):
        run(ctx=None, config={})


# ---- AIME: the thinking gate --------------------------------------------------
class _FakeCompletion:
    def __init__(self, text, ntok):
        self.text = text
        self.token_ids = list(range(ntok))
        self.finish_reason = "stop"


class _FakeOutput:
    def __init__(self, text, ntok, k=1):
        self.outputs = [_FakeCompletion(text, ntok) for _ in range(k)]


class _FakeLLM:
    """Records what it was asked, answers correctly, at a chosen generation length."""
    def __init__(self, ntok):
        self.ntok = ntok
        self.seen_msgs = None

    def chat(self, msgs, sp, chat_template_kwargs=None, use_tqdm=None):
        self.seen_msgs = msgs
        return [_FakeOutput("the answer is \\boxed{42}", self.ntok) for _ in msgs]


class _FakeCtx:
    def __init__(self, llm):
        self.handle = type("H", (), {"llm": llm})()


class _FakeSamplingParams:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _patch_rows(monkeypatch, n=3):
    """Stub the dataset AND vllm.SamplingParams: these tests exercise the adapter's
    prompt-shaping and its thinking gate, both of which must be checkable on CPU in CI
    where vLLM is not installed."""
    import sys
    import types

    from glq.bench.tasks import aime
    monkeypatch.setattr(aime, "_rows",
                        lambda year: [(f"{year}-{i}", f"problem {i}", 42) for i in range(n)])
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = _FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)


def test_aime_omits_system_message_by_default(monkeypatch):
    """A custom system message is what puts a SmolLM3 template into /no_think. Default to
    user-only turns so the correct behaviour is the one you get without thinking about it."""
    from glq.bench.tasks import aime
    _patch_rows(monkeypatch)
    llm = _FakeLLM(ntok=15000)
    aime.run(_FakeCtx(llm), {"sets": ["2026"], "budget": 32768})
    assert all(t["role"] == "user" for turns in llm.seen_msgs for t in turns)

    aime.run(_FakeCtx(llm), {"sets": ["2026"], "budget": 32768, "system": "You are terse."})
    assert llm.seen_msgs[0][0] == {"role": "system", "content": "You are terse."}


def test_aime_rejects_a_run_that_never_engaged_thinking(monkeypatch):
    """The load-bearing guard. A no-think SmolLM3 run answers in ~1.8k tokens and looks like
    a completed thinking eval — it scores low and reads as a quantization regression. The
    only reliable signal is generation length, so it has to be enforced, not just logged."""
    from glq.bench.tasks import aime
    _patch_rows(monkeypatch)
    with pytest.raises(RuntimeError, match="never engaged"):
        aime.run(_FakeCtx(_FakeLLM(ntok=1800)), {"sets": ["2026"], "budget": 32768})


def test_aime_gate_is_scoped_to_thinking_runs(monkeypatch):
    """A deliberate no-think run is a legitimate measurement — the floor must not veto it,
    and an explicit min_mean_gen=0 must be able to turn it off for a genuinely terse model."""
    from glq.bench.tasks import aime
    _patch_rows(monkeypatch)
    res, _ = aime.run(_FakeCtx(_FakeLLM(ntok=1800)),
                      {"sets": ["2026"], "budget": 32768, "thinking": False})
    assert res.value == 1.0
    res2, _ = aime.run(_FakeCtx(_FakeLLM(ntok=1800)),
                       {"sets": ["2026"], "budget": 32768, "min_mean_gen": 0})
    assert res2.value == 1.0
    assert res2.extra["mean_gen_tokens"] == 1800


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
