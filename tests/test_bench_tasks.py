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
    # terminal_bench runs its own vllm server + harbor subprocess, so it cannot share the
    # quality engine; and with no bf16 reference yet it must stay out of the index.
    assert registry.get_task("terminal_bench").kind == "throughput"
    assert registry.get_task("terminal_bench").standardized is False
    # The adapter reads max_chunks; a `nsamples` key here would be silently ignored.
    assert "max_chunks" in registry.get_task("wikitext2_ppl").defaults
    assert "nsamples" not in registry.get_task("wikitext2_ppl").defaults
    assert registry.get_task("aime_2026").defaults["sets"] == ["2026"]
    with pytest.raises(KeyError):
        registry.get_task("does_not_exist")


def test_terminal_bench_serves_with_tool_calling_enabled():
    """An agent that cannot call tools cannot touch the terminal. vLLM answers pi's
    `tool_choice: "auto"` with a 400 unless BOTH flags are present, and the trial then dies
    at setup with no score — so assert the flags, not just that a command was built."""
    from glq.bench.tasks import terminal_bench as tb
    cmd = tb.serve_command("xv/M-GLQ", "glq", {}, port=8000, served_id="glq-model")
    assert "--enable-auto-tool-choice" in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "hermes"
    assert cmd[cmd.index("--served-model-name") + 1] == "glq-model"
    assert "--quantization" in cmd and "--reasoning-parser" not in cmd

    # Per family: gemma-4 does not speak the Hermes <tool_call> markup.
    cmd2 = tb.serve_command("org/M", "none", {"tool_call_parser": "pythonic",
                                              "reasoning_parser": "deepseek_r1"},
                            port=9000, served_id="m")
    assert cmd2[cmd2.index("--tool-call-parser") + 1] == "pythonic"
    assert cmd2[cmd2.index("--reasoning-parser") + 1] == "deepseek_r1"
    assert "--quantization" not in cmd2          # bf16 arm serves unquantized


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


def test_task_config_nested_by_task_name_applies_to_that_task_only():
    """A multi-task sweep needs per-task sampling, and the flat form cannot express it.

    MRCR scores verbatim reproduction and wants greedy; AIME wants the model card's
    0.6/0.95/20. One flat dict holds one `temperature`, so a sweep running both tasks
    in one invocation silently applies the wrong sampling to one of them.
    """
    overrides = {"mrcr": {"per_bucket": 2, "temperature": 0.0},
                 "aime_2025": {"temperature": 0.6, "top_k": 20}}
    cfg = runner._task_config(registry.get_task("mrcr"), n=None, budget=None,
                              overrides=overrides)
    assert cfg["per_bucket"] == 2 and cfg["temperature"] == 0.0
    # The other task's block must not leak in, under its own name or flattened.
    assert "aime_2025" not in cfg and cfg.get("top_k") != 20

    cfg = runner._task_config(registry.get_task("aime_2025"), n=None, budget=None,
                              overrides=overrides)
    assert cfg["temperature"] == 0.6 and cfg["top_k"] == 20
    assert "per_bucket" not in cfg and "mrcr" not in cfg


def test_task_config_nested_beats_flat_for_the_named_task():
    """Flat keys stay the sweep-wide default; the named block is the more specific
    statement and wins for its own task."""
    overrides = {"temperature": 1.0, "mrcr": {"temperature": 0.0}}
    mrcr = runner._task_config(registry.get_task("mrcr"), n=None, budget=None,
                               overrides=overrides)
    aime = runner._task_config(registry.get_task("aime_2025"), n=None, budget=None,
                               overrides=overrides)
    assert mrcr["temperature"] == 0.0
    assert aime["temperature"] == 1.0


def test_task_config_dict_under_a_non_task_key_stays_a_value():
    """Only real task names are treated as per-task blocks. A dict-valued setting that
    happens to be a dict must survive as its own value, or this feature would eat it."""
    overrides = {"system": {"role": "system", "content": "/no_think"}}
    cfg = runner._task_config(registry.get_task("aime_2025"), n=None, budget=None,
                              overrides=overrides)
    assert cfg["system"] == {"role": "system", "content": "/no_think"}


def test_bench_engines_cap_max_num_seqs():
    """vLLM's default is 1024 — a batch-server number. On hybrid-GDN models every decode
    sequence reserves a Mamba cache block up front, and the bf16 27B arm refused to start:
    'max_num_seqs (1024) exceeds available Mamba cache blocks (345)'. Same bug the chat
    supervisor fixed, one layer over. 64 is ample concurrency for every bench task."""
    kw = runtime.build_llm_kwargs("org/M", quant="none")
    assert kw["max_num_seqs"] == 64
    kw = runtime.build_llm_kwargs("org/M", quant="glq", max_num_seqs=16)
    assert kw["max_num_seqs"] == 16
    assert "--max-num-seqs 64" in runtime.serving_command(
        "org/M", runtime.build_llm_kwargs("org/M"))


def test_kv_cache_dtype_reaches_the_engine():
    """A KV-quantization sweep varies exactly one thing, and it has to arrive.

    vLLM has no VLLM_KV_CACHE_DTYPE env var — the setting is an engine argument —
    so a sweep that tries to select fp8 or turboquant through the environment runs
    every arm as bf16 and reports the result as if the KV dtype had changed. That
    is a silent wrong answer, not a crash, so it is pinned here.
    """
    kw = runtime.build_llm_kwargs("org/M", quant="none", kv_cache_dtype="fp8")
    assert kw["kv_cache_dtype"] == "fp8"
    assert "--kv-cache-dtype fp8" in runtime.serving_command("org/M", kw)


def test_kv_cache_dtype_is_absent_when_not_asked_for():
    """Default must stay off: passing kv_cache_dtype='auto' explicitly to older
    engines is not the same as omitting it."""
    kw = runtime.build_llm_kwargs("org/M", quant="none")
    assert "kv_cache_dtype" not in kw
    assert "--kv-cache-dtype" not in runtime.serving_command("org/M", kw)


def test_serving_command_carries_the_glq_kv_env(monkeypatch):
    """The command string is the reproduction recipe. GLQ's KV cache is selected by
    environment, not engine flags, so a command without it reproduces a *bf16* KV
    run while claiming to be the E8 one — silently, which is the failure mode that
    matters. The env belongs in front of the command, as you would type it."""
    monkeypatch.setenv("GLQ_KV_QUANT", "e8_relaxed:2")
    monkeypatch.setenv("GLQ_KV_E8_COMPRESSED_ALLOC", "1")
    kw = runtime.build_llm_kwargs("org/M", quant="none")
    cmd = runtime.serving_command("org/M", kw)
    assert cmd.startswith("GLQ_KV_QUANT=e8_relaxed:2 ") or "GLQ_KV_QUANT=e8_relaxed:2" in cmd
    assert "GLQ_KV_E8_COMPRESSED_ALLOC=1" in cmd
    assert "vllm serve org/M" in cmd


def test_serving_command_is_unchanged_without_glq_env(monkeypatch):
    """A plain run must keep the plain command — no empty env prefix."""
    for k in list(__import__("os").environ):
        if k.startswith("GLQ_"):
            monkeypatch.delenv(k, raising=False)
    kw = runtime.build_llm_kwargs("org/M", quant="none")
    assert runtime.serving_command("org/M", kw).startswith("vllm serve org/M")


def test_terminal_bench_picks_the_parser_from_the_model_family():
    """The default was a hardcoded `hermes`, which silently mis-parses every Qwen we publish
    (their templates emit `<function=`/`<parameter=` XML, not JSON inside `<tool_call>`).
    A TB-2 rollout with a wrong parser does not error — it produces a plausible-looking
    reward of 0.0 — so the default has to come from the same family table glq-code uses."""
    from glq.bench.tasks import terminal_bench as tb
    qwen = tb.serve_command("xv0y5ncu/Qwen3.8-27B-GLQ-trellis-3inst-4bpw", "glq", {},
                            port=8000, served_id="m")
    assert qwen[qwen.index("--tool-call-parser") + 1] == "qwen3_xml"
    assert qwen[qwen.index("--reasoning-parser") + 1] == "qwen3"

    smol = tb.serve_command("xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel", "glq", {},
                            port=8000, served_id="m")
    assert smol[smol.index("--tool-call-parser") + 1] == "hermes"
    assert "--reasoning-parser" not in smol       # the validated SmolLM3 pairing


def test_an_explicit_parser_still_overrides_the_family():
    """The config key is how a new family gets served before the table knows about it."""
    from glq.bench.tasks import terminal_bench as tb
    cmd = tb.serve_command("xv0y5ncu/Qwen3.8-27B-GLQ-trellis-3inst-4bpw", "glq",
                           {"tool_call_parser": "pythonic"}, port=8000, served_id="m")
    assert cmd[cmd.index("--tool-call-parser") + 1] == "pythonic"


def test_an_unknown_family_still_gets_tool_calling():
    """Falling back to hermes is how this behaved before and is better than serving an agent
    with no tool flags at all, which fails at the first turn with a 400."""
    from glq.bench.tasks import terminal_bench as tb
    cmd = tb.serve_command("org/something-new", "glq", {}, port=8000, served_id="m")
    assert "--enable-auto-tool-choice" in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "hermes"


def test_terminal_bench_serves_the_model_families_sampling():
    """Sampling belongs to the model, and on an agentic run the SERVER has to carry it: pi
    cannot send `top_k` at all -- it is not an OpenAI field and pi's config has no slot for it
    -- so a top_k the server does not pin is a top_k that never applies. Qwen's own card asks
    for temperature 1.0 / top_p 0.95 / top_k 20 in thinking mode, and `glq-code` already pins
    exactly that via the same helper. This task served whatever vLLM defaulted to."""
    from glq.bench.tasks import terminal_bench as tb
    cmd = tb.serve_command("xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4", "glq",
                           {}, port=8000, served_id="m")
    cfg = cmd[cmd.index("--override-generation-config") + 1]
    assert '"top_k":20' in cfg and '"temperature":1.0' in cfg and '"top_p":0.95' in cfg


def test_terminal_bench_keeps_the_familys_other_serve_flags():
    """`--language-model-only` is in the family args and was being dropped, because this
    function pulled the two parser VALUES out of `tool_serve_args` and discarded the rest.
    Without it a multimodal checkpoint loads vision and audio towers that a terminal agent
    never uses -- VRAM spent on weights nothing reads, and on some archs a crash on a
    `<|video|>` placeholder."""
    from glq.bench.tasks import terminal_bench as tb
    cmd = tb.serve_command("xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4", "glq",
                           {}, port=8000, served_id="m")
    assert "--language-model-only" in cmd
    # and the trio is still decided here, not duplicated from the family list
    assert cmd.count("--tool-call-parser") == 1
    assert cmd.count("--enable-auto-tool-choice") == 1
    assert cmd.count("--reasoning-parser") == 1


def test_terminal_bench_caps_max_num_seqs():
    """vLLM defaults max_num_seqs to 1024, and on a hybrid-GDN architecture every decode slot
    reserves a Mamba cache block before a single request exists. That default has already been
    measured REFUSING startup on a 96 GB card -- `max_num_seqs (1024) exceeds available Mamba
    cache blocks (399)` -- and Qwen3.8-Flash-Next is exactly such a model, so leaving this to
    vLLM means the agentic run cannot start at all.

    8 covers the concurrent rollouts harbor runs against one server and keeps both the Mamba
    cache and the per-step logits buffer small."""
    from glq.bench.tasks import terminal_bench as tb
    cmd = tb.serve_command("xv/M-GLQ", "glq", {}, port=8000, served_id="m")
    assert cmd[cmd.index("--max-num-seqs") + 1] == "8"
    cmd2 = tb.serve_command("xv/M-GLQ", "glq", {"max_num_seqs": 32}, port=8000, served_id="m")
    assert cmd2[cmd2.index("--max-num-seqs") + 1] == "32"


def test_a_custom_agent_is_passed_as_the_agent_not_an_import_path_flag():
    """harbor removed `--agent-import-path`. Since 0.24.0 `-a/--agent` takes either a builtin
    name or a `module.path:ClassName`, so the old flag is rejected at argument parsing and the
    whole run dies before a container starts.

    The oracle leg CANNOT catch this: it runs a builtin agent and never goes through ours.
    Extracted from run() for the same reason serve_command was -- a flag the agent depends on
    should be assertable without starting Docker, which is how this one rotted unnoticed."""
    from glq.bench.tasks import terminal_bench as tb
    cmd = tb.harbor_command("/usr/bin/harbor",
                            dataset="terminal-bench/terminal-bench@4.0.0", served_id="m",
                            n_attempts=1, n_tasks=3, host_ip="172.17.0.1",
                            jobs_dir="/tmp/j", n_concurrent=3)
    assert "--agent-import-path" not in cmd
    assert cmd[cmd.index("-a") + 1] == "benchmarks.harbor_pi_glq:PiGLQAgent"
    assert cmd[cmd.index("-d") + 1] == "terminal-bench/terminal-bench@4.0.0"
    assert cmd[cmd.index("-m") + 1] == "glq/m"
    assert cmd[cmd.index("-l") + 1] == "3"
    assert cmd[cmd.index("-k") + 1] == "1"
    assert cmd[cmd.index("--allow-agent-host") + 1] == "172.17.0.1"


def test_the_job_is_named_rather_than_discovered():
    """harbor's `--job-name` makes the job directory known BEFORE harbor starts, which the
    artifact mirror needs (there is nothing to sync to if the target is only learned at the
    end) and which replaces a genuinely fragile heuristic.

    `run()` used to locate its result by diffing `jobs_dir` and taking `sorted(new)[-1]`. A
    second harbor job writing a later-sorting timestamped directory into the same `jobs_dir`
    would then have been parsed as the benchmark's own result -- hit for real while running a
    cookbook smoke test alongside a live bench, and avoided only by passing a separate
    `jobs_dir` by hand."""
    from glq.bench.tasks import terminal_bench as tb
    cmd = tb.harbor_command("/usr/bin/harbor", dataset="d", served_id="m", n_attempts=1,
                            n_tasks=1, host_ip="172.17.0.1", jobs_dir="/tmp/j",
                            job_name="glq-tb-abc123")
    assert cmd[cmd.index("--job-name") + 1] == "glq-tb-abc123"


def test_the_job_name_is_unique_per_run():
    """Two runs into one jobs_dir must not collide, which is the whole point of not using a
    bare timestamp the way harbor's default does."""
    from glq.bench.tasks import terminal_bench as tb
    names = {tb.job_name() for _ in range(50)}
    assert len(names) == 50


def test_the_artifact_prefix_names_the_model_and_the_job():
    """A mirrored job has to be findable later: prefix by model and job name, and keep the
    repo id's slash so the S3 listing reads like the Hub does."""
    from glq.bench.tasks import terminal_bench as tb
    prefix = tb.artifact_prefix("xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4",
                                "glq-tb-abc123")
    assert prefix.startswith("terminal_bench/")
    assert "Qwen3.8-Flash-Next" in prefix
    assert prefix.endswith("glq-tb-abc123")
    assert ".." not in prefix and not prefix.startswith("/")


def test_the_default_dataset_is_a_pinned_version_tag():
    """Terminal-Bench is a CONTINUOUS benchmark now: versions are tags on one repo addressed
    `name@version`, where 2.0 had its own `terminal-bench-2` repo. harbor 0.24.0 documents
    --dataset as "Dataset name@version".

    Pinned rather than floating on `main`, because a benchmark whose task set changes silently
    between runs makes two recorded numbers incomparable -- and 4.0 removed 8 of 3.0's tasks
    and revised 20."""
    from glq.bench.tasks import terminal_bench as tb
    assert "@" in tb._DATASET, "a floating dataset id makes two runs incomparable"
    assert not tb._DATASET.endswith("terminal-bench-2")
    # And exactly ONE place holds it. The registry used to carry its own copy, which is the
    # one that would be forgotten on the next version tag -- it sits nowhere near the harbor
    # code, and the adapter already falls back to `_DATASET`.
    assert "dataset" not in registry.get_task("terminal_bench").defaults
