"""Per-model serving settings: what every lookup answers, frozen, so a refactor cannot move it.

`glq/tooling.py` held seven family tables, each with its own substring loop (tool parser,
text-only load, thinks-by-default, plugin, DSpark drafter, sampling, fp16 denylist), and
`supervisor.kv_cost` an eighth. A model's settings were spread over all of them, and glq-chat
and glq-code each assembled their own subset by hand.

`tests/fixtures/serve_profiles_golden.json` records, for every case below, the answer of every
public lookup as the code gave it on main 3f80b7e, BEFORE those tables became one. The test
asserts each answer is unchanged. Regenerate only for a deliberate behaviour change, and name
the change in the commit:

    python tests/test_serve_profiles.py --regen
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

GOLDEN = Path(__file__).parent / "fixtures" / "serve_profiles_golden.json"

#: Hub ids (and ids that look like box paths but do not exist, which match on the name alone).
IDS = [
    # gemma-4
    "xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw",
    "xv0y5ncu/gemma-4-E2B-it-GLQ-trellis-3inst-4bpw",
    "google/gemma-4-26B-A4B-it",
    "XV0Y5NCU/Gemma-4-26B-A4B-IT-GLQ-4BPW",
    # Qwen
    "xv0y5ncu/Qwen3.5-0.8B-GLQ-trellis-3inst-4bpw",
    "xv0y5ncu/Qwen3.8-27B-GLQ-trellis-3inst-4bpw",
    "xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4",
    "Qwen/Qwen3.8-27B",
    # Arete-27B: a Qwen3.8-27B fine-tune whose name does not say so
    "xv0y5ncu/Arete-27B-GLQ-trellis-3inst-5bpw",
    "NousResearch/Arete-27B",
    "/opt/dlami/nvme/Arete-27B-GLQ-trellis-3inst-5bpw",
    "NousResearch/Arete-9B",
    # SmolLM
    "xv0y5ncu/SmolLM3-3B-trellis-3inst-6bpw",
    "HuggingFaceTB/SmolLM3-3B",
    "xv0y5ncu/SmolLM2-360M-Instruct-trellis-3inst-4bpw",
    # Kolibri
    "xv0y5ncu/Kolibri-1-GLQ-trellis-3inst-3bpw",
    "Aleph-Alpha/Kolibri-1",
    "someorg/Kolibri-2-Preview",
    # Nemotron 3.5 and its neighbours
    "xv0y5ncu/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GLQ-trellis-3inst-4bpw",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    "xv0y5ncu/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-3bpw",
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B",
    "someorg/Nemotron-3.5-Nano-VL-8B",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
    # GLM
    "xv0y5ncu/GLM-5.3-Flash-GLQ-trellis-3inst-4bpw",
    "zai-org/GLM-5.3-Flash",
    "nvidia/GLM-5.3-Flash-NVFP4",
    "someorg/glm5-local-name",
    "zai-org/GLM-4.7",
    "THUDM/glm-4-9b-chat",
    # Mistral (dtype only)
    "mistralai/Ministral-3-3B-Reasoning-2512",
    "mistralai/Devstral-Small-2-24B-Instruct-2512",
    "xv0y5ncu/Ministral-3-3B-GLQ-trellis-3inst-4bpw",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    # unknown
    "someorg/BrandNewArch-9B-GLQ-4bpw",
    "",
]

#: Local checkpoints, named by whoever copied them: `(dir name, config.json or None)`. The names
#: carry no family on purpose, so only the config can tell (a None config means no file; a
#: string is written verbatim, to cover an unreadable one).
LOCAL = {
    "lightning30b-glq-3bpw": {"model_type": "nemotron_h",
                              "architectures": ["NemotronHForCausalLM"]},
    "flash-box-copy": {"model_type": "glm5_next",
                       "architectures": ["Glm5NextForConditionalGeneration"]},
    "q27-copy": {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]},
    "g4-local": {"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"]},
    "my-nemotron-build": None,
    "plain-build": None,
    "broken-config": "{not json",
}


def _make_local(root: Path, name: str) -> str:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    cfg = LOCAL[name]
    if isinstance(cfg, dict):
        (d / "config.json").write_text(json.dumps(cfg))
    elif isinstance(cfg, str):
        (d / "config.json").write_text(cfg)
    return str(d)


def _snapshot(model_id: str, templates_dir: Path) -> dict:
    """Every public per-model lookup's answer for `model_id`, as JSON-able values."""
    from glq import tooling as T
    from glq.supervisor import kv_cost

    tdir = str(templates_dir)

    def norm(v):
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        if isinstance(v, dict):
            return {k: norm(x) for k, x in v.items()}
        if isinstance(v, str):
            return v.replace(tdir, "<T>")
        return v

    return norm({
        "tool_serve_args": T.tool_serve_args(model_id, templates_dir=templates_dir),
        "text_only_serve_args": T.text_only_serve_args(model_id),
        "reasoning_serve_args": T.reasoning_serve_args(model_id),
        "required_plugin": T.required_plugin(model_id),
        "dspark_drafter": T.dspark_drafter(model_id),
        "sampling_serve_args": T.sampling_serve_args(model_id),
        "recommended_sampling": T.recommended_sampling(model_id),
        "preferred_dtype": {d: T.preferred_dtype(model_id, d) for d in ("cuda", "cpu", "rocm")},
        "kv_cost": kv_cost(model_id),
    })


def _case_ids():
    return [f"id:{m}" for m in IDS] + [f"local:{n}" for n in LOCAL]


def _model_for(case: str, root: Path) -> str:
    kind, _, val = case.partition(":")
    return val if kind == "id" else _make_local(root, val)


def _all_snapshots(root: Path) -> dict:
    tdir = root / "templates"
    return {case: _snapshot(_model_for(case, root), tdir) for case in _case_ids()}


# ---- the gate ------------------------------------------------------------------------------

def test_the_golden_file_covers_exactly_these_cases():
    """A case added here without regenerating would be checked against nothing."""
    assert sorted(json.loads(GOLDEN.read_text())) == sorted(_case_ids())


@pytest.mark.parametrize("case", _case_ids())
def test_every_lookup_answers_as_before(case, tmp_path):
    want = json.loads(GOLDEN.read_text())[case]
    got = _snapshot(_model_for(case, tmp_path), tmp_path / "templates")
    assert got == want


# ---- the table: one record per family ------------------------------------------------------

def _name_for_key(key) -> str:
    return "org/" + ("-".join(key) if isinstance(key, tuple) else key)


def test_every_profile_is_reached_by_its_own_first_key():
    """The table is first-match over an ordered tuple, so a broad key listed too early would
    hide a narrower profile behind it (nemotron before nemotron-3.5, kolibri before
    kolibri-1) and nothing else would fail."""
    from glq.tooling import PROFILES, profile_for
    for p in PROFILES:
        assert profile_for(_name_for_key(p.match[0]), "arch") is p, p.family


def test_family_names_are_unique():
    from glq.tooling import PROFILES
    names = [p.family for p in PROFILES]
    assert len(names) == len(set(names)), names


def test_a_template_lookup_never_reads_config_json(tmp_path):
    """Two scopes, on purpose. A fine-tune can change its chat template, so the parsers and the
    sampling follow the checkpoint's NAME (plus the aliases of the ones that were compared);
    an architecture fact (dtype) also reads config.json, because a box copy is named by
    whoever copied it."""
    from glq.tooling import profile_for
    d = _make_local(tmp_path, "flash-box-copy")
    assert profile_for(d, "template") is None
    assert profile_for(d, "arch").family == "glm-5"


def test_a_fine_tune_alias_reaches_its_family():
    from glq.tooling import profile_for
    assert profile_for("NousResearch/Arete-27B", "template").family == "qwen"
    assert profile_for("NousResearch/Arete-9B", "template") is None


def test_an_unknown_scope_is_refused():
    from glq.tooling import profile_for
    with pytest.raises(ValueError):
        profile_for("Qwen/Qwen3.8-27B", "checkpoint")


def test_the_fp16_denylist_is_read_from_the_table():
    """`FP16_UNSAFE` stays importable, now derived. Every tag the old literal carried is still
    in it, and nothing in it comes from a profile that is fp16-safe."""
    from glq.tooling import FP16_UNSAFE, PROFILES
    assert {"ministral", "mistral", "devstral", "qwen", "nemotron", "glm-5",
            "glm5"} <= set(FP16_UNSAFE)
    unsafe_keys = {k for p in PROFILES if p.fp16_unsafe for k in p.match + p.config_match}
    assert set(FP16_UNSAFE) <= unsafe_keys


# ---- what each command serves with ---------------------------------------------------------

@pytest.mark.parametrize("case", _case_ids())
def test_glq_code_serves_the_tool_args_plus_the_cards_sampling(case, tmp_path):
    """The composition glq-code, the installer's printed command and Terminal-Bench each did
    by hand. None for a family with no known tool setup: glq-code refuses rather than guess."""
    from glq import tooling as T
    m, tdir = _model_for(case, tmp_path), tmp_path / "templates"
    tool = T.tool_serve_args(m, templates_dir=tdir)
    want = None if tool is None else tool + (T.sampling_serve_args(m) or [])
    assert T.role_serve_args(m, "code", templates_dir=tdir) == want


@pytest.mark.parametrize("case", _case_ids())
def test_glq_chat_serves_the_reasoning_parser_and_the_text_only_load(case, tmp_path):
    """No tool flags: glq-chat sends no tools, and no sampling override: its sliders send
    every sampling field on each request."""
    from glq import tooling as T
    m = _model_for(case, tmp_path)
    assert T.role_serve_args(m, "chat") == T.reasoning_serve_args(m) + T.text_only_serve_args(m)


def test_an_unknown_role_is_refused():
    from glq import tooling as T
    with pytest.raises(ValueError):
        T.role_serve_args("Qwen/Qwen3.8-27B", "bench")


# ---- flags every launcher must pass: GLM-5.3 needs FULL CUDA graphs -------------------------
#
# Under vLLM 0.32.0's default `FULL_AND_PIECEWISE` graphs GLM-5.3-Flash returns NaN logits
# (greedy "lock lock lock…", logprobs refused as "Out of range float values are not JSON
# compliant: nan"); with `cudagraph_mode: FULL` the same server answers " Paris". Measured on
# GLQ's 4 bpw build and on NVIDIA's NVFP4 build alike (RTX PRO 6000, both 2026-11-10/11,
# `benchmarks/glm_5_3_flash_nan_ab/`), so it is the model under vLLM, not GLQ's kernels.
# glq-bench and run_model.py always forced FULL, which is why every benchmark was coherent while
# glq-code printed "!!!!".

GLM = ["xv0y5ncu/GLM-5.3-Flash-GLQ-trellis-3inst-4bpw", "zai-org/GLM-5.3-Flash",
       "nvidia/GLM-5.3-Flash-NVFP4"]
FULL_GRAPHS = ["--compilation-config", '{"cudagraph_mode":"FULL"}']


@pytest.mark.parametrize("model", GLM)
def test_glm_5_is_served_with_full_cuda_graphs(model):
    from glq import tooling as T
    assert T.correctness_serve_args(model, "cuda") == FULL_GRAPHS
    assert T.compilation_config(model) == {"cudagraph_mode": "FULL"}


def test_the_flag_is_compact_json():
    """Handed to subprocess raw; where a command is PRINTED it is shell-quoted separately."""
    from glq import tooling as T
    args = T.correctness_serve_args(GLM[0], "cuda")
    assert json.loads(args[1]) == {"cudagraph_mode": "FULL"} and " " not in args[1]


def test_a_glm_5_box_copy_is_caught_by_its_config(tmp_path):
    """An architecture fact, so a directory named by whoever copied it still gets it."""
    from glq import tooling as T
    assert T.correctness_serve_args(_make_local(tmp_path, "flash-box-copy"), "cuda") \
        == FULL_GRAPHS


def test_the_cpu_backend_gets_no_graph_flags():
    """The CPU path runs `--enforce-eager`: there are no CUDA graphs to choose a mode for."""
    from glq import tooling as T
    assert T.correctness_serve_args(GLM[0], "cpu") == []


@pytest.mark.parametrize("case", _case_ids())
def test_no_other_family_changes_its_graph_mode(case, tmp_path):
    """FULL for every family is a separate, measured change (time to first token and decode
    per family under glq-chat/glq-code). Until then only the measured NaN gets it."""
    from glq import tooling as T
    m = _model_for(case, tmp_path)
    arch = T.profile_for(m, "arch")
    want = FULL_GRAPHS if arch is not None and arch.family == "glm-5" else []
    assert T.correctness_serve_args(m, "cuda") == want


# ---- one compilation config: a caller's settings join ours instead of replacing them ---------
#
# vLLM 0.32.0 does not merge a repeated `--compilation-config` (read from
# `FlexibleArgumentParser.parse_args`, and measured on 18.100.241.207, 2026-11-11):
#   * whole-JSON spellings (`--compilation-config X`, `=X`, `-cc X`, `-cc=X`) are plain argparse,
#     so the LAST one is kept whole — ours + `{"mode":0}` parsed to cudagraph_mode=None;
#   * dotted spellings (`-cc.mode=0`, `--compilation-config.mode 0`) are gathered into one dict
#     that is appended at the END of argv, so they replace a whole-JSON flag too.
# So every launcher collapses all of them into one flag, later settings winning per key.

def _cc_of(argv):
    """The one compilation config in argv, asserting there is exactly one flag for it."""
    hits = [i for i, a in enumerate(argv)
            if a.split("=", 1)[0].split(".", 1)[0] in
            ("--compilation-config", "--compilation_config", "-cc")]
    assert len(hits) == 1 and argv[hits[0]] == "--compilation-config", argv
    return json.loads(argv[hits[0] + 1])


def _compact(cc):
    return json.dumps(cc, separators=(",", ":"), sort_keys=True)


_HEAD = ["/x/vllm", "serve", "m", "--port", "8000"]
_OURS_THEN = {"cudagraph_mode": "FULL"}


@pytest.mark.parametrize("theirs, want", [
    (["--compilation-config", '{"mode":0}'], {**_OURS_THEN, "mode": 0}),
    (['--compilation-config={"mode":0}'], {**_OURS_THEN, "mode": 0}),
    (["-cc", '{"mode":0}'], {**_OURS_THEN, "mode": 0}),
    (['-cc={"mode":0}'], {**_OURS_THEN, "mode": 0}),
    (["--compilation_config", '{"mode":0}'], {**_OURS_THEN, "mode": 0}),
    (["-cc.mode=0"], {**_OURS_THEN, "mode": 0}),
    (["-cc.mode", "0"], {**_OURS_THEN, "mode": 0}),
    (["--compilation-config.mode", "0"], {**_OURS_THEN, "mode": 0}),
    (["--compilation_config.mode=0"], {**_OURS_THEN, "mode": 0}),
    (["-cc.pass_config.fuse_norm_quant=false"],
     {**_OURS_THEN, "pass_config": {"fuse_norm_quant": False}}),
    # vLLM reads a non-JSON dotted value as a human-readable int, else as a plain string
    (["-cc.max_cudagraph_capture_size=1k"], {**_OURS_THEN, "max_cudagraph_capture_size": 1000}),
    (["-cc.max_cudagraph_capture_size=1K"], {**_OURS_THEN, "max_cudagraph_capture_size": 1024}),
    (["-cc.backend=eager"], {**_OURS_THEN, "backend": "eager"}),
    # vLLM keeps this one a string (decimal + binary suffix); its config validation refuses it
    (["-cc.max_cudagraph_capture_size=1.5K"],
     {**_OURS_THEN, "max_cudagraph_capture_size": "1.5K"}),
    (["-cc.splitting_ops+", "a,b"], {**_OURS_THEN, "splitting_ops": ["a", "b"]}),
    (['-cc={"pass_config":{"x":0.5}}'], {**_OURS_THEN, "pass_config": {"x": 0.5}}),
    # the caller's own setting of the same key wins
    (["-cc.cudagraph_mode=PIECEWISE"], {"cudagraph_mode": "PIECEWISE"}),
])
def test_a_callers_compilation_config_joins_ours(theirs, want):
    from glq import tooling as T
    argv = _HEAD + FULL_GRAPHS + ["--max-num-seqs", "1"] + theirs + ["--enforce-eager"]
    assert T.merge_compilation_config(argv) == \
        _HEAD + ["--compilation-config", _compact(want), "--max-num-seqs", "1", "--enforce-eager"]


def test_settings_merge_in_argv_order_and_nested_dicts_merge():
    from glq import tooling as T
    argv = _HEAD + FULL_GRAPHS + ["-cc", '{"pass_config":{"a":1},"mode":1}',
                                  "-cc.pass_config.b=2", "--compilation-config.mode", "3",
                                  "-cc.splitting_ops+", "x", "-cc.splitting_ops+", "y"]
    assert _cc_of(T.merge_compilation_config(argv)) == {
        "cudagraph_mode": "FULL", "pass_config": {"a": 1, "b": 2}, "mode": 3,
        "splitting_ops": ["x", "y"]}


@pytest.mark.parametrize("argv", [
    _HEAD,
    _HEAD + FULL_GRAPHS,
    _HEAD + ["-cc.mode=0"],
    _HEAD + ["--compilation-config", '{ "mode": 0 }'],
])
def test_one_or_no_compilation_config_is_left_exactly_as_written(argv):
    from glq import tooling as T
    assert T.merge_compilation_config(argv) == argv


@pytest.mark.parametrize("theirs", [
    ["--compilation-config", "not json"],
    ["--compilation-config", "[1, 2]"],
    ["--compilation-config", "3"],
    ["-cc"],                                   # no value
])
def test_a_value_it_cannot_read_exactly_is_left_for_vllm_to_judge(theirs):
    """A whole-JSON value that is not a JSON object, or a flag with no value: rewriting what it
    cannot read would change what the user asked for, silently. Left as written, vLLM either
    reports the error or applies its own rules, as it does today."""
    from glq import tooling as T
    argv = _HEAD + FULL_GRAPHS + theirs
    assert T.merge_compilation_config(argv) == argv


def test_vllm_parses_the_merged_flag_as_both_settings():
    """The mechanism, on the real parser: unmerged, vLLM drops FULL; merged, it keeps both."""
    pytest.importorskip("vllm")
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from glq import tooling as T
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    for theirs in (["--compilation-config", '{"mode":0}'], ["-cc.mode=0"]):
        argv = ["--model", "m", *FULL_GRAPHS, *theirs]
        before = parser.parse_args(argv).compilation_config
        after = parser.parse_args(T.merge_compilation_config(argv)).compilation_config
        assert before.cudagraph_mode is None, theirs
        assert after.cudagraph_mode.name == "FULL" and int(after.mode) == 0, theirs


def _sup_argv(model, device, extra=()):
    from glq.supervisor import VllmSupervisor
    return VllmSupervisor(model=model, device=device, vllm_bin="/x/vllm",
                          extra_args=list(extra)).argv()


def test_the_supervisor_serves_glm_5_with_full_graphs():
    """The supervisor is where glq-chat and glq-code meet, so neither can forget the flag."""
    argv = _sup_argv(GLM[0], "cuda")
    i = argv.index("--compilation-config")
    assert argv[i:i + 2] == FULL_GRAPHS


def test_the_supervisor_joins_a_callers_compilation_config_to_ours():
    """A caller's unrelated setting no longer replaces FULL (it did: vLLM keeps the last flag
    whole, and `{"mode":0}` after ours parsed to cudagraph_mode=None — the default that NaNs)."""
    argv = _sup_argv(GLM[0], "cuda", extra=["--compilation-config", '{"mode":0}'])
    assert _cc_of(argv) == {"cudagraph_mode": "FULL", "mode": 0}


def test_a_callers_own_graph_mode_still_wins():
    """Per key, the caller's explicit choice is theirs to make: the convention `--dtype` and
    `--speculative-config` already follow."""
    argv = _sup_argv(GLM[0], "cuda", extra=["-cc.cudagraph_mode=PIECEWISE"])
    assert _cc_of(argv) == {"cudagraph_mode": "PIECEWISE"}


def test_the_supervisor_leaves_other_families_and_the_cpu_alone():
    assert "--compilation-config" not in _sup_argv(GLM[0], "cpu")
    assert "--compilation-config" not in _sup_argv(
        "xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw", "cuda")


def test_decode_sweep_serves_glm_5_with_full_graphs():
    """The sweep's own `vllm serve` is a string vLLM splits with shlex; the JSON must survive."""
    import shlex
    from glq.bench.tasks.decode_sweep import _serve_flags
    flags = _serve_flags(quant="glq", dtype="bfloat16", max_model_len=2048, port=8321,
                         gpu_mem=None, serve_extra="", model=GLM[0])
    toks = shlex.split(flags)
    i = toks.index("--compilation-config")
    assert toks[i:i + 2] == FULL_GRAPHS
    other = _serve_flags(quant="glq", dtype="float16", max_model_len=2048, port=8321,
                         gpu_mem=None, serve_extra="", model="org/gemma-4-26B")
    assert "--compilation-config" not in other


def test_decode_sweep_joins_serve_extras_compilation_config_to_ours():
    import shlex
    from glq.bench.tasks.decode_sweep import _serve_flags
    toks = shlex.split(_serve_flags(
        quant="glq", dtype=None, max_model_len=2048, port=8321, gpu_mem=None, model=GLM[0],
        serve_extra="-cc.mode=0 --enforce-eager"))
    assert _cc_of(toks) == {"cudagraph_mode": "FULL", "mode": 0}
    assert toks[-1] == "--enforce-eager"


def test_decode_sweep_leaves_other_families_serve_extra_as_written():
    """No flag of ours to merge, so the user's string goes through byte for byte."""
    from glq.bench.tasks.decode_sweep import _serve_flags
    extra = "--override-generation-config '{\"top_k\": 20}'  -cc.mode=0"
    flags = _serve_flags(quant="glq", dtype=None, max_model_len=2048, port=8321,
                         gpu_mem=None, model="org/gemma-4-26B", serve_extra=extra)
    assert flags.endswith(" " + extra)


def test_terminal_bench_serves_glm_5_with_full_graphs():
    from glq.bench.tasks import terminal_bench as tb
    cmd = tb.serve_command(GLM[0], "glq", {}, port=8000, served_id="glq-model")
    i = cmd.index("--compilation-config")
    assert cmd[i:i + 2] == FULL_GRAPHS
    other = tb.serve_command("xv0y5ncu/Qwen3.8-27B-GLQ-trellis-3inst-4bpw", "glq", {},
                             port=8000, served_id="glq-model")
    assert "--compilation-config" not in other


if __name__ == "__main__" and "--regen" in sys.argv:
    with tempfile.TemporaryDirectory(prefix="glq-golden-") as tmp:
        snap = _all_snapshots(Path(tmp))
    GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN} ({len(snap)} cases)")
