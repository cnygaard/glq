"""`glq-code` owns a tool-calling vLLM server's lifetime and runs pi against it.

The manual sequence it replaces failed four separate ways in one evening: an unsourced
nvm made `pi` resolve to nothing (and Ubuntu suggested the unrelated Raspberry-Pi
package), the serve command lacked tool flags, the flags that were printed used the
wrong parser family, and the gemma-4 tool template did not exist on disk. One command,
same supervisor architecture as glq-chat: start correct, run pi, free the GPU on exit.

Mirrors the `_run_chat` harness: every process, probe and file-write is injected, so
these tests need neither node, nor a GPU, nor the network.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import glq.code as code  # noqa: E402


class _FakeSup:
    def __init__(self, events, **kw):
        self.events, self.kw = events, kw
        # the real supervisor resolves None to a planned window; the floor stands in
        self.max_model_len = kw.get("max_model_len") or kw.get("max_model_len_floor", 8192)
        self.proc = None

    def __enter__(self):
        self.events.append("start")
        self.proc = object()             # we "spawned" it; attach tests override
        return self

    def __exit__(self, *_exc):
        self.events.append("stop")
        return False


def _run_code(monkeypatch, tmp_path, *, cfg=None, pi_exit=0, pi_missing=False,
              attach=False):
    events, made, ran, wrote = [], [], [], []

    monkeypatch.setattr(code, "_installed_config", lambda: dict(cfg or {}))

    def sup(**kw):
        made.append(kw)
        s = _FakeSup(events, **kw)
        if attach:
            class _Attached(_FakeSup):
                def __enter__(self):
                    self.events.append("start")
                    self.proc = None     # attached: we did not spawn it
                    return self
            s = _Attached(events, **kw)
        return s

    monkeypatch.setattr(code, "VllmSupervisor", sup)

    pi_bin = tmp_path / "node" / "bin" / "pi"
    if not pi_missing:
        pi_bin.parent.mkdir(parents=True, exist_ok=True)
        pi_bin.write_text("#!/usr/bin/env node\n")
    monkeypatch.setattr(code, "_find_pi", lambda: pi_bin if not pi_missing else None)

    def run_pi(cmd, env):
        events.append("pi")
        ran.append((list(map(str, cmd)), dict(env)))
        if isinstance(pi_exit, BaseException):
            raise pi_exit
        return pi_exit

    monkeypatch.setattr(code, "_run_pi", run_pi)
    monkeypatch.setattr(code, "write_pi_models",
                        lambda path, base_url, ids, **kw: wrote.append(
                            (str(path), base_url, list(ids))))
    monkeypatch.setattr(code, "ensure_gemma4_template",
                        lambda: tmp_path / "tool_chat_template_gemma4.jinja")
    return events, made, ran, wrote


GEMMA = "xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw"
SMOL = "xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel"


def test_the_server_starts_before_pi_and_stops_after_it(monkeypatch, tmp_path):
    events, made, ran, _ = _run_code(monkeypatch, tmp_path)
    rc = code.main(["--model", SMOL])
    assert rc == 0
    assert events == ["start", "pi", "stop"]
    assert made[0]["model"] == SMOL


def test_a_crashing_pi_still_frees_the_gpu(monkeypatch, tmp_path):
    events, _, _, _ = _run_code(monkeypatch, tmp_path,
                                pi_exit=RuntimeError("pi fell over"))
    with pytest.raises(RuntimeError):
        code.main(["--model", SMOL])
    assert events[-1] == "stop", "vLLM left running; the GPU stays reserved"


def test_pi_exit_code_is_propagated(monkeypatch, tmp_path):
    _run_code(monkeypatch, tmp_path, pi_exit=7)
    assert code.main(["--model", SMOL]) == 7


def test_gemma4_models_serve_with_the_gemma4_tool_stack(monkeypatch, tmp_path):
    _, made, _, _ = _run_code(monkeypatch, tmp_path)
    code.main(["--model", GEMMA])
    extra = " ".join(made[0]["extra_args"])
    assert "--tool-call-parser gemma4" in extra
    assert "--reasoning-parser gemma4" in extra
    assert "--enable-auto-tool-choice" in extra


def test_smollm3_serves_with_hermes(monkeypatch, tmp_path):
    _, made, _, _ = _run_code(monkeypatch, tmp_path)
    code.main(["--model", SMOL])
    assert "hermes" in " ".join(made[0]["extra_args"])


def test_unknown_families_are_refused_before_any_server_starts(monkeypatch, tmp_path,
                                                               capsys):
    events, made, _, _ = _run_code(monkeypatch, tmp_path)
    rc = code.main(["--model", "mistralai/Devstral-Small-2-24B"])
    assert rc == 2
    assert made == [] and events == []
    err = capsys.readouterr().err
    assert "gemma-4" in err.lower() and "smollm3" in err.lower()


def test_missing_pi_names_the_picode_component(monkeypatch, tmp_path, capsys):
    events, _, _, _ = _run_code(monkeypatch, tmp_path, pi_missing=True)
    rc = code.main(["--model", SMOL])
    assert rc == 3
    assert events == [], "started a server for an agent that cannot run"
    assert "picode" in capsys.readouterr().err


def test_models_json_is_refreshed_for_the_served_model(monkeypatch, tmp_path):
    _, _, _, wrote = _run_code(monkeypatch, tmp_path)
    code.main(["--model", SMOL])
    assert len(wrote) == 1
    path, base_url, ids = wrote[0]
    assert path.endswith(os.path.join(".pi", "agent", "models.json"))
    assert base_url.endswith("/v1")
    assert ids == [SMOL]


def test_pi_gets_the_provider_the_model_and_the_passthrough_args(monkeypatch, tmp_path):
    """Asserted on the WHOLE argv, not its tail. `cmd[-2:]` passed even when the literal
    separator leaked into pi's arguments, because the leak lands at index 5, not the end --
    and argparse does keep the `--` inside a REMAINDER capture (`pi_args` really is
    `['--', '--continue', ...]`), so the strip in code.py is load-bearing, not defensive."""
    _, _, ran, _ = _run_code(monkeypatch, tmp_path)
    code.main(["--model", SMOL, "--", "--continue", "fix the tests"])
    cmd, _env = ran[0]
    assert cmd[1:] == ["--provider", "glq", "--model", SMOL,
                       "--continue", "fix the tests"], cmd


def test_pis_child_path_contains_its_own_bin_dir(monkeypatch, tmp_path):
    """npm bin shims are `#!/usr/bin/env node`: resolving pi's path is not enough, node's
    bin dir must be on the child's PATH or the shim fails at the shebang."""
    _, _, ran, _ = _run_code(monkeypatch, tmp_path)
    code.main(["--model", SMOL])
    _cmd, env = ran[0]
    assert str(tmp_path / "node" / "bin") in env["PATH"].split(os.pathsep)


def test_the_model_defaults_from_the_installed_config(monkeypatch, tmp_path):
    _, made, _, _ = _run_code(monkeypatch, tmp_path, cfg={"model": SMOL})
    assert code.main([]) == 0
    assert made[0]["model"] == SMOL


def test_no_model_anywhere_is_an_error(monkeypatch, tmp_path, capsys):
    _run_code(monkeypatch, tmp_path)
    assert code.main([]) == 2
    assert "--model" in capsys.readouterr().err


def test_attaching_to_an_existing_server_warns_about_tool_flags(monkeypatch, tmp_path,
                                                                capsys):
    """A server someone else started (glq-chat's, say) probably lacks the tool flags, and
    pi's requests will 400 — say so instead of letting it look like a broken model."""
    _run_code(monkeypatch, tmp_path, attach=True)
    code.main(["--model", SMOL])
    assert "tool" in capsys.readouterr().err.lower()


def test_models_json_carries_the_window_and_a_capped_output_budget(monkeypatch,
                                                                   tmp_path):
    """pi treats maxTokens as its per-turn output ask; without a cap it requests the full
    window and vLLM 400s every call (measured: the first live glq-code run produced an
    empty assistant turn and a silent exit). A quarter of the window leaves room for the
    transcript to grow across tool turns."""
    wrote = {}

    def record(path, base_url, ids, **kw):
        wrote.update(kw)

    _run_code(monkeypatch, tmp_path)
    monkeypatch.setattr(code, "write_pi_models", record)
    code.main(["--model", SMOL, "--max-model-len", "16384"])
    assert wrote["context_window"] == 16384
    assert wrote["max_tokens"] == 4096


def test_glq_code_plans_its_window_with_the_coding_floor(monkeypatch, tmp_path):
    """A coding agent's floor is 16384 (file contents + diffs); the tiering above it is
    the supervisor's job."""
    _, made, _, _ = _run_code(monkeypatch, tmp_path)
    monkeypatch.setattr(code, "_model_max_len", lambda repo: 262144)
    code.main(["--model", SMOL])
    assert made[0]["max_model_len"] is None
    assert made[0]["max_model_len_floor"] == 16384
    assert made[0]["model_max_len"] == 262144


def test_the_pi_budget_follows_the_planned_window(monkeypatch, tmp_path):
    """maxTokens = window/4 must use the window actually served, which in auto mode is
    the supervisor's choice, not an args value that no longer exists."""
    wrote = {}

    def record(path, base_url, ids, **kw):
        wrote.update(kw)

    _run_code(monkeypatch, tmp_path)
    monkeypatch.setattr(code, "write_pi_models", record)
    monkeypatch.setattr(code, "_model_max_len", lambda repo: 262144)
    code.main(["--model", SMOL])
    assert wrote["context_window"] == 16384     # the fake supervisor's resolved window
    assert wrote["max_tokens"] == 4096


QWEN = "xv0y5ncu/Qwen3.8-27B-GLQ-trellis-3inst-4bpw"


def test_code_serves_the_code_model_over_the_generic_pick(monkeypatch, tmp_path):
    """The installer records code_model (Qwen: native hermes tool calling, AIME at bf16
    parity) separately from chat's pick; glq-code must serve it when --model is absent."""
    _, made, _, _ = _run_code(monkeypatch, tmp_path,
                              cfg={"model": GEMMA, "code_model": QWEN})
    code.main([])
    assert made[0]["model"] == QWEN


def test_an_explicit_model_flag_still_wins(monkeypatch, tmp_path):
    _, made, _, _ = _run_code(monkeypatch, tmp_path,
                              cfg={"model": GEMMA, "code_model": QWEN})
    code.main(["--model", SMOL])
    assert made[0]["model"] == SMOL


# ------------------------------------------------- one stream, and a window sized for it

# `--max-model-len 262144` was unreachable on the coding path for two reasons at once: the
# window planner priced every window at chat concurrency (8 streams = ~52 GiB of KV), and
# `--max-num-seqs` came from the chat default of 16. pi issues one request at a time, so both
# were describing a server nobody was running.

def test_a_coding_session_is_priced_as_the_single_stream_it_is(monkeypatch, tmp_path):
    """One stream is the difference between 131072 and the model's full 262144 on the same
    card, so this is the parameter that decides whether the window is reachable at all."""
    _, made, _, _ = _run_code(monkeypatch, tmp_path)
    code.main(["--model", SMOL])
    assert made[0]["max_num_seqs"] == 1
    assert made[0]["window_concurrency"] == 1


def test_the_admitted_concurrency_and_the_priced_concurrency_cannot_disagree(monkeypatch,
                                                                            tmp_path):
    """Raising --max-num-seqs must re-price the window too. Letting them drift is how a
    server ends up admitting eight requests into a pool sized for one."""
    _, made, _, _ = _run_code(monkeypatch, tmp_path)
    code.main(["--model", SMOL, "--max-num-seqs", "4"])
    assert made[0]["max_num_seqs"] == 4
    assert made[0]["window_concurrency"] == 4


def test_the_coding_window_actually_reaches_the_declared_maximum(monkeypatch, tmp_path):
    """End of the chain, with the REAL supervisor rather than the fake: a 96 GB card and
    Flash-Next's post-offload resident size must produce 262144 and a pool that holds it.
    The fake supervisor above cannot prove this — it echoes the floor back."""
    import glq.supervisor as sup_mod
    GIB = 2**30
    window = sup_mod.plan_max_model_len(
        weights_bytes=int(49.53 * GIB), vram_bytes=95 * GIB, model_max_len=262144,
        floor=code.DEFAULT_CODE_MAX_MODEL_LEN,
        concurrency=code.DEFAULT_CODE_MAX_NUM_SEQS)
    assert window == 262144
    util = sup_mod.plan_gpu_memory_utilization(
        weights_bytes=int(49.53 * GIB), vram_bytes=95 * GIB,
        kv_bytes=sup_mod.window_kv_bytes(window, code.DEFAULT_CODE_MAX_NUM_SEQS))
    pool = util * 95 * GIB - int(49.53 * GIB) - sup_mod._RUNTIME_OVERHEAD_BYTES
    assert pool >= 6.55 * GIB, f"262144 promised against {pool / GIB:.2f} GiB of KV"


def test_the_coding_floor_stays_small_for_boxes_with_no_vram_reading(monkeypatch, tmp_path):
    """The floor is what a CPU box and an unknown card fall back to. Raising it to 262144
    would hand a CPU server a quarter-million-token window against an 8 GiB pool — the
    tiering is what lifts the window, never the floor."""
    assert code.DEFAULT_CODE_MAX_MODEL_LEN == 16384


def test_the_served_model_s_own_sampling_is_pinned(monkeypatch, tmp_path):
    """pi cannot send top_k — it is not in the OpenAI schema — so a Qwen coding session
    samples at vLLM's top_k=0 unless the server pins the card's 20."""
    import json
    _, made, _, _ = _run_code(monkeypatch, tmp_path)
    code.main(["--model", QWEN])
    extra = made[0]["extra_args"]
    cfg = json.loads(extra[extra.index("--override-generation-config") + 1])
    assert cfg["top_k"] == 20


def test_sampling_args_do_not_displace_the_tool_args(monkeypatch, tmp_path):
    """Both go through extra_args; a coding session with correct sampling and no tool
    parser is useless, so this asserts they coexist rather than one replacing the other."""
    _, made, _, _ = _run_code(monkeypatch, tmp_path)
    code.main(["--model", QWEN])
    extra = " ".join(made[0]["extra_args"])
    assert "--enable-auto-tool-choice" in extra
    assert "--tool-call-parser qwen3_xml" in extra
    assert "--override-generation-config" in extra


def test_all_three_non_resident_counts_reach_the_supervisor(monkeypatch, tmp_path):
    """The wiring, which is the part that silently does nothing when it breaks: a planner
    that never receives `nontext_bytes` sizes exactly as it did before and nothing fails."""
    _, made, _, _ = _run_code(monkeypatch, tmp_path)
    monkeypatch.setattr(code, "checkpoint_offload_bytes", lambda repo: (11, 22, 33))
    code.main(["--model", QWEN])
    assert made[0]["ple_offload_bytes"] == 11
    assert made[0]["expert_offload_bytes"] == 22
    assert made[0]["nontext_bytes"] == 33


def test_cpu_offload_gb_reaches_the_supervisor(monkeypatch, tmp_path):
    """The escape hatch for the offload policy. Without it there is no way to ask for a longer
    window than `WEIGHT_FRACTION` happens to allow: the budget is computed inside the
    supervisor, and hand-running `vllm serve` loses `flashinfer_env()`."""
    _, made, _, _ = _run_code(monkeypatch, tmp_path)
    code.main(["--model", QWEN, "--cpu-offload-gb", "42"])
    assert made[0]["expert_offload_gib"] == 42


def test_an_absent_cpu_offload_gb_leaves_the_planner_in_charge(monkeypatch, tmp_path):
    """None, not 0 -- 0 is a real answer meaning 'serve this resident'."""
    _, made, _, _ = _run_code(monkeypatch, tmp_path)
    code.main(["--model", QWEN])
    assert made[0]["expert_offload_gib"] is None


def test_cpu_offload_gb_zero_is_distinguishable_from_unset(monkeypatch, tmp_path):
    _, made, _, _ = _run_code(monkeypatch, tmp_path)
    code.main(["--model", QWEN, "--cpu-offload-gb", "0"])
    assert made[0]["expert_offload_gib"] == 0


# ------------------------------------- the `--` separator: discoverable, and loud when wrong

# `glq-code -- --continue` already worked; what it lacked was any way to find out. Two failure
# modes, verified by lifting the real parser out of main() and exercising it:
#
#   glq-code --continue   -> exit 2, "unrecognized arguments: --continue", no mention of `--`
#   glq-code --c 3        -> SILENTLY set cpu_offload_gb=3 (prefix match), pi never saw it
#
# The second is the dangerous one: `--c`/`--r`/`--v`/`--no-s` are unique prefixes of glq-code's
# own options, so an abbreviated pi flag is swallowed AND can eat the following token as its
# value. `allow_abbrev=False` turns each into an error that the hint then explains.

def _parse_fails(monkeypatch, tmp_path, argv):
    """(exit_code, stderr) for an invocation that should not reach pi at all."""
    import contextlib, io
    _run_code(monkeypatch, tmp_path)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        with pytest.raises(SystemExit) as e:
            code.main(argv)
    return e.value.code, err.getvalue()


@pytest.mark.parametrize("flag", ["--continue", "--resume", "-c", "-r"])
def test_a_pi_flag_without_the_separator_names_the_separator(monkeypatch, tmp_path, flag):
    """The whole point: the old message was `unrecognized arguments: --continue` and left the
    reader to guess. REMAINDER cannot absorb a leading-dash token -- the option branch claims
    it first -- so the separator is mandatory and the error has to say so."""
    rc, err = _parse_fails(monkeypatch, tmp_path, ["--model", SMOL, flag])
    assert rc == 2
    assert "--" in err and "pi" in err.lower(), err
    assert f"-- {flag}" in err, f"the hint does not show the fix for {flag}:\n{err}"


@pytest.mark.parametrize("argv,swallowed_by", [
    (["--c", "3"], "cpu_offload_gb"),
    (["--r", "99"], "ready_timeout"),
    (["--v"], "verbose"),
    (["--no-s"], "serve"),
])
def test_an_abbreviated_pi_flag_is_not_silently_captured(monkeypatch, tmp_path, argv,
                                                         swallowed_by):
    """Each of these is a unique prefix of a glq-code option, so prefix matching consumed it
    and pi never received the flag the user typed. `--c 3` reading as `--cpu-offload-gb 3` is
    the worst: it looks like it worked, and it also ate the `3`.

    This test fails today by SUCCEEDING -- the command parses fine and silently does the wrong
    thing -- which is why it is the regression guard for `allow_abbrev=False`."""
    rc, err = _parse_fails(monkeypatch, tmp_path, ["--model", SMOL, *argv])
    assert rc == 2, f"{argv} still parses; it is being captured as {swallowed_by}"


@pytest.mark.parametrize("argv", [
    ["--cpu-offload-gb", "42"],
    ["--max-model-len", "65536"],
    ["--max-num-seqs", "4"],
    ["--no-serve"],
    ["--verbose"],
])
def test_the_full_spellings_still_work(monkeypatch, tmp_path, argv):
    """Disabling abbreviation must not touch the real flags."""
    _, made, _, _ = _run_code(monkeypatch, tmp_path)
    code.main(["--model", SMOL, *argv])
    assert made, f"{argv} no longer reaches the supervisor"


def test_an_unrelated_parse_error_gets_no_separator_hint(monkeypatch, tmp_path):
    """A hint on every parse failure is noise, and standing noise trains a reader past the
    error that matters -- the same reasoning spot_scout records for AuthFailure. So the hint
    is conditional on the extras looking like flags."""
    rc, err = _parse_fails(monkeypatch, tmp_path,
                           ["--model", SMOL, "--max-model-len", "not-a-number"])
    assert rc == 2
    assert "separator" not in err.lower(), err


def test_the_help_shows_how_to_reach_pi(monkeypatch, tmp_path):
    """The convention lived in one `help=` string on a positional nobody reads. The examples
    block is where someone looks first."""
    import contextlib, io
    _run_code(monkeypatch, tmp_path)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        with pytest.raises(SystemExit):
            code.main(["--help"])
    text = out.getvalue()
    assert "-- --continue" in text, text
    assert "-- --resume" in text, text
    assert "pi --help" in text, "the help should point at pi's own flag list"


def test_plain_glq_code_passes_pi_nothing_extra(monkeypatch, tmp_path):
    """An empty REMAINDER must not leave a stray separator or empty string in pi's argv."""
    _, _, ran, _ = _run_code(monkeypatch, tmp_path)
    code.main(["--model", SMOL])
    cmd, _env = ran[0]
    assert cmd[1:] == ["--provider", "glq", "--model", SMOL], cmd
