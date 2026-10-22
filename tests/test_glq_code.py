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

import io
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import glq.code as code  # noqa: E402
from glq.installer import configure  # noqa: E402
from glq.installer.discovery import Checkpoint, DiscoveryError  # noqa: E402
from glq.supervisor import VllmSupervisor  # noqa: E402


class _FakeSup(VllmSupervisor):
    """The REAL supervisor, with only its lifecycle stubbed out.

    `VllmSupervisor.__init__` is pure planning — it spawns nothing, writes nothing, and
    `default_log_path()` only builds a Path — so there is no reason to fake the part under
    test. Subclassing means the pre-flight fit check here runs against the real planners and
    the real `resident_bytes` / `pinned_host_bytes` / `vram_bytes` properties.

    A hand-written stand-in would have to re-declare that surface, and a stand-in that
    re-declares what it stands in for is how three tests in this project came to pass against
    a state no real object can be in. `device="cuda"` is injected by `_run_code` because the
    only I/O `__init__` can do is `detect_device()`, which shells out to nvidia-smi.
    """

    def __init__(self, events, **kw):
        # popped BEFORE the real __init__, which rejects unknown kwargs
        self._attached = kw.pop("_attached", False)
        super().__init__(**kw)
        self.events, self.kw = events, kw

    def __enter__(self):
        self.events.append("start")
        # we "spawned" it; the attach tests want proc None
        self.proc = None if self._attached else object()
        return self

    def __exit__(self, *_exc):
        self.events.append("stop")
        return False


class _Tty(io.StringIO):
    """Stands in for /dev/tty: feeds keystrokes, records what the user would see."""

    def __init__(self, keystrokes=""):
        super().__init__(keystrokes)
        self.shown = []

    def write(self, s):
        self.shown.append(s)
        return len(s)

    @property
    def screen(self):
        return "".join(self.shown)


def _run_code(monkeypatch, tmp_path, *, cfg=None, pi_exit=0, pi_missing=False,
              attach=False, vram_gib=None, ram_gib=None, swap_gib=0,
              weights_gib=None, offload=None, collection=None, keystrokes=None,
              discovery_error=False):
    """Inject every process, probe, file-write, hardware reading AND checkpoint lookup.

    The hardware defaults are all "unknown" on purpose: `glq.fit.check_fit` returns
    `fits is None` when it cannot size the box, so every test written before the pre-flight
    check existed keeps its original behaviour, and no test's result depends on the card in
    whatever machine is running pytest. A test that wants a verdict asks for one by passing
    `vram_gib` / `ram_gib` / `weights_gib`.

    `weights_gib` and `offload` are stubbed for the same reason, and they close a hole that
    predates this: `sizing_weights_bytes` and `checkpoint_offload_bytes` reach HuggingFace
    over HTTP, so without stubs the suite's results depend on network reachability and on
    what the live repos happen to declare. They default to None/zeros — the values those
    helpers already return offline, since both have a never-raise broad except.
    """
    events, made, ran, wrote = [], [], [], []

    monkeypatch.setattr(code, "sizing_weights_bytes",
                        lambda args, **kw: (None if weights_gib is None
                                            else int(weights_gib * (2 ** 30))))
    monkeypatch.setattr(code, "checkpoint_offload_bytes",
                        lambda repo: tuple(offload or (0, 0, 0)))

    monkeypatch.setattr(code, "_installed_config", lambda: dict(cfg or {}))

    def sup(**kw):
        made.append(kw)
        # device is pinned so the real __init__ never probes for a GPU.
        return _FakeSup(events, **{"device": "cuda", "_attached": attach, **kw})

    monkeypatch.setattr(code, "VllmSupervisor", sup)

    _gib = lambda v: None if v is None else int(v * (2 ** 30))      # noqa: E731
    monkeypatch.setattr(code, "_vram_bytes", lambda: _gib(vram_gib))
    monkeypatch.setattr(code, "_ram_bytes", lambda: _gib(ram_gib))
    monkeypatch.setattr(code, "_swap_bytes", lambda: _gib(swap_gib))
    monkeypatch.setattr(code, "_available_ram_bytes", lambda: _gib(ram_gib))
    # Only read on the refusal path, but it shells out to nvidia-smi, so it is injected for
    # the same reason as the rest: no test result may depend on the host running pytest.
    monkeypatch.setattr(code, "_gpu_name",
                        lambda: None if vram_gib is None else "Test GPU")

    # The picker's two seams. `collection` is a list of (repo_id, GiB) pairs; the default of
    # None means "discovery finds nothing", which keeps every pre-picker test on the path it
    # was written for. `keystrokes=None` means NO TERMINAL — `select_model` then answers
    # without prompting, which is the CI contract and must stay the default.
    def _discover(*_a, **_kw):
        if discovery_error:
            raise DiscoveryError("no network (test)")
        return [Checkpoint(rid, int(g * (2 ** 30)), True, False, 0, 0, 0)
                for rid, g in (collection or [])]

    monkeypatch.setattr(code, "cached_discover", _discover)
    tty = _Tty(keystrokes) if keystrokes is not None else None
    monkeypatch.setattr(code, "open_tty", lambda: tty)
    # `device` is pinned for the same reason the supervisor's is: detect_device() would
    # otherwise answer for whatever machine runs pytest, and on a GPU-less one every
    # checkpoint is unjudged and nothing is ever recommended.
    monkeypatch.setattr(code, "_fit_device", lambda: "cuda")

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
    empty assistant turn and a silent exit).

    HALF the window, not a quarter: pi clamps the ask against the remaining context itself
    every turn, so this is a ceiling rather than a reservation and a smaller share would cap
    reasoning for no gain. The ratio lives in `configure.pi_max_tokens` — one copy, because
    the harbor bench agent omitted this field entirely and truncated every long reasoning
    turn at pi's own 16384 default."""
    wrote = {}

    def record(path, base_url, ids, **kw):
        wrote.update(kw)

    _run_code(monkeypatch, tmp_path)
    monkeypatch.setattr(code, "write_pi_models", record)
    code.main(["--model", SMOL, "--max-model-len", "16384"])
    assert wrote["context_window"] == 16384
    assert wrote["max_tokens"] == configure.pi_max_tokens(16384) == 8192


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
    """The output budget must follow the window actually SERVED, which in auto mode is the
    supervisor's choice and not an args value that no longer exists."""
    wrote = {}

    def record(path, base_url, ids, **kw):
        wrote.update(kw)

    _run_code(monkeypatch, tmp_path)
    monkeypatch.setattr(code, "write_pi_models", record)
    monkeypatch.setattr(code, "_model_max_len", lambda repo: 262144)
    code.main(["--model", SMOL])
    assert wrote["context_window"] == 16384     # the fake supervisor's resolved window
    assert wrote["max_tokens"] == configure.pi_max_tokens(16384) == 8192


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


def test_the_window_for_offload_trade_is_opted_into(monkeypatch, tmp_path):
    """glq-code buys context with offload; glq-chat does not. The supervisor defaults the trade
    OFF, so this opt-in is the whole difference between the two front-ends -- and asserting the
    argument rather than a resulting window is what makes the test fail if the wiring is
    dropped, since the planner also reaches many windows without spending anything."""
    from glq.supervisor import WINDOW_OFFLOAD_MAX_EXTRA_GIB
    _, made, _, _ = _run_code(monkeypatch, tmp_path)
    code.main(["--model", QWEN])
    assert made[0]["window_offload_extra_gib"] == WINDOW_OFFLOAD_MAX_EXTRA_GIB
    assert WINDOW_OFFLOAD_MAX_EXTRA_GIB > 0


def test_cpu_offload_gb_reaches_the_supervisor(monkeypatch, tmp_path):
    """The escape hatch for the offload policy, now that the plan sizes for the window too: a
    session that will never fill the context can keep every token fast with `0`, and a card
    whose PCIe cost is worse than the g7 figures can disagree with the budget. The plan is
    computed inside the supervisor, and hand-running `vllm serve` loses `flashinfer_env()`."""
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


# ---- the pre-flight fit check -----------------------------------------------------------
#
# A checkpoint larger than the machine used to reach vLLM and fail with a CUDA
# out-of-memory several minutes into the weight load, with nothing naming the cause. These
# pin the refusal, and — more importantly — pin that it happens BEFORE the supervisor is
# entered, since the whole value is not paying for the download first.
#
# The motivating case: Kolibri-1 at 3 bpw (28.70 GiB, 26.77 GiB of it offloadable experts) on
# an RTX PRO 4500 Blackwell with 31.86 GiB of VRAM and 30 GiB of host RAM and no swap. The
# non-obvious part, measured, is that the CARD is fine once the offload is planned and HOST
# RAM is what binds — so a refusal that blamed the GPU would send the user shopping for
# hardware that fixes nothing.

GIB = 2 ** 30
KOLIBRI = dict(weights_gib=28.70, offload=(0, int(26.77 * GIB), 0),
               vram_gib=31.86, ram_gib=30.0, swap_gib=0)


def test_an_oversized_model_is_refused_before_the_supervisor_starts(monkeypatch, tmp_path,
                                                                    capsys):
    events, made, _, _ = _run_code(monkeypatch, tmp_path, **KOLIBRI)
    rc = code.main(["--model", SMOL])
    assert rc == 2
    assert events == [], "the refusal must precede the weight load, not follow it"
    err = capsys.readouterr().err
    assert "does not fit" in err
    assert "--force" in err


def test_the_refusal_names_host_ram_when_that_is_what_binds(monkeypatch, tmp_path, capsys):
    """Blaming the card here would be wrong AND expensive to act on. Also: it must not
    advise raising --cpu-offload-gb, which is advice the user cannot take — pinned pages
    cannot be swapped out, so deeper offload makes a host shortfall worse."""
    _run_code(monkeypatch, tmp_path, **KOLIBRI)
    code.main(["--model", SMOL])
    err = capsys.readouterr().err
    assert "Host RAM is the binding constraint" in err, err
    assert "--cpu-offload-gb" not in err


def test_force_serves_an_oversized_model_anyway(monkeypatch, tmp_path):
    """The user may know something the planner does not; --force must still work."""
    events, _, _, _ = _run_code(monkeypatch, tmp_path, **KOLIBRI)
    rc = code.main(["--model", SMOL, "--force"])
    assert rc == 0
    assert events == ["start", "pi", "stop"]


def test_a_fitting_model_is_not_refused(monkeypatch, tmp_path):
    events, _, _, _ = _run_code(monkeypatch, tmp_path, weights_gib=1.8,
                                vram_gib=31.86, ram_gib=30.0)
    assert code.main(["--model", SMOL]) == 0
    assert events == ["start", "pi", "stop"]


def test_an_unsizeable_checkpoint_is_never_refused(monkeypatch, tmp_path):
    """Offline, or a repo whose size cannot be read. The check must never be the thing that
    stops a setup that works today."""
    events, _, _, _ = _run_code(monkeypatch, tmp_path, weights_gib=None,
                                vram_gib=31.86, ram_gib=30.0)
    assert code.main(["--model", SMOL]) == 0
    assert events == ["start", "pi", "stop"]


def test_a_box_with_no_detectable_gpu_is_never_refused(monkeypatch, tmp_path):
    events, _, _, _ = _run_code(monkeypatch, tmp_path, **{**KOLIBRI, "vram_gib": None})
    assert code.main(["--model", SMOL]) == 0
    assert events == ["start", "pi", "stop"]


def test_no_serve_is_never_refused(monkeypatch, tmp_path):
    """Attaching to a server somebody else started allocates nothing here, so there is
    nothing to refuse."""
    events, _, _, _ = _run_code(monkeypatch, tmp_path, attach=True, **KOLIBRI)
    assert code.main(["--model", SMOL, "--no-serve"]) == 0
    assert "start" in events


def test_an_explicit_gpu_memory_utilization_suppresses_the_refusal(monkeypatch, tmp_path):
    """That flag IS the answer the sizing would have produced, so `vram_bytes` is passed as
    None deliberately and there is nothing left to judge against."""
    events, _, _, _ = _run_code(monkeypatch, tmp_path, **KOLIBRI)
    assert code.main(["--model", SMOL, "--gpu-memory-utilization", "0.9"]) == 0
    assert events == ["start", "pi", "stop"]


def test_swap_saves_the_same_configuration(monkeypatch, tmp_path):
    """Swap absorbs the overshoot around the pinned pages, so the predicted failure cannot
    occur. The failing control for the host gate, at the CLI level."""
    events, _, _, _ = _run_code(monkeypatch, tmp_path, **{**KOLIBRI, "swap_gib": 64.0})
    assert code.main(["--model", SMOL]) == 0
    assert events == ["start", "pi", "stop"]


def test_a_refused_gemma4_invocation_does_not_fetch_the_tool_template(monkeypatch,
                                                                      tmp_path):
    """`ensure_gemma4_template` downloads a file the refused serve will never read. The
    mechanism, not the exit code: assert the call did not happen."""
    calls = []
    _run_code(monkeypatch, tmp_path, **KOLIBRI)
    monkeypatch.setattr(code, "ensure_gemma4_template",
                        lambda: calls.append("fetched") or tmp_path / "t.jinja")
    assert code.main(["--model", GEMMA]) == 2
    assert calls == [], "a refused invocation must not download the tool template"


def test_a_served_gemma4_invocation_still_fetches_the_template(monkeypatch, tmp_path):
    """The failing control for the move: reordering must not make it stop happening."""
    calls = []
    _run_code(monkeypatch, tmp_path, weights_gib=1.8, vram_gib=31.86, ram_gib=30.0)
    monkeypatch.setattr(code, "ensure_gemma4_template",
                        lambda: calls.append("fetched") or tmp_path / "t.jinja")
    assert code.main(["--model", GEMMA]) == 0
    assert calls == ["fetched"]


# ---- the interactive picker ------------------------------------------------------------
#
# Bare `glq-code` used to print "error: no model to serve" and exit 2. It now offers the
# curated collection, sized against this machine, with the best fit preselected.
#
# Two properties carry the value and neither is visible in an exit code:
#
# * the offered list is FAMILY-FILTERED, because glq-code refuses any model whose tool
#   parser it does not know — an unfiltered menu offers models it then rejects, and the two
#   smallest, most inviting entries in the real collection are exactly those;
# * the filter runs BEFORE ranking. `rank` marks one entry of the list it is handed as
#   recommended, so filtering afterwards leaves every survivor unrecommended and
#   `select_model` falls back to index 1 — the LARGEST entry, the worst possible default.

#: SmolLM2 has no tool parser, so glq-code exits 2 on it: the realistic unsupported family.
NO_PARSER = "xv0y5ncu/SmolLM2-135M-Instruct-GLQ-block-diagonal-4bpw"
FITS = [(SMOL, 1.8), (GEMMA, 13.9), ("xv0y5ncu/Qwen3.8-27B-GLQ-trellis-3inst-4bpw", 17.8)]


def test_bare_glq_code_offers_a_menu_and_enter_takes_the_recommendation(monkeypatch,
                                                                        tmp_path):
    events, made, _, _ = _run_code(monkeypatch, tmp_path, collection=FITS,
                                   keystrokes="", vram_gib=31.86, ram_gib=64.0)
    assert code.main([]) == 0
    assert events == ["start", "pi", "stop"]
    # largest that fits, which is the Qwen entry — and `prefer_family="qwen"` for this
    # command, so the two agree here
    assert made[0]["model"].endswith("Qwen3.8-27B-GLQ-trellis-3inst-4bpw"), made[0]["model"]


def test_a_number_picks_that_entry(monkeypatch, tmp_path):
    events, made, _, _ = _run_code(monkeypatch, tmp_path, collection=FITS,
                                   keystrokes="3", vram_gib=31.86, ram_gib=64.0)
    assert code.main([]) == 0
    # largest-first ordering, so 3 is the smallest of the three
    assert made[0]["model"] == SMOL, made[0]["model"]


def test_with_no_terminal_it_auto_picks_and_never_prompts(monkeypatch, tmp_path):
    """The CI contract. `keystrokes=None` means open_tty() returns None."""
    events, made, _, _ = _run_code(monkeypatch, tmp_path, collection=FITS,
                                   vram_gib=31.86, ram_gib=64.0)
    assert code.main([]) == 0
    assert events == ["start", "pi", "stop"]
    assert made[0]["model"]


def test_the_offered_list_excludes_families_glq_code_cannot_serve(monkeypatch, tmp_path):
    """The mechanism: inspect the list handed to select_model, not the model chosen."""
    seen = {}
    _run_code(monkeypatch, tmp_path, collection=[(NO_PARSER, 0.1)] + FITS,
              keystrokes="", vram_gib=31.86, ram_gib=64.0)
    real = code.select_model
    monkeypatch.setattr(code, "select_model",
                        lambda ranked, tty=None: seen.setdefault("ranked", ranked)
                        and None or real(ranked, tty=tty))
    code.main([])
    offered = [r.checkpoint.repo_id for r in seen["ranked"]]
    assert NO_PARSER not in offered, offered
    assert len(offered) == len(FITS)


def test_the_family_filter_runs_before_ranking(monkeypatch, tmp_path):
    """If it ran after, every survivor would be `recommended=False` and the menu's default
    would silently become the largest entry. Pins that exactly one survivor is recommended."""
    seen = {}
    # the LARGEST entry is the unsupported one, which is what makes the ordering matter
    _run_code(monkeypatch, tmp_path, collection=[(NO_PARSER, 400.0)] + FITS,
              keystrokes="", vram_gib=31.86, ram_gib=64.0)
    real = code.select_model
    monkeypatch.setattr(code, "select_model",
                        lambda ranked, tty=None: seen.setdefault("ranked", ranked)
                        and None or real(ranked, tty=tty))
    code.main([])
    assert sum(r.recommended for r in seen["ranked"]) == 1, seen["ranked"]


def test_an_explicit_model_never_consults_discovery(monkeypatch, tmp_path):
    """No new network cost and no behaviour change for scripted use."""
    def boom(*_a, **_kw):
        raise AssertionError("discovery must not be consulted for an explicit --model")

    _run_code(monkeypatch, tmp_path, vram_gib=31.86, ram_gib=64.0, weights_gib=1.8)
    monkeypatch.setattr(code, "cached_discover", boom)
    assert code.main(["--model", SMOL]) == 0


def test_a_fitting_config_default_is_served_without_prompting(monkeypatch, tmp_path):
    """The installer's per-command pick must not start nagging."""
    asked = []
    _run_code(monkeypatch, tmp_path, cfg={"code_model": SMOL}, collection=FITS,
              keystrokes="", vram_gib=31.86, ram_gib=64.0, weights_gib=1.8)
    monkeypatch.setattr(code, "select_model",
                        lambda *a, **k: asked.append(1) or SMOL)
    assert code.main([]) == 0
    assert asked == [], "a config default that fits must be served as-is"


def test_a_menu_pick_that_does_not_fit_is_not_then_refused(monkeypatch, tmp_path):
    """The picker warns and obeys. If the fit check then refused the choice, the menu would
    be a liar — the two features have to agree about who decides."""
    events, _, _, _ = _run_code(monkeypatch, tmp_path,
                                collection=[(SMOL, 28.70)], keystrokes="",
                                vram_gib=31.86, ram_gib=30.0, swap_gib=0,
                                weights_gib=28.70)
    assert code.main([]) == 0
    assert events == ["start", "pi", "stop"]


def test_discovery_failure_falls_back_to_the_config_default(monkeypatch, tmp_path):
    events, made, _, _ = _run_code(monkeypatch, tmp_path, cfg={"code_model": SMOL},
                                   discovery_error=True, keystrokes="",
                                   vram_gib=31.86, ram_gib=64.0, weights_gib=1.8)
    assert code.main([]) == 0
    assert made[0]["model"] == SMOL


def test_discovery_failure_with_nothing_to_fall_back_on_is_an_error(monkeypatch, tmp_path,
                                                                    capsys):
    events, _, _, _ = _run_code(monkeypatch, tmp_path, discovery_error=True,
                                keystrokes="", vram_gib=31.86, ram_gib=64.0)
    assert code.main([]) == 2
    assert events == []
    err = capsys.readouterr().err
    assert "--model" in err, "the original hint must survive"
    assert "no network" in err, "and it should say why the menu could not be shown"
