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
    _, _, ran, _ = _run_code(monkeypatch, tmp_path)
    code.main(["--model", SMOL, "--", "--continue", "fix the tests"])
    cmd, _env = ran[0]
    assert cmd[1:3] == ["--provider", "glq"]
    assert cmd[3:5] == ["--model", SMOL]
    assert cmd[-2:] == ["--continue", "fix the tests"]


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
