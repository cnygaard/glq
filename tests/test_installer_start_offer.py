"""The last thing the installer does: offer to start GLQ.

Without it, a successful `curl … | bash` ends with a wall of numbered steps and a shell
prompt — the user has installed something and has nothing to show for it. Ollama ends with
a model you can talk to; LM Studio opens a window. The offer closes that gap using the one
command that now does everything (`glq-chat` starts vLLM, opens the chat, and stops the
server on exit).

Two things must stay true, because they are the ways this could do harm:

* **It must never start a GPU server nobody asked for.** `--yes`, `--dry-run`, `docker
  build` and CI all run without a terminal, and an installer that seizes a card there is
  worse than one that prints instructions.
* **It must never run after a failed self-check.** That path already refuses to print
  "GLQ is installed"; launching the chat anyway would be the same lie with a UI.
"""
from __future__ import annotations

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.installer import __main__ as M  # noqa: E402
from glq.installer import verify as V  # noqa: E402
from glq.installer.discovery import Checkpoint  # noqa: E402

GIB = 1024 ** 3
FLEET = [Checkpoint("xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel",
                    int(1.8 * GIB), trellis=True)]


@pytest.fixture
def offline(monkeypatch):
    """No network, no GPU, no pip, and no chat actually launched."""
    monkeypatch.setattr(M.discovery, "discover", lambda *a, **k: FLEET)
    monkeypatch.setattr(M.hardware, "gpu_name", lambda *a, **k: "Fake GPU")
    monkeypatch.setattr(M.hardware, "vram_bytes", lambda *a, **k: int(24 * GIB))
    monkeypatch.setattr(M, "_install_python_extras", lambda *a, **k: None)
    monkeypatch.setattr(M.configure, "write_glq_config", lambda *a, **k: None)
    monkeypatch.setattr(M.configure, "write_pi_models", lambda *a, **k: None)
    # The install path runs its self-check as a SUBPROCESS of the venv's python (that
    # venv does not exist here, and in production the check cannot run in-process — pip
    # has just swapped torch underneath it; see M._self_check). Fake that seam, not
    # run_checks, which is what the child would call.
    monkeypatch.setattr(M, "_self_check", lambda *a, **k:
                        (0, V.render([V.Check("glq importable", True, "glq 0.8.6")])))

    started = []
    monkeypatch.setattr(M, "_start_chat", lambda venv: started.append(venv))
    return started


@pytest.fixture
def broken(monkeypatch):
    monkeypatch.setattr(M, "_self_check", lambda *a, **k:
                        (1, V.render([V.Check("glq_vllm importable", False,
                                              "glq_vllm is missing")])))


def _asked(capsys):
    return "start" in capsys.readouterr().out.lower()


# --------------------------------------------------------- never without being asked

def test_yes_installs_and_stops_without_starting_anything(offline):
    """`--yes` is what install.sh uses non-interactively and what CI runs. Starting a
    server there would hold a GPU for the rest of the job."""
    rc = M.main(["--yes", "--components", "core,vllm,chat"])
    assert rc == 0
    assert offline == [], "started the chat during a non-interactive install"


def test_dry_run_starts_nothing(offline):
    M.main(["--yes", "--dry-run", "--components", "core,vllm,chat"])
    assert offline == []


def test_no_start_wins_over_everything(offline):
    M.main(["--yes", "--start", "--no-start", "--components", "core,vllm,chat"])
    assert offline == []


def test_a_failed_self_check_never_gets_as_far_as_the_offer(offline, broken):
    """Same rule as the "GLQ is installed" banner: do not dress a broken install up."""
    rc = M.main(["--yes", "--start", "--components", "core,vllm,chat"])
    assert rc != 0
    assert offline == []


def test_it_is_not_offered_when_the_chat_was_not_installed(offline):
    M.main(["--yes", "--start", "--components", "core,vllm"])
    assert offline == [], "there is no chat UI to start"


# ------------------------------------------------------------------ when asked for

def test_start_launches_the_chat_from_the_installed_venv(offline):
    """The venv is never activated, so this has to be the absolute path — the same reason
    every command in the summary is absolute."""
    rc = M.main(["--yes", "--start", "--components", "core,vllm,chat",
                 "--venv", "/home/u/.glq/venv"])
    assert rc == 0
    assert [str(v) for v in offline] == ["/home/u/.glq/venv"]


def test_the_summary_is_printed_before_the_chat_takes_over(offline, capsys):
    """`glq-chat` replaces this process, so anything printed after it is never printed.
    The steps must already be in the scrollback when the user stops the chat."""
    printed = []
    M.main(["--yes", "--start", "--components", "core,vllm,chat"])
    printed.append(capsys.readouterr().out)
    assert "GLQ is installed" in printed[0]
    assert "glq-chat" in printed[0]


# ------------------------------------------------------------------- the prompt itself

def test_the_offer_is_made_on_the_terminal_when_there_is_one(offline, monkeypatch):
    """Under `curl … | bash` stdin is the script, so the question goes to /dev/tty — the
    same contract as every other prompt in the installer."""
    asked = []

    def fake_confirm(question, default=True, tty=None):
        asked.append((question, default, tty))
        return False

    monkeypatch.setattr(M.prompt, "confirm", fake_confirm)
    monkeypatch.setattr(M.prompt, "open_tty", lambda: io.StringIO())

    M.main(["--components", "core,vllm,chat", "--model", FLEET[0].repo_id])

    # There is more than one prompt now (the KV-cache question comes first, next to the
    # model choice), so name the one under test rather than assuming an order.
    start_qs = [a for a in asked if "start glq" in a[0].lower()]
    assert start_qs, f"the installer never offered to start GLQ; asked: {[a[0] for a in asked]}"
    assert start_qs[0][1] is True, "the offer should default to yes"
    assert start_qs[0][2] is not None, "the question was not put to the terminal"


def test_declining_the_offer_still_reports_a_successful_install(offline, monkeypatch):
    monkeypatch.setattr(M.prompt, "confirm", lambda *a, **k: False)
    monkeypatch.setattr(M.prompt, "open_tty", lambda: io.StringIO())
    rc = M.main(["--components", "core,vllm,chat", "--model", FLEET[0].repo_id])
    assert rc == 0
    assert offline == []


def test_accepting_the_offer_starts_the_chat(offline, monkeypatch):
    monkeypatch.setattr(M.prompt, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(M.prompt, "open_tty", lambda: io.StringIO())
    M.main(["--components", "core,vllm,chat", "--model", FLEET[0].repo_id])
    assert len(offline) == 1


def test_a_missing_glq_chat_is_reported_rather_than_thrown(tmp_path, capsys):
    """This runs after "GLQ is installed." has already been printed. A traceback there
    reads as a failed install, when in fact everything but the handoff worked."""
    M._start_chat(tmp_path)                      # tmp_path has no bin/glq-chat

    out = capsys.readouterr().out
    assert "glq-chat" in out
    assert "Traceback" not in out


# ------------------------------------------------- offering KV-cache compression

# The E8 KV cache buys ~2.7x the context for a measured -22% decode on an L4. That is a real
# choice, not a default — so the installer asks, records the answer, and glq-chat inherits it.

def test_kv_compression_is_off_unless_chosen(offline, monkeypatch):
    written = {}
    monkeypatch.setattr(M.configure, "write_glq_config",
                        lambda path, **kw: written.update(kw))
    M.main(["--yes", "--components", "core,vllm,chat"])
    assert written.get("fp8_kv") in (False, None)


def test_the_flag_records_the_choice_for_the_chat_to_inherit(offline, monkeypatch):
    """Answering once should be enough; glq-chat reads it back out of config.json."""
    written = {}
    monkeypatch.setattr(M.configure, "write_glq_config",
                        lambda path, **kw: written.update(kw))
    M.main(["--yes", "--fp8-kv-cache", "--components", "core,vllm,chat"])
    assert written.get("fp8_kv") is True


def test_the_question_states_the_trade_rather_than_just_asking(offline, monkeypatch):
    """"Enable KV compression? [y/N]" tells a first-time user nothing they can decide on."""
    asked = []
    monkeypatch.setattr(M.prompt, "confirm",
                        lambda q, default=True, tty=None: asked.append(q) or False)
    monkeypatch.setattr(M.prompt, "open_tty", lambda: io.StringIO())

    M.main(["--components", "core,vllm,chat", "--model", FLEET[0].repo_id])

    kv_qs = [q for q in asked if "kv" in q.lower() or "context" in q.lower()]
    assert kv_qs, f"never offered; asked only: {asked}"
    assert "precision" in kv_qs[0].lower(), (
        f"the question states the gain but not the cost: {kv_qs[0]!r}")
