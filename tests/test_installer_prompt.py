"""Interactive selection (glq/installer/prompt.py).

The trap this module exists for: under `curl … | bash`, **stdin is the script itself**. A
naive `input()` reads leftover script text, or hits EOF and the installer either crashes or
silently takes a default the user never saw. The fix is to talk to `/dev/tty` explicitly,
which is the real terminal even when stdin is a pipe — and to degrade cleanly when there is
no terminal at all (CI, a Dockerfile RUN line, a cron job).

So every function here takes an explicit tty handle: a file object when one exists, None
when it doesn't. Tests drive both paths with StringIO, no terminal required.
"""
from __future__ import annotations

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.installer import prompt as P  # noqa: E402
from glq.installer.discovery import Checkpoint  # noqa: E402
from glq.installer.recommend import Ranked  # noqa: E402

GIB = 1024 ** 3


class _Tty(io.StringIO):
    """A StringIO that also collects what was written, standing in for /dev/tty."""

    def __init__(self, keystrokes=""):
        super().__init__(keystrokes)
        self.shown = []

    def write(self, s):          # what the user would see
        self.shown.append(s)
        return len(s)

    @property
    def screen(self):
        return "".join(self.shown)


def _ranked():
    return [
        Ranked(Checkpoint("xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8", int(22.4 * GIB)),
               True, True),
        Ranked(Checkpoint("xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw", int(5.8 * GIB)), True, False),
        Ranked(Checkpoint("xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel", int(1.8 * GIB)),
               True, False),
        Ranked(Checkpoint("xv0y5ncu/too-big", int(400 * GIB)), False, False),
    ]


# ----------------------------------------------------------------- components

def test_no_tty_returns_the_default_set_without_blocking():
    """The `curl | bash` case with no terminal: must not hang waiting for a key."""
    chosen = P.select_components(P.DEFAULT_COMPONENTS, tty=None)
    assert chosen == P.DEFAULT_COMPONENTS


def test_enter_accepts_the_default_set():
    tty = _Tty("\n")
    assert P.select_components(P.DEFAULT_COMPONENTS, tty=tty) == P.DEFAULT_COMPONENTS


def test_a_number_toggles_that_component():
    """picode is off by default because it installs node; typing its number adds it."""
    tty = _Tty("3\n")
    chosen = P.select_components(P.DEFAULT_COMPONENTS, tty=tty)
    assert "picode" in chosen
    assert "picode" not in P.DEFAULT_COMPONENTS


def test_toggling_twice_removes_a_default_component():
    tty = _Tty("4\n")          # 4 = chat, which is on by default
    assert "chat" not in P.select_components(P.DEFAULT_COMPONENTS, tty=tty)


def test_core_cannot_be_switched_off():
    """Everything else depends on the venv; letting it be deselected produces an installer
    that reports success having installed nothing."""
    tty = _Tty("1\n")
    assert "core" in P.select_components(P.DEFAULT_COMPONENTS, tty=tty)


def test_eof_falls_back_to_the_default():
    """User pressed Ctrl-D. Not an error, just 'use the defaults'."""
    assert P.select_components(P.DEFAULT_COMPONENTS, tty=_Tty("")) == P.DEFAULT_COMPONENTS


def test_the_menu_names_every_component():
    tty = _Tty("\n")
    P.select_components(P.DEFAULT_COMPONENTS, tty=tty)
    for name in ("core", "vllm", "picode", "chat"):
        assert name in tty.screen


# ---------------------------------------------------------------------- model

def test_no_tty_takes_the_recommendation():
    picked = P.select_model(_ranked(), tty=None)
    assert picked.repo_id == "xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8"


def test_enter_takes_the_recommendation():
    picked = P.select_model(_ranked(), tty=_Tty("\n"))
    assert picked.repo_id == "xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8"


def test_a_number_picks_that_model():
    picked = P.select_model(_ranked(), tty=_Tty("3\n"))
    assert picked.repo_id == "xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel"


def test_sizes_and_fit_are_shown_so_the_choice_is_informed():
    """A bare list of nine repo ids is not a choice anyone can make; the size is the whole
    point, since picking wrong costs a multi-GiB download."""
    tty = _Tty("\n")
    P.select_model(_ranked(), tty=tty)
    assert "22.4" in tty.screen and "1.8" in tty.screen
    assert "recommended" in tty.screen.lower()


def test_models_that_do_not_fit_are_marked_but_still_listed():
    tty = _Tty("\n")
    P.select_model(_ranked(), tty=tty)
    assert "too-big" in tty.screen
    assert "too large" in tty.screen.lower() or "does not fit" in tty.screen.lower()


def test_an_out_of_range_choice_reprompts_rather_than_crashing():
    picked = P.select_model(_ranked(), tty=_Tty("99\n2\n"))
    assert picked.repo_id == "xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw"


def test_garbage_input_reprompts():
    picked = P.select_model(_ranked(), tty=_Tty("banana\n3\n"))
    assert picked.repo_id == "xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel"


def test_picking_a_model_that_does_not_fit_is_allowed_but_warned():
    """Their card, their call — a user may know something we don't (a second GPU, an
    offload plan). We warn and obey rather than refuse."""
    tty = _Tty("4\n")
    picked = P.select_model(_ranked(), tty=tty)
    assert picked.repo_id == "xv0y5ncu/too-big"
    assert "warn" in tty.screen.lower() or "may not fit" in tty.screen.lower()


def test_no_recommendation_still_lets_the_user_choose():
    """Unknown VRAM: nothing is recommended, but the list must still be usable."""
    unranked = [Ranked(r.checkpoint, None, False) for r in _ranked()]
    picked = P.select_model(unranked, tty=_Tty("2\n"))
    assert picked.repo_id == "xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw"


def test_no_recommendation_and_no_tty_falls_back_to_the_smallest():
    """Nothing known about the GPU and nobody to ask: the smallest checkpoint is the only
    choice that cannot OOM, and it is the fastest to download for a first smoke test."""
    unranked = [Ranked(r.checkpoint, None, False) for r in _ranked()]
    picked = P.select_model(unranked, tty=None)
    assert picked.repo_id == "xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel"


# ------------------------------------------------- the offer to start GLQ straight away
#
# The installer ends by asking whether to start serving and open the chat. That question is
# the difference between "install finished, now read four steps" and "install finished, here
# is your model" — the gap between GLQ and Ollama or LM Studio for a first-time user.
#
# It has to obey the same contract as every other prompt here: under `curl … | bash` stdin is
# the *script*, so the question goes to /dev/tty, and where there is no terminal (CI, docker
# build, --yes) it must answer from the default without blocking.

def test_confirm_with_no_tty_takes_the_default_without_blocking():
    assert P.confirm("Start GLQ now?", default=True, tty=None) is True
    assert P.confirm("Start GLQ now?", default=False, tty=None) is False


def test_enter_accepts_the_default():
    assert P.confirm("Start GLQ now?", default=True, tty=_Tty("\n")) is True
    assert P.confirm("Start GLQ now?", default=False, tty=_Tty("\n")) is False


@pytest.mark.parametrize("keys", ["y\n", "Y\n", "yes\n", " y \n"])
def test_yes_in_its_usual_spellings(keys):
    assert P.confirm("Start GLQ now?", default=False, tty=_Tty(keys)) is True


@pytest.mark.parametrize("keys", ["n\n", "N\n", "no\n"])
def test_no_in_its_usual_spellings(keys):
    assert P.confirm("Start GLQ now?", default=True, tty=_Tty(keys)) is False


def test_confirm_on_eof_falls_back_to_the_default():
    """A closed terminal must not hang the installer or silently start a GPU server."""
    assert P.confirm("Start GLQ now?", default=True, tty=_Tty("")) is True


def test_an_unrecognised_answer_is_asked_again():
    """Same contract as `select_model`: a typo re-asks rather than deciding for the user.
    Guessing here either starts a GPU server nobody asked for or skips the whole point of
    the prompt."""
    tty = _Tty("maybe\ny\n")

    assert P.confirm("Start GLQ now?", default=False, tty=tty) is True
    assert tty.screen.count("Start GLQ now?") >= 1
    assert "y or n" in tty.screen


def test_the_default_is_visible_in_the_question():
    """[Y/n] vs [y/N] is the only cue for what Enter does."""
    tty = _Tty("\n")
    P.confirm("Start GLQ now?", default=True, tty=tty)
    assert "[Y/n]" in tty.screen
    tty = _Tty("\n")
    P.confirm("Start GLQ now?", default=False, tty=tty)
    assert "[y/N]" in tty.screen


def test_quantize_is_offered_but_not_default():
    """`glq-quantize` ships in every install (it is a console script of the base package)
    but its deps do not — the component closes that gap for people who want it, while the
    serving-only default stays lean."""
    assert "quantize" in P.COMPONENTS
    assert "quantize" not in P.DEFAULT_COMPONENTS
