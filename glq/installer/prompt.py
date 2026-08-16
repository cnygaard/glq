"""Interactive menus that survive `curl … | bash`.

Under a pipe, **stdin is the script**. `input()` would consume leftover script bytes or hit
EOF immediately, so the installer would either crash or silently choose for the user. Every
prompt here therefore reads `/dev/tty` — the actual terminal, even when stdin is a pipe —
and degrades to documented defaults when no terminal exists (CI, `docker build`, cron).

`open_tty()` returns None in that case; every function accepts `tty=None` and answers
without blocking.
"""
from __future__ import annotations

#: Component -> one-line description. `core` is mandatory; the rest are toggleable.
COMPONENTS = {
    "core": "venv + glq (always installed)",
    "vllm": "vLLM — OpenAI-compatible server for the chat UI and picode",
    "picode": "pi coding agent (installs node via nvm)",
    "chat": "Gradio chat UI",
}

#: picode is off by default: it is the only component that installs a second language
#: runtime, which is a surprising thing for an unattended install to do.
DEFAULT_COMPONENTS = ("core", "vllm", "chat")


def open_tty():
    """The controlling terminal, or None when there isn't one."""
    try:
        return open("/dev/tty", "r+")                     # noqa: SIM115
    except OSError:
        return None


def _ask(tty, question: str) -> str | None:
    """Write a question to the terminal and read one line. None on EOF."""
    tty.write(question)
    try:
        tty.flush()
    except Exception:                                     # noqa: BLE001 - StringIO in tests
        pass
    line = tty.readline()
    if line == "":
        return None
    return line.strip()


#: Spellings people actually type. Anything else re-asks rather than being guessed at.
_YES = {"y", "yes"}
_NO = {"n", "no"}


def confirm(question: str, default: bool = True, tty=None) -> bool:
    """Ask a yes/no question. Without a terminal, answer `default` without blocking.

    That last part is the whole contract: this runs at the end of `curl … | bash`, where a
    blocking read would hang CI and `docker build` forever, and where guessing wrong either
    starts a GPU server nobody asked for or skips the handoff the prompt exists to make.
    """
    if tty is None:
        return default

    suffix = "[Y/n]" if default else "[y/N]"
    prompt = f"{question} {suffix} "

    while True:
        answer = _ask(tty, prompt)
        if answer is None or answer == "":       # EOF, or Enter
            return default
        answer = answer.strip().lower()
        if answer in _YES:
            return True
        if answer in _NO:
            return False
        prompt = f"  please answer y or n {suffix} "


def select_components(default=DEFAULT_COMPONENTS, tty=None) -> tuple[str, ...]:
    """Let the user toggle components by number. Returns the chosen set.

    `core` is pinned on: everything else needs the venv, and an installer that reports
    success having installed nothing is worse than one that ignores a keystroke.
    """
    if tty is None:
        return tuple(default)

    names = list(COMPONENTS)
    chosen = set(default) | {"core"}

    lines = ["\nGLQ installer — components\n"]
    for i, name in enumerate(names, 1):
        mark = "x" if name in chosen else " "
        lines.append(f"  [{mark}] {i} {name:<7} {COMPONENTS[name]}\n")
    lines.append("\nEnter to accept, or numbers to toggle (e.g. '3'): ")

    answer = _ask(tty, "".join(lines))
    if not answer:
        return tuple(n for n in names if n in chosen)

    for tok in answer.replace(",", " ").split():
        if not tok.isdigit():
            continue
        idx = int(tok) - 1
        if not 0 <= idx < len(names):
            continue
        name = names[idx]
        if name == "core":
            continue
        chosen.symmetric_difference_update({name})

    return tuple(n for n in names if n in chosen)


def _fit_label(ranked) -> str:
    if ranked.recommended:
        return "recommended"
    if ranked.fits is None:
        return "size unknown for this GPU"
    return "fits" if ranked.fits else "too large for this GPU"


def select_model(ranked, tty=None):
    """Choose a checkpoint. Returns a `Checkpoint`.

    Without a terminal: the recommendation, or — when VRAM could not be detected and so
    nothing is recommended — the *smallest* entry. That fallback is deliberate: it is the
    only choice that cannot OOM on an unknown card, and the quickest to download for a
    first smoke test.
    """
    if not ranked:
        raise ValueError("no checkpoints to choose from")

    recommended = next((r for r in ranked if r.recommended), None)

    if tty is None:
        if recommended is not None:
            return recommended.checkpoint
        return min(ranked, key=lambda r: r.checkpoint.size_bytes).checkpoint

    default_idx = ranked.index(recommended) + 1 if recommended else 1

    lines = ["\nGLQ checkpoints (from the published collection)\n\n"]
    for i, r in enumerate(ranked, 1):
        size = f"{r.checkpoint.size_gib:6.1f} GiB"
        lines.append(f"  {i}) {r.checkpoint.short_name:<48} {size}  [{_fit_label(r)}]\n")
    lines.append(f"\nSelect [1-{len(ranked)}] (Enter = {default_idx}): ")
    question = "".join(lines)

    while True:
        answer = _ask(tty, question)
        if answer is None or answer == "":
            picked = ranked[default_idx - 1]
            break
        if answer.isdigit() and 1 <= int(answer) <= len(ranked):
            picked = ranked[int(answer) - 1]
            break
        question = f"  please enter a number 1-{len(ranked)}: "

    if picked.fits is False:
        tty.write(f"\n  warning: {picked.checkpoint.short_name} "
                  f"({picked.checkpoint.size_gib:.1f} GiB) may not fit this GPU; "
                  f"serving it can fail with an out-of-memory error.\n")

    return picked.checkpoint
