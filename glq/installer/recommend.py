"""Rank discovered checkpoints against the detected GPU.

`WEIGHT_FRACTION` is the only tuning knob here, and it is doing two jobs at once:

  * **Weights are not the whole resident set.** vLLM defaults to
    `gpu_memory_utilization=0.9`, and out of that the KV cache and activations must also be
    paid for. Sizing weights to fill the card guarantees an OOM at load.
  * **The input is disk bytes, not resident bytes.** The Hub tree API reports what the
    safetensors weigh on disk (31B = 22.4 GiB) whereas the same checkpoint resides in
    ~16.5 GiB. So the number being compared already over-states VRAM need.

The two errors are not symmetric. Recommending something too big means an OOM *after* a
multi-GiB download — the most expensive way to fail. Recommending something too small means
the user picks a bigger entry from the list, which is one keystroke. So this errs small.
"""
from __future__ import annotations

from dataclasses import dataclass

from .discovery import Checkpoint

#: Share of VRAM that may go to weights; the rest is KV cache, activations, fragmentation.
WEIGHT_FRACTION = 0.75

#: Per-command model-family preference, matched as a substring of the repo id. Measured
#: rationale (2026-08, RTX PRO 6000 evals): Qwen3.8's tool calling is native hermes markup
#: — no external template, no thought-markup leakage — and its GLQ-4bpw AIME ties bf16,
#: which is what a coding agent needs; gemma-4's 26B-A4B MoE has the fastest interactive
#: decode, which is what a chat session feels. Preference is fit-gated: it never forces a
#: checkpoint the card cannot hold.
PREFERRED_FAMILIES = {"code": "qwen", "chat": "gemma-4"}


@dataclass(frozen=True)
class Ranked:
    checkpoint: Checkpoint
    #: True/False when VRAM is known, None when it could not be detected.
    fits: bool | None
    recommended: bool


def usable_weight_bytes(vram_bytes: int) -> int:
    """VRAM a checkpoint's weights may occupy, after reserving non-weight headroom."""
    return int(vram_bytes * WEIGHT_FRACTION)


def rank(checkpoints, vram_bytes: int | None,
         prefer_family: str | None = None) -> list[Ranked]:
    """Largest-first, each marked fits/doesn't, with at most one recommended.

    With `vram_bytes=None` (no nvidia-smi, CPU-only box, container without the device) every
    entry is listed with `fits=None` and nothing is recommended — the installer should ask
    rather than bluff a recommendation it cannot justify.

    `prefer_family` narrows the recommendation to repo ids containing that substring
    (case-insensitive) when at least one such checkpoint FITS; otherwise the preference is
    ignored rather than recommending an OOM-after-download.
    """
    ordered = sorted(checkpoints, key=lambda c: c.size_bytes, reverse=True)

    if vram_bytes is None:
        return [Ranked(c, None, False) for c in ordered]

    budget = usable_weight_bytes(vram_bytes)
    # size_bytes == 0 means the tree API gave us nothing usable; never treat that as
    # "fits anywhere", or a broken repo sorts to the top of the recommendation.
    fitting = [c for c in ordered if 0 < c.size_bytes <= budget]
    if prefer_family:
        fam = [c for c in fitting if prefer_family.lower() in c.repo_id.lower()]
        if fam:
            fitting = fam

    # Prefer trellis, then size. "Biggest that fits" alone is the wrong objective: trellis
    # decodes single-stream at bf16 parity while shell/e8p are materially slower, so a
    # smaller trellis checkpoint is the better default than a larger slow one. Size only
    # breaks ties within a format. `trellis is True` — not truthiness — so an unknown
    # (None) never outranks a confirmed one.
    trellis_fitting = [c for c in fitting if c.trellis is True]
    best = (trellis_fitting or fitting or [None])[0]
    best = best.repo_id if best is not None else None

    return [Ranked(c, 0 < c.size_bytes <= budget, c.repo_id == best) for c in ordered]


def per_command_picks(checkpoints, vram_bytes: int | None, fallback: str) -> dict:
    """The per-command served-model defaults the installer records in config.json.

    `fallback` (the user's generic pick) fills any slot the family preference cannot —
    unknown VRAM, or no fitting checkpoint of that family — so glq-code/glq-chat always
    have a model and old behavior is the floor, never a regression.
    """
    picks = {}
    for command, family in PREFERRED_FAMILIES.items():
        best = next((r.checkpoint.repo_id
                     for r in rank(checkpoints, vram_bytes, prefer_family=family)
                     if r.recommended), None)
        picks[f"{command}_model"] = best or fallback
    return picks
