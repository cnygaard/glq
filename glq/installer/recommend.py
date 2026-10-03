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

#: Share of system RAM that may go to weights on a CPU-only box. Deliberately lower than
#: the VRAM fraction: RAM also holds the OS, page cache, the Python process and the
#: separate VLLM_CPU_KVCACHE_SPACE pool — and oversizing here means swap-death rather
#: than a clean OOM. Erring small costs one keystroke in the menu.
CPU_WEIGHT_FRACTION = 0.5

#: Per-command model-family preference, matched as a substring of the repo id. Measured
#: rationale (2026-08, RTX PRO 6000 evals): Qwen3.8's tool calling needs no external chat
#: template — its own emits the markup, and the reasoning parser keeps thought markup out
#: of prose — and its GLQ-4bpw AIME ties bf16,
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


def usable_weight_bytes(vram_bytes: int, weight_fraction: float = WEIGHT_FRACTION) -> int:
    """Memory a checkpoint's weights may occupy, after reserving non-weight headroom."""
    return int(vram_bytes * weight_fraction)


#: Share of host RAM that pinned offloaded weights may occupy. Must agree with
#: `supervisor._PINNED_RAM_FRACTION`: if this gate credits offload the supervisor will then
#: refuse to perform, the menu offers a checkpoint that cannot serve.
PINNED_RAM_FRACTION = 0.5


def _resident_floor(c, ram_bytes: int | None = None) -> int:
    """VRAM a checkpoint must actually hold, allowing for declared host offload.

    Reads the property when present and falls back to ``size_bytes``, so this works with the
    bare ``Checkpoint(repo_id, size)`` built by ``--model`` on the command line and with any
    stub a test passes in — neither carries the offload fields.

    ``ram_bytes`` caps the credit at what can actually be PINNED. Without it this gate and
    `supervisor.plan_expert_offload_gib` can disagree: a 23 GiB card with 32 GiB of RAM would
    be offered a checkpoint whose floor fits, and then the supervisor would decline to offload
    (pinned pages cannot swap) and serve 48 GiB of weights into 23 GiB of VRAM.
    """
    floor = getattr(c, "resident_floor_bytes", None)
    if not (isinstance(floor, int) and floor > 0):
        return int(c.size_bytes)
    if ram_bytes:
        # Only the OFFLOADED bytes need pinning. `nontext_bytes` (MTP head, vision tower) are
        # never loaded by a text-only serve, so they consume no host memory and must not be
        # charged against the pinnable budget — doing so would withdraw a credit a box with
        # modest RAM is entitled to, on account of weights nothing reads.
        nontext = int(getattr(c, "nontext_bytes", 0) or 0)
        offloadable = int(c.size_bytes) - int(floor) - nontext
        pinnable = int(float(ram_bytes) * PINNED_RAM_FRACTION)
        if offloadable > pinnable:
            return int(c.size_bytes) - nontext - max(0, pinnable)
    return int(floor)


def rank(checkpoints, vram_bytes: int | None,
         prefer_family: str | None = None,
         weight_fraction: float = WEIGHT_FRACTION,
         require_trellis: bool = False,
         ram_bytes: int | None = None) -> list[Ranked]:
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

    budget = usable_weight_bytes(vram_bytes, weight_fraction)
    # size_bytes == 0 means the tree API gave us nothing usable; never treat that as
    # "fits anywhere", or a broken repo sorts to the top of the recommendation.
    #
    # The gate is the RESIDENT floor, not the file size: a checkpoint that declares
    # host-offloadable bytes (a PLE table, MoE experts) does not have to hold them in VRAM.
    # Gating on size_bytes judged Qwen3.8-Flash-Next unservable on EVERY card -- 72.2 GiB of
    # safetensors against a 96 GB card's 71.2 GiB budget -- while it demonstrably serves at
    # 49.53 GiB with the n-gram table offloaded and 12.15 GiB with the experts too. Undeclared
    # offload bytes are 0, so every existing checkpoint is gated exactly as before.
    fitting = [c for c in ordered
               if 0 < c.size_bytes and _resident_floor(c, ram_bytes) <= budget]
    if require_trellis:
        # The CPU backend serves trellis — dense or MoE, the latter since the fused CPU
        # expert kernel landed. e8p/shell still refuse at load (their expert and dequant
        # paths are CUDA-only), and unknown (None) is ineligible rather than assumed:
        # recommending gigabytes of download into a hard refusal at serve time is the
        # worst failure this gate can produce.
        fitting = [c for c in fitting if c.trellis is True]
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

    # `fits` uses the same resident floor as the recommendation gate above. Deriving the two
    # from different measures is how the menu ends up showing a [fits] label on a checkpoint it
    # will not recommend, or the reverse.
    return [Ranked(c, 0 < c.size_bytes and _resident_floor(c, ram_bytes) <= budget,
                   c.repo_id == best) for c in ordered]


def per_command_picks(checkpoints, vram_bytes: int | None, fallback: str,
                      weight_fraction: float = WEIGHT_FRACTION,
                      require_trellis: bool = False) -> dict:
    """The per-command served-model defaults the installer records in config.json.

    `fallback` (the user's generic pick) fills any slot the family preference cannot —
    unknown VRAM, or no fitting checkpoint of that family — so glq-code/glq-chat always
    have a model and old behavior is the floor, never a regression.
    """
    picks = {}
    for command, family in PREFERRED_FAMILIES.items():
        best = next((r.checkpoint.repo_id
                     for r in rank(checkpoints, vram_bytes, prefer_family=family,
                                   weight_fraction=weight_fraction,
                                   require_trellis=require_trellis)
                     if r.recommended), None)
        picks[f"{command}_model"] = best or fallback
    return picks
