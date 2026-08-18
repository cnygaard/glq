"""Which codebook `glq-quantize` produces when the user says nothing.

The default used to be `e8_shell`, the original 8-D lattice. Trellis (QTIP TCQ) has been the
recommended path since v0.7 and wins where it matters most — SmolLM3-3B at 2 bpw is PPL
11.94 against 13.79 for the lattice — so the default now matches the recommendation.

Two things have to move together. The trellis *variant* defaulted to `hyb`, but the fused
CUDA kernels need a **3INST** checkpoint: serving a hyb checkpoint falls back to the
pure-torch decode, "correct but materially slower". A default that quietly produces
checkpoints with no fast path would be worse than the lattice default it replaced.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.quantize_model import build_parser  # noqa: E402

BASE = ["--model", "org/m", "--output", "/tmp/out"]


def test_the_default_codebook_is_trellis():
    assert build_parser().parse_args(BASE).codebook == "trellis"


def test_the_lattice_codebooks_are_still_reachable():
    """Not a removal: e8_shell is the only path for fractional and mixed bit-rates."""
    for name in ("e8_shell", "e8_relaxed", "e8p", "trellis"):
        assert build_parser().parse_args(BASE + ["--codebook", name]).codebook == name


def test_the_default_bpw_is_one_trellis_accepts():
    """trellis takes uniform integer 2-8; a fractional default would make the no-flag
    invocation fail outright."""
    bpw = build_parser().parse_args(BASE).bpw
    assert bpw == int(bpw) and 2 <= bpw <= 8


def test_the_default_trellis_variant_is_3inst():
    """`hyb` has no fused kernel — a checkpoint quantized with it serves on the slow path."""
    from glq.quantize_model import default_trellis_variant
    assert default_trellis_variant() == "3inst"


def test_the_variant_env_override_still_wins(monkeypatch):
    from glq.quantize_model import default_trellis_variant
    monkeypatch.setenv("GLQ_TRELLIS_VARIANT", "hyb")
    assert default_trellis_variant() == "hyb"


def test_fractional_bpw_names_the_codebook_that_supports_it():
    """The new default cannot do mixed precision, so the refusal has to say what can —
    otherwise `--bpw 3.5`, which worked before, becomes a dead end."""
    from glq.quantize_model import _reject_mixed_trellis
    with pytest.raises(ValueError, match=r"--codebook e8_shell"):
        _reject_mixed_trellis(codebook_type="trellis", mixed_precision=True,
                              bpw=3.5, bpw_map=None, has_range=False)
