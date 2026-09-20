"""The meta skeleton must report its true size, or `device_map="auto"` places blindly.

transformers swaps in GLQ's modules at `preprocess_model` (`modeling_utils.py:4033`) and only
then infers the device map from the model it now sees (`:4053`). So whatever GLQ's skeleton
claims to weigh is what accelerate budgets against. Two things made it lie:

* the trellis `E8RHTLinear` registers ``trellis_packed`` **0-size** and resizes it at load
  ("K/bpw is checkpoint-authoritative"), so ~43 GiB of decoder weights were invisible;
* ``replace_with_glq_embedding`` never passed ``bpw``, so ``TrellisRHTEmbedding`` took its
  default of 3 and sized a 23.84 GiB table as 17.88.

Measured consequence on Qwen3.8-Flash-Next + an L40S: `device_map="auto"` OOMed trying to put
the 23.84 GiB PLE on a card with 18.30 GiB free, and with `max_memory={0: "20GiB"}` accelerate
fit the under-reported table inside the budget and then the load grew it to 23.84 GiB.

Two non-causes, checked so nobody designs around them again: accelerate **does** count buffers
(`named_module_tensors(..., include_buffers=True)`), and user `max_memory` **is** honoured
(`transformers/integrations/accelerate.py:347`). Only the inputs were wrong.

Sizes here are a placement *hint*. Resize-on-load stays authoritative, so a wrong hint costs a
bad device map, never wrong weights — the last test pins that.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("safetensors", reason="safetensors not installed")
import torch.nn as nn  # noqa: E402

from glq.quantized_linear import E8RHTLinear, TrellisRHTEmbedding  # noqa: E402

# A real gemma-4-26B-A4B expert: [704, 2816] at 4 bpw packs to (7744, 64) int16 --
# (704//16)*(2816//16) = 44*176 = 7744 rows, 16*K = 64 columns (glq/trellis.py:854).
OUT, IN, K = 704, 2816, 4
PACKED = (7744, 64)


def _write_ckpt(tmp_path, tensors, sharded=False):
    """Write a checkpoint the peek helpers can read: single file or index + shards."""
    import json
    from safetensors.torch import save_file
    d = tmp_path
    d.mkdir(parents=True, exist_ok=True)
    if not sharded:
        save_file(tensors, str(d / "model.safetensors"))
        return str(d)
    weight_map, i = {}, 0
    for k, v in tensors.items():
        fn = f"shard{i % 2}.safetensors"
        weight_map.setdefault(fn, {})[k] = v
        weight_map[fn] = weight_map[fn]
        i += 1
    index = {"weight_map": {}}
    for fn, group in weight_map.items():
        save_file(group, str(d / fn))
        for k in group:
            index["weight_map"][k] = fn
    (d / "model.safetensors.index.json").write_text(json.dumps(index))
    return str(d)


def _linear_tensors(prefix="model.layers.0.mlp.down_proj"):
    return {
        f"{prefix}.trellis_packed": torch.zeros(PACKED, dtype=torch.int16),
        f"{prefix}.SU": torch.ones(OUT, dtype=torch.float16),
        f"{prefix}.SV": torch.ones(IN, dtype=torch.float16),
        f"{prefix}.Wscale": torch.ones((), dtype=torch.float32),
    }


# ---- the shape peek ------------------------------------------------------------------

def test_collect_shapes_reads_a_single_file_checkpoint(tmp_path):
    from glq.hf_integration import _collect_quantized_shapes
    p = _write_ckpt(tmp_path / "a", _linear_tensors())
    shapes = _collect_quantized_shapes(p)
    assert shapes["model.layers.0.mlp.down_proj"]["trellis_packed"] == list(PACKED)


def test_collect_shapes_reads_a_sharded_checkpoint(tmp_path):
    """The real layout: an index plus N shards. Headers only — no tensor data is read."""
    from glq.hf_integration import _collect_quantized_shapes
    p = _write_ckpt(tmp_path / "b", _linear_tensors(), sharded=True)
    shapes = _collect_quantized_shapes(p)
    assert shapes["model.layers.0.mlp.down_proj"]["trellis_packed"] == list(PACKED)


def test_collect_shapes_returns_none_when_unreadable(tmp_path):
    """Older callers pass no path; behaviour must be exactly as before."""
    from glq.hf_integration import _collect_quantized_shapes
    assert _collect_quantized_shapes(str(tmp_path / "nope")) in (None, {})


# ---- the skeleton --------------------------------------------------------------------

def test_trellis_linear_can_be_built_at_its_true_packed_size():
    """The specific bug: this was `torch.zeros(0)`, so the layer weighed nothing."""
    lin = E8RHTLinear(IN, OUT, bias=False, codebook_type="trellis",
                      packed_shape=PACKED)
    assert tuple(lin.trellis_packed.shape) == PACKED
    assert lin.trellis_packed.numel() == PACKED[0] * PACKED[1]


def test_trellis_linear_without_a_hint_is_unchanged():
    """No checkpoint to peek -> the historical 0-size registration, resized at load."""
    lin = E8RHTLinear(IN, OUT, bias=False, codebook_type="trellis")
    assert lin.trellis_packed.numel() == 0


def test_shell_skeleton_size_is_untouched():
    """Shell registers Qidxs at full size already; this change must not perturb it."""
    lin = E8RHTLinear(IN, OUT, bias=False, codebook_type="e8_shell")
    assert lin.Qidxs.numel() > 0


def test_embedding_uses_the_checkpoint_rate_not_the_default():
    """`bpw` defaults to 3. A 4 bpw table built at 3 is under-reported by 25%, which is
    what let accelerate fit it inside a budget it then overran at load."""
    vocab, width = 4096, 160
    at3 = TrellisRHTEmbedding(num_embeddings=vocab, embedding_dim=width, bpw=3)
    at4 = TrellisRHTEmbedding(num_embeddings=vocab, embedding_dim=width, bpw=4)
    assert at4.trellis_packed.numel() > at3.trellis_packed.numel()


# ---- the property that actually governs placement ------------------------------------

def test_skeleton_size_matches_the_checkpoint_bytes(tmp_path):
    """What accelerate measures. Build the module at the peeked shape and compare its
    tensor bytes against the checkpoint's."""
    from glq.hf_integration import _collect_quantized_shapes
    p = _write_ckpt(tmp_path / "c", _linear_tensors())
    shapes = _collect_quantized_shapes(p)["model.layers.0.mlp.down_proj"]
    lin = E8RHTLinear(IN, OUT, bias=False, codebook_type="trellis",
                      packed_shape=tuple(shapes["trellis_packed"]))
    skeleton = sum(b.numel() * b.element_size() for _, b in lin.named_buffers())
    ckpt = sum(v.numel() * v.element_size() for v in _linear_tensors().values())
    assert skeleton >= ckpt * 0.95, (skeleton, ckpt)


def test_a_wrong_hint_is_still_corrected_on_load():
    """The hint must never outrank the checkpoint: resize-on-load stays authoritative, so
    a bad hint costs a bad device map, never wrong weights."""
    lin = E8RHTLinear(IN, OUT, bias=False, codebook_type="trellis",
                      packed_shape=(16, 16))          # deliberately wrong
    real = torch.zeros(PACKED, dtype=torch.int16)
    sd = {"trellis_packed": real,
          "SU": torch.ones(OUT, dtype=torch.float16),
          "SV": torch.ones(IN, dtype=torch.float16),
          "Wscale": torch.ones((), dtype=torch.float32)}
    lin.load_state_dict(sd, strict=False)
    assert tuple(lin.trellis_packed.shape) == PACKED, "load must override the hint"


# ---- stacked experts are the bulk, and a separate code path --------------------------

def test_stacked_experts_are_sized_from_the_checkpoint():
    """Expert linears are built by `_replace_stacked_gated_experts`, not
    `replace_with_glq_linear`, so they need the shapes threaded separately. On
    Qwen3.8-Flash-Next they are ~43 GiB -- the bulk of the model. Left 0-size, the PLE
    alone (23.84 GiB) is all accelerate sees and everything else looks free.

    Only the FUSED gate_up_proj is sized: the gate_proj/up_proj landing pads are transient
    (fuse_gate_up drops them after load), so sizing all three would double-count.
    """
    from glq.fused_experts import _replace_stacked_gated_experts

    E, I, H = 2, 704, 2816
    packed_half = (7744, 64)                       # one [704, 2816] half at 4 bpw

    class _Native(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(E, 2 * I, H))
            self.down_proj = nn.Parameter(torch.zeros(E, H, I))
            self.act_fn = nn.SiLU()

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = _Native()

    shapes = {}
    for e in range(E):
        shapes[f"experts.{e}.gate_proj"] = {"trellis_packed": list(packed_half)}
        shapes[f"experts.{e}.up_proj"] = {"trellis_packed": list(packed_half)}
        shapes[f"experts.{e}.down_proj"] = {"trellis_packed": list(packed_half)}

    m = _M()
    _replace_stacked_gated_experts(m, codebook_type="trellis", shapes=shapes)
    pair = m.experts[0]
    # gate_up is the two halves stacked: twice the rows, same columns.
    assert tuple(pair.gate_up_proj.trellis_packed.shape) == (2 * packed_half[0],
                                                             packed_half[1])
    assert tuple(pair.down_proj.trellis_packed.shape) == packed_half
    # the landing pads stay empty so the fused buffer is not counted three times
    assert pair.gate_proj.trellis_packed.numel() == 0
    assert pair.up_proj.trellis_packed.numel() == 0
