"""The recorded shape manifests, and what they are for.

`tests/models/*.json` hold a checkpoint's tensor keys, shapes and dtypes with no weight data.
They exist because hand-written fixtures encode the author's model of an architecture rather
than the architecture, and three fixtures in this suite were wrong that way: 16x8 shapes the
production filter rejects outright, a bare ``...mlp.experts`` ``layer_bpw`` key no checkpoint
contains, and a fake MoE layer that returned None unconditionally. Each made a test pass
against a state that cannot occur.

These tests guard the manifests themselves — a silently wrong manifest would be worse than no
manifest, because every test built on it would agree with it.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_shapes import ModelShapes, available, collapse, load  # noqa: E402

RECORDED = available()


def test_at_least_the_two_reference_models_are_recorded():
    assert "qwen3_8_flash_next_glq_3bpw" in RECORDED
    assert "gemma_4_26b_a4b_it" in RECORDED


@pytest.mark.parametrize("slug", RECORDED)
def test_every_manifest_names_its_BASE_MODEL_and_that_model_s_LICENSE(slug):
    """A manifest is derived from somebody else's release, and it is checked into a public
    repo. It carries no weights — only tensor names, shapes and dtypes — but the base model's
    terms still govern what was derived from, and the licenses differ between the two
    recorded here: gemma-4 is Apache 2.0 while Qwen3.8-Flash-Next is the **Qwen Community
    License 1.0**, a custom license with its own notice requirement. Guessing "probably
    Apache" for a Qwen model would be wrong.

    Asserted rather than documented because `record.py` writes new manifests mechanically,
    and an unattributed one would otherwise be committed without anybody noticing.
    """
    m = load(slug)
    for field in ("base_model", "license", "license_url"):
        val = m.meta.get(field)
        assert val, f"{slug}: manifest does not record {field}"
    assert m.meta["license_url"].startswith("http"), f"{slug}: license_url is not a link"


@pytest.mark.parametrize("slug", RECORDED)
def test_expansion_reproduces_the_recorded_key_count_exactly(slug):
    """The compression is lossless or it is misleading. Flash-Next collapses 297,256 keys to
    149 patterns (45.8 KiB); if expansion drifted, every test reading the manifest would
    quietly ask about keys the model does not have."""
    m = load(slug)
    keys = m.keys()
    assert len(keys) == m.n_keys, f"{slug}: expanded {len(keys)} vs recorded {m.n_keys}"
    assert len(set(keys)) == m.n_keys, f"{slug}: expansion produced duplicates"


@pytest.mark.parametrize("slug", RECORDED)
def test_every_expanded_key_resolves_to_a_shape(slug):
    """`shape_of` is what production consumes (`mtp_sublayer_weights(shape_of=...)`), so a
    key that expands but does not resolve would make the filter skip a real tensor."""
    m = load(slug)
    for k in m.keys():
        assert m.shape_of(k) is not None, f"{slug}: {k} expands but has no shape"


@pytest.mark.parametrize("slug", RECORDED)
def test_an_unknown_key_resolves_to_none_rather_than_guessing(slug):
    m = load(slug)
    assert m.shape_of("definitely.not.a.real.tensor.weight") is None


def test_collapse_only_touches_whole_numeric_segments():
    """`layers.10` must collapse, but `fc_hidden` and `e8p` must not — a digit inside a name
    is not an index, and collapsing it would merge unrelated tensors into one pattern."""
    assert collapse("a.0.b.12.c") == "a.{i}.b.{i}.c"
    assert collapse("mtp.fc_hidden.weight") == "mtp.fc_hidden.weight"
    assert collapse("x.Qidxs2") == "x.Qidxs2"
    assert collapse("layers.3") == "layers.{i}"


def test_the_two_models_have_genuinely_different_expert_layouts():
    """Why a second model earns its place. Flash-Next stacks 512 experts under
    ``layers.{i}.mlp.experts``; gemma-4 stacks 128 under ``layers.{i}.experts`` with no
    ``.mlp.``. A filter or detector tested against one shape only is tested against one
    naming convention only."""
    fn = load("qwen3_8_flash_next_glq_3bpw")
    g4 = load("gemma_4_26b_a4b_it")

    g4_stacked = [p for p in g4.patterns() if p.endswith("experts.gate_up_proj")]
    assert g4_stacked, "gemma-4 manifest has no stacked gate_up_proj"
    g4_shape = g4.patterns()[g4_stacked[0]]["shape"]
    assert len(g4_shape) == 3 and g4_shape[0] == 128
    assert ".mlp.experts" not in g4_stacked[0], "gemma-4 nests experts under .mlp. after all"

    fn_stacked = [p for p in fn.patterns() if p.endswith("mlp.experts.gate_up_proj")]
    assert fn_stacked, "Flash-Next manifest has no stacked mlp.experts.gate_up_proj"
    assert fn.patterns()[fn_stacked[0]]["shape"][0] == 512


def test_norm_spellings_differ_enough_to_defeat_a_name_rule():
    """The evidence behind "shape is the authority for norms, not spelling". If a future
    refactor reaches for a name rule again, this test says why that cannot work: of the 1-D
    tensors in Flash-Next's MTP head, only a minority are spelled ``*layernorm``."""
    m = load("qwen3_8_flash_next_glq_3bpw")
    one_d = [k for k in m.keys()
             if k.startswith("mtp.") and len(m.shape_of(k)) == 1]
    assert one_d, "no 1-D tensors in the MTP head — manifest looks wrong"
    layernorm_spelled = [k for k in one_d if k.rsplit(".", 2)[-2].endswith("layernorm")]
    assert len(layernorm_spelled) < len(one_d), (
        "every 1-D tensor is spelled *layernorm here, so this architecture would not have "
        "caught the bug — pick a model whose norms are spelled inconsistently")


def test_nbytes_matches_a_hand_computed_tensor():
    """The MTP head's stacked experts are the 4.69 GiB that `--cpu-offload-gb` cannot move,
    and that number drove the whole design — so it is worth asserting rather than quoting."""
    m = load("qwen3_8_flash_next_glq_3bpw")
    gup = m.nbytes_of("mtp.layers.0.mlp.experts.gate_up_proj")
    dwn = m.nbytes_of("mtp.layers.0.mlp.experts.down_proj")
    assert gup == 512 * 1280 * 2560 * 2          # 3.125 GiB
    assert dwn == 512 * 2560 * 640 * 2           # 1.5625 GiB
    assert (gup + dwn) / 2**30 == pytest.approx(4.6875, abs=1e-4)


def test_a_missing_manifest_names_how_to_record_it():
    with pytest.raises(FileNotFoundError, match="record.py"):
        load("no_such_model")


def test_keys_under_matches_segments_not_characters():
    m = load("qwen3_8_flash_next_glq_3bpw")
    assert m.keys_under("mtp.layers.0.mlp.experts")
    assert not m.keys_under("mtp.layers.0.mlp.expert")


def test_shapes_is_not_an_empty_shell():
    """A manifest that recorded nothing would make every consumer test vacuous."""
    for slug in RECORDED:
        m = load(slug)
        assert isinstance(m, ModelShapes)
        assert m.n_keys > 100, f"{slug}: only {m.n_keys} keys recorded"
        assert len(m.patterns()) > 10
