"""LEADERBOARD.md follows the HF collections: recommended checkpoints, then one section per family.

The structure is the curation on the Hub, read live — https://huggingface.co/xv0y5ncu/collections —
so publishing a checkpoint into a collection is what puts it on the leaderboard, and nothing
else is. That last part is a privacy property as much as a layout one: the flat table this
replaces printed every model with a record, which would have published private repo ids the
moment their records were pushed.

Rows inside a section are grouped by base model and run GPU by GPU, bigger card first: the
collection's GLQ checkpoints, then the bf16 base, then the other methods (NVFP4, ...) —
the order of the requested table.
"""
from __future__ import annotations

import os
import sys

from glq.bench import ModelMeta
from glq.bench import report
from glq.bench.hfcollections import Collection

sys.path.insert(0, os.path.dirname(__file__))
from test_bench_model_gpu_table import (                       # noqa: E402,F401 (fixtures)
    BF16, GLQ, L4, NVFP4, OTHER, PRO6000, _hw, aime, rows_of, sweep,
)

G26_T4 = ModelMeta(id="xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw",
                   base_model="google/gemma-4-26B-A4B-it", quant_method="glq", bpw=4.0)
G26_T3 = ModelMeta(id="xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-3bpw",
                   base_model="google/gemma-4-26B-A4B-it", quant_method="glq", bpw=3.0)
BF16_26 = ModelMeta(id="google/gemma-4-26B-A4B-it", quant_method="none")
BF16_12 = ModelMeta(id="google/gemma-4-12B-it", quant_method="none")
SMOL_BF16 = ModelMeta(id="HuggingFaceTB/SmolLM3-3B", quant_method="none")
#: Same base as GLQ, but in no public collection — an R&D build, or a private repo.
PRIVATE = ModelMeta(id="xv0y5ncu/gemma-4-31B-it-GLQ-private-experiment",
                    base_model="google/gemma-4-31B-it", quant_method="glq", bpw=3.0)
#: Benchmarked, but in no collection at all.
UNRELATED = ModelMeta(id="xv0y5ncu/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GLQ-trellis-3inst-3bpw",
                      base_model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
                      quant_method="glq", bpw=3.0)

E4B = "xv0y5ncu/gemma-4-E4B-it-GLQ-trellis-3inst-4bpw"          # in collections, no records
G12 = "xv0y5ncu/gemma-4-12B-it-GLQ-trellis-3inst-4bpw"
KOLIBRI = "xv0y5ncu/Kolibri-1-GLQ-trellis-3inst-3bpw"

START = Collection(
    title="Start here - recommended GLQ checkpoints",
    slug="xv0y5ncu/start-here-recommended-glq-checkpoints-6a69",
    description="One curated pick per model family",
    model_ids=[OTHER.id, G26_T4.id, G26_T3.id, GLQ.id, E4B])
GEMMA = Collection(
    title="Gemma-4 - GLQ quantized", slug="xv0y5ncu/gemma-4-glq-quantized-6a69",
    description="All public Gemma-4 GLQ checkpoints",
    # curator order puts the 31B BETWEEN the two 26B builds
    model_ids=[G26_T4.id, GLQ.id, G26_T3.id, G12, E4B])
KOL = Collection(
    title="Kolibri 1 - GLQ quantized", slug="xv0y5ncu/kolibri-1-glq-quantized-6add",
    description="Aleph Alpha Kolibri 1", model_ids=[KOLIBRI])
SMOL = Collection(
    title="SmolLM2 / SmolLM3 - GLQ quantized", slug="xv0y5ncu/smollm2-smollm3-glq-quantized-6a69",
    description="The small end", model_ids=[OTHER.id])


def records():
    return [
        aime(OTHER, PRO6000, 0.42), aime(SMOL_BF16, PRO6000, 0.47),
        aime(G26_T4, PRO6000, 0.86), aime(G26_T4, L4, 0.86), aime(G26_T3, PRO6000, 0.82),
        aime(BF16_26, PRO6000, 0.90),
        aime(GLQ, PRO6000, 0.90), aime(BF16, PRO6000, 0.88), aime(NVFP4, PRO6000, 0.88),
        aime(BF16_12, PRO6000, 0.85),                       # a base with no measured member
        aime(PRIVATE, PRO6000, 0.70), aime(UNRELATED, PRO6000, 0.95),
    ]


def board(recs=None, families=(GEMMA, KOL, SMOL)):
    return report.leaderboard(records() if recs is None else recs, start_here=START,
                              families=list(families), generated_utc="2026-10-28T00:00:00Z")


def section(md: str, heading: str) -> str:
    start = md.index(heading)
    nxt = md.find("\n## ", start + len(heading))
    return md[start:] if nxt == -1 else md[start:nxt]


def order(md: str, heading: str):
    return [(r[0], r[1]) for r in rows_of(section(md, heading))]


# --------------------------------------------------------------------------- structure

def test_recommended_first_then_each_family_alphabetically_and_nothing_else():
    md = board(families=(SMOL, GEMMA, KOL))                 # passed out of order
    heads = [ln for ln in md.splitlines() if ln.startswith("## ")]
    assert heads == ["## Recommended checkpoints", "## Gemma-4 - GLQ quantized",
                     "## Kolibri 1 - GLQ quantized", "## SmolLM2 / SmolLM3 - GLQ quantized"]
    assert "# Quality index" not in md                     # the %-of-bf16 index is gone
    assert md.startswith("# GLQ benchmark leaderboard")


def test_each_section_links_its_collection_and_shows_its_description():
    md = board()
    rec, gem = section(md, "## Recommended checkpoints"), section(md, "## Gemma-4")
    assert f"https://huggingface.co/collections/{START.slug}" in rec
    assert "One curated pick per model family" in rec
    assert f"https://huggingface.co/collections/{GEMMA.slug}" in gem
    assert "All public Gemma-4 GLQ checkpoints" in gem


# --------------------------------------------------------------------------- rows

def test_recommended_rows_follow_the_curators_order_gpu_by_gpu_with_references():
    assert order(board(), "## Recommended checkpoints") == [
        (OTHER.id, "RTX PRO 6000"), (SMOL_BF16.id, "RTX PRO 6000"),
        (G26_T4.id, "RTX PRO 6000"), (G26_T3.id, "RTX PRO 6000"), (BF16_26.id, "RTX PRO 6000"),
        (G26_T4.id, "L4"),
        (GLQ.id, "RTX PRO 6000"), (BF16.id, "RTX PRO 6000"), (NVFP4.id, "RTX PRO 6000"),
    ]


def test_two_picks_of_one_base_share_one_set_of_references_per_gpu():
    rows = order(board(), "## Recommended checkpoints")
    assert rows.count((BF16_26.id, "RTX PRO 6000")) == 1


def test_a_base_with_no_measured_member_gets_no_reference_rows():
    md = board()
    assert BF16_12.id not in md


def test_a_family_groups_by_base_in_first_appearance_order_curator_order_within():
    """The collection lists 26B-4bpw, 31B, 26B-3bpw; the 26B builds stay together."""
    assert order(board(), "## Gemma-4 - GLQ quantized") == [
        (G26_T4.id, "RTX PRO 6000"), (G26_T3.id, "RTX PRO 6000"), (BF16_26.id, "RTX PRO 6000"),
        (G26_T4.id, "L4"),
        (GLQ.id, "RTX PRO 6000"), (BF16.id, "RTX PRO 6000"), (NVFP4.id, "RTX PRO 6000"),
    ]


def test_unbenchmarked_members_are_listed_under_the_table_in_curator_order():
    gem = section(board(), "## Gemma-4 - GLQ quantized")
    line = next(ln for ln in gem.splitlines() if ln.startswith("Not yet benchmarked:"))
    assert line.index(G12) < line.index(E4B)
    assert f"(https://huggingface.co/{G12})" in line
    assert gem.index("| Model") < gem.index("Not yet benchmarked:")


def test_a_family_with_nothing_benchmarked_shows_only_the_list():
    kol = section(board(), "## Kolibri 1 - GLQ quantized")
    assert "| Model" not in kol
    assert f"[{KOLIBRI}](https://huggingface.co/{KOLIBRI})" in kol


# --------------------------------------------------------------------------- privacy

def test_models_outside_the_public_collections_never_appear():
    """A GLQ build of a listed base is NOT a reference row — references are the base itself
    and non-GLQ builds. Otherwise a private experiment would leak in beside the 31B."""
    md = board()
    assert PRIVATE.id not in md
    assert UNRELATED.id not in md


# --------------------------------------------------------------------------- legend

def test_one_legend_at_the_top_marks_only_when_used():
    md = board()
    assert md.count("— = not measured") == 1
    assert md.index("— = not measured") < md.index("## Recommended checkpoints")
    assert "‡" not in md and "§" not in md
    offloaded = records() + [aime(BF16_26, L4, 0.9, vram=19.1, offload=40.0)]
    md2 = board(offloaded)
    assert md2.count("‡ resident share only") == 1
    assert "§" not in md2


# --------------------------------------------------------------------------- fetching

LISTING = [
    {"slug": "xv0y5ncu/start-here-recommended-glq-checkpoints-6a69", "private": False,
     "title": "Start here - recommended GLQ checkpoints", "items": []},
    {"slug": "xv0y5ncu/smollm2-smollm3-glq-quantized-6a69", "private": False,
     "title": "SmolLM2 / SmolLM3 - GLQ quantized", "description": "small", "items": []},
    {"slug": "xv0y5ncu/gemma-4-glq-quantized-6a69", "private": False,
     "title": "Gemma-4 - GLQ quantized", "description": "gemma", "items": []},
    {"slug": "xv0y5ncu/bit-width-sweeps-quality-vs-footprint-6a69", "private": False,
     "title": "Bit-width sweeps - quality vs footprint", "items": []},
    {"slug": "xv0y5ncu/internal-glq-quantized-6a69", "private": True,
     "title": "Internal - GLQ quantized", "items": []},
]


def _item(i, **kw):
    return {"type": "model", "id": i, "private": False, "gated": False, **kw}


DETAIL = {
    "xv0y5ncu/start-here-recommended-glq-checkpoints": {
        "title": "Start here - recommended GLQ checkpoints", "description": "picks",
        "slug": "xv0y5ncu/start-here-recommended-glq-checkpoints-6a69",
        "items": [_item("a/one"), _item("a/two")]},
    "xv0y5ncu/gemma-4-glq-quantized-6a69": {
        "title": "Gemma-4 - GLQ quantized", "description": "gemma",
        "slug": "xv0y5ncu/gemma-4-glq-quantized-6a69",
        # the listing truncates to 4 items; the detail call carries them all
        "items": [_item("g/1"), _item("g/2"), _item("g/3"), _item("g/4"), _item("g/5"),
                  {"type": "paper", "id": "2406.11235"}, _item("g/priv", private=True),
                  _item("g/gated", gated=True)]},
    "xv0y5ncu/smollm2-smollm3-glq-quantized-6a69": {
        "title": "SmolLM2 / SmolLM3 - GLQ quantized", "description": "small",
        "slug": "xv0y5ncu/smollm2-smollm3-glq-quantized-6a69", "items": [_item("s/1")]},
}


def fake_fetch(url):
    if "/collections?" in url:
        return LISTING
    for slug, body in DETAIL.items():
        if url.endswith("/collections/" + slug):
            return body
    raise AssertionError(f"unexpected fetch: {url}")


def test_fetch_takes_public_family_collections_alphabetically_in_full():
    from glq.bench.hfcollections import fetch_collections
    start, families = fetch_collections(fetch=fake_fetch)
    assert start.title == "Start here - recommended GLQ checkpoints"
    assert start.model_ids == ["a/one", "a/two"]
    assert [f.title for f in families] == ["Gemma-4 - GLQ quantized",
                                           "SmolLM2 / SmolLM3 - GLQ quantized"]
    gem = families[0]
    assert gem.model_ids == ["g/1", "g/2", "g/3", "g/4", "g/5"]   # paper/private/gated dropped
    assert gem.url == "https://huggingface.co/collections/xv0y5ncu/gemma-4-glq-quantized-6a69"


def test_the_installer_and_the_leaderboard_share_one_item_filter():
    from glq.installer import discovery
    data = DETAIL["xv0y5ncu/gemma-4-glq-quantized-6a69"]
    assert discovery.public_model_ids(data) == ["g/1", "g/2", "g/3", "g/4", "g/5"]
    assert discovery.collection_repo_ids(
        fetch=lambda url: DETAIL["xv0y5ncu/start-here-recommended-glq-checkpoints"]) == \
        ["a/one", "a/two"]


def test_a_snapshot_round_trips():
    from glq.bench import hfcollections
    text = hfcollections.to_json(START, [GEMMA, SMOL])
    s, fams = hfcollections.from_json(text)
    assert s == START and fams == [GEMMA, SMOL]


# --------------------------------------------------------------------------- CLI

def _write(tmp_path, recs):
    from glq.bench.record import write_jsonl
    p = tmp_path / "r.jsonl"
    write_jsonl(recs, p)
    return p


def test_cli_leaderboard_renders_from_a_collections_snapshot(tmp_path):
    from glq.bench import cli, hfcollections
    snap = tmp_path / "c.json"
    snap.write_text(hfcollections.to_json(START, [GEMMA, KOL, SMOL]), encoding="utf-8")
    out = tmp_path / "LEADERBOARD.md"
    rc = cli.main(["leaderboard", "--records", str(_write(tmp_path, records())),
                   "--collections", str(snap), "--out", str(out), "--no-timestamp"])
    assert rc == 0
    assert "## Recommended checkpoints" in out.read_text(encoding="utf-8")


def test_cli_collections_writes_the_snapshot_leaderboard_reads(tmp_path, monkeypatch):
    from glq.bench import cli, hfcollections
    monkeypatch.setattr(hfcollections, "fetch_collections",
                        lambda fetch=None: (START, [GEMMA, SMOL]))
    snap = tmp_path / "c.json"
    assert cli.main(["collections", "--out", str(snap)]) == 0
    assert hfcollections.from_json(snap.read_text(encoding="utf-8")) == (START, [GEMMA, SMOL])


def test_cli_leaderboard_writes_nothing_when_the_collections_cannot_be_read(tmp_path,
                                                                           monkeypatch):
    """A failed fetch must not quietly produce a differently structured file."""
    from glq.bench import cli, hfcollections

    def boom(fetch=None):
        raise OSError("network unreachable")
    monkeypatch.setattr(hfcollections, "fetch_collections", boom)
    out = tmp_path / "LEADERBOARD.md"
    rc = cli.main(["leaderboard", "--records", str(_write(tmp_path, records())),
                   "--out", str(out)])
    assert rc == 2
    assert not out.exists()
