"""The xv0y5ncu Hugging Face collections, as the structure of LEADERBOARD.md.

The leaderboard follows the curation on the Hub (https://huggingface.co/xv0y5ncu/collections):
the "Start here" collection first, then one section per model-family collection. Reading it
live means curating a collection is what changes the leaderboard — no list here to keep in sync.

A family collection is a public one whose slug contains ``-glq-quantized``: every family is
titled ``"<Family> - GLQ quantized"`` (Gemma-4, SmolLM2 / SmolLM3, Kolibri 1). Collections that
cut across families — the bit-width sweeps — and private ones are not sections.

The fetcher, the start-here slug and the item filter (public models only) are the installer's
own, so the leaderboard and ``glq-setup --list`` can never disagree about what is published.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from glq.installer.discovery import COLLECTION_SLUG, _API, _fetch_json, public_model_ids

OWNER = COLLECTION_SLUG.split("/", 1)[0]
#: Marks a model-family collection; see the module docstring.
FAMILY_MARK = "-glq-quantized"


@dataclass
class Collection:
    title: str
    slug: str
    description: str = ""
    model_ids: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://huggingface.co/collections/{self.slug}"


def _collection(data: dict, fallback_slug: str) -> Collection:
    return Collection(title=data.get("title") or fallback_slug,
                      slug=data.get("slug") or fallback_slug,
                      description=(data.get("description") or "").strip(),
                      model_ids=public_model_ids(data))


def fetch_collections(fetch=_fetch_json) -> tuple[Collection, list[Collection]]:
    """``(start_here, families)``, families ordered by title.

    The owner listing returns at most four items per collection, so each family is fetched
    again on its own for its full, ordered item list. Private collections are dropped
    explicitly: an authenticated listing includes them, and the leaderboard is public.
    """
    start = _collection(fetch(f"{_API}/collections/{COLLECTION_SLUG}"), COLLECTION_SLUG)
    listing = fetch(f"{_API}/collections?owner={OWNER}&limit=100")
    families = []
    for entry in listing:
        slug = entry.get("slug", "")
        if entry.get("private") or FAMILY_MARK not in slug:
            continue
        families.append(_collection(fetch(f"{_API}/collections/{slug}"), slug))
    families.sort(key=lambda c: c.title.lower())
    return start, families


def to_json(start: Collection, families: list[Collection]) -> str:
    """A snapshot ``glq-bench leaderboard --collections`` can read back."""
    return json.dumps({"start_here": asdict(start), "families": [asdict(f) for f in families]},
                      indent=2) + "\n"


def from_json(text: str) -> tuple[Collection, list[Collection]]:
    doc = json.loads(text)
    return Collection(**doc["start_here"]), [Collection(**f) for f in doc["families"]]
