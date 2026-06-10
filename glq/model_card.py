"""Generate a GLQ HuggingFace model card.

Injects GLQ-specific sections (install, vLLM/Transformers usage, pi-code/opencode
coding-agent configs, E8 KV cache, benchmarks) into the *original* base-model card:
the base README's YAML frontmatter (license, language, pipeline_tag, tags) is
inherited, its body is appended verbatim under a collapsible section, and a
derivative-work notice + license are added. The result respects the base model's
license (it is carried over, never relabeled).

Usage (library):
    from glq.model_card import build_card
    build_card("/path/to/quant_out", "google/gemma-4-e4b-it",
               repo_id="xv0y5ncu/Gemma-4-E4B-it-GLQ-4bpw",
               benchmarks=[{"task": "MMLU-Pro", "metric": "exact_match",
                            "n": 247, "value": "65.2%"}])

Usage (CLI):
    python -m glq.model_card --dir ./out --base google/gemma-4-e4b-it \
        [--repo-id xv0y5ncu/...] [--benchmarks bench.json]
"""
from __future__ import annotations

import json
import os
from pathlib import Path

GLQ_REPO = "https://github.com/cnygaard/glq"
_TEMPLATE = "model_card.md.j2"

# Frontmatter keys carried over verbatim from the base model's card.
_INHERIT_KEYS = ("license", "license_name", "license_link", "language",
                 "pipeline_tag", "datasets", "tags")


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Split a README string into (frontmatter_dict, body). Robust to no/blank
    frontmatter. Uses PyYAML if available, else returns an empty dict + full body."""
    s = text.lstrip("﻿")
    if not s.startswith("---"):
        return {}, text
    rest = s[3:]
    end = rest.find("\n---")
    if end == -1:
        return {}, text
    fm_raw = rest[:end]
    body = rest[end + 4:].lstrip("\n")
    try:
        import yaml
        fm = yaml.safe_load(fm_raw)
    except Exception:  # noqa: BLE001 — missing PyYAML or malformed YAML
        return {}, text
    return (fm if isinstance(fm, dict) else {}), body


def _fetch_original_readme(base_model_id: str, token: str | None) -> tuple[dict, str]:
    """Download the base model's README.md and split it. Returns ({}, "") if absent."""
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(base_model_id, "README.md", repo_type="model", token=token)
        return _split_frontmatter(Path(p).read_text(encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 — gated/no-README/offline
        return {}, ""


def _merged_frontmatter(orig_fm: dict, base_model_id: str) -> dict:
    fm: dict = {}
    for k in _INHERIT_KEYS:
        if k in orig_fm and orig_fm[k] is not None:
            fm[k] = orig_fm[k]
    fm.setdefault("license", "other")
    fm.setdefault("library_name", "transformers")
    fm.setdefault("pipeline_tag", "text-generation")
    fm["base_model"] = base_model_id
    fm["base_model_relation"] = "quantized"
    tags = list(fm.get("tags") or [])
    for t in ("glq", "quantization", "e8-lattice"):
        if t not in tags:
            tags.append(t)
    fm["tags"] = tags
    return fm


def _dump_frontmatter(fm: dict) -> str:
    try:
        import yaml
        body = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True,
                              default_flow_style=False).strip()
    except Exception:  # noqa: BLE001 — no PyYAML: emit a minimal hand-rolled block
        lines = []
        for k, v in fm.items():
            if isinstance(v, list):
                lines.append(f"{k}:")
                lines.extend(f"- {x}" for x in v)
            else:
                lines.append(f"{k}: {v}")
        body = "\n".join(lines)
    return f"---\n{body}\n---\n"


def _bpw_label(avg_bpw: float, is_mixed: bool, lo, hi) -> str:
    if is_mixed:
        return f"{avg_bpw:.1f}bpw (mixed {lo}–{hi})"
    return f"{int(round(avg_bpw))}bpw"


def _stages_blurb(avg_bpw: float) -> str:
    if avg_bpw <= 2.5:
        return "At ~2 bpw a single E8 codebook is used."
    if avg_bpw <= 4.5:
        return ("At 3–4 bpw a two-stage residual vector quantization "
                "(primary + residual codebook) is used.")
    return ("At 5–8 bpw an N-stage residual vector quantization is used "
            "(multiple stacked E8 residual codebooks).")


def _pi_config(repo_id: str) -> str:
    return json.dumps({
        "providers": {
            "glq": {
                "baseUrl": "http://localhost:8000/v1",
                "api": "openai-completions",
                "apiKey": "glq",
                "models": [{"id": repo_id}],
            }
        }
    }, indent=2)


def _opencode_config(repo_id: str) -> str:
    return json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "glq": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "GLQ (local vLLM)",
                "options": {"baseURL": "http://localhost:8000/v1", "apiKey": "glq"},
                "models": {repo_id: {"name": repo_id.split("/")[-1]}},
            }
        }
    }, indent=2)


def build_card(out_dir, base_model_id: str, *, repo_id: str | None = None,
               benchmarks=None, hf_token: str | None = None,
               write: bool = True) -> str:
    """Render the GLQ model card for a quantized output dir and (optionally) write
    it to ``out_dir/README.md``. Returns the rendered markdown."""
    out = Path(out_dir)
    qcfg = json.loads((out / "quantize_config.json").read_text()) \
        if (out / "quantize_config.json").exists() else {}
    cfg = json.loads((out / "config.json").read_text()) \
        if (out / "config.json").exists() else {}

    repo_id = repo_id or f"xv0y5ncu/{out.name}"
    avg_bpw = float(qcfg.get("bpw", cfg.get("quantization_config", {}).get("bpw", 4)))

    # Mixed-precision detection from the per-layer bpw map (config.json).
    layer_bpw = (cfg.get("quantization_config") or {}).get("layer_bpw") or {}
    if layer_bpw:
        vals = list(layer_bpw.values())
        lo, hi = min(vals), max(vals)
    else:
        lo = hi = int(round(avg_bpw))
    is_mixed = lo != hi

    arch = (cfg.get("architectures") or [""])[0]
    multimodal = ("ConditionalGeneration" in arch
                  or "vision_config" in cfg or "audio_config" in cfg)
    trust_remote_code = bool(cfg.get("auto_map"))

    orig_fm, orig_body = _fetch_original_readme(base_model_id, hf_token)
    fm = _merged_frontmatter(orig_fm, base_model_id)

    if isinstance(benchmarks, dict):  # accept {"MMLU-Pro": "65.2%"} convenience form
        benchmarks = [{"task": k, "metric": "score", "value": v}
                      for k, v in benchmarks.items()]

    ctx = {
        "repo_id": repo_id,
        "title": base_model_id.rstrip("/").split("/")[-1],
        "base_model": base_model_id,
        "avg_bpw": avg_bpw,
        "bpw_label": _bpw_label(avg_bpw, is_mixed, lo, hi),
        "is_mixed": is_mixed, "min_bpw": lo, "max_bpw": hi,
        "avg_sqnr_db": float(qcfg.get("avg_sqnr_db", 0.0)),
        "n_quantized_layers": qcfg.get("n_quantized_layers"),
        "nsamples": qcfg.get("nsamples", 128),
        "seqlen": qcfg.get("seqlen", 2048),
        "trust_remote_code": trust_remote_code,
        "multimodal": multimodal,
        "auto_cls": "AutoModelForImageTextToText" if multimodal else "AutoModelForCausalLM",
        "benchmarks": benchmarks,
        "stages_blurb": _stages_blurb(avg_bpw),
        "pi_config": _pi_config(repo_id),
        "opencode_config": _opencode_config(repo_id),
        "orig_body": orig_body,
        "license": fm.get("license"),
        "glq_repo": GLQ_REPO,
    }

    from jinja2 import Environment, FileSystemLoader
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
        autoescape=False, keep_trailing_newline=True,
        trim_blocks=True, lstrip_blocks=True)
    body = env.get_template(_TEMPLATE).render(**ctx)
    card = _dump_frontmatter(fm) + "\n" + body

    if write:
        (out / "README.md").write_text(card, encoding="utf-8")
    return card


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Generate a GLQ model card (README.md).")
    ap.add_argument("--dir", required=True, help="quantized output dir (has quantize_config.json)")
    ap.add_argument("--base", required=True, help="base model HF id (for license + original card)")
    ap.add_argument("--repo-id", default=None, help="target HF repo id (default xv0y5ncu/<dirname>)")
    ap.add_argument("--benchmarks", default=None, help="JSON file: list of {task,metric,n,value}")
    args = ap.parse_args()
    bench = json.loads(Path(args.benchmarks).read_text()) if args.benchmarks else None
    card = build_card(args.dir, args.base, repo_id=args.repo_id, benchmarks=bench,
                      hf_token=os.environ.get("HF_TOKEN"))
    print(f"wrote {Path(args.dir) / 'README.md'} ({len(card)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
