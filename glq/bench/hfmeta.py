"""Derive a ``ModelMeta`` (base model, quant method, bpw, arch, links, size) from a
model id or local dir.

Works for any checkpoint, not just GLQ: quant method is read from the HF
``quantization_config`` (glq / modelopt-NVFP4 / gptq / awq / compressed-tensors)
or GLQ's ``quantize_config.json``, falling back to ``none`` (bf16). Mirrors the
config-reading already done in ``glq/model_card.py`` (``build_card``), kept
separate so it has no Jinja/HF-upload dependencies.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .record import ModelMeta

HF_BASE = "https://huggingface.co"

# Map raw config quant-method strings to our normalized tags.
_QUANT_ALIASES = {
    "modelopt_fp4": "modelopt", "modelopt": "modelopt", "nvfp4": "modelopt",
    "fp4": "modelopt", "glq": "glq", "gptq": "gptq", "awq": "awq",
    "compressed-tensors": "compressed-tensors", "bitsandbytes": "bnb",
    "fp8": "fp8",
}


def _looks_like_local_dir(model: str) -> bool:
    return os.path.isdir(model)


def _hf_url(repo_id: str | None) -> str | None:
    if not repo_id or _looks_like_local_dir(repo_id) or "/" not in repo_id:
        return None
    return f"{HF_BASE}/{repo_id}"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _load_configs(model: str, hf_token: str | None) -> tuple[dict, dict, Path | None]:
    """Return (config.json, quantize_config.json, local_dir_or_None)."""
    if _looks_like_local_dir(model):
        d = Path(model)
        return _read_json(d / "config.json"), _read_json(d / "quantize_config.json"), d
    # Remote: download just the two small JSONs.
    cfg, qcfg = {}, {}
    try:
        from huggingface_hub import hf_hub_download
        cfg = _read_json(Path(hf_hub_download(model, "config.json", token=hf_token)))
        try:
            qcfg = _read_json(Path(hf_hub_download(model, "quantize_config.json", token=hf_token)))
        except Exception:  # noqa: BLE001 — most non-GLQ repos have no quantize_config.json
            qcfg = {}
    except Exception:  # noqa: BLE001 — offline / gated / no huggingface_hub
        pass
    return cfg, qcfg, None


def _detect_quant_method(cfg: dict, qcfg: dict, override: str | None) -> str:
    if override:
        return _QUANT_ALIASES.get(override.lower(), override.lower())
    qc = cfg.get("quantization_config") or {}
    raw = (qcfg.get("quant_method") or qc.get("quant_method")
           or qc.get("quant_algo") or "")
    raw = str(raw).strip().lower()
    if not raw:
        return "none"
    return _QUANT_ALIASES.get(raw, raw)


def _bpw_info(cfg: dict, qcfg: dict) -> tuple[float | None, int | None, int | None, bool]:
    qc = cfg.get("quantization_config") or {}
    avg = qcfg.get("bpw", qc.get("bpw"))
    avg = float(avg) if avg is not None else None
    layer_bpw = (qcfg.get("layer_bpw") or qc.get("layer_bpw") or {})
    if layer_bpw:
        vals = [int(v) for v in layer_bpw.values()]
        lo, hi = min(vals), max(vals)
    elif avg is not None:
        lo = hi = int(round(avg))
    else:
        lo = hi = None
    is_mixed = lo is not None and hi is not None and lo != hi
    return avg, lo, hi, is_mixed


def _disk_gb(local_dir: Path | None, model: str, hf_token: str | None) -> float | None:
    if local_dir is not None:
        total = sum(p.stat().st_size for p in local_dir.glob("*.safetensors"))
        return round(total / 2**30, 2) if total else None
    try:
        from huggingface_hub import model_info
        mi = model_info(model, files_metadata=True, token=hf_token)
        total = sum((s.size or 0) for s in mi.siblings
                    if s.rfilename.endswith(".safetensors"))
        return round(total / 2**30, 2) if total else None
    except Exception:  # noqa: BLE001
        return None


def _first(v):
    """A base_model value as one repo id: cards store it as a string or a list."""
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    return str(v).strip().strip("'\"") if v else None


def _card_base_model(model: str, local_dir: Path | None, hf_token: str | None) -> str | None:
    """``base_model`` from the checkpoint's model card, or None.

    Needed because a quantized checkpoint's config often does not name its parent —
    nvidia/Gemma-4-31B-IT-NVFP4's config.json has no ``_name_or_path`` — while its card's
    frontmatter does. A local dir reads its own README.md (the card GLQ writes at quantize
    time); a repo id reads the Hub's parsed card metadata.
    """
    if local_dir is not None:
        readme = local_dir / "README.md"
        if not readme.is_file():
            return None
        text = readme.read_text(encoding="utf-8", errors="replace")
        if not text.startswith("---"):
            return None
        front = text.split("---", 2)[1] if text.count("---") >= 2 else ""
        lines = front.splitlines()
        for i, line in enumerate(lines):
            if not line.startswith("base_model:"):
                continue
            rest = line[len("base_model:"):].strip()
            if rest.startswith("["):
                return _first([s for s in rest.strip("[]").split(",") if s.strip()])
            if rest:
                return _first(rest)
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            return _first(nxt[1:]) if nxt.startswith("-") else None
        return None
    try:
        from huggingface_hub import model_info
        data = model_info(model, token=hf_token).card_data
        return _first((data.to_dict() if data is not None else {}).get("base_model"))
    except Exception:  # noqa: BLE001 — offline / gated / no card: no base, not an error
        return None


def model_meta(model: str, *, quant_override: str | None = None,
               hf_token: str | None = None) -> ModelMeta:
    """Build a ``ModelMeta`` for an HF repo id or a local checkpoint dir.

    ``quant_override`` (e.g. the runtime ``--quant`` flag) wins over config
    detection — useful when serving a bf16 model under a quantization flag or
    vice-versa.
    """
    cfg, qcfg, local_dir = _load_configs(model, hf_token)

    base = cfg.get("_name_or_path") or cfg.get("base_model")
    if isinstance(base, dict):                # some cards store base_model as a list/obj
        base = base.get("base_model") or None
    if isinstance(base, list):
        base = base[0] if base else None
    # A local dir's _name_or_path may point back at itself; only keep a real repo id.
    if base and _looks_like_local_dir(str(base)):
        base = None

    quant_method = _detect_quant_method(cfg, qcfg, quant_override)
    # No base in the config: a QUANTIZED checkpoint's card names the model it was quantized
    # from, so use that. An unquantized one is skipped on purpose — google/gemma-4-31B-it's
    # card names its pretraining parent (google/gemma-4-31B), and taking it would pull the
    # bf16 baseline out of its own family. Unquantized means it IS the base.
    if not base and quant_method != "none":
        base = _card_base_model(model, local_dir, hf_token)
        if base and _looks_like_local_dir(base):
            base = None
    avg, lo, hi, is_mixed = _bpw_info(cfg, qcfg)
    arch = (cfg.get("architectures") or [None])[0]

    return ModelMeta(
        id=model,
        hf_url=_hf_url(model),
        base_model=base,
        base_hf_url=_hf_url(base),
        quant_method=quant_method,
        bpw=avg if quant_method != "none" else None,
        bpw_min=lo if quant_method != "none" else None,
        bpw_max=hi if quant_method != "none" else None,
        is_mixed=is_mixed if quant_method != "none" else None,
        weights_disk_gb=_disk_gb(local_dir, model, hf_token),
        architecture=arch,
    )
