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
