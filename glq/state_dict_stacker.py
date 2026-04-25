"""State-dict key remapping helpers for loading legacy GLQ checkpoints into
the native HF transformers `nemotron_h` integration.

Our published GLQ checkpoints were quantized through NVIDIA's
trust-remote-code modeling file, which calls the body submodule `backbone`.
The native transformers integration calls it `model`. Saved keys therefore
look like

    backbone.layers.X.mixer.experts.{i}.up_proj.Qidxs

while the native model expects

    model.layers.X.mixer.experts.{i}.up_proj.Qidxs

(once we've replaced the stacked-tensor `NemotronHExperts` with our
per-expert `E8RHTFusedExperts` — see `glq/fused_experts.py`).

Transformers' modern loader (`core_model_loading.convert_and_load_state_dict_in_model`)
applies a class-level regex mapping called `_checkpoint_conversion_mapping`
during `_load_pretrained_model`. We exploit that hook by injecting the
prefix swap on the model class at load-prep time.

For the mixed-bpw case (some experts at 5bpw, others at 4bpw within the
same MoE layer's expert ensemble — verified via the safetensors index), no
explicit padding is needed: `E8RHTLinear` initializes `Qidxs3` and
`inv_resid_scale2` buffers to zeros by default, and 4bpw experts just leave
them at zero — the inference path in `glq_fused_linear_cuda` and the Triton
fallback both treat `inv_resid_scale == 0` as "stage absent" per expert.
"""
from __future__ import annotations

import logging

import torch.nn as nn

_LOG = logging.getLogger(__name__)

# Regex → replacement for the trust-remote-code → native prefix swap.
# Keys that don't match (`lm_head.weight`, etc.) are passed through untouched.
NEMOTRON_H_PREFIX_RENAMES = {
    r"^backbone\.": "model.",
}


def install_nemotron_h_state_dict_renames(model: nn.Module) -> bool:
    """Inject `_checkpoint_conversion_mapping` on the model's class so the HF
    loader rewrites legacy `backbone.*` keys to `model.*` while populating
    parameters.

    Idempotent. Returns True if a rename was added (or already present),
    False if the model isn't a NemotronH variant.
    """
    cfg = getattr(model, "config", None)
    if cfg is None or getattr(cfg, "model_type", None) != "nemotron_h":
        return False

    cls = type(model)
    existing = dict(getattr(cls, "_checkpoint_conversion_mapping", {}) or {})
    if existing.get(r"^backbone\.") == "model.":
        return True
    existing.update(NEMOTRON_H_PREFIX_RENAMES)
    # Class-level set is intentional — the rename is harmless on freshly
    # quantized native-prefix checkpoints (regex won't match anything) and
    # all NemotronH instances should treat legacy keys consistently.
    cls._checkpoint_conversion_mapping = existing
    _LOG.debug(
        "glq: registered NemotronH state-dict prefix rename "
        "(backbone. -> model.) on %s", cls.__name__,
    )
    return True
