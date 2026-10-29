"""Let transformers' native NemotronH Omni config answer the field name vLLM reads.

transformers 5.18 ported ``NemotronH_Omni_Reasoning_V3_Config`` natively and renamed
``vit_hidden_size`` to ``vision_hidden_size``. The class records the rename itself::

    _legacy_field_names = {..., "vit_hidden_size": "vision_hidden_size"}

but applies it only when a config is LOADED (``__post_init__`` pops the legacy kwarg). vLLM
0.32's ``nano_nemotron_vl.py`` was written against NVIDIA's remote-code config and READS the
old name while building the vision projector::

    vit_hidden_size = config.vit_hidden_size
    AttributeError: 'NemotronH_Omni_Reasoning_V3_Config' object has no attribute
                    'vit_hidden_size'. Did you mean: 'vision_hidden_size'?

So every NemotronH Omni checkpoint served without remote code -- GLQ's Super-VL builds and
NVIDIA's own BF16 one alike -- stops at engine init. Measured on vLLM 0.32.0 with
transformers 5.18.0: of every top-level ``config.X`` that file reads, this is the only one the
native class does not answer. Its other legacy read, ``getattr(config, "sound_config", None)``,
has a default, so it is left alone on purpose: aliasing it to ``audio_config`` would switch on
vLLM's audio path, with a differently shaped config object, for the audio-capable Omni models.

The alias goes through ``attribute_map``, the mechanism ``PreTrainedConfig.__getattribute__``
and ``__setattr__`` already consult. It is a read alias only: nothing is added to
``to_dict()``, so a config.json written through this class is unchanged.
``tests/test_nemotron_omni_config.py`` pins that, and that ``register()`` installs it.

The right long-term fix is upstream: ``nano_nemotron_vl`` should read ``vision_hidden_size``.
"""

from __future__ import annotations

import inspect
import warnings

_LEGACY = "vit_hidden_size"
_FIELD = "vision_hidden_size"
_MISSING = object()


def _declares_field(cls, name) -> bool:
    if name in getattr(cls, "__dataclass_fields__", {}):
        return True
    return any(name in inspect.get_annotations(klass) for klass in cls.__mro__)


def _already_answers(cls) -> bool:
    """True when ``cls().vit_hidden_size`` already resolves: aliased (a second ``install()``,
    or someone else's alias), or a real field, property or class attribute -- which is what a
    transformers release that fixed this itself would look like."""
    if _LEGACY in (getattr(cls, "attribute_map", None) or {}):
        return True
    return (_declares_field(cls, _LEGACY)
            or inspect.getattr_static(cls, _LEGACY, _MISSING) is not _MISSING)


def install(cls=None) -> bool:
    """Alias ``vit_hidden_size`` -> ``vision_hidden_size`` on the native config class.

    Returns True when the class answers the legacy name afterwards. No-op -- deliberately
    silent -- when transformers has no native class (< 5.18): that serves through NVIDIA's
    remote-code config, which carries ``vit_hidden_size``, and ``register()`` calls this in
    every process regardless of which model is being served.

    If the native class exists but no longer declares the field, or no longer records the
    rename, this warns instead of aliasing into a name it cannot vouch for.
    """
    if cls is None:
        try:
            from transformers.models.nemotron_h_omni.configuration_nemotron_h_omni import (
                NemotronH_Omni_Reasoning_V3_Config as cls)
        except ImportError:
            return False

    if _already_answers(cls):
        return True

    renamed = (getattr(cls, "_legacy_field_names", None) or {}).get(_LEGACY) == _FIELD
    if not (renamed and _declares_field(cls, _FIELD)):
        warnings.warn(
            f"GLQ: transformers' {cls.__name__} no longer declares `{_FIELD}` as the rename of "
            f"`{_LEGACY}`, so glq_vllm cannot alias it. vLLM's nano_nemotron_vl reads "
            f"`config.{_LEGACY}`: expect `AttributeError: ... no attribute '{_LEGACY}'` at "
            f"engine start for NemotronH Omni (Super-VL) checkpoints served without remote "
            f"code. This transformers version needs a new hook in "
            f"glq_vllm/_nemotron_omni_config.py.",
            RuntimeWarning, stacklevel=2)
        return False

    # Assign a new dict on the class itself: mutating an inherited one would alias the name
    # on every PreTrainedConfig in the process.
    cls.attribute_map = {**(getattr(cls, "attribute_map", None) or {}), _LEGACY: _FIELD}
    return True
