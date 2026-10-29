"""NemotronH Omni (Super-VL) on vLLM with transformers' native config: the `vit_hidden_size` alias.

transformers 5.18 ported `NemotronH_Omni_Reasoning_V3_Config` natively and renamed
`vit_hidden_size` to `vision_hidden_size`. The class declares the rename itself
(`_legacy_field_names`), but applies it only when LOADING a config. vLLM 0.32's
`nano_nemotron_vl.py` was written against NVIDIA's remote-code config and READS the old name:

    vit_hidden_size = config.vit_hidden_size
    AttributeError: 'NemotronH_Omni_Reasoning_V3_Config' object has no attribute
                    'vit_hidden_size'. Did you mean: 'vision_hidden_size'?

So the published GLQ Super-VL builds -- and NVIDIA's own BF16 checkpoint -- stop at engine
init without remote code. `glq_vllm/_nemotron_omni_config.py` adds one read alias. These tests
pin that the alias answers, that it changes nothing transformers writes or loads (GLQ's
quantize writes config.json through this class), and that the plugin entry point installs it.
"""
from __future__ import annotations

import sys
import warnings

import pytest

pytest.importorskip("transformers", minversion="5.18.0",
                    reason="nemotron_h_omni is native from transformers 5.18")
try:
    from transformers.models.nemotron_h_omni.configuration_nemotron_h_omni import (
        NemotronH_Omni_Reasoning_V3_Config as OmniConfig)
except ImportError as exc:                                            # pragma: no cover
    pytest.skip(f"transformers has no nemotron_h_omni: {exc}", allow_module_level=True)

from transformers import PreTrainedConfig  # noqa: E402

from glq_vllm import _nemotron_omni_config as omni  # noqa: E402

LEGACY, FIELD = "vit_hidden_size", "vision_hidden_size"
_MODULE = "transformers.models.nemotron_h_omni.configuration_nemotron_h_omni"


@pytest.fixture(autouse=True)
def _native_class():
    """Each test starts from the class as transformers ships it, and leaves it as found.

    `attribute_map` is class-level state shared by every config in the process, and the
    import above already ran `register()` -- glq_vllm calls it at import time for manual use
    -- so where vLLM is installed the alias is in place before any test starts."""
    missing = object()
    own = OmniConfig.__dict__.get("attribute_map", missing)
    OmniConfig.attribute_map = {k: v for k, v in OmniConfig.attribute_map.items() if k != LEGACY}
    yield
    if own is missing:
        if "attribute_map" in OmniConfig.__dict__:
            delattr(OmniConfig, "attribute_map")
    else:
        OmniConfig.attribute_map = own


def _nvidia_style(**extra):
    """A config as NVIDIA's released checkpoints store it: under the legacy field name."""
    return OmniConfig.from_dict({LEGACY: 1152, **extra})


# ---- the gap -------------------------------------------------------------------------

def test_the_native_config_does_not_answer_the_name_vllm_reads():
    """The bug, at unit level. If this starts failing, transformers answers the legacy name
    itself and the alias in glq_vllm can go."""
    cfg = _nvidia_style()
    assert cfg.vision_hidden_size == 1152, "the legacy value was not carried across on load"
    with pytest.raises(AttributeError, match=LEGACY):
        getattr(cfg, LEGACY)


# ---- the alias ----------------------------------------------------------------------

def test_after_install_the_legacy_name_reads_the_new_field():
    built_before = _nvidia_style()
    assert omni.install() is True
    built_after = _nvidia_style()
    for cfg in (built_before, built_after):     # class-level: covers configs vLLM already built
        assert getattr(cfg, LEGACY) == cfg.vision_hidden_size == 1152


def test_the_alias_changes_nothing_transformers_writes():
    """GLQ's quantize writes config.json through this class, and checkpoints must stay
    byte-identical: an alias that leaked into to_dict() would add a key to every one."""
    before = _nvidia_style()
    dict_before, json_before = before.to_dict(), before.to_json_string()
    omni.install()
    assert before.to_dict() == dict_before
    after = _nvidia_style()
    assert after.to_json_string() == json_before
    assert LEGACY not in after.to_dict()


def test_the_alias_changes_nothing_transformers_loads():
    """transformers documents that a config carrying both names keeps the new one. The alias
    also routes WRITES through `__setattr__`, so it must not let the legacy value win."""
    both = {LEGACY: 1152, FIELD: 2048}
    expected = OmniConfig.from_dict(dict(both)).vision_hidden_size
    assert expected == 2048
    omni.install()
    assert OmniConfig.from_dict(dict(both)).vision_hidden_size == expected
    assert OmniConfig.from_dict({FIELD: 2048}).vision_hidden_size == 2048
    assert OmniConfig.from_dict({}).vision_hidden_size == OmniConfig.vision_hidden_size


def test_install_is_idempotent_and_keeps_existing_aliases():
    OmniConfig.attribute_map = {"some_alias": "projector_hidden_size"}
    assert omni.install() is True
    assert omni.install() is True
    assert OmniConfig.attribute_map == {"some_alias": "projector_hidden_size", LEGACY: FIELD}


def test_install_touches_only_the_omni_class():
    """Assigning on the subclass must not mutate the dict it may inherit."""
    from transformers import NemotronHConfig
    omni.install()
    assert LEGACY not in PreTrainedConfig.attribute_map
    assert LEGACY not in NemotronHConfig.attribute_map


# ---- the edges ----------------------------------------------------------------------

def test_install_is_silent_without_the_native_class(monkeypatch):
    """transformers < 5.18 serves this model through the remote-code config, which carries
    `vit_hidden_size` already -- and `register()` runs for every model."""
    monkeypatch.setitem(sys.modules, _MODULE, None)        # makes the import raise
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert omni.install() is False


class _Renamed(PreTrainedConfig):
    """A future class that dropped the field the alias would point at."""
    attribute_map = {}
    _legacy_field_names = {}


class _Unmapped(PreTrainedConfig):
    """Still has the field, but no longer declares the rename."""
    attribute_map = {}
    _legacy_field_names = {}
    vision_hidden_size: int = 1280


@pytest.mark.parametrize("cls", [_Renamed, _Unmapped], ids=["field-gone", "rename-undeclared"])
def test_install_warns_instead_of_aliasing_into_the_unknown(cls):
    """A silent no-op is how a rename went unnoticed in `_qwen4exp_ple`; say what breaks."""
    with pytest.warns(RuntimeWarning, match=LEGACY):
        assert omni.install(cls) is False
    assert cls.attribute_map == {}


def test_install_leaves_a_class_that_already_has_the_field_alone():
    class _Fixed(PreTrainedConfig):
        attribute_map = {}
        _legacy_field_names = {LEGACY: FIELD}
        vision_hidden_size: int = 1280

        @property
        def vit_hidden_size(self):
            return self.vision_hidden_size

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert omni.install(_Fixed) is True
    assert _Fixed.attribute_map == {}


# ---- the mechanism: the plugin entry point ------------------------------------------

def test_the_vllm_plugin_entry_point_installs_the_alias():
    """`register()` is the `vllm.general_plugins` entry point, so it runs in the EngineCore
    child that builds the model. Plain `import vllm` succeeds on boxes without a real vLLM,
    hence the deeper module."""
    pytest.importorskip("vllm.model_executor.layers.quantization", reason="vLLM not installed")
    import glq_vllm
    assert LEGACY not in OmniConfig.attribute_map
    glq_vllm.register()
    assert OmniConfig.attribute_map.get(LEGACY) == FIELD
