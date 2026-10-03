"""`kv_transfer_config` has to be reachable from the CLI, and has to stay JSON-serialisable.

vLLM's own recipe for Qwen3.8-Flash-Next configures CPU KV offload through
``--kv-transfer-config '{"kv_connector": "OffloadingConnector", ...}'``. `glq-bench` could not
express it: `build_llm_kwargs` accepts a fixed set of engine kwargs and this was not among
them, so the whole connector family was unreachable no matter what the hardware could do.
That is the same gap `--dtype` had — accepted by the callee, unreachable from the caller.

Two constraints shape the design, and both are pinned below.

**The record is JSONL.** `BenchRecord.to_json` runs `json.dumps(asdict(self), default=str)`
and `ServingMeta.llm_kwargs` is a verbatim copy of the engine kwargs. A `KVTransferConfig`
dataclass there would serialise to its `repr` via `default=str` — provenance you cannot parse
back. So the dict stays a dict everywhere the record can see it, and is converted to
`KVTransferConfig` only for the `LLM(**kw)` call itself.

**`build_llm_kwargs` must not need vLLM.** Several tests call it directly on a machine with no
vLLM installed, so the conversion cannot happen there.
"""
from __future__ import annotations

import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glq.bench.runtime import build_llm_kwargs, serving_command  # noqa: E402

# The shape vLLM's recipe configurator emits, scaled to a box with ~124 GB of RAM rather than
# the 512 GiB the published recipe assumes.
OFFLOAD = {
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"cpu_bytes_to_use": 68719476736, "blocks_per_chunk": 4},
}


# ---- it reaches the engine kwargs, as a dict -----------------------------------------------

def test_absent_by_default():
    """Every existing recorded run must be unaffected."""
    assert "kv_transfer_config" not in build_llm_kwargs("m")


def test_it_reaches_the_engine_kwargs():
    kw = build_llm_kwargs("m", kv_transfer_config=OFFLOAD)
    assert kw["kv_transfer_config"] == OFFLOAD


def test_it_stays_json_serialisable():
    """The record is JSONL. A KVTransferConfig object here would serialise to its repr via
    `default=str`, giving provenance nobody can parse back into a config."""
    kw = build_llm_kwargs("m", kv_transfer_config=OFFLOAD)
    assert json.loads(json.dumps(kw["kv_transfer_config"])) == OFFLOAD


def test_it_is_copied_not_aliased():
    """Mutating the caller's dict afterwards must not rewrite what the run recorded."""
    src = dict(OFFLOAD)
    kw = build_llm_kwargs("m", kv_transfer_config=src)
    src["kv_role"] = "mutated"
    assert kw["kv_transfer_config"]["kv_role"] == "kv_both"


def test_build_llm_kwargs_does_not_need_vllm(monkeypatch):
    """It is called directly by tests on machines with no vLLM, so the KVTransferConfig
    conversion must NOT live here."""
    monkeypatch.setitem(sys.modules, "vllm", None)          # any import would raise
    monkeypatch.setitem(sys.modules, "vllm.config", None)
    assert build_llm_kwargs("m", kv_transfer_config=OFFLOAD)["kv_transfer_config"] == OFFLOAD


# ---- provenance ---------------------------------------------------------------------------

def test_the_serving_command_can_be_copy_pasted():
    """A run that used CPU offload and a command string that omits it are not the same run.

    Split the command the way a shell would: the JSON contains spaces, so the real assertion
    is that it survives quoting as ONE argument and parses back. Substring matching would pass
    even if the quoting were broken and the shell split it into six arguments.
    """
    import shlex
    cmd = serving_command("m", build_llm_kwargs("m", kv_transfer_config=OFFLOAD))
    argv = shlex.split(cmd)
    assert "--kv-transfer-config" in argv, cmd
    payload = argv[argv.index("--kv-transfer-config") + 1]
    assert json.loads(payload) == OFFLOAD


def test_serving_command_omits_it_when_unset():
    assert "--kv-transfer-config" not in serving_command("m", build_llm_kwargs("m"))


# ---- load() converts, and records ---------------------------------------------------------

class _Stop(Exception):
    """Sentinel: LLM() reached, so every kwarg was already assembled."""


def _fake_vllm(monkeypatch, seen):
    """A vLLM stub whose KVTransferConfig records the kwargs it was built from."""
    class KVTransferConfig:
        def __init__(self, **kw):
            self.kw = kw
            seen["config_built_from"] = kw

    def _llm(**kw):
        seen["llm_kwargs"] = kw
        raise _Stop

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=_llm))
    monkeypatch.setitem(sys.modules, "vllm.config",
                        types.SimpleNamespace(KVTransferConfig=KVTransferConfig))
    return KVTransferConfig


def test_load_converts_the_dict_for_the_engine(monkeypatch):
    """vLLM's EngineArgs wants a KVTransferConfig, not a dict — `kv_transfer_config` is not
    even an explicit LLM() parameter, it rides **kwargs into EngineArgs."""
    import glq.bench.runtime as rt
    seen: dict = {}
    KVTransferConfig = _fake_vllm(monkeypatch, seen)
    with pytest.raises(_Stop):
        rt.load("m", kv_transfer_config=OFFLOAD)
    assert seen["config_built_from"] == OFFLOAD, "the dict was not converted"
    assert isinstance(seen["llm_kwargs"]["kv_transfer_config"], KVTransferConfig)


def test_load_leaves_the_recorded_kwargs_as_a_dict(monkeypatch):
    """The conversion is for the engine only. What the record keeps must still be a dict, or
    ServingMeta.llm_kwargs stops round-tripping through JSON."""
    import glq.bench.runtime as rt
    seen: dict = {}
    _fake_vllm(monkeypatch, seen)
    captured = {}
    orig = rt.serving_command
    monkeypatch.setattr(rt, "serving_command",
                        lambda model, kw: captured.setdefault("kw", kw) and "" or orig(model, kw))
    with pytest.raises(_Stop):
        rt.load("m", kv_transfer_config=OFFLOAD)
    # LLM() got the object; anything the record sees must still be the plain dict.
    assert captured.get("kw", {}).get("kv_transfer_config", OFFLOAD) == OFFLOAD


def test_serving_meta_has_the_field():
    """So a record states its own KV topology instead of leaving it only inside llm_kwargs —
    the mistake kv_cache_dtype already makes (declared on ServingMeta, never populated)."""
    from glq.bench.record import ServingMeta
    assert "kv_transfer_config" in ServingMeta.__dataclass_fields__


# ---- reachable from the CLI and the runner ------------------------------------------------

def test_cli_exposes_it_and_parses_json():
    from glq.bench.cli import build_parser
    args = build_parser().parse_args(
        ["run", "--model", "m", "--tasks", "aime_2026",
         "--kv-transfer-config", json.dumps(OFFLOAD)])
    assert args.kv_transfer_config == OFFLOAD, "the CLI must hand the runner a parsed dict"


def test_cli_rejects_malformed_json():
    """A typo in a long JSON blob should fail at argument parsing, not 4 minutes into an
    engine start."""
    from glq.bench.cli import build_parser
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["run", "--model", "m", "--tasks", "aime_2026",
             "--kv-transfer-config", "{not json"])


def test_runner_forwards_it():
    """The runner is the only path from CLI to engine; a parameter it drops is unreachable."""
    import inspect

    from glq.bench.runner import run
    assert "kv_transfer_config" in inspect.signature(run).parameters
