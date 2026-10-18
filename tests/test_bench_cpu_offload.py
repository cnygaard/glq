"""`--cpu-offload-gb` has to be reachable from the CLI, and has to reach the record.

WEIGHT offload, which is a different mechanism from `--kv-transfer-config` (KV offload).
Keeping them distinct is the whole reason this flag exists: it maps PARAMETERS into host
memory so a model larger than the card can be measured at all.

The motivating case is a baseline. Kolibri-1 bf16 is 145.5 GiB against a 96 GiB card, so
without this the GLQ 3 bpw perplexity (23.672) has nothing to be compared against and cannot
be interpreted either way. Proven on a g7e: 70.55 GiB resident + 75.01 GiB offloaded =
145.56 GiB, matching the source.

Two properties the measurement depends on:

* **Absent unless asked.** Every existing record was produced without the key, and a
  silently-added `cpu_offload_gb: 0` would change the `llm_kwargs` blob that comparisons
  key on.
* **Visible in the record.** `load_gpu_mem_gib` means something different under offload — it
  is the RESIDENT share, not the model's footprint — so a reader who could not see the
  offload would read 70.55 GiB as the size of a 145.5 GiB model.
"""
from __future__ import annotations

from glq.bench.record import ServingMeta
from glq.bench.runtime import build_llm_kwargs, serving_command


def test_absent_unless_requested():
    """Existing runs must keep byte-identical recorded kwargs."""
    assert "cpu_offload_gb" not in build_llm_kwargs("m")
    assert "cpu_offload_gb" not in build_llm_kwargs("m", cpu_offload_gb=0)


def test_present_when_requested():
    assert build_llm_kwargs("m", cpu_offload_gb=75)["cpu_offload_gb"] == 75.0


def test_it_is_not_the_kv_offload_key():
    """Separate mechanisms; setting one must not imply the other."""
    kw = build_llm_kwargs("m", cpu_offload_gb=75)
    assert "kv_transfer_config" not in kw
    kv = build_llm_kwargs("m", kv_transfer_config={"kv_connector": "OffloadingConnector"})
    assert "cpu_offload_gb" not in kv


def test_serving_command_shows_the_offload():
    """A run with 75 GiB of weights in host memory is not the same run, and the command
    string is what a reader of the record actually has."""
    kw = build_llm_kwargs("m", cpu_offload_gb=75)
    assert "--cpu-offload-gb 75.0" in serving_command("m", kw)
    assert "--cpu-offload-gb" not in serving_command("m", build_llm_kwargs("m"))


def test_serving_meta_carries_it():
    """Promoted to its own field rather than left in llm_kwargs: `kv_cache_dtype` was
    declared on ServingMeta and never passed, so every record reported None for it
    regardless of what was served. Same failure shape, so pin the field."""
    assert "cpu_offload_gb" in ServingMeta.__dataclass_fields__
    assert ServingMeta(runtime="vllm").cpu_offload_gb is None
    assert ServingMeta(runtime="vllm", cpu_offload_gb=75.0).cpu_offload_gb == 75.0


def test_cli_exposes_the_flag_and_defaults_to_off():
    from glq.bench.cli import build_parser

    p = build_parser()
    off = p.parse_args(["run", "--model", "m", "--tasks", "wikitext2_ppl_vllm"])
    assert off.cpu_offload_gb == 0, "default must leave existing invocations unchanged"
    on = p.parse_args(["run", "--model", "m", "--tasks", "wikitext2_ppl_vllm",
                       "--cpu-offload-gb", "75"])
    assert on.cpu_offload_gb == 75.0


def test_runner_accepts_and_forwards_it():
    """The CLI flag is useless if `runner.run` drops it before `runtime.load`."""
    import inspect

    from glq.bench import runner, runtime
    assert "cpu_offload_gb" in inspect.signature(runner.run).parameters
    assert "cpu_offload_gb" in inspect.signature(runtime.load).parameters
    src = inspect.getsource(runner.run)
    assert "cpu_offload_gb=cpu_offload_gb" in src, (
        "runner.run accepts the kwarg but never passes it on — the flag would be silently "
        "inert, which is the failure mode this whole file guards")
