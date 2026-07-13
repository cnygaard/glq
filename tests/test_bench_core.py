"""CPU-only unit tests for the glq-bench capture/record/index core.

No torch / vllm / GPU required — covers record (de)serialization, the
% -of-bf16 index math, HF/quant metadata derivation, and provenance capture.
"""
from __future__ import annotations

import json

import pytest

from glq.bench import (
    BenchmarkResult,
    BenchRecord,
    EnvMeta,
    HardwareMeta,
    ModelMeta,
    ServingMeta,
    ThroughputResult,
)
from glq.bench import hfmeta, index, provenance
from glq.bench.record import read_jsonl, write_jsonl


def _rec(model_id, task, value, *, quant="glq", base="google/gemma-4-31B-it",
         metric="accuracy", standardized=True, ts="2026-06-20T00:00:00Z"):
    return BenchRecord(
        model=ModelMeta(id=model_id, base_model=base, quant_method=quant),
        benchmark=BenchmarkResult(task=task, metric=metric, value=value,
                                  standardized=standardized),
        timestamp_utc=ts,
    )


# --------------------------------------------------------------------------- #
# record (de)serialization
# --------------------------------------------------------------------------- #
def test_record_roundtrip_dict_and_json():
    r = BenchRecord(
        model=ModelMeta(id="xv0y5ncu/M-GLQ", base_model="org/M", quant_method="glq",
                        bpw=5.0, bpw_min=3, bpw_max=8, is_mixed=True,
                        weights_disk_gb=16.5, architecture="Gemma4ForConditionalGeneration",
                        hf_url="https://huggingface.co/xv0y5ncu/M-GLQ"),
        env=EnvMeta(python="3.12.0", torch="2.11.0+cu128", vllm="0.23.0"),
        hardware=HardwareMeta(gpu_model="RTX PRO 6000", gpu_count=1, gpu_total_vram_gib=95.6),
        serving=ServingMeta(runtime="vllm", load_gpu_mem_gib=16.5, max_model_len=20480,
                            llm_kwargs={"quantization": "glq"}),
        benchmark=BenchmarkResult(task="mmlu_pro", metric="accuracy", value=0.90,
                                  standardized=True, config={"n": 60, "budget": 16384},
                                  extra={"correct": 54, "n": 60}),
        throughput=ThroughputResult(output_tok_s=187.0, batch=60, measure="in_run_tqdm"),
    )
    back = BenchRecord.from_dict(r.to_dict())
    assert back.to_dict() == r.to_dict()
    back2 = BenchRecord.from_json(r.to_json())
    assert back2.model.bpw == 5.0
    assert back2.benchmark.extra["correct"] == 54
    assert back2.throughput.output_tok_s == 187.0


def test_record_forward_compatible_unknown_keys():
    d = _rec("m", "mmlu_pro", 0.9).to_dict()
    d["future_top_level"] = {"x": 1}
    d["model"]["future_field"] = "ignored"
    r = BenchRecord.from_dict(d)            # must not raise
    assert r.model.id == "m"


def test_jsonl_roundtrip(tmp_path):
    recs = [_rec("a", "mmlu_pro", 0.9), _rec("b", "aime_2026", 0.83)]
    p = tmp_path / "r.jsonl"
    write_jsonl(recs, p)
    write_jsonl([_rec("c", "mmlu_pro", 0.78, quant="none")], p, append=True)  # accumulate
    got = read_jsonl(p)
    assert len(got) == 3
    assert {g.model.id for g in got} == {"a", "b", "c"}
    # default mode overwrites (write_*, not append_*)
    write_jsonl([_rec("d", "mmlu_pro", 0.5)], p)
    assert {g.model.id for g in read_jsonl(p)} == {"d"}


# --------------------------------------------------------------------------- #
# index math (% of bf16)
# --------------------------------------------------------------------------- #
def test_index_percent_of_bf16_higher_and_lower_better():
    recs = [
        # bf16 baseline for base B on two tasks
        _rec("google/gemma-4-31B-it", "mmlu_pro", 0.867, quant="none", base=None),
        _rec("google/gemma-4-31B-it", "wikitext2_ppl", 7.0, quant="none", base=None,
             metric="perplexity"),
        # GLQ quant of B
        _rec("xv0y5ncu/B-GLQ", "mmlu_pro", 0.90, quant="glq"),
        _rec("xv0y5ncu/B-GLQ", "wikitext2_ppl", 7.2, quant="glq", metric="perplexity"),
    ]
    idx = index.compute_index(recs)
    glq = idx["xv0y5ncu/B-GLQ"]
    # accuracy: higher better -> 0.90/0.867 ; ppl: lower better -> 7.0/7.2
    assert glq["per_task"]["mmlu_pro"]["retention"] == pytest.approx(0.90 / 0.867, rel=1e-6)
    assert glq["per_task"]["wikitext2_ppl"]["retention"] == pytest.approx(7.0 / 7.2, rel=1e-6)
    assert glq["index"] == pytest.approx(((0.90 / 0.867) + (7.0 / 7.2)) / 2, rel=1e-6)
    # the bf16 baseline scores exactly 1.0 of itself
    assert idx["google/gemma-4-31B-it"]["index"] == pytest.approx(1.0, rel=1e-9)


def test_index_missing_baseline_and_weights_and_rank():
    recs = [
        _rec("google/gemma-4-31B-it", "mmlu_pro", 0.80, quant="none", base=None),
        _rec("xv0y5ncu/B-GLQ", "mmlu_pro", 0.84, quant="glq"),
        _rec("xv0y5ncu/B-GLQ", "aime_2026", 0.83, quant="glq"),   # no bf16 baseline for aime
    ]
    idx = index.compute_index(recs)
    g = idx["xv0y5ncu/B-GLQ"]
    assert g["missing_baselines"] == ["aime_2026"]
    assert g["n_tasks"] == 1
    # weighting changes nothing with a single task but must be honoured
    idx_w = index.compute_index(recs, weights={"mmlu_pro": 3.0})
    assert idx_w["xv0y5ncu/B-GLQ"]["index"] == pytest.approx(0.84 / 0.80, rel=1e-6)
    order = [m for m, _, _ in index.rank(recs)]
    assert order[0] == "xv0y5ncu/B-GLQ"        # 1.05 > baseline 1.0


def test_index_latest_record_wins():
    recs = [
        _rec("google/gemma-4-31B-it", "mmlu_pro", 0.80, quant="none", base=None),
        _rec("xv0y5ncu/B-GLQ", "mmlu_pro", 0.70, quant="glq", ts="2026-06-19T00:00:00Z"),
        _rec("xv0y5ncu/B-GLQ", "mmlu_pro", 0.84, quant="glq", ts="2026-06-20T00:00:00Z"),
    ]
    idx = index.compute_index(recs)
    assert idx["xv0y5ncu/B-GLQ"]["per_task"]["mmlu_pro"]["value"] == 0.84


# --------------------------------------------------------------------------- #
# hfmeta derivation from a local checkpoint dir
# --------------------------------------------------------------------------- #
def _write_ckpt(d, config, qconfig=None, blob_mb=1):
    (d / "config.json").write_text(json.dumps(config))
    if qconfig is not None:
        (d / "quantize_config.json").write_text(json.dumps(qconfig))
    (d / "model.safetensors").write_bytes(b"\0" * (blob_mb * 1024 * 1024))


def test_hfmeta_glq_mixed_from_config(tmp_path):
    _write_ckpt(tmp_path, {
        "_name_or_path": "google/gemma-4-31B-it",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "quantization_config": {"quant_method": "glq", "bpw": 5.0,
                                "layer_bpw": {"a": 3, "b": 8, "c": 5}},
    })
    m = hfmeta.model_meta(str(tmp_path))
    assert m.base_model == "google/gemma-4-31B-it"
    assert m.base_hf_url == "https://huggingface.co/google/gemma-4-31B-it"
    assert m.hf_url is None                 # local dir
    assert m.quant_method == "glq"
    assert (m.bpw, m.bpw_min, m.bpw_max, m.is_mixed) == (5.0, 3, 8, True)
    assert m.architecture == "Gemma4ForConditionalGeneration"
    assert m.weights_disk_gb is not None    # found + summed the safetensors (rounds to 0.0 GiB for a toy blob)


def test_hfmeta_glq_from_quantize_config(tmp_path):
    _write_ckpt(tmp_path,
                {"architectures": ["LlamaForCausalLM"], "_name_or_path": "org/base"},
                qconfig={"quant_method": "glq", "bpw": 4})
    m = hfmeta.model_meta(str(tmp_path))
    assert m.quant_method == "glq"
    assert m.bpw == 4 and m.bpw_min == 4 and m.bpw_max == 4 and m.is_mixed is False


def test_hfmeta_bf16_and_modelopt(tmp_path):
    (tmp_path / "bf16").mkdir()
    _write_ckpt(tmp_path / "bf16", {"architectures": ["Gemma4ForConditionalGeneration"]})
    m = hfmeta.model_meta(str(tmp_path / "bf16"))
    assert m.quant_method == "none"
    assert m.bpw is None and m.is_mixed is None

    (tmp_path / "nvfp4").mkdir()
    _write_ckpt(tmp_path / "nvfp4", {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "_name_or_path": "google/gemma-4-31B-it",
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    })
    m2 = hfmeta.model_meta(str(tmp_path / "nvfp4"))
    assert m2.quant_method == "modelopt"


def test_hfmeta_quant_override(tmp_path):
    _write_ckpt(tmp_path, {"architectures": ["X"]})       # no quantization_config
    m = hfmeta.model_meta(str(tmp_path), quant_override="glq")
    assert m.quant_method == "glq"


# --------------------------------------------------------------------------- #
# provenance (degrades gracefully on CPU)
# --------------------------------------------------------------------------- #
def test_provenance_env_and_hardware_do_not_raise():
    env = provenance.env_snapshot()
    assert env.python and env.python.count(".") == 2     # always available
    hw = provenance.hardware_snapshot()
    assert isinstance(hw, HardwareMeta)                  # gpu fields may be None on CPU
