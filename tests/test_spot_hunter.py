"""spot_hunter's pure logic — the parts that decide what gets launched.

No AWS here: everything below is table-building, mapping and filtering, which is where the
mistakes that cost a provisioning cycle live. The one live-API failure mode this file DOES
guard is the missing-comma trap the module documents: Python concatenates adjacent string
literals, so a dropped comma silently merges two instance types into one unmatchable
string. It cost seven types once, including the cheapest single-GPU L40S and Blackwell
entry points, which simply never appeared in any hunt.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "infra"))

import spot_hunter as SH  # noqa: E402

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "infra", "spot_hunter.py")


# ---- the lineups are well-formed ---------------------------------------------------------

@pytest.mark.parametrize("lineup", ["DEFAULT_TYPES", "DEFAULT_CPU_TYPES"])
def test_every_instance_type_can_actually_match(lineup):
    """A merged or space-padded entry matches nothing and fails SILENTLY: AWS treats an
    unknown instance type as a filter that returns no rows."""
    bad = [t for t in getattr(SH, lineup) if not SH._TYPE_RE.match(t)]
    assert not bad, f"{lineup} has unmatchable entries (missing comma?): {bad}"


def test_the_cpu_lineup_holds_no_gpu_families():
    """--cpu-only must not quietly price GPU boxes; the whole point is a cheaper machine."""
    gpu_families = set(SH.GPU_INFO)
    overlap = [t for t in SH.DEFAULT_CPU_TYPES if SH._family(t) in gpu_families]
    assert not overlap, overlap


def test_the_cpu_lineup_is_x86_only():
    """The published wheels are manylinux_2_28_x86_64, so `pip install glq` has nothing to
    install on Graviton, and the CPU kernels' SIMD tiers are x86 intrinsics — an ARM box
    would fall to the scalar tier even built from source."""
    arm = [t for t in SH.DEFAULT_CPU_TYPES
           if SH._family(t).endswith("g") or SH._family(t).endswith("gd")]
    assert not arm, f"ARM families in the CPU lineup: {arm}"


def test_every_cpu_family_declares_its_simd_tier():
    """A row with an unknown tier tells the reader nothing about decode speed."""
    missing = sorted({SH._family(t) for t in SH.DEFAULT_CPU_TYPES} - set(SH.CPU_INFO))
    assert not missing, f"families with no CPU_INFO entry: {missing}"


def test_the_declared_tiers_are_ones_the_kernel_actually_has():
    """The labels must match glq_cpu_dispatch.cpp's tier names, or the table promises a
    kernel that does not exist."""
    known = {"scalar", "avx2", "avx512", "avx512fp16"}
    for family, (_cpu, isa) in SH.CPU_INFO.items():
        assert isa in known, f"{family} claims unknown tier {isa!r}"


def test_unknown_families_report_a_question_mark_rather_than_guessing():
    assert SH._cpu_cols("zz9.2xlarge") == ("?", "?")


# ---- row shape ---------------------------------------------------------------------------

def _price_rows(cpu_only):
    """latest_prices with boto3 faked out — the mapping is what is under test."""
    import types as _t

    class _Paginator:
        def paginate(self, **kw):
            return [{"SpotPriceHistory": [
                {"InstanceType": "c7i.4xlarge", "AvailabilityZone": "eu-north-1a",
                 "Timestamp": SH.dt.datetime.now(SH.dt.timezone.utc), "SpotPrice": "0.30"},
                {"InstanceType": "g6.xlarge", "AvailabilityZone": "eu-north-1a",
                 "Timestamp": SH.dt.datetime.now(SH.dt.timezone.utc), "SpotPrice": "0.40"},
            ]}]

    class _Client:
        def get_paginator(self, _name):
            return _Paginator()

    fake = _t.ModuleType("boto3")
    fake.Session = lambda **kw: _t.SimpleNamespace(client=lambda *a, **k: _Client())
    sys.modules["boto3"] = fake
    try:
        return {r["instance"]: r for r in
                SH.latest_prices("eu-north-1", [], "Linux/UNIX", {}, cpu_only=cpu_only)}
    finally:
        del sys.modules["boto3"]


def test_cpu_mode_reports_the_simd_tier_and_no_vram():
    rows = _price_rows(cpu_only=True)
    c7i = rows["c7i.4xlarge"]
    assert c7i["gpu"] == "Xeon Sapphire Rapids"
    assert c7i["cc"] == "avx512fp16"
    assert c7i["vram_gb"] == 0, "a CPU box has no VRAM to report"


def test_gpu_mode_is_unchanged():
    g6 = _price_rows(cpu_only=False)["g6.xlarge"]
    assert (g6["gpu"], g6["vram"], g6["cc"]) == ("L4", "24GB", "sm_89")
    assert g6["vram_gb"] == 24


# ---- generated terraform ------------------------------------------------------------------

def test_the_generated_tf_says_cpu_when_hunting_cpu():
    """It is a committed file someone reads later; calling a c7i a GPU instance is how a
    stale comment starts."""
    row = {"instance": "c7i.4xlarge", "price": 0.3, "region": "eu-north-1",
           "az": "eu-north-1a", "gpu": "Xeon Sapphire Rapids", "vram": "-",
           "cc": "avx512fp16", "vcpus": 16, "ram": 32.0}
    tf = SH.render_spot_tf(row, cpu_only=True)
    assert "Cheapest CPU spot instance" in tf
    assert "VRAM" not in tf
    assert "avx512fp16 tier" in tf
    assert 'default     = "c7i.4xlarge"' in tf


def test_the_generated_tf_still_says_gpu_by_default():
    row = {"instance": "g6.xlarge", "price": 0.4, "region": "eu-north-1",
           "az": "eu-north-1a", "gpu": "L4", "vram": "24GB", "cc": "sm_89",
           "vcpus": 4, "ram": 16.0}
    tf = SH.render_spot_tf(row)
    assert "Cheapest GPU spot instance" in tf and "24GB VRAM" in tf


# ---- flag combinations that would otherwise fail confusingly ------------------------------

@pytest.mark.parametrize("argv,expected", [
    (["--isa", "avx512"], "only applies with --cpu-only"),
    (["--cpu-only", "--cc", "sm_89"], "filter with --isa"),
    (["--cpu-only", "--vram", "24"], "CPU instances have none"),
])
def test_mismatched_filters_are_refused_with_an_explanation(argv, expected):
    """These combinations are silently empty result sets otherwise — the reader is left
    thinking there was no capacity."""
    proc = subprocess.run([sys.executable, SCRIPT, *argv],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert expected in proc.stderr, proc.stderr
