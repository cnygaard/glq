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

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "infra", "spot_hunter.py")

# `infra/` is gitignored (it holds the SSH key), so spot_hunter.py is not in the
# repository — these run for whoever is editing the script locally and skip everywhere
# else, rather than failing a checkout that legitimately does not contain it.
SH = pytest.importorskip("spot_hunter",
                         reason="infra/spot_hunter.py is not present (infra/ is gitignored)")


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


# ---- --try-create: walking the price ladder ---------------------------------------------
#
# The cheapest rung often has no capacity, and today that only surfaces from `tofu apply`
# waiting out `timeouts { create = "3m" }`. RunInstances with InstanceMarketOptions answers
# capacity synchronously, so the ladder can be walked in ~10s a rung. What follows guards
# the decisions that make that walk useful rather than noisy — and it creates no instances:
# the probe is injected.

def _rows(*prices):
    """Ladder rows in the shape latest_prices emits, cheapest first."""
    return [{"region": "eu-north-1", "az": "eu-north-1a", "instance": f"g{i}.xlarge",
             "price": p, "gpu": "L40S", "vram": "48GB", "vram_gb": 48, "cc": "sm_89"}
            for i, p in enumerate(sorted(prices))]


def test_capacity_errors_advance_to_the_next_rung():
    """The whole point: no capacity here does not mean no capacity anywhere."""
    for code in ("InsufficientInstanceCapacity", "SpotMaxPriceTooLow", "Unsupported"):
        assert SH.classify_error(code) == "next", code


def test_account_level_errors_stop_the_walk():
    """A quota or permissions failure is not about this rung. Treating it as 'no capacity'
    walks the entire ladder burning API calls and then reports, wrongly, that nothing is
    available anywhere."""
    for code in ("VcpuLimitExceeded", "InstanceLimitExceeded",
                 "MaxSpotInstanceCountExceeded", "UnauthorizedOperation", "AuthFailure"):
        assert SH.classify_error(code) == "stop", code


def test_an_unrecognised_error_stops_rather_than_being_swallowed():
    """Silently mapping the unknown to 'no capacity' is how a real misconfiguration reads
    as an empty market."""
    assert SH.classify_error("SomethingNobodyHasSeen") == "stop"


def test_the_ladder_is_walked_cheapest_first():
    tried = []
    rows = _rows(0.90, 0.30, 0.55)

    def probe(row):
        tried.append(row["price"])
        return SH.ProbeResult(ok=row["price"] == 0.55, reason="", fatal=False)

    win = SH.walk_ladder(rows, probe=probe, log=lambda *_: None)
    assert [r for r in tried] == [0.30, 0.55], f"order was {tried}"
    assert win is not None and win["price"] == 0.55
    assert 0.90 not in tried, "the walk must stop at the first success"


def test_max_price_caps_the_walk():
    tried = []

    def probe(row):
        tried.append(row["price"])
        return SH.ProbeResult(ok=False, reason="no capacity", fatal=False)

    win = SH.walk_ladder(_rows(0.30, 0.55, 2.50), max_price=1.0,
                         probe=probe, log=lambda *_: None)
    assert tried == [0.30, 0.55], f"walked past --max-price: {tried}"
    assert win is None


def test_a_fatal_probe_stops_the_walk_immediately():
    tried = []

    def probe(row):
        tried.append(row["price"])
        return SH.ProbeResult(ok=False, reason="VcpuLimitExceeded", fatal=True)

    win = SH.walk_ladder(_rows(0.30, 0.55, 0.90), probe=probe, log=lambda *_: None)
    assert tried == [0.30], f"kept walking after a fatal error: {tried}"
    assert win is None


def test_the_winner_is_what_reaches_the_tf_file_not_the_cheapest():
    """The reason the tool exists: the launchable rung and the cheapest rung differ."""
    rows = _rows(0.30, 0.55)
    win = SH.walk_ladder(rows, probe=lambda r: SH.ProbeResult(r["price"] == 0.55, "", False),
                         log=lambda *_: None)
    tf = SH.render_spot_tf({**win, "vcpus": 4, "ram": 16.0})
    assert win["instance"] in tf
    assert rows[0]["instance"] not in tf, "the cheapest, unlaunchable rung leaked into the tf"


def test_a_probe_instance_is_always_terminated(monkeypatch):
    """A probe creates a real billable instance. If anything between create and terminate
    raises, the instance must still be killed — a leak runs until somebody notices."""
    killed = []

    class _Client:
        def run_instances(self, **kw):
            return {"Instances": [{"InstanceId": "i-probe"}]}

        def describe_instances(self, **kw):
            raise RuntimeError("boom between create and terminate")

        def terminate_instances(self, InstanceIds=None, **kw):
            killed.extend(InstanceIds or [])
            return {}

    res = SH.probe_rung(_rows(0.30)[0], client=_Client(), image_id="ami-x",
                        subnet_id="subnet-x")
    assert killed == ["i-probe"], "probe instance leaked"
    assert res.ok is False


def test_without_try_create_nothing_is_ever_launched(monkeypatch, capsys, tmp_path):
    """The default path is a read-only price scan and must stay one. A plain
    `spot_hunter --top 10` that quietly started billing would be a nasty surprise."""
    import types as _t

    launched = []

    class _Paginator:
        def paginate(self, **kw):
            return [{"SpotPriceHistory": [
                {"InstanceType": "g6.xlarge", "AvailabilityZone": "eu-north-1a",
                 "Timestamp": SH.dt.datetime.now(SH.dt.timezone.utc),
                 "SpotPrice": "0.40"}]}]

    class _Client:
        def get_paginator(self, _name):
            return _Paginator()

        def run_instances(self, **kw):
            launched.append(kw)
            raise AssertionError("a read-only price scan launched an instance")

        def describe_instance_types(self, **kw):
            return {"InstanceTypes": []}

    fake = _t.ModuleType("boto3")
    fake.Session = lambda **kw: _t.SimpleNamespace(
        client=lambda *a, **k: _Client(), get_credentials=lambda: object())
    monkeypatch.setitem(sys.modules, "boto3", fake)
    monkeypatch.setattr(sys, "argv", ["spot_hunter", "--regions", "eu-north-1",
                                      "--write-tf", str(tmp_path / "out.tf")])

    assert SH.main() == 0
    assert launched == []


def test_max_price_without_try_create_is_refused():
    """Silently ignoring it would let someone believe a cap was applied to a walk that
    never happened."""
    proc = subprocess.run([sys.executable, SCRIPT, "--max-price", "2.0"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2
    assert "only applies with --try-create" in proc.stderr


def test_probes_are_confined_to_the_scanned_regions(monkeypatch):
    """A probe must only ever launch in a region the price scan already covered.

    If the probe path could reach a region outside --regions, a leaked instance would be
    somewhere nobody thinks to look — and --cleanup-probes, which sweeps the same flag,
    would report all-clear without having looked there."""
    touched = []
    monkeypatch.setattr(SH, "_probe_client",
                        lambda sk, region, t: touched.append(region) or object())
    monkeypatch.setattr(SH, "_al2023_ami", lambda sk, region: "ami-x")
    monkeypatch.setattr(SH, "_default_subnet", lambda c, az: "subnet-x")
    monkeypatch.setattr(SH, "probe_rung",
                        lambda *a, **k: SH.ProbeResult(False, "InsufficientInstanceCapacity",
                                                       False))

    scanned = {"eu-north-1", "eu-south-2"}
    rows = [{"region": "eu-north-1", "az": "eu-north-1a", "instance": "g7e.2xlarge",
             "price": 1.11, "gpu": "RTX PRO 6000", "vram": "96GB", "vram_gb": 96,
             "cc": "sm_120"},
            {"region": "eu-south-2", "az": "eu-south-2b", "instance": "g7e.2xlarge",
             "price": 1.16, "gpu": "RTX PRO 6000", "vram": "96GB", "vram_gb": 96,
             "cc": "sm_120"}]

    SH.try_create(rows, {}, max_price=2.0)
    assert set(touched) <= scanned, f"probed outside the scan: {set(touched) - scanned}"
    assert touched, "the walk did not probe at all"


def test_cleanup_names_the_regions_it_swept(capsys):
    """"No probe instances found" is a clean bill of health. It has to say which regions it
    is a clean bill of health FOR — a walk run with --regions us-east-1 and then killed
    leaves instances a default-region sweep never looks at."""
    import types as _t

    class _Client:
        def describe_instances(self, **kw):
            return {"Reservations": []}

    fake = _t.ModuleType("boto3")
    fake.Session = lambda **kw: _t.SimpleNamespace(client=lambda *a, **k: _Client())
    sys.modules["boto3"] = fake
    try:
        SH.cleanup_probes(["eu-north-1", "eu-south-2"], {})
    finally:
        del sys.modules["boto3"]
    out = capsys.readouterr().out
    assert "eu-north-1" in out and "eu-south-2" in out, out
