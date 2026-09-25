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

@pytest.mark.parametrize("lineup", ["DEFAULT_TYPES", "DEFAULT_CPU_TYPES",
                                    "DEFAULT_METAL_TYPES"])
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


# ---- bare metal (--metal) ----------------------------------------------------------------

#: The sizes this flag exists to price. Both spellings are real and both must pass: older
#: families are a bare `.metal`, newer ones carry the socket count.
METAL_SIZES = ["c7i.metal-24xl", "r8a.metal-24xl", "m7i.metal-24xl",
               "m8azn.metal-12xl", "m8a.metal-24xl", "m8azn.metal-24xl",
               "c5.metal", "m6i.metal"]


@pytest.mark.parametrize("itype", METAL_SIZES)
def test_metal_sizes_are_recognised(itype):
    """Before --metal existed the regex only knew `[0-9]*x?large`, so every metal row warned
    'will never match' on every run — the standing noise the regex exists to prevent."""
    assert SH._TYPE_RE.match(itype), itype


@pytest.mark.parametrize("itype", [
    "c7i.metal-24xlarge",   # metal sizes are `xl`, not `xlarge`
    "c7i.metal-",           # dangling separator
    "c7i.metalx",           # not a size
    "c7i.24xl",             # `xl` is metal-only; a virtual size is `24xlarge`
    "c7i.metal-24xl ",      # trailing space, the documented silent-vanish
])
def test_widening_the_regex_did_not_make_it_accept_junk(itype):
    """The regex earns its place by REJECTING things. Adding the metal spellings must not
    turn it into a rubber stamp, or the missing-comma trap comes back unnoticed."""
    assert not SH._TYPE_RE.match(itype), itype


def test_the_metal_lineup_holds_no_gpu_families():
    overlap = [t for t in SH.DEFAULT_METAL_TYPES if SH._family(t) in set(SH.GPU_INFO)]
    assert not overlap, overlap


def test_the_metal_lineup_is_x86_only():
    """Same reason as the CPU lineup: the wheels are manylinux_2_28_x86_64 and the SIMD
    tiers are x86 intrinsics."""
    arm = [t for t in SH.DEFAULT_METAL_TYPES
           if SH._family(t).endswith("g") or SH._family(t).endswith("gd")]
    assert not arm, f"ARM families in the metal lineup: {arm}"


def test_every_metal_family_declares_its_simd_tier():
    missing = sorted({SH._family(t) for t in SH.DEFAULT_METAL_TYPES} - set(SH.CPU_INFO))
    assert not missing, f"families with no CPU_INFO entry: {missing}"


def test_turin_is_not_claimed_to_have_fp16():
    """AVX512_FP16 is believed Intel-only (Sapphire Rapids and newer). Understating is the
    safe direction: the module's rule is that a wrong ISA claim sends someone to a box whose
    decode is a tier slower than the table promised. If Turin turns out to have it,
    `glq_cpu_active_isa()` on the box is the authority — not this table."""
    for family in ("m8a", "r8a", "c8a", "m8azn"):
        cpu, isa = SH.CPU_INFO[family]
        assert "Turin" in cpu, family
        assert isa == "avx512", f"{family} claims {isa!r}; FP16 is unverified on Zen 5"


# ---- the silent-no-match failure, from the other end -------------------------------------

def test_a_requested_type_that_never_priced_is_named():
    """`_validate_types` catches malformed spellings; this catches well-formed WRONG ones — a
    retired size, a typo'd socket count, or a family not offered in the scanned regions. AWS
    treats an unknown type as a filter matching nothing, so without this it vanishes."""
    rows = [{"instance": "c7i.metal-24xl"}]
    missing = SH.report_unpriced(["c7i.metal-24xl", "m8a.metal-96xl"], rows)
    assert missing == ["m8a.metal-96xl"]


def test_nothing_is_reported_when_every_type_priced(capsys):
    rows = [{"instance": "c5.metal"}, {"instance": "m6i.metal"}]
    assert SH.report_unpriced(["c5.metal", "m6i.metal"], rows) == []
    assert capsys.readouterr().err == "", "silence when there is nothing to report"


# ---- specs are region-scoped, even though the values are not -----------------------------

def _fake_specs_boto3(catalogue):
    """boto3 whose describe_instance_types knows only `catalogue[region]`, and raises for
    the whole call on anything else — which is exactly what AWS does."""
    import types as _t

    class _Client:
        def __init__(self, region):
            self.region = region

        def describe_instance_types(self, InstanceTypes):  # noqa: N803 - boto3's casing
            known = catalogue.get(self.region, set())
            bad = [t for t in InstanceTypes if t not in known]
            if bad:
                raise RuntimeError(f"InvalidInstanceType: do not exist: {bad}")
            return {"InstanceTypes": [
                {"InstanceType": t, "VCpuInfo": {"DefaultVCpus": 96},
                 "MemoryInfo": {"SizeInMiB": 192 * 1024}} for t in InstanceTypes]}

    fake = _t.ModuleType("boto3")
    fake.Session = lambda **kw: _t.SimpleNamespace(
        client=lambda _svc, region_name: _Client(region_name))
    return fake


def test_specs_are_asked_of_each_region_about_only_its_own_types():
    """`describe_instance_types` only knows the types OFFERED in the region it is called
    against, and rejects the WHOLE call for the rest. This used to be one call in whichever
    region happened to be cheapest, so a single type priced only elsewhere blanked vCPU and
    RAM for every row. --metal hits it constantly: metal availability varies far more by
    region than virtual does."""
    catalogue = {"eu-north-1": {"c5.metal"}, "eu-west-1": {"m8a.metal-24xl"}}
    sys.modules["boto3"] = _fake_specs_boto3(catalogue)
    try:
        specs = SH.fetch_specs(
            {"eu-north-1": ["c5.metal"], "eu-west-1": ["m8a.metal-24xl"]}, {})
    finally:
        del sys.modules["boto3"]
    assert set(specs) == {"c5.metal", "m8a.metal-24xl"}, \
        "a type priced in one region must get its specs from THAT region"


def test_one_region_failing_does_not_blank_the_others():
    """Best-effort per region: a table with some specs beats one with none."""
    catalogue = {"eu-north-1": {"c5.metal"}}          # eu-west-1 knows nothing -> raises
    sys.modules["boto3"] = _fake_specs_boto3(catalogue)
    try:
        specs = SH.fetch_specs(
            {"eu-north-1": ["c5.metal"], "eu-west-1": ["m8a.metal-24xl"]}, {})
    finally:
        del sys.modules["boto3"]
    assert set(specs) == {"c5.metal"}


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
    # --metal implies --cpu-only, so it must inherit the same guards rather than silently
    # accepting GPU filters that can never match a metal CPU box.
    (["--metal", "--cc", "sm_89"], "filter with --isa"),
    (["--metal", "--vram", "24"], "CPU instances have none"),
])
def test_mismatched_filters_are_refused_with_an_explanation(argv, expected):
    """These combinations are silently empty result sets otherwise — the reader is left
    thinking there was no capacity."""
    proc = subprocess.run([sys.executable, SCRIPT, *argv],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert expected in proc.stderr, proc.stderr


def _selected_lineup(argv):
    """Which types main() would price, with the AWS calls stubbed out.

    Exercises the real argument wiring rather than re-deriving it: the flag ordering is the
    bug risk here, since --metal implies --cpu-only and would never reach its own lineup if
    the branches were checked the other way round.
    """
    import types as _t
    seen: list[list[str]] = []

    def _fake_latest_prices(region, types, product, session_kwargs, cpu_only=False):
        seen.append(list(types))
        return []

    orig = SH.latest_prices
    SH.latest_prices = _fake_latest_prices
    argv_orig = sys.argv
    sys.argv = ["spot_hunter.py", "--regions", "eu-north-1", *argv]
    fake = _t.ModuleType("boto3")
    fake.Session = lambda **kw: _t.SimpleNamespace(client=lambda *a, **k: None)
    sys.modules["boto3"] = fake
    try:
        SH.main()
    except SystemExit:
        pass
    finally:
        SH.latest_prices = orig
        sys.argv = argv_orig
        sys.modules.pop("boto3", None)
    return seen[0] if seen else []


def test_metal_selects_the_metal_lineup_not_the_cpu_one():
    """--metal sets cpu_only, so if the lineup branches were ordered the other way it would
    silently price the virtual CPU lineup and nobody would notice from the output."""
    types = _selected_lineup(["--metal"])
    assert "c7i.metal-24xl" in types
    assert "m8azn.metal-12xl" in types
    assert not any(t.endswith("xlarge") for t in types), \
        f"virtual sizes leaked into the metal hunt: {[t for t in types if t.endswith('xlarge')]}"


def test_cpu_only_is_unaffected_by_the_new_flag():
    types = _selected_lineup(["--cpu-only"])
    assert types == SH.DEFAULT_CPU_TYPES
    assert not any(".metal" in t for t in types)


def test_an_explicit_instance_type_still_overrides_metal():
    assert _selected_lineup(["--metal", "--instance-types", "r8a.metal-24xl"]) == \
        ["r8a.metal-24xl"]


def test_the_default_hunt_is_still_gpu():
    assert _selected_lineup([]) == SH.DEFAULT_TYPES


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


# ---- --list-running: "is anything of mine still up" -------------------------------------
#
# A different question from --cleanup-probes, which only matches the glq-spot-probe tag. The
# case that prompted this flag was a real box left running under an OpenTofu Name, which the
# probe sweep reported as absent because it never looked for it.

def _fake_ec2(instances_by_region, terminated=None, regions=None):
    """boto3 whose describe_instances honours the tag + state filters, like AWS does."""
    import types as _t

    class _Paginator:
        def __init__(self, region, filters):
            self.region, self.filters = region, filters

        def paginate(self, **kw):
            want = {f["Name"]: f["Values"] for f in self.filters}
            out = []
            for inst in instances_by_region.get(self.region, []):
                if "tag:Name" in want and inst.get("_name") not in want["tag:Name"]:
                    continue
                if inst["State"]["Name"] not in want["instance-state-name"]:
                    continue
                out.append(inst)
            return [{"Reservations": [{"Instances": out}]}] if out else [{"Reservations": []}]

    class _Client:
        def __init__(self, region):
            self.region = region

        def get_paginator(self, _name):
            region = self.region

            class _P:  # boto3 takes the filters on paginate(), not get_paginator()
                def paginate(self, Filters=None, **kw):  # noqa: N803 - boto3's casing
                    return _Paginator(region, Filters or []).paginate()
            return _P()

        def describe_regions(self, AllRegions=False):  # noqa: N803
            return {"Regions": regions or []}

        def terminate_instances(self, InstanceIds):  # noqa: N803
            if terminated is not None:
                terminated.extend(InstanceIds)
            return {}

    fake = _t.ModuleType("boto3")
    fake.Session = lambda **kw: _t.SimpleNamespace(
        client=lambda _svc, region_name=None, **k: _Client(region_name),
        get_available_regions=lambda _svc: list(instances_by_region))
    return fake


def _inst(iid, itype="g6e.2xlarge", state="running", life="spot", name="-"):
    return {"InstanceId": iid, "InstanceType": itype, "State": {"Name": state},
            "InstanceLifecycle": life, "LaunchTime": SH.dt.datetime(2026, 9, 25, 19, 48,
                                                                   tzinfo=SH.dt.timezone.utc),
            "PublicIpAddress": "13.60.87.125",
            "Tags": [{"Key": "Name", "Value": name}], "_name": name}


def test_the_sweep_finds_a_box_the_probe_tag_filter_misses():
    """THE case this flag exists for. A box stood up by OpenTofu carries its own Name, so
    --cleanup-probes reports a clean bill of health while it is still billing."""
    inst = _inst("i-091e3dc56e0e31b48", name="golay-leech-quant-eval")
    sys.modules["boto3"] = _fake_ec2({"eu-north-1": [inst]})
    try:
        untagged = SH.sweep_instances(["eu-north-1"], {})
        probe_only = SH.sweep_instances(["eu-north-1"], {}, tag=SH.PROBE_TAG_NAME)
    finally:
        del sys.modules["boto3"]
    assert [r["id"] for r in untagged] == ["i-091e3dc56e0e31b48"]
    assert probe_only == [], "the probe-tag sweep must NOT see a differently-tagged box"


def test_the_sweep_reports_lifecycle_so_spot_is_distinguishable():
    """'Are there stray SPOT instances' turns on this field; AWS omits it for on-demand."""
    od = _inst("i-ondemand", name="build")
    del od["InstanceLifecycle"]
    sys.modules["boto3"] = _fake_ec2({"eu-west-1": [_inst("i-spot"), od]})
    try:
        rows = {r["id"]: r["life"] for r in SH.sweep_instances(["eu-west-1"], {})}
    finally:
        del sys.modules["boto3"]
    assert rows == {"i-spot": "spot", "i-ondemand": "on-demand"}


def test_the_sweep_terminates_nothing():
    """Read-only is the contract: --list-running must be safe to run without thinking."""
    killed = []
    sys.modules["boto3"] = _fake_ec2({"eu-north-1": [_inst("i-1")]}, terminated=killed)
    try:
        SH.sweep_instances(["eu-north-1"], {})
    finally:
        del sys.modules["boto3"]
    assert killed == [], "the sweep called terminate_instances"


def test_cleanup_still_terminates_through_the_shared_sweep():
    """The refactor must not have turned cleanup into a no-op."""
    killed = []
    probe = _inst("i-probe", name=SH.PROBE_TAG_NAME)
    sys.modules["boto3"] = _fake_ec2({"eu-north-1": [probe]}, terminated=killed)
    try:
        assert SH.cleanup_probes(["eu-north-1"], {}) == 1
    finally:
        del sys.modules["boto3"]
    assert killed == ["i-probe"]


def test_enabled_regions_skips_the_ones_that_cannot_hold_an_instance():
    """A not-opted-in region cannot hold an instance, so scanning it only produces
    AuthFailure noise — the standing error that trains a reader to ignore errors."""
    regions = [{"RegionName": "us-east-1", "OptInStatus": "opt-in-not-required"},
               {"RegionName": "eu-south-2", "OptInStatus": "opted-in"},
               {"RegionName": "eu-south-1", "OptInStatus": "not-opted-in"},
               {"RegionName": "ap-south-1", "OptInStatus": "opt-in-not-required"}]
    sys.modules["boto3"] = _fake_ec2({}, regions=regions)
    try:
        got = SH.enabled_regions({})
    finally:
        del sys.modules["boto3"]
    assert got == ["eu-south-2", "us-east-1"], got
    assert "eu-south-1" not in got and "ap-south-1" not in got


def test_a_clean_bill_of_health_says_what_it_covers(capsys):
    """"Nothing running" is only meaningful with the scope attached — the same reasoning
    cleanup_probes' own output already follows."""
    SH.print_running([], ["us-east-1", "eu-west-1"])
    out = capsys.readouterr().out
    assert "us-east-1" in out and "eu-west-1" in out and "2 region(s)" in out


def test_list_running_defaults_to_everywhere_not_the_hunt_default(monkeypatch):
    """--regions defaults to two cheap regions for a HUNT; for 'is anything of mine running'
    that would print a clean bill of health with a box up in us-east-1."""
    seen = {}
    monkeypatch.setattr(SH, "enabled_regions", lambda kw, **k: ["us-east-1", "eu-west-1"])
    monkeypatch.setattr(SH, "sweep_instances",
                        lambda regions, kw, **k: seen.setdefault("scope", regions) and [])
    monkeypatch.setattr(SH, "print_running", lambda rows, scope: None)
    monkeypatch.setattr(SH, "resolve_session_kwargs", lambda p: {})
    monkeypatch.setattr(sys, "argv", ["spot_hunter.py", "--list-running"])
    SH.main()
    assert seen["scope"] == ["us-east-1", "eu-west-1"]
    assert SH.DEFAULT_REGIONS != seen["scope"], "must not silently use the hunt default"


def test_list_running_honours_an_explicit_region(monkeypatch):
    seen = {}
    monkeypatch.setattr(SH, "enabled_regions",
                        lambda kw, **k: pytest.fail("should not widen when told a region"))
    monkeypatch.setattr(SH, "sweep_instances",
                        lambda regions, kw, **k: seen.setdefault("scope", regions) and [])
    monkeypatch.setattr(SH, "print_running", lambda rows, scope: None)
    monkeypatch.setattr(SH, "resolve_session_kwargs", lambda p: {})
    monkeypatch.setattr(sys, "argv",
                        ["spot_hunter.py", "--list-running", "--regions", "us-west-2"])
    SH.main()
    assert seen["scope"] == ["us-west-2"]


def test_the_hunt_still_defaults_to_the_two_cheap_regions(monkeypatch):
    """Guards the --regions default=None change: every non-list path must be unaffected."""
    seen = {}
    monkeypatch.setattr(SH, "latest_prices",
                        lambda r, t, p, kw, cpu_only=False: seen.setdefault(
                            "regions", []).append(r) or [])
    monkeypatch.setattr(SH, "resolve_session_kwargs", lambda p: {})
    monkeypatch.setattr(sys, "argv", ["spot_hunter.py", "--cpu-only"])
    try:
        SH.main()
    except SystemExit:
        pass
    assert sorted(seen["regions"]) == sorted(SH.DEFAULT_REGIONS)


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
