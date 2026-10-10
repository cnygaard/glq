"""Mirroring a bench job directory to S3 so a spot reclaim cannot take it.

Same failure `glq/resume.py` exists for, one level up: a Terminal-Bench run is hours of Docker
rollouts whose trajectories, per-trial `result.json` and verifier output live only on the box.
Two reclaims during one session destroyed evidence mid-investigation.

Nothing here touches AWS. The subprocess runner and the IMDS opener are injected, exactly as
`ResumeStore` injects its `backend`.
"""
from __future__ import annotations

import io
import json
import os
import sys
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

from glq.bench.artifact_sync import ArtifactSync, spot_interruption_pending  # noqa: E402


class FakeRunner:
    """Records argv instead of shelling out. Returncode is settable per test."""

    def __init__(self, returncode=0, stderr=""):
        self.calls: list[list[str]] = []
        self.returncode = returncode
        self.stderr = stderr

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))

        class R:
            returncode = self.returncode
            stdout = ""
            stderr = self.stderr
        return R()


def _sync(tmp_path, runner, **kw):
    kw.setdefault("bucket", "glq-resume-test")
    kw.setdefault("prefix", "terminal_bench/org__m/job-1")
    return ArtifactSync(local_dir=str(tmp_path), runner=runner,
                        imds=lambda: False, out=io.StringIO(), **kw)


# ---- the command ------------------------------------------------------------------------

def test_the_sync_command_mirrors_the_directory_to_the_prefix(tmp_path):
    runner = FakeRunner()
    with _sync(tmp_path, runner) as s:
        s.sync_now()
    assert runner.calls, "no sync ran"
    argv = runner.calls[0]
    assert argv[:3] == ["aws", "s3", "sync"]
    assert argv[3] == str(tmp_path)
    assert argv[4] == "s3://glq-resume-test/terminal_bench/org__m/job-1"


def test_delete_is_never_passed():
    """The dangerous flag, asserted on its own. `aws s3 sync --delete` would propagate a
    locally-lost file as a REMOTE deletion -- making the mirror destroy the thing it exists to
    protect, at exactly the moment the local copy went away."""
    runner = FakeRunner()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with ArtifactSync(local_dir=d, bucket="b", prefix="p", runner=runner,
                          imds=lambda: False, out=io.StringIO()) as s:
            s.sync_now()
    for argv in runner.calls:
        assert "--delete" not in argv
        assert not any(a.startswith("--delete") for a in argv)


def test_errors_only_so_a_30s_cadence_does_not_bury_the_log(tmp_path):
    runner = FakeRunner()
    with _sync(tmp_path, runner) as s:
        s.sync_now()
    assert "--only-show-errors" in runner.calls[0]


def test_excludes_are_passed_through(tmp_path):
    """An escape hatch, NOT a cost mitigation -- the default is to mirror everything.

    A task's own output artifact was measured at 178 MB, which sounds like it wants excluding
    until you price it: uploads into S3 are free, inter-region transfer is ~$0.02/GB, so that
    artifact costs ~$0.004 and a full 66-task suite ~$0.23, moving over the AWS backbone in
    seconds. This key exists for a caller with a reason of their own, not because bulk
    artifacts are a problem to design around."""
    runner = FakeRunner()
    with _sync(tmp_path, runner, exclude=["*.safetensors", "*.bin"]) as s:
        s.sync_now()
    argv = runner.calls[0]
    assert argv.count("--exclude") == 2
    assert "*.safetensors" in argv and "*.bin" in argv


# ---- inert by default -------------------------------------------------------------------

def test_no_bucket_means_no_thread_and_no_subprocess(tmp_path):
    """The guard that keeps this change invisible to every box without a bucket -- a laptop,
    a box provisioned before the resume stack existed, anyone running the bench by hand."""
    runner = FakeRunner()
    out = io.StringIO()
    with ArtifactSync(local_dir=str(tmp_path), bucket=None, prefix="p",
                      runner=runner, imds=lambda: False, out=out) as s:
        s.sync_now()
        assert s.enabled is False
        assert s.thread is None
    assert runner.calls == []


def test_a_disabled_sync_says_so_rather_than_failing_silently(tmp_path):
    """Silent-but-visible: a run whose artifacts are NOT protected must say so, or the first
    time anyone notices is after a reclaim has already taken them."""
    out = io.StringIO()
    with ArtifactSync(local_dir=str(tmp_path), bucket=None, prefix="p",
                      runner=FakeRunner(), imds=lambda: False, out=out):
        pass
    said = out.getvalue().lower()
    assert "not" in said and ("s3" in said or "sync" in said)


def test_the_destination_is_reported_once_when_enabled(tmp_path):
    """An artifact nobody can find is not saved."""
    out = io.StringIO()
    with ArtifactSync(local_dir=str(tmp_path), bucket="b", prefix="terminal_bench/x",
                      runner=FakeRunner(), imds=lambda: False, out=out):
        pass
    assert "s3://b/terminal_bench/x" in out.getvalue()


# ---- durability discipline --------------------------------------------------------------

def test_a_final_sync_runs_even_when_the_body_raises(tmp_path):
    """The run's own failure is the case most likely to leave something worth reading."""
    runner = FakeRunner()
    with pytest.raises(RuntimeError, match="boom"):
        with _sync(tmp_path, runner):
            raise RuntimeError("boom")
    assert runner.calls, "nothing was mirrored on the failure path"


def test_a_sync_failure_is_reported_but_never_propagates(tmp_path):
    """A broken mirror must not fail a benchmark that otherwise succeeded -- losing hours of
    GPU time to an S3 permission problem would be worse than losing the mirror."""
    runner = FakeRunner(returncode=1, stderr="AccessDenied")
    out = io.StringIO()
    with ArtifactSync(local_dir=str(tmp_path), bucket="b", prefix="p", runner=runner,
                      imds=lambda: False, out=out) as s:
        assert s.sync_now() is False
    assert "accessdenied" in out.getvalue().lower() or "failed" in out.getvalue().lower()


def test_nothing_secret_is_ever_logged(tmp_path):
    """`resume.py` promises it "never accepts, logs, or stores a key" and so must this."""
    out = io.StringIO()
    runner = FakeRunner(returncode=1, stderr="x")
    with ArtifactSync(local_dir=str(tmp_path), bucket="b", prefix="p", runner=runner,
                      imds=lambda: False, out=out) as s:
        s.sync_now()
    said = out.getvalue()
    for leak in ("aws_access_key", "AKIA", "X-aws-ec2-metadata-token", "secret"):
        assert leak.lower() not in said.lower()


# ---- the spot-interruption probe, and its trap ------------------------------------------

class FakeIMDS:
    """Minimal IMDS. `token_required` reproduces IMDSv2, which is what makes 401 a trap."""

    def __init__(self, action=None, token_required=True, raise_transport=False):
        self.action = action
        self.token_required = token_required
        self.raise_transport = raise_transport
        self.token_fetches = 0
        self.gets = 0

    def __call__(self, url, *, method="GET", headers=None, timeout=None):
        headers = headers or {}
        if self.raise_transport:
            raise urllib.error.URLError("unroutable")
        if method == "PUT" and url.endswith("/latest/api/token"):
            self.token_fetches += 1
            return 200, b"a-token"
        self.gets += 1
        if self.token_required and not headers.get("X-aws-ec2-metadata-token"):
            return 401, b""
        if self.action is None:
            return 404, b""
        return 200, json.dumps({"action": self.action, "time": "2026-10-10T12:00:00Z"}).encode()


def test_a_tokenless_probe_401_is_not_read_as_no_interruption():
    """THE trap, measured on the box: without a token `spot/instance-action` returns 401
    forever, and the obvious reading is "nothing scheduled" -- backwards exactly when it
    matters. A token must be fetched and the probe retried.

    The same omission made a box with role glq-bake-2026... look like it had none."""
    imds = FakeIMDS(action="terminate", token_required=True)
    assert spot_interruption_pending(http=imds) is True
    assert imds.token_fetches == 1, (
        f"expected exactly one token fetch, got {imds.token_fetches} -- 0 means the 401 was "
        f"taken at face value, >1 means the token endpoint is being hit redundantly")


def test_404_with_a_token_is_the_healthy_state():
    imds = FakeIMDS(action=None, token_required=True)
    assert spot_interruption_pending(http=imds) is False
    assert imds.token_fetches == 1
    assert imds.gets == 1, "a healthy 404 should not be retried"


def test_an_announced_interruption_is_detected():
    assert spot_interruption_pending(http=FakeIMDS(action="terminate")) is True
    assert spot_interruption_pending(http=FakeIMDS(action="stop")) is True


def test_an_unroutable_imds_is_not_an_error():
    """On a laptop the link-local address does not route. That must disable the probe, not
    raise into the run and not retry forever."""
    assert spot_interruption_pending(http=FakeIMDS(raise_transport=True)) is False


def test_an_interruption_forces_an_immediate_sync(tmp_path):
    """The point of the probe: a ~2 minute warning turns "lose up to one interval" into
    "lose almost nothing"."""
    runner = FakeRunner()
    with ArtifactSync(local_dir=str(tmp_path), bucket="b", prefix="p", runner=runner,
                      imds=lambda: True, out=io.StringIO(), interval=3600) as s:
        drained = s.tick()                      # one loop iteration, no sleeping
    assert drained is True, "an announced interruption did not trigger a sync"
    assert runner.calls, "interruption did not mirror anything"
