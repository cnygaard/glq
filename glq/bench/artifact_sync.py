"""Mirror a bench job directory to S3 while the run is still going.

`glq/resume.py` solves this one level down, for `glq-quantize`: persist work "locally, and
optionally to a private S3 bucket that outlives the instance. On-box checkpoints alone would
not help, because the failure to protect against *is* losing the box." A Terminal-Bench run has
the same shape — hours of Docker rollouts whose trajectories, per-trial `result.json` and
verifier output exist only on a spot instance. Two reclaims in a single session destroyed
evidence mid-investigation, each time because the data was there at the moment of the event and
gone by the time the question was asked.

**`aws s3 sync` rather than a boto3 walk.** Incremental change detection, parallelism, multipart
and retry are all already in it; doing this with boto3 means writing a change-detector.
`resume.py` uses boto3 because it puts *named artifacts it has just produced* — a different
problem. This mirrors a tree that another process is still writing.

**Never `--delete`.** A file lost locally must not propagate as a remote deletion. That would
make the mirror destroy what it exists to protect, at precisely the moment the local copy went
away.

**A broken mirror never fails a run.** Losing hours of GPU time to an S3 permission problem
would be worse than losing the mirror. Failures are reported and swallowed.

This module never accepts, logs, or stores a credential — the same promise `resume.py` makes.
Credentials come from the instance profile.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request

__all__ = ["ArtifactSync", "spot_interruption_pending"]

#: How often to mirror, in seconds. 30 keeps a reclaim's worst case to half a minute of
#: rollout while staying far below the cost of re-walking a few hundred files.
DEFAULT_INTERVAL = 30.0

#: Loop granularity. The interruption probe runs every tick, the sync only when the interval
#: has elapsed — so the ~2 minute spot warning is noticed promptly without mirroring every
#: few seconds.
_TICK = 5.0

_IMDS = "http://169.254.169.254"
#: Long enough to outlive any single bench run, so the token is fetched about once.
_TOKEN_TTL = "21600"
#: Link-local: either it answers at once or it is not there. A laptop must not stall here.
_IMDS_TIMEOUT = 2.0
#: Consecutive transport failures after which the probe switches off for good. On any
#: non-EC2 machine this fires immediately and then costs nothing for the rest of the run.
_IMDS_MAX_FAILURES = 3


def _http(url, *, method="GET", headers=None, timeout=_IMDS_TIMEOUT):
    """Minimal request returning ``(status, body)``. Injected in tests."""
    req = urllib.request.Request(url, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:   # noqa: S310 - link-local
        return resp.status, resp.read()


def spot_interruption_pending(*, http=_http) -> bool:
    """Has AWS scheduled this spot instance for interruption?

    **The trap this function exists to avoid.** IMDSv2 requires a token, and a token-less
    ``GET /latest/meta-data/spot/instance-action`` returns **401** — which reads as "nothing
    scheduled" and is backwards exactly when it matters. Measured on a live box: 401 without a
    token, **404 with** one (the healthy state), and 200 with a JSON body once an interruption
    is announced. The same omission made that box's `iam/security-credentials/` come back empty
    and look like it carried no instance profile when it did.

    So: fetch a token, retry once on 401, treat 404 as healthy and 200 as the notice. Any
    transport error means this is not EC2 (or IMDS is blocked) — return False rather than
    raising into a benchmark.
    """
    def _token():
        status, body = http(f"{_IMDS}/latest/api/token", method="PUT",
                            headers={"X-aws-ec2-metadata-token-ttl-seconds": _TOKEN_TTL})
        return body.decode(errors="replace").strip() if status == 200 else None

    def _get(token):
        headers = {"X-aws-ec2-metadata-token": token} if token else {}
        return http(f"{_IMDS}/latest/meta-data/spot/instance-action", headers=headers)

    try:
        status, body = _get(_token())
        if status == 401:                  # token rejected or expired — one retry, then give up
            status, body = _get(_token())
        if status != 200:
            return False                   # 404 is the healthy state
        try:
            action = (json.loads(body or b"{}") or {}).get("action")
        except (ValueError, TypeError):
            return True                    # a 200 we cannot parse still means "announced"
        return bool(action)
    except (urllib.error.URLError, OSError, ValueError):
        return False


class ArtifactSync:
    """Periodic `aws s3 sync` of one directory, plus an immediate sync on spot interruption.

    Inert unless a bucket is given, which is what keeps this invisible on a laptop or on a box
    provisioned before the resume stack existed. Use as a context manager: the background
    thread runs for the body, and `__exit__` always mirrors once more — including on the
    failure path, which is the case most likely to have left something worth reading.
    """

    def __init__(self, local_dir, bucket=None, prefix="", *, interval=DEFAULT_INTERVAL,
                 exclude=None, runner=None, imds=None, out=None):
        self.local_dir = str(local_dir)
        self.bucket = bucket or None
        self.prefix = str(prefix or "").strip("/")
        self.interval = float(interval)
        self.exclude = list(exclude or [])
        self._run = runner or subprocess.run
        self._imds = imds if imds is not None else spot_interruption_pending
        self._out = out if out is not None else sys.stderr
        self.thread = None
        self._stop = threading.Event()
        self._elapsed = 0.0
        self._imds_failures = 0
        self._interrupted = False

    # ---- plumbing ----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.bucket is not None

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}" if self.prefix else f"s3://{self.bucket}"

    def _say(self, msg):
        print(msg, file=self._out, flush=True)

    def argv(self) -> list[str]:
        """The sync command. Separate so the absence of `--delete` is assertable."""
        cmd = ["aws", "s3", "sync", self.local_dir, self.uri, "--only-show-errors"]
        for pattern in self.exclude:
            cmd += ["--exclude", pattern]
        return cmd

    # ---- the one operation -------------------------------------------------

    def sync_now(self) -> bool:
        """Mirror once. Returns success; never raises."""
        if not self.enabled:
            return False
        try:
            res = self._run(self.argv(), capture_output=True, text=True, check=False)
        except FileNotFoundError:
            # Named fix, in resume.py's style: a missing CLI is a packaging problem, not a
            # mystery to debug while a benchmark burns.
            self._say("  note: artifact sync needs the AWS CLI, which is not on PATH. "
                      "Install it (https://aws.amazon.com/cli/) or unset the bucket; the run "
                      "continues unmirrored")
            self.bucket = None
            return False
        except Exception as exc:                            # noqa: BLE001 - never fail a run
            self._say(f"  note: artifact sync failed ({type(exc).__name__}); "
                      f"the run continues unmirrored")
            return False
        if res.returncode != 0:
            tail = (getattr(res, "stderr", "") or "").strip().splitlines()
            self._say(f"  note: artifact sync failed (rc={res.returncode}): "
                      f"{tail[-1] if tail else 'no output'}")
            return False
        return True

    def tick(self) -> bool:
        """One loop iteration without sleeping. Returns True if it synced.

        Separate from the thread so the interruption path is testable without timing.
        """
        interrupted = False
        if self._imds_failures < _IMDS_MAX_FAILURES:
            try:
                interrupted = bool(self._imds())
            except Exception:                               # noqa: BLE001
                self._imds_failures += 1
                interrupted = False
        if interrupted and not self._interrupted:
            self._interrupted = True
            self._say("  spot interruption announced — mirroring artifacts now")
        if interrupted or self._elapsed >= self.interval:
            self._elapsed = 0.0
            self.sync_now()
            return True
        return False

    def _loop(self):
        while not self._stop.is_set():
            if self._stop.wait(_TICK):
                return
            self._elapsed += _TICK
            self.tick()

    # ---- context manager ---------------------------------------------------

    def __enter__(self):
        if not self.enabled:
            self._say("  note: artifact sync is OFF (no bucket configured) — job artifacts "
                      "will not survive this instance")
            return self
        self._say(f"  mirroring artifacts to {self.uri} every {self.interval:.0f}s")
        self.thread = threading.Thread(target=self._loop, daemon=True,
                                       name="glq-artifact-sync")
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=30)
            self.thread = None
        if self.enabled:
            # Always one last pass, including when the body raised.
            if self.sync_now():
                self._say(f"  artifacts mirrored to {self.uri}")
        return False


def aws_cli_present() -> bool:
    """Is `aws` reachable? For callers that want to warn before starting a long run."""
    return shutil.which("aws") is not None
