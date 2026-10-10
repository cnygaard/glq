"""Terminal-Bench via harbor — agentic terminal ability of a served GLQ checkpoint.

Every other task here is single-turn: one prompt, one completion, score it. This one asks
whether the model can *drive a terminal to finish a job*, where errors compound across turns
instead of being scored independently — the failure mode a multiple-choice benchmark cannot
see.

Two moving parts:

* the **host**, where `vllm serve --quantization glq` and `harbor run` both live in the
  serving venv. Harbor is an orchestrator — its dependencies are pydantic/typer/litellm/
  fastapi, with no torch, vllm or transformers (those appear only under optional extras) —
  so it installs alongside vLLM without disturbing it. Verified by dry run: pydantic and
  httpx already satisfied and unchanged. (LiveCodeBench is the opposite case and genuinely
  needs its own venv: Python 3.11 against our 3.12, plus its own heavy stack.)
* the **task container**, which runs pi and reaches the host server via
  `benchmarks/harbor_pi_glq.py`.

``kind="throughput"``: this owns a server and subprocesses, so it must not join the quality
tasks' shared in-process engine.

Cost is not comparable to the other tasks. Each task is a Docker rollout with many
sequential LLM calls, so wall-clock scales with tasks x attempts x turns. Use
``n_tasks`` while iterating — but note that a handful of tasks is a capability SMOKE, not a
score: 66 pass/fail tasks already has a worse power problem than MMLU-Pro, so a short run can
show that a model drives a terminal at all and cannot rank two models.

``max_model_len`` defaults small enough for any model and is usually worth raising. A thinking
model's card is the authority: Qwen3.8-Flash-Next asks for up to 262,144 reasoning tokens, and
output-token-exceeded is a named error mode of TB 4.0 — a window chosen for the harness rather
than the model shows up as truncated rollouts, which score like incapability.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
import urllib.error
import urllib.request

from glq.tooling import sampling_serve_args, tool_serve_args

from ..artifact_sync import DEFAULT_INTERVAL, ArtifactSync

from ..record import BenchmarkResult, ServingMeta, ThroughputResult

#: Terminal-Bench is a **continuous** benchmark: versions are tags on one repo, addressed
#: `name@version` (harbor 0.24.0 documents --dataset as "Dataset name@version"). 2.0 had its own
#: `terminal-bench-2` repo, which is why this used to be a bare path.
#:
#: Pinned, never floating on `main`: 4.0 removed 8 of 3.0's 74 tasks — saturated, refusal-prone
#: or publicly solved — and revised 20, so a floating id makes two recorded numbers
#: incomparable without saying so.
_DATASET = "terminal-bench/terminal-bench@4.0.0"

#: Concurrent sequences for the server behind the rollouts. NOT vLLM's default of 1024: on a
#: hybrid-GDN architecture every decode slot reserves a Mamba cache block before any request
#: exists, and 1024 has been measured refusing startup outright on a 96 GB card
#: ("max_num_seqs (1024) exceeds available Mamba cache blocks (399)"). 8 covers the rollouts
#: harbor runs concurrently against one server while keeping that cache and the per-step logits
#: buffer small.
_MAX_NUM_SEQS = 8

#: Where a custom agent goes on harbor's command line. `-a/--agent` takes either a builtin name
#: or a `module.path:ClassName`; the `--agent-import-path` flag this used to pass was removed
#: and is now rejected at argument parsing, before any container starts.
_AGENT = "benchmarks.harbor_pi_glq:PiGLQAgent"

#: The tool/reasoning flags `serve_command` decides for itself, so the rest of a family's serve
#: args can be passed through without duplicating them.
_TOOL_FLAGS = ("--enable-auto-tool-choice", "--tool-call-parser", "--reasoning-parser")


def _family_extras(family: list[str]) -> list[str]:
    """A family's serve flags minus the tool/reasoning trio decided below.

    This used to pull the two parser *values* out of `tool_serve_args` and discard everything
    else, which silently dropped `--language-model-only` — and without that a multimodal
    checkpoint loads vision and audio towers a terminal agent never touches: VRAM spent on
    weights nothing reads, and on some architectures a crash on a `<|video|>` placeholder.
    """
    out: list[str] = []
    skip = 0
    for tok in family:
        if skip:
            skip -= 1
            continue
        if tok in _TOOL_FLAGS:
            # --enable-auto-tool-choice is a bare switch; the other two take a value.
            skip = 0 if tok == "--enable-auto-tool-choice" else 1
            continue
        out.append(tok)
    return out
# Address of the host as seen from inside a container. The docker0 bridge is the portable
# answer on Linux; host.docker.internal needs an explicit --add-host and is a Docker Desktop
# idiom. Overridable because this is exactly the kind of thing that differs per box.
_DEFAULT_HOST_IP = "172.17.0.1"


def _harbor_cli() -> str:
    """The harbor CLI from this interpreter's venv, falling back to PATH."""
    cand = os.path.join(os.path.dirname(os.sys.executable), "harbor")
    if os.path.exists(cand):
        return cand
    found = shutil.which("harbor")
    if found:
        return found
    raise RuntimeError("`harbor` not found — install it into the serving venv with "
                       "`pip install harbor` (it is orchestration-only: no torch/vllm "
                       "deps, so it does not disturb the serving stack).")


def _wait_healthy(port: int, timeout_s: int, proc) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm serve exited early (rc={proc.returncode}) — see log")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(5)
    raise RuntimeError(f"vllm serve not healthy after {timeout_s}s at {url}")


def _parse_result(job_dir: str) -> tuple[float, dict]:
    """Score from harbor's own result.json.

    Shape (confirmed against a real oracle run, not inferred):
      stats.evals["<agent>__<dataset>"].metrics[0].mean  -> reward mean
                                       .n_trials/.n_errors, .pass_at_k
    """
    path = os.path.join(job_dir, "result.json")
    with open(path) as fh:
        doc = json.load(fh)
    evals = (doc.get("stats") or {}).get("evals") or {}
    if not evals:
        raise RuntimeError(f"no evals block in {path}")
    key, ev = next(iter(evals.items()))
    metrics = ev.get("metrics") or [{}]
    mean = metrics[0].get("mean")
    if mean is None:
        raise RuntimeError(f"no mean metric in {path} for {key}")
    return float(mean), {"eval_key": key, "n_trials": ev.get("n_trials"),
                         "n_errors": ev.get("n_errors"),
                         "pass_at_k": ev.get("pass_at_k"),
                         "exception_stats": ev.get("exception_stats"),
                         "n_input_tokens": (doc.get("stats") or {}).get("n_input_tokens"),
                         "n_output_tokens": (doc.get("stats") or {}).get("n_output_tokens")}


def serve_command(model: str, quant: str | None, config: dict, port: int,
                  served_id: str) -> list[str]:
    """The `vllm serve` argv for an agentic run.

    Separate from run() so the flags an agent depends on are assertable without starting a
    server — the tool-calling pair in particular is invisible until a rollout 400s.
    """
    vllm = shutil.which("vllm") or os.path.join(os.path.dirname(os.sys.executable), "vllm")
    cmd = [vllm, "serve", model, "--port", str(port),
           "--served-model-name", served_id,
           "--max-model-len", str(int(config.get("max_model_len", 32768))),
           # Explicit, because vLLM's 1024 default cannot start a GDN hybrid. See _MAX_NUM_SEQS.
           "--max-num-seqs", str(int(config.get("max_num_seqs", _MAX_NUM_SEQS)))]
    # Tool calling is not optional here: an agent that cannot call tools cannot touch the
    # terminal, and vLLM rejects pi's `tool_choice: "auto"` with a 400 unless both flags are
    # set. The parser is per model family, and the family table in glq.tooling is the one
    # source for it — this used to default to a hardcoded `hermes`, which reads
    # <tool_call>{...}</tool_call> and therefore mis-parses every published Qwen (their
    # templates emit <function=.../<parameter=... XML instead). That failure is silent: the
    # rollout completes and scores 0.0, which reads as a bad model.
    family = tool_serve_args(model) or []
    if "tool_call_parser" in config:
        parser = config["tool_call_parser"]
        reasoning = config.get("reasoning_parser")
    elif family:
        parser = family[family.index("--tool-call-parser") + 1]
        reasoning = (family[family.index("--reasoning-parser") + 1]
                     if "--reasoning-parser" in family else config.get("reasoning_parser"))
    else:
        # Unknown family: hermes is what this did before, and tool flags with a possibly
        # wrong parser still beat no tool flags, which 400s on the agent's first turn.
        parser, reasoning = "hermes", config.get("reasoning_parser")
    if parser:
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", parser]
    # Off unless the family asks for it or the config does, because a mismatched reasoning
    # parser mangles output. Needed for a thinking model: without it the <think> block
    # arrives inside `content`, where the tool-call parser has to look past it.
    if reasoning:
        cmd += ["--reasoning-parser", reasoning]
    cmd += _family_extras(family)
    # The model's own recommended sampling, pinned SERVER-side because the agent cannot send
    # it: `top_k` is not an OpenAI field and pi has no slot for it, so a top_k the server does
    # not pin is one that never applies. Same helper glq-code uses, so an agentic bench run and
    # an interactive coding session sample identically.
    cmd += sampling_serve_args(model) or []
    if quant and quant not in ("none", "bf16"):
        cmd += ["--quantization", quant]
    return cmd


def job_name() -> str:
    """A unique directory name for this run's harbor job.

    Named rather than discovered. `run()` used to find its result by diffing `jobs_dir` and
    taking `sorted(new)[-1]`, which breaks the moment a second harbor job lands a
    later-sorting timestamped directory there — and it cannot support mirroring artifacts at
    all, because the sync target would only be known once the run had finished.

    harbor's own default is a bare timestamp, which is exactly what collides.
    """
    return f"glq-tb-{uuid.uuid4().hex[:12]}"


def artifact_prefix(model: str, name: str) -> str:
    """S3 key prefix for one job's artifacts: `terminal_bench/<model>/<job-name>`.

    The repo id keeps its slash, so an `aws s3 ls` listing reads the way the Hub does. Path
    separators are the only thing worth sanitising — a `..` or a leading `/` in a key is
    legal in S3 and merely confusing, but it makes a later `aws s3 cp` to a local tree
    surprising.
    """
    safe = "/".join(p for p in str(model).split("/") if p not in ("", ".", ".."))
    return f"terminal_bench/{safe}/{name}".strip("/")


def harbor_command(harbor: str, *, dataset: str, served_id: str, n_attempts: int,
                   n_tasks, host_ip: str, jobs_dir: str, job_name: str = None,
                   n_concurrent=None) -> list[str]:
    """The `harbor run` argv.

    Separate from run() for the same reason `serve_command` is: a flag the agent depends on
    should be assertable without starting Docker. Skipping that is how the old
    `--agent-import-path` spelling survived harbor's removal of it — the oracle leg runs a
    builtin agent, so it exercises none of this.
    """
    cmd = [harbor, "run", "-d", dataset,
           # A custom agent travels on -a, as `module.path:ClassName`; see _AGENT.
           "-a", _AGENT,
           # Split by Pi.run() into `--provider glq --model <served-id>`, which is why the
           # provider written into models.json is named glq.
           "-m", f"glq/{served_id}", "-k", str(int(n_attempts)),
           # Only consulted by tasks that declare network_mode="allowlist"; harmless
           # otherwise, and without it those tasks cannot see the model at all.
           "--allow-agent-host", host_ip,
           "--jobs-dir", jobs_dir]
    if job_name:
        # So the job directory is known before harbor starts: it is both the mirror source
        # and the result path. See `job_name`.
        cmd += ["--job-name", job_name]
    if n_tasks:
        cmd += ["-l", str(int(n_tasks))]
    # One server feeds every rollout, so concurrency here is free throughput: the bottleneck
    # is container setup and agent think-time, not the GPU.
    if n_concurrent:
        cmd += ["-n", str(int(n_concurrent))]
    return cmd


def run(ctx, config: dict):
    harbor = _harbor_cli()
    dataset = config.get("dataset", _DATASET)
    n_attempts = int(config.get("n_attempts", 1))
    n_tasks = config.get("n_tasks")
    port = int(config.get("port", 8000))
    host_ip = config.get("host_ip") or os.environ.get("GLQ_DOCKER_HOST_IP",
                                                      _DEFAULT_HOST_IP)
    served_id = config.get("served_id") or "glq-model"
    serve_timeout = int(config.get("serve_timeout", 1800))
    run_timeout = int(config.get("run_timeout", 86400))
    jobs_dir = config.get("jobs_dir") or os.environ.get(
        "GLQ_HARBOR_JOBS_DIR", "/opt/dlami/nvme/harbor_jobs")
    # Named up front: the job directory is the artifact-mirror source, so it cannot be
    # something we only learn once harbor has finished. See `job_name`.
    name = config.get("job_name") or job_name()
    job_dir = os.path.join(jobs_dir, name)
    # A spot box loses everything not mirrored. Resolves config -> a bench-specific var ->
    # GLQ_RESUME_BUCKET, the last because `infra/setup.sh.tftpl` already exports that on
    # every provisioned box, so no box-side plumbing is needed. Absent => inert.
    bucket = (config.get("artifacts_bucket")
              or os.environ.get("GLQ_BENCH_ARTIFACTS_BUCKET")
              or os.environ.get("GLQ_RESUME_BUCKET") or None)

    serve_cmd = serve_command(ctx.model, ctx.quant, config, port, served_id)

    log_path = os.path.join(jobs_dir, "vllm_serve.log")
    os.makedirs(jobs_dir, exist_ok=True)

    proc = None
    try:
        with open(log_path, "w") as log:
            proc = subprocess.Popen(serve_cmd, stdout=log, stderr=subprocess.STDOUT)
            _wait_healthy(port, serve_timeout, proc)

        cmd = harbor_command(harbor, dataset=dataset, served_id=served_id,
                             n_attempts=n_attempts, n_tasks=n_tasks, host_ip=host_ip,
                             jobs_dir=jobs_dir, job_name=name,
                             n_concurrent=config.get("n_concurrent"))

        env = dict(os.environ)
        env["GLQ_VLLM_BASE_URL"] = f"http://{host_ip}:{port}/v1"
        # `benchmarks.harbor_pi_glq` is a repo path, not an installed package, so it is only
        # importable if the repo root is on the path — harbor resolves a `-a module:Class`
        # agent with a plain import.
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [os.getcwd(), env.get("PYTHONPATH", "")]))
        t0 = time.time()
        # Mirrored WHILE harbor runs, not afterwards: a reclaim three hours into a six-hour
        # job is exactly the case a post-run upload cannot help with. Inert without a bucket.
        prefix = config.get("artifacts_prefix") or artifact_prefix(ctx.model, name)
        with ArtifactSync(local_dir=job_dir, bucket=bucket, prefix=prefix,
                          interval=float(config.get("sync_interval",
                                                    DEFAULT_INTERVAL)),
                          exclude=config.get("artifacts_exclude")) as mirror:
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=run_timeout, check=False, env=env)
        dt = time.time() - t0
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()

    if not os.path.isdir(job_dir):
        raise RuntimeError(f"harbor produced no job dir at {job_dir} "
                           f"(rc={res.returncode}). stderr tail: "
                           f"{(res.stderr or '')[-600:]}")
    score, extra = _parse_result(job_dir)

    # A low score with most trials errored is an integration failure, not a quality result,
    # and only the counts tell them apart — so they travel with the number.
    res_obj = BenchmarkResult(
        task=config.get("task_name", "terminal_bench"), metric="reward_mean", value=score,
        standardized=bool(config.get("standardized", False)),
        config={"dataset": dataset, "n_attempts": n_attempts, "n_tasks": n_tasks,
                "agent": "pi", "harness": "harbor", "served_id": served_id,
                "host_ip": host_ip, "job_dir": job_dir, "job_name": name,
                # So a record points at its own artifacts. A mirrored job nobody can find
                # later is not much better than one that was never mirrored.
                "artifacts_uri": mirror.uri if mirror.enabled else None},
        extra={**extra, "wall_s": round(dt, 1), "rc": res.returncode})
    tp = ThroughputResult(output_tok_s=None, batch=None, measure="agentic_rollout")
    ctx.standalone_serving = ServingMeta(runtime="vllm", command=" ".join(serve_cmd))
    return res_obj, tp
