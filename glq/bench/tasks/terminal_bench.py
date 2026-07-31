"""Terminal-Bench 2 via harbor — agentic terminal ability of a served GLQ checkpoint.

Every other task here is single-turn: one prompt, one completion, score it. This one asks
whether the model can *drive a terminal to finish a job*, where errors compound across turns
instead of being scored independently — the failure mode a multiple-choice benchmark cannot
see.

Three moving parts, in three environments, for reasons that are all version isolation:

* the **serving venv** runs `vllm serve --quantization glq` on the host (that is where glq
  and its CUDA extension live);
* the **harbor venv** (`GLQ_HARBOR_HOME`, Python ≥3.12) runs the benchmark — harbor pins its
  own dependency set and must not resolve into the serving venv;
* the **task container** runs pi, reaching the host server via
  `benchmarks/harbor_pi_glq.py`.

``kind="throughput"``: this owns a server and subprocesses, so it must not join the quality
tasks' shared in-process engine.

Cost is not comparable to the other tasks. Each task is a Docker rollout with many
sequential LLM calls, so wall-clock scales with tasks x attempts x turns. Use
``n_tasks`` while iterating.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request

from ..record import BenchmarkResult, ServingMeta, ThroughputResult

_DATASET = "terminal-bench/terminal-bench-2"
# Address of the host as seen from inside a container. The docker0 bridge is the portable
# answer on Linux; host.docker.internal needs an explicit --add-host and is a Docker Desktop
# idiom. Overridable because this is exactly the kind of thing that differs per box.
_DEFAULT_HOST_IP = "172.17.0.1"


def _harbor_python(home: str) -> str:
    py = os.path.join(home, ".venv", "bin", "harbor")
    if not os.path.exists(py):
        raise RuntimeError(
            f"no harbor CLI at {py}. Create it with `uv venv --python 3.12 && "
            f"uv pip install harbor` and point GLQ_HARBOR_HOME at that directory. It is "
            f"deliberately separate: harbor's pins would move vllm/transformers under "
            f"every other benchmark.")
    return py


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


def run(ctx, config: dict):
    home = config.get("harbor_home") or os.environ.get(
        "GLQ_HARBOR_HOME", "/opt/dlami/nvme/harbor-env")
    harbor = _harbor_python(home)
    dataset = config.get("dataset", _DATASET)
    n_attempts = int(config.get("n_attempts", 1))
    n_tasks = config.get("n_tasks")
    port = int(config.get("port", 8000))
    host_ip = config.get("host_ip") or os.environ.get("GLQ_DOCKER_HOST_IP",
                                                      _DEFAULT_HOST_IP)
    served_id = config.get("served_id") or "glq-model"
    serve_timeout = int(config.get("serve_timeout", 1800))
    run_timeout = int(config.get("run_timeout", 86400))
    jobs_dir = config.get("jobs_dir") or os.path.join(home, "jobs")

    vllm = shutil.which("vllm") or os.path.join(os.path.dirname(os.sys.executable), "vllm")
    serve_cmd = [vllm, "serve", ctx.model, "--port", str(port),
                 "--served-model-name", served_id,
                 "--max-model-len", str(int(config.get("max_model_len", 32768)))]
    if ctx.quant and ctx.quant not in ("none", "bf16"):
        serve_cmd += ["--quantization", ctx.quant]

    log_path = os.path.join(jobs_dir, "vllm_serve.log")
    os.makedirs(jobs_dir, exist_ok=True)
    before = set(os.listdir(jobs_dir))

    proc = None
    try:
        with open(log_path, "w") as log:
            proc = subprocess.Popen(serve_cmd, stdout=log, stderr=subprocess.STDOUT)
            _wait_healthy(port, serve_timeout, proc)

        cmd = [harbor, "run", "-d", dataset,
               "--agent-import-path", "benchmarks.harbor_pi_glq:PiGLQAgent",
               "-m", f"glq/{served_id}", "-k", str(n_attempts),
               # Only consulted by tasks that declare network_mode="allowlist"; harmless
               # otherwise, and without it those tasks cannot see the model at all.
               "--allow-agent-host", host_ip,
               "--jobs-dir", jobs_dir]
        if n_tasks:
            cmd += ["-l", str(int(n_tasks))]

        env = dict(os.environ)
        env["GLQ_VLLM_BASE_URL"] = f"http://{host_ip}:{port}/v1"
        # The agent module lives in this repo, not in harbor's venv.
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [os.getcwd(), env.get("PYTHONPATH", "")]))
        t0 = time.time()
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

    new = [d for d in os.listdir(jobs_dir) if d not in before
           and os.path.isdir(os.path.join(jobs_dir, d))]
    if not new:
        raise RuntimeError(f"harbor produced no job dir (rc={res.returncode}). "
                           f"stderr tail: {(res.stderr or '')[-600:]}")
    job_dir = os.path.join(jobs_dir, sorted(new)[-1])
    score, extra = _parse_result(job_dir)

    # A low score with most trials errored is an integration failure, not a quality result,
    # and only the counts tell them apart — so they travel with the number.
    res_obj = BenchmarkResult(
        task=config.get("task_name", "terminal_bench"), metric="reward_mean", value=score,
        standardized=bool(config.get("standardized", False)),
        config={"dataset": dataset, "n_attempts": n_attempts, "n_tasks": n_tasks,
                "agent": "pi", "harness": "harbor", "served_id": served_id,
                "host_ip": host_ip, "job_dir": job_dir},
        extra={**extra, "wall_s": round(dt, 1), "rc": res.returncode})
    tp = ThroughputResult(output_tok_s=None, batch=None, measure="agentic_rollout")
    ctx.standalone_serving = ServingMeta(runtime="vllm", command=" ".join(serve_cmd))
    return res_obj, tp
