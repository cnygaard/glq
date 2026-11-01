"""Run glq-code as it would run on a smaller machine, on one big card.

The planner sees a card of TIER_VRAM_GIB and a host of TIER_RAM_GIB, so the window, offload,
arena, DSpark depth and the fit check are exactly that tier's. The `--gpu-memory-utilization`
handed to vLLM is rescaled, so the absolute budget on this card equals the tier's, and below
70 GiB `--max-num-batched-tokens 2048` is passed, the default vLLM picks on a card that size
(see `BATCHED_TOKENS`). Host RAM is enforced separately by the cgroup cap in `tier_run.sh`.

Once the server is ready, a probe measures TTFT and single-stream decode, scrapes `/metrics`,
and then glq-code goes on to its pi prompt as usual. One JSON line per phase is printed with a
`TIER ` prefix.

    TIER_VRAM_GIB=8 TIER_RAM_GIB=16 python tier_launch.py --model M [glq-code flags] -- -p "..."
    TIER_PLAN_ONLY=1 ...    print the plan and exit, start nothing
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request

import glq.code as code
from glq.supervisor import VllmSupervisor, child_env

GIB = 2 ** 30
VRAM = float(os.environ["TIER_VRAM_GIB"])
RAM = float(os.environ["TIER_RAM_GIB"])
PLAN_ONLY = os.environ.get("TIER_PLAN_ONLY") == "1"
PROBE_REPEATS = int(os.environ.get("TIER_PROBE_REPEATS", "3"))
PROBE_TOKENS = int(os.environ.get("TIER_PROBE_TOKENS", "256"))
REAL = int(subprocess.check_output(
    ["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
     "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True).strip()) * 2 ** 20

#: vLLM sizes its batch from the physical card (`EngineArgs._set_default_args`, 0.32.0): the
#: API server gets `max_num_batched_tokens` 8192 at >= 70 GiB and 2048 below. Left alone,
#: every tier here would run with a 96 GB card's 8192, which costs more activation memory
#: and, on gemma-4, ~3 GiB more sliding-window KV per request than a real card of the tier.
BATCHED_TOKENS = 2048 if VRAM < 70 else None
#: vLLM flags to try before glq passes them itself, e.g. `--language-model-only` on gemma-4.
EXTRA_VLLM_ARGS = os.environ.get("TIER_VLLM_ARGS", "").split()
#: The rest of vLLM's rule (`arg_utils.py`, "multimodal prefix-LM models (e.g., Gemma 4)"):
#: when the batch is NOT set explicitly, it is raised to the largest single multimodal item,
#: because these models prefill an image atomically. Passing 2048 explicitly skips that and
#: vLLM refuses to start. 2496 is what vLLM reported for gemma-4 26B-A4B and 31B in run 2
#: ("max_tokens_per_mm_item (2496)"); E4B started at 2048, so its item fits.
MM_ITEM_FLOOR = {"gemma-4-26b": 2496, "gemma-4-31b": 2496}


def _batched_tokens(model, argv):
    if not BATCHED_TOKENS:
        return None
    if "--language-model-only" in argv:          # every modality off: no item to fit
        return BATCHED_TOKENS
    name = (model or "").lower()
    return max([BATCHED_TOKENS] + [v for k, v in MM_ITEM_FLOOR.items() if k in name])

code._vram_bytes = lambda: int(VRAM * GIB)
code._ram_bytes = lambda: int(RAM * GIB)
code._available_ram_bytes = lambda: int(RAM * GIB * 0.9)
code._swap_bytes = lambda: 0


def emit(phase, **kw):
    print("TIER " + json.dumps({"phase": phase, "vram": VRAM, "ram": RAM, **kw}), flush=True)


def _probe(base_url, model):
    """TTFT and decode for one stream: temperature 0, ignore_eos, PROBE_TOKENS tokens."""
    body = json.dumps({
        "model": model, "stream": True, "temperature": 0, "max_tokens": PROBE_TOKENS,
        "ignore_eos": True, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "Write a short story about a lighthouse "
                      "keeper who learns to repair clocks."}]}).encode()
    runs = []
    for _ in range(1 + PROBE_REPEATS):                        # the first is warm-up
        req = urllib.request.Request(f"{base_url}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        first = None
        n = 0
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:") or line == b"data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    n = d["usage"]["completion_tokens"]
                ch = (d.get("choices") or [{}])[0].get("delta") or {}
                if first is None and (ch.get("content") or ch.get("reasoning_content")
                                      or ch.get("reasoning")):
                    first = time.perf_counter()
        t1 = time.perf_counter()
        runs.append(dict(ttft_s=round((first or t1) - t0, 3), tokens=n,
                         decode_tps=round((n - 1) / (t1 - first), 2) if first and n > 1 else None))
    return runs[1:], runs[0]


def _metrics(base_url):
    root = base_url.rsplit("/v1", 1)[0]
    try:
        text = urllib.request.urlopen(f"{root}/metrics", timeout=10).read().decode()
    except Exception as e:                                    # noqa: BLE001
        return {"error": str(e)}
    keep = ("spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
            "spec_decode_num_accepted_tokens_total", "prefix_cache_hits_total",
            "prefix_cache_queries_total")
    out = {}
    for line in text.splitlines():
        for k in keep:
            if line.startswith("vllm:" + k):
                out[k] = out.get(k, 0) + float(line.rsplit(" ", 1)[1])
    return out


#: vLLM's own wrappers, printed after the error that caused them.
_WRAPPERS = ("See root cause above", "exited with status", "Engine core initialization failed")


def _vllm_error():
    """The root-cause `...Error: ...` line in ~/.glq/vllm.log, the reason vLLM refused.

    The last error line is vLLM's wrapper ("Engine core initialization failed. See root cause
    above"), which names nothing, so wrappers are skipped and the EngineCore log prefix dropped.
    """
    try:
        lines = open(os.path.expanduser("~/.glq/vllm.log"), errors="replace").read().splitlines()
    except OSError as e:
        return f"no vllm.log: {e}"
    errs = []
    for l in lines:
        m = re.search(r"\b(\w*(?:Error|Exception)): (.*)", l)
        if m and not l.lstrip().startswith("File ") and not any(w in l for w in _WRAPPERS):
            errs.append(f"{m.group(1)}: {m.group(2)}"[:400])
    return errs[-1] if errs else None


class _Tier(VllmSupervisor):
    def __init__(self, **kw):
        super().__init__(**kw)
        env = {k: v for k, v in child_env(
            device=self.device, expert_arena_gib=self.expert_arena_gib,
            max_num_seqs=self.step_tokens, min_step_tokens=self.min_step_tokens).items()
            if k.startswith("GLQ_MOE")}
        emit("plan", model=self.model, window=self.max_model_len,
             offload_gib=self.expert_offload_gib, arena_gib=float(self.expert_arena_gib or 0),
             spec_tokens=self.spec_tokens, util=self.gpu_memory_utilization,
             util_here=self._util_here(), resident_gib=round(self.resident_bytes / GIB, 3),
             pinned_gib=round(getattr(self, "pinned_host_bytes", 0) / GIB, 3),
             batched_tokens=_batched_tokens(self.model, EXTRA_VLLM_ARGS) or "vllm default",
             extra_vllm_args=EXTRA_VLLM_ARGS,
             note=self._spec_note, env=env)
        if PLAN_ONLY:
            raise SystemExit(0)

    def _util_here(self):
        u = self.gpu_memory_utilization
        return None if u is None else round(float(u) * VRAM * GIB / REAL, 4)

    def argv(self):
        a = super().argv()
        if "--gpu-memory-utilization" in a:
            i = a.index("--gpu-memory-utilization") + 1
            a[i] = f"{self._util_here():.4f}"
        a = a + [x for x in EXTRA_VLLM_ARGS if x not in a]
        n = _batched_tokens(self.model, a)
        if n and "--max-num-batched-tokens" not in a:
            a += ["--max-num-batched-tokens", str(n)]
        return a

    def start(self):
        t0 = time.perf_counter()
        try:
            ok = super().start()                  # True: started; False: attached
        except Exception as e:                    # vLLM refused or exited during startup
            emit("failed", after_s=round(time.perf_counter() - t0, 1),
                 error=f"{type(e).__name__}: {str(e).splitlines()[0][:200]}",
                 vllm=_vllm_error())
            raise
        self._ready = True
        emit("ready", ok=bool(ok), ready_s=round(time.perf_counter() - t0, 1))
        if ok and self.serve:
            try:
                runs, warm = _probe(self.base_url, self.model)
                emit("probe", runs=runs, warmup=warm)
            except Exception as e:                            # noqa: BLE001
                emit("probe", error=f"{type(e).__name__}: {e}")
        return ok

    def stop(self):
        # Only a server that came up has /metrics; one vLLM refused has nothing on the port.
        if self.serve and self.proc is not None and getattr(self, "_ready", False):
            emit("metrics", **_metrics(self.base_url))
        return super().stop()


code.VllmSupervisor = _Tier

if __name__ == "__main__":
    rc = code.main(sys.argv[1:])
    emit("exit", rc=rc)
    sys.exit(rc)
