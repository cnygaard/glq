"""Step 5 of the Arete pi-adapter plan: glq-code --pi-adapter end to end, on the real card or
emulating a smaller one. A copy of benchmarks/arete_24gb/tier24_launch.py (PR #202) with two
changes:

* TIER_VRAM_GIB=- (or unset) runs on the real card with nothing patched, so the full-card arm
  goes through the same probe and log lines as the emulated one.
* TIER_PROBE_MODEL names the served model the probe asks (default: the base model). With the
  adapter on, it is the adapter's served name, so TTFT, decode and the reasoning length are the
  adapter's.

The planner and the fit check see a card of TIER_VRAM_GIB and a host of TIER_RAM_GIB. The
`--gpu-memory-utilization` handed to vLLM is rescaled so the absolute budget on this card equals
the tier's. Host RAM is enforced by the cgroup cap in step5_run.sh.

    TIER_CMD=code TIER_VRAM_GIB=23.99 TIER_RAM_GIB=32 python step5_launch.py --model M [flags]
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

import glq.chat as chat
import glq.code as code
from glq.supervisor import VllmSupervisor

GIB = 2 ** 30
CMD = os.environ.get("TIER_CMD", "code")
_V = os.environ.get("TIER_VRAM_GIB", "-")
EMULATE = _V not in ("", "-")
VRAM = float(_V) if EMULATE else None
RAM = float(os.environ.get("TIER_RAM_GIB") or 0) or None
PROBE_MODEL = os.environ.get("TIER_PROBE_MODEL") or None
PLAN_ONLY = os.environ.get("TIER_PLAN_ONLY") == "1"
PROBE_REPEATS = int(os.environ.get("TIER_PROBE_REPEATS", "3"))
PROBE_TOKENS = int(os.environ.get("TIER_PROBE_TOKENS", "256"))
REAL = int(subprocess.check_output(
    ["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
     "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True).strip()) * 2 ** 20
#: vLLM's own default below 70 GiB of card (EngineArgs._set_default_args).
BATCHED_TOKENS = 2048 if EMULATE and VRAM < 70 else None

if EMULATE:
    for _mod in (chat, code):
        _mod._vram_bytes = lambda: int(VRAM * GIB)
        _mod._ram_bytes = lambda: int(RAM * GIB)
        _mod._available_ram_bytes = lambda: int(RAM * GIB * 0.9)
        _mod._swap_bytes = lambda: 0


def emit(phase, **kw):
    print("TIER " + json.dumps({"phase": phase, "cmd": CMD, "vram": VRAM, "ram": RAM, **kw}),
          flush=True)


def _stream(base_url, model, messages, max_tokens, ignore_eos):
    body = json.dumps({
        "model": model, "stream": True, "temperature": 0, "max_tokens": max_tokens,
        "ignore_eos": ignore_eos, "stream_options": {"include_usage": True},
        "messages": messages}).encode()
    req = urllib.request.Request(f"{base_url}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first, n, content, reasoning = None, 0, [], []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
            ch = (d.get("choices") or [{}])[0].get("delta") or {}
            r_ = ch.get("reasoning_content") or ch.get("reasoning")
            if first is None and (ch.get("content") or r_):
                first = time.perf_counter()
            if ch.get("content"):
                content.append(ch["content"])
            if r_:
                reasoning.append(r_)
    t1 = time.perf_counter()
    return dict(ttft_s=round((first or t1) - t0, 3), tokens=n,
                decode_tps=round((n - 1) / (t1 - first), 2) if first and n > 1 else None,
                content="".join(content), reasoning_chars=len("".join(reasoning)))


def _probe(base_url, model):
    """TTFT and decode for one stream (temperature 0, ignore_eos), then one known answer."""
    story = [{"role": "user", "content": "Write a short story about a lighthouse keeper who "
                                         "learns to repair clocks."}]
    runs = [_stream(base_url, model, story, PROBE_TOKENS, True)
            for _ in range(1 + PROBE_REPEATS)]                      # the first is warm-up
    for r in runs:
        r.pop("content")
    q = [{"role": "user", "content": "What is the capital of France? Answer with one word."}]
    answer = _stream(base_url, model, q, 4096, False)
    return runs[1:], runs[0], answer


def _vllm_error():
    """The root-cause `...Error: ...` line in ~/.glq/vllm.log, skipping vLLM's wrappers."""
    try:
        lines = open(os.path.expanduser("~/.glq/vllm.log"), errors="replace").read().splitlines()
    except OSError as e:
        return f"no vllm.log: {e}"
    wrappers = ("See root cause above", "exited with status", "Engine core initialization failed")
    errs = [f"{m.group(1)}: {m.group(2)}"[:400] for l in lines
            if (m := re.search(r"\b(\w*(?:Error|Exception)): (.*)", l))
            and not l.lstrip().startswith("File ") and not any(w in l for w in wrappers)]
    return errs[-1] if errs else None


class _Tier(VllmSupervisor):
    def __init__(self, **kw):
        super().__init__(**kw)
        emit("plan", model=self.model, window=self.max_model_len,
             offload_gib=self.expert_offload_gib, arena_gib=float(self.expert_arena_gib or 0),
             spec_tokens=self.spec_tokens, util=self.gpu_memory_utilization,
             util_here=self._util_here(), batched_tokens=BATCHED_TOKENS or "vllm default",
             max_num_seqs=getattr(self, "max_num_seqs", None))
        if PLAN_ONLY:
            raise SystemExit(0)

    def _util_here(self):
        u = self.gpu_memory_utilization
        if u is None or not EMULATE:
            return u
        return round(float(u) * VRAM * GIB / REAL, 4)

    def argv(self):
        a = super().argv()
        if EMULATE and "--gpu-memory-utilization" in a:
            a[a.index("--gpu-memory-utilization") + 1] = f"{self._util_here():.4f}"
        if BATCHED_TOKENS and "--max-num-batched-tokens" not in a:
            a += ["--max-num-batched-tokens", str(BATCHED_TOKENS)]
        emit("argv", argv=a)
        return a

    def start(self):
        t0 = time.perf_counter()
        try:
            ok = super().start()
        except Exception as e:                    # vLLM refused or exited during startup
            emit("failed", after_s=round(time.perf_counter() - t0, 1),
                 error=f"{type(e).__name__}: {str(e).splitlines()[0][:200]}",
                 vllm=_vllm_error())
            raise
        emit("ready", ok=bool(ok), ready_s=round(time.perf_counter() - t0, 1))
        if ok and self.serve:
            try:
                runs, warm, answer = _probe(self.base_url, PROBE_MODEL or self.model)
                emit("probe", runs=runs, warmup=warm, answer=answer)
            except Exception as e:                            # noqa: BLE001
                emit("probe", error=f"{type(e).__name__}: {e}")
        if CMD == "chat":                          # measured; do not open the UI
            self.stop()
            emit("exit", rc=0)
            raise SystemExit(0)
        return ok


if __name__ == "__main__":
    mod = chat if CMD == "chat" else code
    mod.VllmSupervisor = _Tier
    rc = mod.main(sys.argv[1:])
    emit("exit", rc=rc)
    sys.exit(rc)
