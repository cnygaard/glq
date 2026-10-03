# AIME-2026 on Qwen3.8-Flash-Next-GLQ-3bpw, under PLE host-memory offload

Two records. `*_avg8_seqs240.jsonl` is the citable one (n=30, avg@8); `*_262k.jsonl` is a
10-problem smoke test kept **deliberately** — side by side they show why sizing the KV pool
from a subset's `mean_gen` is wrong (10,352 on the first 10 problems vs **26,464** across all
30, a 2.6x gap that inverted a conclusion about whether expert offload would help).

| record | value | n | avg_k | mean_gen | truncated | no_answer |
|---|--:|--:|--:|--:|--:|--:|
| `aime2026_flashnext_avg8_seqs240.jsonl` | **0.9958** (239/240) | 30 | 8 | 26,464 | 0 | 0 |
| `aime2026_flashnext_262k.jsonl` | 1.0000 (10/10) | 10 | 1 | 10,352 | 0 | 0 |

`solved_all=29/30`, `solved_any=30/30`, aggregate **209.1 tok/s**, wall-clock **8h26m**,
resident **49.53 GiB**, KV pool **1,045,613 tokens**.

## Exact reproduction

Measured 2026-10-03T06:15:32Z. Provenance from the record itself:

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GiB, PCIe 5 x16 |
| glq / vLLM / torch | **0.8.23** / **0.30.0** / **2.13.0** (cuda 13.0, driver 595.91.07) |
| transformers / triton / python | 5.17.0 / 3.7.1 / 3.12.3 |
| checkpoint | `xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4` (3.0 bpw avg, mixed 3-4, 77.56 GB on disk) |

### 1. Provision

```bash
# `bench` is required: it carries `datasets` (quality tasks) and `pandas` (decode_sweep).
# Without it `glq-bench run --tasks aime_2026` ends in ModuleNotFoundError: datasets.
bash install.sh --yes --components core,vllm,bench
```

### 2. Fetch the checkpoint

```bash
# Token from a mode-0600 file, never on a command line (it would land in shell history and ps).
HF_TOKEN=$(cat /opt/dlami/nvme/.hftok) HF_HOME=/opt/dlami/nvme/hf_cache \
  hf download xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4
```

### 3. Run — the exact command used

```bash
HF_TOKEN=$(cat /opt/dlami/nvme/.hftok) \
HF_HOME=/opt/dlami/nvme/hf_cache \
VLLM_PLE_CPU_OFFLOAD=1 \
PYTHONFAULTHANDLER=1 \
glq-bench run \
    --model xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4 \
    --quant glq \
    --tasks aime_2026 \
    --n 30 \
    --avg-k 8 \
    --budget 262144 \
    --max-model-len 262144 \
    --max-num-seqs 240 \
    --gpu-mem-util 0.90 \
    --task-config '{"temperature": 1.0, "top_p": 0.95, "top_k": 20}' \
    --out aime2026_flashnext_avg8_seqs240.jsonl
```

The smoke record used the same command with `--n 10 --avg-k 1 --max-num-seqs 8`.

### Equivalent serve line

`glq-bench` drives an in-process `LLM(...)`; the record captures the equivalent server form:

```bash
vllm serve xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4 \
    --quantization glq --gpu-memory-utilization 0.9 --max-num-seqs 240 \
    --max-model-len 262144 --dtype bfloat16 --trust-remote-code \
    --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}'
```

plus `compilation_config={"cudagraph_mode": "FULL",
"cudagraph_capture_sizes": [1,2,4,8,16,32]}`.

## Why each flag is what it is

- **`VLLM_PLE_CPU_OFFLOAD=1` is a precondition, not a tuning knob.** With it off, startup fails:
  *"To serve at least one request with the model's max seq len (262144), 6.55 GiB KV cache is
  [needed, more than available]"* — resident 73.3 GiB leaves nothing for KV. With it on,
  resident is 49.53 GiB and the KV pool is 1,045,613 tokens. (It is vLLM's default anyway.)
- **`--task-config` top_k=20 must be set.** The AIME task defaults to gemma-4's `top_k=64`;
  the Qwen3.8 card's thinking mode wants **20**. `min_p=0.0`, `presence_penalty=0.0` and
  `repetition_penalty=1.0` from the card are **not** settable here — `glq-bench`'s `sampling()`
  takes only temperature/top_p/top_k/seed/max_tokens and silently drops anything else — but
  vLLM's defaults already equal those three, so the run still matches the card.
- **No system message.** `system` defaults to `None`, which is what the thinking path needs.
- **`--budget 262144` is necessary, not generous.** At a 26,464 mean with a long tail, 32k
  would truncate and 65k (the task's own registry default) plausibly would too. Truncation
  came out at **0**.
- **`--max-num-seqs 240`** = 30 problems x avg@8, so the eval schedules in one wave. The hard
  ceiling is **685**, which vLLM reports if you overshoot: *"max_num_seqs (1024) exceeds
  available Mamba cache blocks (685). Each decode sequence requires one Mamba cache block"*.
  Exceeding the Mamba cap **refuses at startup**; exceeding KV only **queues**.

## What this result does and does not show

**Does:** 3 bpw GLQ with the PLE table in host memory does not degrade this model —
thinking engaged (mean_gen 26,464, far above the harness's 4,000 floor), zero truncations,
zero unparsed answers. It is also the first quality-under-offload measurement.

**Does not:** resolve a quantization delta. At 99.58% the score is near the ceiling, and no
AIME run at this accuracy could separate 3 bpw from bf16. No bf16 baseline is possible for this
checkpoint either — the base model is 335.3 GiB.

**Effective concurrency was ~39, not 240.** `1,045,613 / 26,464` is the real limit: KV, not
`max_num_seqs`. Raising `max_num_seqs` 8 -> 240 bought 72.0 -> 209.1 tok/s aggregate (**2.9x,
not 30x**). So adding `--cpu-offload-params experts` should help — it frees VRAM into the pool
that is actually binding.
