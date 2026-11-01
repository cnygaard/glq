# Super-VL 3 bpw on a 24 GB budget: speculative decoding + GLQ expert arena, AIPerf coding

Model `xv0y5ncu/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-3bpw`, draft
`RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B.dspark` (bf16, 5-layer sliding-window Qwen3,
loaded without remote code). Plan: `~/.claude/plans/make-arena-on-the-linked-scroll.md`.

**Setup, every arm:** AWS g7 RTX PRO 4500 Blackwell (31.38 GiB, PCIe 5 x16) capped at
`--gpu-memory-utilization 0.70` = 21.96 GiB (0.93 of a 24 GB card), glq 0.8.33, vLLM 0.32.0,
transformers 5.19.0 overlay, bf16, `--max-num-seqs 1`, all routed experts (39.52 GiB) on host via
UVA, `GLQ_MOE_ARENA_MIN/MAX_TOKENS = 1 + k`. AIPerf 0.11.0, SPEED-Bench coding, concurrency 1,
streaming chat, temperature 1.0 / top_p 0.95 sent by the client (the GLQ repo has no
generation_config.json), thinking on (template default), seed 42 so every arm gets the same prompts.

## Stage 3 (final): 10 requests, osl 4096, window 16,384 — replacement box, same GPU type

| arm | arena | weights + arena + draft | KV | ITL | output tok/s per request (min–max) | aggregate tok/s | acceptance (tokens per step) |
|---|--:|--:|--:|--:|--:|--:|--:|
| A0 no draft | 7.95 GiB, 103 slots | 12.72 GiB | 6.02 GiB | 49.62 ms | 20.15 (19.88–20.38) | 17.53 | – |
| A2 DSpark k=4 | 8.95 GiB, 116 slots | 16.34 GiB | 1.97 GiB | 31.96 ms | **31.50 (26.86–35.78), 1.56×** | 25.77 (1.47×) | 70.8% (3.83); per position 0.882 / 0.755 / 0.645 / 0.549 |
| A3 DSpark k=1, 1,048,576 window | 3.47 GiB, 45 slots | – | 7.5 GiB available | – | **did not start** | – | – |

- A0 and A2: 10/10 requests, 0 errors, 40 arena lines and 0 refusals each. TTFT averages
  11.4 s in both (0.9–36.6 s).
- Output lengths averaged 1,536 tokens (A0) and 1,873 (A2), max 2,433 / 4,004. No request hit
  the 4,096 limit, and sampling at temperature 1.0 makes lengths differ between arms. So compare
  the per-token rates (ITL, per-request tok/s), not request latency.
- A3: vLLM refused at start-up. One 1,048,576-token sequence needs 10.43 GiB of KV with the
  drafter loaded, but 7.5 GiB was available. vLLM's estimate of the largest window that fits is
  740,544 tokens. Without a drafter the 1M window fit (10.3 GiB of KV, ~8.2 GiB per 1M
  sequence; `../supervl3_offload_24gb/`).
- k=4 leaves 1.97 GiB of KV at this window (38,027 tokens, 2.32× concurrency at 16,384), against
  6.02 GiB with no draft. On a 24 GB card the deeper draft is paid for in context length.
- `aiperf speed-bench-report` keys rows by model name and category, so A0 and A2 (one model)
  collapse into one row showing A2's figures (`aiperf_report/`). Use the per-arm
  `profile_export_aiperf.json` to compare draft configurations.

## 524,288-token window (`L*_512k`): 10 requests, osl 4096 — replacement box

`scripts/spec_512k.sh`. Same setup as above except `--max-model-len 524288`.

| arm | arena | weights + arena + draft | KV (tokens, concurrency at 512k) | ITL | output tok/s per request (min–max) | aggregate tok/s | acceptance (tokens per step) |
|---|--:|--:|--:|--:|--:|--:|--:|
| L2 DSpark k=2 | 5.09 GiB, 66 slots | 12.42 GiB | 5.92 GiB (561,457, 1.07×) | 37.67 ms | **26.60 (24.15–27.84)** | 22.34 | 85.8% (2.72) |
| L1 DSpark k=1 | 3.47 GiB, 45 slots | 10.85 GiB | 7.51 GiB (723,830, 1.38×) | 43.72 ms | 22.88 (22.16–23.77) | 19.52 | 90.8% (1.91) |

- Both arms: 10/10 requests, 40 arena lines and 0 refusals. Acceptance is summed over every
  `SpecDecoding metrics` line in the serve log; for L2 that includes the needle requests below.
- Per-request rates match the 16,384-window arms (k=2 26.85–26.98, k=1 23.17), so on this
  workload the longer window does not slow decode. The prompts are short; see the needle for a
  long one.
- KV bytes needed per 524,288-token request with the drafter loaded: 5.92 GiB / 1.07 = 5.53 GiB
  (~11.3 KB/token), and 7.51 / 1.38 = 5.44 GiB for k=1.
- **Needle (L2, `scripts/needle_spec.py 500000`, thinking off, temperature 0, 256 tokens,
  `ignore_eos`):**

  | prompt tokens | TTFT | prefill tok/s | decode tok/s | acceptance (tokens per step) | needle found |
  |--:|--:|--:|--:|--:|:-:|
  | 15,746 | 114.5 s | 138 | 24.21 | 71.9% (2.44) | yes |
  | 493,572 | 2,424.5 s (40.4 min) | 204 | 14.31 | 71.4% (2.43) | yes |

  Decode at 493k is 41% slower than at 16k with the same acceptance. The attention over half a
  million cached tokens is the likely cause; it was not profiled.

## Depth re-screen on the replacement box (`S2b_*`): 5 requests, osl 1024, window 16,384

| arm | arena | weights + arena + draft | KV | output tok/s per request (min–max) | acceptance (tokens per step) | per position |
|---|--:|--:|--:|--:|--:|---|
| DSpark k=2 | 5.48 GiB | 12.73 GiB | 5.61 GiB | 26.85 (25.81–27.89) | 85.9% (2.72) | 0.905 / 0.813 |
| DSpark k=3 | 6.95 GiB | 14.30 GiB | 4.04 GiB | 29.17 (28.01–30.05) | 80.3% (3.41) | 0.903 / 0.804 / 0.703 |
| DSpark k=4 | 8.95 GiB | 16.34 GiB | 1.97 GiB | 31.98 (27.39–35.77) | 74.7% (3.99) | 0.910 / 0.786 / 0.689 / 0.604 |

k=2 reproduced the old box's 26.98 within 0.5%. `spec_resume.sh` picked k=4 for Stage 3: it was
more than 3% faster than k=2.

## Stage 2 (depth screening, first box): 5 requests, osl 1024, window 16,384

| arm | arena | weights + arena + draft | KV | ITL | output tok/s per request | aggregate tok/s | accepted / drafted (tokens per step) |
|---|--:|--:|--:|--:|--:|--:|--:|
| no draft | 7.95 GiB, 103 slots | 12.72 GiB | 6.02 GiB | 49.68 ms | 20.13 (19.87–20.43) | 16.46 | – |
| DSpark k=1 | 3.94 GiB, 51 slots | 11.32 GiB | 7.04 GiB | 43.18 ms | 23.17 (22.72–23.71), 1.15× | 18.62 | 92.9% (1.93) |
| DSpark k=2 | 5.48 GiB, 71 slots | 12.73 GiB | 5.61 GiB | 37.10 ms | **26.98 (26.00–27.97), 1.34×** | 20.35 | 86.7% (2.73); pos 1 92.1%, pos 2 81.4% |

- Every arm: 40 arena decision lines, 0 refusals, 5/5 requests, 0 errors.
- TTFT averages ~10.6 s in all three (min ~0.9 s, max ~31 s on the same prompt each time). That
  is the slow prefill under expert offload, not a speculation effect.
- Acceptance comes from aiperf's `server_metrics_export.json`, profiling phase only.
- n=5 at temperature 1.0 gives a feel only and cannot rank close configurations.
- k=4 was stopped here when the first box was shut down (`logs/spec_serve_S2_ds4.log`). It was
  re-screened on the replacement box above. k=4 with adaptive verification was never run.

## Stage 1 (run_model.py, B=1, 256 tokens, 3 repeats, window 16,384)

- MTP k=1: **fails to load**. vLLM builds the draft from the VL wrapper's text config, which has no
  quantization_config, so it reads `quantize_config.json`. That file lacks variant / layer_bpw,
  so GLQ claims the bf16 MTP MoE and refuses it (`GLQ trellis MoE supports only the 3inst
  variant`). Fix not implemented yet.
- DSpark k=4: RUN_OK, 21.735 tok/s (no-draft arena reference 20.83), acceptance 40.4%,
  per-position 0.709 / 0.456 / 0.177 / 0.101 on run_model's generic prompt. On coding the same
  k=4 accepts 0.88 / 0.76 / 0.65 / 0.55 and runs 1.56× (Stage 3). A depth judged on a generic
  prompt would have been ruled out wrongly.

## Files

- `scripts/`: `setup_box.sh` (fresh box), `make_speed_bench_coding.py` (writes
  `speed_bench_coding.jsonl`, the 70 prompts used), `spec_serve_aiperf.sh` (one arm),
  `spec_stage{0,1,2,3}.sh`, `needle_spec.py`.
- Raw artifacts are not in this repository and go to glq-benchmarks: `aiperf/S2_*`
  (`profile_export_aiperf.json`, `server_metrics_export.json`, …), and `logs/` with the serve
  and aiperf logs per arm. `logs/failed_first_attempt/` is the first Stage 2 run, where every
  aiperf call failed on config (see below).

## aiperf 0.11.0 traps

- `--custom-dataset-type speed-bench-coding` needs `--input-file`. It does not download anything.
  Without it the config fails with pydantic `extra_forbidden` on `synthetic.format`.
- The HF parquet stores `turns`; aiperf wants `messages`. 10 of 80 coding rows are placeholders
  to be fetched from their source with specdec_bench, so 70 rows are usable.
- `--tokenizer <local snapshot path>` fails with HFValidationError. Pass the repo id with
  `HF_HUB_OFFLINE=1`.
- aiperf needs Python < 3.14, so it lives in its own venv, not in a glq extra.

## Resume

1. Copy `install.sh`, `scripts/*` to `/opt/dlami/nvme/bench/`, put the read token in
   `/opt/dlami/nvme/.hfreadtok` (mode 600), run `setup_box.sh`.
2. `bash bench/spec_resume.sh <setup PID>` does the rest unattended. It waits for setup,
   re-screens DSpark k=2/3/4 on coding (`S2b_*`, same box so the depths compare), and keeps k=2
   unless a deeper draft is >3% faster per request. Then it runs Stage 3 via `spec_stage3.sh`:
   10 requests at osl 4096, comparing no draft, the picked depth, and DSpark k=1 at the
   1,048,576 window with a 3.5 GiB arena.
   Then `aiperf speed-bench-report /opt/dlami/nvme/aiperf/A* --metric throughput --format both`.
   Started 2026-10-30 16:19 UTC on the replacement box (glq 0.8.33, vLLM 0.32.0, tf 5.19.0,
   torch 2.13.0+cu130). The model revision moved f554c116 → 5eda4b5b, but only README.md changed.
4. Stop a server by its exact PID. SIGTERM sent during engine start-up can be dropped, so re-send
   it after `Application startup complete`. Stop an arm wrapper before its server, or it starts
   aiperf against the server it is waiting on.
