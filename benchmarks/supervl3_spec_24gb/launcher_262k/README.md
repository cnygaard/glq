# `glq-code --dspark` end to end, Super-VL 3 bpw, 262,144-token window

Branch `feat/dspark-launchers` (overlaid on an install.sh install: glq 0.8.33, vLLM 0.32.0,
transformers 5.19.0 overlay). Box: AWS g7, RTX PRO 4500 Blackwell (31.86 GiB, PCIe 5 x16),
60.3 GiB host RAM, 16 vCPU. The real launcher runs every arm, with one non-interactive pi prompt:
read `hello.py` and say what it prints (correct answer 63). `code_dspark.sh` is the wrapper, and
it scrapes `/metrics` every 10 s while the server is up. `plan_code.py` / `plan_vram.py` print
the plan without starting vLLM; `plan_vram.py` replaces the VRAM the planner sees.

| arm | flags | plan | vLLM | result |
|---|---|---|---|---|
| pin3, vision loaded (`*_pin3_mm.log`) | `--dspark 3` | window 262,144, 36 GiB experts on host, arena 6.80 GiB, util 0.92 | loaded 19.17 GiB, KV **1.41 GiB** < 3.05 needed | **refused to start** |
| pin3 (`*_pin3.log`) | `--dspark 3` (+ `--language-model-only` from the launcher) | same | loaded 17.61 GiB, KV 9.31 GiB = 798,915 tokens (3.05× 262,144) | EXIT 0, answer 63, 495 s wall, ready in 316 s |
| auto (`*_auto.log`) | `--dspark` | DSpark off; window 262,144, 36 GiB on host, arena 4 GiB, util 0.892 | loaded 12.26 GiB, KV 13.99 GiB = 1,623,420 tokens (6.19×) | EXIT 0, answer 63, 330 s wall, ready in 230 s |
| 24 GiB budget (`*_v24pin3.log`) | `--dspark 3 --max-model-len 262144 --cpu-offload-gb 40 --gpu-memory-utilization 0.6931` | arena 6.80 GiB, MIN/MAX tokens 4; util 0.694 = 22.11 GiB (0.92 of 24 GiB) | 39.52 GiB experts on host, loaded 14.15 GiB, KV 5.68 GiB = 486,838 tokens (1.86×) | EXIT 0, answer 63, 436 s wall, ready in 258 s |

The 24 GiB row is a utilization cap on the 31.86 GiB card: it shows the plan fits that budget,
not how a real 24 GB card's driver overhead behaves. `plan_vram.py` with `PLAN_VRAM_GIB=24.0`
prints the same argv and env (offload 40 given by hand; on this 60.3 GiB host the planner's
pinned-RAM cap would stop at 30 GiB and warn). 39.52 GiB pinned on the 60.3 GiB host ran without
trouble. DSpark there: 176 of 228 draft tokens accepted (77.2%), per position 0.88 / 0.75 / 0.68;
TTFT mean 37.9 s over 4 requests (7,524 prompt tokens).

## Why the first run refused

The launchers served Super-VL with its vision and audio towers. vLLM loaded the tower (the 1.56
GiB difference between the two pin3 runs; the safetensors headers give 1.57 GiB for vision +
projector) and profiled the encoder with "1 video items of the maximum feature size" at a
262,144-token encoder budget. That profile logged a caching-allocator OOM on a 2.29 GiB request
and left 1.41 GiB for KV. Fix on the branch: `glq.tooling.text_only_serve_args` gives Super-VL
`--language-model-only` in both glq-chat and glq-code.

## pin3, the mechanism

- Arena: 40/40 MoE layers, `88 slots/layer x 40 layers = 6.79 GiB, 1.976 MiB/slot, serving steps
  up to 4 token(s)`, no refusal.
- DSpark: 96 drafts, 235 of 288 draft tokens accepted (81.6%), per position 0.948 / 0.792 /
  0.708, over 334 generated tokens in 3 requests. Too few tokens for a speed figure; the AIPerf
  numbers in `../README.md` are the speed reference.
- CUDA graphs captured (0.54 GiB target + 0.44 GiB drafter).
- Prefix cache: 0 hits in 7,730 queries over 3 turns. Each prompt (1,703 to ~2,000 tokens) is
  shorter than the 2,176-token attention block this hybrid model aligns its cache to, so 0 is
  expected here and says nothing about a long session.

## TTFT

Mean 34.6 s over pin3's 3 requests (sum 103.68 s), 32.8 s for auto's first request (1,703 prompt
tokens, no drafter). It is not caused by DSpark. Each pi turn re-prefills its whole prompt, so on
this setup an interactive turn waits about half a minute before the first token. Not investigated
yet.

## What auto leaves on the table here

Auto turned DSpark off ("one draft token needs 3.4 GiB"), while vLLM then had 10.9 GiB of KV
beyond what the window needs. The planner counts 7.05 GiB this serve never loads:

- the 5.48 GiB MTP head (`language_model.mtp.*`), which the checkpoint does not declare as
  `nontext_bytes` because `quantize_model._is_nontext` only checks the first name segment;
- the 1.57 GiB vision tower, which `--language-model-only` now skips.

The planner's resident estimate at 36 GiB offloaded is 15.26 GiB; the weights vLLM loaded were
8.32 GiB (12.26 minus the 3.94 GiB arena).
