# Hardware tiers: glq-code plans on 8-96 GiB cards and 16-128 GiB hosts

Every tier runs on one RTX PRO 6000 Blackwell (95.59 GiB, PCIe 5), made to look like the
smaller machine:

* the planner sees the tier's VRAM and RAM (`tier_launch.py` patches `glq.code`'s readers),
  so the window, offload, arena, DSpark depth and fit check are that tier's;
* `--gpu-memory-utilization` is rescaled, so vLLM's absolute budget equals the tier's;
* below 70 GiB, `--max-num-batched-tokens` is what vLLM itself picks on a card that size
  (2048, raised to the largest multimodal item for gemma-4 26B/31B), because vLLM chooses it
  from the physical card's memory and a 96 GB card gets 8192;
* host RAM is capped for the whole process tree with `systemd-run --scope -p MemoryMax`;
* rows share one FlashInfer JIT cache, as one machine would.

`tier_run.sh TAG GPU VRAM RAM PORT MODEL [glq-code flags]` runs one row. It prints one
`TIER {json}` line per phase (plan, ready, probe, metrics, exit, or failed with vLLM's root
cause), then glq-code hands pi a one-file task. `queues.sh`, `queue_textonly.sh` and
`queue_gemma.sh` hold the matrix, one queue per GPU. `--force` rows were refused by the planner
and run anyway, to see whether the refusal was right. `summarize.py <dir>` prints the table
below from the logs.

Decode is B=1, 256 tokens at temperature 0 with `ignore_eos`, mean of 3 repeats after a
warm-up, with the range. It comes from a capped RTX PRO 6000 with its full memory bandwidth
and PCIe 5 link, so it does not predict a consumer card's speed. Footprint, KV and whether a
plan starts do carry over.

## Run 2 (2026-11-01, vLLM 0.32.0, glq-code from `feat/dspark-launchers`)

"loaded" is vLLM's `Model loading took`, "KV" its `Available KV cache memory`. `x` rows add
`--language-model-only` (gemma-4 text-only, which glq-code does not pass yet).

| tag | VRAM / RAM GiB | model | forced | window | offload GiB | arena GiB | k | loaded GiB | KV GiB | result |
|---|---|---|---|--:|--:|--:|--:|--:|--:|---|
| t8_e4b | 8 / 16 | gemma-4-E4B-it-GLQ-trellis-3inst-4bpw | yes | 16384 | 0 | 0 | 0 | 6.29 | 0.06 | failed: KV 0.43 GiB needed, 0.06 available |
| t8_l3_r16 | 8 / 16 | NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GLQ-trellis-3inst-3bpw | yes | 16384 | 8 | 0 | 0 | 4.12 | 2.1 | 57.3 tok/s (56.8-58.2), TTFT 191 ms |
| t8_smol3 | 8 / 16 | SmolLM3-3B-trellis-3inst-4bpw-kernel | yes | 32768 | 0 | 0 | 0 | 1.81 | 4.51 | 165.3 tok/s (165.2-165.3), TTFT 24 ms |
| x8_e4b_lmo | 8 / 16 | gemma-4-E4B-it-GLQ-trellis-3inst-4bpw | yes | 16384 | 0 | 0 | 0 | 5.39 | 0.24 | failed: KV 0.43 GiB needed, 0.24 available |
| x8_e4b_lmo_8k | 8 / 16 | gemma-4-E4B-it-GLQ-trellis-3inst-4bpw | yes | 8192 | 0 | 0 | 0 | 5.39 | 0.24 | failed: KV 0.3 GiB needed, 0.24 available |
| t8_l3_r32 | 8 / 32 | NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GLQ-trellis-3inst-3bpw | yes | 32768 | 10 | 0 | 0 | 2.09 | 4.12 | 51.6 tok/s (51.5-51.6), TTFT 230 ms |
| t12_e4b | 12 / 16 | gemma-4-E4B-it-GLQ-trellis-3inst-4bpw | yes | 16384 | 0 | 0 | 0 | 6.29 | 3.71 | 123.0 tok/s (122.9-123.0), TTFT 60 ms |
| t12_g26b3 | 12 / 16 | gemma-4-26B-A4B-it-GLQ-trellis-3inst-3bpw | yes | 16384 | 0 | 0 | 0 | 11.49 | - | failed: no memory for KV cache blocks |
| t12_l4_r32 | 12 / 32 | NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GLQ-trellis-3inst-4bpw |  | 32768 | 13 | 2 | 0 | 4.65 | 5.19 | 67.7 tok/s (67.4-68.1), TTFT 288 ms |
| t16_g26b3 | 16 / 32 | gemma-4-26B-A4B-it-GLQ-trellis-3inst-3bpw | yes | 16384 | 0 | 0 | 0 | 11.49 | 1.98 | 96.6 tok/s (96.5-96.6), TTFT 71 ms |
| t16_g26b4 | 16 / 32 | gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw | yes | 16384 | 0 | 0 | 0 | 14.4 | - | failed: no memory for KV cache blocks |
| t16_l4 | 16 / 32 | NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GLQ-trellis-3inst-4bpw |  | 32768 | 13 | 5 | 0 | - | - | 80.9 tok/s (80.8-80.9), TTFT 285 ms |
| t16_sv3 | 16 / 64 | NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-3bpw | yes | 16384 | 32 | 0 | 0 | 12.13 | 0.71 | 14.2 tok/s (14.2-14.2), TTFT 675 ms |
| t24_g26b4 | 24 / 64 | gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw |  | 65536 | 0 | 0 | 0 | 14.4 | 4.7 | 97.6 tok/s (97.4-97.7), TTFT 71 ms |
| t24_l4 | 24 / 64 | NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GLQ-trellis-3inst-4bpw |  | 131072 | 13 | 10 | 0 | 12.66 | 7.87 | 91.2 tok/s (91.0-91.4), TTFT 279 ms |
| t24_sv3_auto64 | 24 / 64 | NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-3bpw |  | 262144 | 38 | 5 | 0 | 11.22 | 8.5 | 23.4 tok/s (23.4-23.4), TTFT 775 ms |
| t24_sv3_pin3 | 24 / 64 | NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-3bpw |  | 262144 | 38 | 6.8 | 3 | 15.63 | 4.28 | 25.2 tok/s (25.2-25.2), TTFT 864 ms |
| x24_g26b4_lmo | 24 / 64 | gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw |  | 65536 | 0 | 0 | 0 | 13.32 | 5.15 | 97.1 tok/s (97.1-97.2), TTFT 71 ms |
| t24_sv3_auto128 | 24 / 128 | NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-3bpw |  | 262144 | 40 | 5.1 | 2 | 12.42 | 7.46 | 24.1 tok/s (24.1-24.1), TTFT 894 ms |
| t32_g31mix | 32 / 64 | Gemma-4-31B-it-GLQ-5.0bpw-mix3-8 |  | 65536 | 0 | 0 | 0 | 17.38 | 7.81 | failed: KV 9.6 GiB needed, 7.81 available |
| t32_sv4_r64 | 32 / 64 | NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-4bpw |  | 131072 | 38 | 0 | 0 | 20.02 | 6.48 | 12.9 tok/s (12.9-12.9), TTFT 755 ms |
| t32_sv4_auto128 | 32 / 128 | NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-4bpw |  | 262144 | 53 | 11.31 | 4 | 19.5 | 7.29 | 21.1 tok/s (21.1-21.1), TTFT 1012 ms |
| t48_g31t4 | 48 / 64 | gemma-4-31B-it-GLQ-trellis-3inst-4bpw |  | 262144 | 0 | 0 | 0 | 18.04 | 9.96 | failed: KV 24.6 GiB needed, 9.96 available |
| t48_sv4_auto64 | 48 / 64 | NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-4bpw |  | 262144 | 38 | 10 | 3 | 32.57 | 8.33 | 25.6 tok/s (25.6-25.6), TTFT 866 ms |
| t48_sv4_auto128 | 48 / 128 | NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-4bpw |  | 262144 | 53 | 26 | 4 | 34.08 | 7.37 | 27.5 tok/s (27.4-27.5), TTFT 1011 ms |
| t96_sv3_auto | 95.6 / 128 | NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-3bpw |  | 262144 | 0 | 0 | 4 | 46.77 | 6.43 | 69.9 tok/s (69.9-69.9), TTFT 97 ms |
| t96_sv4_1m | 95.6 / 128 | NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-4bpw |  | 1048576 | 0 | 0 | 4 | 60.71 | 14.94 | 73.1 tok/s (73.1-73.1), TTFT 98 ms |
| t96_sv4_auto | 95.6 / 128 | NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-4bpw |  | 262144 | 0 | 0 | 4 | 60.71 | 6.57 | 73.1 tok/s (73.1-73.1), TTFT 99 ms |

"loaded" includes the arena and, on k > 0 rows, the DSpark drafter. t16_l4's vLLM log was not
kept. t24_sv3_pin3's last line (`EXIT=`) was lost when its script was replaced while it ran.
Its TIER lines are intact, including `exit rc 0` and the acceptance (791 of 1,380 draft
tokens).

What the run shows:

* **Six planner refusals were wrong.** SmolLM3 on 8 GiB, Lightning 3 bpw on 8 GiB (16 and 32
  GiB hosts), E4B on 12 GiB, 26B-A4B 3 bpw on 16 GiB and Super-VL 3 bpw on 16 GiB all served
  when forced. The planner's 4 GiB overhead floor and 2 GiB KV floor are too large for small
  cards; recalibrating them is the next step.
* **Five refusals were right**: E4B on 8 GiB (also text-only, also at an 8,192 window), 26B-A4B
  3 bpw on 12 GiB and 26B-A4B 4 bpw on 16 GiB.
* **Two plans the planner offered did not start: both gemma-4 31B rows.** vLLM needs 9.6 GiB
  of KV for a 65,536 window and 24.6 GiB for 262,144, roughly 4.6 GiB per request plus
  ~82 KB per token, far above the 28,201 B/token envelope the planner prices 31B with. This
  predates this branch and is not fixed here.
* Lightning 4 bpw on 24 GiB is the case `supervisor.kv_floor_bytes` fixed for glq-chat; glq-code
  planned 13 GiB offloaded and a 10 GiB arena and served 131,072 tokens.

Raw logs go to the glq-benchmarks repository.
