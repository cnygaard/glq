# Arete-27B on a 24 GB card: glq-code and glq-chat, before and after the fit fix

The question: does Arete-27B 4 bpw (`xv0y5ncu/Arete-27B-GLQ-trellis-3inst-4bpw`) serve on a
24 GB card through `glq-code` and `glq-chat`, as it must before it can join the start-here
collection (the installer menu)?

## Method

One RTX PRO 4500 Blackwell (32 GB, 32,623 MiB in nvidia-smi, 31.43 GiB CUDA total), AWS g7.4xlarge,
vLLM 0.33.0, torch 2.14.0, 2026-11-19. Rows marked 24 are that card capped to 23.99 GiB, which is
the 24,564 MiB nvidia-smi reports for an RTX 4090.

* `tier24_launch.py` makes glq-code / glq-chat plan for the capped card: it patches the
  VRAM/RAM readers in `glq.chat` and `glq.code`, and rescales `--gpu-memory-utilization` by the
  ratio of the two nvidia-smi totals. The cap gives vLLM 21.69-21.78 GiB, close to the
  0.92 x 23.65 = 21.76 GiB a real RTX 4090 would give it.
* vLLM's API server takes 2,048 tokens a step on any card under 70 GiB, the capped one included.
* `tier24_run.sh TAG code|chat VRAM_GIB RAM_GIB MODEL [flags]` runs one row in a 32 GiB host
  cgroup with its own HOME. glq-code then has pi read `hello.py` and say what it prints (63).
  glq-chat is asked the capital of France. Both print `TIER {json}` lines for the plan, ready,
  probe and exit.
* `arete24_queue.sh` ran glq 0.8.37 from PyPI. `fit24_queue.sh` ran the fix branch, with no
  `--force` anywhere.

Decode is B=1, 256 tokens at temperature 0, three repeats after a warm-up. "loaded" is vLLM's
`Model loading took`, "KV" its `Available KV cache memory`. The activation, non-torch and graph
figures are vLLM's own startup breakdown.

## glq 0.8.37

| row | window | result |
|---|--:|---|
| glq-code, 24, 4 bpw | 16,384 | refused: "SHORT by 1.42 GiB" |
| glq-code, 24, 4 bpw, `--force` | 16,384 | served: loaded 16.58, KV 2.60, 28.0 tok/s (27.99-28.09), pi answered 63 |
| glq-chat, 24, 4 bpw | 8,192 | refused: "SHORT by 1.42 GiB" |
| glq-chat, 24, 4 bpw, `--force` | 8,192 | served: KV 2.47, 27.9 tok/s, "Paris" |
| glq-code, 32, 4 bpw | 65,536 | served: KV 7.15, 27.9 tok/s, pi answered 63 |

The fit check reserved 4.51 GiB for runtime overhead (4 GiB + 3% of the resident), and counted
the file, 16.98 GiB, as resident. It also asked for at least 2 GiB of KV. vLLM used 2.64 GiB
(activation 2.50, non-torch 0.10, graphs 0.04), loaded 16.58 GiB because the 0.86 GiB vision
tower is not loaded under `--language-model-only`, and needed ~1.47 GiB of KV for 16,384 tokens.

## The fix branch (`fix/fit-arete-24gb`)

| row | window | loaded | activation | non-torch | graphs | KV | result |
|---|--:|--:|--:|--:|--:|--:|---|
| glq-code, 24, 4 bpw | 16,384 | 16.58 | 2.50 | 0.10 | 0.04 | 2.52 | 28.2 tok/s (28.22-28.28), pi answered 63 |
| glq-chat, 24, 4 bpw | 16,384 | 16.58 | 2.62 | 0.11 | 0.08 | 2.38 | "Paris" |
| glq-code, 22.49 (L4), 4 bpw | - | - | - | - | - | - | refused: "SHORT by 1.28 GiB" |
| glq-code, 32, 5 bpw | 65,536 | 19.44 | 2.51 | 0.37 | 0.04 | 5.31 | pi answered 63 |
| glq-chat, 32, 5 bpw | 65,536 | 19.44 | 2.63 | 0.38 | 0.08 | 5.18 | "Paris", 19.7 tok/s |
| glq-code, 32, 6 bpw | 32,768 | 22.44 | 2.51 | 0.13 | 0.04 | 3.08 | pi answered 63 |
| glq-chat, 32, 6 bpw | 32,768 | 22.44 | 2.63 | 0.14 | 0.08 | 2.95 | "Paris", 19.5 tok/s |

0.8.37 planned 5 bpw at 32,768 on the 32 GB card and refused 6 bpw there.

How these runs relate to the branch as committed:
- **They served on a draft.** The rows above ran with a 3.0 GiB family overhead and the
  2 GiB KV floor.
- **The 5 bpw glq-chat row set the final figure.** As the planner counts it, that row needed
  3.006 GiB, so the committed figure is 3.25 GiB: that run plus the 0.22-0.24 GiB more non-torch
  memory the same checkpoints showed on an RTX PRO 6000.
- **The committed figure also replaces the 2 GiB KV minimum.** This family's floor window is
  now priced at its measured KV cost, 1.74 GiB for 16,384 tokens.
- **The committed plans give vLLM at least as much as the draft did.** Re-planned with the
  committed code on the same box, every row keeps its window, and the utilization is equal or
  higher: 0.92 at 24, 0.887 for 5 bpw and 0.904 for 6 bpw at 32. The 22.49 GiB row still
  refuses, "SHORT by 1.28 GiB".

## Caveats

* The 24 GB card is emulated on a Blackwell card. A real 3090 or 4090 that also drives a
  display has less free VRAM, and its attention backend may allocate differently. The margin
  above covers the non-torch spread seen between two cards, not a measured consumer card.
* tok/s is the RTX PRO 4500's.
* The raw logs are not in the repo (`*.log` is ignored). They were copied off the box with
  sha256 checked, and are kept with the maintainer.
