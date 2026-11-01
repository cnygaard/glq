# Super-VL 3 bpw with its experts on host, on a 24 GB-class budget

`xv0y5ncu/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GLQ-trellis-3inst-3bpw`, served through
`benchmarks/run_model.py` on a g7 (RTX PRO 4500 Blackwell, 31.9 GiB, PCIe 5) capped with
`--gpu-mem 0.70`, vLLM 0.32.0, bf16, text only (`--mm0`), UVA expert offload. B=1, 256
decode tokens at temperature 0 with `ignore_eos`, 3 repeats. Scripts: `offload_supervl3*.sh`.

| arm | window | `--cpu-offload-gb` | GLQ expert arena | weights in VRAM | decode tok/s (range) | TTFT |
|---|--:|--:|---|--:|--:|--:|
| A: offload only | 8,192 | 42 (every expert) | off | 4.72 GiB | 9.661 (9.661-9.662) | 470 ms |
| B: + arena | 8,192 | 42 | 103 slots/layer, 7.95 GiB | 12.72 GiB | 20.833 (20.829-20.891) | 470 ms |
| C: iso-VRAM control, no arena | 8,192 | 31.7 | off | 12.13 GiB | 10.351 (10.351-10.351) | 403 ms |
| D: arena, 262,144 window | 262,144 | 42 | 51 slots/layer, 3.94 GiB | 8.80 GiB | 20.717 (20.714-20.732) | 469 ms |
| F: arena, 1,048,576 window | 1,048,576 | 42 | 51 slots/layer, 3.94 GiB | 8.80 GiB | 20.680 (20.679-20.689) | 469 ms |

The arena printed `1.976 MiB/slot` on all 40 MoE layers, the figure
`tests/test_expert_arena_sizing.py` pins for this model. Arm C spends the arena's VRAM on more
resident experts instead, which buys +7% over A against the arena's 2.16x.

Long prompts (`needle_262k.py`, `needle_1m.py`, against a `vllm serve` of arm F's setup): a
256,011-token prompt found its needle with a 1,297 s TTFT (197 tok/s prefill) and then decoded
at 19.32 tok/s. The full 1M prompt was skipped, projected at over an hour of prefill. Prefill
through UVA is the cost of this setup, not decode.

Speed here reflects this card and its PCIe 5 link; quality under offload was not evaluated.
