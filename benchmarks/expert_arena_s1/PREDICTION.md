# S1 prediction, written before the measurement

Recorded **before** the arena-on arm produced a number, so the comparison cannot be
rationalised after the fact. Nothing here is a measurement; the measured result goes in a
separate file beside it.

## The configuration being measured

L4 (23,034 MiB, PCIe 4), `xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4-3mtp`,
`--cpu-offload-gb 44` (saturates at 41.48), `--offload-params experts --offload-backend uva`,
`--max-model-len 4096 --gpu-mem 0.85 --dtype bfloat16`, B=1, 256 decode tokens, 3 repeats,
`GLQ_MOE_EXPERT_CACHE=8`, `GLQ_MOE_ARENA_MAX_TOKENS=8`.

**49 MoE containers share the budget** — 48 decoder layers plus the MTP draft block, each
with its own 512-expert MoE. At 1,850,900 B/slot that is **94 slots/layer** for 8 GiB.

## Predictions

| quantity | predicted | where it comes from |
|---|--:|---|
| decode hit rate | **0.7869** | the recorded routing trace replayed through `ResidencyPolicy.plan_step` at 94 slots/layer, LRU |
| decode rate | **10.02 tok/s** | `_expert_cache_ceiling.py`, fitted 12.23 GB/s effective PCIe and an 84.24 ms floor |
| speedup | **1.56×** | against the 41.48 GiB-offload baseline |
| hard ceiling | 1.85× | the same model at hit rate 1.0 |

## What would falsify what

* **Rate well below ~1.4× with the hit rate near 0.79** — the step is less PCIe-bound than the
  slope fit says, or the fill costs more than the UVA reads it replaces. The fitted bandwidth
  is the *uncoalesced UVA* one, so a bulk fill should be cheaper per byte; if it is dearer,
  that assumption is wrong.
* **Hit rate well below 0.79** — the live policy diverges from the scored one, or the trace
  does not describe this run's routing. Both are checkable offline.
* **Hit rate near 0.79 and the rate near 1.56×** — the model holds, and the remaining 84.24
  ms/token floor is confirmed as the next target.
* **`capture_gib` of 0, or a load that fails on KV** — the arena is costing the CUDA graph or
  the memory budget, which is a larger loss than the win.

A graphed arm emits no hit-rate report (under a full graph `apply`'s Python runs only at
capture), so the hit rate comes from a matched eager arm with identical routing.
