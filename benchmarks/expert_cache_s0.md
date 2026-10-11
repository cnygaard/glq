# S0 — what hit rate would a GLQ expert cache reach? (measured, not modelled)

Before building an expert cache, score the routing a real run actually produced. Strata sized
its own design this way (`Memory/cache_allocation.py`) rather than by reasoning, and the trace
is what told it its cache was *"never short of capacity — it was handing all of it to two
layers."*

## ⚠ The trace itself is not in this repo

`routing_flashnext.npz` lives in the **glq-benchmarks** repo
(`records/expert_cache_trace/routing_flashnext.npz`), not here, for two reasons:

* it is raw per-run data, and this project already routes that to glq-benchmarks rather than
  into the source tree;
* it is **model output** — the sequence of expert *indices* each token selected — derived from
  a GLQ quantization of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
  which is released under the **Qwen Community License 1.0**
  ([licence text](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)) — a custom
  licence, not Apache 2.0. glq itself is GPL-3.0-only, so third-party-derived data stays out of
  the source tree rather than sitting under a licence it was never offered under.

Everything below is measurement and analysis, which is this project's own. Point the scorer at
wherever you have the file:

```bash
python benchmarks/_expert_cache_score.py \
    ../glq-benchmarks/records/expert_cache_trace/routing_flashnext.npz --gib 10
```

**Provenance of that file.** Recorded 2026-10-11 on an **L4** (`gr6.4xlarge`, 22.5 GiB,
PCIe 4), model `xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4-3mtp` (48 layers x 512
experts, top_k 10), vLLM 0.31.0, `--max-model-len 4096 --gpu-mem 0.85 --dtype bfloat16
--cpu-offload-gb 40 --offload-params experts --offload-backend uva`, B=1, 256 decode tokens.
220,211 records / 2,202,110 lookups. To regenerate it from scratch:

```bash
GLQ_MOE_TRACE_ROUTING=/opt/dlami/nvme/routing.npz GLQ_MOE_TRACE_FLUSH_EVERY=500 \
python benchmarks/run_model.py --model <ckpt> --runtime vllm \
    --batches 1 --decode 256 --repeats 1 --eager \
    --max-model-len 4096 --gpu-mem 0.85 --mm0 --dtype bfloat16 \
    --cpu-offload-gb 40 --offload-params experts --offload-backend uva
```

`--eager` was required to record it at all: the recorder sits in `GLQFusedMoEMethod.apply`,
and under CUDA graphs that Python runs only at capture time. Routing does not depend on graph
capture, so an eager trace is a valid trace — only its tok/s would be unrepresentative, and
tok/s is not what a trace is for.

## ★★ The trap: 89.3% of the trace is not real routing

| forward width | records | share | distinct (layer, expert) pairs |
|---|--:|--:|--:|
| **4096 tokens** — vLLM's dummy KV-sizing pass | 196,608 | 89.3% | **495** of 24,576 |
| 64 tokens — the real prompt prefill | 3,072 | 1.4% | 7,650 |
| 15 tokens — warmup | 2,880 | 1.3% | — |
| **1 token — decode** | 17,651 | 8.0% | **14,109** |

vLLM runs one synthetic max-model-len forward at startup to size its KV pool. Its input is
dummy data, so it routes to about ten experts per layer and then repeats that 4096 times. Any
cache nails a 495-pair working set.

**Scoring everything therefore reported `0.9400` where decode's truth was `0.5263`** — an 79%
relative overstatement, and it is the number this harness printed on its first run. The fix is
in the tool, not in the reader's vigilance: records now carry the width of the forward they
came from, and `_expert_cache_score.py --phase` **defaults to `decode`**. Decode is also the
phase the 6.517 tok/s baseline measures, so it is the only phase the decision can use.

The general lesson is the one this repo keeps relearning: a benchmark that silently includes a
warmup or profiling artifact does not fail, it just answers a different question convincingly.

## ⚠ Update (S1): these rates were scored with the wrong semantics, and are a floor

The table below scored **per-routing** admission (`lookup_or_admit`), which is what the
offline policy did when S0 was written. The live arena cannot work that way: a per-routing
admit can hand out as an eviction candidate a slot that a *later routing of the same step* is
about to read, and because the fused op reads the arena **after** the fill, the matmul would
then read a slot holding a different expert — plausible output from the wrong weights.

So `ResidencyPolicy.plan_step` resolves a whole step at once and protects the slots that step
reads. Minimal demonstration of the difference, 2 slots, warm `{A, B}`, step `[C, A]`:
per-routing scores **0 hits / 2 misses**, step semantics **1 / 1**.

Because protection strictly avoids evicting a would-be hit, **step semantics is never worse**,
so every rate below is a floor. Re-scored on the same trace (`--per-routing` reproduces the old
column):

| slots/layer | per-routing | step (live) | delta |
|--:|--:|--:|--:|
| 16 | 0.3823 | 0.4159 | +0.0335 |
| 32 | 0.5409 | 0.5540 | +0.0131 |
| 64 | 0.6959 | 0.7019 | +0.0061 |
| 92 | 0.7791 | 0.7824 | +0.0033 |
| **120** | 0.8311 | **0.8325** | +0.0014 |
| 185 | 0.8916 | 0.8919 | +0.0003 |

The gap closes as the arena grows, which is the signature of a pathology that needs a tight
cache to bite. At the shipping rung it is +0.0014, so **S0's conclusion is unchanged** and the
prediction moves 1.58× → **1.59×**.

At the corrected blob size (1,850,900 B — the arena excludes the per-layer `SV`, see
`moe_shapes.bytes_per_slot`), 10 GiB is exactly **120 slots/layer**, and the live policy
comparison there is:

| admission | eviction | hit rate | evictions |
|---|---|--:|--:|
| GLOBAL | none | 0.5341 | 0 |
| per-layer | none | 0.5421 | 0 |
| per-layer | **LRU** | **0.8327** | 23,765 |
| per-layer | LFU | 0.8062 | 28,451 |

LRU still beats LFU under live semantics, so S0's policy choice stands.

## The decode-only curve

Replayed through the same `ResidencyPolicy` the live cache would use, so the number means
something about the cache that gets built. Blob size 1.84 MiB/expert (gate_up + down at 3 bpw).

| slots/layer | arena GiB | GLOBAL | per-layer | **per-layer + LRU** | + LFU | LRU evictions |
|--:|--:|--:|--:|--:|--:|--:|
| 8 | 0.7 | 0.0588 | 0.0600 | 0.1089 | **0.1498** | 156,899 |
| 16 | 1.4 | 0.1107 | 0.1116 | **0.3823** | 0.2542 | 108,256 |
| 32 | 2.8 | 0.1834 | 0.1873 | **0.5409** | 0.3962 | 79,503 |
| 64 | 5.5 | 0.3225 | 0.3272 | **0.6959** | 0.5855 | 50,612 |
| 92 | 7.9 | 0.4453 | 0.4355 | **0.7791** | 0.6938 | 34,578 |
| **115** | **9.9** | 0.5200 | 0.5263 | **0.8233** | 0.7586 | 25,663 |
| 139 | 12.0 | 0.5849 | 0.5992 | **0.8560** | 0.8086 | 18,753 |
| 185 | 16.0 | 0.6776 | 0.7360 | **0.8916** | 0.8690 | 10,259 |
| 256 | 22.1 | 0.8995 | 0.8911 | **0.9142** | 0.9099 | 3,024 |
| 512 (everything) | 44.2 | 0.9201 | 0.9201 | 0.9201 | 0.9201 | 0 |

**0.9201 is this trace's ceiling, not 1.0.** Decode touched 14,109 distinct pairs across 176,510
lookups, so 7.99% of lookups are compulsory first-touches that no cache can serve. A longer run
would amortise them further, which means **every rate here is a slight underestimate** of a
steady-state cache — the conservative direction.

Read that way, a **10 GiB arena captures 89.5% of what full 44.2 GiB residency achieves, using
22.4% of the bytes.** As a policy result that is strong.

## Three findings, two of which invert the plan

**1. ★ Per-layer admission is a non-finding here — 0.5236 vs 0.5300.** Strata's 2.97%-vs-21.4%
was the plan's headline transferable result. It does not transfer, for a structural reason
worth keeping: the pathology needs `slots < n_layers * top_k`. Strata had 256 slots against
48x10 = 480 lookups per position, so its cache filled inside the first position it ever saw. A
10 GiB arena affords 5,565 against 480 — **11.6 positions** — so no single position can
monopolise it. Keep per-layer quotas anyway (free, and makes a layer's budget predictable), but
below ~1 GiB the comparison should be re-run rather than inherited.

**2. ★ Eviction is the lever, and LRU is the policy — the opposite of the synthetic result.**
LRU moves the 10 GiB arena from 0.5263 to **0.8233**, and beats LFU at every size from 16
slots/layer up. On the synthetic fixture in `tests/test_expert_cache_policy.py` LRU was **12x
worse** than no eviction at 2 slots/layer (total thrash). Both measurements are real; the
synthetic one simply sits in a different regime. Strata declined to guess this policy and
called it "a measured question" — it was right to, and a cache shipped with the intuitive
choice would have been a straight loss at small arenas.

**3. Routing is diffuse.** 14,109 of 24,576 pairs (57.4%) in 367 decode steps, i.e. ~294 of 512
experts per layer. There is no small hot set to exploit; the gains above come from *recency*,
not from concentration. That also means a profile-ranked warm fill (the plan's S2) has less to
offer here than Strata's 0.4864 -> 0.6447 suggests — and a profile trained on a trace that
includes the dummy profiling pass would be trained on 495 pairs of noise.

## ⚠ What this does NOT establish

**A hit rate is not a speedup**, and the conversion is where this decision actually lives.
Removing PCIe traffic only helps to the extent the step is PCIe-bound:

* measured step at the 6.517 tok/s baseline: **153.4 ms/token** with 40.3 GiB offloaded;
* measured offload price on two **PCIe 5** cards (`benchmarks/flashnext_offload_slope/`):
  **0.41–0.49 ms/token/GiB**, stable across cards whose decode rates differ 2.2x.

If the L4's PCIe-4 slope is ~2x that (the expectation stated in that README, **not** a
measurement), offload is ~36 ms of the 153 ms step and **a perfect cache is capped at ~1.31x**,
with 0.8233 worth ~1.24x. That is far below WiSP's 2.63x (measured on Qwen3-30B-A3B, 8-of-128
routing, batch rather than single-stream) and nothing like Strata's ~45 tok/s on this card —
which would imply the other ~117 ms/token is not offload at all.

So the gate on building S1 is the **L4's own offload slope**, which has never been measured.
That is a four-arm, ~35-minute run varying only `--cpu-offload-gb` (32/36/40/44), and it either
justifies the arena or ends the line of work. It must come before the arena, not after.
