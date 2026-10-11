# What a GiB of expert offload costs on an L4 — and the ceiling it puts on an expert cache

This exists to answer one question before any code was written for it: **a routing-aware expert
cache raises the hit rate, but how much decode time does that actually buy?** S0 measured a hit
rate of **0.8233** for a 10 GiB arena (`benchmarks/expert_cache_trace/`). A hit rate is not a
speedup. The conversion is the offload slope, and the only slope on record was measured on two
**PCIe 5** cards (`benchmarks/flashnext_offload_slope/`: 0.41–0.49 ms/token/GiB). That README
says to "expect roughly double on PCIe 4" — an expectation, not a measurement, and the whole
build/don't-build decision rested on it.

It was wrong by 1.8×, in the favourable direction.

**Base model and licence.** The checkpoint measured here is a GLQ quantization of
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), released under the
**Qwen Community License 1.0** ([licence
text](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)) — a custom licence,
not Apache 2.0. Everything in this directory is timing and memory measurement plus rescued
engine log lines; no model content.

**Provenance.** L4 (`gr6.4xlarge`, 23,034 MiB, PCIe 4), 2026-10-11, vLLM 0.31.0, model
`xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4-3mtp`. Every arm identical except
`--cpu-offload-gb`: `--batches 1 --decode 256 --repeats 3 --max-model-len 4096 --gpu-mem 0.85
--mm0 --dtype bfloat16 --offload-params experts --offload-backend uva`, B=1, `ignore_eos` at
temperature 0, warmup discarded. Raw per-arm logs on the box as
`/opt/dlami/nvme/l4_slope_off*.raw.log`.

| requested | vLLM actually offloaded | resident | KV tokens | tok/s | range | ms/token | Δ ms/GiB |
|--:|--:|--:|--:|--:|--|--:|--:|
| 44 | **41.48** (saturated) | 6.85 | 81,920 | 6.434 | 6.434–6.434 | 155.42 | — |
| 40 | 40.30 | 8.03 | 71,338 | 6.521 | 6.521–6.521 | 153.35 | **1.757** |
| 38 | 38.54 | 9.80 | 54,954 | 6.653 | 6.653–6.653 | 150.31 | **1.729** |
| 36 | 36.18 | 12.15 | 33,450 | 6.834 | 6.832–6.836 | 146.33 | **1.687** |
| 32 | 32.36 | 15.98 | — | *refused* | | | |

`Δ ms/GiB` is against the row above. **Requesting 44 GiB offloads only 41.48** — the pageable
set is the experts, and they total 42.62 GiB, so anything above that is the same point.

**The 32 GiB arm is a legitimate refusal, not a failure to fix.** Resident reaches 15.98 GiB
against the 19.1 GiB that `--gpu-mem 0.85` allows, leaving nothing for the KV pool, and vLLM
says so: `ValueError: No available memory for the cache blocks`. Raising `--gpu-mem` for one arm
would change the configuration mid-sweep, and a slope fitted across two budgets is not a slope.
It still earns its place: its `Model loading took 15.98 GiB` is a third point for the accounting
check below.

## ★ The slope is ~1.72 ms/token/GiB — and it is linear

Three consecutive intervals across the four arms: **1.757, 1.729, 1.687 ms/GiB**, a spread of
**4.0%** over a 5.3 GiB span. Repeats within each arm agree to ±0.01–0.06%, so the intervals
are far outside the noise floor, and the mild monotone drift says the curve is very slightly
convex rather than noisy.

That is what licenses extrapolating down to an arena-sized offload, and the fit is insensitive
to how much of the data it sees: effective bandwidth comes out **12.30 GB/s on two arms, 12.24
on three, 12.23 on four**. A relationship that answers the same on half the data is a line.

**That is 3.6× the PCIe-5 slope (0.47) and 1.8× the "expect 2×" guess.** The link alone accounts
for ~2×; the rest is UVA efficiency — the fit below puts this card at 12.24 GB/s effective
against a ~25 GB/s PCIe-4 x16 peak, i.e. ~49%, because a routed expert is a fine-grained
uncoalesced read rather than a bulk transfer.

**The byte accounting checks against the engine, four times.** Un-offloading a GiB must put a
GiB of experts back on the card, 1:1, and vLLM's own `Model loading took` says it does — to
within **0.01 GiB** at every pair (40.30→36.18 gives +4.12/−4.12; 41.48→36.18 gives
+5.30/−5.30; 40.30→38.54 gives +1.76/−1.77; 40.30→32.36 gives +7.94/−7.95). Note the last two
use arms whose *rate* is unavailable — the refused 32 GiB arm and the in-flight 38 GiB one —
which is why they were worth keeping. The *absolute* resident figure is not checked by this,
because non-expert weights are not modelled; only the delta, which is all the fit uses.

## What it means for the expert cache

`benchmarks/_expert_cache_ceiling.py` does this arithmetic, fitting effective bandwidth from the
arms rather than from a spec sheet:

```
$ python benchmarks/_expert_cache_ceiling.py \
    --arm 41.48:6.434 --arm 40.3:6.521 --arm 38.54:6.653 --arm 36.18:6.834 --baseline 40.3

expert bytes      : 42.62 GiB (48 x 512 x 1.776 MiB)
touched per token : 852.5 MiB (48 layers x top_k 10)
fitted effective PCIe bandwidth : 12.23 GB/s
fitted step floor (zero traffic): 84.24 ms/token -> 11.87 tok/s
```

| | ms/token | tok/s | vs baseline |
|---|--:|--:|--:|
| today, `--cpu-offload-gb 40` (hit rate 0.055, resident *by luck*) | 153.35 | 6.521 | 1.00x |
| 10 GiB arena at the **measured** 0.8233 | 97.16 | 10.293 | **1.58x** |
| 10 GiB arena at a **perfect** 1.0 — the hard ceiling | 84.24 | 11.871 | **1.82x** |

So **offload is 69.1 of the 153.35 ms step — 45.1%** — and a routing-aware arena is predicted to
recover about three quarters of it. That is worth building: 1.58× on the single configuration
GLQ exists to serve (a large MoE on a 24 GB card).

**Note what the baseline's own hit rate is: 0.055.** `--cpu-offload-gb` leaves 5.5% of expert
bytes on the card and they are resident *by luck*, not by routing — vLLM offloads whole
parameters with no idea which experts get picked. The entire value of an arena is that it makes
the same VRAM routing-aware. That framing also gives the cheapest possible sanity check on the
whole model: the four arms span hit rates 0.027→0.151 and the fit reproduces all four rates,
so the curve the arena would move along is the one already measured, not a new one.

Two reasons the prediction is conservative: an arena **fill is a bulk contiguous copy** where
the UVA read it replaces is fine-grained, so a miss should cost less per byte than the fitted
12.24 GB/s implies; and the 0.8233 came from a 367-step trace whose compulsory misses cap it at
0.9201, which a longer run amortises.

## ⚠ The bigger number is the floor, and the cache cannot touch it

**84.33 ms/token remains with every expert byte already resident.** That is 55% of the current
step and it is not PCIe at all — it is the trellis decode for 480 experts plus the dense,
GDN and attention work. For scale, the same checkpoint runs at **31.30 ms/token with zero
offload on a PRO 6000** (`benchmarks/flashnext_offload_slope/`), so the L4's floor is 2.7× that,
which is roughly its bandwidth and compute deficit — i.e. the floor looks like an honest cost,
not a bug.

This matters for scoping. If a competing runtime serves this model at ~45 tok/s (22 ms/token) on
**this class of card**, then a perfect expert cache still leaves GLQ 3.8× behind, and the gap is
in the decode step rather than in paging. **The hardware behind that 45 tok/s figure was not
specified**, so it is not a like-for-like comparison and should not be treated as one — but the
floor is now measured either way, and it bounds what any caching work can deliver.

## Reproducing

```bash
# the sweep (one arm at a time; ~8 min each, mostly model load)
for OFF in 36 38 40 44; do
  python benchmarks/run_model.py --model <ckpt> --runtime vllm \
    --batches 1 --decode 256 --repeats 3 --max-model-len 4096 --gpu-mem 0.85 \
    --mm0 --dtype bfloat16 --cpu-offload-gb $OFF \
    --offload-params experts --offload-backend uva --label off$OFF > off$OFF.log 2>&1
  echo "ARM_EXIT=$?"      # NOT after a pipe: `cmd | grep` makes $? the grep's
done
```

That last comment is load-bearing — piping an arm into `grep` or `tail` made four earlier arms
in this project report `EXIT=0` while failing, and it caught the 32 GiB refusal correctly here
only because the status is read from the python directly.
