# What a GiB of expert offload costs — two cards

Why this exists: `glq-code` now spends a bounded amount of expert offload to reach a longer
context window (`supervisor.WINDOW_OFFLOAD_MAX_EXTRA_GIB`), and that policy needs a price. It
also needed a check that the window it buys is **served** rather than announced.

Model `xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4` throughout, UVA backend
(`--offload-backend uva --cpu-offload-params experts`), `VLLM_PLE_CPU_OFFLOAD=1`, B=1 fixed
concurrency, `ignore_eos` at temperature 0, 256 tokens, warmup discarded, 3 repeats.

`Δms/GiB` and `%/GiB` are measured against each card's own first row. Raw records with full
provenance in `decode_slope.jsonl`; rescued vLLM lines in `vllm_lines.txt`.

| card | window | offload | resident | tok/s | ms/token | Δms/GiB | cummulative % | %/GiB | KV pool | KV tokens | max ctx @c=1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| PRO 4500 (32 GiB) | 8192 | 27 | 21.67 | 14.496 | 68.98 | — | — | — | 1.52 | ~60,600 ˟ | 32,768 ˟ |
| PRO 4500 | 8192 | 34 | 14.60 | 13.877 | 72.06 | 0.440 | 4.3% | 0.610 | 8.54 | ~340,700 ˟ | 262,144 ˟ |
| PRO 4500 | 8192 | 42 | 7.25 | 13.310 | 75.13 | 0.410 | 8.2% | 0.545 | 15.87 | ~633,200 ˟ | 262,144 ˟ |
| PRO 6000 (96 GiB) | 262144 | 0 | 48.65 | 31.949 | 31.30 | — | — | — | 8.24 | **328,790** | 262,144 |
| PRO 6000 | 262144 | 8 | 40.11 | 28.379 | 35.24 | 0.492 | 11.2% | 1.397 | 16.74 | **669,429** | 262,144 |
| PRO 6000 | 262144 | 24 | 24.22 | 23.472 | 42.60 | 0.471 | 26.5% | 1.106 | 32.55 | **1,301,833** | 262,144 |

Resident and KV pool in GiB, from vLLM's own `Model loading took` and `Available KV cache
memory` lines. Bold KV tokens are vLLM's `GPU KV cache size`; ˟ marks values derived from the
26,912 B/token measured here, because the PRO 4500 run recorded only pool GiB.

## What it says

**The percentage cost is not portable; do not quote it as the price.** It doubles on the faster
card — 0.55%/GiB against 1.10%/GiB — for the reason any fixed cost does: the same PCIe transfer
is a larger share of a shorter step. The PRO 6000's baseline is 2.2× the PRO 4500's and its
slope is 2.0× steeper.

**The absolute cost is stable: ~0.41–0.49 ms/token per GiB offloaded**, across two cards whose
decode rates differ 2.2×, and linear over three points on the PRO 6000 (0.492 then 0.471). It is
stable because **both boxes are PCIe 5 x16**, not because it is a property of the GPU. Expect
roughly double on PCIe 4, and more on a model with denser routing.

**The link is the constraint, not host memory.** PCIe 5 x16 is ~55 GB/s effective against
~358 GB/s for DDR5-5600 across the 8 channels of the Xeon 8559C — a 6.5× margin. Per token the
GPU fetches only the *routed* experts from pinned host memory.

**`max ctx` saturates at 262,144 on five of six rows** — the model's declared ceiling. Past that
point more KV buys *concurrency* (1.25× → 2.55× → 4.97×), not context. Offloading beyond the
first budget that reaches the top tier is therefore pure decode cost, which is why the planner
walks tiers downward and stops at the first that fits.

**The asymmetry is structural, not lucky.** 8 GiB costs ~4.4% where this policy fires and ~9%
where it does not: the trade only fires on a card too small to hold the model, which is the
slower card with the gentler percentage. A 96 GiB card plans zero offload and never pays the
steep slope that card would charge.

## The window gate

The PRO 6000 `offload 0` row is the planner's own unedited choice — `--gpu-memory-utilization
0.652 --max-model-len 262144 --max-num-seqs 1`, read out of `glq-code --no-serve`. Against it:

| | predicted | vLLM reported |
|---|--:|--:|
| resident | 48.86 GiB | **48.65 GiB** |
| KV reserved for the window | 7.92 GiB | **8.24 GiB available** |

So `Maximum concurrency for 262,144 tokens per request: 1.25x`, and a real request completed
coherently. vLLM's own per-token cost is 8.24 GiB / 328,790 = **26,912 B**, making
`supervisor._KV_BYTES_PER_TOKEN = 28_201` conservative by 4.8% — the safe direction, and the
first check of that anchor **at** a 262144 window rather than extrapolated from pools measured
at 8192. It also reproduces the 26,903 B/token figure already in that constant's docstring to
within 0.03%, on a different box.

## Caveats, including one against this table

- **The PRO 4500 `offload 27` row is the least trustworthy number here.** Its 1.52 GiB pool
  implies only 32,768, yet a live `glq-code` session on that same box served **65536** at the
  same offload. Those arms ran at `--max-model-len 8192`, so their pools are
  configuration-sensitive and the derived context column for that card should not be cited.
  The PRO 6000 rows were measured *at* 262144, which is why they carry reported token counts.
- **Inferring routing sparsity from these numbers is an upper bound, not an estimate.**
  0.47 ms/GiB × ~55 GB/s suggests ~26 MB of traffic per GiB offloaded per token (~2.5% of the
  offloaded bytes), but fine-grained uncoalesced UVA reads land well under peak link bandwidth,
  so fewer bytes may be moving than that arithmetic implies. `bytes / PCIe_peak` is a floor.
- **`pcie.link.gen.current` reads 1 on an idle GPU.** The link downshifts to save power; 5 is
  what it reports under load. Query `pcie.link.gen.max`, or read it while serving.
- The two cards' offload ranges do not overlap, out of necessity: Flash-Next does not fit a
  32 GiB card at offload 0, and a 96 GiB card needs none. The per-GiB normalisation is what
  makes the rows comparable, and the three PRO 6000 points are what license it.
- B=1 only, one model, one quantization. Nothing here speaks to batch throughput, quality under
  offload, or a dense (non-MoE) model, where there is no routing sparsity to exploit.

Related: `benchmarks/aime_flashnext_offload/` (the 262k AIME run under PLE offload), and the
constant this priced, `glq/supervisor.py:WINDOW_OFFLOAD_MAX_EXTRA_GIB`.
