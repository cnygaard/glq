# S1 measured: the VRAM expert arena on an L4

Compare against `PREDICTION.md` in this directory, written before any of these numbers
existed. The headline **beat** the prediction's own hard ceiling, which means the prediction's
model was wrong; that is written up below rather than quietly dropped.

**Provenance.** L4 (23,034 MiB, PCIe 4), 2026-10-12, vLLM 0.31.0, torch 2.13.0+cu130, triton
3.7.1, `xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4-3mtp`,
`--cpu-offload-gb 44` (saturates at 41.48) `--offload-params experts --offload-backend uva`,
`--max-model-len 4096 --gpu-mem 0.85 --mm0 --dtype bfloat16`, B=1, 256 decode tokens,
`ignore_eos` at temperature 0, warmup discarded. Arena arms: `GLQ_MOE_EXPERT_CACHE=8`,
`GLQ_MOE_ARENA_MAX_TOKENS=8` → **94 slots/layer × 49 MoE containers = 7.94 GiB**
(48 decoder layers + the MTP draft block, each a 512-expert MoE).

## The result

| arm | arena | mode | resident | KV tok | capture | tok/s | repeats | vs its own baseline |
|---|---|---|--:|--:|--:|--:|--|--:|
| off_graph | — | graph | 6.85 | 81,920 | 0.3 | 6.436 | 6.436–6.436 | 1.00× |
| **g8_graph** | **8 GiB** | **graph** | **14.79** | 10,581 | 0.08 | **17.211** | 17.163–17.543 | **2.674×** |
| off_eager | — | eager | 6.85 | 84,992 | 0 | 4.518 | — | 1.00× |
| g8_eager | 8 GiB | eager | 14.79 | 11,946 | 0 | 5.636 | — | 1.247× |

**Decode hit rate, measured live: 0.8267 / 0.8171** (two layers reporting), against **0.7869**
predicted offline at 94 slots/layer. The arena is serving ~82% of routed experts from VRAM.

## ★ The iso-VRAM control: routing-awareness is worth 2.441× of it

The obvious objection to 2.674× is that the arena arm simply holds 7.94 GiB more weight on the
card. So: arena OFF with the same footprint, by offloading 7.94 GiB less.

| arm | arena | offload | **resident** | KV tok | tok/s | repeats |
|---|---|--:|--:|--:|--:|--|
| off_graph | — | 41.48 | 6.85 | 81,920 | 6.436 | 6.436–6.436 |
| **isovram_off** | **—** | **33.54** | **14.80** | 8,874 | **7.052** | 7.048–7.053 |
| **g8_graph** | **8 GiB** | 41.48 | **14.79** | 10,581 | **17.211** | 17.163–17.543 |

At **matched resident bytes (14.79 vs 14.80 GiB)** the arena is worth **2.441×**. The extra
VRAM on its own buys **+9.6%**.

That is the claim the arena exists to make: `--cpu-offload-gb` chooses what to keep with no
knowledge of routing — its effective hit rate is the 2.7% that happens to be resident — and
the arena makes *the same VRAM* routing-aware.

**All three arms produce the identical output md5** `1e44e07f7d9a6c450a8d95492132dcdb`, so the
comparison is between three ways of fetching the same weights, not three different models.

It also shows the offload-slope model is sound where it was actually fitted: extrapolating it
to "7.94 GiB more resident, no routing awareness" predicted **7.05 tok/s** and the measurement
came in at **7.052**. The slope describes residency correctly; what it cannot describe — and
what the falsified ceiling below tried to bound — is routing-awareness.

**Output is byte-identical.** Same prompt at temperature 0, 256 tokens: the `SAMPLE` line's
md5 is `1e44e07f7d9a6c450a8d95492132dcdb` for both the off and arena graphed arms. That is the
end-to-end counterpart to the op-level `torch.equal` gate, and it simultaneously proves the
arena is engaged — identical tokens at 2.674× the rate is what a correct cache looks like.

## ★ The eager/graphed gap is the finding, not a footnote

The arena is worth **2.674× graphed but only 1.247× eager**. `plan_step` adds ~15 torch ops
per layer per step — 49 layers × 256 steps — so run eagerly its own launch overhead eats most
of the PCIe saving. Under CUDA graphs that work is replayed away and the full saving lands.

Graphs are also worth more *with* the arena than without: 6.436→4.518 is 1.42× for the
baseline, 17.211→5.636 is 3.05× with the arena.

So the capture-safety work was not hygiene, it was most of the result. A version of `plan_step`
using `torch.unique`/`bool(t.any())`/`int(t.sum())`/`nonzero` passed every eager test
including bit-exactness and would have required `enforce_eager` — delivering ~1.25× at best,
and in practice less, since `enforce_eager` also forfeits the graph's own 1.42×.

## ⚠ The prediction's ceiling was wrong, and why

`PREDICTION.md` said 1.56× with a **hard ceiling of 1.85×** (11.871 tok/s). Measured 2.674×
(17.211 tok/s), i.e. 58.1 ms/token against a fitted "floor" of 84.24 ms. A result below its
model's floor falsifies the model.

The fault is the extrapolation, and it was visible in advance. `_expert_cache_ceiling.py` fits
`step = floor + traffic/bandwidth` from four arms spanning offload **36.18–41.48 GiB** — a
5.3 GiB window — and the floor is that line's intercept at *zero* traffic, roughly 36 GiB
outside the fitted range. Local linearity over 5.3 GiB (three intervals within 4.0%) was
verified and then treated as global, which it is not: working back from the measurement, the
real floor is **at most ~45 ms**, not 84 ms.

The second error compounds it in the same direction: the fitted 12.23 GB/s is the bandwidth of
**fine-grained uncoalesced UVA reads**, while an arena miss is a **bulk contiguous copy**.
`PREDICTION.md` flagged this as "the prediction is conservative on this point" — it was
conservative by about 1.7×.

**Consequence for the tooling:** `_expert_cache_ceiling.py`'s floor should be reported as an
extrapolated intercept with its fitted range stated, never as a ceiling. Its *hit-rate* input
remains sound — that part was predicted to 0.79 and measured 0.82.

## ⚠ Scope

One model, one card, one checkpoint, **B=1 single-stream, 256 decode tokens**. The arena path
is gated at 8 tokens/step because every expert routed in a step must be resident at once
(`n_slots >= max_tokens * top_k`), so this says nothing about concurrent serving. KV drops from
81,920 to 10,581 tokens to pay for the arena, which does not matter at a 4,096 window and B=1
but would at higher concurrency.
