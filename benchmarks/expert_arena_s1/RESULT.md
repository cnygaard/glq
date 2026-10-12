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

## MTP speculation, arena OFF — the earlier finding reproduces

Measured on the way to the arena+MTP arms, when a 6 GiB arena was **correctly refused** (see
below) and all three arms therefore ran with no arena. `--gpu-mem 0.90`, offload 41.48 GiB:

| arm | tok/s | range | mean acceptance length | vs no-spec |
|---|--:|--|--|--:|
| no speculation | 6.436 | 6.436–6.436 | — | 1.00× |
| `num_speculative_tokens=3` | 7.384 | 7.177–7.450 | 1.91–2.70 | 1.147× |
| **`num_speculative_tokens=1`** | **7.743** | 7.743–7.929 | 1.56–1.72 | **1.203×** |

**1 still beats 3 under offload without the arena**, reproducing the earlier result: step cost
is steeply linear in drafted positions, so the extra acceptance that 3 buys does not cover the
extra positions. `num_spec=3` also costs more KV (34,474 vs 54,016 tokens) and more capture
(0.91 vs 0.60 GiB).

### ★ The arena refused this configuration, and was right to

The 6 GiB budget over **49** MoE containers is 71 slots/layer, against a cap of 8 tokens ×
top_k 10 = 80 simultaneous routings:

```
expert arena disabled: an arena of 71 slots/layer cannot serve 8 tokens x top_k 10 = 80
SIMULTANEOUS routings. Every expert routed in a step must be resident at once...
```

Serving it anyway would have evicted a routing before the matmul read it. The refusal names
its arithmetic, which is the only reason the arms were recognisable as no-arena runs at all —
their rates and `weights_gib=6.85` would otherwise have looked like a plausible arena result.

The fix is to match the cap to the step width rather than to enlarge the arena: an MTP verify
step carries `1 + num_spec = 4` tokens, so `GLQ_MOE_ARENA_MAX_TOKENS=4` brings the requirement
to 40 ≤ 71 and leaves the extra VRAM to KV, which MTP needs.

### Two things that stopped arena+MTP before this

* **`VLLM_LOGGING_LEVEL=DEBUG` breaks CUDA-graph capture on the speculator path.** vLLM's
  DEBUG formatter stringifies a tensor (`vllm/ir/op.py` → `tensors_str_no_data` →
  `Tensor.__repr__`), which needs a device→host copy and is illegal mid-capture:
  `cudaErrorStreamCaptureUnsupported`. The *reported* failure was an unrelated `rms_norm` in
  the draft head with no GLQ frame, because the message is "operation failed due to a
  **previous** error during capture". The first error in the log, 300k lines earlier, named
  the real one. DEBUG is therefore a cause, not just an observer, when chasing a capture bug.
* **An 8 GiB arena plus MTP does not fit** at `--gpu-mem 0.85`: `Available KV cache memory:
  -0.31 GiB`. Only just — which is why the retry raises the budget to 0.90 and shrinks the
  arena rather than abandoning the pairing.

## Arena + MTP: the best combination is arena + `num_spec=1`, at 3.060×

6 GiB arena (71 slots/layer × 49), `GLQ_MOE_ARENA_MAX_TOKENS=4`, `--gpu-mem 0.90`, offload
41.48 GiB. The no-spec arm at the same arena size and budget is the reference, so the
speculation effect is not confounded with arena size or memory budget.

| config | arena | tok/s | range | ms/token | mean acceptance | vs baseline |
|---|---|--:|--|--:|--:|--:|
| baseline | off | 6.436 | 6.436–6.436 | 155.4 | — | 1.00× |
| mtp3 | off | 7.384 | 7.177–7.450 | 135.4 | 2.16 | 1.147× |
| mtp1 | off | 7.743 | 7.743–7.929 | 129.1 | 1.62 | 1.203× |
| arena | **on** | 16.816 | 16.795–17.065 | 59.5 | — | 2.613× |
| arena + mtp3 | **on** | 18.569 | 17.556–18.598 | 53.9 | 2.16 | 2.886× |
| **arena + mtp1** | **on** | **19.692** | 19.681–19.912 | **50.8** | 1.62 | **3.060×** |

### ★ `num_spec=1` still wins, and that is a finding about where the time goes

The hypothesis under test was that the optimum flipped to 1 *because* step cost was steeply
linear in drafted positions, dominated by expert PCIe traffic — so removing that traffic
should have made 3 affordable again. It did not: 19.692 vs 18.569, the same ranking as with
the arena off (7.743 vs 7.384).

**So the per-position cost that makes 1 win is not mostly expert paging.** The arena removes
about two thirds of the step (155.4 → 59.5 ms) and the ranking is unchanged, which places the
remaining per-position cost in the rest of the step: trellis decode for the routed experts,
the GDN/linear-attention blocks, attention and the dense layers. That is where the next piece
of work has to go, and it is the same conclusion the falsified ceiling pointed at from the
other direction.

### The two techniques overlap rather than compound

Speculation is worth **less** once the arena is in place:

| | arena off | arena on |
|---|--:|--:|
| mtp1 over its own no-spec | 1.203× | 1.171× |
| mtp3 over its own no-spec | 1.147× | 1.104× |

Part of what speculation buys is amortising one weight fetch across several tokens, and the
arena has already removed most of that fetch. They still combine usefully — 3.060× together
against 2.613× and 1.203× apart — but not multiplicatively.

### ⚠ MTP is not byte-identical here; the arena is

| arm | sample md5 |
|---|---|
| baseline, iso-VRAM control, arena 8 GiB, arena 6 GiB | `1e44e07f7d9a6c450a8d95492132dcdb` |
| arena + mtp3 | `c5d54df0e516c283a5240fc591aa7c30` |
| arena + mtp1 | `2f2be2fe8b4651fec5142572c6c15ed2` |

**Every non-speculative arm agrees**, including both arena sizes — so the arena is
byte-identical, confirmed a third time at 6 GiB. The speculative arms each differ, and they
differ with the arena off too, so this is MTP's property rather than the arena's: a verify
step computes logits for several positions in one forward, and a batched GEMM does not reduce
in the same order as a single-position one, so greedy decode can diverge after enough steps.

Worth stating because speculative decoding is often described as lossless at temperature 0.
It is lossless in *distribution*; it is not bit-identical in this implementation. Any quality
claim about an MTP arm therefore needs its own eval, not an appeal to exactness.
