# QTIP trellis coding → GLQ — verdict: **WIN, adopt** (all gates passed)

Integrating QTIP trellis-coded quantization (TCQ) as a 2-bit codebook for GLQ, validated
end-to-end vs GLQ's e8p. Motivation: deep-research + our codebook study flagged TCQ as the
top adoptable low-bpw lever (dimensionality, not a richer lattice). Pure PTQ, no fine-tuning.

Harness (Option B, pure scratch): `benchmarks/_qtip_trellis.py` (extracted QTIP `bitshift_codebook`,
GPL-3=GPL-3), `_qtip_ldlq.py` (GLQ `block_LDL`+LDLQ loop, block=16, tile-trellis codebook —
GLQ's LDLQ is provably identical to QTIP's BlockLDLQ), `_qtip_e2e.py` (independent-per-layer
quant → bf16 checkpoints), `_qtip_layer_proxy.py`, `_qtip_mse.py`, `tests/test_trellis.py`.

## Gates (all PASS)

| Gate | Test | Result |
| :-- | :-- | :-- |
| **codebook** | 2-bit iid+real-RHT'd MSE | e8p 0.091 → **qtip-3inst 0.069 (+1.2 dB)**; RD bound 0.0625 |
| **P0** unit (CPU, TDD) | tiling / roundtrip / feedback / MSE anchor | **6/6** |
| **P1** layer proxy (6 real SmolLM3 layers) | trellis-2b vs e8p-2b | **6/6 layers, median +1.32 dB** (SQNR uniformly higher too) |
| **P2** e2e PPL (SmolLM2-360M) | dequant→bf16→wikitext-2 | trellis **20.61** < e8p **27.97** (bf16 11.47) |
| **P3** e2e PPL (SmolLM3-3B) | same | **trellis 11.86 < e8p 13.79** (bf16 9.12) |

## Why it works (consistent with `_codebook_sqnr_study.md`)
The +1.2 dB codebook win comes from **quantization dimensionality** — the trellis reaches
effective dim 256 vs E8's 8, lookup-free — breaking past our SQNR study's lattice ceiling
(E8→∞ only 0.88 dB, which is why lattice swaps and learned lattices [GLVQ, `_glvq_study.md`]
failed). That +1.2 dB survives the full RHT+LDLQ pipeline (P1 +1.32 dB) and compounds into a
**1.93 PPL** e2e win at 3B (P3). Trellis wins on BOTH proxy and SQNR (unlike GLVQ's
proxy-only, H-null-space-exploiting artifact) → the gain is real, not a metric mirage.

## Method (verified from `qtip/`)
Per layer: GLQ RHT of W,H → scale `Wr /= rms/(rms(lut)·0.9)` → GLQ `block_LDL(H,16)` → LDLQ
reverse sweep over 16-col blocks with `R@L` feedback, each block's (m,16) residual reshaped
into 16×16=256-element tiles that the trellis quantizes as tail-biting Viterbi sequences
(K=2 → 2 bpw; lookup-free `3inst` variant won). GLQ's LDLQ == QTIP's BlockLDLQ (both
identity-diagonal-block L; feedback reads only rows below the block, so QTIP's diag-zeroing
is a no-op) — so GLQ's machinery is reused verbatim, only the codebook changes.

## Caveats / scope
- **Cost:** the pure-torch Viterbi is slow — SmolLM3-3B trellis quant ~1–2 GPU-hr (vs ~min for
  e8p). A one-time quantization cost; QTIP's CUDA Viterbi would cut it. Decode is the *fast*
  direction (`G@z`-free, ≤4 instr/weight) but NOT yet built here.
- **Independent-per-layer** (no sequential GPTQ prop) — conservative; production sequential
  quant (Option A, glq driver) would only *widen* the trellis lead. Absolute PPLs sit above a
  sequential run for BOTH arms equally.
- **bf16-dequant gate** discards TCQ's compressed storage + lookup-free decode — those are the
  productization payoff, deliberately out of scope for the quality question.
- Pure PTQ, no fine-tuning (QTIP's default adds per-layer FT, a further lever).

## Verdict → next
**Adopt TCQ.** It's a real 2-bit quality win over e8p AND (once the decode kernel is built) a
speed win (removes the L2 codebook-gather ceiling).

---

# Productization S0 — **DONE** (shipped-format gate PASSED)

The gates above used a bf16-dequant scratch harness (quality only). S0 makes it **real**:
compressed storage + HF load, end-to-end through the production path.

Ship variant = **HYB** (`quantlut_sym` V=2 tlut_bits=9) — QTIP's *only* variant with a shipped
CUDA kernel (`has_kernel`). Costs ~0.1 PPL vs our best (3inst), still crushes e8p.

## S0 gates (all PASS)

| Gate | Test | Result |
| :-- | :-- | :-- |
| **storage round-trip** | quantize → `pack_layer`(kernel layout) → pure-torch `decode_layer` | **bit-exact** (`tests/test_trellis_storage.py`) |
| **layer inference** | compressed buffers → `E8RHTLinear` forward vs dense `W_hat` | match (`tests/test_trellis_layer.py`) |
| **HF factory** | one shared `TrellisCodebook` rebuilt from stored tlut + config | pass (`tests/test_trellis_hf.py`) |
| **regression** | full suite incl. e8p/shell/quant/hf paths | **83 passed, 6 skipped** |
| **e2e 135M** (production CLI) | `--codebook trellis` → safetensors → `from_pretrained` → PPL | **trellis 34.89 < e8p 45.56** (bf16 15.43); 83 vs 84 MB |
| **e2e 3B** (production CLI) | same | **trellis 11.74 < e8p 13.21** (bf16 9.12); **both 1.2 GB** |

`benchmarks/_qtip_s0_gate.sh` is the box runner.

**Trellis beats e8p by 1.47 PPL at IDENTICAL footprint** on SmolLM3-3B, closing 36% of e8p's
degradation vs bf16. Footprint is genuinely 2 bpw (135M: 83 MB = 26 MB 2-bpw linears + 57 MB
bf16 embeddings — the compression is real, not a dequantized shell).

**Honest caveat:** the production *sequential* driver improved BOTH arms over the scratch
independent-per-layer harness (trellis 11.94→11.74, e8p 13.79→13.21), so the gap narrowed from
1.85 to 1.47. The win is smaller than the scratch implied, but decisive and now on the real format.

---

# Productization S1 — **DONE** (CUDA kernel; all 9 gates green)

`glq/csrc/glq_trellis.cu` ports QTIP's `kernel_decompress_matvec` with two deliberate
deviations: **runtime-generic** (m,k are kernel args, not template params — GLQ serves
arbitrary models from one build) and **cudagraph-safe** (upstream's per-launch
`cudaGetDeviceProperties` + `cudaFuncSetAttribute` hoisted into a one-time `call_once`).

A **decompress kernel** sharing the matvec's exact decode path is the keystone: it gives a
`torch.equal` oracle (the tlut is stored fp16, so the kernel's fp16 LUT values are bit-identical
to torch's), and doubles as the B>1 prefill fallback.

| Gate | Result |
| :-- | :-- |
| decompress vs `decode_layer` (3 shapes) | **BIT-EXACT** (`torch.equal`) |
| B=1 GEMV vs `x @ W.T` | SQNR > 40 dB |
| determinism | bit-stable (block-owns-m + smem reduce, no atomics) |
| fused linear vs S0 dense reference (B=1, B=4) | SQNR > 35 dB |
| **`torch.cuda.CUDAGraph` capture + replay** | **PASS** |
| real-model PPL (SmolLM3-3B) | **11.7450** via kernel ≈ **11.7424** pure-torch |
| **VRAM** | **1176 MiB vs bf16 5866 → 5.0×** (weights stay compressed) |
| **B=1 decode** | **48.2 tok/s vs e8p 36.8 → 1.31×** |

So vs e8p at 2 bpw: **better quality (11.74 < 13.21), faster decode (1.31×), same footprint.**

**Caveats.** (1) Trellis peak VRAM 1315 MiB > e8p 1187 — the B>1 path decompresses to a dense
**fp32** W transiently; e8p uses a compressed batched GEMM. Fix = fp16 W or a batched trellis
GEMM. (2) HF-eager 48 tok/s is Python-dispatch bound, and bf16 is still faster (73.8) on a 96 GB
Blackwell — GLQ's speed win lives on 24/32 GB cards and under vLLM cudagraphs (S2).
(3) Kernel needs `m % 32`, `k % 64`; other shapes fall back to the pure-torch decode.

---

# Productization S2 — **DONE**: vLLM serving + batched GEMM. **Speed gate FAILED.**

Two things shipped: a **batched trellis GEMM** (the B=1 GEMV filled only column 0 of the
`m16n8k16` tile and discarded 7/8 — the batched fork puts one token per N-column, so B≤8 costs
the same decode and the same tensor-core work; ragged tail predicated, not padded) with a
hybrid dispatch (`B=1` GEMV → `B≤64` batched → `B>64` decompress-fp16 + cuBLAS, so no dense
weight exists on any *decode* step while prefill keeps the fast cuBLAS path); and the full vLLM
wiring (op triple + fake, `config.variant`, full-size `create_weights`, `_setup_trellis_weights`,
`_glq_apply_trellis`).

Deliberately **not** copied from e8p: its split-K scratch uses a per-call
`raw_alloc`/`raw_delete`, which `glq_cuda.cu:3405-3412` itself flags as illegal during graph
capture. Our in-block reduce (each block owns a disjoint m-range; no atomics, no scratch) is
deterministic *and* capture-safe by construction.

## Gates

| Gate | Result |
| :-- | :-- |
| batched GEMM vs `x @ decompress(packed).T` (B=1,2,7,8,9,63,64,65) | SQNR > 40 dB |
| **batched row `b` vs the B=1 GEMV on `x[b]`** | **BIT-EXACT** (`torch.equal`) |
| determinism, cudagraph capture/replay | pass |
| vLLM serve, **FULL cudagraph** (no `enforce_eager`) | coherent |
| PPL (3B) | 11.7448 (unchanged) |
| **peak VRAM** | **1315 → 1186 MiB** — the dense-materialization wart is gone (e8p: 1187) |
| **decode tok/s > e8p** | ❌ **FAILED** |

## SmolLM3-3B @ 2 bpw — vLLM, FULL cudagraph

| | PPL | weights | peak VRAM | B=1 | B=32 total |
| :-- | --: | --: | --: | --: | --: |
| e8p | 13.21 | 1176 MiB | 1187 MiB | **150.9 tok/s** | **3173 tok/s** |
| **trellis** | **11.74** | 1176 MiB | **1186 MiB** | 147.9 | 2366 |

## ⚠️ Correction to S1

**The S1 claim that trellis decodes "1.31× faster than e8p" was WRONG** — an artifact of
HF-eager being Python-dispatch-bound. Under cudagraph, e8p gained 4.1× (36.8→150.9) while
trellis gained 3.0× (48.8→144.5): e8p was simply penalised *more* by Python overhead. Once both
are freed by graph capture — the setting that actually matters for serving — **e8p's kernel is
the faster one.** *Never generalise an eager-mode tok/s number.*

**The real proposition: trellis buys better quality (−1.47 PPL) at IDENTICAL footprint, for a
~2% (B=1) / ~25% (B=32) decode-throughput cost.** Not "better and faster".

The cost is intrinsic, not a tuning miss: trellis decode is a smem LUT gather + `idx*(idx+1)` +
sign-flip per 2 weights, vs e8p's cheaper packed-int32 bit-expand; and the 64 KB LUT + 1024
threads caps occupancy at 1 block/SM (67%). We *did* find a real defect — QTIP hardcodes
`grid.x=128` (a ~108-SM A100), leaving 60 of this card's 188 SMs idle — and made it SM-adaptive,
but it bought only +2%/+1%, which **disproves** the under-subscription theory rather than
confirming it.

## ⚠️ Scope limit on the S2 speed loss — see S3

Everything above was measured at **2 bpw**, which is the *one* rate where e8p is single-stage.
S3 shows the comparison **inverts at 4 bpw**, and the "S4/3inst is the high-value lever"
conclusion that used to sit here was drawn from this too-narrow measurement. It is superseded.

---

# Productization S3 — **native K=3/4. The 4-bpw gate: trellis WINS on quality AND speed.**

The trellis has a **native rate K**; e8p reaches 3-8 bpw by **RVQ stacking**, which runs one full
decode+matmul pass per stage (`glq_cuda.cu:2902`: *"matmul linearity lets us decode each stage
independently and sum"*). And `tr_decode_regw<R>` loops `j=0..3` for **every** R — always 4 half2
from 4 states — so LUT gathers, sign-flips and mma count are identical at K=2 and K=4; only the
bytes read per lane double.

| bpw | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| e8p decode passes | 1 | 2 | 2 | 3 | 3 | 4 | 4 |
| trellis decode passes | 1 | **1** | **1** | 2 | 2 | 2 | 2 |

## The mechanism, confirmed end-to-end (SmolLM3-3B, vLLM FULL cudagraph)

**Trellis decode is rate-independent; e8p pays ~linearly in bpw.** This is the whole thesis:

| B=1 decode | @2 bpw | @4 bpw | change |
| :-- | --: | --: | --: |
| **trellis** | 147.9 | **146.9** | **−0.7% — flat** |
| e8p | 150.9 | 121.9 | **−19%** |

| B=32 total | @2 bpw | @4 bpw | change |
| :-- | --: | --: | --: |
| **trellis** | 2366 | **2388** | **+1% — flat** |
| e8p | 3173 | 2119 | **−33%** |

## S3 gate — SmolLM3-3B @ 4 bpw (all PASS)

| | PPL (wikitext-2, 141 windows) | on-disk | B=1 | B=32 | quant time | SQNR |
| :-- | --: | --: | --: | --: | --: | --: |
| bf16 | 9.122 | — | — | — | — | — |
| **trellis K=4** | **9.245** | 1.9 G | **146.9** | **2388.5** | 29.0 m | 21.10 dB |
| e8p 4bpw | 9.329 | 1.9 G | 121.9 | 2118.8 | **4.5 m** | 18.81 dB |
| trellis K=3 | 9.651 | 1.5 G | — | — | 29.6 m | 15.94 dB |

**At 4 bpw — GLQ's production operating point — trellis beats e8p on quality AND speed at
identical footprint:** −0.08 PPL, **+20.5% B=1**, **+12.7% B=32**. PPL is monotone in rate
(2/3/4 bpw = 11.74 / 9.65 / 9.25) and K=3 is a genuine 3-bpw footprint (1.5 G vs 1.9 G).
Both arms generate coherently. *Scope: one model, wikitext-2 PPL, single run — not yet a
downstream-quality claim (see Next).*

**Prediction scorecard — direction right, magnitudes wrong.** I predicted the win would show at
B=32 and that B=1 would be near parity ("both DRAM-bound, both read 4 bpw of weight bytes").
The opposite: **B=1 wins more (+20.5%) than B=32 (+12.7%)**. B=1 is *latency*-bound, so e8p's
second pass costs a whole extra round of launches, `x_rht` re-reads, `y_rht` read-modify-write
and split-K scratch+reduce. At B=32 the mma work matters more, and trellis's LUT-gather decode
is intrinsically pricier per weight than e8p's bit-expand — so halving the pass count nets
+12.7%, not the ~2× a naive pass-count argument implies.

## Kernel gates: R=3 / R=4 executed for the FIRST TIME (100 passed / 0 failed)
The `R=3`/`R=4` branches had been compiled and smem-configured since S1 but **never launched**.
Both are **bit-exact** (`torch.equal`) vs the pure-torch `decode_layer` oracle, at 3 shapes each.
The risk was the per-lane load width (`2R` u16 → uint2/uint3/uint4) failing to line up with the
packed width (`16R`); it lines up. `test_decode_cost_is_rate_independent` also passes: a K=4
GEMV is not ~2× a K=2 GEMV.

## Two real defects fixed (both would have shipped SILENTLY WRONG checkpoints)
1. **`--codebook trellis --bpw 3` crashed at model load** — `hf_integration`'s shell RVQ chain
   fell through to `codebook.make_small(256)`, which `TrellisCodebook` doesn't have. The trellis
   is single-stage and needs no residual grid; it is now branched out explicitly. K=4 had only
   worked *by accident* (its `>=4` branch aliased the codebook harmlessly). **Only the K=3
   parameterization could have caught this** — every prior trellis test pinned K=2.
2. **Mixed-bpw trellis was silently wrong** — `--bpw-map` collapsed every layer to K=2 and
   `--bpw 3.5` to K=3, while `config.json` still advertised the per-layer map; vLLM sizes
   `trellis_packed` from that map, so it would mismatch and strand `param.data` on CPU. Now a
   hard `ValueError`, plus a bpw 2-4 bound check (a `--bpw 5` run would otherwise Viterbi for
   GPU-hours and only fail at serve).

## Quantization cost — trellis is ~6.5× slower to quantize (29.0 m vs 4.5 m at 3B)
Not the 1-2 GPU-hr the S0 note implied. **K=3 and K=4 take the same time (29.6 vs 29.0 m)** even
though `from_state` is 4× smaller at K=4 (2^8 vs 2^10 wide) — so that buffer's bandwidth is *not*
the bottleneck, and the "shrink `from_state`" lever is worth little. The Viterbi is
**launch/compute-bound**: `trellis_ldlq` issues `(n/16) × 2 × ceil((m/16)/bs) × 127` sequential
`update` calls (~16 M for a 3B model), each on a *small* batch (`B = m/16` = 128-256 vs 65536
states) that cannot saturate 188 SMs. Levers, best first:
1. **CUDA-graph the 127-step Viterbi loop** (shapes are static per chunk) — kills the launches;
   the only thing that helps `down_proj`, whose `B=128` cannot grow.
2. **Raise `bs = min(2**(24-L), NO)`** — a small-VRAM-era cap; on 96 GB, wide layers (gate/up/
   down = most of the params) needlessly split into 3 serialized Viterbi chunks.
3. `state_err` as a GEMM: `‖r_s−t_b‖² = ‖r_s‖² − 2·r_s·t_b + ‖t_b‖²`; `‖r_s‖²` is **constant**
   (precompute once), the cross term is a (65536×2)@(2×B) tensor-core GEMM, and the 128 MiB
   `(2,B,65536)` transient disappears. Measure first — `@torch.compile` may already fuse it.
4. Row-stack q/k/v and gate/up before LDLQ — LDLQ is **row-independent** and they share `H`, so
   it is mathematically identical: 3×/2× the batch, ⅓ the launches.

*(QTIP does **not** ship a CUDA Viterbi — `qtip-kernels/src/` is decode-side only.)*

## MMLU-Pro downstream gate (SmolLM3-3B @4bpw, n=1811 = 15%, vLLM + chat template)

| | MMLU-Pro | 95% CI |
| :-- | --: | --: |
| bf16 | 40.3% | ±2.3 |
| **trellis-4** | **38.9%** | ±2.3 |
| e8p-4 | 37.0% | ±2.2 |

**Serving-runtime side-note (same 1811-prompt run, `batch_size=auto`, prefill-heavy):** total
wall-clock bf16 **2:44** / trellis-4 **5:19** / e8p-4 **13:27** — i.e. e8p is **2.5×** trellis,
not the 2× its extra RVQ pass predicts. The extra 0.5× is a *separate* e8p deficit: at high
batch trellis routes to decompress→fp16→**one cuBLAS GEMM**, while e8p has no dense-prefill path
and runs its custom TC-GEMM **twice**. So the S3 decode win (+12.7% B=32) *widens* to ~2.9×
output-tok/s at serving batch. (bf16 fastest — expected on a 96 GB card; GLQ's big-GPU edge is
footprint, per [[glq_vs_bf16_rtx6000]]. Confound: aggregate throughput on one workload, not a
controlled bench. The actionable item is an e8p prefill fast-path, not a trellis change.)

**trellis-4 − e8p-4 = +1.9 pts → WITHIN NOISE (band ~4.5).** A NULL result on task accuracy,
not a win: the +1.9 is consistent with the PPL edge but well inside the band. What it *does*
establish — the thing that mattered — is that **neither 4-bpw arm collapses**: both are 1.4-3.3
pts under bf16 (also within noise), so there's no reasoning-cliff like gemma-4-12B @3bpw. Caveat:
3B is a weak base (bf16 only 40%), so MMLU-Pro here has little power to resolve a small gap and
says nothing about hard reasoning. Checkpoints on private HF `xv0y5ncu/SmolLM3-3B-{trellis,e8p}-4bpw-s3`.

## Next
- **Flagship quality still the deciding test.** The 3B MMLU-Pro cleared the collapse risk but is
  underpowered. The genuinely decisive gate is **gemma-4-12B, trellis-4 vs bf16, MMLU-Pro + AIME
  in thinking mode** — that's what would change a default or ship. (PPL and even MMLU-Pro have
  looked fine here while AIME collapsed: `gemma4_12b_glq3bpw_e8p_aime_collapse`.)
- **3inst (old S4) is deprioritized**: trellis already wins at 4 bpw without it, and it is a
  from-scratch storage↔kernel co-design (V=1 vs the shipped kernel's hardwired V=2 — its smem LUT
  entries *are* `half2`), with no reference implementation to check bit-exactness against.
- **Trellis RVQ (5-8 bpw)** — now more attractive: 2 passes where e8p needs 3-4. Gate on demand.
- MoE trellis (no grouped kernel exists).

---

# Productization S4-speed — CUDA-graph the Viterbi (quant ~4× faster, bit-exact)

Trellis *quantization* was ~6.5× slower than e8p (SmolLM3-3B 29 min vs 4.5 min). nsys pinned the
encoder-side Viterbi as **launch-bound**: ~380k kernel launches for 5 layers, **96% of CUDA-API
time is launch+memcpy dispatch**, GPU busy only ~17%. Fix: capture the whole `viterbi` into a CUDA
graph per `(T, B, has_overlap)`, replay per column-block. Encode-only (decode never calls it),
deterministic → replay is `torch.equal` to eager, the gate.

## Results (SmolLM2-135M shapes, K=4, RTX PRO 6000)

**Launch collapse (nsys, 5-layer replay pass):** eager ~440k dispatch calls (255k cudaLaunchKernel
+ 122k cuLaunchKernel + 62k cudaMemcpyAsync) → **4.7k cudaLaunchKernel + 480 cudaGraphLaunch**.
The 480 = exactly the Viterbi-call count (72+72+72+72+192 over the 5 shapes) — each viterbi is now
**one graph replay** in place of ~600 kernel launches.

**Wall-clock A/B (per-layer `trellis_ldlq`, steady replay):**

| shape | eager ms | graphed ms | speedup |
| :-- | --: | --: | --: |
| q/o_proj (576²) | ~790 | ~173 | 4.3–4.9× |
| k/v_proj (192×576) | ~838 | ~108 | **7.8×** |
| gate/up (1536×576) | ~836 | ~318 | 2.6× |
| down (576×1536) | 2231 | 460 | 4.9× |
| **1 layer (7 linears)** | **7160** | **1659** | **4.32×** |

**Projected full SmolLM2-135M Viterbi-quant: 214.8 s → 52.2 s = 4.1×** (incl. 2.4 s one-time
capture). **End-to-end confirmed: real SmolLM2-135M trellis-4bpw quant 4.2 min → 1.3 min = 3.2×
(210 sublayers), and the two checkpoints are BYTE-IDENTICAL (sha256 `5a1bc5d2…`)** — the CUDA
graph produces a bit-identical quant, just faster (3.2× e2e < 4.1× Viterbi-only because RHT /
block_LDL / embedding stay eager). All 25 GPU parity tests green
(`tests/test_trellis_cudagraph.py`), bit-exact `torch.equal` graphed-vs-eager across K∈{2,3,4} ×
B∈{12,36,256} × overlap∈{None,tensor} and full-layer `trellis_ldlq`.

## Two bugs the first run exposed (both now fixed)
1. **Capture silently failed** — the overlap tail-biting pass did `torch.arange(2^KV).to(X.device)`,
   a CPU→CUDA copy that's illegal during capture. Only the overlap=False graph captured; a global
   disable flag then forced everything eager → 0.94× (no win). Fix: hoist that constant to a device
   buffer (`_kv_arange`). Bit-exact.
2. **A test blind spot hid #1** — the parity path fell back to bit-exact eager on capture failure,
   so all 24 parity tests passed while NO graph ran. Fix: per-key `None`-sentinel fallback + a loud
   `RuntimeWarning` (not a silent global disable), and a new test that asserts a live `_VitGraph`
   was captured for BOTH passes. **Lesson: a test that can pass via a bit-exact fallback proves
   nothing about whether the optimized path engaged — assert the mechanism, not just the output.**

## Implementation (glq/trellis.py, encode-only)
`bitshift_codebook`: `_vit_graphs` dict + shared `graph_pool_handle()` + `threading.Lock` (MoE
experts quantize on a ThreadPoolExecutor sharing one codebook; capture is process-global/exclusive);
`_capture_viterbi` (side-stream warmup compiles `update` + primes allocator, then capture) +
`_viterbi_graphed` (copy-in→replay→clone-out); `quantize_seq` routes to it when `X.is_cuda`. Kept
`@torch.compile` on `update`. Env `GLQ_TRELLIS_NO_CUDAGRAPH` + `GLQ_TRELLIS_CUDAGRAPH_MAX_B` gates.
