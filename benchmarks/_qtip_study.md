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

## Remaining (S3, S4)
**S4 (3inst-inline kernel) is now the high-value lever, not a nicety:** 3inst is **lookup-free**
(3 integer ops, no tlut) → deletes the 64 KB smem, lifts occupancy, *and* reclaims ~0.1 PPL. It
is the one change that could plausibly close the speed gap. · S3 sequential requant +
MMLU-Pro/AIME + release.
