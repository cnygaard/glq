# Trellis 3INST 5–8 bpw — CUDA decode results

Validation for the stacked-RVQ decode (stage 1 K=4 writes, stage 2 K=bpw−4 accumulates).
Raw logs are gitignored (`*.log`); reproduce with `benchmarks/_ppl_checkpoint.py` and
`tests/test_trellis_3inst_kernel.py`. Hardware: RTX PRO 6000 Blackwell (sm_120),
torch 2.11.0+cu128, transformers 5.14.0.

## Quality — SmolLM3-3B, wikitext-2 test, seqlen 2048, 141 windows

| | PPL | vs bf16 | safetensors |
|---|---|---|---|
| bf16 | **9.1220** | — | — |
| trellis 6 bpw (4+2), fused kernel | **9.1310** | +0.0090 (+0.10%) | 2516.55 MiB |
| trellis 6 bpw (4+2), pure-torch decode | **9.1297** | +0.0077 | *(same checkpoint)* |
| trellis 4 bpw (single stage) | **9.2299** | +0.1079 (+1.18%) | 1846.00 MiB |

Quantized with `--codebook trellis --nsamples 128`, `GLQ_TRELLIS_VARIANT=3inst`.

**Fused vs pure-torch is the decisive gate**: same checkpoint, same scales, only the kernel
differs — Δ 0.0013. Run it with `GLQ_FUSED_TRELLIS=0`, which is the *only* switch that
reaches the torch path (`GLQ_TRELLIS_DENSE` picks the dense branch **inside** the fused op
and still consumes `trellis_packed2`, so it cannot isolate the kernel).

The mechanism is confirmed by warning asymmetry, not just by the numbers: the
"no usable fused CUDA entry" `RuntimeWarning` fires on the pure-torch leg only. Two agreeing
PPLs would prove nothing if both legs had quietly fallen back to eager.

6 bpw closes **92%** of the 4 bpw → bf16 gap. `PPL(6) ≈ PPL(4)` would be the dropped-stage
alarm; the 0.099 separation says stage 2 contributes at the model level.

## Footprint — decomposes exactly

4→6 bpw delta = **670.55 MiB**. If stage 2 (K=2) is exactly half of stage 1 (K=4), then
stage 1 = 1341.10 MiB and the unquantized remainder is 504.90 MiB — against SmolLM3-3B's
tied embedding of 128256×2048×2 B = **501.0 MiB**, leaving ~3.9 MiB for SU/SV/norms/scales.
Back-solving gives 2.813e9 quantized weights and **exactly 4.00 and 6.00 bits/weight on
disk**. (Verify *bytes*, never nominal bpw.)

## Decode cost — B=1 GEMV, m=n=4096

| arm | time | vs 4 bpw | stage 2 alone | two-launch overhead |
|---|---|---|---|---|
| 1-stage K=4 (4 bpw) | 7.3 µs | 1.00× | — | — |
| 4+1 (5 bpw) | 14.5 µs | **1.99×** | 7.3 µs | −1.0% |
| 4+2 (6 bpw) | 13.9 µs | **1.91×** | 6.9 µs | −1.9% |
| 4+3 (7 bpw) | 14.0 µs | **1.92×** | 7.1 µs | −3.3% |
| 4+4 (8 bpw) | 13.7 µs | **1.89×** | 7.0 µs | −3.8% |

Stage 2's cost is **flat in K2** — K2=1 carries a quarter the bits of K2=4 and costs the
same ~7 µs — so the cost is per *state-decode*, not per bit. The ~1.9× is therefore doubled
decode ALU (stacked 4+2 decodes 512 states per 16×16 tile where a native K=6 would decode
256) and is irreducible: no fusion strategy avoids it. The two-launch overhead is ~0
(negative = noise), far below the ~10% that would justify a one-kernel variant.

**Do not conflate the two wall-clocks.** The PPL runs above show 6 bpw at 29 s vs 4 bpw at
24 s (1.21×), but seqlen 2048 drives B=2048 ≫ `GLQ_TRELLIS_BATCH_MAX=64`, so PPL exercises
the **dense prefill** branch (two decompresses + one fused add + one GEMM). The 1.9× is the
**B=1 decode** GEMV. Different paths; neither generalizes to the other.

## Test gates

300 on GPU + 138 local. On GPU: 28 stacked-RVQ gates (R=1 bit-exact decompress at every
residual rate, decode-vs-`W_hat` SQNR > 40 dB for bpw 5–8 × B ∈ {1,4}, stage-2-contributes,
zero-scale refusal, determinism, cudagraph capture), 192 regression proving 2/3/4 bpw is
untouched, 24 vLLM op-surface, 56 CPU suites.

Decode is gated against `W_hat` — the quantizer's own dequantized weight — never against
another decode. A fused-vs-eager A/B shares the scale wiring, so a stage dropped in *both*
legs passes it while quality collapses.
