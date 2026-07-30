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
256) and is irreducible: no fusion strategy avoids it.

### The two-launch overhead is hardware-dependent — do not quote one number

On sm_120 it is ~0 (−1.0% to −3.8%, i.e. noise), which says a one-kernel variant would buy
nothing there. On **sm_89 (L4) the same microbench gives +10.5% / +12.7% / +15.3% / +21.1%**
at bpw 5/6/7/8 — above the ~10% that justifies fusing, and growing with K2:

| arm | sm_120 (RTX PRO 6000) | sm_89 (L4) |
|---|---|---|
| 4 bpw baseline | 7.3 µs | 27.8 µs |
| 4+2 (6 bpw) | 13.9 µs — **1.91×**, overhead −1.9% | 65.6 µs — **2.36×**, overhead **+12.7%** |
| 4+4 (8 bpw) | 13.7 µs — **1.89×**, overhead −3.8% | 75.8 µs — **2.73×**, overhead **+21.1%** |

The doubled-ALU floor is universal (stage-2 cost is flat in K2 on both). The launch/locality
penalty is not. Quote the decode cost per architecture.

**Mechanism not established.** The obvious L2-capacity explanation is falsified — the L4 has
48 MiB of L2 against 8+8 MiB of weight buffers — and the weight loads use `ld.global.cs`
(evict-first), which undercuts any L2-residency story either way. An ncu comparison of the
two-launch sequence against each pass alone would settle it. Note that recovering this needs
*per-tile* interleaving, which is the variant that spills at the 64-register wall; a
one-kernel phase-sequential variant is still two full sweeps and would not help.

Correctness is fully portable: 28 RVQ gates and 192 regression tests pass unchanged on
sm_89, including the R=1 bit-exact decompress.

**Do not conflate the two wall-clocks.** The PPL runs above show 6 bpw at 29 s vs 4 bpw at
24 s (1.21×), but seqlen 2048 drives B=2048 ≫ `GLQ_TRELLIS_BATCH_MAX=64`, so PPL exercises
the **dense prefill** branch (two decompresses + one fused add + one GEMM). The 1.9× is the
**B=1 decode** GEMV. Different paths; neither generalizes to the other.

## vLLM serving — SmolLM3-3B on L40S (sm_89)

vLLM 0.25.1, torch 2.11.0+cu128, transformers 5.14.0, fp16, `max_model_len=2048`,
`gpu_memory_utilization=0.85`, capture sizes 1–32. Both checkpoints come from the same
quantization run (`…-6bpw` and `…-4bpw-paired`), so this is a matched pair.
Driver: `benchmarks/run_model.py`; footprint is vLLM's own `Model loading took` line.

| | 4 bpw (1 stage) | 6 bpw (4+2) | ratio |
|---|---|---|---|
| weights | 1.81 GiB | **2.46 GiB** | 1.36× |
| KV cache | 495,776 tok | 486,240 tok | — |
| decode B=1 | 131.0 tok/s | **89.7 tok/s** | 0.68× |
| decode B=32 | 1768.9 tok/s (55.3/seq) | **1076.7 tok/s** (33.6/seq) | 0.61× |
| TTFT B=1 | 31 ms | 33 ms | — |
| load | 125.9 s | 128.9 s | — |

The footprint decomposes: +0.65 GiB for +2 bits over 2.813e9 quantized weights is 0.655 GiB.
Nothing else moved, which is what says the residual is resident and the rest of the model is
not silently dense.

`cudagraph_mode=FULL` was requested; vLLM downgraded it to `FULL_AND_PIECEWISE` itself
("not supported with FlashAttentionBackend"). Graphs were captured on both arms — 0.12 GiB
at 6 bpw, 0.10 at 4 bpw — and `run_model.py` fails the run outright if they are not, so the
tok/s above are on-graph numbers.

### Loader gate — vLLM ≡ HF, token for token

`benchmarks/_trellis_rvq2_vllm_hf_parity.py`, greedy, 64 tokens, same prompt **token ids**
fed to both runtimes:

| checkpoint | vLLM vs HF eager | greedy B=32 |
|---|---|---|
| 6 bpw (4+2) | **64/64 identical** | all 32 identical to each other and to B=1 |
| 4 bpw (control) | **64/64 identical** | all 32 identical to each other and to B=1 |

This is the check the unit tests structurally cannot make. They prove the *arithmetic* (both
vLLM apply paths are `torch.equal` to the shared HF staticmethod); this proves the **loader**
— that a real merged qkv/gate_up parameter puts each shard's `trellis_packed2` and its own
`inv_resid_scale2` in the right slot. A residual loaded one shard over still produces fluent
text. The two checkpoints diverge from *each other* by token ~8, so 6 bpw is demonstrably not
serving the 4 bpw weights.

### Stage-2 gate: the fused per-tile kernel is NOT worth building yet

The microbench's sm_89 two-launch overhead (+12.7% at 6 bpw) does survive end-to-end, but
small. Working backwards from B=1: 7.63 ms/token at 4 bpw vs 11.15 at 6 bpw is Δ 3.52 ms;
at the microbench's 2.36× that puts the trellis matvec at ~2.6 ms, ~34% of the step.
Removing the whole 12.7% would give ~95.7 tok/s — **+6.7%**.

That is the optimistic bound: it assumes every microsecond of the overhead is recoverable,
and the variant that would recover it is per-tile interleaving — the one the register
analysis says spills at the 64-register wall (`__launch_bounds__(1024,1)`, zero headroom).
A spill costs occupancy and would plausibly eat the whole 6.7%. 8 bpw is where the case is
strongest (+21.1% microbench), so if this is revisited, revisit it there, and run the ncu
pass first — the mechanism is still unknown and the obvious explanation is falsified.

## Test gates

300 on GPU + 138 local. On GPU: 28 stacked-RVQ gates (R=1 bit-exact decompress at every
residual rate, decode-vs-`W_hat` SQNR > 40 dB for bpw 5–8 × B ∈ {1,4}, stage-2-contributes,
zero-scale refusal, determinism, cudagraph capture), 192 regression proving 2/3/4 bpw is
untouched, 24 vLLM op-surface, 56 CPU suites.

Decode is gated against `W_hat` — the quantizer's own dequantized weight — never against
another decode. A fused-vs-eager A/B shares the scale wiring, so a stage dropped in *both*
legs passes it while quality collapses.
