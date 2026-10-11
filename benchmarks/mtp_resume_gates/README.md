# Gates for the MTP draft head + resume work

Raw logs for the validation behind the MTP post-pass, the resume-replay expert fix, and the
`--streaming` rotary fix. Run 2026-10-10 on a g7e.4xlarge (RTX PRO 6000 Blackwell, sm_120),
transformers 5.17.0 / torch 2.13.0+cu130 / vLLM 0.31.0, CUDA 13.2.

Both arms ran the **same command**, with only `PYTHONPATH` selecting the source tree — glq was
installed non-editable, so no reinstall sits between the arms and nothing but the code differs:

* arm A `/opt/dlami/nvme/glq_head` — git HEAD, `5c145c1`
* arm B `/opt/dlami/nvme/glq_src` — `release/0.8.26` + this work

## 1. Byte-identity: a checkpoint with no MTP head must be unchanged

`bytecmp_nonstreaming.log` · `gate_head.log` (arm A) · `gate_branch.log` (arm B)

SmolLM3-3B, trellis-3inst, 3 bpw, `nsamples 128`, non-streaming.

```
keys equal: True (1082 vs 1082 tensors)
BYTE-IDENTICAL
```

`config.json` and `layer_metrics.json` identical too; both arms report `Average SQNR = 16.19 dB`
over 252 matrices. The only difference anywhere is `quantize_config.json`'s `total_time_s`
(884.2 vs 876.2) — wall-clock, which differs between any two runs.

This is the gate every quantize-path change has to clear, and it is why the post-pass is
**opt-in** behind `--mtp-bpw`: with the flag absent nothing calls the new code, so the result
holds by construction and the measurement confirms it.

## 2. The `--streaming` fix, with a control

`stream_head.log` (control) · `stream_branch.log`

A default-profile arch (Llama / SmolLM2 / SmolLM3) never had a rotary built under
`--streaming`, because the capture was gated on `profile['rotary_attr']` and only the *named*
profiles set one. Arm A, layer 1:

```
  File ".../transformers/models/smollm3/modeling_smollm3.py", line 223, in forward
    cos, sin = position_embeddings
TypeError: cannot unpack non-iterable NoneType object
```

Arm B, same command, reached layer 17/35 at SQNR 15.4–16.1 dB and stopped only on the
harness's own `timeout`. **The control is the point**: without arm A, "streaming works now" is
a claim about a path nobody had watched fail.

The fix routes through `get_rotary_emb`, which already encoded the contract — its guard
discriminates on `layers_attr`, so NemotronH (`rotary_attr: None` **with** `layers_attr`) keeps
getting `None` deliberately, while the default profile (both `None`, comment: *"use heuristic"*)
falls through as always intended. Only archs that previously crashed change behaviour.

## 3. Resume + `--streaming` reproduces an uninterrupted run byte-for-byte

`resume_gate.sh` (the driver) · `resume_gate.log` · `r1.log` (the uninterrupted reference)

SmolLM3-3B, 3 bpw, `nsamples 8`, `--streaming --resume`. Phase 1 runs and is stopped part-way
(standing in for a spot reclaim); phase 2 restarts the identical command; the result is compared
against a reference run that was never interrupted.

```
PHASE1_EXIT=137                                 (stopped on purpose)
  resume: 7 layer(s) checkpointed, replaying to layer 7
  "replayed from checkpoint" x 7
Quantized 203 sublayers in 11.1m                (252 - 7x7 = 203, exactly)
PHASE2_EXIT=0
keys equal: True (1082 vs 1082 tensors)
BYTE-IDENTICAL
```

The replay count and the sublayer arithmetic are both asserted, not inferred — a resume that
silently re-quantized everything would also have produced an identical checkpoint, so the
byte verdict alone could not distinguish the two.

## What these do NOT cover

* **A MoE resume.** SmolLM3 is dense, so every restored matrix is an `nn.Linear`. The fix for
  stacked routed experts (3-D Parameters, which the old `isinstance(mod, nn.Linear)` walk
  skipped entirely — leaving the experts at bf16 through the replay forward) is covered only by
  `tests/test_resume_replay_experts.py`. A real MoE replay is still unmeasured.
* **The MTP post-pass end to end.** No checkpoint has yet been produced with `--mtp-bpw`; the
  head's only published home is Qwen3.8-Flash-Next, which is a 77 GiB download and hours of GPU
  time. Everything here establishes that the path is *inert when unused*, not that it works
  when used.
* **Serving a quantized draft head.** The acceptance rate and the decode gain against the
  6.271 tok/s L4 baseline remain to be measured.
