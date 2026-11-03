# Kolibri-1 3 bpw with the `intuit/Kolibri-1-dspark` drafter on a 32 GB card

Model `xv0y5ncu/Kolibri-1-GLQ-trellis-3inst-3bpw`, drafter `intuit/Kolibri-1-dspark`.
- The drafter is bf16, 0.77 GiB on the Hub: 5 sliding-window Qwen3 layers that read Kolibri-1's
  layers 4/14/24/34/44.
- vLLM resolves it to `Qwen3DSparkModel` with no `--trust-remote-code`.

**Box, every run (2026-11-02):**
- AWS RTX PRO 4500 Blackwell, 32,623 MiB, uncapped, PCIe 5 x16.
- 30.07 GiB host RAM + 32 GiB swap.
- vLLM 0.29.0, aleph-alpha-inference 1.0.0, transformers 5.17.0, torch 2.13.0+cu130.
- glq 0.8.34 for the kernels and plugin. The `feat/kolibri-dspark` tree was on `PYTHONPATH`
  for the launcher runs.

## A/B: `run_model.py`, same flags, with and without the drafter (`kolibri_dspark_ab.sh`)

Both arms used the plan `glq-code --dspark` makes for this box (`kolibri_dspark_plan.log`):
- fp16;
- `--max-model-len 262144`;
- `--gpu-mem 0.902`;
- UVA offload of 18 GiB of experts;
- `GLQ_MOE_EXPERT_CACHE=4 GLQ_MOE_ARENA_MAX_TOKENS=5`.

Timing: B=1, 256 decode tokens, `ignore_eos`, temperature 0, 3 repeats after a 16-token warmup.

| arm | weights + arena (+ draft) | KV pool | TTFT | decode tok/s (range) | acceptance |
|---|--:|--:|--:|--:|--:|
| A, no drafter | 14.57 GiB | 508,810 tokens (1.94× the window) | 121 ms | 53.31 (53.18–53.32) | – |
| B, DSpark k=4 | 15.37 GiB | 452,852 tokens (1.73×) | 150 ms | **118.56 (118.52–118.69), 2.22×** | 55.5%: 611 of 1,100 draft tokens, 2.22 per draft |

- Both arms: `RUN_OK`, and `expert arena: 57 slots/layer x 50 layers = 3.93 GiB, ... serving
  steps up to 5 token(s)` on all 50 MoE layers, with 0 refusals.
- The SAMPLE (the harness's chat-template prompt about primary colours) opens a coherent
  `<think>` in both arms.
- The drafter added 0.80 GiB to vLLM's `Model loading took`.
- Arm A ran first, on a cold page cache (load 577 s against 385 s). Load time is not compared.
- One prompt, decoded greedily past EOS. This shows the mechanism engages and the scale of the
  gain on this box. It is not a benchmark of the drafter across workloads. Acceptance depends
  on the text.

## Launcher: `glq-chat --dspark`, headless (`kolibri_dspark_chat_serve.log`, `kolibri_dspark_session.*`)

- Plan: as above, with `--max-num-seqs 16`.
  - Startup message: "speculating 4 draft tokens per step with DSpark (intuit/Kolibri-1-dspark,
    0.77 GiB of VRAM)".
  - The arena cap is lowered from 80 to 9 tokens per step (57 slots / top_k 6). The 5-token
    single-stream verify step still fits.
  - Ready in 324 s. KV 515,625 tokens (1.97×).
- Three chat requests at the card's sampling (temperature 1.0, top_p 0.97, top_k 128), max 1,024
  tokens:
  - All three returned reasoning through the `kolibri1` parser.
  - Two finished with coherent answers. The third spent its 1,024 tokens reasoning.
  - 79.6 / 102.7 / 66.4 tok/s, prefill included.
  - 1,378 of 2,740 draft tokens accepted (50.3%, 2.01 per draft), from vLLM's
    `spec_decode_num_*_total` counters.
  - The script's own ratio missed vLLM's `_total` suffix, so these were worked out from the raw
    counters it printed.
- Teardown: SIGTERM to glq-chat → "stopping vLLM — the GPU is free again", and 0 MiB in use.
  - SIGINT did nothing here, because a job started with `&` from a non-interactive shell ignores
    it (`SigIgn` bit 2).

## Planning

`tests/test_dspark_planning.py::test_kolibri_on_a_32_gb_card_plans_what_served` pins this
plan, built from these inputs, for both launchers.
- The drafter is paid for out of the arena (5 → 4 GiB).
- The window and the offload are unchanged.
- The default KV price was enough: vLLM held 1.73× the window with the drafter loaded, so
  Kolibri needs no `_KV_COST_BY_FAMILY` entry.
