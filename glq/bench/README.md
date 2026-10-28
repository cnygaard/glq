# glq-bench — model-performance benchmarking toolkit

Run benchmarks on any model (any quant method), capture full reproducibility
provenance, append structured records to a **durable centralized git repo**, and
produce comparison tables, plots, and a weighted **% -of-bf16 quality index**.

Not GLQ-specific: quant method is read from the checkpoint config
(`glq` / `modelopt`-NVFP4 / `gptq` / `awq` / `none`=bf16), so GLQ, NVFP4, GPTQ,
AWQ and bf16 are all comparable against the shared base.

```bash
pip install "glq[bench]"          # vllm + datasets + matplotlib + pandas + ...
export GLQ_BENCH_RESULTS_REPO=git@github.com:cnygaard/glq-benchmarks.git
```

## Run benchmarks (captures provenance + pushes records)
```bash
# quality + throughput on a GLQ model, push the records to the results repo
glq-bench run \
  --model xv0y5ncu/Gemma-4-31B-it-GLQ-5.0bpw-mix3-8 \
  --tasks mmlu_pro,aime_2026,throughput --quant glq --push

# the bf16 baseline of the same base (needed for the % -of-bf16 index)
glq-bench run --model google/gemma-4-31B-it --tasks mmlu_pro,aime_2026 --quant none --push
```
Each run records: exact serving command, torch/vllm/transformers/sglang versions,
GPU model, **GPU memory at load**, **tokens/sec**, base/parent model, quant method,
HF links — one `BenchRecord` per (model × task × GPU).

Tasks: `mmlu_pro`, `aime_2024`, `aime_2025`, `aime_2026` (thinking), `wikitext2_ppl`,
`throughput` (`vllm bench`). Quality tasks share one vLLM engine; throughput uses
`vllm bench throughput`; perplexity loads via HF.

## Compare / report / index / plot (CPU, reads the results repo)
```bash
glq-bench index                 # weighted % -of-bf16 quality table (+ VRAM, tok/s)
glq-bench report --filter base=google/gemma-4-31B-it
glq-bench compare --models A,B,C --tasks mmlu_pro,aime_2026
glq-bench plot --kind all --out plots/     # index / quality-vs-bpw / Pareto / tok-s-by-gpu
glq-bench pull                  # sync the results repo
```

`LEADERBOARD.md` lives at the root of **this** repo, not in the results repo. Regenerate it
from the glq repo root, so it lands next to the code that produces it:
```bash
glq-bench leaderboard --out LEADERBOARD.md
```
Do not generate it inside a results-repo checkout: `glq-bench run --push` commits with
`git add -A`, so a stray copy there would be pushed back into the results repo.
Pass `--records file.jsonl` to any read command to work offline against a local file.

## Index
`index(M) = Σ wₜ·(scoreₜ(M)/scoreₜ(bf16_of_same_base)) / Σ wₜ` over the standardized
quality tasks (perplexity inverts). bf16 = 100%; quality-only (tok/s + VRAM reported
beside it, not folded in). Override task weights with `--weights weights.yaml`.

## Speed and quality by GPU (LEADERBOARD.md and model cards)
`glq-bench leaderboard` leads with one row per **(model, GPU)**: VRAM, bpw, AIME 2026
avg@8, Terminal-Bench, and decode tok/s and TTFT at batch 1 / 8 / 32. Keyed on the GPU too,
because speed and VRAM do not transfer between cards. Each column has exactly one source:

| column | from |
|---|---|
| VRAM GiB | the weights vLLM reports at load (`Model loading took X GiB`), from any in-process task (`aime_2026`, `mmlu_pro`, ...) — the KV pool is not included, and `decode_sweep` runs its own server so it does not record this |
| AIME 2026 avg@8 | `aime_2026` run with `--avg-k 8` — single-pass records are never shown |
| Terminal-Bench | a full `terminal_bench` run at the version the adapter pins — other versions and `n_tasks`-limited smoke runs are never shown |
| Batch=N tok/s, TTFT | `decode_sweep` with `--task-config '{"concurrencies": [1, 8, 32]}'` |

The same table, limited to one model's base family, goes into that model's HF card:
```bash
glq-bench card --model <repo-id>          # prints the diff against the live card; uploads nothing
glq-bench card --model <repo-id> --push   # uploads README.md (HF_TOKEN from the environment)
```
It replaces the block between `<!-- glq-bench:model-gpu-table:begin/end -->` markers, or
inserts one before `## How GLQ works`. A card without that heading ahead of the appended
base-model card is refused, not guessed at.

## Layout
`provenance` (versions/GPU) · `hfmeta` (base/quant/bpw/links) · `runtime` (vLLM load
+ serving command + load-mem) · `record` (schema/JSONL) · `tasks/` (adapters +
registry) · `runner` (orchestrator) · `store` (git results repo) · `index` /
`report` / `plot` · `cli`.
