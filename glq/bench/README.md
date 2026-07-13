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
Pass `--records file.jsonl` to any read command to work offline against a local file.

## Index
`index(M) = Σ wₜ·(scoreₜ(M)/scoreₜ(bf16_of_same_base)) / Σ wₜ` over the standardized
quality tasks (perplexity inverts). bf16 = 100%; quality-only (tok/s + VRAM reported
beside it, not folded in). Override task weights with `--weights weights.yaml`.

## Layout
`provenance` (versions/GPU) · `hfmeta` (base/quant/bpw/links) · `runtime` (vLLM load
+ serving command + load-mem) · `record` (schema/JSONL) · `tasks/` (adapters +
registry) · `runner` (orchestrator) · `store` (git results repo) · `index` /
`report` / `plot` · `cli`.
