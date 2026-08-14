"""WikiText-2 perplexity through **vLLM** (``kind="quality"``, shared handle).

Why this exists alongside ``perplexity.py``: the HF adapter cannot measure every quantized
format. ``Firworks/SmolLM3-3B-nvfp4`` ships no ``input_scale`` for any linear, so
transformers newly-initializes 252 of them, decompresses, and then dies in the forward
(``size of tensor a (2048) must match tensor b (128)``). The same checkpoint serves fine
under vLLM, which drops the A4 half and runs weight-only FP4 through Marlin. Without a
vLLM-side PPL there is simply no quality number for that arm.

**These numbers are NOT interchangeable with ``wikitext2_ppl``.** Different attention
kernels and accumulation order give a different absolute value for the same model. Both
are honest measurements of the same quantity by different routes; mixing them in one table
is the error. Hence a distinct task name, and ``standardized=False`` so it cannot silently
enter the quality index next to the HF series. Compare vLLM-PPL only against vLLM-PPL.

The token stream and chunking deliberately mirror ``perplexity.py`` exactly — same dataset,
same non-overlapping ``seqlen`` windows, same mean-CE-per-chunk then ``exp(mean)`` — so the
two series differ only by runtime, not by construction.
"""
from __future__ import annotations

import math

from ..record import BenchmarkResult, ThroughputResult


def run(ctx, config: dict):
    seqlen = int(config.get("seqlen", 2048))
    max_chunks = int(config.get("max_chunks", 80))

    llm = ctx.handle.llm
    tok = ctx.handle.tokenizer
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(ctx.model, trust_remote_code=True)

    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"])).input_ids
    n_chunks = min(len(ids) // seqlen, max_chunks)
    if n_chunks == 0:
        raise RuntimeError(f"wikitext2 tokenized to {len(ids)} tokens, < seqlen {seqlen}")

    chunks = [ids[i * seqlen:(i + 1) * seqlen] for i in range(n_chunks)]

    from vllm import SamplingParams
    # prompt_logprobs=0 returns the logprob of the ACTUAL prompt token at each position,
    # which is exactly the teacher-forced term the HF loop takes cross-entropy over.
    # max_tokens=1 because vLLM requires generating at least one token; it is discarded.
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outs = llm.generate([{"prompt_token_ids": c} for c in chunks], sp,
                        use_tqdm=getattr(ctx, "pbar_factory", None) or True)

    nlls = []
    for out, chunk in zip(outs, chunks):
        plp = out.prompt_logprobs
        if not plp:
            raise RuntimeError(
                "vLLM returned no prompt_logprobs — this backend/config does not support "
                "them, so PPL cannot be computed this way (do NOT fall back to a different "
                "PPL definition; report the arm as skipped instead).")
        # position 0 has no conditional distribution, matching HF's logits[:, :-1] shift.
        lp = []
        for i in range(1, len(chunk)):
            entry = plp[i]
            if entry is None or chunk[i] not in entry:
                raise RuntimeError(f"missing prompt logprob for position {i}")
            lp.append(entry[chunk[i]].logprob)
        nlls.append(-sum(lp) / len(lp))               # mean CE for this chunk

    ppl = float(math.exp(sum(nlls) / len(nlls)))      # exp(mean of per-chunk mean CE)

    res = BenchmarkResult(
        task=config.get("task_name", "wikitext2_ppl_vllm"), metric="perplexity", value=ppl,
        # NOT standardized: a different runtime's PPL must not land in the same index
        # column as the HF series. Flip only once the two are shown to be calibrated.
        standardized=bool(config.get("standardized", False)),
        config={"dataset": "wikitext-2-raw-v1", "seqlen": seqlen, "n_chunks": n_chunks,
                "measure": "vllm_prompt_logprobs/exp(mean per-chunk CE)"},
        extra={"n_chunks": n_chunks, "tokens_scored": n_chunks * (seqlen - 1)})
    return res, ThroughputResult(measure="n/a")
