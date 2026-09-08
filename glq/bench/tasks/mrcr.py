"""OpenAI MRCR — long-context retrieval, bucketed by context length.

The instrument for KV-cache work. Reasoning evals stress the model; MRCR stresses
the *cache*: the answer is verbatim text sitting tens or hundreds of thousands of
tokens back, so any KV quantization error shows up as retrieval failure at exactly
the lengths where a smaller cache is supposed to buy you something. vLLM's
TurboQuant blog used it for that reason, and it is where its 3-bit variants fell
over while looking fine on short-context reasoning.

Multi-round coreference resolution: the conversation contains ``n_needles``
near-identical user requests ("write a poem about X"), each answered differently,
and the model must reproduce the i-th assistant response — prefixed by a random
string that proves it followed the instruction rather than pattern-matching.

Scoring is the reference implementation's, verbatim
(https://huggingface.co/datasets/openai/mrcr) — deterministic, no judge model:

    if not response.startswith(random_string_to_prepend):
        return 0
    return SequenceMatcher(None, response_stripped, answer_stripped).ratio()

Missing the prefix scores zero however good the recall was; that is the point of
the prefix.

Reported per length bucket, plus **AUC** over log2(context) as the aggregate —
a method that holds up to 32k and collapses at 128k should not average out to
"fine", and the area under the curve is what the blog and Context Arena quote.
"""
from __future__ import annotations

import json
import math
import time
from difflib import SequenceMatcher

from glq.bench.record import BenchmarkResult
from glq.bench.tasks.thinking import sampling

#: Bucket edges in tokens, the dataset card's bins. Buckets whose lower edge is
#: beyond the served context window are skipped rather than truncated — a
#: truncated long-context sample measures nothing except the truncation.
_BINS = [(4096, 8192), (8192, 16384), (16384, 32768), (32768, 65536),
         (65536, 131072), (131072, 262144), (262144, 524288), (524288, 1048576)]

#: The answer is a reproduced writing sample plus the 10-char prefix. The
#: reference answers are short; this is generous without inviting rambling.
_DEFAULT_MAX_TOKENS = 1024


def grade(response: str, answer: str, random_string_to_prepend: str) -> float:
    """The reference metric, verbatim from the dataset card."""
    if not response.startswith(random_string_to_prepend):
        return 0.0
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())


def _bucket_of(n_tokens: int):
    for lo, hi in _BINS:
        if lo <= n_tokens <= hi:
            return (lo, hi)
    return None


def auc(scores_by_bucket: dict) -> float:
    """Area under the score-vs-log2(context) curve, normalised to [0, 1].

    Trapezoid over log2 of the bucket upper edges, divided by the log2 span, so
    the result is a weighted mean that treats each *doubling* of context equally
    rather than each bucket equally. One bucket alone has no area and returns its
    own score, which keeps a partial run interpretable.
    """
    pts = sorted((math.log2(hi), s) for (_, hi), s in scores_by_bucket.items())
    if not pts:
        return 0.0
    if len(pts) == 1:
        return float(pts[0][1])
    area = sum((pts[i + 1][0] - pts[i][0]) * (pts[i + 1][1] + pts[i][1]) / 2.0
               for i in range(len(pts) - 1))
    return float(area / (pts[-1][0] - pts[0][0]))


def _rows(needles: int, per_bucket: int, max_ctx: int, tokenizer):
    """Dataset rows grouped into buckets, capped at `per_bucket` each.

    Token counts come from the served model's own tokenizer rather than the
    dataset's `n_chars`, because the bucket boundary is what decides whether a
    sample fits the window, and chars-per-token varies enough to misplace samples
    at the edges.
    """
    from datasets import load_dataset

    # Streamed, and stopped as soon as every reachable bucket is full. The dataset
    # carries samples out to 1M tokens; a non-streaming load pulls all of it to
    # screen a few 8k ones, which is minutes of download and tens of GB for
    # samples this run will never look at.
    ds = load_dataset("openai/mrcr", split="train", streaming=True)
    wanted = {b for b in _BINS if b[1] <= max_ctx}
    buckets: dict = {}
    for row in ds:
        if int(row["n_needles"]) != needles:
            continue
        # n_chars is a cheap pre-filter: tokenising a 1M-token conversation to
        # discover it is out of range costs more than the sample is worth. One
        # token is never fewer than ~2 chars for this text, so this cannot drop a
        # sample that would have fitted.
        if int(row.get("n_chars", 0)) > max_ctx * 8:
            continue
        msgs = json.loads(row["prompt"])
        n_tok = sum(len(tokenizer.encode(m["content"])) for m in msgs)
        if n_tok > max_ctx:
            continue
        b = _bucket_of(n_tok)
        if b is None or b not in wanted:
            continue
        got = buckets.setdefault(b, [])
        if len(got) < per_bucket:
            got.append((msgs, row["answer"], row["random_string_to_prepend"], n_tok))
        if all(len(buckets.get(w, [])) >= per_bucket for w in wanted):
            break
    return buckets


def run(ctx, config: dict):
    needles = int(config.get("needles", 8))
    per_bucket = int(config.get("per_bucket", 8))
    max_tokens = int(config.get("max_tokens", _DEFAULT_MAX_TOKENS))
    # Never ask for more context than the engine was started with: vLLM rejects
    # the request, and a bucket of rejections is not a score of zero, it is no
    # measurement at all.
    max_ctx = int(config.get("max_ctx", getattr(ctx.handle, "max_model_len", 0) or 131072))

    buckets = _rows(needles, per_bucket, max_ctx, ctx.handle.tokenizer)
    if not buckets:
        raise RuntimeError(
            f"no MRCR samples fit: needles={needles}, max_ctx={max_ctx}. The served "
            f"context window is below the smallest bucket ({_BINS[0][0]} tokens).")

    # Greedy by default, unlike the reasoning tasks. MRCR asks for verbatim
    # reproduction of text already in the context and is scored by similarity to
    # an exact string, so the model-card sampling those tasks inherit (1.0/0.95/64)
    # would add variance to the measurement without representing anything the task
    # is trying to capture. Overridable via task-config like everything else.
    sp = sampling({"temperature": 0.0, "top_p": 1.0, "top_k": None, **config},
                  max_tokens)
    per_bucket_scores: dict = {}
    per_item: list = []
    prefix_misses = 0
    truncated = 0
    t0 = time.time()

    for b in sorted(buckets):
        batch = buckets[b]
        outs = ctx.handle.llm.chat([m for m, _, _, _ in batch], sp, use_tqdm=True)
        scores = []
        for out, (_, answer, prefix, n_tok) in zip(outs, batch):
            text = out.outputs[0].text
            s = grade(text, answer, prefix)
            if not text.startswith(prefix):
                prefix_misses += 1
            if getattr(out.outputs[0], "finish_reason", None) == "length":
                truncated += 1
            scores.append(s)
            per_item.append({"bucket": b[1], "tokens": n_tok, "score": round(s, 4)})
        per_bucket_scores[b] = sum(scores) / len(scores)

    dt = time.time() - t0
    aggregate = auc(per_bucket_scores)

    return BenchmarkResult(
        task=config.get("task_name", f"mrcr_{needles}needle"),
        metric="auc", value=aggregate,
        standardized=bool(config.get("standardized", False)),
        config={"needles": needles, "per_bucket": per_bucket, "max_ctx": max_ctx,
                "max_tokens": max_tokens, "temperature": sp.temperature,
                "top_p": sp.top_p, "seed": sp.seed,
                "buckets": [hi for _, hi in sorted(per_bucket_scores)]},
        extra={
            # Per bucket, keyed by the upper edge — the curve behind the AUC, and
            # the thing to read when a method holds short and fails long.
            "by_bucket": {str(hi): round(v, 4)
                          for (_, hi), v in sorted(per_bucket_scores.items())},
            "n": len(per_item),
            # Prefix misses are instruction-following failures, not retrieval
            # failures; both score 0 and only this number separates them.
            "prefix_misses": prefix_misses,
            # Ran to the token cap rather than stopping: rambling or looping, not a
            # retrieval result. Greedy decoding on a thinking model produces exactly
            # this (Qwen's card warns of "endless repetitions"), and the greedy
            # default below is right for a non-thinking model and wrong for that one.
            # A near-zero score with most items truncated is a sampling mistake; the
            # same score with none truncated is the cache losing the needle.
            "truncated": truncated,
            "elapsed_s": round(dt, 1),
            "per_item": per_item,
        },
    ), None
