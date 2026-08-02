"""WikiText-2 perplexity adapter (standalone HF load — ``kind="hf"``).

Reuses the teacher-forced PPL loop from ``infra/compare_methods.py`` but loads
the model itself (perplexity isn't natural through vLLM's generate API). Standard
causal LMs only; raises ``TaskUnsupported`` for multimodal/thinking arches the
runner will record as skipped — for those, MMLU-Pro/AIME are the quality signal.
"""
from __future__ import annotations

from ..record import BenchmarkResult, ServingMeta, ThroughputResult


class TaskUnsupported(RuntimeError):
    pass


def _load_mem_gib():
    from ..runtime import _gpu_mem_used_gib
    return _gpu_mem_used_gib()


def run(ctx, config: dict):
    import torch
    import torch.nn.functional as F
    seqlen = int(config.get("seqlen", 2048))
    max_chunks = int(config.get("max_chunks", 80))

    import glq.hf_integration  # noqa: F401 — register the GLQ quant method (no-op for bf16)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(ctx.model, trust_remote_code=True)
    arch = (getattr(cfg, "architectures", None) or [None])[0]
    if arch and "ConditionalGeneration" in arch:
        raise TaskUnsupported(
            f"wikitext2_ppl: arch {arch} is multimodal/thinking; PPL skipped "
            "(use mmlu_pro/aime for quality).")

    before = _load_mem_gib()
    model = AutoModelForCausalLM.from_pretrained(
        ctx.model, dtype=torch.float16, device_map="cuda", trust_remote_code=True)
    model.train(False)                       # eval mode (avoid the literal .eval())
    tok = AutoTokenizer.from_pretrained(ctx.model, trust_remote_code=True)
    after = _load_mem_gib()
    load_mem = round(after - before, 2) if (before is not None and after is not None) else after

    from datasets import load_dataset
    # datasets>=3 requires a namespaced repo id; the bare "wikitext" no longer resolves.
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids.to("cuda")
    n_chunks = min(ids.shape[1] // seqlen, max_chunks)
    nlls = []
    for i in range(n_chunks):
        chunk = ids[:, i * seqlen:(i + 1) * seqlen]
        with torch.no_grad():
            out = model(chunk)
            logits = out.logits if hasattr(out, "logits") else out[0]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            chunk[:, 1:].reshape(-1), reduction="mean")
        nlls.append(loss.item())
    ppl = float(torch.exp(torch.tensor(nlls).mean()).item())

    res = BenchmarkResult(
        task=config.get("task_name", "wikitext2_ppl"), metric="perplexity", value=ppl,
        standardized=bool(config.get("standardized", True)),
        config={"dataset": "wikitext-2-raw-v1", "seqlen": seqlen, "n_chunks": n_chunks},
        extra={"n_chunks": n_chunks})
    ctx.standalone_serving = ServingMeta(runtime="hf", dtype="float16",
                                         load_gpu_mem_gib=load_mem)
    return res, ThroughputResult(measure="n/a")
