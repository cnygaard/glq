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


def _resolve_loader(cfg, causal_cls):
    """Pick the auto-class that can produce next-token logits for ``cfg``, or None.

    Prefers the plain causal-LM class so the ordinary path is untouched, then falls back
    to the image-text-to-text class, which is how transformers exposes vision-language
    models whose text half is a normal causal transformer. ``_model_mapping`` membership
    is the same test ``from_pretrained`` applies internally, so this decides without
    loading 50+ GB of weights first.
    """
    candidates = [causal_cls]
    try:
        from transformers import AutoModelForImageTextToText
        candidates.append(AutoModelForImageTextToText)
    except ImportError:                      # older transformers: causal-only
        pass
    for cls in candidates:
        mapping = getattr(cls, "_model_mapping", None)
        if mapping is None or type(cfg) in mapping:
            return cls
    return None


def run(ctx, config: dict):
    import torch
    import torch.nn.functional as F
    seqlen = int(config.get("seqlen", 2048))
    max_chunks = int(config.get("max_chunks", 80))

    import glq.hf_integration  # noqa: F401 — register the GLQ quant method (no-op for bf16)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(ctx.model, trust_remote_code=True)
    arch = (getattr(cfg, "architectures", None) or [None])[0]
    # Route on CAPABILITY, not on the architecture's name. The previous rule refused any
    # arch containing "ConditionalGeneration", which also rejects vision-language models
    # whose text half is an ordinary dense causal transformer — teacher-forced PPL over a
    # text-only batch is well defined for those, and it is the only instrument that
    # resolves adjacent bpw rungs cheaply. We still refuse, cleanly, when no auto-class
    # can produce next-token logits, so a genuinely unsupported model is recorded as
    # skipped-with-reason rather than dying in an obscure load error.
    loader = _resolve_loader(cfg, AutoModelForCausalLM)
    if loader is None:
        raise TaskUnsupported(
            f"wikitext2_ppl: arch {arch} exposes no causal text tower; PPL skipped "
            "(use mmlu_pro/aime for quality).")

    # float16 by DEFAULT because every stored wikitext2_ppl record was measured that way;
    # moving it would silently break comparability with the existing series. bf16-native
    # models (logit softcapping, large activation outliers) can overflow in fp16 — those
    # opt in with dtype="bfloat16", and the choice is recorded so the number stays scoped.
    dtype_name = str(config.get("dtype", "float16"))
    dtype = getattr(torch, dtype_name)

    before = _load_mem_gib()
    model = loader.from_pretrained(
        ctx.model, dtype=dtype, device_map="cuda", trust_remote_code=True)
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
        config={"dataset": "wikitext-2-raw-v1", "seqlen": seqlen, "n_chunks": n_chunks,
                "dtype": dtype_name, "loader": loader.__name__},
        extra={"n_chunks": n_chunks})
    ctx.standalone_serving = ServingMeta(runtime="hf", dtype=dtype_name,
                                         load_gpu_mem_gib=load_mem)
    return res, ThroughputResult(measure="n/a")
