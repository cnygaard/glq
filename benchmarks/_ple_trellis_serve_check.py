"""Step 3: does a trellis-coded PLE actually load and serve — through the trellis path?

The trap this is built around: a checkpoint whose PLE silently fell back to the shell decoder
would still load, still generate fluent text, and still look like a pass. Coherent output is
therefore not evidence. Each check below asserts the mechanism — which module class was
substituted, which op was called — and only then looks at the text.

    python benchmarks/_ple_trellis_serve_check.py <ckpt> [--shell-ckpt <ckpt>] [--skip-vllm]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROMPT = "Explain in two sentences why the sky appears blue."


def check_config(path: str) -> dict:
    cfg = json.load(open(os.path.join(path, "config.json")))
    qc = cfg.get("quantization_config", {})
    print("  quantization_config.codebook     :", qc.get("codebook"))
    print("  quantization_config.ple_codebook :", qc.get("ple_codebook"))
    print("  quantization_config.ple_bpw      :", qc.get("ple_bpw"))
    if not os.environ.get("GLQ_CHECK_CONTROL"):
        assert qc.get("ple_codebook") == "trellis", (
            "the marker vLLM dispatches on is absent — create_weights would build shell "
            "buffers")
    return qc


def check_hf(path: str) -> None:
    # REQUIRED, and its absence is quiet: this import is what registers quant_method="glq"
    # with transformers (@register_quantization_config / @register_quantizer). Without it
    # from_pretrained loads the architecture, finds none of the GLQ tensors, reports every
    # weight as newly initialized, and hands back a randomly-weighted model that still
    # generates text. Importing glq.quantized_linear alone is not enough.
    import glq.hf_integration  # noqa: F401
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from glq.quantized_linear import E8RHTEmbedding, TrellisRHTEmbedding

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(
        path, dtype=torch.bfloat16, device_map="cuda")

    control = bool(os.environ.get("GLQ_CHECK_CONTROL"))
    tre = [n for n, m in model.named_modules() if isinstance(m, TrellisRHTEmbedding)]
    shell = [n for n, m in model.named_modules() if isinstance(m, E8RHTEmbedding)]
    print(f"  TrellisRHTEmbedding modules: {len(tre)} {tre[:2]}")
    print(f"  E8RHTEmbedding modules     : {len(shell)} {shell[:2]}")
    if control:
        # Control arm: the shell build of the same model, same prompt and sampling. If this
        # also produces nothing, the empty output is the harness, not the trellis path.
        assert shell, "control checkpoint has no GLQ embedding either"
    else:
        assert tre, "no TrellisRHTEmbedding was substituted — the PLE fell back to shell"

    mod = dict(model.named_modules())[(tre or shell)[0]]
    if tre:
        print(f"  recovered rate K={mod.K}, packed {tuple(mod.trellis_packed.shape)}, "
              f"blocks {[int(b) for b in mod.rht_blocks.tolist()]}")
        assert sum(int(b) for b in mod.rht_blocks.tolist()) == mod.embedding_dim, (
            "block sizes do not sum to the width — the row would inverse-transform wrongly")

    # Chat template + the model card's sampling, NOT greedy. gemma-4 is RL-collapsed and
    # produces nothing at temperature 0 — reproduced in this project before (700/700 tokens,
    # no content). Greedy here would read as a broken checkpoint when it is a sampling
    # mistake, which is the whole failure mode this script exists to avoid.
    msgs = [{"role": "user", "content": PROMPT}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to("cuda")
    out = model.generate(**ids, max_new_tokens=64,
                         do_sample=True, temperature=1.0, top_p=0.95, top_k=64)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  HF generation: {text.strip()[:220]!r}")
    assert text.strip(), "empty generation"


def check_vllm(path: str) -> None:
    """Serve it, and prove the trellis op is the one that ran."""
    import glq_vllm  # noqa: F401  (registers the plugin + ops)
    from vllm import LLM, SamplingParams

    calls = {"trellis": 0, "shell": 0}
    real_t = torch.ops.glq.embedding_dequant_trellis
    real_s = torch.ops.glq.embedding_dequant

    # Count through the dispatcher rather than trusting the config: this is the difference
    # between "a trellis checkpoint loaded" and "the trellis decode actually ran".
    def wrap(fn, key):
        def inner(*a, **k):
            calls[key] += 1
            return fn(*a, **k)
        return inner

    torch.ops.glq.embedding_dequant_trellis = wrap(real_t, "trellis")   # type: ignore[assignment]
    torch.ops.glq.embedding_dequant = wrap(real_s, "shell")             # type: ignore[assignment]
    try:
        llm = LLM(model=path, dtype="bfloat16", max_model_len=2048,
                  gpu_memory_utilization=0.55, enforce_eager=True)
        tok = llm.get_tokenizer()
        prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                         add_generation_prompt=True, tokenize=False)
        # The model card's sampling, not greedy — same reason as the HF arm: gemma-4 emits
        # nothing at temperature 0, and an empty string would sail past a check that only
        # counts op dispatches.
        outs = llm.generate([prompt], SamplingParams(temperature=1.0, top_p=0.95,
                                                     top_k=64, max_tokens=64))
        text = outs[0].outputs[0].text
        print(f"  vLLM generation: {text.strip()[:220]!r}")
    finally:
        torch.ops.glq.embedding_dequant_trellis = real_t                # type: ignore[assignment]
        torch.ops.glq.embedding_dequant = real_s                        # type: ignore[assignment]

    print(f"  op calls: embedding_dequant_trellis={calls['trellis']}  "
          f"embedding_dequant(shell)={calls['shell']}")
    assert calls["trellis"] > 0, (
        "the trellis embedding op never ran — coherent text here came from somewhere else")
    assert calls["shell"] == 0, "the shell embedding op ran too — both paths were active"
    # Both halves are required. Op counts alone would pass on an empty string; text alone
    # would pass on a silent fallback to shell.
    assert text.strip(), "empty generation despite the trellis op running"


def footprint(path: str) -> float:
    """On-disk bytes of the PLE tensors, which is what the 1.83x claim is about."""
    from safetensors import safe_open
    total = 0
    with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
        for k in f.keys():
            if "embed_tokens_per_layer" in k:
                sl = f.get_slice(k)
                n = 1
                for d in sl.get_shape():
                    n *= d
                total += n * (2 if "16" in str(sl.get_dtype()) else 4)
    return total / 2 ** 30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--shell-ckpt", default=None,
                    help="same model with a shell PLE, for the footprint comparison")
    ap.add_argument("--skip-vllm", action="store_true")
    args = ap.parse_args()

    print("== config ==")
    check_config(args.ckpt)
    print("== HF load ==")
    check_hf(args.ckpt)
    if args.shell_ckpt:
        t, s = footprint(args.ckpt), footprint(args.shell_ckpt)
        print(f"== footprint ==\n  trellis PLE {t:.2f} GiB   shell PLE {s:.2f} GiB   "
              f"ratio {s / max(t, 1e-9):.2f}x")
    if not args.skip_vllm:
        print("== vLLM serve ==")
        check_vllm(args.ckpt)
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
