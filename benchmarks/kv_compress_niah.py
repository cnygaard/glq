"""Phase 4.3: long-context needle-in-haystack across the KV bpw ladder.

For every (bpw_recipe, ctx_len, depth, seed):
  1. Generate a unique 5-digit passkey from the seed.
  2. Build a long distractor passage from Project Gutenberg, insert the
     needle sentence at the target depth fraction, append a question.
  3. Tokenize to *exactly* ``ctx_len`` tokens (truncate distractor as
     needed), wrap in the chat template.
  4. Prefill with a fresh cache, then generate ``--n-decode`` tokens.
  5. Extract digits from the output and compare to the passkey.

Why this matters: PPL on Gutenberg measures averaged next-token
prediction (mostly local context). NIAH isolates targeted long-range
recall — the actual failure mode for KV compression.

Default model: Ministral-3-3B-Instruct-2512 (head_dim=128, no sliding
window, 256k native context) — works with flash-attn-2 out of the box.

Run:
    python benchmarks/kv_compress_niah.py \\
        --ctx-lens 16384 32768 65536 131072 \\
        --depths 0.05 0.25 0.5 0.75 0.95 \\
        --bpws 2 3 4 \\
        --bpw-map /tmp/kv_bpw_min3.json \\
        --seeds 3 \\
        --out /tmp/logs/phase4_niah.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.request

import torch


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

NEEDLE_TEMPLATE = (
    "Important: the special passkey is {key}. Remember it carefully."
)
QUESTION_TEMPLATE = (
    "Question: what is the special passkey mentioned in the text above? "
    "Answer with the 5-digit number only."
)


def read_gutenberg_text(url: str,
                        cache_dir: str = "/opt/dlami/nvme/hf_cache") -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    s = text.find("*** START OF")
    if s != -1:
        text = text[text.find("\n", s) + 1:]
    e = text.find("*** END OF")
    if e != -1:
        text = text[:e]
    # Strip stray 4-6 digit runs that could be confused with the passkey.
    return re.sub(r"\d{4,6}", "", text)


def make_passkey(seed: int) -> str:
    rng = random.Random(seed)
    return f"{rng.randint(10_000, 99_999)}"


def build_prompt(tok, distractor_ids: torch.Tensor, *, ctx_len: int,
                 depth: float, seed: int) -> tuple[torch.Tensor, str]:
    """Tokenize a prompt of exactly ``ctx_len`` tokens with the needle
    placed at ``depth`` (0..1) within the distractor.

    Returns (input_ids, passkey)."""
    passkey = make_passkey(seed)
    needle = "\n" + NEEDLE_TEMPLATE.format(key=passkey) + "\n"
    question = "\n\n" + QUESTION_TEMPLATE
    needle_ids = tok(needle, add_special_tokens=False,
                     return_tensors="pt").input_ids[0]
    question_ids = tok(question, add_special_tokens=False,
                       return_tensors="pt").input_ids[0]
    # Reserve room for chat-template wrapping (~10 tokens) and bos.
    overhead = 24
    budget = ctx_len - len(needle_ids) - len(question_ids) - overhead
    if budget < 16:
        raise ValueError(
            f"ctx_len={ctx_len} too short for needle+question (budget={budget})")
    if budget > distractor_ids.shape[-1]:
        raise ValueError(
            f"distractor pool too small ({distractor_ids.shape[-1]}) "
            f"for budget={budget}")
    distractor = distractor_ids[:budget]
    split = max(1, min(budget - 1, int(round(depth * budget))))
    pre = distractor[:split]
    post = distractor[split:]
    body_ids = torch.cat([pre, needle_ids, post, question_ids])

    body_text = tok.decode(body_ids, skip_special_tokens=True)
    msgs = [{"role": "user", "content": body_text}]
    inp = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt",
        return_dict=True)
    ids = inp["input_ids"]
    if ids.shape[-1] > ctx_len:
        # Trim from the start of the distractor (after BOS) to fit.
        excess = ids.shape[-1] - ctx_len
        head = ids[:, :1]                  # BOS
        tail = ids[:, 1 + excess:]
        ids = torch.cat([head, tail], dim=1)
    return ids, passkey


def extract_passkey(text: str) -> str | None:
    m = re.findall(r"\b(\d{5})\b", text)
    return m[0] if m else None


# --------------------------------------------------------------------------- #
# Cache variants
# --------------------------------------------------------------------------- #

def build_variants(model_config, *, e8_method: str, residual_length: int,
                   bpw_map_path: str | None,
                   bpws: tuple[int, ...]) -> list:
    """Return [(label, bpw_label, factory_or_None)] for the sweep."""
    from glq.kv_cache import GLQQuantizedCache

    def factory_for(method, n_stages=1, bpw_map=None):
        def make():
            return GLQQuantizedCache(
                model_config, quant_method=method, n_stages=n_stages,
                bpw_map=bpw_map, residual_length=residual_length)
        return make

    cfg = (model_config.get_text_config(decoder=True)
           if hasattr(model_config, "get_text_config") else model_config)
    n_layers = getattr(cfg, "num_hidden_layers", 32)

    variants: list = [("baseline (fp16)", "16", None)]
    for b in bpws:
        if b == 8:
            variants.append(("int8 absmax", "8", factory_for("int8")))
            continue
        uniform = {i: b for i in range(n_layers)}
        variants.append(
            (f"{e8_method} {b} bpw", str(b),
             factory_for(e8_method, bpw_map=uniform)))

    if bpw_map_path and os.path.exists(bpw_map_path):
        raw = json.load(open(bpw_map_path))
        bpw_map = {int(k): int(v) for k, v in raw.items()}
        avg = sum(bpw_map.values()) / len(bpw_map)
        variants.append(
            (f"{e8_method} mix avg {avg:.2f}", f"~{avg:.2f}",
             factory_for(e8_method, bpw_map=bpw_map)))
    return variants


# --------------------------------------------------------------------------- #
# One trial
# --------------------------------------------------------------------------- #

def run_one(model, tok, distractor_ids: torch.Tensor, *,
            ctx_len: int, depth: float, seed: int, cache_factory,
            n_decode: int = 16) -> dict:
    ids, passkey = build_prompt(
        tok, distractor_ids, ctx_len=ctx_len, depth=depth, seed=seed)
    ids = ids.to("cuda")
    actual_ctx = ids.shape[-1]

    torch.cuda.empty_cache()
    cache = cache_factory() if cache_factory else None
    gen_kwargs = dict(
        input_ids=ids, max_new_tokens=n_decode, do_sample=False,
        temperature=1.0, pad_token_id=tok.eos_token_id)
    if cache is not None:
        gen_kwargs["past_key_values"] = cache

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    dt = time.time() - t0
    text = tok.decode(out[0, actual_ctx:], skip_special_tokens=True)
    pred = extract_passkey(text)
    return {
        "ctx_len": ctx_len,
        "actual_ctx": actual_ctx,
        "depth": depth,
        "seed": seed,
        "passkey": passkey,
        "pred": pred,
        "ok": pred == passkey,
        "elapsed_sec": dt,
        "output": text,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mistralai/Ministral-3-3B-Instruct-2512")
    p.add_argument("--text-url",
                   default="https://www.gutenberg.org/cache/epub/6593/pg6593.txt")
    p.add_argument("--ctx-lens", type=int, nargs="+",
                   default=[16384, 32768, 65536, 131072])
    p.add_argument("--depths", type=float, nargs="+",
                   default=[0.05, 0.25, 0.5, 0.75, 0.95])
    p.add_argument("--bpws", type=int, nargs="+", default=[2, 3, 4],
                   help="Uniform-bpw recipes to test (in addition to fp16)")
    p.add_argument("--bpw-map", default=None,
                   help="Optional per-layer mixed-precision JSON")
    p.add_argument("--seeds", type=int, default=3,
                   help="Number of distinct passkey seeds per cell")
    p.add_argument("--n-decode", type=int, default=16)
    p.add_argument("--residual", type=int, default=8)
    p.add_argument("--e8-method", default="e8_relaxed",
                   choices=("e8_strict", "e8_relaxed"))
    p.add_argument("--out", default="/tmp/logs/phase4_niah.json")
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf_cache")
    from transformers import AutoTokenizer

    print(f"Loading {args.model} with flash_attention_2 ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    # Multimodal model — load the text-generation entry point.
    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="flash_attention_2")
    except Exception:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="flash_attention_2")
    print(f"  load: {time.time()-t0:.1f}s  cls={type(model).__name__}",
          flush=True)

    text = read_gutenberg_text(args.text_url)
    max_ctx = max(args.ctx_lens)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    distractor_ids = enc[0]
    if distractor_ids.shape[-1] < max_ctx:
        # Repeat the text enough times to cover the longest context.
        n_reps = max_ctx // distractor_ids.shape[-1] + 2
        distractor_ids = distractor_ids.repeat(n_reps)
    print(f"  distractor pool: {distractor_ids.shape[-1]:,} tokens",
          flush=True)

    variants = build_variants(
        model.config, e8_method=args.e8_method,
        residual_length=args.residual, bpw_map_path=args.bpw_map,
        bpws=tuple(args.bpws))
    print(f"  variants: {[v[0] for v in variants]}", flush=True)

    results = {"args": vars(args), "rows": []}
    total = len(variants) * len(args.ctx_lens) * len(args.depths) * args.seeds
    done = 0
    t_start = time.time()
    print(f"\n  total trials: {total}", flush=True)
    print(f"\n  {'variant':<24} {'ctx':>6} {'depth':>6} {'seed':>4} "
          f"{'pred':>6} {'gold':>6} {'ok':>3} {'sec':>6}", flush=True)
    for label, bpw_label, factory in variants:
        for ctx_len in args.ctx_lens:
            for depth in args.depths:
                for seed in range(args.seeds):
                    try:
                        res = run_one(
                            model, tok, distractor_ids,
                            ctx_len=ctx_len, depth=depth, seed=seed,
                            cache_factory=factory, n_decode=args.n_decode)
                    except torch.cuda.OutOfMemoryError:
                        res = {"ctx_len": ctx_len, "depth": depth,
                               "seed": seed, "oom": True, "ok": False}
                        torch.cuda.empty_cache()
                    row = {"variant": label, "bpw_label": bpw_label, **res}
                    results["rows"].append(row)
                    done += 1
                    pred = res.get("pred", "OOM" if res.get("oom") else "?")
                    print(f"  {label:<24} {ctx_len:>6} {depth:>5.2f} "
                          f"{seed:>4} {str(pred):>6} "
                          f"{res.get('passkey', '?'):>6} "
                          f"{'Y' if res.get('ok') else 'N':>3} "
                          f"{res.get('elapsed_sec', 0):>6.1f}  "
                          f"[{done}/{total}]",
                          flush=True)
                    os.makedirs(os.path.dirname(args.out), exist_ok=True)
                    with open(args.out, "w") as f:
                        json.dump(results, f, indent=2)

    dt = time.time() - t_start
    print(f"\n  total wall: {dt/60:.1f} min", flush=True)
    print(f"  wrote {args.out}", flush=True)

    # Quick accuracy summary by variant + ctx_len.
    print(f"\n=== accuracy by (variant, ctx) ===", flush=True)
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    for r in results["rows"]:
        agg[(r["variant"], r["ctx_len"])][0] += int(bool(r.get("ok")))
        agg[(r["variant"], r["ctx_len"])][1] += 1
    print(f"  {'variant':<24} " + " ".join(
        f"{c:>9}" for c in args.ctx_lens), flush=True)
    seen_variants = []
    for r in results["rows"]:
        if r["variant"] not in seen_variants:
            seen_variants.append(r["variant"])
    for v in seen_variants:
        print(f"  {v:<24} " + " ".join(
            f"{agg[(v, c)][0]:>3}/{agg[(v, c)][1]:<5}" for c in args.ctx_lens),
            flush=True)


if __name__ == "__main__":
    main()
