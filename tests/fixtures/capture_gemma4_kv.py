"""Capture real K/V tensors from a Gemma-4-E4B-it forward pass.

The forked-kernel Phase 2 acceptance gate compares the kernel against
``reference_e8_attention`` on synthetic K/V (Phase 2.0, 2.1) AND on
real K/V from a Gemma-4-E4B-it prefill (Phase 2.2). This script
captures the latter: it runs a short prompt through the bf16 model,
hooks ``k_proj`` / ``v_proj`` via ``glq.kv_sensitivity._KVCapture``,
and saves a small fixture file covering:

- ``layer 0``: sliding-window layer (head_dim=256, sliding_window=512)
- ``layer 5``: full-attention layer (head_dim=512, no sliding)

The fixture is keyed by ``(model_id, prompt_hash)`` and lands under
``~/.cache/glq/gemma4_e4b_kv_<hash>.pt`` so a re-run picks up the
cached version instead of re-downloading the 4 GB model.

Usage (one-time, env-gated):

    HF_TOKEN=hf_... python tests/fixtures/capture_gemma4_kv.py

The companion ``tests/test_triton_attention_e8_gemma4.py`` consumes
the fixture (and skips if it's missing).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


DEFAULT_PROMPT = (
    "Project Gutenberg's etext of The Adventures of Tom Sawyer by Mark "
    "Twain. This etext is for the use of anyone anywhere at no cost and "
    "with almost no restrictions whatsoever. The whole village turned "
    "out and went down to the river to see the body, and presently the "
    "missing children, who had been hidden in the cave, came running "
    "across the meadow toward home."
)


def _cache_path(model_id: str, prompt: str) -> Path:
    h = hashlib.sha256(
        (model_id + "\n" + prompt).encode()
    ).hexdigest()[:16]
    cache = Path.home() / ".cache" / "glq"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"gemma4_e4b_kv_{h}.pt"


def capture(
    model_id: str = "google/gemma-4-E4B-it",
    prompt: str = DEFAULT_PROMPT,
    *,
    sliding_layer_idx: int = 0,
    full_layer_idx: int = 5,
    device: str = "cuda",
    force: bool = False,
) -> Path:
    """Run a prefill, capture K/V from two layers, save to disk.

    Returns the path of the saved fixture.
    """
    out_path = _cache_path(model_id, prompt)
    if out_path.exists() and not force:
        print(f"[capture_gemma4_kv] cached fixture at {out_path}; "
              "pass force=True to regenerate")
        return out_path

    print(f"[capture_gemma4_kv] loading {model_id} on {device}...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device,
    )

    # Drill into the text-only sub-model.
    # Gemma-4 layout: ``Gemma4ForConditionalGeneration.model.language_model``
    # is the ``Gemma4TextModel``.
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        text_model = model.model.language_model
    elif hasattr(model, "language_model"):
        text_model = model.language_model
    else:
        text_model = model

    # Hook ``k_norm`` / ``v_norm`` (NOT ``k_proj`` / ``v_proj``): the cache
    # stores K/V *after* RMSNorm, and the pre-norm magnitudes are
    # 100× larger (e.g. ~430 on sliding layers). Quantising pre-norm
    # values triggers fp16 overflow in the kernel intermediates,
    # producing max-abs ≈ 8 against the reference. Post-norm K/V
    # has unit-RMS magnitudes — what ``E8KVQuantizer`` expects.
    #
    # ``_KVCapture`` (which hooks k_proj/v_proj) is therefore not used
    # here; we walk ``text_model.layers`` and hook each layer's
    # ``self_attn.k_norm`` / ``self_attn.v_norm`` outputs directly,
    # keyed by the absolute layer index so the saved fixture indexes
    # match the config's ``layer_types`` list.
    k_per_layer: dict[int, torch.Tensor] = {}
    v_per_layer: dict[int, torch.Tensor] = {}
    handles = []

    def _make_hook(target: dict, idx: int):
        def _hook(_module, _inputs, output):
            target[idx] = output.detach().to("cpu", torch.float32)
        return _hook

    for i, layer in enumerate(text_model.layers):
        attn = layer.self_attn
        if not (hasattr(attn, "k_norm") and hasattr(attn, "v_norm")):
            # Shared-KV layers don't own their k_norm/v_norm — skip
            continue
        handles.append(attn.k_norm.register_forward_hook(
            _make_hook(k_per_layer, i)))
        handles.append(attn.v_norm.register_forward_hook(
            _make_hook(v_per_layer, i)))
    print(f"[capture_gemma4_kv] hooked k_norm/v_norm on "
          f"{len(handles) // 2} layers (of {len(text_model.layers)} total)")

    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    print(f"[capture_gemma4_kv] prefill {input_ids.shape[1]} tokens...")
    with torch.no_grad():
        text_model(input_ids)
    for h in handles:
        h.remove()

    # Discover Gemma-4 attention surface from the config.
    cfg = model.config.text_config
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    sliding_head_dim = cfg.head_dim
    full_head_dim = cfg.global_head_dim or cfg.head_dim
    sliding_window = cfg.sliding_window
    layer_types = cfg.layer_types

    def _layer(idx: int) -> dict:
        # k_norm / v_norm output shape: [batch=1, seq, num_kv_heads, head_dim]
        # (already per-head, post-RMSNorm; the layer applies the view before
        # the norm at modeling_gemma4.py:910-911).
        k_buf = k_per_layer[idx]
        v_buf = v_per_layer[idx]
        is_sliding = layer_types[idx] == "sliding_attention"
        head_dim = sliding_head_dim if is_sliding else full_head_dim
        assert k_buf.shape[-2:] == (num_kv_heads, head_dim), (
            f"layer {idx}: k_norm output {tuple(k_buf.shape)} mismatches "
            f"num_kv_heads={num_kv_heads}, head_dim={head_dim}"
        )
        k = k_buf.squeeze(0).to(torch.float16)
        v = v_buf.squeeze(0).to(torch.float16)
        return dict(
            layer_idx=idx,
            layer_type=layer_types[idx],
            is_sliding=is_sliding,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            num_q_heads=num_q_heads,
            sliding_window=(sliding_window if is_sliding else 0),
            k=k,
            v=v,
        )

    assert layer_types[sliding_layer_idx] == "sliding_attention", (
        f"sliding_layer_idx={sliding_layer_idx} but "
        f"layer_types[{sliding_layer_idx}] = "
        f"{layer_types[sliding_layer_idx]!r}; want sliding_attention"
    )
    assert layer_types[full_layer_idx] == "full_attention", (
        f"full_layer_idx={full_layer_idx} but "
        f"layer_types[{full_layer_idx}] = "
        f"{layer_types[full_layer_idx]!r}; want full_attention"
    )

    fixture = dict(
        model_id=model_id,
        prompt=prompt,
        input_ids=input_ids.cpu(),
        sliding_layer=_layer(sliding_layer_idx),
        full_layer=_layer(full_layer_idx),
    )

    torch.save(fixture, out_path)
    print(f"[capture_gemma4_kv] saved fixture "
          f"({out_path.stat().st_size / 1e6:.2f} MB) to {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    out = capture(model_id=args.model, prompt=args.prompt, force=args.force)
    print(out)


if __name__ == "__main__":
    main()
