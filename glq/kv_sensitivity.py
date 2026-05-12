"""Per-layer sensitivity profiling + allocator for mixed-precision KV cache.

Mirrors ``glq/sensitivity.py`` for KV: instead of weight-quantization
Hessian-trace proxy losses, here we compute simple reconstruction-MSE
proxy losses by quantizing real captured KV tensors at each candidate
bpw and comparing to the bf16 original. The greedy marginal-gain
allocator then assigns a bpw to each cache layer subject to an average
bpw budget.

Pipeline:
  1. Forward N calibration sequences through the model with output_attentions
     and a forward hook on each attention layer's K and V projections.
  2. For each (layer, bpw) in candidate menu {2, 4, 8, 16}: round-trip
     the captured (K, V) tensor and accumulate MSE.
  3. Take ``sensitivity[L] = MSE_at_min_bpw[L]`` (the per-layer cost at
     the most aggressive compression). Pass to a greedy allocator that
     upgrades layers in order of marginal-gain-per-bit until the budget
     is exhausted.

Outputs ``{layer_idx: int bpw}`` consumable by ``GLQQuantizedCache(bpw_map=...)``.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .kv_cache import VALID_KV_BPW


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #

class _KVCapture:
    """Forward-hook helper that collects K, V tensors per attention layer.

    The hook attaches to the ``k_proj`` and ``v_proj`` Linear modules of
    every attention block: their OUTPUT is the post-projection key/value
    states that the cache normally stores.
    """

    def __init__(self):
        self.k_per_layer: dict[int, list[torch.Tensor]] = {}
        self.v_per_layer: dict[int, list[torch.Tensor]] = {}
        self._handles = []

    def attach(self, model):
        """Walk the model, hook k_proj/v_proj on every attention layer."""
        layer_idx_counter = 0
        for name, module in model.named_modules():
            cls_name = type(module).__name__
            # Heuristic: attention blocks have submodules named k_proj/v_proj.
            if hasattr(module, "k_proj") and hasattr(module, "v_proj") and \
                    "Attention" in cls_name:
                idx = layer_idx_counter
                layer_idx_counter += 1
                self.k_per_layer[idx] = []
                self.v_per_layer[idx] = []
                self._handles.append(
                    module.k_proj.register_forward_hook(
                        self._make_hook(self.k_per_layer, idx)))
                self._handles.append(
                    module.v_proj.register_forward_hook(
                        self._make_hook(self.v_per_layer, idx)))
        if layer_idx_counter == 0:
            raise RuntimeError(
                "No attention layers with k_proj/v_proj submodules found. "
                "Pass a model with the standard HF attention layout.")
        return layer_idx_counter

    @staticmethod
    def _make_hook(target_list_dict, idx):
        def hook(_module, _inputs, output):
            # output: [batch, seq, kv_dim] — pre-reshape into heads.
            # We store in CPU fp32 to keep the GPU free for the model.
            target_list_dict[idx].append(output.detach().to("cpu", torch.float32))
        return hook

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# --------------------------------------------------------------------------- #
# Per-layer reconstruction MSE under each candidate bpw
# --------------------------------------------------------------------------- #

def _mse_for_bpw(tensor: torch.Tensor, bpw: int, *, e8_method: str,
                 cb_strict, cb_relaxed) -> float:
    """Reconstruction MSE if we quantize ``tensor`` at the given bpw.

    Mirrors what ``GLQQuantizedCache`` will actually do per layer:
      bpw 16 → identity, MSE = 0
      bpw 8  → per-row INT8 absmax
      bpw 4  → E8 RVQ (2 stages)
      bpw 2  → E8 stage 1
    """
    if bpw == 16:
        return 0.0
    if bpw == 8:
        # Per-row absmax INT8 (same math as INT8QuantizedLayer with axis=0).
        scale = tensor.abs().amax(dim=0, keepdim=True).clamp_min(1e-12) / 127
        q = (tensor / scale).round().clamp(-128, 127)
        rec = q * scale
        return ((tensor - rec) ** 2).mean().item()
    # E8 paths.
    from .kv_e8 import E8KVQuantizer
    cb = cb_relaxed if e8_method == "e8_relaxed" else cb_strict
    n_stages = 2 if bpw == 4 else 1
    quant = E8KVQuantizer(cb, n_stages=n_stages)
    qt = quant.quantize(tensor)
    rec = quant.dequantize(qt)
    return ((tensor - rec) ** 2).mean().item()


def compute_kv_proxy_losses(
    capture: _KVCapture,
    *,
    candidate_bpws: tuple[int, ...] = VALID_KV_BPW,
    e8_method: str = "e8_relaxed",
    cb_strict=None,
    cb_relaxed=None,
    device: str = "cuda",
    verbose: bool = True,
) -> tuple[dict[int, dict[int, float]], dict[int, int]]:
    """Compute reconstruction MSE per (layer, bpw) and per-layer sizes.

    The codebook NN match is a 65,536-way distance computation per group
    of 8 values — only tractable on GPU. Captures live on CPU; this
    function pages them to ``device`` per layer, computes MSEs, frees.
    """
    if cb_strict is None:
        from .codebook import E8ShellCodebook
        cb_strict = E8ShellCodebook(device=device, verbose=False)
    if cb_relaxed is None:
        from .codebook_relaxed import E8RelaxedCodebook
        cb_relaxed = E8RelaxedCodebook(device=device, verbose=False)

    proxy: dict[int, dict[int, float]] = {}
    sizes: dict[int, int] = {}
    n_layers = len(capture.k_per_layer)
    if verbose:
        n_active = sum(1 for idx in range(n_layers)
                       if capture.k_per_layer[idx])
        print(f"  scoring {n_active} active layers × {len(candidate_bpws)} "
              f"bpws on {device}", flush=True)
    for idx in range(n_layers):
        # KV-sharing layers (e.g. Gemma-4 readers) have no k_proj/v_proj
        # calls of their own; skip them so the cache layer index keeps
        # its position but the layer is absent from the allocation.
        if not capture.k_per_layer[idx] or not capture.v_per_layer[idx]:
            continue
        k_cat = torch.cat(capture.k_per_layer[idx], dim=0).to(device)
        v_cat = torch.cat(capture.v_per_layer[idx], dim=0).to(device)
        sizes[idx] = k_cat.numel() + v_cat.numel()
        proxy[idx] = {}
        for bpw in candidate_bpws:
            mse_k = _mse_for_bpw(
                k_cat, bpw, e8_method=e8_method,
                cb_strict=cb_strict, cb_relaxed=cb_relaxed)
            mse_v = _mse_for_bpw(
                v_cat, bpw, e8_method=e8_method,
                cb_strict=cb_strict, cb_relaxed=cb_relaxed)
            proxy[idx][bpw] = 0.5 * (mse_k + mse_v)
        # Free GPU memory for the next layer.
        del k_cat, v_cat
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        if verbose:
            top = min(proxy[idx])
            print(f"    layer {idx:3d}  MSE@{top}={proxy[idx][top]:.4e}",
                  flush=True)
    return proxy, sizes


# --------------------------------------------------------------------------- #
# Allocator
# --------------------------------------------------------------------------- #

def allocate_kv_bpw(
    proxy_losses: dict[int, dict[int, float]],
    layer_sizes: dict[int, int],
    target_avg_bpw: float,
    *,
    allowed_bpws: tuple[int, ...] = VALID_KV_BPW,
) -> dict[int, int]:
    """Greedy marginal-gain allocation of KV bpw to minimize total loss.

    Identical algorithm to ``glq.sensitivity.allocate_bpw`` but works
    directly on per-(layer, bpw) measured proxy losses rather than the
    closed-form 4× decay. Start everything at the minimum allowed bpw;
    repeatedly upgrade the (layer, target_bpw) with best
    loss-reduction-per-bit until the budget is exhausted.
    """
    allowed = sorted(set(allowed_bpws))
    if not allowed:
        raise ValueError("allowed_bpws is empty")
    base = allowed[0]
    total_size = sum(layer_sizes.values())
    budget = target_avg_bpw * total_size

    allocation = {idx: base for idx in proxy_losses}
    current_bits = base * total_size

    while True:
        best_gain = -1.0
        best_idx = None
        best_target = None
        best_cost = 0

        for idx, cur in allocation.items():
            for target_bpw in allowed:
                if target_bpw <= cur:
                    continue
                extra = (target_bpw - cur) * layer_sizes[idx]
                if current_bits + extra > budget:
                    continue
                loss_cur = proxy_losses[idx][cur]
                loss_tgt = proxy_losses[idx][target_bpw]
                gain = (loss_cur - loss_tgt) / extra
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_target = target_bpw
                    best_cost = extra

        if best_idx is None:
            break
        allocation[best_idx] = best_target
        current_bits += best_cost

    return allocation


def print_kv_allocation_summary(
    allocation: dict[int, int],
    layer_sizes: dict[int, int],
    proxy_losses: dict[int, dict[int, float]] | None = None,
) -> None:
    """Pretty-print the allocator output (same conventions as the
    weight-quant allocator)."""
    total_size = sum(layer_sizes.values())
    total_bits = sum(allocation[i] * layer_sizes[i] for i in allocation)
    avg = total_bits / total_size

    by_bpw: dict[int, list[int]] = {}
    for idx, bpw in allocation.items():
        by_bpw.setdefault(bpw, []).append(idx)

    print("\nKV bit allocation summary:")
    print(f"  Average bpw: {avg:.3f}")
    print(f"  Total elements: {total_size:,}")
    for bpw in sorted(by_bpw):
        layers = by_bpw[bpw]
        size = sum(layer_sizes[i] for i in layers)
        pct = size / total_size
        print(f"  {bpw} bpw: {len(layers):3d} layers "
              f"({size:>12,} elems, {pct:.1%})")

    if proxy_losses:
        ranked = sorted(
            allocation.items(),
            key=lambda kv: proxy_losses[kv[0]][min(proxy_losses[kv[0]])],
            reverse=True,
        )
        print("\n  Top 10 most sensitive layers (by MSE at min bpw):")
        for idx, bpw in ranked[:10]:
            min_bpw = min(proxy_losses[idx])
            print(f"    layer.{idx:<3d}  bpw={bpw}  "
                  f"MSE@{min_bpw}bpw={proxy_losses[idx][min_bpw]:.4e}")


# --------------------------------------------------------------------------- #
# Orchestrator: load model + forward calibration + allocate
# --------------------------------------------------------------------------- #

def profile_and_allocate(
    model,
    tokenizer,
    *,
    target_avg_bpw: float,
    nsamples: int = 32,
    seqlen: int = 512,
    candidate_bpws: tuple[int, ...] = VALID_KV_BPW,
    e8_method: str = "e8_relaxed",
    calibration_corpus: str = "wikitext",
    device: str = "cuda",
    verbose: bool = True,
) -> dict[int, int]:
    """End-to-end: capture per-layer K/V, score at candidate bpws, allocate.

    Returns ``{layer_idx: bpw}`` ready to pass to ``GLQQuantizedCache(bpw_map=...)``.
    """
    if verbose:
        print(f"KV sensitivity profile — target avg {target_avg_bpw} bpw, "
              f"nsamples={nsamples}, seqlen={seqlen}, e8={e8_method}",
              flush=True)

    # ---- Calibration data ----
    if calibration_corpus == "wikitext":
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(ds["text"])
    elif isinstance(calibration_corpus, str) and calibration_corpus.startswith("c4"):
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split="train",
                          streaming=True)
        text = "\n\n".join(next(iter(ds))["text"] for _ in range(nsamples * 4))
    else:
        raise ValueError(
            f"unknown calibration_corpus {calibration_corpus!r}")
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]
    n = min(nsamples, input_ids.shape[0] // seqlen)
    calib_ids = input_ids[:n * seqlen].reshape(n, seqlen).to(device)
    if verbose:
        print(f"  {n} calibration sequences x {seqlen} tokens", flush=True)

    # ---- Forward with hooks ----
    capture = _KVCapture()
    n_layers = capture.attach(model)
    if verbose:
        print(f"  hooked {n_layers} attention layers", flush=True)
    try:
        with torch.no_grad():
            for i in range(n):
                model(input_ids=calib_ids[i:i + 1])
    finally:
        capture.detach()

    # ---- Proxy losses + allocation ----
    if verbose:
        print("  computing proxy losses at candidate bpws ...", flush=True)
    proxy, sizes = compute_kv_proxy_losses(
        capture, candidate_bpws=candidate_bpws, e8_method=e8_method,
        device=device, verbose=verbose)
    allocation = allocate_kv_bpw(proxy, sizes, target_avg_bpw,
                                 allowed_bpws=candidate_bpws)
    if verbose:
        print_kv_allocation_summary(allocation, sizes, proxy)
    return allocation
