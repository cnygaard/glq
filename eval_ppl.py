"""
eval_ppl.py — Evaluate LLM perplexity under quantization.

GPTQ-style layer-by-layer quantization:
  1. Capture per-layer input activations from calibration data
  2. Compute Hessian H = X^T X / n_samples
  3. Quantize W via chosen method
  4. Replace weight in-place
  5. Evaluate perplexity on test data

Methods:
  bw16_shell         — BW₁₆ shell codebook LDLQ (1-2 bpw)
  bw16_babai         — BW₁₆ Babai LDLQ with Wscale grid search (2-4 bpw)
  bw16_babai_hessian — BW₁₆ Babai LDLQ with Hessian-aware decoder (2-4 bpw)
  bw16_babai_rht     — BW₁₆ Babai LDLQ with RHT incoherence (2-4 bpw)
  e8_shell           — E₈ shell codebook LDLQ (2 or 4 bpw)
  e8_shell_rht       — E₈ shell codebook LDLQ + RHT (2 or 4 bpw) [QuIP#-style]
  e8_babai_rht       — E₈ Babai LDLQ + RHT (2-4 bpw)
  scalar_gptq        — Standard scalar GPTQ baseline (any bpw)

Diagnostic methods:
  diag_scalar_noprop   — Exp A: scalar block-16, no within-block propagation
  diag_bw16_unbounded  — Exp B: BW16 LDLQ unbounded (no clamping)
  diag_gptq_bw16       — Exp C: GPTQ Hinv + BW16 Euclidean blocks
  diag_bw16_hessian    — Exp D: GPTQ Hinv + BW16 Hessian-aware blocks
  diag_gptq_bw16_rht   — Exp E: GPTQ Hinv + BW16 + RHT incoherence

Usage:
    python eval_ppl.py --model MODEL --device cuda --method e8_shell_rht --bpw 2
    python eval_ppl.py --model MODEL --device cuda --method e8_shell_rht --bpw 4
    python eval_ppl.py --model MODEL --device cuda --method bw16_babai --bpw 3
    python eval_ppl.py --model MODEL --device cuda --method scalar_gptq --bpw 3
    python eval_ppl.py --model MODEL --device cuda --method bw16_shell --bpw 1
    python eval_ppl.py --model MODEL --device cuda --baseline-only
"""

import sys
import time
import argparse
import math
import gc
import json
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime, timedelta

# ---- import quantization functions from prototype ----
sys.path.insert(0, str(Path(__file__).parent))

# Load prototype module without triggering __main__ block
_proto_path = str(Path(__file__).parent / "golay-leech-quant-prototype-v2.py")
_source = Path(_proto_path).read_text()
_source = _source.replace('if __name__ == "__main__":', 'if __name__ == "__never_run__":')

import types
lattice_quant = types.ModuleType("lattice_quant")
lattice_quant.__file__ = _proto_path
_code = compile(_source, _proto_path, 'exec')
exec(_code, lattice_quant.__dict__)

BW16ShellCodebook = lattice_quant.BW16ShellCodebook
E8ShellCodebook = lattice_quant.E8ShellCodebook
quantize_ldlq_codebook = lattice_quant.quantize_ldlq_codebook
quantize_ldlq_codebook_rvq = lattice_quant.quantize_ldlq_codebook_rvq
BW16Quantizer = lattice_quant.BW16Quantizer
LatticeQuantizer = lattice_quant.LatticeQuantizer
RHT = lattice_quant.RHT
E8Quantizer = lattice_quant.E8Quantizer

METHOD_NAMES = {
    'bw16_shell': 'BW16 Shell LDLQ',
    'bw16_babai': 'BW16 Babai LDLQ',
    'bw16_babai_hessian': 'BW16 Babai LDLQ Hessian-Aware',
    'scalar_gptq': 'Scalar GPTQ',
    'diag_scalar_noprop': 'Scalar Block-16 No-Prop (Exp A)',
    'diag_bw16_unbounded': 'BW16 LDLQ Unbounded (Exp B)',
    'diag_gptq_bw16': 'GPTQ + BW16 Blocks (Exp C)',
    'diag_bw16_hessian': 'GPTQ + BW16 Hessian-Aware (Exp D)',
    'bw16_babai_rht': 'BW16 Babai LDLQ + RHT',
    'diag_gptq_bw16_rht': 'GPTQ + BW16 + RHT (Exp E)',
    'e8_babai_rht': 'E8 Babai LDLQ + RHT',
    'e8_shell': 'E8 Shell LDLQ',
    'e8_shell_rht': 'E8 Shell LDLQ + RHT',
}


# ---- logging helper ----

_log_path = None

def log(msg=""):
    """Print and optionally log to file."""
    print(msg, flush=True)
    if _log_path:
        with open(_log_path, "a") as f:
            f.write(msg + "\n")


# ---- model / data loading ----

def load_model_and_tokenizer(model_name, device, dtype=torch.bfloat16):
    from transformers import AutoTokenizer, AutoConfig

    log(f"Loading {model_name} ...")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cfg = AutoConfig.from_pretrained(model_name)
    arch = cfg.architectures[0] if cfg.architectures else ""
    device_map = "auto" if device == "cuda" else "cpu"

    if "Mistral3" in arch:
        from transformers import Mistral3ForConditionalGeneration
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device_map,
            low_cpu_mem_usage=True,
        )
        text_model = model.model.language_model
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device_map,
            low_cpu_mem_usage=True,
        )
        text_model = model

    dt = time.perf_counter() - t0
    n_params = sum(p.numel() for p in text_model.parameters()) / 1e9
    log(f"  Loaded in {dt:.1f}s — {n_params:.2f}B params, device_map={device_map}")

    return model, text_model, tokenizer


def get_wikitext2(tokenizer, split="test", seqlen=2048, nsamples=None):
    """Load WikiText-2 and tokenize into (nsamples, seqlen) batches."""
    from datasets import load_dataset

    log(f"Loading WikiText-2 {split} ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]
    log(f"  {len(input_ids):,} tokens")

    if nsamples is not None:
        max_tokens = nsamples * seqlen
        input_ids = input_ids[:max_tokens]

    n_chunks = input_ids.shape[0] // seqlen
    input_ids = input_ids[:n_chunks * seqlen].reshape(n_chunks, seqlen)
    log(f"  {n_chunks} sequences of length {seqlen}")

    return input_ids


# ---- Hessian capture ----

class HessianCapture:
    """Hook-based Hessian (X^T X) accumulator for a linear layer."""

    def __init__(self, layer: nn.Linear):
        self.n_samples = 0
        self.H = None
        self._hook = layer.register_forward_pre_hook(self._hook_fn)

    def _hook_fn(self, module, inp):
        x = inp[0]
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        x = x.float()
        n = x.shape[0]
        if self.H is None:
            self.H = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32,
                                 device=x.device)
        self.H.addmm_(x.T, x)
        self.n_samples += n

    def finalize(self):
        self._hook.remove()
        if self.H is not None and self.n_samples > 0:
            self.H /= self.n_samples
        return self.H


# ---- layer quantization helpers ----

def pad_to_multiple(W, block_size=16):
    """Pad columns to multiple of block_size."""
    n = W.shape[1]
    pad = (block_size - n % block_size) % block_size
    if pad > 0:
        W = torch.cat([W, torch.zeros(W.shape[0], pad, dtype=W.dtype, device=W.device)], dim=1)
    return W, n, pad


def pad_hessian(H, block_size=16):
    """Pad Hessian to match padded weight columns."""
    n = H.shape[0]
    pad = (block_size - n % block_size) % block_size
    if pad > 0:
        H_pad = torch.zeros(n + pad, n + pad, dtype=H.dtype, device=H.device)
        H_pad[:n, :n] = H
        for i in range(n, n + pad):
            H_pad[i, i] = 1e-6
        return H_pad
    return H


def _metrics(W_orig, W_hat, H):
    """Compute SQNR, proxy loss, Wscale from original and quantized weights."""
    diff = W_orig - W_hat
    mse = (diff ** 2).mean().item()
    signal = (W_orig ** 2).mean().item()
    sqnr = 10 * math.log10(signal / mse) if mse > 0 else float('inf')
    proxy_loss = (diff @ H @ diff.T).diagonal().mean().item()
    return sqnr, proxy_loss


# ---- method: BW16 shell codebook (1-2 bpw) ----

def quantize_layer_shell(W, H, codebook, bpw=1, tune_iters=0):
    """Quantize using BW16 shell codebook LDLQ."""
    dev = codebook.device
    W_pad, n_orig, _ = pad_to_multiple(W.float().to(dev), 16)
    H_pad = pad_hessian(H.float().to(dev), 16)

    # No damping here — codebook functions damp internally
    if bpw == 1:
        result = quantize_ldlq_codebook(W_pad, H_pad, codebook, tune_iters=tune_iters)
    else:
        result = quantize_ldlq_codebook_rvq(W_pad, H_pad, codebook, tune_iters=tune_iters)

    W_hat = result['W_hat'][:, :n_orig]
    W_f = W.float().to(dev)
    sqnr, _ = _metrics(W_f, W_hat, H.float().to(dev))

    return W_hat, {
        'sqnr': sqnr,
        'proxy_loss': result['proxy_loss'],
        'bpw': result['bpw'],
        'Wscale': result['Wscale'],
    }


# ---- method: BW16 Babai LDLQ (2-4 bpw) ----

def quantize_layer_babai(W, H, quantizer, bpw=3, tune_iters=0):
    """Quantize using BW16 Babai nearest-plane LDLQ with Wscale grid search."""
    dev = quantizer.device
    W_pad, n_orig, _ = pad_to_multiple(W.float().to(dev), 16)
    H_pad = pad_hessian(H.float().to(dev), 16)

    # No damping here — quantizer.quantize_ldlq() damps internally
    result = quantizer.quantize_ldlq(W_pad, H_pad, coord_bits=bpw, tune_iters=tune_iters)

    W_hat = result['W_hat'][:, :n_orig]
    W_f = W.float().to(dev)
    sqnr, _ = _metrics(W_f, W_hat, H.float().to(dev))

    return W_hat, {
        'sqnr': sqnr,
        'proxy_loss': result['proxy_loss'],
        'bpw': result['bpw'],
        'Wscale': result['Wscale'],
    }


# ---- method: BW16 Babai LDLQ Hessian-aware (2-4 bpw) ----

def quantize_layer_babai_hessian(W, H, quantizer, bpw=3, tune_iters=0):
    """Quantize using BW16 Babai LDLQ with Hessian-aware decoder.

    Same as quantize_layer_babai but enables hessian_aware=True, so the
    Babai decoder whitens each block by its D_k from the LDL decomposition
    and minimizes (x-target)^T D_k (x-target) instead of ||x-target||^2.
    """
    dev = quantizer.device
    W_pad, n_orig, _ = pad_to_multiple(W.float().to(dev), 16)
    H_pad = pad_hessian(H.float().to(dev), 16)

    result = quantizer.quantize_ldlq(W_pad, H_pad, coord_bits=bpw,
                                      tune_iters=tune_iters,
                                      hessian_aware=True)

    W_hat = result['W_hat'][:, :n_orig]
    W_f = W.float().to(dev)
    sqnr, _ = _metrics(W_f, W_hat, H.float().to(dev))

    return W_hat, {
        'sqnr': sqnr,
        'proxy_loss': result['proxy_loss'],
        'bpw': result['bpw'],
        'Wscale': result['Wscale'],
    }


# ---- method: BW16 Babai LDLQ + RHT (incoherence processing) ----

def quantize_layer_babai_rht(W, H, quantizer, bpw=3, tune_iters=0):
    """Quantize using BW16 Babai LDLQ with Randomized Hadamard Transform.

    Applies RHT to W and H before quantization so that D_k blocks from
    the LDL decomposition become approximately proportional to I.
    This makes Euclidean nearest-neighbor ≈ Hessian-optimal decoding.
    """
    dev = quantizer.device
    m, n = W.shape
    W_f = W.float().to(dev)
    H_f = H.float().to(dev)

    # Apply RHT (pads to power-of-2 internally)
    rht = RHT(m, n, device=dev)
    W_tilde = rht.transform_weights(W_f)
    H_tilde = rht.transform_hessian(H_f)

    # Pad to multiple of 16 for BW16 block structure
    n_tilde = W_tilde.shape[1]
    W_pad, n_rht, _ = pad_to_multiple(W_tilde, 16)
    H_pad = pad_hessian(H_tilde, 16)

    # Quantize in RHT domain (Euclidean ≈ Hessian-optimal)
    result = quantizer.quantize_ldlq(W_pad, H_pad, coord_bits=bpw,
                                      tune_iters=tune_iters)

    # Inverse transform back to original domain
    W_hat_tilde = result['W_hat'][:, :n_tilde]  # unpad BW16 padding
    W_hat = rht.inverse_transform_weights(W_hat_tilde)

    # Metrics in original domain
    sqnr, proxy = _metrics(W_f, W_hat, H_f)
    return W_hat, {
        'sqnr': sqnr,
        'proxy_loss': proxy,
        'bpw': result['bpw'],
        'Wscale': result['Wscale'],
    }


# ---- method: E8 Babai LDLQ + RHT (incoherence processing) ----

def quantize_layer_e8_rht(W, H, quantizer, bpw=3, tune_iters=0):
    """Quantize using E8 Babai LDLQ with Randomized Hadamard Transform.

    Same as BW16+RHT but with E8 (8D) lattice. Benefits:
    - 8x8 D_k blocks concentrate better under incoherence
    - 2x finer block-level error propagation (n/8 vs n/16)
    - Less padding waste (multiple of 8 vs 16)
    """
    dev = quantizer.device
    m, n = W.shape
    W_f = W.float().to(dev)
    H_f = H.float().to(dev)

    # Apply RHT (pads to power-of-2 internally)
    rht = RHT(m, n, device=dev)
    W_tilde = rht.transform_weights(W_f)
    H_tilde = rht.transform_hessian(H_f)

    # Pad to multiple of 8 for E8 block structure
    n_tilde = W_tilde.shape[1]
    W_pad, n_rht, _ = pad_to_multiple(W_tilde, 8)
    H_pad = pad_hessian(H_tilde, 8)

    # Quantize in RHT domain (Euclidean ≈ Hessian-optimal)
    result = quantizer.quantize_ldlq(W_pad, H_pad, coord_bits=bpw,
                                      tune_iters=tune_iters)

    # Inverse transform back to original domain
    W_hat_tilde = result['W_hat'][:, :n_tilde]  # unpad E8 padding
    W_hat = rht.inverse_transform_weights(W_hat_tilde)

    # Metrics in original domain
    sqnr, proxy = _metrics(W_f, W_hat, H_f)
    return W_hat, {
        'sqnr': sqnr,
        'proxy_loss': proxy,
        'bpw': result['bpw'],
        'Wscale': result['Wscale'],
    }


# ---- method: E8 shell codebook (2 bpw, 4 bpw with RVQ) ----

def quantize_layer_e8_shell(W, H, codebook, bpw=2, tune_iters=0):
    """Quantize using E8 shell codebook LDLQ (no RHT).

    At 2 bpw: single-stage (16 bits / 8 dims).
    At 4 bpw: two-stage RVQ (32 bits / 8 dims).
    """
    dev = codebook.device
    W_pad, n_orig, _ = pad_to_multiple(W.float().to(dev), 8)
    H_pad = pad_hessian(H.float().to(dev), 8)

    if bpw == 2:
        result = quantize_ldlq_codebook(W_pad, H_pad, codebook, tune_iters=tune_iters)
    else:
        result = quantize_ldlq_codebook_rvq(W_pad, H_pad, codebook, tune_iters=tune_iters)

    W_hat = result['W_hat'][:, :n_orig]
    W_f = W.float().to(dev)
    sqnr, _ = _metrics(W_f, W_hat, H.float().to(dev))

    return W_hat, {
        'sqnr': sqnr,
        'proxy_loss': result['proxy_loss'],
        'bpw': result['bpw'],
        'Wscale': result['Wscale'],
    }


# ---- method: E8 shell codebook + RHT (2 bpw, 4 bpw with RVQ) ----

def quantize_layer_e8_shell_rht(W, H, codebook, bpw=2, tune_iters=0):
    """Quantize using E8 shell codebook LDLQ with Randomized Hadamard Transform.

    This is the QuIP#-style pipeline:
      1. Apply RHT to W and H for incoherence (D_k blocks -> ~c*I)
      2. Pad to multiple of 8 for E8 block structure
      3. Quantize via E8 shell codebook with LDLQ
      4. Inverse RHT back to original domain

    At 2 bpw: single-stage (16 bits / 8 dims).
    At 4 bpw: two-stage RVQ (32 bits / 8 dims).
    """
    dev = codebook.device
    m, n = W.shape
    W_f = W.float().to(dev)
    H_f = H.float().to(dev)

    # Apply RHT (pads to power-of-2 internally)
    rht = RHT(m, n, device=dev)
    W_tilde = rht.transform_weights(W_f)
    H_tilde = rht.transform_hessian(H_f)

    # Pad to multiple of 8 for E8 block structure
    n_tilde = W_tilde.shape[1]
    W_pad, n_rht, _ = pad_to_multiple(W_tilde, 8)
    H_pad = pad_hessian(H_tilde, 8)

    # Quantize in RHT domain (Euclidean ~ Hessian-optimal)
    if bpw == 2:
        result = quantize_ldlq_codebook(W_pad, H_pad, codebook, tune_iters=tune_iters)
    else:
        result = quantize_ldlq_codebook_rvq(W_pad, H_pad, codebook, tune_iters=tune_iters)

    # Inverse transform back to original domain
    W_hat_tilde = result['W_hat'][:, :n_tilde]  # unpad E8 padding
    W_hat = rht.inverse_transform_weights(W_hat_tilde)

    # Metrics in original domain
    sqnr, proxy = _metrics(W_f, W_hat, H_f)
    return W_hat, {
        'sqnr': sqnr,
        'proxy_loss': proxy,
        'bpw': result['bpw'],
        'Wscale': result['Wscale'],
    }


# ---- method: scalar GPTQ baseline ----

def quantize_layer_scalar_gptq(W, H, bits=3, block_size=128):
    """Standard GPTQ (Frantar et al. 2022) with per-tensor symmetric quantization.

    Uses upper Cholesky of H^{-1} for numerically stable error propagation,
    matching the reference implementation. Scale is optimized via lightweight
    grid search over RMS-based clipping factors.
    """
    dev = W.device
    m, n = W.shape
    W = W.float().clone()
    H = H.float().to(dev)

    # Damping — increase until Cholesky succeeds
    diag_mean = torch.diag(H).mean()
    Hinv = None
    for damp_factor in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        try:
            H_reg = H + damp_factor * diag_mean * torch.eye(n, device=dev)
            L = torch.linalg.cholesky(H_reg)
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
            break
        except torch._C._LinAlgError:
            continue
    if Hinv is None:
        raise RuntimeError("Cholesky failed even with heavy damping")

    # Dead columns: if Cholesky diagonal is tiny, skip those columns
    dead = Hinv.diag() < 1e-6
    if dead.any():
        Hinv[dead, :] = 0
        Hinv[:, dead] = 0
        Hinv[dead, dead] = 1.0
        W[:, dead] = 0.0

    # Optimize scale: search over RMS-based clipping factors
    W_orig = W.clone()
    maxq = 2 ** bits - 1  # e.g., 7 for 3 bits
    half_q = maxq / 2
    W_rms = W.pow(2).mean().sqrt()

    best_scale, best_mse = None, float('inf')
    for clip_mult in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
        trial_max = (clip_mult * W_rms).clamp(min=1e-8)
        trial_scale = trial_max / half_q
        q = torch.clamp(torch.round(W / trial_scale + half_q), 0, maxq)
        q_hat = trial_scale * (q - half_q)
        trial_mse = ((W - q_hat) ** 2).mean().item()
        if trial_mse < best_mse:
            best_mse = trial_mse
            best_scale = trial_scale

    scale = best_scale

    # Block-wise left-to-right GPTQ (Frantar et al. 2022)
    for i1 in range(0, n, block_size):
        i2 = min(i1 + block_size, n)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for j in range(count):
            w = W1[:, j]

            # Quantize
            q = torch.clamp(torch.round(w / scale + half_q), 0, maxq)
            q_hat = scale * (q - half_q)
            Q1[:, j] = q_hat

            # Normalized error for propagation
            err = (w - q_hat) / Hinv1[j, j]
            Err1[:, j] = err
            if j + 1 < count:
                W1[:, j + 1:] -= err.unsqueeze(1) * Hinv1[j, j + 1:count].unsqueeze(0)

        W[:, i1:i2] = Q1

        # Cross-block error propagation (normalized errors × off-diagonal Cholesky)
        if i2 < n:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    W_hat = W
    sqnr, proxy_loss = _metrics(W_orig, W_hat, H_reg)

    return W_hat, {
        'sqnr': sqnr,
        'proxy_loss': proxy_loss,
        'bpw': float(bits),
        'Wscale': scale.item(),
    }


# ---- diagnostic: Exp A — scalar block-16, no within-block propagation ----

def quantize_layer_scalar_block16_noprop(W, H, bits=3):
    """Scalar GPTQ framework with block_size=16 and NO within-block error cascade.

    Isolates H1 (propagation granularity). Same Hinv-based cross-block
    propagation as GPTQ, but within each 16-column block, each column is
    quantized independently — matching LDLQ's propagation granularity.
    """
    dev = W.device
    m, n = W.shape
    W = W.float().clone()
    H = H.float().to(dev)
    block_size = 16

    # Damping + Cholesky of H^{-1} — identical to scalar GPTQ
    diag_mean = torch.diag(H).mean()
    Hinv = None
    for damp_factor in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        try:
            H_reg = H + damp_factor * diag_mean * torch.eye(n, device=dev)
            L = torch.linalg.cholesky(H_reg)
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
            break
        except torch._C._LinAlgError:
            continue
    if Hinv is None:
        raise RuntimeError("Cholesky failed even with heavy damping")

    dead = Hinv.diag() < 1e-6
    if dead.any():
        Hinv[dead, :] = 0
        Hinv[:, dead] = 0
        Hinv[dead, dead] = 1.0
        W[:, dead] = 0.0

    # Scale optimization — identical to scalar GPTQ
    W_orig = W.clone()
    maxq = 2 ** bits - 1
    half_q = maxq / 2
    W_rms = W.pow(2).mean().sqrt()

    best_scale, best_mse = None, float('inf')
    for clip_mult in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
        trial_max = (clip_mult * W_rms).clamp(min=1e-8)
        trial_scale = trial_max / half_q
        q = torch.clamp(torch.round(W / trial_scale + half_q), 0, maxq)
        q_hat = trial_scale * (q - half_q)
        trial_mse = ((W - q_hat) ** 2).mean().item()
        if trial_mse < best_mse:
            best_mse = trial_mse
            best_scale = trial_scale
    scale = best_scale

    # Block-wise processing — NO within-block column cascade
    for i1 in range(0, n, block_size):
        i2 = min(i1 + block_size, n)
        W_block = W[:, i1:i2].clone()

        # Quantize each column independently
        Q_block = scale * (torch.clamp(
            torch.round(W_block / scale + half_q), 0, maxq) - half_q)

        # Block error and cross-block propagation
        E = W_block - Q_block  # (m, count)
        W[:, i1:i2] = Q_block

        if i2 < n:
            Hinv_blk = Hinv[i1:i2, i1:i2]
            E_norm = E @ torch.linalg.inv(Hinv_blk)
            W[:, i2:] -= E_norm @ Hinv[i1:i2, i2:]

    W_hat = W
    sqnr, proxy_loss = _metrics(W_orig, W_hat, H_reg)
    return W_hat, {'sqnr': sqnr, 'proxy_loss': proxy_loss,
                   'bpw': float(bits), 'Wscale': scale.item()}


# ---- diagnostic: Exp B — BW16 LDLQ unbounded ----

def quantize_layer_babai_unbounded(W, H, quantizer, bpw=3):
    """BW16 LDLQ with no coordinate clamping (coord_bits=32).

    Isolates H3 (clamping damage). Effectively unbounded Babai decoder.
    """
    dev = quantizer.device
    W_pad, n_orig, _ = pad_to_multiple(W.float().to(dev), 16)
    H_pad = pad_hessian(H.float().to(dev), 16)
    result = quantizer.quantize_ldlq(W_pad, H_pad, coord_bits=32, tune_iters=0)
    W_hat = result['W_hat'][:, :n_orig]
    W_f = W.float().to(dev)
    sqnr, _ = _metrics(W_f, W_hat, H.float().to(dev))
    return W_hat, {'sqnr': sqnr, 'proxy_loss': result['proxy_loss'],
                   'bpw': float(bpw), 'Wscale': result['Wscale']}


# ---- diagnostic: Exp C — GPTQ framework + BW16 lattice blocks ----

def quantize_layer_gptq_bw16(W, H, quantizer, bits=3):
    """GPTQ Hinv framework with BW16 lattice rounding per 16-col block.

    Isolates H2 (metric mismatch). Same cross-block propagation as Exp A,
    but uses BW16 Babai decoder instead of scalar rounding within each block.
    Exp A - Exp C = lattice effect (positive => lattice hurts).
    """
    dev = W.device
    m, n = W.shape
    W = W.float().clone()
    H = H.float().to(dev)
    block_size = 16

    # Pad to multiple of 16
    orig_n = n
    pad = (block_size - n % block_size) % block_size
    if pad > 0:
        W = torch.cat([W, torch.zeros(m, pad, dtype=W.dtype, device=dev)], dim=1)
        H_new = torch.eye(n + pad, dtype=H.dtype, device=dev) * 1e-6
        H_new[:n, :n] = H
        H = H_new
        n = n + pad

    # Damping + Cholesky of H^{-1}
    diag_mean = torch.diag(H).mean()
    Hinv = None
    for damp_factor in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        try:
            H_reg = H + damp_factor * diag_mean * torch.eye(n, device=dev)
            L = torch.linalg.cholesky(H_reg)
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
            break
        except torch._C._LinAlgError:
            continue
    if Hinv is None:
        raise RuntimeError("Cholesky failed even with heavy damping")

    dead = Hinv.diag() < 1e-6
    if dead.any():
        Hinv[dead, :] = 0
        Hinv[:, dead] = 0
        Hinv[dead, dead] = 1.0
        W[:, dead] = 0.0

    W_orig = W[:, :orig_n].clone()

    # Wscale grid search (LDLQ-style): find best global scale for BW16
    W_rms = W.pow(2).mean().sqrt()
    best_Wscale, best_mse = None, float('inf')
    for opt_s in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0,
                  1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]:
        trial_scale = (W_rms / opt_s).item() if W_rms > 1e-10 else 1.0
        Wr = W / trial_scale
        groups = Wr.reshape(-1, block_size)
        decoded, _ = quantizer.babai_round_global(
            groups.to(quantizer.device), coord_bits=bits)
        trial_hat = decoded.to(dev).reshape(m, n) * trial_scale
        trial_mse = ((W - trial_hat) ** 2).mean().item()
        if trial_mse < best_mse:
            best_mse = trial_mse
            best_Wscale = trial_scale
    Wscale = best_Wscale

    # Left-to-right block processing with BW16 rounding
    for i1 in range(0, n, block_size):
        i2 = i1 + block_size
        W_block = W[:, i1:i2].clone()

        # BW16 lattice decode
        target = W_block / Wscale
        decoded, _ = quantizer.babai_round_global(
            target.to(quantizer.device), coord_bits=bits)
        Q_block = decoded.to(dev) * Wscale

        # Block error and cross-block propagation
        E = W_block - Q_block
        W[:, i1:i2] = Q_block

        if i2 < n:
            Hinv_blk = Hinv[i1:i2, i1:i2]
            E_norm = E @ torch.linalg.inv(Hinv_blk)
            W[:, i2:] -= E_norm @ Hinv[i1:i2, i2:]

    W_hat = W[:, :orig_n]
    sqnr, proxy_loss = _metrics(W_orig, W_hat, H_reg[:orig_n, :orig_n])
    return W_hat, {'sqnr': sqnr, 'proxy_loss': proxy_loss,
                   'bpw': float(bits), 'Wscale': Wscale}


# ---- diagnostic: Exp D — GPTQ framework + Hessian-aware BW16 lattice blocks ----

def quantize_layer_gptq_bw16_hessian(W, H, quantizer, bits=3):
    """GPTQ Hinv framework with Hessian-aware BW16 lattice rounding per block.

    Same as Exp C (quantize_layer_gptq_bw16) but uses the Hessian-aware Babai
    decoder that whitens each 16-col block by H_reg[kb:ke, kb:ke] before
    running Babai nearest-plane. This minimizes (x-target)^T D_k (x-target)
    instead of ||x-target||^2, closing the metric mismatch gap.

    Exp D - Exp C = improvement from Hessian-aware decoding.
    Exp A - Exp D = remaining gap (lattice packing loss vs scalar at this bpw).
    """
    dev = W.device
    m, n = W.shape
    W = W.float().clone()
    H = H.float().to(dev)
    block_size = 16

    # Pad to multiple of 16
    orig_n = n
    pad = (block_size - n % block_size) % block_size
    if pad > 0:
        W = torch.cat([W, torch.zeros(m, pad, dtype=W.dtype, device=dev)], dim=1)
        H_new = torch.eye(n + pad, dtype=H.dtype, device=dev) * 1e-6
        H_new[:n, :n] = H
        H = H_new
        n = n + pad

    # Damping + Cholesky of H^{-1}
    diag_mean = torch.diag(H).mean()
    Hinv = None
    for damp_factor in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        try:
            H_reg = H + damp_factor * diag_mean * torch.eye(n, device=dev)
            L = torch.linalg.cholesky(H_reg)
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
            break
        except torch._C._LinAlgError:
            continue
    if Hinv is None:
        raise RuntimeError("Cholesky failed even with heavy damping")

    dead = Hinv.diag() < 1e-6
    if dead.any():
        Hinv[dead, :] = 0
        Hinv[:, dead] = 0
        Hinv[dead, dead] = 1.0
        W[:, dead] = 0.0

    W_orig = W[:, :orig_n].clone()

    # Wscale grid search (LDLQ-style): find best global scale for BW16
    # Use Hessian-aware decoder during grid search too for consistency
    W_rms = W.pow(2).mean().sqrt()
    best_Wscale, best_mse = None, float('inf')
    for opt_s in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0,
                  1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]:
        trial_scale = (W_rms / opt_s).item() if W_rms > 1e-10 else 1.0
        Wr = W / trial_scale
        # Quick proxy: use Euclidean decode for grid search (much faster)
        groups = Wr.reshape(-1, block_size)
        decoded, _ = quantizer.babai_round_global(
            groups.to(quantizer.device), coord_bits=bits)
        trial_hat = decoded.to(dev).reshape(m, n) * trial_scale
        trial_mse = ((W - trial_hat) ** 2).mean().item()
        if trial_mse < best_mse:
            best_mse = trial_mse
            best_Wscale = trial_scale
    Wscale = best_Wscale

    # Left-to-right block processing with Hessian-aware BW16 rounding
    for i1 in range(0, n, block_size):
        i2 = i1 + block_size
        W_block = W[:, i1:i2].clone()

        # Per-block Hessian: H_reg[i1:i2, i1:i2]
        D_k = H_reg[i1:i2, i1:i2]

        # Hessian-aware BW16 lattice decode
        target = W_block / Wscale
        decoded, _ = quantizer.babai_round_global_weighted(
            target.to(quantizer.device), D_k.to(quantizer.device),
            coord_bits=bits)
        Q_block = decoded.to(dev) * Wscale

        # Block error and cross-block propagation
        E = W_block - Q_block
        W[:, i1:i2] = Q_block

        if i2 < n:
            Hinv_blk = Hinv[i1:i2, i1:i2]
            E_norm = E @ torch.linalg.inv(Hinv_blk)
            W[:, i2:] -= E_norm @ Hinv[i1:i2, i2:]

    W_hat = W[:, :orig_n]
    sqnr, proxy_loss = _metrics(W_orig, W_hat, H_reg[:orig_n, :orig_n])
    return W_hat, {'sqnr': sqnr, 'proxy_loss': proxy_loss,
                   'bpw': float(bits), 'Wscale': Wscale}


# ---- diagnostic: Exp E — GPTQ framework + BW16 blocks + RHT ----

def quantize_layer_gptq_bw16_rht(W, H, quantizer, bits=3):
    """GPTQ Hinv framework with BW16 lattice rounding + RHT incoherence.

    Same as Exp C but with RHT preprocessing. RHT makes D_k ≈ c*I so
    Euclidean BW16 decoding ≈ Hessian-optimal. This is the QuIP#-style
    pipeline with BW16 instead of E8.
    """
    dev = W.device
    m_orig, n_orig = W.shape
    W_input = W.float().clone()
    H_input = H.float().to(dev)
    block_size = 16

    # Apply RHT (pads to power-of-2 internally)
    rht = RHT(m_orig, n_orig, device=dev)
    W = rht.transform_weights(W_input)
    H = rht.transform_hessian(H_input)
    m, n = W.shape  # padded to power-of-2 (always multiple of 16)

    # Damping + Cholesky of H^{-1}
    diag_mean = torch.diag(H).mean()
    Hinv = None
    for damp_factor in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        try:
            H_reg = H + damp_factor * diag_mean * torch.eye(n, device=dev)
            L_chol = torch.linalg.cholesky(H_reg)
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L_chol), upper=True)
            break
        except torch._C._LinAlgError:
            continue
    if Hinv is None:
        raise RuntimeError("Cholesky failed even with heavy damping")

    dead = Hinv.diag() < 1e-6
    if dead.any():
        Hinv[dead, :] = 0
        Hinv[:, dead] = 0
        Hinv[dead, dead] = 1.0
        W[:, dead] = 0.0

    # Wscale grid search
    W_rms = W.pow(2).mean().sqrt()
    best_Wscale, best_mse = None, float('inf')
    for opt_s in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0,
                  1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]:
        trial_scale = (W_rms / opt_s).item() if W_rms > 1e-10 else 1.0
        Wr = W / trial_scale
        groups = Wr.reshape(-1, block_size)
        decoded, _ = quantizer.babai_round_global(
            groups.to(quantizer.device), coord_bits=bits)
        trial_hat = decoded.to(dev).reshape(m, n) * trial_scale
        trial_mse = ((W - trial_hat) ** 2).mean().item()
        if trial_mse < best_mse:
            best_mse = trial_mse
            best_Wscale = trial_scale
    Wscale = best_Wscale

    # Left-to-right block processing with BW16 rounding
    for i1 in range(0, n, block_size):
        i2 = i1 + block_size
        W_block = W[:, i1:i2].clone()

        target = W_block / Wscale
        decoded, _ = quantizer.babai_round_global(
            target.to(quantizer.device), coord_bits=bits)
        Q_block = decoded.to(dev) * Wscale

        E = W_block - Q_block
        W[:, i1:i2] = Q_block

        if i2 < n:
            Hinv_blk = Hinv[i1:i2, i1:i2]
            E_norm = E @ torch.linalg.inv(Hinv_blk)
            W[:, i2:] -= E_norm @ Hinv[i1:i2, i2:]

    # Inverse RHT to original domain
    W_hat = rht.inverse_transform_weights(W)

    # Metrics in original domain
    sqnr, proxy_loss = _metrics(W_input.to(dev), W_hat, H_input)
    return W_hat, {'sqnr': sqnr, 'proxy_loss': proxy_loss,
                   'bpw': float(bits), 'Wscale': Wscale}


# ---- dispatch ----

def quantize_layer(W, H, method, quantizer, bpw, tune_iters=0, block_size=128):
    """Dispatch quantization by method."""
    if method == 'bw16_shell':
        return quantize_layer_shell(W, H, quantizer, bpw=bpw, tune_iters=tune_iters)
    elif method == 'bw16_babai':
        return quantize_layer_babai(W, H, quantizer, bpw=bpw, tune_iters=tune_iters)
    elif method == 'bw16_babai_hessian':
        return quantize_layer_babai_hessian(W, H, quantizer, bpw=bpw, tune_iters=tune_iters)
    elif method == 'scalar_gptq':
        return quantize_layer_scalar_gptq(W, H, bits=bpw, block_size=block_size)
    elif method == 'diag_scalar_noprop':
        return quantize_layer_scalar_block16_noprop(W, H, bits=bpw)
    elif method == 'diag_bw16_unbounded':
        return quantize_layer_babai_unbounded(W, H, quantizer, bpw=bpw)
    elif method == 'diag_gptq_bw16':
        return quantize_layer_gptq_bw16(W, H, quantizer, bits=bpw)
    elif method == 'diag_bw16_hessian':
        return quantize_layer_gptq_bw16_hessian(W, H, quantizer, bits=bpw)
    elif method == 'bw16_babai_rht':
        return quantize_layer_babai_rht(W, H, quantizer, bpw=bpw, tune_iters=tune_iters)
    elif method == 'diag_gptq_bw16_rht':
        return quantize_layer_gptq_bw16_rht(W, H, quantizer, bits=bpw)
    elif method == 'e8_babai_rht':
        return quantize_layer_e8_rht(W, H, quantizer, bpw=bpw, tune_iters=tune_iters)
    elif method == 'e8_shell':
        return quantize_layer_e8_shell(W, H, quantizer, bpw=bpw, tune_iters=tune_iters)
    elif method == 'e8_shell_rht':
        return quantize_layer_e8_shell_rht(W, H, quantizer, bpw=bpw, tune_iters=tune_iters)
    else:
        raise ValueError(f"Unknown method: {method}")


# ---- perplexity evaluation ----

@torch.no_grad()
def evaluate_perplexity(model, input_ids, device, batch_size=1):
    model.eval()
    n_seqs = input_ids.shape[0]
    total_nll = 0.0
    total_tokens = 0

    for i in range(0, n_seqs, batch_size):
        batch = input_ids[i:i + batch_size].to(device)
        logits = model(input_ids=batch, use_cache=False).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        nll = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
        ).item()

        total_nll += nll
        total_tokens += shift_labels.numel()

        if i % max(1, n_seqs // 10) == 0 or i + batch_size >= n_seqs:
            ppl_so_far = math.exp(total_nll / total_tokens)
            log(f"  [{i+1}/{n_seqs}] ppl={ppl_so_far:.2f}")

    return math.exp(total_nll / total_tokens)


# ---- model structure helpers ----

def get_decoder_layers(text_model):
    if hasattr(text_model, 'layers'):
        return text_model.layers
    if hasattr(text_model, 'model') and hasattr(text_model.model, 'layers'):
        return text_model.model.layers
    raise ValueError("Cannot find transformer layers")


def get_embed(text_model):
    if hasattr(text_model, 'embed_tokens'):
        return text_model.embed_tokens
    if hasattr(text_model, 'model') and hasattr(text_model.model, 'embed_tokens'):
        return text_model.model.embed_tokens
    raise ValueError("Cannot find embedding layer")


def get_rotary_emb(text_model):
    if hasattr(text_model, 'rotary_emb'):
        return text_model.rotary_emb
    if hasattr(text_model, 'model') and hasattr(text_model.model, 'rotary_emb'):
        return text_model.model.rotary_emb
    return None


# ---- GPTQ-style quantization pipeline ----

def quantize_model(text_model, calib_ids, quantizer, device,
                   method='bw16_shell', bpw=1, tune_iters=0, block_size=128):
    """
    GPTQ-style layer-by-layer quantization.

    Processes one decoder layer at a time:
      1. Forward calibration data, capturing Hessians for each linear sublayer
      2. Quantize each linear using chosen method with the captured Hessian
      3. Forward calibration data through the quantized layer for next layer
    """
    method_display = METHOD_NAMES[method]
    if method == 'bw16_shell' and bpw == 2:
        method_display += ' RVQ'
    if method in ('e8_shell', 'e8_shell_rht') and bpw == 4:
        method_display += ' RVQ'

    log(f"\n{'='*60}")
    log(f"Quantizing — {method_display} @ {bpw} bpw, tune={tune_iters}")
    log(f"{'='*60}")

    decoder_layers = get_decoder_layers(text_model)
    embed = get_embed(text_model)
    rotary_emb = get_rotary_emb(text_model)
    n_layers = len(decoder_layers)
    all_sqnr = []
    t_start = time.perf_counter()

    bw16_methods = ('bw16_shell', 'bw16_babai', 'bw16_babai_hessian',
                     'diag_bw16_unbounded', 'diag_gptq_bw16',
                     'diag_bw16_hessian', 'diag_scalar_noprop')
    e8_methods = ('e8_shell', 'e8_shell_rht', 'e8_babai_rht')
    if method in bw16_methods:
        min_cols = 16
    elif method in e8_methods:
        min_cols = 8
    else:
        min_cols = 1

    # Embed calibration data
    log(f"\nEmbedding calibration data ({calib_ids.shape[0]} seqs) ...")
    hidden_states = []
    with torch.no_grad():
        for i in range(calib_ids.shape[0]):
            hidden_states.append(embed(calib_ids[i:i+1].to(device)))
    hidden_states = torch.cat(hidden_states, dim=0)
    log(f"  Shape: {hidden_states.shape}")

    for layer_idx in range(n_layers):
        layer = decoder_layers[layer_idx]
        t_layer = time.perf_counter()
        log(f"\n--- Layer {layer_idx}/{n_layers-1} ---")

        # Collect linear sublayers
        linears = {}
        for name, mod in layer.named_modules():
            if isinstance(mod, nn.Linear):
                linears[name] = mod

        # Install Hessian hooks
        captures = {}
        for name, mod in linears.items():
            captures[name] = HessianCapture(mod)

        # Forward calibration through this layer (hooks capture inputs)
        with torch.no_grad():
            for i in range(hidden_states.shape[0]):
                h = hidden_states[i:i+1]
                seq_len = h.shape[1]
                cache_position = torch.arange(seq_len, device=h.device)
                position_ids = cache_position.unsqueeze(0)
                kwargs = dict(position_ids=position_ids, cache_position=cache_position,
                              use_cache=False)
                if rotary_emb is not None:
                    kwargs['position_embeddings'] = rotary_emb(h, position_ids=position_ids)
                layer(h, **kwargs)

        # Quantize each linear
        for name, cap in captures.items():
            H = cap.finalize()
            if H is None:
                log(f"  {name}: no activations, skipping")
                continue

            mod = linears[name]
            W = mod.weight.data

            if W.shape[1] < min_cols:
                log(f"  {name}: in={W.shape[1]} < {min_cols}, skipping")
                continue

            t0 = time.perf_counter()
            W_hat, metrics = quantize_layer(W, H, method, quantizer,
                                            bpw=bpw, tune_iters=tune_iters,
                                            block_size=block_size)
            dt = time.perf_counter() - t0

            mod.weight.data = W_hat.to(dtype=mod.weight.dtype, device=mod.weight.device)
            all_sqnr.append(metrics['sqnr'])

            log(f"  {name:30s} {str(tuple(W.shape)):20s} "
                f"SQNR={metrics['sqnr']:5.1f}dB  "
                f"Ws={metrics['Wscale']:.3f}  "
                f"{dt:.1f}s")

            del H
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        # Forward calibration through quantized layer for next layer's input
        new_hidden = []
        with torch.no_grad():
            for i in range(hidden_states.shape[0]):
                h = hidden_states[i:i+1]
                seq_len = h.shape[1]
                cache_position = torch.arange(seq_len, device=h.device)
                position_ids = cache_position.unsqueeze(0)
                kwargs = dict(position_ids=position_ids, cache_position=cache_position,
                              use_cache=False)
                if rotary_emb is not None:
                    kwargs['position_embeddings'] = rotary_emb(h, position_ids=position_ids)
                out = layer(h, **kwargs)
                new_hidden.append(out[0] if isinstance(out, tuple) else out)
        hidden_states = torch.cat(new_hidden, dim=0)

        dt_layer = time.perf_counter() - t_layer
        elapsed = time.perf_counter() - t_start
        remaining = elapsed / (layer_idx + 1) * (n_layers - layer_idx - 1)
        eta = datetime.now() + timedelta(seconds=remaining)
        log(f"  Layer time: {dt_layer:.0f}s  |  "
            f"Elapsed: {elapsed/3600:.1f}h  |  "
            f"ETA: {eta.strftime('%Y-%m-%d %H:%M')}")

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    total_time = time.perf_counter() - t_start
    avg_sqnr = sum(all_sqnr) / len(all_sqnr) if all_sqnr else 0
    log(f"\n{'='*60}")
    log(f"Quantized {len(all_sqnr)} sublayers in {total_time/3600:.1f}h")
    log(f"Average SQNR = {avg_sqnr:.2f} dB")
    log(f"{'='*60}")

    return avg_sqnr


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM perplexity under quantization")
    parser.add_argument("--model", type=str,
                        default="mistralai/Ministral-3-3B-Reasoning-2512")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--method", type=str, default="bw16_shell",
                        choices=list(METHOD_NAMES.keys()),
                        help="Quantization method")
    parser.add_argument("--bpw", type=int, default=1,
                        choices=[1, 2, 3, 4])
    parser.add_argument("--block-size", type=int, default=128,
                        help="GPTQ block size (default 128; use 16 for Exp D)")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--n-calib", type=int, default=16,
                        help="Calibration sequences")
    parser.add_argument("--n-test", type=int, default=None,
                        help="Max test sequences (None=all)")
    parser.add_argument("--tune-iters", type=int, default=0,
                        help="LDLQ refinement passes (0=fastest)")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--log", type=str, default=None,
                        help="Log file path")
    args = parser.parse_args()

    # Validate method/bpw combinations
    if args.method == 'bw16_shell' and args.bpw not in (1, 2):
        parser.error("bw16_shell only supports --bpw 1 or 2")
    if args.method == 'bw16_babai' and args.bpw < 2:
        parser.error("bw16_babai needs --bpw >= 2")
    if args.method == 'bw16_babai_hessian' and args.bpw < 2:
        parser.error("bw16_babai_hessian needs --bpw >= 2")
    if args.method in ('diag_scalar_noprop', 'diag_gptq_bw16', 'diag_bw16_hessian',
                       'diag_gptq_bw16_rht') and args.bpw < 2:
        parser.error(f"{args.method} needs --bpw >= 2")
    if args.method in ('bw16_babai_rht', 'e8_babai_rht') and args.bpw < 2:
        parser.error(f"{args.method} needs --bpw >= 2")
    if args.method in ('e8_shell', 'e8_shell_rht') and args.bpw not in (2, 4):
        parser.error(f"{args.method} only supports --bpw 2 or 4")

    if args.device == "cuda" and not torch.cuda.is_available():
        log("CUDA not available, falling back to CPU")
        args.device = "cpu"

    global _log_path
    _log_path = args.log or str(
        Path(__file__).parent / f"eval_ppl_{args.method}_{args.bpw}bpw.log")

    with open(_log_path, "w") as f:
        f.write(f"# eval_ppl.py — {datetime.now().isoformat()}\n")
        f.write(f"# {vars(args)}\n\n")

    log(f"Args: {vars(args)}")
    log(f"Log:  {_log_path}")
    log(f"Start: {datetime.now().isoformat()}")

    if args.device == "cuda":
        log(f"GPU:  {torch.cuda.get_device_name()}")
        log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Initialize quantizer/codebook
    needs_bw16 = args.method in ('bw16_babai', 'bw16_babai_hessian',
                                 'diag_bw16_unbounded', 'diag_gptq_bw16',
                                 'diag_bw16_hessian',
                                 'bw16_babai_rht', 'diag_gptq_bw16_rht')
    needs_e8 = args.method in ('e8_babai_rht',)
    needs_e8_shell = args.method in ('e8_shell', 'e8_shell_rht')
    quantizer = None
    if not args.baseline_only:
        if args.method == 'bw16_shell':
            log("\nBuilding BW16 shell codebook ...")
            t0 = time.perf_counter()
            quantizer = BW16ShellCodebook(device=args.device, verbose=True)
            log(f"  Built in {time.perf_counter()-t0:.1f}s")
        elif needs_e8_shell:
            log("\nBuilding E8 shell codebook ...")
            t0 = time.perf_counter()
            quantizer = E8ShellCodebook(device=args.device, verbose=True)
            log(f"  Built in {time.perf_counter()-t0:.1f}s")
        elif needs_bw16:
            log("\nInitializing BW16 Babai quantizer ...")
            t0 = time.perf_counter()
            quantizer = BW16Quantizer(device=args.device)
            log(f"  Ready in {time.perf_counter()-t0:.1f}s")
        elif needs_e8:
            log("\nInitializing E8 Babai quantizer ...")
            t0 = time.perf_counter()
            quantizer = E8Quantizer(device=args.device)
            log(f"  Ready in {time.perf_counter()-t0:.1f}s")
        elif args.method in ('scalar_gptq', 'diag_scalar_noprop'):
            log(f"\n{METHOD_NAMES[args.method]} — no codebook needed")

    # Load model
    model, text_model, tokenizer = load_model_and_tokenizer(args.model, args.device)

    # Load data
    calib_ids = get_wikitext2(tokenizer, split="train",
                              seqlen=args.seqlen, nsamples=args.n_calib)
    test_ids = get_wikitext2(tokenizer, split="test",
                             seqlen=args.seqlen, nsamples=args.n_test)

    # Baseline
    log(f"\n{'='*60}")
    log("BF16 Baseline Perplexity")
    log(f"{'='*60}")
    t0 = time.perf_counter()
    ppl_baseline = evaluate_perplexity(model, test_ids, args.device)
    log(f"  BF16 ppl: {ppl_baseline:.2f}  ({time.perf_counter()-t0:.1f}s)")

    if args.baseline_only:
        log(f"\nDone: {datetime.now().isoformat()}")
        return

    # Quantize
    avg_sqnr = quantize_model(text_model, calib_ids, quantizer, args.device,
                               method=args.method, bpw=args.bpw,
                               tune_iters=args.tune_iters,
                               block_size=args.block_size)

    # Quantized perplexity
    method_display = METHOD_NAMES[args.method]
    if args.method == 'bw16_shell' and args.bpw == 2:
        method_display += ' RVQ'
    if args.method in ('e8_shell', 'e8_shell_rht') and args.bpw == 4:
        method_display += ' RVQ'

    log(f"\n{'='*60}")
    log(f"{method_display} @ {args.bpw} bpw Perplexity")
    log(f"{'='*60}")
    t0 = time.perf_counter()
    ppl_quant = evaluate_perplexity(model, test_ids, args.device)
    log(f"  Quant ppl: {ppl_quant:.2f}  ({time.perf_counter()-t0:.1f}s)")

    # ---- summary ----
    log(f"\n{'='*60}")
    log("SUMMARY")
    log(f"{'='*60}")
    log(f"  Model:       {args.model}")
    log(f"  Device:      {args.device}")
    log(f"  Method:      {method_display}")
    log(f"  bpw:         {args.bpw}")
    log(f"  tune_iters:  {args.tune_iters}")
    log(f"  n_calib:     {args.n_calib}")
    log(f"  seqlen:      {args.seqlen}")
    log(f"  Avg SQNR:    {avg_sqnr:.2f} dB")
    log(f"  BF16 ppl:    {ppl_baseline:.2f}")
    log(f"  Quant ppl:   {ppl_quant:.2f}")
    log(f"  Delta ppl:   {ppl_quant - ppl_baseline:+.2f}")
    log(f"  Ratio:       {ppl_quant / ppl_baseline:.3f}x")
    log(f"\nDone: {datetime.now().isoformat()}")

    # Save results JSON
    results = {
        'model': args.model,
        'device': args.device,
        'method': method_display,
        'bpw': args.bpw,
        'tune_iters': args.tune_iters,
        'n_calib': args.n_calib,
        'seqlen': args.seqlen,
        'avg_sqnr': avg_sqnr,
        'ppl_baseline': ppl_baseline,
        'ppl_quant': ppl_quant,
        'ppl_delta': ppl_quant - ppl_baseline,
        'ppl_ratio': ppl_quant / ppl_baseline,
    }
    json_path = str(Path(__file__).parent /
                     f"eval_ppl_{args.method}_{args.bpw}bpw_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
