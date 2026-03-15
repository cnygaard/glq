"""Profile per-phase GPU timings for GLQ quantization.

Instruments quantize_layer_e8_shell_rht to measure:
  - RHT forward (weight + Hessian transforms)
  - Block LDL decomposition
  - LDLQ inner loop (feedback matmul vs codebook NN)
  - RHT inverse transform

Usage:
    python benchmarks/profile_quantize.py                    # SmolLM2-360M 3bpw
    python benchmarks/profile_quantize.py --bpw 2
    python benchmarks/profile_quantize.py --sublayer gate_proj
"""

import argparse
import gc
import time
import torch
import torch.nn as nn

from glq.codebook import E8ShellCodebook
from glq.rht import RHT
from glq.ldlq import block_LDL
from glq.quantize_model import pad_to_multiple, pad_hessian


def profile_ldlq_2stage(W_pad, H_pad, codebook, codebook2, resid_scale):
    """Instrumented 2-stage LDLQ with per-phase CUDA event timing."""
    device = codebook.device
    m, n = W_pad.shape
    b = codebook.codesz
    num_blocks = n // b

    W_pad = W_pad.float().to(device)
    H_pad = H_pad.float().to(device)

    # --- Block LDL ---
    torch.cuda.synchronize()
    ev_ldl_s = torch.cuda.Event(enable_timing=True)
    ev_ldl_e = torch.cuda.Event(enable_timing=True)
    ev_ldl_s.record()

    damp = 0.01 * torch.diag(H_pad).mean()
    H_reg = H_pad + damp * torch.eye(n, device=device)
    L, D = block_LDL(H_reg, block_size=b)

    ev_ldl_e.record()

    # --- Wscale ---
    W_rms = W_pad.pow(2).mean().sqrt().item()
    Wscale = W_rms / codebook.opt_scale if W_rms > 1e-10 else 1.0
    Wr = W_pad / Wscale
    hatWr = torch.zeros_like(Wr)
    all_indices1 = torch.zeros(m, num_blocks, dtype=torch.long, device=device)
    all_indices2 = torch.zeros(m, num_blocks, dtype=torch.long, device=device)

    # --- LDLQ inner loop with sub-phase timing ---
    t_resid_ms = 0.0
    t_feedback_ms = 0.0
    t_cb1_ms = 0.0
    t_cb2_ms = 0.0
    t_update_ms = 0.0

    torch.cuda.synchronize()
    ev_loop_s = torch.cuda.Event(enable_timing=True)
    ev_loop_e = torch.cuda.Event(enable_timing=True)
    ev_loop_s.record()

    for k in reversed(range(num_blocks)):
        kb, ke = k * b, (k + 1) * b

        if ke < n:
            # (a) Residual subtraction
            ev_a_s = torch.cuda.Event(enable_timing=True)
            ev_a_e = torch.cuda.Event(enable_timing=True)
            ev_a_s.record()
            resid = Wr[:, ke:] - hatWr[:, ke:]
            ev_a_e.record()

            # (b) Feedback matmul
            ev_b_s = torch.cuda.Event(enable_timing=True)
            ev_b_e = torch.cuda.Event(enable_timing=True)
            ev_b_s.record()
            feedback = resid @ L[ke:, kb:ke]
            ev_b_e.record()

            # Sync to read sub-phase times
            ev_b_e.synchronize()
            t_resid_ms += ev_a_s.elapsed_time(ev_a_e)
            t_feedback_ms += ev_b_s.elapsed_time(ev_b_e)
        else:
            feedback = 0.0

        target = Wr[:, kb:ke] + feedback

        # (c) Codebook 1 quantize
        ev_c_s = torch.cuda.Event(enable_timing=True)
        ev_c_e = torch.cuda.Event(enable_timing=True)
        ev_c_s.record()
        dec1, idx1 = codebook.quantize(target)
        ev_c_e.record()

        # (d) Codebook 2 quantize (residual)
        ev_d_s = torch.cuda.Event(enable_timing=True)
        ev_d_e = torch.cuda.Event(enable_timing=True)
        ev_d_s.record()
        residual = target - dec1
        dec2, idx2 = codebook2.quantize(residual * resid_scale)
        ev_d_e.record()

        # (e) Update hatWr
        ev_e_s = torch.cuda.Event(enable_timing=True)
        ev_e_e = torch.cuda.Event(enable_timing=True)
        ev_e_s.record()
        all_indices1[:, k] = idx1
        all_indices2[:, k] = idx2
        hatWr[:, kb:ke] = dec1 + dec2 / resid_scale
        ev_e_e.record()

        ev_e_e.synchronize()
        t_cb1_ms += ev_c_s.elapsed_time(ev_c_e)
        t_cb2_ms += ev_d_s.elapsed_time(ev_d_e)
        t_update_ms += ev_e_s.elapsed_time(ev_e_e)

    ev_loop_e.record()
    torch.cuda.synchronize()

    ldl_ms = ev_ldl_s.elapsed_time(ev_ldl_e)
    loop_ms = ev_loop_s.elapsed_time(ev_loop_e)

    return {
        'block_ldl_ms': ldl_ms,
        'loop_total_ms': loop_ms,
        'resid_sub_ms': t_resid_ms,
        'feedback_matmul_ms': t_feedback_ms,
        'codebook1_ms': t_cb1_ms,
        'codebook2_ms': t_cb2_ms,
        'update_ms': t_update_ms,
        'W_hat': hatWr * Wscale,
        'Wscale': Wscale,
    }


def profile_ldlq_1stage(W_pad, H_pad, codebook):
    """Instrumented 1-stage LDLQ (2bpw) with per-phase timing."""
    device = codebook.device
    m, n = W_pad.shape
    b = codebook.codesz
    num_blocks = n // b

    W_pad = W_pad.float().to(device)
    H_pad = H_pad.float().to(device)

    torch.cuda.synchronize()
    ev_ldl_s = torch.cuda.Event(enable_timing=True)
    ev_ldl_e = torch.cuda.Event(enable_timing=True)
    ev_ldl_s.record()

    damp = 0.01 * torch.diag(H_pad).mean()
    H_reg = H_pad + damp * torch.eye(n, device=device)
    L, D = block_LDL(H_reg, block_size=b)

    ev_ldl_e.record()

    W_rms = W_pad.pow(2).mean().sqrt().item()
    Wscale = W_rms / codebook.opt_scale if W_rms > 1e-10 else 1.0
    Wr = W_pad / Wscale
    hatWr = torch.zeros_like(Wr)
    all_indices = torch.zeros(m, num_blocks, dtype=torch.long, device=device)

    t_resid_ms = 0.0
    t_feedback_ms = 0.0
    t_cb_ms = 0.0
    t_update_ms = 0.0

    torch.cuda.synchronize()
    ev_loop_s = torch.cuda.Event(enable_timing=True)
    ev_loop_e = torch.cuda.Event(enable_timing=True)
    ev_loop_s.record()

    for k in reversed(range(num_blocks)):
        kb, ke = k * b, (k + 1) * b

        if ke < n:
            ev_a_s = torch.cuda.Event(enable_timing=True)
            ev_a_e = torch.cuda.Event(enable_timing=True)
            ev_a_s.record()
            resid = Wr[:, ke:] - hatWr[:, ke:]
            ev_a_e.record()

            ev_b_s = torch.cuda.Event(enable_timing=True)
            ev_b_e = torch.cuda.Event(enable_timing=True)
            ev_b_s.record()
            feedback = resid @ L[ke:, kb:ke]
            ev_b_e.record()

            ev_b_e.synchronize()
            t_resid_ms += ev_a_s.elapsed_time(ev_a_e)
            t_feedback_ms += ev_b_s.elapsed_time(ev_b_e)
        else:
            feedback = 0.0

        WXWX = Wr[:, kb:ke] + feedback

        ev_c_s = torch.cuda.Event(enable_timing=True)
        ev_c_e = torch.cuda.Event(enable_timing=True)
        ev_c_s.record()
        hatWr[:, kb:ke], all_indices[:, k] = codebook.quantize(WXWX)
        ev_c_e.record()

        ev_c_e.synchronize()
        t_cb_ms += ev_c_s.elapsed_time(ev_c_e)

    ev_loop_e.record()
    torch.cuda.synchronize()

    ldl_ms = ev_ldl_s.elapsed_time(ev_ldl_e)
    loop_ms = ev_loop_s.elapsed_time(ev_loop_e)

    return {
        'block_ldl_ms': ldl_ms,
        'loop_total_ms': loop_ms,
        'resid_sub_ms': t_resid_ms,
        'feedback_matmul_ms': t_feedback_ms,
        'codebook1_ms': t_cb_ms,
        'codebook2_ms': 0.0,
        'update_ms': 0.0,
        'W_hat': hatWr * Wscale,
        'Wscale': Wscale,
    }


def profile_sublayer(W, H, codebook, bpw=3):
    """Profile one sublayer quantization with full phase breakdown."""
    dev = codebook.device
    m, n = W.shape
    W_f = W.float().to(dev)
    H_f = H.float().to(dev)

    # Dampen
    damp = 0.01 * torch.mean(torch.diag(H_f))
    diag = torch.arange(H_f.shape[-1], device=dev)
    H_f[diag, diag] += damp

    # --- RHT Forward ---
    torch.cuda.synchronize()
    ev_rht_s = torch.cuda.Event(enable_timing=True)
    ev_rht_e = torch.cuda.Event(enable_timing=True)
    ev_rht_s.record()

    rht = RHT(m, n, device=dev)
    W_tilde = rht.transform_weights(W_f)
    H_tilde = rht.transform_hessian(H_f)

    ev_rht_e.record()

    # --- Pad ---
    torch.cuda.synchronize()
    ev_pad_s = torch.cuda.Event(enable_timing=True)
    ev_pad_e = torch.cuda.Event(enable_timing=True)
    ev_pad_s.record()

    n_tilde = W_tilde.shape[1]
    W_pad, _, _ = pad_to_multiple(W_tilde, 8)
    H_pad = pad_hessian(H_tilde, 8)

    ev_pad_e.record()
    torch.cuda.synchronize()

    rht_ms = ev_rht_s.elapsed_time(ev_rht_e)
    pad_ms = ev_pad_s.elapsed_time(ev_pad_e)

    # --- LDLQ (instrumented) ---
    if bpw == 2:
        ldlq_result = profile_ldlq_1stage(W_pad, H_pad, codebook)
    elif bpw == 3:
        codebook_small = codebook.make_small(256)
        ldlq_result = profile_ldlq_2stage(
            W_pad, H_pad, codebook, codebook_small,
            resid_scale=codebook.resid_scale)
    else:  # 4bpw
        ldlq_result = profile_ldlq_2stage(
            W_pad, H_pad, codebook, codebook,
            resid_scale=codebook.resid_scale)

    # --- RHT Inverse ---
    torch.cuda.synchronize()
    ev_inv_s = torch.cuda.Event(enable_timing=True)
    ev_inv_e = torch.cuda.Event(enable_timing=True)
    ev_inv_s.record()

    W_hat_tilde = ldlq_result['W_hat'][:, :n_tilde]
    W_hat = rht.inverse_transform_weights(W_hat_tilde)

    ev_inv_e.record()
    torch.cuda.synchronize()
    inv_ms = ev_inv_s.elapsed_time(ev_inv_e)

    # SQNR
    diff = W_f - W_hat
    sqnr = 10 * torch.log10(W_f.pow(2).sum() / diff.pow(2).sum().clamp(min=1e-20)).item()

    return {
        'rht_forward_ms': rht_ms,
        'pad_ms': pad_ms,
        'block_ldl_ms': ldlq_result['block_ldl_ms'],
        'loop_total_ms': ldlq_result['loop_total_ms'],
        'resid_sub_ms': ldlq_result['resid_sub_ms'],
        'feedback_matmul_ms': ldlq_result['feedback_matmul_ms'],
        'codebook1_ms': ldlq_result['codebook1_ms'],
        'codebook2_ms': ldlq_result['codebook2_ms'],
        'update_ms': ldlq_result['update_ms'],
        'rht_inverse_ms': inv_ms,
        'sqnr': sqnr,
        'shape': (m, n),
    }


def print_profile(name, result):
    """Print a formatted timing breakdown."""
    total = (result['rht_forward_ms'] + result['pad_ms'] +
             result['block_ldl_ms'] + result['loop_total_ms'] +
             result['rht_inverse_ms'])

    loop = result['loop_total_ms']
    print(f"\n{'='*65}")
    print(f"  {name}  shape={result['shape']}  SQNR={result['sqnr']:.1f} dB")
    print(f"{'='*65}")
    print(f"  {'Phase':<25s} {'Time (ms)':>10s} {'% Total':>8s} {'% Loop':>8s}")
    print(f"  {'-'*51}")

    rows = [
        ('RHT forward',       result['rht_forward_ms'],   True,  False),
        ('Padding',            result['pad_ms'],           True,  False),
        ('Block LDL',          result['block_ldl_ms'],     True,  False),
        ('LDLQ loop total',    result['loop_total_ms'],    True,  False),
        ('  residual sub',     result['resid_sub_ms'],     False, True),
        ('  feedback matmul',  result['feedback_matmul_ms'], False, True),
        ('  codebook1 NN',     result['codebook1_ms'],     False, True),
        ('  codebook2 NN',     result['codebook2_ms'],     False, True),
        ('  update hatWr',     result['update_ms'],        False, True),
        ('RHT inverse',        result['rht_inverse_ms'],   True,  False),
    ]

    for label, ms, show_total_pct, show_loop_pct in rows:
        pct_total = f"{ms/total*100:6.1f}%" if show_total_pct and total > 0 else ""
        pct_loop = f"{ms/loop*100:6.1f}%" if show_loop_pct and loop > 0 else ""
        print(f"  {label:<25s} {ms:10.1f} {pct_total:>8s} {pct_loop:>8s}")

    print(f"  {'-'*51}")
    print(f"  {'TOTAL':<25s} {total:10.1f}")
    # Also show loop sub-phase sum vs loop total
    sub_sum = (result['resid_sub_ms'] + result['feedback_matmul_ms'] +
               result['codebook1_ms'] + result['codebook2_ms'] +
               result['update_ms'])
    overhead = loop - sub_sum
    if overhead > 0:
        print(f"  {'  loop overhead':<25s} {overhead:10.1f} {'':>8s} {overhead/loop*100:6.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Profile GLQ quantization phases")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M",
                        help="HuggingFace model ID")
    parser.add_argument("--bpw", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--sublayer", default="all",
                        help="Which sublayer to profile: all, gate_proj, q_proj, etc.")
    parser.add_argument("--nsamples", type=int, default=4,
                        help="Calibration samples (fewer = faster profiling)")
    parser.add_argument("--layer-idx", type=int, default=0,
                        help="Which decoder layer to profile")
    args = parser.parse_args()

    device = "cuda"
    assert torch.cuda.is_available(), "CUDA required"

    print(f"Profiling GLQ quantization: {args.model} {args.bpw}bpw")
    print(f"  Device: {torch.cuda.get_device_name()}")
    print(f"  Calibration samples: {args.nsamples}")

    # Load model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True)

    # Get decoder layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        decoder_layers = model.model.layers
    elif hasattr(model, 'layers'):
        decoder_layers = model.layers
    else:
        raise ValueError("Cannot find decoder layers")

    # Get embeddings
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed = model.model.embed_tokens
    elif hasattr(model, 'embed_tokens'):
        embed = model.embed_tokens
    else:
        raise ValueError("Cannot find embed_tokens")

    # Calibration data
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    seqlen = 2048
    calibration = []
    for i in range(args.nsamples):
        start = i * seqlen
        calibration.append(tokens[start:start + seqlen].unsqueeze(0))

    # Embed calibration data
    embed.to(device)
    hidden_states = []
    with torch.no_grad():
        for seq in calibration:
            hidden_states.append(embed(seq.to(device)))
    hidden_states = torch.cat(hidden_states, dim=0)
    embed.to("cpu")

    # Move target layer to GPU and capture Hessians
    layer_idx = args.layer_idx
    layer = decoder_layers[layer_idx]
    layer.to(device)

    linears = {}
    for name, mod in layer.named_modules():
        if isinstance(mod, nn.Linear):
            linears[name] = mod

    # Install Hessian hooks
    from glq.quantize_model import HessianCapture, get_rotary_emb
    captures = {}
    for name, mod in linears.items():
        captures[name] = HessianCapture(mod)

    # Get rotary embeddings
    rotary_emb = get_rotary_emb(model)
    if rotary_emb is not None:
        rotary_emb.to(device)

    # Forward pass to capture Hessians
    with torch.no_grad():
        for i in range(hidden_states.shape[0]):
            h = hidden_states[i:i+1]
            seq_len = h.shape[1]
            cache_position = torch.arange(seq_len, device=device)
            position_ids = cache_position.unsqueeze(0)
            kwargs = dict(position_ids=position_ids, cache_position=cache_position,
                          use_cache=False)
            if rotary_emb is not None:
                kwargs['position_embeddings'] = rotary_emb(h, position_ids=position_ids)
            layer(h, **kwargs)

    # Build codebook
    codebook = E8ShellCodebook(device=device)

    # Profile each sublayer
    print(f"\nLayer {layer_idx}: {len(linears)} linear sublayers")

    for name, cap in captures.items():
        if args.sublayer != "all" and args.sublayer not in name:
            cap.finalize()
            continue

        H = cap.finalize()
        if H is None:
            continue
        mod = linears[name]
        W = mod.weight.data
        if W.shape[1] < 8:
            continue

        # Warmup run
        _ = profile_sublayer(W, H, codebook, bpw=args.bpw)
        torch.cuda.synchronize()

        # Profiling run
        result = profile_sublayer(W, H, codebook, bpw=args.bpw)
        print_profile(name, result)

        del H
        gc.collect()
        torch.cuda.empty_cache()

    # Cleanup
    layer.to("cpu")
    if rotary_emb is not None:
        rotary_emb.to("cpu")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
