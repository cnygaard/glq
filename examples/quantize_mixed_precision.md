# Two-pass mixed-precision quantization

A walk-through of GLQ's **sensitivity-allocated mixed precision** workflow —
the one that produced most of our published models (e.g.
`xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw`).

This is the recommended path when you want better quality-per-bit than
a uniform bpw setting: instead of quantizing every layer at the same
rate, the allocator gives more bits to the layers that need them and
fewer bits to the robust ones.

## Why not just pick one `--bpw`?

If you run

```bash
glq-quantize --model HuggingFaceTB/SmolLM3-3B --output m_uniform_4bpw --bpw 4
```

every one of the model's ~250 linear sublayers is quantized at the same
rate. That's simple, but it over-spends bits on the *robust* layers
(MLP output projections at the tail of the network, where errors
average out) and under-spends on the *sensitive* ones (attention K/V
projections in the middle of the network, where errors compound
across all downstream tokens).

Mixed precision fixes this by looking at **how much each layer's
quantization error actually hurts the output** and giving each layer
a bit budget accordingly. You'll typically get within ~1 % of full
bf16 quality at ~0.5 bpw lower average than the uniform setting would
need.

## The two passes, at a glance

```
                           ┌──────────────────┐
model + calibration data ─►│ pass 1: profile  │─► bpw_allocation.json
                           └──────────────────┘
                                              │
                                              ▼
                           ┌──────────────────┐
  same model + calib data ►│ pass 2: quantize │─► GLQ checkpoint
                           └──────────────────┘
```

**Pass 1** quantizes every layer at the smallest allowed bpw, measures
the per-layer *proxy loss* (trace of `H · (W - W_hat)ᵀ · (W - W_hat)`),
then runs a greedy marginal-gain allocator: whichever layer's loss
drops the most per extra bit gets promoted next, until the bit budget
is spent. The output is a tiny JSON file mapping each layer name to
its allocated bpw.

**Pass 2** re-quantizes every layer, this time at the per-layer bpw
from `bpw_allocation.json`. Same calibration Hessians, same E8 codebook
— just different precision per layer.

## Concrete walk-through: SmolLM3-3B at 4.5 bpw average

We'll target `4.5 bpw average`, bounded below by 4 and above by 5.
(Only GLQ's N-stage RVQ sizes are legal per-layer: 2, 3, 4, 5, 6, 7, 8 —
the allocator will pick from whichever of those fall in `[min-bpw,
max-bpw]`.)

### Pass 1 — profile

```bash
glq-quantize \
    --model HuggingFaceTB/SmolLM3-3B \
    --output /tmp/smollm3_4_5bpw_profile \
    --bpw 4.5 --min-bpw 4 --max-bpw 5 \
    --nsamples 128 --device cuda
```

This runs the sensitivity profiler. A few things happen inside:

1. **Calibration.** The quantizer grabs 128 sequences of length 2048
   from WikiText-2 train and runs them through the model on-the-fly,
   capturing the Hessian `H = Xᵀ X / n` per layer.
2. **Cheap quant at the floor.** Every layer is quantized at
   `--min-bpw 4` to establish the baseline proxy loss
   `tr(H · (W - W_hat)ᵀ · (W - W_hat))`.
3. **Greedy allocator.** For each candidate promotion (e.g.
   `4 → 5` for a single layer), the allocator predicts the loss drop
   per extra bit. The layer with the highest marginal gain gets
   promoted. Repeat until the total bpw budget matches `--bpw 4.5`.
4. **Write `bpw_allocation.json`** into the output directory.

The script exits when done — no quantized model is saved yet, just
the allocation. The output dir will contain a
`bpw_allocation.json` that looks roughly like:

```json
{
  "model.layers.0.self_attn.q_proj": 4,
  "model.layers.0.self_attn.k_proj": 5,
  "model.layers.0.self_attn.v_proj": 5,
  "model.layers.0.self_attn.o_proj": 4,
  "model.layers.0.mlp.gate_proj": 4,
  "model.layers.0.mlp.up_proj": 4,
  "model.layers.0.mlp.down_proj": 5,
  "...": "..."
}
```

A typical pattern: **attention K/V and MLP down_proj in the middle
of the network get promoted**, while MLP gate/up at the head and tail
stay at the floor.

### Copy the allocation somewhere safe

The profile directory is large (weights + Hessians). You only need the
tiny allocation JSON for pass 2:

```bash
mkdir -p /opt/dlami/nvme/allocs
cp /tmp/smollm3_4_5bpw_profile/bpw_allocation.json \
   /opt/dlami/nvme/allocs/smollm3_4_5bpw_alloc.json
rm -rf /tmp/smollm3_4_5bpw_profile   # optional, frees ~6 GB
```

### Pass 2 — quantize with the allocation

```bash
glq-quantize \
    --model HuggingFaceTB/SmolLM3-3B \
    --output ./SmolLM3-3B-GLQ-4.5bpw \
    --bpw-map /opt/dlami/nvme/allocs/smollm3_4_5bpw_alloc.json \
    --nsamples 128 --device cuda
```

With `--bpw-map` set, `--bpw` / `--min-bpw` / `--max-bpw` are ignored
— the per-layer bpw comes straight from the JSON. The second pass
uses the same calibration data and the same LDLQ algorithm as a
uniform run; only the bits-per-layer changes.

Output is a standard GLQ checkpoint: `model.safetensors`,
`config.json`, tokenizer files. It loads through
`AutoModelForCausalLM.from_pretrained(...)` after
`import glq.hf_integration`, same as any other GLQ model.

## Sanity-checking the allocation

Inside the driver, or in an ad-hoc REPL:

```python
import json, collections
alloc = json.load(open("alloc.json"))
counts = collections.Counter(alloc.values())
avg = sum(int(b) * n for b, n in counts.items()) / sum(counts.values())
print(f"{dict(sorted(counts.items()))}  avg={avg:.2f} bpw "
      f"over {sum(counts.values())} layers")
```

Typical output for SmolLM3-3B at target 4.5:

```
{'4': 130, '5': 120}  avg=4.48 bpw over 250 layers
```

`avg` won't always land exactly on `--bpw`: the allocator stops when
promoting the next-best layer would exceed the budget, so the realised
average is always ≤ the target.

## Picking `--min-bpw` / `--max-bpw`

Rule of thumb:

| Goal | min | max | avg | Notes |
|---|---|---|---|---|
| Maximum compression at the edge of coherence | 2 | 4 | 2.5–3 | 2bpw floor can fail on math/code; check quality. |
| Balanced storage / quality | 3 | 4 | 3.5 | The `-3.5bpw` family of published models. |
| Small loss budget, wide range | 4 | 6 | 4.5–5 | Good for instruction-tuned models where no individual layer can be heavily compressed. |
| Near-bf16 fidelity at ½ storage | 5 | 8 | 6 | Published `SmolLM3-3B-GLQ-6bpw` uses this style. |

**Wider range = more room for the allocator** to trade bits. `--min-bpw 3 --max-bpw 6` at target 4.5 often beats `--min-bpw 4 --max-bpw 5` at the same target, because the extra dynamic range lets the allocator compress robust layers harder and spend the saved bits on sensitive ones. Downside: the spread makes inference-time kernel selection more variable across layers.

## What it actually costs

Numbers below are from a Blackwell RTX PRO 6000, SmolLM3-3B, 128 calibration samples:

| Pass | Wall time |
|---|---|
| Pass 1 (profile @ 4.5bpw min=4 max=5) | ~7 min |
| Pass 2 (quantize with allocation)     | ~7 min |

So the full two-pass flow is ~15 min on a 3B model. For a 30B model, multiply by ~10 and use `--streaming` to keep memory bounded (see `quantize_model.py`'s example 3).

## Under the hood

- **Proxy loss**: `glq/sensitivity.py::layer_proxy_loss` returns `tr(H · Δ · Δᵀ)` where `Δ = W - dequant(Q(W))`. This is the second-order Taylor term of the output loss w.r.t. the weight perturbation.
- **Greedy allocator**: `glq/sensitivity.py::allocate_bpws_greedy` — at each step picks `argmax_layer (loss_at_current_bpw - loss_at_next_bpw) / bits_per_layer`, promotes that layer by one N-stage step, stops when the bit budget is hit.
- **Supported per-layer bpws**: `{2, 3, 4, 5, 6, 7, 8}`. 3 and 4 are 2-stage RVQ; 5 and 6 are 3-stage; 7 and 8 are 4-stage. The allocator skips non-supported values, so `--min-bpw 2 --max-bpw 4` has a candidate set of `{2, 3, 4}`.
- **Where the `--bpw-map` path takes over**: `glq/quantize_model.py::quantize` detects a non-None `bpw_map` argument and routes each layer through `quantize_ldlq_codebook_nstage` with its assigned stage count. No allocation is recomputed.

## Common mistakes

1. **Using different calibration data between pass 1 and pass 2.** The proxy losses from pass 1 implicitly assume the same Hessians at quantization time. Switching datasets or `--nsamples` between passes can flip which layers the allocator thought were sensitive. Stick with one setup.
2. **Moving the profile output before copying the JSON.** The whole point of the profile pass is the allocation; if you delete the directory before saving the JSON, you have to redo pass 1. Copy the JSON first, then clean up.
3. **Expecting `avg == --bpw` exactly.** The allocator is greedy and discretised; the realised average is usually 0.01–0.05 bpw below target.
4. **Picking `--bpw` outside `[min-bpw, max-bpw]`.** If `--bpw 4.5` but `--min-bpw 5 --max-bpw 6`, the minimum achievable average is 5, so the allocator will end up at 5 bpw regardless of the 4.5 target.

## Related

- [`quantize_model.py`](quantize_model.py) — the underlying CLI entry point (both passes run through it)
- [`inference_hf.py`](inference_hf.py) — load and run the produced checkpoint
- Top-level `README.md` Quickstart → *Mixed-precision quantization* section for a more concise variant of this walkthrough
