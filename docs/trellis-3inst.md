# Trellis / 3INST: how a GLQ checkpoint is made and read

The trellis codebook (QTIP-style TCQ) is GLQ's flagship path — 2–8 bits/weight, end to end,
as of 0.8.0. This document traces one linear layer from bf16 weights to a served matmul.

The pipeline is five stages, and each exists because of the one before it:

```
W  ──RHT──►  W̃  ──LDLQ+Viterbi──►  trellis states  ──pack──►  int16 blob
                                                                   │
x ──input RHT──►  x̂ ─────────── fused decode+matmul ─────────────►─┘
                                          │
                                    ──output RHT──►  y = Wx
```

Every claim below is traced to a source line, and the snippets run as written.

---

## 1. RHT — make the weights look Gaussian

`glq/rht.py`

Real weight matrices have outlier channels. A fixed codebook sized for those outliers wastes
almost all its points on a range nothing occupies, so quantization error explodes. The
randomized Hadamard transform fixes this before any quantization happens.

Two ±1 sign vectors, `su` (m,) and `sv` (n,), are drawn from a generator seeded at 42
(`rht.py:80`) and stored per layer as the `SU`/`SV` artifacts. `transform_weights`
(`rht.py:102`) computes

```
W̃ = H_m · diag(su) · W · diag(sv) · H_n
```

— scale columns by `sv`, FHT along columns, scale rows by `su`, FHT along rows.
`transform_hessian` (`rht.py:124`) applies the **input-side rotation only**,
`H_n diag(sv) H diag(sv) H_n`: the Hessian lives on the input axis, so there is no `su` in it.

Each entry of `W̃` is a sum of many ±-weighted originals, so by CLT it is approximately
Gaussian and outliers are smeared across the whole matrix:

```python
import torch; from glq.rht import RHT
torch.manual_seed(0)
W = torch.randn(256, 512); W[:, 7] *= 30.0          # one outlier channel
Wt = RHT(256, 512, device="cpu").transform_weights(W)
# peak/rms:   W 51.4  ->  W_rht 4.2
# kurtosis:   W 599.9 ->  W_rht 2.93   (3.0 == Gaussian)
```

That is the whole point: after the RHT, one fixed codebook fits every layer of every model.

### Block-diagonal: why trellis never pads

`_block_decompose` splits a dimension into a sum of powers of two and FHTs each block
independently, so no padding is needed:

```python
>>> from glq.hadamard import _block_decompose
>>> _block_decompose(2688)
[2048, 512, 128]
```

`m_pad == m` and `n_pad == n`, asserted at `trellis.py:874`. This is a real footprint
difference, not a detail — the e8p codebook pads to the next power of two instead, which
inflates gemma-4-31B by 2.1×.

## 2. Input RHT and output RHT

The rotation is **never undone in storage**. Weights stay rotated, and inference brackets the
matmul instead (`rht.py:220–229`):

```python
transform_input(x)          →  x̂ = H_n (sv ⊙ x)
inverse_transform_output(y) →  su ⊙ H_m(y)
```

The algebra closes because `H·H = I` and `diag(s)² = I`:

```
su ⊙ H_m(W̃ @ x̂) = diag(su)H_m · H_m diag(su) W diag(sv)H_n · H_n diag(sv) x = W x
```

Verifiable in four lines:

```python
rht = RHT(64, 128, device="cpu")
y = rht.inverse_transform_output(rht.transform_weights(W) @ rht.transform_input(x))
# max|y - W@x| ~ 1e-5     (fp32 rounding only; the transform itself is exact)
```

So input RHT and output RHT are two halves of a rotation that cancels exactly. **The only
lossy step in the entire pipeline is quantizing `W̃` in the middle.** Both are O(n log n) and
get fused into the matmul kernel rather than launched separately.

## 3. The 3INST codebook — a trellis, not a table

`glq/trellis.py:122`, `glq_trellis.cu:138`

A 16-bit state decodes to a weight by arithmetic, with no lookup table at all:

```python
h = s*89226354 + 64248484          # uint32, wrapping
r = (h & 0x8FFF8FFF) ^ 0x3B603B60
w = fp16(r >> 16) + fp16(r & 0xFFFF)    # bit-cast both halves, add IN fp16
```

The CUDA side is the same four lines. `__ushort_as_half` (bit-cast, not convert) and `__hadd`
(fp16 add, not fp32) are both load-bearing for bit-exactness against the Python oracle — see
the comment at `glq_trellis.cu:134`.

The **trellis** part: `L=16` is a shift register and each step consumes `K` bits (`V=1` for
3INST), so consecutive weights share `L−K` bits of state. Weights are not chosen
independently — the encoder picks a *path*, and that constraint is what buys coding gain over
scalar quantization at the same rate. `K` bits per weight *is* the bit rate: K=2/3/4 → 2/3/4
bpw. `bitshift_codebook.__init__` refuses `K*V >= L` (`trellis.py:174`), where the trellis
degenerates and the bit-packing silently slices to empty.

| | 3INST | HYB |
|---|---|---|
| V | 1 | 2 |
| decode | arithmetic, lookup-free | 9-bit kmeans `tlut`, stored in the checkpoint |
| shared memory | **zero** | tlut resident |
| 5–8 bpw | supported | refused — no 2-stage entry |

3INST's advantage is not accuracy — measured 22.06 vs 21.1 dB SQNR, a tie. It is that no
table means no shared-memory codebook gather, which was ~35% of matvec stalls under ncu.

## 4. Quantization: LDLQ + Viterbi

`trellis_ldlq`, `trellis.py:649`

```python
L, _ = block_LDL(H + damp·I, block_size=16)      # damp = 1% of mean(diag(H))
Wscale = rms(W) * cb.opt_scale                    # opt_scale = 1/(rms(lut)*0.9) ≈ 0.8934
for k in reversed(range(n//16)):                  # reverse sweep, 16 columns at a time
    feedback = R[:, ke:] @ L[ke:, kb:ke]          # error already made, projected via H
    tiles = (Wr[:, kb:ke] + feedback).reshape(m//16, 256)[:, _PERMUTE]
    hatX, state = cb.quantize_tiles(tiles)        # ← Viterbi over the whole tile
    R[:, kb:ke] = Wr[:, kb:ke] - hatWr[:, kb:ke]
```

Two ideas stacked:

* **LDLQ** is Hessian error feedback. Rounding error from one block is pushed into the
  not-yet-quantized blocks, weighted by the LDL factor, so error lands where the model is
  least sensitive. GLQ's `block_LDL` is QTIP's BlockLDLQ — feedback reads only rows below the
  block, which makes QTIP's diagonal-zeroing a no-op.
* **Viterbi** finds the minimum-distortion trellis path for a whole 16×16 tile: 256 weights as
  one tail-biting cycle, not 256 independent roundings. `_PERMUTE` reorders into MMA-fragment
  order first, so the stored bits already match what the kernel wants to read.

The ACS inner loop (`bitshift_codebook.update`, `trellis.py:231`) is written as a strided
view-min rather than expand+gather — same candidate ordering, hence the same tie-breaks and
bit-identical output, but coalesced and with no index traffic. `glq/trellis_step_kernel.py`
fuses it in Triton for a 1.6–1.9× quantization speedup with byte-identical checkpoints.

### 5–8 bpw is stacked RVQ, not a bigger K

```python
>>> {b: trellis_rvq_recipe(b) for b in range(2, 9)}
{2: [2], 3: [3], 4: [4], 5: [4, 1], 6: [4, 2], 7: [4, 3], 8: [4, 4]}
```

Stage 1 quantizes at K=4; stage 2 quantizes the **residual** at K=bpw−4 with a per-layer
fitted scale (`trellis.py:694`). Native K=5..8 was measured worse — the coding gain decays as
K grows because the L=16 window collapses. The presence of `trellis_packed2` *is* the stage
count; there is no config key for it.

## 5. Storage

`pack_layer`, `trellis.py:608`

```
(m, n//V) states  →  [(m//16)*(n//16), ceil(256·K/16)] int16
```

Axis 0 is a **flattened (row-block, col-block) index, row-block-major**. Per 16×16 tile:
256 weights × K bits, so at K=4 that's 64 int16 = exactly 4.00 bits/weight. This is why
published footprints back-solve to exact integer rates.

Then `kernel_tile_flip` (`trellis.py:587`) permutes bytes into MMA-fragment order:

```python
(m//16//2, 2, n//16//2, 2, 32, K) → permute(0, 2, 4, 3, 1, 5)
```

> **This makes a PAIR of 16-row blocks one self-contained unit.** Any row-wise slicing of a
> packed buffer — e.g. splitting a jointly-quantized `[gate; up]` MoE expert — is therefore
> valid only on **32-row** boundaries. `split_trellis_packed` (`quantize_model.py`) asserts
> this; cutting inside a pair interleaves the two halves' bytes and produces a buffer that
> still loads and still decodes, to the wrong weights.

### Artifact taxonomy

Anything that slices or reassembles a layer needs this table:

| artifact | scope | on split |
|---|---|---|
| `trellis_packed`, `trellis_packed2` | block-tiled | block-aware split, 32-row aligned |
| `SU` | per row (m,) | plain row split |
| `SV`, `Wscale`, `inv_resid_scale2`, `tlut` | whole layer | duplicated |

## 6. Dequantization

**Reference oracle** — `decode_layer` (`trellis.py:629`): un-flip bytes → unpack the
tail-biting bit stream to 256 states per tile → `recons` → undo `_PERMUTE` → re-tile to
(m, n). Stacked is `decode_layer_nstage`: `Σ_s decode(stage_s) · cum_inv_rs[s]`.

The encode → pack → decode round trip is bit-exact:

```python
hat, Qidxs, Wscale = trellis_ldlq(W, H, cb)
packed = pack_layer(cb, Qidxs, m, n, has_kernel=True)
torch.equal(decode_layer(cb, packed, m, n, has_kernel=True), hat)   # True
```

**Production** never materialises `W`. `_trellis_linear_apply`
(`quantized_linear.py:838`) calls one fused CUDA op doing
`input-RHT → decode+matmul → ×Wscale → output-RHT`, dispatching on:

| condition | entry |
|---|---|
| `tlut` non-empty | `fused_linear_trellis` (HYB) |
| stage 2 present | `fused_linear_trellis_3inst_rvq2` |
| otherwise | `fused_linear_trellis_3inst` |
| `tlut` **and** stage 2 | hard `NotImplementedError` |

Weights stay compressed in VRAM; each 16-bit state is decoded in-register at the point of use.
That last row matters: taking the 1-stage tlut op for a stacked layer would decode stage 1
only and return plausible-but-wrong output, so the refusal is deliberate and lives at the one
choke point shared by HF and vLLM.

Because matmul is linear, the kernel may decode each stage and sum the **outputs** rather than
summing decoded weights — which is why `decode_layer_nstage` documents itself as the oracle
for either implementation.

## Known limits

* **MoE + trellis has no vLLM path.** `glq_vllm/fused_moe_method.py:107` refuses it at any
  bpw. Such checkpoints load under HF transformers only.
* **HYB cannot do 5–8 bpw.** The fused RVQ entries take no tlut.
* Trellis requires uniform integer bpw 2–8; mixed/fractional is refused at
  `quantize_model.py:828`.

## Where to look next

| topic | file |
|---|---|
| incoherence processing | `glq/rht.py`, `glq/hadamard.py` |
| codebook, encode, pack/decode | `glq/trellis.py` |
| Triton ACS step | `glq/trellis_step_kernel.py` |
| CUDA decode + fused RHT | `glq/csrc/glq_trellis.cu` (read the header comment first) |
| HF inference | `glq/quantized_linear.py` |
| serving | `glq_vllm/linear_method.py` |
| bit-exactness gates | `tests/test_trellis_3inst_kernel.py`, `tests/test_trellis_gate_up_split.py` |
