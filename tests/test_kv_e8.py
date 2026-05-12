"""Phase 1 tests for the relaxed-E8 (D̃8) codebook + MSE microbench.

Goal: verify the new codebook and compare quantization MSE against strict E8
and a KIVI-style INT8 baseline on synthetic sub-Gaussian KV-shaped data. The
relaxed lattice should reduce MSE by ~22% over strict according to the math
in ``infra/kv-cache.md``.

Pytest skips heavy tests automatically if the relaxed codebook can't be
constructed (e.g., enumeration radius too small).
"""

import math

import pytest
import torch

from glq.codebook import E8ShellCodebook
from glq.codebook_relaxed import E8RelaxedCodebook, enumerate_dtilde8


# --------------------------------------------------------------------------- #
# Codebook construction
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def strict():
    """Strict E8 codebook (shared across tests in this module)."""
    return E8ShellCodebook(device="cpu", verbose=False)


@pytest.fixture(scope="module")
def relaxed():
    """Relaxed E8 (D̃8) codebook."""
    return E8RelaxedCodebook(device="cpu", verbose=False)


class TestRelaxedCodebookConstruction:
    def test_strict_count(self, strict):
        assert strict.codebook.shape == (65536, 8)

    def test_relaxed_count(self, relaxed):
        assert relaxed.codebook.shape == (65536, 8)

    def test_strict_even_sum_parity(self, strict):
        """Every strict E8 point has 2·sum(coords) ∈ Z (even-sum / D8+).

        E8 = D8 ∪ (D8 + (½)^8) with even number of half-integers.
        Sum of coords is always an integer (since each coord is in ½ Z and
        the half-integer coset has even count of halves).
        """
        sums = 2 * strict.codebook.sum(dim=-1)
        # 2*sum should be integer; check distance to nearest integer.
        residual = (sums - sums.round()).abs().max().item()
        assert residual < 1e-5

    def test_relaxed_contains_odd_parity_points(self, relaxed):
        """Relaxed lattice should include half-integer points with odd
        number of negatives, which strict E8 forbids.

        Concrete witness: (½, -½, ½, ½, ½, ½, ½, ½) has odd # of minus
        signs in the half-integer coset; in strict E8 this would be
        filtered out by the even-sum parity rule.
        """
        # Find a half-integer point with odd negative count in the codebook.
        # A point is half-integer if any coord ends in .5.
        cb = relaxed.codebook.double()
        is_half = (cb - cb.round()).abs().max(dim=-1).values > 0.1
        half_pts = cb[is_half]
        # Among half-integer pts at norm²=2 (the 256 ±½ patterns), count
        # how many have odd number of negative signs.
        norms = (half_pts ** 2).sum(-1)
        shell1 = half_pts[norms < 2.5]
        assert shell1.shape[0] > 0, "no half-integer shell-1 points found"
        # All shell-1 half points have all coords = ±½
        signs = (shell1 < 0).sum(dim=-1)
        odd_count = (signs % 2 == 1).sum().item()
        # Strict E8 has 128 even-parity points at norm²=2 from the
        # half-integer coset. Relaxed includes the other 128 odd-parity ones.
        assert odd_count > 0, (
            "relaxed codebook contains zero odd-parity half-integer "
            "shell-1 points — it's not actually relaxed")

    def test_first_shell_size(self, strict, relaxed):
        """Shell-1 (||v||²=2) of strict E8 = 240 kissing-number points.

        D̃8 shell-1 = 112 (±e_i ± e_j) + 256 (±½)^8 = 368 points.
        """
        s_norms = strict.codebook_norms.double()
        s_shell1 = (s_norms - 2.0).abs() < 1e-4
        assert s_shell1.sum().item() == 240

        r_norms = relaxed.codebook_norms.double()
        r_shell1 = (r_norms - 2.0).abs() < 1e-4
        assert r_shell1.sum().item() == 368

    def test_origin_present(self, strict, relaxed):
        """Both lattices include the origin (the all-zeros codeword)."""
        for cb in (strict.codebook, relaxed.codebook):
            zero_dist = cb.norm(dim=-1).min().item()
            assert zero_dist < 1e-6


# --------------------------------------------------------------------------- #
# MSE microbench: relaxed vs strict vs INT8 on synthetic KV-shaped data
# --------------------------------------------------------------------------- #

def _int8_absmax_quant(x: torch.Tensor) -> torch.Tensor:
    """Per-row INT8 absmax quantization (KIVI-style group-of-8)."""
    s = x.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-8) / 127.0
    q = (x / s).round().clamp(-127, 127)
    return q * s


def _gauss_kv_like(n: int, seed: int = 0) -> torch.Tensor:
    """Standard Gaussian in 8D, post-Hadamard.

    Hadamard of N(0, I_8) is N(0, I_8) (rotation invariant), so we return
    Gaussian samples directly. Heavy tails relative to real KV cache.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 8, generator=g)


def _hadamard8() -> torch.Tensor:
    """8×8 normalized Walsh-Hadamard matrix (H @ H^T = I)."""
    H1 = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    H2 = torch.kron(H1, H1)        # 4x4
    H3 = torch.kron(H2, H1)        # 8x8
    return H3 / math.sqrt(8.0)


def _subgauss_kv_like(n: int, seed: int = 0) -> torch.Tensor:
    """KV-cache-like sub-Gaussian data: Hadamard-rotated, per-row
    NSN-normalized, with outlier channels (the standard KV pattern).

    Construction:
      1. Draw a heavy-tailed pre-Hadamard signal (Gaussian + sparse spikes
         to mimic attention sink / outlier channels).
      2. Per-row NSN: scale so max|x| = 1.
      3. Apply 8×8 Hadamard (in-place rotation).

    The result has bounded support per coordinate (≤ 1/√8 ≈ 0.354 of the
    pre-scale max), light tails, and mass concentrated near origin.
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n, 8, generator=g)
    # Add outlier spikes on ~5% of rows, mimicking attention-sink columns.
    n_spike = int(n * 0.05)
    spike_rows = torch.randperm(n, generator=g)[:n_spike]
    spike_cols = torch.randint(0, 8, (n_spike,), generator=g)
    base[spike_rows, spike_cols] += torch.randn(n_spike, generator=g) * 5.0
    # Per-row NSN: divide by max|x|.
    nsn = base.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-8)
    base = base / nsn
    # Hadamard rotate.
    H = _hadamard8()
    return base @ H


def _quantize_codebook(x: torch.Tensor, cb) -> torch.Tensor:
    """Single-stage codebook NN quantize, returning the decoded vectors."""
    dec, _ = cb.quantize(x)
    return dec


def _quantize_codebook_rvq(x: torch.Tensor, cb) -> torch.Tensor:
    """Two-stage RVQ (4bpw effective)."""
    dec, _ = cb.quantize_rvq(x)
    return dec


def _mse(x_ref: torch.Tensor, x_q: torch.Tensor) -> float:
    return ((x_ref - x_q) ** 2).sum(-1).mean().item()


def _mse_table_for_dist(label: str, x: torch.Tensor, strict, relaxed):
    """Compute INT8 / strict-E8 / relaxed-E8 (+ RVQ) MSE on a fixed sample."""
    int8 = _int8_absmax_quant(x)
    s = strict.opt_scale
    r = relaxed.opt_scale
    strict_q = _quantize_codebook(x / s, strict) * s
    strict_rvq = _quantize_codebook_rvq(x / s, strict) * s
    relax_q = _quantize_codebook(x / r, relaxed) * r
    relax_rvq = _quantize_codebook_rvq(x / r, relaxed) * r

    rows = [
        ("INT8 absmax", 8, _mse(x, int8)),
        ("E8 strict (2bpw)", 2, _mse(x, strict_q)),
        ("E8 relaxed (2bpw)", 2, _mse(x, relax_q)),
        ("E8 strict RVQ (4bpw)", 4, _mse(x, strict_rvq)),
        ("E8 relaxed RVQ (4bpw)", 4, _mse(x, relax_rvq)),
    ]
    base = rows[0][2]  # INT8 baseline
    print(f"\n  ===== {label} ({x.shape[0]} samples) =====")
    print(f"  {'Quantizer':<24} {'bpw':>4}  {'MSE':>10}  {'norm. (INT8=1)':>15}")
    for name, bpw, mse in rows:
        print(f"  {name:<24} {bpw:>4}  {mse:>10.5f}  {mse/base:>15.4f}")
    # Return relaxed/strict ratios for assertion convenience.
    return {
        "single_ratio": rows[2][2] / rows[1][2],
        "rvq_ratio": rows[4][2] / rows[3][2],
        "mse_strict_single": rows[1][2],
        "mse_relax_single": rows[2][2],
    }


def test_mse_gaussian(strict, relaxed):
    """Gaussian data — strict E8 expected to win (heavier tails than
    Hadamard-rotated KV; outer codebook shells matter).

    No pass/fail; diagnostic only. Confirms expectation that strict is
    the right choice for unbounded-tail distributions.
    """
    x = _gauss_kv_like(n=16384)
    r = _mse_table_for_dist("Gaussian N(0, I_8)", x, strict, relaxed)
    print(f"  ratio (relaxed / strict, single): {r['single_ratio']:.3f}")
    print(f"  ratio (relaxed / strict, RVQ)   : {r['rvq_ratio']:.3f}")


def test_mse_subgaussian_kvlike(strict, relaxed):
    """Sub-Gaussian KV-like data (per-row NSN + Hadamard, with outlier
    spikes). Relaxed E8 should win here per kv-cache.md.

    Asserts MSE_relaxed / MSE_strict ≤ 0.95 (≥5% improvement) on the
    single-stage 2bpw codebook.
    """
    x = _subgauss_kv_like(n=16384)
    r = _mse_table_for_dist("Sub-Gaussian KV-like (NSN + Hadamard)",
                            x, strict, relaxed)
    print(f"  ratio (relaxed / strict, single): {r['single_ratio']:.3f}")
    print(f"  ratio (relaxed / strict, RVQ)   : {r['rvq_ratio']:.3f}")
    assert r["single_ratio"] <= 0.95, (
        f"relaxed/strict single-stage ratio = {r['single_ratio']:.3f}, "
        f"expected ≤ 0.95 on KV-like data")


# --------------------------------------------------------------------------- #
# Enumerator sanity
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# E8KVQuantizer: HF-cache-shaped tensor roundtrip
# --------------------------------------------------------------------------- #

def _kv_tensor(batch=1, n_heads=4, seq=64, head_dim=64, seed=0):
    """Synthetic KV-shaped tensor mimicking attention output: each row
    has a few outlier channels then per-row normalized."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, n_heads, seq, head_dim, generator=g)
    # Inject sparse outlier channels (attention sink pattern).
    n_spikes = max(1, int(batch * n_heads * seq * 0.02))
    sb = torch.randint(0, batch, (n_spikes,), generator=g)
    sh = torch.randint(0, n_heads, (n_spikes,), generator=g)
    ss = torch.randint(0, seq, (n_spikes,), generator=g)
    sc = torch.randint(0, head_dim, (n_spikes,), generator=g)
    x[sb, sh, ss, sc] += torch.randn(n_spikes, generator=g) * 5.0
    return x


@pytest.mark.parametrize("n_stages,bpw", [(1, 2), (2, 4)])
@pytest.mark.parametrize("codebook_kind", ["strict", "relaxed"])
def test_kv_quantizer_roundtrip(strict, relaxed, n_stages, bpw, codebook_kind):
    """E8KVQuantizer encode → decode should reconstruct KV-shaped tensors
    within a tolerance that depends on bpw."""
    from glq.kv_e8 import E8KVQuantizer
    cb = strict if codebook_kind == "strict" else relaxed
    quantizer = E8KVQuantizer(cb, n_stages=n_stages)

    x = _kv_tensor(batch=2, n_heads=4, seq=32, head_dim=64)
    qt = quantizer.quantize(x)

    # Shape + dtype invariants
    assert qt["shape"] == tuple(x.shape)
    assert qt["dtype"] == x.dtype
    assert qt["idx1"].dtype == torch.int16
    assert "idx2" in qt if n_stages == 2 else "idx2" not in qt

    rec = quantizer.dequantize(qt)
    assert rec.shape == x.shape
    assert rec.dtype == x.dtype

    # Reconstruction MSE depends on bpw budget. At 2 bpw, error is large;
    # at 4 bpw (RVQ), error should be modest.
    err = ((x - rec) ** 2).mean().item()
    var = x.var().item()
    rel = err / (var + 1e-12)
    print(f"\n  {codebook_kind} {bpw}bpw: MSE={err:.4f} relMSE={rel:.4f}")
    # Loose tolerance; the headline goal is "decoder works", quality is
    # in the separate microbench.
    if bpw == 4:
        assert rel < 0.5, f"4 bpw rel MSE = {rel:.4f}, expected < 0.5"
    else:
        assert rel < 2.0, f"2 bpw rel MSE = {rel:.4f}, expected < 2.0"


def test_kv_quantizer_relaxed_beats_strict_on_real_shape(strict, relaxed):
    """End-to-end E8KVQuantizer (FHT + scale + NN + RVQ) on a 4D KV-shaped
    tensor: relaxed should beat strict on this synthetic distribution."""
    from glq.kv_e8 import E8KVQuantizer
    x = _kv_tensor(batch=4, n_heads=8, seq=128, head_dim=64)

    def measure(cb, n_stages):
        q = E8KVQuantizer(cb, n_stages=n_stages)
        rec = q.dequantize(q.quantize(x))
        return ((x - rec) ** 2).mean().item()

    mse_strict_1 = measure(strict, 1)
    mse_relax_1 = measure(relaxed, 1)
    mse_strict_2 = measure(strict, 2)
    mse_relax_2 = measure(relaxed, 2)
    print(f"\n  KV-shaped, 4D tensor: strict 2bpw={mse_strict_1:.4f} "
          f"relaxed 2bpw={mse_relax_1:.4f} ratio={mse_relax_1/mse_strict_1:.3f}")
    print(f"  KV-shaped, 4D tensor: strict 4bpw={mse_strict_2:.4f} "
          f"relaxed 4bpw={mse_relax_2:.4f} ratio={mse_relax_2/mse_strict_2:.3f}")
    # Relaxed should win on both 2 bpw and 4 bpw on this distribution.
    assert mse_relax_1 / mse_strict_1 <= 0.95
    assert mse_relax_2 / mse_strict_2 <= 0.97


# --------------------------------------------------------------------------- #
# Enumerator sanity
# --------------------------------------------------------------------------- #

def test_enumerator_size():
    """D̃8 enumeration at norm²=2.

    Expected points:
      - origin: 1
      - ±e_i (integer, norm²=1): 16
      - ±e_i ± e_j (integer, norm²=2): 4 × C(8,2) = 112
      - (±½)^8 (half-integer, norm²=2): 256
    Total: 1 + 16 + 112 + 256 = 385.
    """
    coords, norms = enumerate_dtilde8(max_norm_sq=2.0)
    assert coords.shape[0] == 385
    assert norms.max().item() <= 2.0 + 1e-9
