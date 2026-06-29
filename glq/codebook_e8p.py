"""E8P (QuIP# padded-D̂8) codebook for GLQ's ``--codebook e8p`` path.

Ported from the validated prototype (benchmarks/_e8p_codebook_min.py = QuIP# latticee8_padded12) + glq's
codebook interface. These are SINGLE-STAGE codebooks; glq's N-stage LDLQ (ldlq.quantize_ldlq_codebook_nstage)
does the RVQ stacking with explicit ``resid_scales`` (2bpw=[E8P]; 3bpw=[E8P,E81B] rs 2.04; 4bpw=[E8P,E8P] rs 3.45).

Scale convention is glq's: the LDLQ uses ``Wscale = rms·opt_scale`` and feeds ``Wr = W/Wscale`` to
``quantize_fast``. The QuIP# E8P codebook is calibrated for input ≈ unit·1.03, so ``opt_scale = 1/1.03`` makes
``Wr ≈ (W/rms)·1.03`` — matching the prototype's validated input. Decode at inference uses the TC kernels in
glq/csrc/glq_e8p.cu (grid_packed_abs int32[256] for E8P; e81b_grid fp16 256×8 for E81B).
"""
import torch

_E8P_CODESZ = 8


# ----------------------------------------------------------------- grid construction (QuIP# verbatim)
def get_norm12():
    return torch.tensor([
        [3, 1, 1, 1, 3, 3, 3, 3], [1, 3, 1, 1, 3, 3, 3, 3], [1, 1, 3, 1, 3, 3, 3, 3],
        [1, 1, 1, 3, 3, 3, 3, 3], [3, 3, 3, 1, 3, 3, 1, 1], [3, 3, 3, 1, 3, 1, 3, 1],
        [3, 3, 3, 1, 1, 3, 3, 1], [3, 3, 3, 1, 3, 1, 1, 3], [3, 3, 3, 1, 1, 3, 1, 3],
        [3, 3, 3, 1, 1, 1, 3, 3], [3, 3, 1, 3, 3, 3, 1, 1], [3, 3, 1, 3, 3, 1, 3, 1],
        [3, 3, 1, 3, 1, 3, 3, 1], [3, 3, 1, 3, 3, 1, 1, 3], [3, 3, 1, 3, 1, 3, 1, 3],
        [3, 3, 1, 3, 1, 1, 3, 3], [3, 1, 3, 3, 3, 3, 1, 1], [3, 1, 3, 3, 3, 1, 3, 1],
        [3, 1, 3, 3, 1, 3, 3, 1], [3, 1, 3, 3, 3, 1, 1, 3], [3, 1, 3, 3, 1, 3, 1, 3],
        [1, 3, 3, 3, 1, 1, 3, 3], [1, 3, 3, 3, 3, 3, 1, 1], [1, 3, 3, 3, 3, 1, 3, 1],
        [1, 3, 3, 3, 1, 3, 3, 1], [1, 3, 3, 3, 3, 1, 1, 3], [1, 3, 3, 3, 1, 3, 1, 3],
        [1, 1, 3, 3, 1, 3, 3, 3], [3, 3, 1, 1, 3, 3, 3, 1],
    ]) / 2


def get_packed_abs_grid():
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * 8).float() + 1 / 2
    d8m2 = (d8.sum(dim=-1) % 2 == 0)
    d8n = d8.norm(dim=-1) ** 2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)
    cba = torch.concat([d8abs, get_norm12()], dim=0)
    cba = cba[:, [0, 2, 4, 6, 1, 3, 5, 7]]
    cba[:, 7] *= (1 - 2 * (cba.sum(1) % 2))
    cba = (cba * 2 + 8).to(torch.int32)
    acc = cba[:, 0]
    for i in range(7):
        acc = acc | (cba[:, (i + 1)] << ((i + 1) * 4))
    return acc


def get_abs_grid():
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * _E8P_CODESZ).float() + 1 / 2
    d8m2 = (d8.sum(dim=-1) % 2 == 0)
    d8n = d8.norm(dim=-1) ** 2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)
    return torch.concat([d8abs, get_norm12()], dim=0)


def get_full_grid(packed_abs_grid):
    # Explicit fp32 — built at module import, which may run under a non-fp32 default
    # dtype (e.g. vLLM sets float16 during model init); the grid must stay fp32.
    synth_codebook = torch.zeros(1 << 16, 8, dtype=torch.float32)
    parity_idx = []
    shuffle_map = [0, 4, 1, 5, 2, 6, 3, 7]
    for c in range(1 << 16):
        signs = c & 255
        abs = c >> 8
        parity = 0
        for i in range(8):
            parity = parity ^ ((signs >> i) & 1)
        signs = signs ^ parity
        abs_code = packed_abs_grid[abs].item()
        for i in range(8):
            ii = shuffle_map[i]
            synth_codebook[c, i] = (((abs_code >> (4 * ii)) & 15) - 8) * 0.5
            if ((signs >> ii) & 1):
                synth_codebook[c, i] *= -1
        if parity:
            synth_codebook[c, :] -= 0.25
            parity_idx.append(c)
        else:
            synth_codebook[c, :] += 0.25
    return synth_codebook, parity_idx


def get_e81bgrid():
    """E81B residual codebook (norm²≤2 even-sum E8 + 15 norm-4 padding axis vectors) — 255 entries."""
    intr = torch.arange(-4, 4)
    hintr = intr + 1 / 2
    ge8 = torch.concat([torch.cartesian_prod(*[intr] * 8), torch.cartesian_prod(*[hintr] * 8)], dim=0)
    e8 = ge8[torch.where((ge8.sum(dim=-1) % 2 == 0) * (ge8.norm(dim=-1) ** 2 <= 2))[0]]
    norm4 = torch.eye(8) * 2
    norm4 = torch.concat([norm4, -norm4[:7]], dim=0)
    return torch.concat([e8, norm4], dim=0)


_E8P_PACKED_ABS = get_packed_abs_grid()
_E8P_GRID, _PARITY_IDX = get_full_grid(_E8P_PACKED_ABS)


# ----------------------------------------------------------------- E8P codebook (2-bit stage)
class E8PCodebook:
    DIM = 8
    is_e8p = True                                                 # marker for quantize_model / hf_integration dispatch

    def __init__(self, device='cpu', verbose=True, opt_scale=1.0 / 1.03):
        self.device = device
        self.codesz = _E8P_CODESZ
        self.opt_scale = opt_scale
        self.grid_packed_abs = _E8P_PACKED_ABS.to(device)         # int32[256] for the TC kernel
        self.codebook = _E8P_GRID.to(device)                      # (65536, 8) dense (decode / opt_scale calib)
        self.codebook_half = self.codebook.half()
        self.codebook_norms = (self.codebook ** 2).sum(-1)
        # quantize machinery (parity-half subset NN), from latticee8_padded12 inference=False
        grid_part = self.codebook[_PARITY_IDX] + 0.25
        grid_part = grid_part[torch.where(((grid_part[:, :7] < 0).sum(dim=-1) <= 1) *
                                          (grid_part[:, :7].min(dim=-1).values >= -0.5))[0]]
        self.grid_part = grid_part
        self.grid_part_norm = grid_part.norm(dim=-1) ** 2
        abs_grid = get_abs_grid().to(device)
        self.grid_abs_odd = abs_grid.sum(dim=-1) % 2 == 1
        self.part_abs_map = self.round(grid_part.abs(), abs_grid, abs_grid.norm(dim=-1) ** 2)[1]
        self.bit_map = (2 ** torch.arange(8)).to(device)
        self.resid_scale = None                                   # set per-recipe by the caller
        if verbose:
            print(f"E8PCodebook: 65536-entry padded-D̂8, opt_scale={self.opt_scale:.4f}")

    def round(self, X, grid, grid_norm):
        # fp32-safe: construction may run under a non-fp32 default dtype.
        Xqidx = (2 * X.float() @ grid.float().T - grid_norm.float()).argmax(-1)
        return grid[Xqidx], Xqidx

    def _fast_quantize_part(self, X, parity):
        X_part = torch.abs(X)
        X_odd = torch.where((X < 0).sum(dim=-1) % 2 != 0)[0]
        X_part[X_odd, 7] = -X_part[X_odd, 7]
        mask = 1 - 2 * (X < 0).to(torch.float32)
        mask[X_odd, 7] = -mask[X_odd, 7]
        roundout, Xqidx = self.round(X_part, self.grid_part, self.grid_part_norm)
        vals = roundout * mask
        abs_idx = self.part_abs_map[Xqidx]
        sign_mask = (((roundout < 0) ^ (mask < 0))[:, [0, 2, 4, 6, 1, 3, 5, 7]])
        sign_mask[:, 7] = sign_mask[:, 7] ^ self.grid_abs_odd[abs_idx]
        sign_mask[:, 0] = sign_mask[:, 0] ^ parity
        mask_idx = (sign_mask * self.bit_map).sum(dim=-1).int()
        return vals, (abs_idx << 8) + mask_idx, (X - vals).norm(dim=-1)

    def quantize(self, x):
        """Structured E8P nearest-codeword -> (decoded_fp, idx int64 in [0,65535])."""
        x = x.float()
        pv, pi, pe = self._fast_quantize_part(x + 1 / 4, True)
        mv, mi, me = self._fast_quantize_part(x - 1 / 4, False)
        which = pe < me
        vals = torch.where(which.unsqueeze(-1), pv - 1 / 4, mv + 1 / 4)
        idx = torch.where(which, pi, mi).long()
        return vals, idx

    def quantize_fast(self, x_half, decoded_out=None, idx_out=None):
        dec, idx = self.quantize(x_half)
        dec = dec.to(torch.float16)
        if decoded_out is not None:
            decoded_out.copy_(dec)
        if idx_out is not None:
            idx_out.copy_(idx)
        return dec, idx

    def decode(self, indices):
        return self.codebook[indices]

    def maybe_pack_idxs(self, idxs):
        """E8P 16-bit idx (m, n/8 int64) -> packed (m, n/4) int64 (caller views to (m//16,n//64,8,4))."""
        m, n = idxs.shape
        idxs = idxs.view(m // 2, 2, (n * 8) // 16, 2).transpose(1, 2).contiguous()
        abs32 = (idxs[:, :, 0, 0] >> 8) + ((idxs[:, :, 1, 0] >> 8) << 8) + \
            ((idxs[:, :, 0, 1] >> 8) << 16) + ((idxs[:, :, 1, 1] >> 8) << 24)
        # Vectorized sign-pack (was a 4x8 Python loop, CPU-serial per expert/stage):
        # sign32 = sum_{i<4, j<8} bit_j(wt_i) << (4j + i), wt_i = idxs[:, :, i%2, i//2].
        wts = torch.stack([idxs[:, :, i % 2, i // 2] for i in range(4)], dim=-1)        # (.., 4)
        bits = (wts.unsqueeze(-1) >> torch.arange(8, device=idxs.device)) & 1           # (.., 4, 8)
        shifts = (4 * torch.arange(8, device=idxs.device)).view(1, 8) \
            + torch.arange(4, device=idxs.device).view(4, 1)                            # (4, 8) = 4j+i
        sign32 = (bits << shifts).sum(dim=(-1, -2))                                     # (..)
        out = (sign32 << 32) + abs32
        return out.reshape(m // 16, 8, n // 8, 4).transpose(1, 2).contiguous().view(m, n // 4)

    def compute_paired_resid_scale(self, secondary, n_samples=8_000):
        """glq-convention rs that minimises 2-stage error (self primary, secondary stage-2)."""
        x = torch.randn(n_samples, self.DIM, device=self.device)
        x_s = x / self.opt_scale
        dec1, _ = self.quantize(x_s)
        residual = x_s - dec1
        best_nmse, best_rs = float('inf'), 1.0
        for rs in [0.5, 1.0, 2.0, 3.0, 3.45, 5.0, 7.0, 10.0]:
            dec2, _ = secondary.quantize(residual * rs)
            total = dec1 + dec2 / rs
            nmse = ((x_s - total) ** 2).sum(-1).mean().item() * (self.opt_scale ** 2)
            if nmse < best_nmse:
                best_nmse, best_rs = nmse, rs
        return best_rs

    def save(self, path):
        """Written for parity with the shell artifact layout (quantize_model always
        saves e8_codebook.pt). The E8P grid is rebuilt deterministically at load
        (hf_integration constructs E8PCodebook directly), so this file isn't read back."""
        torch.save({
            'codebook_type': 'e8p',
            'codebook': self.codebook.cpu(),
            'opt_scale': self.opt_scale,
            'grid_packed_abs': self.grid_packed_abs.cpu(),
        }, path)


# ----------------------------------------------------------------- E81B codebook (3-bit residual stage)
class E81BCodebook:
    DIM = 8

    def __init__(self, device='cpu', verbose=True):
        self.device = device
        self.codesz = _E8P_CODESZ
        self.opt_scale = 1.0                                       # stage-2; primary's opt_scale governs Wscale
        g = get_e81bgrid().to(device)                             # 255 entries
        self.codebook = g
        self.codebook_norms = (g ** 2).sum(-1)
        pad = torch.zeros(256 - g.shape[0], 8, dtype=torch.float16, device=device)
        self.e81b_grid = torch.cat([g.half(), pad], 0).contiguous()   # fp16 256×8 for the kernels
        if verbose:
            print(f"E81BCodebook: {g.shape[0]}-entry E8 residual (padded to 256 for kernel)")

    def quantize(self, x):
        x = x.float()
        idx = (2.0 * x @ self.codebook.T - self.codebook_norms).argmax(-1).long()
        return self.codebook[idx], idx

    def quantize_fast(self, x_half, decoded_out=None, idx_out=None):
        dec, idx = self.quantize(x_half)
        dec = dec.to(torch.float16)
        if decoded_out is not None:
            decoded_out.copy_(dec)
        if idx_out is not None:
            idx_out.copy_(idx)
        return dec, idx

    def decode(self, indices):
        return self.codebook[indices]

    @staticmethod
    def pack_e81b(idxs):
        """E81B 8-bit idx (m, n/8 int64) -> packed (m, n/64) int64 (8 stride-8 indices per int64).

        Static so quantize_model can pack the stage-2 indices without holding an E81BCodebook instance.
        """
        accum = idxs[:, 0::8].clone()
        for i in range(1, 8):
            accum = accum + (idxs[:, i::8] << (8 * i))
        return accum
