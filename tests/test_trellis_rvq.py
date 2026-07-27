"""Trellis RVQ (stacked 5-8 bpw): recipe, multi-stage LDLQ, and the decode stage-sum.

The adopted architecture is two trellis stages, each K<=4, residual-quantized — stage 1 is
always K=4 and stage 2 is K=(bpw-4), so 5=4+1 / 6=4+2 / 7=4+3 / 8=4+4. Chosen over a native
K=5..8 trellis because native ties at 5 and loses 0.3/0.9/2.4 dB at 6/7/8 (the L=16 decode
window shrinks to ceil(L/K) symbols), and because every stage staying at R<=4 reuses the
existing CUDA kernels — `WIDTH = 8*R` overflows a uint32 at R>=5.

CPU-only and small on purpose: the Viterbi cost is 2**L per step regardless of K, so these
run in the local CPU subset.
"""
import math
import warnings

import pytest
import torch

import glq.trellis as gt


def _sqnr(W, Wh):
    num = W.float().pow(2).mean().item()
    den = (W.float() - Wh.float()).pow(2).mean().item()
    return 10 * math.log10(num / den) if den > 0 else 99.0


def _cbs(bpw, device="cpu"):
    return [gt.TrellisCodebook(variant="3inst", K=k, device=device)
            for k in gt.trellis_rvq_recipe(bpw)]


class TestRecipe:
    @pytest.mark.parametrize("bpw,want", [(2, [2]), (3, [3]), (4, [4]),
                                          (5, [4, 1]), (6, [4, 2]),
                                          (7, [4, 3]), (8, [4, 4])])
    def test_recipe(self, bpw, want):
        assert gt.trellis_rvq_recipe(bpw) == want

    def test_stage1_is_always_k4_above_4bpw(self):
        """One primary codebook at every stacked rate — only the residual rate varies."""
        for bpw in (5, 6, 7, 8):
            assert gt.trellis_rvq_recipe(bpw)[0] == 4

    def test_every_stage_is_kernel_serviceable(self):
        """No stage may exceed R=4: glq_trellis.cu packs each chunk into a uint32 whose
        width is 8*R, so R>=5 has no kernel. This is the whole reason for stacking."""
        for bpw in range(2, 9):
            assert all(1 <= k <= 4 for k in gt.trellis_rvq_recipe(bpw))

    def test_bits_sum_to_bpw(self):
        for bpw in range(2, 9):
            assert sum(gt.trellis_rvq_recipe(bpw)) == bpw

    @pytest.mark.parametrize("bpw", [1, 0, 9, 12])
    def test_out_of_range_rejected(self, bpw):
        with pytest.raises(ValueError, match="2-8"):
            gt.trellis_rvq_recipe(bpw)


class TestNStageLDLQ:
    def test_single_stage_delegates_bit_identically(self):
        """bpw 2-4 must keep going through the ORIGINAL single-stage code path — existing
        checkpoints are byte-identical-gated, so a refactor that perturbs them is a bug."""
        torch.manual_seed(3)
        W, H = torch.randn(64, 64), torch.eye(64)
        cb = gt.TrellisCodebook(variant="3inst", K=4)
        ref_hat, ref_idx, ref_ws = gt.trellis_ldlq(W, H, cb)
        got_hat, got_idxs, got_ws, got_rs = gt.trellis_ldlq_nstage(W, H, [cb])
        assert torch.equal(ref_hat, got_hat)
        assert torch.equal(ref_idx, got_idxs[0])
        assert ref_ws == got_ws
        assert got_rs == [1.0]

    @pytest.mark.parametrize("bpw", [5, 6, 8])
    def test_stage_count_and_shapes(self, bpw):
        torch.manual_seed(4)
        W, H = torch.randn(64, 64), torch.eye(64)
        cbs = _cbs(bpw)
        _, idxs, _, cum_inv_rs = gt.trellis_ldlq_nstage(W, H, cbs)
        assert len(idxs) == len(cbs) == 2
        assert len(cum_inv_rs) == 2 and cum_inv_rs[0] == 1.0
        for q in idxs:
            assert q.shape == (64, 64)          # V=1

    def test_second_stage_strictly_improves(self):
        """The residual stage must actually reduce error — a dropped/ignored stage 2 would
        leave this equal to the 4 bpw result."""
        torch.manual_seed(5)
        W, H = torch.randn(128, 64), torch.eye(64)
        cb4 = gt.TrellisCodebook(variant="3inst", K=4)
        hat1, _, ws1 = gt.trellis_ldlq(W, H, cb4)
        hat2, _, ws2, _ = gt.trellis_ldlq_nstage(W, H, _cbs(6))
        assert _sqnr(W / ws2, hat2) > _sqnr(W / ws1, hat1) + 3.0

    def test_sqnr_monotonic_in_bpw(self):
        """Each added residual bit must buy error reduction; a flat step means the extra
        bits never reached the Viterbi."""
        torch.manual_seed(6)
        W, H = torch.randn(128, 64), torch.eye(64)
        sq = {}
        for bpw in (5, 6, 7, 8):
            hat, _, ws, _ = gt.trellis_ldlq_nstage(W, H, _cbs(bpw))
            sq[bpw] = _sqnr(W / ws, hat)
        assert sq[8] > sq[7] > sq[6] > sq[5], sq

    def test_fitted_resid_scale_is_positive_and_amplifying(self):
        """The residual is amplified into the codebook's calibrated range, so the encode-side
        scale is >1 and cum_inv_rs shrinks it back on reconstruction."""
        torch.manual_seed(7)
        W, H = torch.randn(64, 64), torch.eye(64)
        _, _, _, cum_inv_rs = gt.trellis_ldlq_nstage(W, H, _cbs(6))
        assert 0.0 < cum_inv_rs[1] < 1.0

    def test_resid_scale_applies_the_per_rate_calibration(self):
        """Mechanism: the shipped scale is rms_ratio * C[K2], not the bare rms ratio.

        The bare ratio is computed from the STAGE-1 residual, so it is identical at every
        bpw on a layer — measured, that costs up to 3.6 dB at K2=4. Asserting the ratio of
        the fitted scales across two bpws equals the ratio of their constants proves the
        rate correction is really applied and did not silently cancel.
        """
        torch.manual_seed(11)
        W, H = torch.randn(128, 64), torch.eye(64)
        rs = {}
        for bpw in (5, 8):
            _, _, _, cum = gt.trellis_ldlq_nstage(W, H, _cbs(bpw))
            rs[bpw] = 1.0 / cum[1]
        want = gt._RESID_SCALE_CAL[1] / gt._RESID_SCALE_CAL[4]
        assert rs[5] / rs[8] == pytest.approx(want, rel=1e-6), (rs, want)

    def test_calibration_covers_every_residual_rate(self):
        """Every K a stacked recipe can produce must have a constant, or that rate silently
        falls back to the uncorrected ratio."""
        for bpw in (5, 6, 7, 8):
            assert gt.trellis_rvq_recipe(bpw)[-1] in gt._RESID_SCALE_CAL


class TestDecodeStageSum:
    @pytest.mark.parametrize("bpw", [5, 6, 7, 8])
    def test_packed_roundtrip_reproduces_hatW(self, bpw):
        """THE gate: decode from the stored per-stage packed buffers must reproduce the
        quantizer's own hatW. Compared against hatW, not against another decode — a
        decode-vs-decode A/B shares the scale wiring and passes even with a dropped stage."""
        torch.manual_seed(8 + bpw)
        m, n = 64, 64
        W, H = torch.randn(m, n), torch.eye(n)
        cbs = _cbs(bpw)
        hat, idxs, _, cum_inv_rs = gt.trellis_ldlq_nstage(W, H, cbs)
        packed = [gt.pack_layer(cb, q, m, n, has_kernel=True) for cb, q in zip(cbs, idxs)]
        dec = gt.decode_layer_nstage(cbs, packed, cum_inv_rs, m, n, has_kernel=True)
        assert torch.allclose(dec, hat, atol=1e-5), (dec - hat).abs().max().item()

    def test_dropping_the_residual_stage_changes_the_result(self):
        """Companion to the round-trip: proves stage 2 contributes. The e8p stage-3/4 silent
        drop passed every A/B precisely because nothing asserted the top stage mattered."""
        torch.manual_seed(9)
        m, n = 64, 64
        W, H = torch.randn(m, n), torch.eye(n)
        cbs = _cbs(6)
        _, idxs, _, cum_inv_rs = gt.trellis_ldlq_nstage(W, H, cbs)
        packed = [gt.pack_layer(cb, q, m, n, has_kernel=True) for cb, q in zip(cbs, idxs)]
        full = gt.decode_layer_nstage(cbs, packed, cum_inv_rs, m, n, has_kernel=True)
        stage1 = gt.decode_layer_nstage(cbs[:1], packed[:1], cum_inv_rs[:1], m, n,
                                        has_kernel=True)
        assert (full - stage1).abs().max().item() > 1e-3

    def test_stage_sum_matches_manual_weighted_sum(self):
        """decode_layer_nstage is exactly sum_s decode(packed_s) * cum_inv_rs[s]."""
        torch.manual_seed(10)
        m, n = 64, 64
        W, H = torch.randn(m, n), torch.eye(n)
        cbs = _cbs(7)
        _, idxs, _, cum_inv_rs = gt.trellis_ldlq_nstage(W, H, cbs)
        packed = [gt.pack_layer(cb, q, m, n, has_kernel=True) for cb, q in zip(cbs, idxs)]
        got = gt.decode_layer_nstage(cbs, packed, cum_inv_rs, m, n, has_kernel=True)
        want = sum(gt.decode_layer(cb, p, m, n, has_kernel=True) * s
                   for cb, p, s in zip(cbs, packed, cum_inv_rs))
        assert torch.equal(got, want)


class TestCheckpointRoundTrip:
    """quantize -> artifacts -> E8RHTLinear -> forward, i.e. the shipping path."""

    @staticmethod
    def _quantize(bpw, m=64, n=64, seed=20):
        from glq.quantize_model import quantize_layer_e8_shell_rht
        torch.manual_seed(seed)
        W = torch.randn(m, n)
        H = torch.eye(n) + 0.05 * torch.randn(n, n) @ torch.randn(n, n).T / n
        cbs = _cbs(bpw)
        primary = cbs[0]
        primary.rvq_stages = cbs                       # how the driver threads the recipe
        W_hat, arts, metrics = quantize_layer_e8_shell_rht(
            W, H, primary, bpw=bpw, block_diagonal=True)
        return W, H, W_hat, arts, metrics

    @pytest.mark.parametrize("bpw", [2, 3, 4])
    def test_single_stage_emits_no_residual_keys(self, bpw):
        """A 2-4 bpw checkpoint must not carry stage-2 keys — the loader keys the stage
        count off `trellis_packed2`'s presence, so an empty one would falsely claim two."""
        _, _, _, arts, _ = self._quantize(bpw)
        assert "trellis_packed" in arts
        assert "trellis_packed2" not in arts
        assert "inv_resid_scale2" not in arts

    @pytest.mark.parametrize("bpw", [5, 6, 7, 8])
    def test_stacked_emits_residual_keys_at_the_right_rate(self, bpw):
        _, _, _, arts, metrics = self._quantize(bpw)
        assert "trellis_packed2" in arts and "inv_resid_scale2" in arts
        # Each stage's rate is self-describing from its own packed width (cols == 16*K) —
        # this is why no config.json stage marker is needed.
        assert arts["trellis_packed"].shape[1] // 16 == 4
        assert arts["trellis_packed2"].shape[1] // 16 == bpw - 4
        assert metrics["bpw"] == bpw, "metrics must report the SUMMED rate, not stage 1's"

    @pytest.mark.parametrize("bpw", [5, 6, 8])
    def test_layer_forward_matches_W_hat(self, bpw):
        """THE gate: load the artifacts into the real module and compare its forward
        against the quantizer's own W_hat — not against another decode, which would share
        the scale wiring and pass even with the residual stage dropped."""
        from glq.quantized_linear import E8RHTLinear
        W, _, W_hat, arts, _ = self._quantize(bpw)
        m, n = W.shape
        lin = E8RHTLinear(n, m, bias=False, codebook_type="trellis")
        lin.load_state_dict({k: v for k, v in arts.items()}, strict=False)
        cbs = _cbs(bpw)
        lin.set_codebook(cbs[0], cbs[1])
        x = torch.randn(4, n)
        got = lin(x).float()
        want = x @ W_hat.float().T
        num = want.pow(2).mean().item()
        den = (want - got).pow(2).mean().item()
        sqnr = 10 * math.log10(num / den) if den > 0 else 99.0
        assert sqnr > 40.0, f"bpw {bpw}: decode-vs-W_hat only {sqnr:.1f} dB"

    def test_dropping_stage2_buffer_changes_the_forward(self):
        """Companion: proves the residual stage reaches the forward. The e8p stage-3/4
        silent drop survived every A/B precisely because nothing asserted this."""
        from glq.quantized_linear import E8RHTLinear
        W, _, _, arts, _ = self._quantize(6)
        m, n = W.shape
        x = torch.randn(4, n)
        cbs = _cbs(6)

        full = E8RHTLinear(n, m, bias=False, codebook_type="trellis")
        full.load_state_dict(dict(arts), strict=False)
        full.set_codebook(cbs[0], cbs[1])

        stage1 = E8RHTLinear(n, m, bias=False, codebook_type="trellis")
        stripped = {k: v for k, v in arts.items()
                    if k not in ("trellis_packed2", "inv_resid_scale2")}
        stage1.load_state_dict(stripped, strict=False)
        stage1.set_codebook(cbs[0], None)

        assert (full(x) - stage1(x)).abs().max().item() > 1e-3

    def test_stage2_flag_and_scale_resolve_together(self):
        """The e8p bug was a flag that said 'stage present' beside a scale still at 0.0.
        Both come from one block now; assert they agree."""
        from glq.quantized_linear import E8RHTLinear
        W, _, _, arts, _ = self._quantize(7)
        m, n = W.shape
        lin = E8RHTLinear(n, m, bias=False, codebook_type="trellis")
        lin.load_state_dict(dict(arts), strict=False)
        cbs = _cbs(7)
        lin.set_codebook(cbs[0], cbs[1])
        assert lin._trellis_has_stage2 is True
        assert lin._inv_rs2_float != 0.0

    def test_stacked_layer_refuses_the_fused_cuda_op(self, monkeypatch):
        """Mechanism, not output: the fused op takes a single `trellis_packed`, so taking
        it for a 2-stage layer would decode stage 1 only and return PLAUSIBLE numbers.
        Comparing values cannot catch that; asserting the dispatch can.

        The notice is a one-shot per-process latch (else a 3B model emits it ~200 times),
        so reset it here rather than depending on test order.
        """
        import glq.quantized_linear as ql
        from glq.quantized_linear import E8RHTLinear
        monkeypatch.setattr(ql, "_WARNED_TRELLIS_RVQ_EAGER", False)
        W, _, _, arts, _ = self._quantize(6)
        m, n = W.shape
        lin = E8RHTLinear(n, m, bias=False, codebook_type="trellis")
        lin.load_state_dict(dict(arts), strict=False)
        cbs = _cbs(6)
        lin.set_codebook(cbs[0], cbs[1])
        with pytest.warns(RuntimeWarning, match="no fused CUDA kernel"):
            assert lin._trellis_op_usable(torch.zeros(1, n)) is False

    def test_single_stage_layer_does_not_warn(self):
        """The notice must be specific to the stacked path — 2-4 bpw is unaffected and
        must not start emitting warnings on every load."""
        import glq.quantized_linear as ql
        from glq.quantized_linear import E8RHTLinear
        ql._WARNED_TRELLIS_RVQ_EAGER = False
        W, _, _, arts, _ = self._quantize(4)
        m, n = W.shape
        lin = E8RHTLinear(n, m, bias=False, codebook_type="trellis")
        lin.load_state_dict(dict(arts), strict=False)
        lin.set_codebook(_cbs(4)[0], None)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            lin._trellis_op_usable(torch.zeros(1, n))     # must not raise


def test_kv_le_l_asserted():
    """K*V > L silently slices pack_trellis to empty; K*V == L degenerates the trellis."""
    with pytest.raises((AssertionError, ValueError)):
        gt.TrellisCodebook(variant="3inst", K=17)
