"""Tests for quantize_model.py — HessianCapture, worker functions, model profiles."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

from glq.codebook import E8ShellCodebook
from glq.quantize_model import (
    HessianCapture,
    pad_to_multiple,
    pad_hessian,
    quantize_layer_e8_shell_rht,
    _detect_profile,
    _resolve_attr,
    _init_worker,
    _quantize_sublayer,
    _worker_codebook,
    _DEFAULT_PROFILE,
    get_decoder_layers,
    get_embed,
    get_rotary_emb,
    _build_forward_kwargs,
    _load_layer_state,
    _load_tensor_from_shards,
    main,
)


@pytest.fixture(scope="module")
def codebook():
    return E8ShellCodebook.build(device="cpu", verbose=False)


def _random_psd(n):
    A = torch.randn(n, n)
    return A @ A.T + 0.1 * torch.eye(n)


# ---- P1a: HessianCapture ----

class TestHessianCapture:
    def test_hessian_shape(self):
        """H should be (in_features, in_features)."""
        linear = nn.Linear(64, 32)
        cap = HessianCapture(linear)
        with torch.no_grad():
            linear(torch.randn(16, 64))
        H = cap.finalize()
        assert H.shape == (64, 64)

    def test_hessian_accumulates(self):
        """H should equal (X1^T X1 + X2^T X2) / n_total."""
        linear = nn.Linear(64, 32)
        cap = HessianCapture(linear)
        x1 = torch.randn(8, 64)
        x2 = torch.randn(4, 64)
        with torch.no_grad():
            linear(x1)
            linear(x2)
        H = cap.finalize()

        expected = (x1.float().T @ x1.float() + x2.float().T @ x2.float()) / 12.0
        torch.testing.assert_close(H, expected, atol=1e-4, rtol=1e-4)

    def test_hessian_3d_input(self):
        """3D input (batch, seq, features) should be flattened to 2D."""
        linear = nn.Linear(64, 32)
        cap = HessianCapture(linear)
        with torch.no_grad():
            linear(torch.randn(2, 8, 64))  # 16 samples total
        H = cap.finalize()
        assert H.shape == (64, 64)
        assert cap.n_samples == 16

    def test_hessian_symmetric(self):
        linear = nn.Linear(32, 16)
        cap = HessianCapture(linear)
        with torch.no_grad():
            linear(torch.randn(10, 32))
        H = cap.finalize()
        torch.testing.assert_close(H, H.T, atol=1e-6, rtol=1e-6)

    def test_hessian_no_samples(self):
        """No forward pass → finalize returns None."""
        linear = nn.Linear(32, 16)
        cap = HessianCapture(linear)
        H = cap.finalize()
        assert H is None

    def test_hook_removed(self):
        """After finalize, the hook should no longer accumulate."""
        linear = nn.Linear(32, 16)
        cap = HessianCapture(linear)
        with torch.no_grad():
            linear(torch.randn(4, 32))
        H = cap.finalize()
        n_before = cap.n_samples

        # Another forward should NOT change the accumulator
        with torch.no_grad():
            linear(torch.randn(4, 32))
        assert cap.n_samples == n_before


# ---- P1b: Worker functions ----

class TestWorkerFunctions:
    def test_init_worker(self, codebook):
        """_init_worker should set the global _worker_codebook."""
        import glq.quantize_model as qm
        _init_worker(codebook.codebook.cpu(), codebook.opt_scale,
                     codebook.resid_scale, 1)
        assert qm._worker_codebook is not None
        assert qm._worker_codebook.codebook.shape == codebook.codebook.shape

    def test_quantize_sublayer_2bpw(self, codebook):
        """_quantize_sublayer should return (name, W_hat, artifacts, metrics)."""
        import glq.quantize_model as qm
        qm._worker_codebook = codebook

        torch.manual_seed(0)
        W = torch.randn(16, 64)
        H = _random_psd(64)
        name, W_hat, artifacts, metrics = _quantize_sublayer(
            ("test_layer", W, H, 2, 0))

        assert name == "test_layer"
        assert W_hat.shape == (16, 64)
        assert set(artifacts.keys()) == {'Qidxs', 'SU', 'SV', 'Wscale'}
        assert all(v.device.type == 'cpu' for v in artifacts.values())
        assert metrics['sqnr'] > 0

    def test_quantize_sublayer_3bpw(self, codebook):
        """3bpw should produce non-zero Qidxs2."""
        import glq.quantize_model as qm
        qm._worker_codebook = codebook

        torch.manual_seed(0)
        W = torch.randn(16, 64)
        H = _random_psd(64)
        name, W_hat, artifacts, metrics = _quantize_sublayer(
            ("test_3bpw", W, H, 3, 0))

        assert name == "test_3bpw"
        assert artifacts['Qidxs2'].abs().max() > 0
        assert artifacts['inv_resid_scale'].abs().item() > 0


# ---- P1c: Model profile detection ----

class TestModelProfiles:
    def test_detect_nemotron(self):
        class Cfg:
            architectures = ["NemotronHForCausalLM"]
        profile = _detect_profile(Cfg())
        assert profile['forward_kwargs'] == 'nemotron_h'
        assert profile['trust_remote_code'] is True

    def test_detect_default(self):
        class Cfg:
            architectures = ["LlamaForCausalLM"]
        profile = _detect_profile(Cfg())
        assert profile == _DEFAULT_PROFILE

    def test_detect_no_architectures(self):
        class Cfg:
            pass
        profile = _detect_profile(Cfg())
        assert profile == _DEFAULT_PROFILE

    def test_resolve_attr(self):
        class Inner:
            val = 42
        class Outer:
            inner = Inner()
        assert _resolve_attr(Outer(), "inner.val") == 42

    def test_resolve_attr_single(self):
        class Obj:
            x = 7
        assert _resolve_attr(Obj(), "x") == 7


# ---- P1d: Cholesky fallback ----

class TestCholeskyFallback:
    def test_near_singular_hessian(self, codebook):
        """Near-singular H should succeed via heavy damping fallback."""
        torch.manual_seed(42)
        W = torch.randn(8, 64)
        # Rank-1 Hessian (singular)
        v = torch.randn(64)
        H = v.unsqueeze(1) @ v.unsqueeze(0)
        # Should not raise — 3-tier fallback handles this
        W_hat, artifacts, metrics = quantize_layer_e8_shell_rht(
            W, H, codebook, bpw=2)
        assert W_hat.shape == W.shape
        assert metrics['sqnr'] > 0

    def test_zero_hessian(self, codebook):
        """Zero H should succeed via identity fallback."""
        torch.manual_seed(42)
        W = torch.randn(8, 64)
        H = torch.zeros(64, 64)
        W_hat, artifacts, metrics = quantize_layer_e8_shell_rht(
            W, H, codebook, bpw=2)
        assert W_hat.shape == W.shape


# ---- P4: Parallel integration ----

class TestParallelIntegration:
    def test_parallel_matches_sequential(self, codebook):
        """Parallel worker results should match sequential quantization."""
        from concurrent.futures import ProcessPoolExecutor
        import os

        # Generate data once, clone for each run to avoid in-place H modification
        torch.manual_seed(42)
        base_data = [
            (f"layer_{i}", torch.randn(16, 64), _random_psd(64), 2, 0)
            for i in range(3)
        ]

        # Sequential (clone H to avoid in-place damping corruption)
        seq_results = []
        for name, W, H, bpw, ti in base_data:
            W_hat, artifacts, metrics = quantize_layer_e8_shell_rht(
                W.clone(), H.clone(), codebook, bpw=bpw, tune_iters=ti)
            seq_results.append(metrics['sqnr'])

        # Parallel via worker pool (H is serialized → fresh copy per worker)
        # Use 'spawn' context to avoid fork+CUDA deadlock when pytest has
        # initialized CUDA in earlier tests.
        import multiprocessing
        n_threads = max(1, (os.cpu_count() or 1) // 2)
        with ProcessPoolExecutor(
            max_workers=2,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_init_worker,
            initargs=(codebook.codebook.cpu(), codebook.opt_scale,
                      codebook.resid_scale, n_threads),
        ) as pool:
            par_results_raw = list(pool.map(_quantize_sublayer, base_data))
        par_sqnrs = [r[3]['sqnr'] for r in par_results_raw]

        for s, p in zip(seq_results, par_sqnrs):
            assert s == pytest.approx(p, abs=1e-4)


# ---- P5: Model structure helpers ----

_SENTINEL = object()


class TestGetDecoderLayers:
    def test_profile_attr(self):
        """Profile layers_attr overrides heuristics."""
        class M:
            class backbone:
                layers = _SENTINEL
        result = get_decoder_layers(M(), profile={'layers_attr': 'backbone.layers'})
        assert result is _SENTINEL

    def test_direct_layers(self):
        """model.layers is the first heuristic."""
        class M:
            layers = _SENTINEL
        assert get_decoder_layers(M()) is _SENTINEL

    def test_model_dot_layers(self):
        """model.model.layers fallback."""
        class Inner:
            layers = _SENTINEL
        class M:
            model = Inner()
        assert get_decoder_layers(M()) is _SENTINEL

    def test_backbone_dot_layers(self):
        """model.backbone.layers fallback."""
        class Backbone:
            layers = _SENTINEL
        class M:
            backbone = Backbone()
        assert get_decoder_layers(M()) is _SENTINEL

    def test_raises_when_missing(self):
        """No matching attribute → ValueError."""
        class M:
            pass
        with pytest.raises(ValueError, match="Cannot find transformer layers"):
            get_decoder_layers(M())


class TestGetEmbed:
    def test_profile_attr(self):
        class M:
            class backbone:
                embeddings = _SENTINEL
        result = get_embed(M(), profile={'embed_attr': 'backbone.embeddings'})
        assert result is _SENTINEL

    def test_direct_embed_tokens(self):
        class M:
            embed_tokens = _SENTINEL
        assert get_embed(M()) is _SENTINEL

    def test_model_dot_embed_tokens(self):
        class Inner:
            embed_tokens = _SENTINEL
        class M:
            model = Inner()
        assert get_embed(M()) is _SENTINEL

    def test_direct_embeddings(self):
        """NemotronH-style: model.embeddings."""
        class M:
            embeddings = _SENTINEL
        assert get_embed(M()) is _SENTINEL

    def test_model_dot_embeddings(self):
        class Inner:
            embeddings = _SENTINEL
        class M:
            model = Inner()
        assert get_embed(M()) is _SENTINEL

    def test_raises_when_missing(self):
        class M:
            pass
        with pytest.raises(ValueError, match="Cannot find embedding layer"):
            get_embed(M())


class TestGetRotaryEmb:
    def test_profile_attr(self):
        class Inner:
            rotary_emb = _SENTINEL
        class M:
            model = Inner()
        result = get_rotary_emb(M(), profile={'rotary_attr': 'model.rotary_emb'})
        assert result is _SENTINEL

    def test_profile_no_rotary(self):
        """Profile explicitly declares no rotary (e.g. NemotronH)."""
        class M:
            pass
        result = get_rotary_emb(M(), profile={'rotary_attr': None, 'layers_attr': 'x'})
        assert result is None

    def test_direct_rotary_emb(self):
        """model.rotary_emb heuristic (no .model wrapper)."""
        class M:
            rotary_emb = _SENTINEL
        assert get_rotary_emb(M()) is _SENTINEL

    def test_heuristic_found(self):
        class Inner:
            rotary_emb = _SENTINEL
        class M:
            model = Inner()
        assert get_rotary_emb(M()) is _SENTINEL

    def test_heuristic_missing(self):
        class M:
            pass
        assert get_rotary_emb(M()) is None


class TestBuildForwardKwargs:
    def test_nemotron_h(self):
        h = torch.randn(1, 16, 64)
        kwargs = _build_forward_kwargs({'forward_kwargs': 'nemotron_h'}, h, None)
        assert kwargs['cache_params'] is None
        assert kwargs['cache_position'].shape == (16,)
        assert 'position_ids' not in kwargs

    def test_default_no_rotary(self):
        h = torch.randn(1, 8, 32)
        kwargs = _build_forward_kwargs({}, h, rotary_emb=None)
        assert kwargs['position_ids'].shape == (1, 8)
        assert kwargs['use_cache'] is False
        assert 'position_embeddings' not in kwargs

    def test_default_with_rotary(self):
        h = torch.randn(1, 8, 32)
        cos = torch.randn(1, 8, 16)
        sin = torch.randn(1, 8, 16)
        rotary_emb = lambda x, position_ids: (cos, sin)
        kwargs = _build_forward_kwargs({}, h, rotary_emb=rotary_emb)
        assert 'position_embeddings' in kwargs
        assert kwargs['position_embeddings'] == (cos, sin)


# ---- P6: Safetensors I/O helpers ----

safetensors = pytest.importorskip("safetensors")


class TestSafetensorsIO:
    def test_load_tensor_from_shards(self, tmp_path):
        from safetensors.torch import save_file
        t = torch.randn(16, 64)
        shard_file = tmp_path / "shard-00001.safetensors"
        key = "model.layers.0.self_attn.q_proj.weight"
        save_file({key: t}, str(shard_file))

        weight_map = {key: shard_file.name}
        shard_paths = {shard_file.name: str(shard_file)}
        loaded = _load_tensor_from_shards(weight_map, shard_paths, key)
        torch.testing.assert_close(loaded, t)

    def test_load_layer_state_single_shard(self, tmp_path):
        from safetensors.torch import save_file
        t1 = torch.randn(16, 64)
        t2 = torch.randn(64, 128)
        shard_file = tmp_path / "shard-00001.safetensors"
        save_file({
            "model.layers.0.self_attn.q_proj.weight": t1,
            "model.layers.0.mlp.gate_proj.weight": t2,
        }, str(shard_file))

        weight_map = {
            "model.layers.0.self_attn.q_proj.weight": shard_file.name,
            "model.layers.0.mlp.gate_proj.weight": shard_file.name,
        }
        shard_paths = {shard_file.name: str(shard_file)}
        state = _load_layer_state(weight_map, shard_paths, layer_idx=0,
                                  sd_prefix="model.layers")
        assert set(state.keys()) == {
            "self_attn.q_proj.weight", "mlp.gate_proj.weight"}
        torch.testing.assert_close(state["self_attn.q_proj.weight"], t1)
        torch.testing.assert_close(state["mlp.gate_proj.weight"], t2)

    def test_load_layer_state_multi_shard(self, tmp_path):
        from safetensors.torch import save_file
        t1 = torch.randn(16, 64)
        t2 = torch.randn(64, 128)
        shard1 = tmp_path / "shard-00001.safetensors"
        shard2 = tmp_path / "shard-00002.safetensors"
        save_file({"model.layers.1.attn.weight": t1}, str(shard1))
        save_file({"model.layers.1.mlp.weight": t2}, str(shard2))

        weight_map = {
            "model.layers.1.attn.weight": shard1.name,
            "model.layers.1.mlp.weight": shard2.name,
        }
        shard_paths = {
            shard1.name: str(shard1),
            shard2.name: str(shard2),
        }
        state = _load_layer_state(weight_map, shard_paths, layer_idx=1,
                                  sd_prefix="model.layers")
        assert set(state.keys()) == {"attn.weight", "mlp.weight"}
        torch.testing.assert_close(state["attn.weight"], t1)
        torch.testing.assert_close(state["mlp.weight"], t2)

    def test_load_layer_state_filters_other_layers(self, tmp_path):
        from safetensors.torch import save_file
        t0 = torch.randn(8, 8)
        t1 = torch.randn(16, 16)
        shard = tmp_path / "shard.safetensors"
        save_file({
            "model.layers.0.weight": t0,
            "model.layers.1.weight": t1,
        }, str(shard))

        weight_map = {
            "model.layers.0.weight": shard.name,
            "model.layers.1.weight": shard.name,
        }
        shard_paths = {shard.name: str(shard)}
        state = _load_layer_state(weight_map, shard_paths, layer_idx=0,
                                  sd_prefix="model.layers")
        assert set(state.keys()) == {"weight"}
        torch.testing.assert_close(state["weight"], t0)


# ---- P10: CLI argparse ----

class TestCLIArgparse:
    def test_defaults(self):
        """--model and --output are required; defaults for rest."""
        with patch('sys.argv', ['prog', '--model', 'test/model', '--output', '/tmp/out']):
            with patch('glq.quantize_model.quantize') as mock_q:
                main()
                mock_q.assert_called_once()
                kwargs = mock_q.call_args[1]
                assert kwargs['model_name'] == 'test/model'
                assert kwargs['output_dir'] == '/tmp/out'
                assert kwargs['bpw'] == 2
                assert kwargs['tune_iters'] == 0
                assert kwargs['nsamples'] == 128
                assert kwargs['seqlen'] == 2048
                assert kwargs['device'] == 'cuda'
                assert kwargs['trust_remote_code'] is False
                assert kwargs['streaming'] is False
                assert kwargs['workers'] == 0

    def test_all_options(self):
        """All CLI flags parsed correctly."""
        argv = [
            'prog', '--model', 'my/model', '--output', '/out',
            '--bpw', '4', '--tune-iters', '3', '--nsamples', '32',
            '--seqlen', '4096', '--device', 'cpu',
            '--trust-remote-code', '--streaming', '--workers', '8',
        ]
        with patch('sys.argv', argv):
            with patch('glq.quantize_model.quantize') as mock_q:
                main()
                kwargs = mock_q.call_args[1]
                assert kwargs['bpw'] == 4
                assert kwargs['tune_iters'] == 3
                assert kwargs['nsamples'] == 32
                assert kwargs['seqlen'] == 4096
                assert kwargs['device'] == 'cpu'
                assert kwargs['trust_remote_code'] is True
                assert kwargs['streaming'] is True
                assert kwargs['workers'] == 8
