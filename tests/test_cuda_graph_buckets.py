"""Unit tests for CUDAGraphBucketWrapper (shape-bucketed CUDA graphs).

Uses a tiny randomly-initialised LlamaForCausalLM — correctness of the
wrapper (padding, bucket selection, replay) is independent of the GLQ
kernel path, which is covered by tests/test_block_diagonal_cuda.py.
"""

import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for graph wrapper"
)


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to("cuda").half()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _close(a, b, tol=1e-3):
    diff = (a.float() - b.float()).abs().max().item()
    scale = a.float().abs().max().item() + 1e-9
    return diff / scale < tol, diff, scale


class TestCUDAGraphBucketWrapper:
    def test_right_pad_matches_eager(self, tiny_model):
        from glq.cuda_graph import CUDAGraphBucketWrapper
        buckets = [(4, 128), (8, 256)]
        wrapper = CUDAGraphBucketWrapper(tiny_model, buckets=buckets, padding="right")

        for B, S in [(2, 100), (4, 128), (5, 200)]:
            ids = torch.randint(0, 256, (B, S), dtype=torch.long, device="cuda")
            eager = tiny_model(input_ids=ids, use_cache=False).logits
            out = wrapper(ids).logits
            ok, diff, scale = _close(eager, out)
            assert ok, f"(B={B},S={S}) diff={diff:.5f} scale={scale:.3f}"

    def test_left_pad_matches_eager(self, tiny_model):
        from glq.cuda_graph import CUDAGraphBucketWrapper
        buckets = [(4, 128), (8, 256)]
        wrapper = CUDAGraphBucketWrapper(tiny_model, buckets=buckets, padding="left")

        # S == bucket S means padding amount is zero, so output must match
        # a vanilla eager forward.
        B, S = 3, 128
        ids = torch.randint(0, 256, (B, S), dtype=torch.long, device="cuda")
        eager = tiny_model(input_ids=ids, use_cache=False).logits
        out = wrapper(ids).logits
        ok, diff, _ = _close(eager, out)
        assert ok, f"S==bucket diff={diff:.5f}"

        # With S < bucket, build an eager forward that mirrors the
        # wrapper's left-pad + position_ids handling and compare on the
        # valid slice only.
        B, S = 3, 70
        S_cap = 128
        ids = torch.randint(0, 256, (B, S), dtype=torch.long, device="cuda")
        pad_id = tiny_model.config.eos_token_id or 0
        pad_ids = torch.full((B, S_cap - S), fill_value=pad_id,
                             dtype=torch.long, device="cuda")
        padded_ids = torch.cat([pad_ids, ids], dim=1)
        attn = torch.cat([torch.zeros(B, S_cap - S, dtype=torch.long, device="cuda"),
                          torch.ones(B, S, dtype=torch.long, device="cuda")], dim=1)
        pos1d = (torch.arange(S_cap, device="cuda") - (S_cap - S)).clamp_min(0)
        eager = tiny_model(
            input_ids=padded_ids, attention_mask=attn,
            position_ids=pos1d.unsqueeze(0).expand(B, S_cap).contiguous(),
            use_cache=False).logits
        eager_valid = eager[:, S_cap - S:]
        out = wrapper(ids).logits
        ok, diff, scale = _close(eager_valid, out, tol=2e-3)
        assert ok, f"(B={B},S={S}) left-pad diff={diff:.5f} scale={scale:.3f}"

    def test_oversized_input_falls_back(self, tiny_model):
        from glq.cuda_graph import CUDAGraphBucketWrapper
        buckets = [(4, 128)]
        wrapper = CUDAGraphBucketWrapper(tiny_model, buckets=buckets, padding="right")

        # Larger than max bucket — must go straight to eager; no capture.
        ids = torch.randint(0, 256, (8, 256), dtype=torch.long, device="cuda")
        out = wrapper(ids)
        assert hasattr(out, "logits")
        assert out.logits.shape == (8, 256, tiny_model.config.vocab_size)
        assert not wrapper._graphs, "should not have captured any graph"

    def test_use_cache_falls_back(self, tiny_model):
        from glq.cuda_graph import CUDAGraphBucketWrapper
        buckets = [(4, 128)]
        wrapper = CUDAGraphBucketWrapper(tiny_model, buckets=buckets, padding="right")

        ids = torch.randint(0, 256, (2, 64), dtype=torch.long, device="cuda")
        out = wrapper(ids, use_cache=True)
        assert hasattr(out, "logits")
        assert out.logits.shape == (2, 64, tiny_model.config.vocab_size)
        assert not wrapper._graphs, "use_cache=True must bypass graph capture"

    def test_replay_faster_than_capture(self, tiny_model):
        from glq.cuda_graph import CUDAGraphBucketWrapper
        buckets = [(4, 128)]
        wrapper = CUDAGraphBucketWrapper(tiny_model, buckets=buckets, padding="right")

        ids = torch.randint(0, 256, (4, 128), dtype=torch.long, device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        _ = wrapper(ids)
        torch.cuda.synchronize()
        t_first = time.time() - t0

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            _ = wrapper(ids)
        torch.cuda.synchronize()
        t_replay = (time.time() - t0) / 5

        # Tiny model: capture cost is bounded but replay must still beat
        # the first (capture + warmup) call by a clear margin.
        assert t_replay < t_first * 0.8, (
            f"replay not materially faster than capture: "
            f"first={t_first*1000:.2f}ms replay={t_replay*1000:.2f}ms")


class TestPhaseAWrapperStillWorks:
    """Ensure adding CUDAGraphBucketWrapper didn't break the decode
    wrapper or its imports."""

    def test_decode_wrapper_still_runs(self, tiny_model):
        from glq.cuda_graph import CUDAGraphWrapper
        wrapper = CUDAGraphWrapper(tiny_model, max_cache_len=256)
        ids = torch.randint(0, 256, (1, 8), dtype=torch.long, device="cuda")
        out = wrapper.generate(ids, max_new_tokens=8)
        assert out.shape[0] == 1 and out.shape[1] == 16
