"""Tests for GLQ vLLM plugin — registration, correctness, VRAM, throughput."""

import os
import time

# vLLM v1 serializes model state between processes; GLQ params have
# function references (weight_loader) that aren't msgpack-serializable.
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import pytest
import torch

try:
    from vllm.model_executor.layers.quantization import get_quantization_config
    _HAS_VLLM = True
except (ImportError, ModuleNotFoundError):
    _HAS_VLLM = False

try:
    import transformers
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

requires_vllm = pytest.mark.skipif(not _HAS_VLLM, reason="vllm not installed")
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required"
)

MODEL_ID = "xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw"
TOKENIZER_ID = "HuggingFaceTB/SmolLM3-3B"


# ── Test 1: Config registration (no GPU needed) ────────────────────────

@requires_vllm
def test_glq_config_registration():
    """GLQvLLMConfig should be discoverable after import glq_vllm."""
    import glq_vllm  # noqa: F401
    from glq_vllm.config import GLQvLLMConfig
    assert get_quantization_config("glq") == GLQvLLMConfig


# ── Test 2: Config parsing (no GPU needed) ──────────────────────────────

@requires_vllm
@requires_gpu
def test_fused_kernel_custom_ops_registered():
    """All 9 GLQ pybind entrypoints are registered as torch.ops.glq.*.

    The 6 dequant + RHT kernels were registered first; the 3 fused-linear /
    fused-MoE entrypoints were added later so vLLM's torch.compile (mode>=3)
    can trace through them as opaque ops. Without this registration,
    dynamo crashes with ``Attempted to call function marked as skipped`` on
    the pybind11 functions.
    """
    import glq_vllm.custom_ops
    glq_vllm.custom_ops._ensure_registered()
    expected = [
        "dequant_matvec", "dequant_matmul",
        "dequant_matvec_packed", "dequant_matmul_packed",
        "input_rht", "output_rht",
        "fused_linear", "fused_linear_block_diag", "fused_moe_block_diag",
        # v0.3.3: Triton-fallback RHT wrappers (n_pad > 16384).
        "input_rht_triton", "output_rht_triton",
        # v0.3.3: KV gather/scatter Triton wrappers (preparatory for
        # full-graph mode; the attention region is already a piecewise
        # graph-break boundary today).
        "gather_kv_paged_dequant", "scatter_kv_paged_quant",
    ]
    missing = [op for op in expected if not hasattr(torch.ops.glq, op)]
    assert not missing, f"unregistered: {missing}"


@requires_vllm
def test_fused_linear_fake_shape():
    """Fake implementation must match the live kernel's output shape so
    torch.compile shape inference doesn't disagree with runtime.

    Uses meta-device tensors so the fake impl runs without needing the
    real kernel — exercises only the shape inference path.
    """
    import glq_vllm.custom_ops
    glq_vllm.custom_ops._ensure_registered()
    if not hasattr(torch.ops.glq, "fused_linear"):
        pytest.skip("fused_linear not registered (no CUDA ext loaded)")

    B, in_features, out_features = 2, 1024, 768
    n_pad, m_pad = 1024, 1024
    meta = torch.device("meta")
    x = torch.empty(B, in_features, dtype=torch.float16, device=meta)
    sv = torch.empty(n_pad, dtype=torch.float16, device=meta)
    su = torch.empty(m_pad, dtype=torch.float16, device=meta)
    qidxs = torch.empty(m_pad, n_pad // 8, dtype=torch.int16, device=meta)
    cb = torch.empty(65536, 8, dtype=torch.float16, device=meta)
    empty_i16 = torch.empty(0, dtype=torch.int16, device=meta)
    empty_f16 = torch.empty(0, dtype=torch.float16, device=meta)

    fy = torch.ops.glq.fused_linear(
        x, sv, su, qidxs, cb, 1.0,
        in_features, out_features, n_pad, m_pad, 10, 10,
        empty_i16, empty_f16, 0.0,
        empty_i16, empty_f16, 0.0,
        empty_i16, empty_f16, 0.0,
    )
    assert fy.shape == (B, out_features), f"got {tuple(fy.shape)}"
    assert fy.dtype == torch.float16
    assert fy.device.type == "meta"


@requires_vllm
def test_glq_config_from_config():
    """GLQvLLMConfig should parse bpw and layer_bpw from dict."""
    from glq_vllm.config import GLQvLLMConfig
    cfg = GLQvLLMConfig.from_config({"bpw": 4, "layer_bpw": {"layers.0.q_proj": 2}})
    assert cfg.bpw == 4
    assert cfg.layer_bpw["layers.0.q_proj"] == 2
    assert cfg.get_name() == "glq"
    assert cfg.get_config_filenames() == ["quantize_config.json"]
    assert torch.float16 in cfg.get_supported_act_dtypes()


@requires_vllm
def test_glq_config_block_diagonal_flag():
    """``block_diagonal`` is parsed from config and threaded to the per-layer
    method. It records the e8p RHT layout so the loader sizes the weight
    buffers to match the checkpoint: True = block-diagonal padding (the
    0.6.6+ default), False = legacy full pow2 Hadamard. Absent → True for
    back-compat with the block-diagonal default, NOT the older pow2 uploads
    (which must set it explicitly)."""
    from vllm.model_executor.layers.linear import LinearBase
    from glq_vllm.config import GLQvLLMConfig
    from glq_vllm.linear_method import GLQLinearMethod

    # Absent → defaults to True (block-diagonal e8p).
    cfg = GLQvLLMConfig.from_config({"bpw": 3, "codebook": "e8p"})
    assert cfg.block_diagonal is True

    # Explicit False (legacy pow2 e8p checkpoint).
    cfg = GLQvLLMConfig.from_config({
        "bpw": 3, "codebook": "e8p", "block_diagonal": False,
        "layer_bpw": {"model.layers.0.mlp.down_proj": 3},
    })
    assert cfg.block_diagonal is False

    # Threaded into the per-layer GLQLinearMethod.
    layer = LinearBase.__new__(LinearBase)
    method = cfg.get_quant_method(layer, "model.layers.0.mlp.down_proj")
    assert isinstance(method, GLQLinearMethod)
    assert method.block_diagonal is False
    assert method.codebook_type == "e8p"


@requires_vllm
def test_e8p_create_weights_block_diagonal_sizing():
    """e8p create_weights sizes the Qidxs buffers per the ``block_diagonal``
    flag — block-diag pads to a mult-of-64 (cols) / mult-of-16 (rows), pow2
    pads each dim to the next power of two. A non-pow2 shape distinguishes
    them: 11008 stays 11008 block-diag but inflates to 16384 pow2; 3072 stays
    3072 block-diag but inflates to 4096 pow2. CPU-only (no codebook load)."""
    from glq_vllm.linear_method import GLQLinearMethod, _glq_pad, _e8p_pad

    in_sz, out_sz = 11008, 3072
    assert _e8p_pad(in_sz, 64) == 11008 and _glq_pad(in_sz) == 16384
    assert _e8p_pad(out_sz, 16) == 3072 and _glq_pad(out_sz) == 4096

    def make(block_diagonal):
        m = GLQLinearMethod(None, bpw=3, codebook_type="e8p",
                            block_diagonal=block_diagonal)
        layer = torch.nn.Module()
        m.create_weights(layer, in_sz, [out_sz], in_sz, out_sz,
                         torch.float16)
        return layer

    bd = make(True)
    assert tuple(bd.Qidxs_e8p.shape) == (3072 // 16, 11008 // 64, 8, 4)
    assert bd.glq_n_pad == 11008 and bd.glq_m_pad == 3072

    pw = make(False)
    assert tuple(pw.Qidxs_e8p.shape) == (4096 // 16, 16384 // 64, 8, 4)
    assert pw.glq_n_pad == 16384 and pw.glq_m_pad == 4096


# ── GLQ embedding-method dispatch + dequant equivalence ───────────────

@requires_vllm
def test_glq_embedding_method_registered():
    """get_quant_method returns GLQEmbeddingMethod for VocabParallelEmbedding
    when the prefix is in layer_bpw, else UnquantizedEmbeddingMethod."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod, VocabParallelEmbedding,
    )
    from glq_vllm.config import GLQvLLMConfig
    from glq_vllm.embedding_method import GLQEmbeddingMethod

    cfg = GLQvLLMConfig.from_config({
        "bpw": 4,
        "layer_bpw": {
            # checkpoint-form key, like quantize_model.py emits
            "model.language_model.embed_tokens_per_layer": 4,
        },
    })
    # Avoid full __init__ (needs torch.distributed setup); just satisfy isinstance.
    layer = VocabParallelEmbedding.__new__(VocabParallelEmbedding)

    # Quantized embedding — runtime prefix is the multimodal mm-rewritten form.
    method = cfg.get_quant_method(
        layer, "language_model.model.embed_tokens_per_layer")
    assert isinstance(method, GLQEmbeddingMethod), type(method).__name__
    assert method.bpw == 4

    # Unrelated embedding (e.g. main embed_tokens) — not in layer_bpw → unquant.
    method = cfg.get_quant_method(layer, "language_model.model.embed_tokens")
    assert isinstance(method, UnquantizedEmbeddingMethod), type(method).__name__


@pytest.mark.skipif(
    not _HAS_TRANSFORMERS, reason="transformers not installed"
)
def test_glq_embedding_dequant_matches_hf():
    """``_dequant_embedding_rows`` must reproduce ``E8RHTEmbedding.forward``
    (with embed_scale=1.0, the convention vLLM uses) byte-for-byte.

    Uses synthetic GLQ buffers + a tiny codebook so the test is fast and
    needs no GPU.
    """
    import glq.hf_integration  # noqa: F401 (registers types)
    from glq.codebook import E8ShellCodebook
    from glq.quantized_linear import E8RHTEmbedding, _dequant_embedding_rows

    torch.manual_seed(0)
    vocab, dim = 64, 32
    n_pad = 32  # already pow2
    cb = E8ShellCodebook(device="cpu", verbose=False)

    # Stage-1 only (2bpw equivalent).
    emb = E8RHTEmbedding(num_embeddings=vocab, embedding_dim=dim,
                          embed_scale=1.0)
    emb.set_codebook(cb)
    emb.Qidxs.copy_(torch.randint(
        0, 65536, emb.Qidxs.shape, dtype=torch.int32).to(torch.int16))
    emb.SV.copy_(torch.randn_like(emb.SV))
    emb.Wscale.copy_(torch.rand_like(emb.Wscale) + 0.1)

    ids = torch.tensor([0, 5, 17, 63, 42], dtype=torch.long)
    hf_out = emb(ids)

    direct_out = _dequant_embedding_rows(
        ids,
        emb.Qidxs, emb.SV, emb.Wscale, cb.codebook,
        None, None, None,
        n_pad=n_pad, embedding_dim=dim,
        embed_scale=1.0, out_dtype=hf_out.dtype,
    )
    torch.testing.assert_close(hf_out, direct_out, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(
    not _HAS_TRANSFORMERS, reason="transformers not installed"
)
def test_shell_ple_quant_decode_roundtrip():
    """A PLE chunk quantized with the SHELL codebook via quantize_layer_e8_shell_rht
    (apply_left=False, bpw=4) — the path a Gemma-4 E2B/E4B now takes under
    --codebook e8p — loads into E8RHTEmbedding and reconstructs the original rows.
    Proves shell indices + a shell codebook decode correctly (no Qidxs_e8p, no
    codebook mismatch) end-to-end."""
    import glq.hf_integration  # noqa: F401 (registers types)
    from glq.codebook import E8ShellCodebook
    from glq.quantize_model import quantize_layer_e8_shell_rht
    from glq.quantized_linear import E8RHTEmbedding

    torch.manual_seed(0)
    vocab, dim = 64, 64
    cb = E8ShellCodebook.build(device="cpu", verbose=False)
    W = torch.randn(vocab, dim) * 0.1
    _, arts, _ = quantize_layer_e8_shell_rht(
        W, torch.eye(dim), cb, bpw=4, apply_left=False, block_diagonal=False)
    assert {'Qidxs', 'Qidxs2', 'SV', 'Wscale', 'inv_resid_scale'} <= set(arts)
    assert 'Qidxs_e8p' not in arts

    emb = E8RHTEmbedding(num_embeddings=vocab, embedding_dim=dim, embed_scale=1.0)
    assert emb.Qidxs.shape == arts['Qidxs'].shape  # n_pad matches the quant output
    emb.Qidxs.copy_(arts['Qidxs'])
    emb.Qidxs2.copy_(arts['Qidxs2'])
    emb.SV.copy_(arts['SV'])
    emb.Wscale.fill_(float(arts['Wscale']))          # one chunk -> one scalar, broadcast per-row
    emb.inv_resid_scale.fill_(float(arts['inv_resid_scale']))
    emb.set_codebook(cb, codebook2=cb)               # full-shell stage-2 (4bpw PLE)
    assert emb._n_stages == 2

    out = emb(torch.arange(vocab))
    assert out.shape == (vocab, dim) and torch.isfinite(out).all()
    cos = torch.nn.functional.cosine_similarity(
        out.flatten().float(), W.flatten().float(), dim=0).item()
    assert cos > 0.8, f"reconstruction cosine {cos:.3f} too low"


# ── Test 2b: _lookup_bpw across vLLM prefix transforms ─────────────────

@requires_vllm
def test_lookup_bpw_prefix_forms():
    """Whitelist must match the same logical layer across the prefix forms
    vLLM produces at runtime:

    - text-only ``Gemma4ForCausalLM`` strips ``model.language_model.`` to ``model.``
    - multimodal ``Gemma4ForConditionalGeneration`` rewrites
      ``model.language_model.X`` to ``language_model.model.X``
    - stacked-merge: ``qkv_proj`` matches when any of q/k/v are listed unmerged

    Storing keys in safetensors-checkpoint form (``model.language_model.X``)
    and asking the whitelist to map every runtime form back to it is the
    convention quantize_model writes.
    """
    from glq_vllm.config import GLQvLLMConfig
    cfg = GLQvLLMConfig.from_config({
        "bpw": 4,
        "layer_bpw": {
            # checkpoint-form keys, as quantize_model writes them
            "model.language_model.layers.0.mlp.down_proj": 4,
            "model.language_model.layers.0.self_attn.q_proj": 4,
            "model.language_model.layers.0.self_attn.k_proj": 4,
            "model.language_model.layers.0.self_attn.v_proj": 4,
            "model.language_model.layers.0.per_layer_input_gate": 4,
        },
    })

    # Direct match.
    assert cfg._lookup_bpw("model.language_model.layers.0.mlp.down_proj") == 4
    # Text-only Gemma-4 path: vLLM strips `language_model.` so prefix is
    # `model.layers.0...`. Whitelist re-adds the missing `language_model.`.
    assert cfg._lookup_bpw("model.layers.0.mlp.down_proj") == 4
    # Multimodal Gemma-4 path: vLLM mapper renames
    # `model.language_model.X` -> `language_model.model.X`.
    assert cfg._lookup_bpw("language_model.model.layers.0.mlp.down_proj") == 4
    assert cfg._lookup_bpw(
        "language_model.model.layers.0.per_layer_input_gate") == 4
    # Stacked-merge: vLLM packs q/k/v into qkv_proj at runtime; whitelist
    # finds at least one of the three subnames and returns the max bpw.
    assert cfg._lookup_bpw("model.layers.0.self_attn.qkv_proj") == 4
    assert cfg._lookup_bpw(
        "language_model.model.layers.0.self_attn.qkv_proj") == 4

    # Layer absent from whitelist returns None, so get_quant_method falls
    # through to UnquantizedLinearMethod (e.g. Gemma-4
    # `per_layer_model_projection`, which is bf16 in the checkpoint).
    assert cfg._lookup_bpw(
        "model.language_model.per_layer_model_projection") is None
    assert cfg._lookup_bpw(
        "language_model.model.per_layer_model_projection") is None
    assert cfg._lookup_bpw("model.per_layer_model_projection") is None


# ── Test 3: Dequant correctness (CPU, no vLLM server) ──────────────────

@pytest.mark.skipif(
    not _HAS_TRANSFORMERS, reason="transformers not installed"
)
def test_dequant_matches_hf():
    """dequantize_glq_weight must match E8RHTLinear.dequantize()."""
    import glq.hf_integration  # noqa: F401
    from transformers import AutoModelForCausalLM
    from glq_vllm.dequant import dequantize_glq_weight, get_codebook, get_codebook2

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cpu", torch_dtype=torch.float16
    )
    layer = model.model.layers[0].self_attn.q_proj
    hf_weight = layer.dequantize()

    cb = get_codebook()
    inv_rs = layer.inv_resid_scale.item()
    has_stage2 = inv_rs != 0.0
    cb2 = get_codebook2(4) if has_stage2 else None

    vllm_weight = dequantize_glq_weight(
        layer.Qidxs, layer.SU, layer.SV, layer.Wscale, cb,
        layer.Qidxs2 if has_stage2 else None, inv_rs, cb2,
        layer.out_features, layer.in_features,
    )

    cos = torch.nn.functional.cosine_similarity(
        hf_weight.flatten().float(), vllm_weight.flatten().float(), dim=0
    ).item()
    assert cos > 0.999, f"Dequant cosine similarity {cos:.4f} too low"


# ── Tests 4-7: GPU tests with shared LLM instance ──────────────────────

@requires_vllm
@requires_gpu
def test_glq_vllm_gpu():
    """All GPU tests in one: weights, generation, VRAM, throughput.

    Uses a single LLM instance to avoid repeated 5-min model loads.
    """
    import glq_vllm  # noqa: F401
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_ID,
        tokenizer=TOKENIZER_ID,
        quantization="glq",
        enforce_eager=True,
        dtype="half",
        gpu_memory_utilization=0.9,
        max_model_len=512,
    )

    # --- 4. Weights stay compressed ---
    found_qidxs = [False]
    def check_layers(model):
        for name, mod in model.named_modules():
            if hasattr(mod, "Qidxs"):
                assert mod.Qidxs.dtype == torch.int16, f"{name}.Qidxs is {mod.Qidxs.dtype}"
                assert not hasattr(mod, "weight"), f"{name} has dense weight"
                found_qidxs[0] = True
    llm.apply_model(check_layers)
    assert found_qidxs[0], "No GLQ layers found with Qidxs"

    # --- 5. Generation correctness ---
    params = SamplingParams(max_tokens=30, temperature=0)
    outputs = llm.generate(["The capital of France is"], params)
    text = outputs[0].outputs[0].text
    assert len(text) > 10, f"Output too short: {text!r}"
    assert "Paris" in text, f"Expected 'Paris' in output: {text!r}"

    # --- 6. Throughput ---
    params = SamplingParams(max_tokens=128, temperature=0)
    t0 = time.time()
    outputs = llm.generate(["The theory of relativity states that"], params)
    t1 = time.time()
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tok_per_sec = total_tokens / (t1 - t0)
    assert tok_per_sec >= 10, f"Throughput {tok_per_sec:.1f} tok/s below 10 tok/s minimum"

    del llm
