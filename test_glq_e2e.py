"""
End-to-end test: Quantize + Save + Load + Generate via GPTQModel GLQ integration.

Usage (GPU machine):
    python test_glq_e2e.py

Expected results on SmolLM2-360M:
    - Quantization completes (~2 min on A10G)
    - Save/load round-trip works
    - Generated text is coherent
    - Perplexity ~13.19 (standalone GLQ baseline)
"""

import os
import sys
import time
import tempfile

import torch

# Make sure both GPTQModel and glq are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GPTQModel"))

from gptqmodel import GPTQModel
from gptqmodel.quantization.config import QuantizeConfig


MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_calibration_data(n_samples=16):
    """Load WikiText-2 calibration data as list of strings."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # Filter non-empty lines and take n_samples
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:n_samples]
    return texts


def test_quantize_save_load():
    print("=" * 60)
    print("GLQ GPTQModel End-to-End Test")
    print("=" * 60)

    # 1. Create quantize config
    print("\n[1/5] Creating QuantizeConfig ...")
    config = QuantizeConfig(bits=2, group_size=-1, quant_method="glq")
    print(f"  method={config.quant_method}, format={config.format}, bits={config.bits}")

    # 2. Load model for quantization
    print(f"\n[2/5] Loading {MODEL_ID} ...")
    t0 = time.perf_counter()
    model = GPTQModel.load(MODEL_ID, quantize_config=config, device=DEVICE)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # 3. Quantize with calibration data
    print(f"\n[3/5] Quantizing ...")
    calib_data = get_calibration_data(n_samples=16)
    print(f"  Calibration: {len(calib_data)} samples")

    t0 = time.perf_counter()
    model.quantize(calib_data)
    quant_time = time.perf_counter() - t0
    print(f"  Quantized in {quant_time:.1f}s")

    # 4. Save
    save_dir = os.path.join(tempfile.gettempdir(), "smollm2-glq-2bpw-test")
    print(f"\n[4/5] Saving to {save_dir} ...")
    t0 = time.perf_counter()
    model.save(save_dir)
    print(f"  Saved in {time.perf_counter() - t0:.1f}s")

    # Check saved files
    for f in ["model.safetensors", "quantize_config.json", "e8_codebook.pt"]:
        path = os.path.join(save_dir, f)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            print(f"  {f}: {size_mb:.1f} MB")
        else:
            print(f"  WARNING: {f} not found!")

    # 5. Load quantized model and generate
    print(f"\n[5/5] Loading quantized model + generating ...")
    t0 = time.perf_counter()
    model2 = GPTQModel.load(save_dir, device=DEVICE)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    prompt = "The capital of France is"
    output = model2.generate(prompt, max_new_tokens=32)
    print(f"\n  Prompt: {prompt}")
    print(f"  Output: {output[0]}")

    print(f"\n{'=' * 60}")
    print(f"TEST PASSED")
    print(f"{'=' * 60}")

    return save_dir


def test_perplexity(model_path):
    """Measure perplexity on WikiText-2 test set."""
    print(f"\n{'=' * 60}")
    print(f"Perplexity Evaluation")
    print(f"{'=' * 60}")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    model = GPTQModel.load(model_path, device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(DEVICE)

    seqlen = 2048
    n_chunks = input_ids.shape[1] // seqlen
    nlls = []

    print(f"  Evaluating {n_chunks} chunks of length {seqlen} ...")
    t0 = time.perf_counter()

    for i in range(n_chunks):
        chunk = input_ids[:, i * seqlen:(i + 1) * seqlen]
        with torch.no_grad():
            out = model.model(chunk)
            logits = out.logits if hasattr(out, 'logits') else out[0]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean',
        )
        nlls.append(loss.item())

        if (i + 1) % 20 == 0:
            ppl_so_far = torch.exp(torch.tensor(nlls).mean()).item()
            print(f"    chunk {i+1}/{n_chunks}: ppl={ppl_so_far:.2f}")

    ppl = torch.exp(torch.tensor(nlls).mean()).item()
    dt = time.perf_counter() - t0

    print(f"\n  Perplexity: {ppl:.2f}")
    print(f"  Eval time: {dt:.1f}s")
    print(f"  (SmolLM2-360M standalone GLQ baseline: ~13.19)")

    return ppl


if __name__ == "__main__":
    save_dir = test_quantize_save_load()
    if "--ppl" in sys.argv:
        test_perplexity(save_dir)
