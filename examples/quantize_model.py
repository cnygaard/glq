#!/usr/bin/env python3
"""
Quantize a HuggingFace model to GLQ format.

GLQ (E8 Lattice Quantization) compresses LLM weights using the E8 lattice
codebook — 65,536 vectors in 8 dimensions. Each 8-weight block maps to a
16-bit index, achieving exactly 2.0 bits per weight at the base level.
For higher quality, two-stage residual quantization (RVQ) adds a secondary
codebook to reach 3.0 or 4.0 bits per weight.

Examples:
    # Quantize SmolLM3-3B at 3.5 bits per weight (mixed 2+4 bpw)
    python quantize_model.py \\
        --model HuggingFaceTB/SmolLM3-3B \\
        --output ./SmolLM3-3B-GLQ-3.5bpw \\
        --bpw 3.5 --min-bpw 2 --max-bpw 4

    # Quantize at uniform 4 bits per weight
    python quantize_model.py \\
        --model HuggingFaceTB/SmolLM3-3B \\
        --output ./SmolLM3-3B-GLQ-4bpw \\
        --bpw 4

    # Quantize a large model (>30B) using streaming mode to avoid OOM
    python quantize_model.py \\
        --model mistralai/Devstral-Small-2-24B-Instruct-2512 \\
        --output ./Devstral-24B-GLQ-4bpw \\
        --bpw 4 --streaming

    # Quantize a model with custom code (e.g., NemotronH)
    python quantize_model.py \\
        --model nvidia/Nemotron-3-Nano-30B-A3B \\
        --output ./Nemotron-30B-GLQ-4bpw \\
        --bpw 4 --streaming --trust-remote-code
"""

import argparse
from glq.quantize_model import main

if __name__ == "__main__":
    main()
