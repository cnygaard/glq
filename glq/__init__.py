"""
glq — Golay-Leech Quantization

E8 lattice shell codebook + Randomized Hadamard Transform (RHT)
for 2-bit post-training quantization of LLM weights.
"""

__version__ = "0.1.7"

from .codebook import E8ShellCodebook
from .hadamard import fast_hadamard_transform
from .rht import RHT
from .ldlq import block_LDL, quantize_ldlq_codebook, quantize_ldlq_codebook_2stage
from .quantized_linear import E8RHTLinear
from .quantize_model import quantize
