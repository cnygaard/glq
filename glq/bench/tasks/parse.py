"""Pure parsing helpers shared by the task adapters (no torch/vllm import).

Answer extraction is lifted verbatim from the session harnesses
(`benchmarks/_aime_multiyear.py`, `_mmlu_think_sweep_glq.py`) so re-quantized
results stay comparable to the hand-run numbers. The vLLM log/stdout parsers
turn engine output into the structured fields the record schema wants. All
unit-testable on CPU.
"""
from __future__ import annotations

import re

# ---- AIME: last \boxed{N}, then loosening fallbacks --------------------------
_BOX = re.compile(r"\\boxed\{\s*([0-9]{1,3})\s*\}")
_AIME_FALLBACK = [
    r"answer is[:\s]*\$?\(?([0-9]{1,3})\b",
    r"final answer[^0-9]{0,12}([0-9]{1,3})\b",
    r"\b([0-9]{1,3})\s*$",
]


def extract_boxed_int(text: str) -> int | None:
    ms = list(_BOX.finditer(text))
    if ms:
        return int(ms[-1].group(1))
    for p in _AIME_FALLBACK:
        ms = list(re.finditer(p, text, re.IGNORECASE | re.MULTILINE))
        if ms:
            return int(ms[-1].group(1))
    return None


# ---- MMLU-Pro: "The answer is (X)." letter, with fallbacks -------------------
_MMLU_PATS = [
    r"answer is[:\s]*\(([A-P])\)",
    r"answer is[:\s]*([A-P])\b",
    r"\\boxed\{\(?([A-P])\)?\}",
    r"\banswer[:\s]*\(?([A-P])\)?",
    r"\(([A-P])\)\s*$",
    r"\b([A-P])\)\s*$",
]


def extract_mmlu_letter(text: str) -> str | None:
    for p in _MMLU_PATS:
        ms = list(re.finditer(p, text, re.IGNORECASE | re.MULTILINE))
        if ms:
            return ms[-1].group(1).upper()
    return None


# ---- vLLM engine log: GPU memory at load -------------------------------------
# Matches: "Model loading took 16.51 GiB memory and 54.7 seconds"
_LOAD_MEM = re.compile(r"Model loading took\s+([0-9.]+)\s*GiB", re.IGNORECASE)


def parse_load_mem_gib(log_text: str) -> float | None:
    ms = list(_LOAD_MEM.finditer(log_text))
    return float(ms[-1].group(1)) if ms else None


# ---- `vllm bench throughput` stdout: output tok/s ----------------------------
# vLLM prints lines like "Output token throughput (tok/s): 430.56" or
# "Throughput: 12.34 requests/s, 1234.5 total tokens/s, 430.6 output tokens/s".
_TP_LABELLED = re.compile(
    r"[Oo]utput token throughput\s*\(tok/?s\)\s*[:=]\s*([0-9.]+)")
_TP_INLINE = re.compile(r"([0-9.]+)\s*output tokens?/s")
_TP_TOTAL = re.compile(r"([0-9.]+)\s*total tokens?/s")


def parse_vllm_bench_throughput(stdout: str) -> dict[str, float | None]:
    """Return ``{output_tok_s, total_tok_s}`` parsed from `vllm bench` stdout."""
    out: dict[str, float | None] = {"output_tok_s": None, "total_tok_s": None}
    m = _TP_LABELLED.search(stdout) or _TP_INLINE.search(stdout)
    if m:
        out["output_tok_s"] = float(m.group(1))
    m = _TP_TOTAL.search(stdout)
    if m:
        out["total_tok_s"] = float(m.group(1))
    return out
