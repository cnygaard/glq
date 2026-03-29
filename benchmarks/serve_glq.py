#!/usr/bin/env python
"""Wrapper to run `vllm serve` with GLQ plugin registered.

Usage:
    python serve_glq.py serve <model> [vllm args...]
    python serve_glq.py bench serve [vllm bench args...]
"""
import os
import sys

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import glq_vllm  # noqa: F401 — registers "glq" quantization

from vllm.scripts import main

if __name__ == "__main__":
    sys.argv[0] = "vllm"
    main()
