"""Family-aware vLLM tool-calling serve args — one source for knowledge that drifted.

pi (and any tool-using client) needs the server started with `--enable-auto-tool-choice`
and a parser that matches the model's tool markup. Which parser — and for gemma-4, which
*template*, since its bundled chat template is plain chat and the tool template lives in
vLLM's repo examples — used to be duplicated between the installer's printed commands and
the bench harness, and drifted once already: hermes was printed for every model, which
matches SmolLM3/Qwen-style ``<tool_call>`` markup and silently mangles gemma-4's into
tool calls that never parse. That is the worst failure mode, because it reads as a bad
model rather than a bad flag.

Stdlib only: the installer's core profile has no huggingface_hub and no requests.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

GEMMA4_TOOL_TEMPLATE = "tool_chat_template_gemma4.jinja"
#: The version pin matters: templates track vLLM's parser expectations, and this pairing
#: is what the README's tool-calling recipe was validated against.
GEMMA4_TOOL_TEMPLATE_URL = ("https://raw.githubusercontent.com/vllm-project/vllm/"
                            f"v0.20.2/examples/{GEMMA4_TOOL_TEMPLATE}")


def _default_templates_dir() -> Path:
    return Path(os.environ.get("GLQ_HOME", Path.home() / ".glq")) / "templates"


def tool_serve_args(model_id: str, templates_dir=None) -> list[str] | None:
    """vLLM serve args that make tool calling work for this model's family.

    None for an unknown family — the caller must refuse rather than guess, because a
    wrong parser fails silently at the worst layer.
    """
    name = model_id.lower()
    if "gemma-4" in name:
        tpl = Path(templates_dir or _default_templates_dir()) / GEMMA4_TOOL_TEMPLATE
        return ["--enable-auto-tool-choice",
                "--tool-call-parser", "gemma4",
                "--reasoning-parser", "gemma4",
                "--chat-template", str(tpl),
                # Without this the template never opens a thought section, but the
                # RL-trained model thinks anyway — measured live in a pi session:
                # <|thought|> markers and tool-call syntax leaking into prose, and a
                # "thoughtthoughtthought" repetition loop in the reasoning field. The
                # README's validated recipe always carried it. Compact JSON (no spaces)
                # keeps the printed shell command copy-pasteable without quoting.
                "--default-chat-template-kwargs", '{"enable_thinking":true}']
    if "qwen" in name:
        # hermes matches Qwen's <tool_call> markup, but Qwen3.x are THINKING models:
        # without --reasoning-parser qwen3 the <think> block stays in `content` and
        # leaks into the agent's prose — the same failure class as gemma-4's
        # enable_thinking leak (#85), and in a pi session it reads as rambling or
        # repetition. Parser name per vLLM's official Qwen3.5 recipe
        # (vllm-project/recipes Qwen/Qwen3.5.md). --language-model-only skips the
        # multimodal wrapper's bf16 vision tower, which text-only agent/chat serving
        # never uses — pure VRAM back on 24 GB cards.
        return ["--enable-auto-tool-choice", "--tool-call-parser", "hermes",
                "--reasoning-parser", "qwen3",
                "--language-model-only"]
    if "smollm3" in name:
        # SmolLM3 emits hermes-style <tool_call> markup too; SmolLM3+hermes WITHOUT a
        # reasoning parser is the pairing the Terminal-Bench integration validated end
        # to end — don't drift it as a side effect of Qwen changes.
        return ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    return None


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as resp:            # noqa: S310 - pinned https
        return resp.read()


def ensure_gemma4_template(templates_dir=None, fetch=_fetch) -> Path:
    """The cached gemma-4 tool template, fetching it if this install never got one.

    The installer's picode component downloads it, but glq-code must also work on
    installs that predate that or skipped the component. A failed fetch raises with the
    manual command — "it failed" without "do this instead" strands the user exactly
    where the missing-template ValueError from vLLM did.
    """
    tpl = Path(templates_dir or _default_templates_dir()) / GEMMA4_TOOL_TEMPLATE
    if tpl.exists():
        return tpl
    try:
        data = fetch(GEMMA4_TOOL_TEMPLATE_URL)
        tpl.parent.mkdir(parents=True, exist_ok=True)
        tpl.write_bytes(data)
    except OSError as exc:
        raise RuntimeError(
            f"could not fetch gemma-4's tool template ({exc}); get it yourself:\n"
            f"  curl --proto '=https' --tlsv1.2 -fsSL {GEMMA4_TOOL_TEMPLATE_URL} "
            f"-o {tpl}") from exc
    return tpl


#: Architectures that must NOT run in fp16. Matched on the repo id, lowercased, because that
#: is all the caller reliably has before load.
#:
#: Why a DENYLIST here when `tool_serve_args` above deliberately refuses unknown families:
#: that rule exists because a wrong tool parser "fails silently at the worst layer". A wrong
#: dtype has THREE failure modes, and only the third is silent:
#:
#:   1. loud  -- fp16 overflow: NaN, visibly garbage on the first token (the Mistral case).
#:   2. loud  -- the architecture refuses the dtype outright and the engine will not start
#:               (the Qwen4Exp case below).
#:   3. SILENT -- a kernel that wanted bf16 quietly picks a slower implementation. This one
#:               is real: `qwen_gdn_linear_attn.py:535` drops the fused CUDA GDN decode
#:               kernel when `model_config.dtype != torch.bfloat16`, and since
#:               VLLM_GDN_DECODE_KERNEL defaults to "cuda" while its raise fires only when
#:               the env var was set explicitly, the default path is a `logger.info_once`.
#:               `cpu_moe.py:466/502/535` do the same via `return False`.
#:
#: Mode 3 was the reason to doubt a denylist, so it was measured rather than argued: forcing
#: the Triton GDN path on Qwen3.8-Flash-Next-GLQ-3bpw (both arms bf16, sm_120, vLLM 0.29.0,
#: decode 64, repeats 3) cost -0.26% at B=1 -- 32.035 vs 31.951 tok/s with overlapping
#: ranges, and triton marginally AHEAD at B=8.
#:
#: A null that small is indistinguishable from a flag that does nothing, and vLLM's init-time
#: "GDN decode kernel: …" line proves only that the flag was SET. So the substitution was
#: confirmed at the op level (`benchmarks/_gdn_cuda_engagement.py`, in-process + eager so the
#: calls are visible; a FULL cudagraph replay does not re-enter Python):
#:
#:   kernel=cuda    qwen_gdn_attention_core 0 calls   ..._fused_norm_packed 1536 calls
#:   kernel=triton  qwen_gdn_attention_core 1536      ..._fused_norm_packed 0
#:
#: 1536 = 48 GDN layers x 32 decode tokens: a complete swap, so the flag does change what
#: runs. The op is **0.70-0.82% of total device time**, which caps any end-to-end effect at
#: ~0.8% and explains the null instead of merely reporting it. On a GLQ checkpoint decode is
#: dominated by the trellis dequant+matvec, not the GDN recurrence -- the same reason the CPU
#: GDN kernel's in-situ share is ~0%. So the silent mode exists but is cheap, and that -- not
#: its nonexistence -- is what makes a denylist acceptable here.
FP16_UNSAFE = (
    # Activation outliers exceed fp16's 65504 range -> hard NaN. The observed case; see
    # glq/quantized_linear.py's fp32-accum note and the GLQ fp16-overflow finding.
    "ministral", "mistral", "devstral",
    # vLLM REFUSES fp16 for Qwen4Exp: `models/qwen4_exp/nvidia/qsa.py` and `indexer_qsa.py`
    # raise NotImplementedError("Qwen4Exp QSA ... requires BF16") in four places. Confirmed
    # on 0.29.0 -- the bf16 arm scored wikitext PPL 4.8917 against 4.8898 on record (0.04%,
    # so the harness was sound) while the fp16 arm exited on that raise. An upstream
    # architecture constraint, not a GLQ bug and not an overflow.
    #
    # The entry is a family substring, and it stays that broad because fp16 buys the OTHER
    # published Qwen nothing. Both are GDN hybrids (`qwen3_5.py:145` and qwen4_exp both build
    # QwenGatedDeltaNetAttention), and a GDN model in fp16 silently forfeits the fused CUDA
    # GDN decode kernel -- which cancels the conversion saving exactly. Measured on
    # Qwen3.8-27B-GLQ-4bpw (qwen3_5: GDN, no QSA, so fp16 is not refused), sm_120,
    # vLLM 0.29.0, B=1, decode 64, repeats 3:
    #
    #   bf16  39.509 tok/s (39.452-40.111)  GDN decode kernel: cuda
    #   fp16  39.472 tok/s (39.434-39.801)  GDN decode kernel: triton
    #
    # Overlapping ranges, both samples coherent. So fp16 is SAFE here but pointless, and
    # narrowing this to qwen4_exp would add a special case to buy a measured tie. That run
    # also shows the earlier Qwen3.5-2B fp16 failure ("expected mat1 and mat2 to have the same
    # dtype ... c10::BFloat16", against a bf16 control that was itself broken at PPL 2.9e6)
    # was NOT arch-level -- do not cite it as evidence for anything.
    #
    # Known gap, not an oversight: non-Qwen models import the same GDN class
    # (`interns2_mobius.py`), so a substring cannot catch them. Acceptable because mode 3 is
    # what they would hit, and mode 3 measured as a null.
    "qwen",
)

#: The dtype to serve with, per device. The two entries are chosen for DIFFERENT reasons and
#: must not be collapsed into one value. Measured 2026-09-26, decode tok/s, repeats 5, swap 0:
#:
#:   CUDA  gemma-4-26B-A4B 3bpw, RTX PRO 6000: fp16 9.370 > fp32 9.224 > bf16 8.304
#:         fp16 and bf16 have IDENTICAL peak alloc (16.46 GiB), so fp16 is free.
#:   CPU   Qwen3.8-Flash-Next 3bpw, Xeon 8559C: fp32 3.072 @ ~93.5 GiB
#:                                              fp16 3.017 @ 89.11
#:                                              bf16 2.803 @ 82.74
#:
#: **CUDA: fp16 because that is what the kernels compute in.** Every CUDA entry point casts
#: (`quantized_linear.py:772/795/917/925/934`, `x2d.half()`), so bf16 pays a conversion at
#: every quantized-layer boundary for nothing. +12.8% at equal footprint.
#:
#: **CPU: bf16, and NOT by the same argument.** The CPU kernels compute in fp32
#: (`quantized_linear.py:888`, `:1082`), so no 16-bit dtype is aligned there and the choice
#: is a footprint/speed trade. fp16 is a genuine Pareto middle -- it was picked here
#: originally on that basis -- but it is the wrong point on the frontier: +6.4 GiB buys +7.6%
#: on the one platform where exhausting RAM costs far more than 7.6%, and two findings on
#: this project were invalidated outright by swap contamination. bf16 is additionally the
#: only 16-bit dtype vLLM's CPU kernels accept: cpu_moe.py:466/502/535 (bf16 only, a silent
#: downgrade to the per-expert loop), scaled_mm/cpu.py:257 (bf16/fp32), and a hard assert in
#: mamba/ops/cpu/gdn_attention.py:54.
_DEFAULT_DTYPE = {"cuda": "float16", "cpu": "bfloat16"}


def preferred_dtype(model_id: str, device: str = "cuda") -> str:
    """The dtype to serve `model_id` with on `device`.

    See :data:`_DEFAULT_DTYPE` for why the devices differ, and :data:`FP16_UNSAFE` for the
    exceptions. An unrecognised device gets bf16: it has not been measured, and bf16 is what
    the checkpoint declares, so it is the answer that cannot be worse than today's.

    On CUDA, quality is not traded for the speed: wikitext-2 PPL is marginally BETTER in fp16
    than bf16 (SmolLM3-3B trellis 6bpw: 9.1145 vs 9.1337), because fp16 carries 10 mantissa
    bits against bf16's 7. bf16 buys exponent range by spending precision, so where
    activations fit fp16's range fp16 is the more accurate of the two and the only risk is
    overflow.
    """
    if any(tag in model_id.lower() for tag in FP16_UNSAFE):
        return "bfloat16"
    return _DEFAULT_DTYPE.get(device, "bfloat16")
