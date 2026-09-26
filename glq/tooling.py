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


#: Architectures that must NOT run in fp16. The failure is a hard NaN from activation
#: outliers exceeding fp16's 65504 range, measured on Ministral-3 (see
#: glq/quantized_linear.py's fp32-accum note and the GLQ fp16-overflow finding). Matched on
#: the repo id, lowercased, because that is all the caller reliably has before load.
#:
#: Why a DENYLIST here when `tool_serve_args` above deliberately refuses unknown families:
#: that rule exists because a wrong tool parser "fails silently at the worst layer". A wrong
#: dtype does not fail silently -- fp16 overflow produces NaN and visibly garbage output on
#: the first token. Loud failure justifies opting everything in; silent failure does not.
FP16_UNSAFE = (
    # Activation outliers exceed fp16's 65504 range -> hard NaN. The observed case.
    "ministral", "mistral", "devstral",
    # PROVISIONAL, not a measured fp16 failure. Qwen3.5-2B-4bpw raised
    # "expected mat1 and mat2 to have the same dtype ... c10::BFloat16" under fp16, but its
    # bf16 CONTROL is also broken (wikitext PPL 2.9e6, no usable recorded baseline), so
    # neither arm is evidence about fp16 and the family stays on bf16 until a working
    # verification vehicle exists. Suspected cause is NOT arch-level: hf_integration derives
    # `_compute_dtype` from cfg.torch_dtype rather than the REQUESTED dtype, so a quantized
    # embedding emits bf16 into an fp16 graph. If that is fixed and a Qwen checkpoint
    # verifies, delete this line rather than special-casing further.
    "qwen",
)

#: The dtype each device's GLQ kernels already compute in, so there is no conversion at the
#: quantized-layer boundary. Measured 2026-09-26, decode tok/s, repeats 5, swap 0:
#:
#:   CUDA  gemma-4-26B-A4B 3bpw, RTX PRO 6000: fp16 9.370 > fp32 9.224 > bf16 8.304
#:         fp16 and bf16 have IDENTICAL peak alloc (16.46 GiB), so fp16 is free.
#:   CPU   Qwen3.8-Flash-Next 3bpw, Xeon 8559C: fp32 3.072 > fp16 3.017 > bf16 2.803
#:         fp16 costs +6.4 GiB RSS over bf16 (89.11 vs 82.74), so it is a trade, not free.
#:
#: bf16 loses on BOTH paths because it is aligned with neither kernel path and pays
#: conversions everywhere. fp16 is preferred over fp32 on CPU despite being marginally
#: slower, because fp32 costs a further +4.4 GiB and footprint is this project's point.
_ALIGNED_DTYPE = {"cuda": "float16", "cpu": "float16"}


def preferred_dtype(model_id: str, device: str = "cuda") -> str:
    """The dtype to serve `model_id` with on `device`.

    fp16 unless the model is in :data:`FP16_UNSAFE`, in which case bf16 -- which is slower
    on every path measured, and correct where fp16 would NaN.

    Quality is not traded for the speed: wikitext-2 PPL is marginally BETTER in fp16 than
    bf16 (SmolLM3-3B trellis 6bpw: 9.1145 vs 9.1337), because fp16 carries 10 mantissa bits
    against bf16's 7. bf16 buys exponent range by spending precision, so where activations
    fit fp16's range fp16 is the more accurate of the two and the only risk is overflow.
    """
    if any(tag in model_id.lower() for tag in FP16_UNSAFE):
        return "bfloat16"
    return _ALIGNED_DTYPE.get(device, "float16")
