"""Per-model serving settings: one table, and the lookups every launcher reads it through.

What a model needs from vLLM used to live in seven tables here, each with its own substring
loop, plus an eighth in `supervisor.kv_cost`; glq-chat and glq-code then each assembled their
own subset by hand. Every family is now one `ModelProfile` in `PROFILES`, and each public
function below is a view onto it, so a family is added, and its evidence read, in one place.

Why it is worth one place: this knowledge drifted twice when it was duplicated. First, hermes
was printed for every model, which silently mangled gemma-4's markup into tool calls that never
parse. Then hermes stayed on Qwen on the belief that Qwen emits "hermes-style ``<tool_call>``"
markup — false for every Qwen this project publishes, whose templates emit
``<function=``/``<parameter=`` XML (see the ``qwen`` profile for the per-checkpoint evidence).
Both are the worst failure mode, because they read as a bad model rather than a bad flag.

**Two lookup scopes.** `profile_for(model_id, scope)`:

  "template"  the repo id (lowercased) plus `_DERIVED_FAMILIES` aliases. For facts about ONE
              checkpoint's chat template and card: parsers, sampling, drafter. A fine-tune can
              change its template, so a config's `model_type` must never pick a tool parser.
  "arch"      the above plus a local checkpoint's `config.json` model_type/architectures, and
              each profile's `config_match` spellings. For facts about the architecture under
              vLLM: dtype. A box copy is named by whoever copied it, and its config still says
              what it is.

Matching is first-match over `PROFILES`, which is ordered most specific first. The fp16
denylist is the one any-match lookup (`preferred_dtype`), because there bf16 is the safe
direction.

Stdlib only: the installer's core profile has no huggingface_hub and no requests.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

GEMMA4_TOOL_TEMPLATE = "tool_chat_template_gemma4.jinja"
#: The version pin matters: templates track vLLM's parser expectations, and this pairing
#: is what the README's tool-calling recipe was validated against.
GEMMA4_TOOL_TEMPLATE_URL = ("https://raw.githubusercontent.com/vllm-project/vllm/"
                            f"v0.20.2/examples/{GEMMA4_TOOL_TEMPLATE}")


def _default_templates_dir() -> Path:
    return Path(os.environ.get("GLQ_HOME", Path.home() / ".glq")) / "templates"


#: Fine-tunes whose name does not carry their family, as `{substring: family}`. Every lookup
#: in this module matches on the name, so without an entry here such a model is an unknown
#: family: glq-code refuses it and glq-chat serves it with none of the family's settings.
#:
#: An entry is evidence about ONE checkpoint, not a rule about an architecture. Mapping by
#: `config.json`'s `model_type` instead would treat every fine-tune of the architecture as
#: using the base's tool markup, and a fine-tune can change its template, which is the silent
#: failure `tool_serve_args` refuses to risk. So each entry names what was compared.
#:
#:   arete-27b  NousResearch/Arete-27B: Qwen3.8-27B after supervised fine-tuning, same
#:              `Qwen3_5ForConditionalGeneration`. Chat template compared with Qwen3.8-27B's
#:              on 2026-11-08: identical `<tool_call>` / `<function=` / `<think>` markup; the
#:              only change drops the `{% generation %}` training-mask markers. Its
#:              generation_config.json is Qwen's 1.0 / 0.95 / top_k 20.
_DERIVED_FAMILIES = {
    "arete-27b": "qwen",
}


def _family_name(model_id: str) -> str:
    """The "template" scope's text: the id, lowercased, plus the family of any fine-tune in
    `_DERIVED_FAMILIES` whose name does not say it."""
    name = (model_id or "").lower()
    extra = [family for alias, family in _DERIVED_FAMILIES.items() if alias in name]
    return " ".join([name, *extra])


def _family_text(model_id: str) -> str:
    """The "arch" scope's text: the id, plus — for a local checkpoint — the family it declares
    about itself.

    The id alone fails exactly where it matters most. A checkpoint on a box is served from a
    directory named by whoever copied it there (`/opt/dlami/nvme/lightning30b-glq-3bpw`
    carries no family tag), so every NemotronH run on 2026-10-27 got fp16 and lost its
    reasoning without a word. `config.json`'s `model_type` and `architectures` name the
    family regardless of the directory name. Hub ids are left as they are: resolving them
    would mean a network call inside a function that decides a CLI default.
    """
    text = _family_name(model_id)
    cfg_path = os.path.join(model_id, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            return text
        parts = [str(cfg.get("model_type", ""))]
        parts += [str(a) for a in cfg.get("architectures") or []]
        text += " " + " ".join(parts).lower()
    return text


@dataclass(frozen=True, eq=False)
class ModelProfile:
    """What one family needs from vLLM, and what its card asks for.

    A `match` key is a lowercase substring, or a tuple of substrings that must ALL appear.
    `config_match` keys are matched only in the "arch" scope (see the module docstring).
    """
    family: str
    match: tuple = ()
    config_match: tuple = ()
    #: vLLM tool calling, for glq-code. None: glq-code refuses the family rather than guess.
    tool_parser: str | None = None
    reasoning_parser: str | None = None
    #: A tool template that ships in neither the checkpoint nor the vLLM wheel (gemma-4),
    #: by file name under the templates dir.
    chat_template: str | None = None
    chat_template_kwargs: dict | None = None
    #: The chat template thinks unless told not to, so glq-chat needs the reasoning parser.
    thinks_by_default: bool = False
    #: `--language-model-only` for a multimodal wrapper served text-only.
    text_only: bool = False
    #: The model card's sampling; only parameters that are NOT already vLLM's default.
    sampling: dict | None = None
    #: `(importable module, pip command)` of a vLLM plugin this family needs.
    plugin: tuple | None = None
    dspark_drafter: str | None = None
    #: `(fixed bytes per request, bytes per token)`, MEASURED, for `supervisor.kv_cost`.
    kv_cost: tuple | None = None
    #: Never serve this family in fp16.
    fp16_unsafe: bool = False
    #: A vLLM compilation config this family must be served with on CUDA, because vLLM's
    #: default gives wrong output for it. Every launcher passes it: the supervisor (glq-chat,
    #: glq-code), the installer's printed command, the model card, glq-bench's own servers.
    compilation_config: dict | None = None


_GEMMA4 = ModelProfile(
    family="gemma-4", match=("gemma-4",),
    tool_parser="gemma4", reasoning_parser="gemma4",
    chat_template=GEMMA4_TOOL_TEMPLATE,
    # Without this the template never opens a thought section, but the RL-trained model
    # thinks anyway — measured live in a pi session: <|thought|> markers and tool-call syntax
    # leaking into prose, and a "thoughtthoughtthought" repetition loop in the reasoning
    # field. The README's validated recipe always carried it.
    #
    # NOT `thinks_by_default`: the template defaults `enable_thinking` to false, which is why
    # glq-code has to pass this to make it think at all, and why glq-chat gets no reasoning
    # block from it.
    chat_template_kwargs={"enable_thinking": True},
    # The values glq-chat was already shipping.
    sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    # gemma-4 is NOT `text_only`, although the flag works there (vLLM 0.32.0,
    # `benchmarks/hw_tiers/`): 26B-A4B 4 bpw loaded 13.32 GiB with it against 14.40 without
    # (decode 97.1 against 97.6 tok/s), and E4B 5.39 against 6.29. Its towers have always been
    # served, and the planner does not yet subtract them, so turning it on is a separate change.
)

_QWEN = ModelProfile(
    family="qwen", match=("qwen",),
    # `qwen3_xml`, NOT hermes. hermes was carried here for a long time on the belief that
    # Qwen emits "hermes-style <tool_call> markup", and that is false for every Qwen this
    # project publishes — wrong in the silent direction, which is the exact failure this
    # module exists to prevent. hermes reads JSON inside <tool_call>; all five published Qwen
    # checkpoints emit XML tags. From their own chat templates:
    #
    #   Qwen3.5-0.8B / Qwen3.5-2B / Qwen3.8-27B (3 and 4 bpw) / Qwen3.8-Flash-Next
    #       <function= x5     "arguments" x0
    #   SmolLM3 (for contrast)  <function= x0     "arguments" x2   -> hermes is right
    #
    # Flash-Next's template both instructs the format ("If you choose to call a function
    # ONLY reply in the following format ... <tool_call>\n<function=example_function_name>
    # \n<parameter=...") and renders past calls in it, and vLLM's own matcher expects
    # "\n\n<tool_call>" — the same bytes the template emits.
    #
    # `qwen3_xml` and `qwen3_coder` are ALIASES in vLLM: `tool_parsers/__init__.py` maps both
    # to `Qwen3EngineToolParser`, which wraps the single `Qwen3Parser` whose docstring is that
    # `<tool_call>/<function=/<parameter=` grammar. `qwen3_xml` is the name in Qwen's own
    # serving recipe for Qwen3.8-27B, so that is the one carried here.
    tool_parser="qwen3_xml",
    # Qwen3.x are THINKING models: without --reasoning-parser qwen3 the <think> block stays in
    # `content` and leaks into the agent's prose — the same failure class as gemma-4's
    # enable_thinking leak (#85), reading as rambling or repetition in a pi session.
    reasoning_parser="qwen3", thinks_by_default=True,
    # The bf16 vision tower — VRAM back on 24 GB cards, and why the vendor recipe's
    # `--mm-encoder-tp-mode` has nothing to act on here.
    text_only=True,
    # Qwen3.x thinking-mode card: 1.0 / 0.95 / 20. The 20 is the whole reason the sampling
    # table exists — a 64 borrowed from gemma-4 is a different sampler.
    sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    # vLLM REFUSES fp16 for Qwen4Exp: `models/qwen4_exp/nvidia/qsa.py` and `indexer_qsa.py`
    # raise NotImplementedError("Qwen4Exp QSA ... requires BF16") in four places. Confirmed
    # on 0.29.0 -- the bf16 arm scored wikitext PPL 4.8917 against 4.8898 on record (0.04%,
    # so the harness was sound) while the fp16 arm exited on that raise. An upstream
    # architecture constraint, not a GLQ bug and not an overflow.
    #
    # The key is the family substring, and it stays that broad because fp16 buys the OTHER
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
    # (`interns2_mobius.py`), so a substring cannot catch them. Acceptable because the silent
    # mode is what they would hit, and it measured as a null (see `preferred_dtype`).
    fp16_unsafe=True,
)

_SMOLLM3 = ModelProfile(
    family="smollm3", match=("smollm3",),
    # SmolLM3 emits hermes-style <tool_call> markup; SmolLM3+hermes WITHOUT a reasoning parser
    # is the pairing the Terminal-Bench integration validated end to end — don't drift it as a
    # side effect of Qwen changes.
    tool_parser="hermes",
    # SmolLM3's card asks for 0.6 and specifies NO top_k, so none is emitted. Leaving the key
    # out is not the same as sending 0: it keeps whatever the checkpoint's own
    # generation_config.json says, which is the thing we have no reason to overrule.
    sampling={"temperature": 0.6, "top_p": 0.95},
)

_KOLIBRI = ModelProfile(
    family="kolibri", match=("kolibri",),
    # Both names are registered by the `aleph-alpha-inference` PLUGIN, not by vLLM — see
    # `required_plugin` for the consequence. A vanilla vLLM has neither parser, and cannot load
    # `Kolibri1ForCausalLM` natively either (upstream `registry.py` routes it to the
    # transformers backend).
    #
    # The two are NOT symmetric. The reasoning parser is genuinely the plugin's
    # (`Kolibri1ParserReasoningAdapter`, which reads `reasoning_effort`); the tool parser is a
    # lazy ALIAS to vLLM's own `Hermes2ProToolParser`, with the plugin's comment "Kolibri 1
    # currently shares the Hermes <tool_call>...</tool_call> format". So `hermes` would behave
    # identically today — and is still the wrong thing to write here, because that alias is
    # exactly where Aleph Alpha would change it if the markup diverges, and `kolibri1` is what
    # the plugin's README and the published model card both document.
    #
    # No `chat_template_kwargs`, unlike gemma-4: the plugin has thinking ON by default and it is
    # turned OFF per request (`reasoning_effort: "none"`), so a server-side default would
    # assert something the model already does. No `chat_template` either — Kolibri's ships
    # with the plugin rather than living in vLLM's examples directory.
    tool_parser="kolibri1", reasoning_parser="kolibri1", thinks_by_default=True,
    # Kolibri 1's card: 1.0 / 0.97 / 128. The 128 is again the point — pi cannot send top_k
    # at all, so an unpinned one silently becomes vLLM's default.
    sampling={"temperature": 1.0, "top_p": 0.97, "top_k": 128},
    # The module name is the IMPORTABLE one (underscores), not the distribution name
    # (hyphens) — `importlib.util.find_spec` only accepts the former, and a hyphen here could
    # never be found, so the guard would refuse a perfectly good install.
    plugin=("aleph_alpha_inference", "pip install 'aleph-alpha-inference>=1'"),
)

_KOLIBRI_1 = dataclasses.replace(
    _KOLIBRI, family="kolibri-1", match=("kolibri-1",),
    # Serves on vLLM 0.29.x with the aleph-alpha-inference plugin, the one vLLM that plugin
    # supports. It reads Kolibri-1's hidden states at layers 4, 14, 24, 34 and 44, hence
    # `kolibri-1` rather than the family-wide `kolibri`. Measured with GLQ's 3 bpw build on an
    # RTX PRO 4500 (vLLM 0.29.0, 18 GiB of experts on host, 4 GiB arena, 262,144 window, B=1,
    # temperature 0): 118.56 tok/s at 4 draft tokens against 53.31 without
    # (`benchmarks/kolibri_dspark_rtxpro4500/`).
    dspark_drafter="intuit/Kolibri-1-dspark",
)

_NEMOTRON_35 = ModelProfile(
    family="nemotron-3.5", match=("nemotron-3.5",),
    # NVIDIA's card for Nemotron-3.5-Lightning-30B-A3B serves with exactly these two, and vLLM
    # 0.32.0 has both built in. Keyed on "nemotron-3.5" rather than "nemotron" because the wider
    # name spans formats that differ: Nemotron-3-Nano's card uses a `nano_v3` reasoning parser
    # loaded from a plugin file, and Llama-Nemotron is a Llama model with Llama's tool format.
    #
    # No `chat_template_kwargs`: the template opens the <think> block by default
    # (`enable_thinking` defaults to True; the prompt ends in `<think>\n`). Serving in bf16
    # matters more here than any flag (fp16 skips the reasoning; see `_NEMOTRON`).
    tool_parser="qwen3_coder", reasoning_parser="nemotron_v3", thinks_by_default=True,
    # Nemotron 3.5's card: 1.0 / 0.95 in every example, reasoning on or off, and no top_k.
    sampling={"temperature": 1.0, "top_p": 0.95},
    fp16_unsafe=True,
)

_NEMOTRON_35_VL = dataclasses.replace(
    _NEMOTRON_35, family="nemotron-3.5-vl", match=(("nemotron-3.5", "-vl"),),
    # A multimodal wrapper, served text-only: vision + projector 1.57 GiB, and the encoder
    # profile: vLLM profiled it with one maximum-size video at the window's budget. With
    # `glq-code --dspark 3` at 262,144 on a 31.86 GiB card that left 1.41 GiB of KV against
    # 3.05 GiB for one request, and vLLM refused to start.
    text_only=True,
)

_NEMOTRON_35_SUPER_VL = dataclasses.replace(
    _NEMOTRON_35_VL, family="nemotron-3.5-super-vl", match=("nemotron-3.5-super-vl",),
    # Measured with GLQ's 3 bpw build on a 21.96 GiB budget, vLLM 0.32.0, AIPerf coding
    # prompts: 31.50 tok/s per request at 4 draft tokens against 20.15 without
    # (`benchmarks/supervl3_spec_24gb/`). A drafter is a small separate model trained against
    # the BASE model; a GLQ build keeps the base's tokenizer, so the base's drafter serves it.
    dspark_drafter="RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B.dspark",
    # KV in only 8 of its 88 layers (40 are Mamba), so a long window costs a third of the
    # envelope, but each request also carries a fixed share. vLLM's own figures, with the
    # DSpark drafter loaded (RTX PRO 4500 capped at 21.96 GiB, vLLM 0.32.0,
    # `benchmarks/supervl3_spec_24gb/`), as pool / "Maximum concurrency":
    #
    #        16,384 window   0.67 GiB per request
    #       262,144          3.12
    #       524,288          5.53
    #     1,048,576         10.43   (vLLM's refusal: what one request needs)
    #
    # One per-token figure cannot cover that: 262,144's 12.8 KB/token under-prices 131,072, and
    # 131,072's would price 262,144 out of a 24 GB card. 0.75 GiB + 10,000 B/token sits above
    # every point (3.19 GiB at 262,144 against 3.12, 10.52 at 1M against 10.43), before
    # `_KV_SAFETY_FACTOR`. Without the drafter the cost is lower still (2.26 GiB at 262,144), so
    # one figure serves both.
    kv_cost=(int(0.75 * 1024**3), 10_000),
)

_NEMOTRON = ModelProfile(
    family="nemotron", match=("nemotron",),
    # Every other Nemotron line (Nemotron-3-Nano, Llama-Nemotron): no tool setup, because their
    # recipes differ (see `_NEMOTRON_35`), but the dtype rule still covers the NemotronH ones.
    #
    # NemotronH: fp16 SILENTLY DISABLES REASONING. The chat template ends the prompt in
    # `<think>\n`; in fp16 the model closes that block at once and answers directly — fluent,
    # even correct, and no NaN anywhere. Measured 2026-10-27, Nemotron-3.5-Lightning-30B,
    # vLLM 0.32.0, RTX PRO 6000, greedy, identical prompt:
    #
    #   bf16 base  vLLM bf16   "Here's a thinking process: ..."
    #   GLQ 3bpw   HF   bf16   "Here's a thinking process: ..."
    #   GLQ 3bpw   vLLM bf16   "Here's a thinking process: ..."
    #   GLQ 3bpw   vLLM fp16   "Red, Blue, Yellow. Red is the color of fire ..."
    #
    # The worst failure shape for a reasoning model: an AIME run would score like a much
    # weaker model with nothing in the output to say why. Residual stream peaks at 952, inside
    # fp16's range, so the overflow is most likely within a block — the MoE squares its
    # intermediate (relu^2), which overflows fp16 above ~256. Not yet localised.
    #
    # The price is small and measured. GLQ 3bpw, CUDA graphs, decode 128, repeats 3:
    #   fp16  B=1 130.915 (130.840-130.946)  B=8 913.6   -- does not reason
    #   bf16  B=1 127.549 (127.408-127.684)  B=8 882.3   -- reasons
    # i.e. -2.6% / -3.4%, non-overlapping ranges, for a model that actually thinks.
    fp16_unsafe=True,
)

_GLM5 = ModelProfile(
    family="glm-5", match=("glm-5",),
    # Two spellings: the repo id says "GLM-5.3", while config.json says `glm5_next` /
    # `Glm5NextForConditionalGeneration`. The second is an architecture fact only, so it
    # cannot pick a tool parser for a directory or a fine-tune that merely says "glm5".
    config_match=("glm5",),
    # vLLM's GLM-5.3-Flash recipe: `glm47` for both. vLLM 0.32.0 registers `glm45` and `glm47`
    # as aliases of the same two classes; the recipe's name is carried. Keyed on "glm-5"
    # because GLM-4 checkpoints use other templates and were never checked.
    #
    # No `chat_template_kwargs`: the template opens `<think>` at every generation prompt, with
    # no switch. The recipe also passes `--tokenizer-mode glm5_next`, which vLLM 0.32.0 does
    # not have.
    tool_parser="glm47", reasoning_parser="glm47", thinks_by_default=True,
    # GLM-5.3's generation_config.json, and the card's evaluation settings: 1.0 / 0.95, no
    # top_k. `--language-model-only` has not been run on GLM-5.3 yet, so it is not `text_only`.
    sampling={"temperature": 1.0, "top_p": 0.95},
    # vLLM 0.32.0's GLM-5 kernels take bf16 only. `models/glm5next/nvidia/ops/
    # kpool_compress.py` asserts it at seven sites, and `common/kda.py` names bf16 as a
    # FlashKDA requirement. Read in the source, not run in fp16.
    fp16_unsafe=True,
    # FULL CUDA graphs, because vLLM's default FULL_AND_PIECEWISE serves NaN logits. One-
    # variable A/Bs, prompt "The capital of France is", greedy, vLLM 0.32.0, RTX PRO 6000,
    # bf16, experts offloaded over UVA, max-num-seqs 1, window 16,384:
    #
    #   GLQ 4 bpw (glq-code's own argv)  default  "locklocklock…", logprobs NaN (HTTP 400)
    #                                    FULL     " Paris. In French, Paris is …"
    #   NVIDIA NVFP4 (modelopt_fp4)      default  "locklocklock…", logprobs NaN (HTTP 400)
    #                                    FULL     " Paris. Paris is the largest city …"
    #
    # (`benchmarks/glm_5_3_flash_nan_ab/`, 2026-11-10/11.) The NVFP4 build runs no GLQ code,
    # so this is the model under vLLM's piecewise graphs, not GLQ's kernels; glq-code first
    # showed it as "!!!!". Not yet separated: the SM120 card and the UVA expert offload, which
    # every failing run shared, and which piecewise piece produces the NaN. FULL also left more
    # KV (GLQ 391,850 tokens against 299,008; NVFP4 697,685 against 604,842). glq-bench and
    # run_model.py already forced FULL for every model, which is why their runs were coherent.
    compilation_config={"cudagraph_mode": "FULL"},
)

_MISTRAL = ModelProfile(
    family="mistral", match=("ministral", "mistral", "devstral"),
    # Activation outliers exceed fp16's 65504 range -> hard NaN. The observed case; see
    # glq/quantized_linear.py's fp32-accum note and the GLQ fp16-overflow finding. No tool
    # setup: none was ever checked, so glq-code refuses these.
    fp16_unsafe=True,
)

#: Every family, MOST SPECIFIC FIRST: lookups take the first profile that matches, so a
#: broader key above a narrower one would hide it (`test_serve_profiles` checks each profile
#: is reachable by its own key).
PROFILES = (
    _GEMMA4,
    _QWEN,
    _SMOLLM3,
    _KOLIBRI_1,
    _KOLIBRI,
    _NEMOTRON_35_SUPER_VL,
    _NEMOTRON_35_VL,
    _NEMOTRON_35,
    _NEMOTRON,
    _GLM5,
    _MISTRAL,
)

_SCOPE_TEXT = {"template": _family_name, "arch": _family_text}


def _key_in(key, text: str) -> bool:
    if isinstance(key, tuple):
        return all(part in text for part in key)
    return key in text


def profiles_for(model_id: str, scope: str) -> list[ModelProfile]:
    """Every profile matching `model_id` in `scope`, in table order."""
    try:
        text = _SCOPE_TEXT[scope](model_id or "")
    except KeyError:
        raise ValueError(f"unknown scope {scope!r}; expected one of {sorted(_SCOPE_TEXT)}")
    out = []
    for p in PROFILES:
        keys = p.match + (p.config_match if scope == "arch" else ())
        if any(_key_in(k, text) for k in keys):
            out.append(p)
    return out


def profile_for(model_id: str, scope: str) -> ModelProfile | None:
    """The family `model_id` belongs to in `scope` ("template" or "arch"), or None."""
    found = profiles_for(model_id, scope)
    return found[0] if found else None


# ---- views onto the table ------------------------------------------------------------------

def tool_serve_args(model_id: str, templates_dir=None) -> list[str] | None:
    """vLLM serve args that make tool calling work for this model's family.

    None for an unknown family — the caller must refuse rather than guess, because a
    wrong parser fails silently at the worst layer.

    `--default-chat-template-kwargs` is compact JSON, and that is NOT what keeps a printed
    command copy-pasteable (an earlier comment here claimed so; running the printed command
    through bash disproved it): compactness prevents word-splitting, but the shell still strips
    the double quotes (vLLM then received `{enable_thinking:true}`, not JSON) and `{a,b}` is
    brace expansion. Shell quoting is applied where commands are PRINTED
    (`installer.__main__._shell_arg`); these args stay raw because they are handed to
    subprocess as a list, with no shell in between.
    """
    p = profile_for(model_id, "template")
    if p is None or p.tool_parser is None:
        return None
    args = ["--enable-auto-tool-choice", "--tool-call-parser", p.tool_parser]
    if p.reasoning_parser:
        args += ["--reasoning-parser", p.reasoning_parser]
    if p.chat_template:
        args += ["--chat-template",
                 str(Path(templates_dir or _default_templates_dir()) / p.chat_template)]
    if p.chat_template_kwargs:
        args += ["--default-chat-template-kwargs",
                 json.dumps(p.chat_template_kwargs, separators=(",", ":"))]
    return args + text_only_serve_args(model_id)


def text_only_serve_args(model_id: str) -> list[str]:
    """`--language-model-only` for a multimodal wrapper served text-only, else `[]`.

    glq-chat and glq-code send text only, so a vision tower is VRAM and startup memory spent
    on nothing. vLLM reads the flag as every modality's limit at 0, which skips loading the
    towers and profiling the encoder; it ignores the flag on a text-only model. Set per family
    only where it was measured (`ModelProfile.text_only`).
    """
    p = profile_for(model_id, "template")
    return ["--language-model-only"] if p is not None and p.text_only else []


def reasoning_serve_args(model_id: str) -> list[str]:
    """`--reasoning-parser <name>` for a family that thinks by default, else `[]`.

    glq-chat sends no `chat_template_kwargs`, so these are the only families whose chat replies
    carry a reasoning block at all. Without a parser vLLM returns the reasoning inside
    `content`, so a chat shows the whole trace — for Nemotron 3.5 followed by a literal
    `</think>`, which is not a special token. The name is the profile's, the same one
    `tool_serve_args` hands glq-code, so the two commands cannot disagree on it.
    """
    p = profile_for(model_id, "template")
    if p is None or not p.thinks_by_default or not p.reasoning_parser:
        return []
    return ["--reasoning-parser", p.reasoning_parser]


def compilation_config(model_id: str) -> dict | None:
    """The vLLM compilation config this model must be served with, or None for vLLM's own.

    An architecture fact, so matched in the "arch" scope: a box copy of GLM-5.3 named
    `/opt/dlami/nvme/flash-box-copy` still gets it from its config.json.
    """
    p = profile_for(model_id, "arch")
    if p is None or p.compilation_config is None:
        return None
    return dict(p.compilation_config)


def correctness_serve_args(model_id: str, device: str = "cuda") -> list[str]:
    """Flags every `vllm serve` of this model needs to produce correct output, else `[]`.

    Unlike `role_serve_args`, not a choice of what a command offers: without these the model
    answers wrongly whoever launched it, so the supervisor applies them for glq-chat and
    glq-code, and the printed command, the model card and glq-bench's own servers carry them.

    CUDA only. The CPU backend runs `--enforce-eager`, so there is no graph mode to choose.
    Compact JSON, raw: shell quoting is applied where a command is PRINTED.
    """
    if device != "cuda":
        return []
    cc = compilation_config(model_id)
    if cc is None:
        return []
    return ["--compilation-config", json.dumps(cc, separators=(",", ":"), sort_keys=True)]


#: vLLM's two names for the compilation config. A long name's underscores read as dashes up to
#: its first ".", as vLLM's own parser normalises them; the short one is taken as written.
_CC_NAMES = ("--compilation-config", "-cc")


def _cc_flag(token: str) -> tuple[str, str | None] | None:
    """(dotted key path, "" for whole JSON; the value after "=", or None) for a token that
    names the compilation config, else None. Split on "=" first, as vLLM does, so a "." inside
    a JSON value is not read as a key path."""
    name, eq, inline = token.partition("=")
    head, _, path = name.partition(".")
    if head.startswith("--"):
        head = head.replace("_", "-")
    if head not in _CC_NAMES:
        return None
    return path, (inline if eq else None)


def _vllm_dotted_value(raw: str):
    """A dotted flag's value exactly as vLLM 0.32.0 reads it (`FlexibleArgumentParser`): JSON,
    else a human-readable int ("1k" = 1000, "1K" = 1024), else the string as written."""
    try:
        return json.loads(raw)
    except ValueError:
        pass
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([kKmMgGtT])", raw.strip())
    try:
        if m is None:
            return int(raw)
        number, suffix = m.groups()
        if suffix in "kmgt":
            return int(float(number) * 10 ** (3 * ("kmgt".index(suffix) + 1)))
        return int(number) * 2 ** (10 * ("KMGT".index(suffix) + 1))
    except ValueError:      # e.g. "1.5K": vLLM refuses decimals with binary suffixes
        return raw


def _cc_update(dst: dict, src: dict, *, append_lists: bool) -> None:
    """Later settings win per key; nested dicts merge. Lists append for dotted flags, which
    is what vLLM does when it gathers those itself, and are replaced by a whole-JSON one."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _cc_update(dst[k], v, append_lists=append_lists)
        elif append_lists and isinstance(v, list) and isinstance(dst.get(k), list):
            dst[k] = dst[k] + v
        else:
            dst[k] = v


def merge_compilation_config(argv: list[str]) -> list[str]:
    """`argv` with every vLLM compilation-config flag collapsed into ONE, settings merged in
    order, so a later flag adds to an earlier one instead of replacing it.

    vLLM 0.32.0 does not merge them itself (read from `FlexibleArgumentParser.parse_args`, and
    measured on an RTX PRO 6000 box, 2026-11-11):

      * a whole-JSON flag (`--compilation-config X`, `=X`, `-cc X`, `-cc=X`) is plain argparse,
        so the last one is kept WHOLE. Ours plus a caller's `{"mode":0}` parsed to
        `cudagraph_mode=None`: back to the default under which GLM-5.3 returns NaN logits;
      * dotted flags (`-cc.mode=0`, `--compilation-config.mode 0`, `-cc.ops+ a,b`) are gathered
        into one dict that is appended at the END of argv, so they replace a whole-JSON flag too.

    The merged flag takes the place of the first one; later settings win per key, so a caller
    who sets `cudagraph_mode` themselves still gets theirs. With fewer than two such flags, or
    a value it cannot read exactly as vLLM would (whole JSON that is not an object, a missing
    value), argv comes back unchanged: then vLLM applies its own rules, or reports the error.
    Not recognised: argparse abbreviations of the long name, and `--config` YAML files.
    """
    hits = []                               # (index, token count, key path, raw value)
    i = 0
    while i < len(argv):
        flag = _cc_flag(argv[i])
        if flag is None:
            i += 1
            continue
        path, inline = flag
        if inline is None:
            if i + 1 >= len(argv):
                return list(argv)
            hits.append((i, 2, path, argv[i + 1]))
            i += 2
        else:
            hits.append((i, 1, path, inline))
            i += 1
    if len(hits) < 2:
        return list(argv)

    merged: dict = {}
    for _, _, path, raw in hits:
        if not path:
            try:
                value = json.loads(raw)
            except ValueError:
                return list(argv)
            if not isinstance(value, dict):
                return list(argv)
            _cc_update(merged, value, append_lists=False)
            continue
        keys = path.split(".")
        if keys[-1].endswith("+"):          # vLLM's list form: `-cc.splitting_ops+ a,b`
            keys[-1] = keys[-1][:-1]
            value = raw.split(",")
        else:
            value = _vllm_dotted_value(raw)
        for key in reversed(keys):
            value = {key: value}
        _cc_update(merged, value, append_lists=True)

    first = hits[0][0]
    taken = {j for start, n, _, _ in hits for j in range(start, start + n)}
    out = []
    for j, arg in enumerate(argv):
        if j == first:
            out += ["--compilation-config",
                    json.dumps(merged, separators=(",", ":"), sort_keys=True)]
        if j not in taken:
            out.append(arg)
    return out


#: The roles `role_serve_args` composes for.
ROLES = ("code", "chat")


def role_serve_args(model_id: str, role: str, templates_dir=None) -> list[str] | None:
    """The family's vLLM flags for one command, so glq-chat and glq-code cannot drift apart.

      "code"  tool calling plus the card's sampling, for a tool-using agent (glq-code, and the
              installer's printed command). None for a family with no known tool setup: the
              caller refuses rather than guess. Sampling is appended, not substituted: correct
              sampling with no tool parser is a useless coding session.
      "chat"  the reasoning parser and the text-only load, for glq-chat. No tool flags (the
              chat sends no tools) and no sampling override (its sliders send every field).
    """
    if role == "code":
        tool = tool_serve_args(model_id, templates_dir=templates_dir)
        if tool is None:
            return None
        return tool + (sampling_serve_args(model_id) or [])
    if role == "chat":
        return reasoning_serve_args(model_id) + text_only_serve_args(model_id)
    raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")


def required_plugin(model_id: str) -> tuple[str, str] | None:
    """`(importable module, pip command)` this family needs beyond glq + vLLM, or None.

    Only Kolibri 1 today. vLLM has no native `Kolibri1ForCausalLM` and no `kolibri1`
    parsers, so without the plugin a serve fails at vLLM startup — minutes into a weight
    load, with an unknown-architecture or unknown-parser error that names neither the
    plugin nor the install command.

    Callers should test presence with `importlib.util.find_spec`, **never by importing**:
    importing the plugin runs its registration side effects (`ModelRegistry.register_model`,
    `AutoConfig.register`, two parser registrations), which is not something a check should
    do. Note the plugin pins `vllm>=0.29.0,<0.30.0` and each of its releases supports
    exactly one vLLM minor, so `glq[vllm]` would move vLLM out of range — plain `glq`
    installed *after* the plugin is the order that works.
    """
    p = profile_for(model_id, "template")
    return p.plugin if p is not None else None


def dspark_drafter(model_id: str) -> str | None:
    """The DSpark drafter repo for this model's family, or None when there is none.

    Listed only when it loads through vLLM's own `dspark` method without remote code: the
    launchers never pass `--trust-remote-code` for one. Matched on the id, like
    `required_plugin`: a local directory carries whatever name it was copied under, so
    `--draft-model` names the drafter for one.
    """
    p = profile_for(model_id, "template")
    return p.dspark_drafter if p is not None else None


#: Neutral values for a family we have no card for — usable on a slider, opinionated about
#: nothing. 0 is how vLLM spells "top_k off".
_NEUTRAL_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": 0}


def _family_sampling(model_id: str) -> dict | None:
    """The card's sampling for this family. Only parameters that are NOT already vLLM's
    default appear: `min_p=0.0`, `presence_penalty=0.0` and `repetition_penalty=1.0` are on
    Qwen's card but are also `_DEFAULT_SAMPLING_PARAMS` in vLLM's OpenAI path, so listing them
    would change nothing while implying it did. `top_p`'s OpenAI default IS 1.0, so 0.95 is a
    real setting."""
    p = profile_for(model_id, "template")
    return dict(p.sampling) if p is not None and p.sampling is not None else None


def sampling_serve_args(model_id: str) -> list[str] | None:
    """vLLM serve args pinning this family's recommended sampling, or None if unknown.

    **Why this has to be server-side.** `top_k` is not in the OpenAI schema, so an
    OpenAI-dialect client cannot send it — pi does not, and the installer's pi config
    (`configure.pi_models_json`) has no field for it either. So the one parameter most likely
    to be wrong is the one only the server can set.

    **What it does and does not override.** `--override-generation-config` feeds
    `default_sampling_params`, and vLLM resolves each field "user -> server default ->
    OpenAI default" (`entrypoints/openai/chat_completion/protocol.py:to_sampling_params`,
    verified on 0.30.0). Every field here is `None` on an incoming request unless the client
    set it, so: `top_k` always takes effect, `top_p` takes effect for a client that omits it,
    and `temperature` is inert against any client that sends its own — which pi does. That is
    worth stating plainly rather than implying the server has the last word.

    None for an unknown family, which is the opposite of `tool_serve_args`'s refusal and for
    a reason: a wrong tool parser fails silently at the worst layer, whereas no sampling
    override just means the checkpoint's own `generation_config.json` is used. That is a
    better answer than a guess, not a worse one.
    """
    params = _family_sampling(model_id)
    if params is None:
        return None
    # Compact separators: the installer prints these as copy-pasteable shell commands, and a
    # space inside an unquoted JSON argument splits it into two.
    return ["--override-generation-config",
            json.dumps(params, separators=(",", ":"), sort_keys=True)]


def recommended_sampling(model_id: str) -> dict:
    """Slider defaults for `glq-chat`, from the same table, always fully populated.

    The UI cannot render None on a slider, so a family with no `top_k` (SmolLM3) gets 0 —
    which is how vLLM spells "off" — and an unknown model gets neutral values rather than
    one family's numbers.
    """
    return {**_NEUTRAL_SAMPLING, **(_family_sampling(model_id) or {})}


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


#: The substrings that keep a model off fp16, read from the profiles (`fp16_unsafe`). Kept
#: importable as a flat tuple for callers and tests; `preferred_dtype` reads the profiles.
FP16_UNSAFE = tuple(dict.fromkeys(
    k for p in PROFILES if p.fp16_unsafe for k in p.match + p.config_match
    if isinstance(k, str)))

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

    See :data:`_DEFAULT_DTYPE` for why the devices differ. Any matching profile marked
    `fp16_unsafe` gets bf16 (the one any-match lookup: bf16 is the safe direction), matched in
    the "arch" scope so a box copy is judged by its config. An unrecognised device gets bf16:
    it has not been measured, and bf16 is what the checkpoint declares, so it is the answer
    that cannot be worse than today's.

    Why a DENYLIST when `tool_serve_args` deliberately refuses unknown families: that rule
    exists because a wrong tool parser "fails silently at the worst layer". A wrong dtype has
    THREE failure modes, and only the third is silent:

      1. loud  -- fp16 overflow: NaN, visibly garbage on the first token (the Mistral case).
      2. loud  -- the architecture refuses the dtype outright and the engine will not start
                  (the Qwen4Exp case).
      3. SILENT -- a kernel that wanted bf16 quietly picks a slower implementation. This one
                  is real: `qwen_gdn_linear_attn.py:535` drops the fused CUDA GDN decode
                  kernel when `model_config.dtype != torch.bfloat16`, and since
                  VLLM_GDN_DECODE_KERNEL defaults to "cuda" while its raise fires only when
                  the env var was set explicitly, the default path is a `logger.info_once`.
                  `cpu_moe.py:466/502/535` do the same via `return False`.

    Mode 3 was the reason to doubt a denylist, so it was measured rather than argued: forcing
    the Triton GDN path on Qwen3.8-Flash-Next-GLQ-3bpw (both arms bf16, sm_120, vLLM 0.29.0,
    decode 64, repeats 3) cost -0.26% at B=1 -- 32.035 vs 31.951 tok/s with overlapping
    ranges, and triton marginally AHEAD at B=8.

    A null that small is indistinguishable from a flag that does nothing, and vLLM's init-time
    "GDN decode kernel: …" line proves only that the flag was SET. So the substitution was
    confirmed at the op level (`benchmarks/_gdn_cuda_engagement.py`, in-process + eager so the
    calls are visible; a FULL cudagraph replay does not re-enter Python):

      kernel=cuda    qwen_gdn_attention_core 0 calls   ..._fused_norm_packed 1536 calls
      kernel=triton  qwen_gdn_attention_core 1536      ..._fused_norm_packed 0

    1536 = 48 GDN layers x 32 decode tokens: a complete swap, so the flag does change what
    runs. The op is **0.70-0.82% of total device time**, which caps any end-to-end effect at
    ~0.8% and explains the null instead of merely reporting it. On a GLQ checkpoint decode is
    dominated by the trellis dequant+matvec, not the GDN recurrence -- the same reason the CPU
    GDN kernel's in-situ share is ~0%. So the silent mode exists but is cheap, and that -- not
    its nonexistence -- is what makes a denylist acceptable here.

    On CUDA, quality is not traded for the speed: wikitext-2 PPL is marginally BETTER in fp16
    than bf16 (SmolLM3-3B trellis 6bpw: 9.1145 vs 9.1337), because fp16 carries 10 mantissa
    bits against bf16's 7. bf16 buys exponent range by spending precision, so where
    activations fit fp16's range fp16 is the more accurate of the two and the only risk is
    overflow.
    """
    if any(p.fp16_unsafe for p in profiles_for(model_id, "arch")):
        return "bfloat16"
    return _DEFAULT_DTYPE.get(device, "bfloat16")
