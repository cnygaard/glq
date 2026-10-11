# Recorded shape manifests

`*.json` here hold a checkpoint's tensor keys, shapes and dtypes — **no weight data**. Numeric
path segments collapse to `{i}` with their index ranges recorded, so Flash-Next's 297,256 keys
become 149 patterns in 46 KiB.

They exist because a hand-written fixture encodes the author's *model* of an architecture
rather than the architecture. Three fixtures in this suite were wrong that way and each made a
test pass against a state no checkpoint can be in: 16×8 MTP shapes the production `out % 32`
gate rejects outright, a bare `…mlp.experts` `layer_bpw` key (that map lists the per-expert
*children*), and a fake MoE layer that returned `None` unconditionally.

`tests/model_shapes.py` is the loader; `tests/test_model_shapes.py` guards the manifests
themselves, since a silently wrong manifest is worse than none — every test built on it would
agree with it.

## Attribution

A manifest carries no weights, but it is derived from somebody else's released model and this
is a public repo, so each one records its base model and **that model's** licence, inside the
JSON (`base_model`, `license`, `license_url`) rather than only here — the manifest is what gets
read programmatically and copied around.

| manifest | base model | licence |
|---|---|---|
| `qwen3_8_flash_next_glq_3bpw.json` | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) |
| `gemma_4_26b_a4b_it.json` | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) | [apache-2.0](https://ai.google.dev/gemma/docs/gemma_4_license) |

**The licences differ, and the Qwen one is not Apache 2.0.** It is a custom licence with its
own notice requirement and a restriction on operating model-as-a-service businesses. Assuming
"probably Apache" for a Qwen model would be wrong, which is why `record.py` *requires*
`--base-model`, `--license` and `--license-url` instead of defaulting them, and why
`test_model_shapes.py` asserts every manifest carries all three. A new manifest cannot be
committed unattributed by accident.

## Recording another model

```bash
python tests/models/record.py --repo <hf-repo-id> \
    --base-model Qwen/Qwen3.8-Flash-Next \
    --license "Qwen Community License 1.0" \
    --license-url https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE \
    --out tests/models/<slug>.json
```

Use `--checkpoint <dir>` for a local copy. The hub path fetches **only the safetensors
header**, never the weights — but HF caps that read, and Flash-Next's header is 39.45 MB, so
that model has to be recorded from a local checkpoint.
