# Recorded shape manifests

`*.json` here hold a checkpoint's tensor keys, shapes and dtypes — **no weight data**. Numeric
path segments collapse to `{i}` with their index ranges recorded, so Flash-Next's 297,256 keys
become 149 patterns in 46 KiB.

Index sets are recorded in whichever of three forms is exact — see `record.py`'s header. The
middle one, `values`, exists for hybrid towers: Nemotron-3.5-Super-VL interleaves mamba, MoE
and attention blocks, so its 40 MoE layers sit at non-contiguous indices among 88 and `ranges`
would over-claim. Every (layer, expert) combination does exist, though, so the two per-position
value lists are lossless in 40 + 512 numbers where the explicit pairs take 20,480 — the
difference between a 1,280.6 KiB manifest and the 37.8 KiB one recorded here.

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
| `kolibri_1_bf16.json` | [`Aleph-Alpha/Kolibri-1-BF16`](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16) | [apache-2.0](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16/blob/main/LICENSE) |
| `nemotron_3_5_super_vl_120b_a12b.json` | [`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16) | [openmdw-1.1](https://openmdw.ai/license/1-1/) |

**The licences differ, and the Qwen one is not Apache 2.0.** It is a custom licence with its
own notice requirement and a restriction on operating model-as-a-service businesses. Assuming
"probably Apache" for a Qwen model would be wrong, which is why `record.py` *requires*
`--base-model`, `--license` and `--license-url` instead of defaulting them, and why
`test_model_shapes.py` asserts every manifest carries all three. A new manifest cannot be
committed unattributed by accident.

Nemotron-3.5-Super-VL is a second example of the same hazard: its card declares
`license: other` with `license_name: openmdw-1.1` — the [OpenMDW 1.1](https://openmdw.ai/license/1-1/)
licence, **not** the NVIDIA Open Model License an earlier Nemotron draft card in this repo
used. Take the name and the link from the model's own `cardData`, never from a sibling model.

## Recording another model

```bash
python tests/models/record.py --repo <hf-repo-id> \
    --base-model Qwen/Qwen3.8-Flash-Next \
    --license "Qwen Community License 1.0" \
    --license-url https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE \
    --out tests/models/<slug>.json
```

Use `--checkpoint <dir>` for a local copy. The hub path fetches **only the safetensors
header**, never the weights — but HF caps that read at 25 MB, and Flash-Next's header is
39.45 MB, so that model has to be recorded from a local checkpoint.

Shard count is not the predictor: Nemotron-3.5-Super-VL is 63 shards and 231.8 GiB, and its
headers sum to **6.23 MB**, so `--repo` records it in about 5 seconds. What blows the cap is
key count (Flash-Next has 297,256 against Nemotron's 43,079).
