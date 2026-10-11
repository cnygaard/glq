"""Per-layer resume checkpoints for ``glq-quantize``.

A 30B quantize runs ~2 hours (52 layers × ~148 s measured on Muse-Glimmer-30B) and
``quantize()`` writes nothing until the single ``save_file`` at the end, so a spot reclaim
at 1 h 50 m destroys every layer. This module persists each layer's artifacts as soon as
they are produced — locally, and optionally to a private S3 bucket that outlives the
instance. On-box checkpoints alone would not help, because the failure to protect against
*is* losing the box.

Shard layout deliberately matches the final checkpoint's flattening
(``quantize_model.py:1983-1985``): ``state_dict[f"{layer_prefix}.{key}"]``. One layout, one
code path, and the final assembly can consume shards without translation.

**The ordering invariant.** The manifest is updated only after a layer's shard is fully
stored. A crash mid-upload therefore leaves the layer *not done* and it is re-quantized on
resume. The alternative — recording the layer first — turns a crash into a silently
truncated shard that resume happily loads, which is worse than no resume at all. sha256 in
the manifest enforces the same thing for data that was corrupted after the fact.

Credentials come from the EC2 instance profile; this module never accepts, logs, or stores
a key.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile

__all__ = ["ResumeStore", "ResumeKeyMismatch", "compute_run_key", "shard_from_checkpoint",
           "quantized_prefix_leaves",
           "detect_sd_prefix", "derive_run_key",
           "dequantize_artifacts"]

_MANIFEST = "manifest.json"


class ResumeKeyMismatch(RuntimeError):
    """The manifest was written by a run with different settings."""


def compute_run_key(*, model, codebook, bpw, nsamples, seqlen, variant, sd_prefix,
                    version, bpw_map=None, mtp_bpw=None):
    """Stable digest of every setting that changes the produced weights.

    Resuming a 4 bpw manifest into a 3 bpw run would blend two configurations into one
    checkpoint without erroring anywhere, so the key must cover anything that alters
    output. Extra settings are cheap to add here and expensive to discover later.

    ``mtp_bpw`` is added **only when set**, which is deliberate and not merely tidy: the key
    addresses the S3 manifest, so unconditionally widening the payload would change every
    existing key and orphan the stores already uploaded — a resumable 77 GiB run would
    restart from layer 0 with nothing reporting why. A checkpoint with no draft head
    therefore keeps the digest it has always had.
    """
    fields = {
        "model": str(model), "codebook": str(codebook), "bpw": bpw,
        "nsamples": nsamples, "seqlen": seqlen, "variant": str(variant),
        "sd_prefix": str(sd_prefix), "version": str(version),
        "bpw_map": bpw_map if bpw_map is None else dict(sorted(bpw_map.items())),
    }
    if mtp_bpw is not None:
        fields["mtp_bpw"] = int(mtp_bpw)
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _S3Backend:
    """Thin boto3 wrapper. Imported lazily so boto3 stays an optional dependency."""

    def __init__(self, bucket):
        try:
            import boto3                               # noqa: PLC0415 — optional dep
        except ImportError as e:                       # pragma: no cover - env-dependent
            # Declared by the `quantize` extra, but a venv assembled by hand can still
            # miss it. Failing with a bare ModuleNotFoundError here costs whatever the
            # run had already done, so name the fix.
            raise RuntimeError(
                "--resume-bucket needs boto3, which is not installed. "
                "Install it with:  pip install 'glq[quantize]'  (or: pip install boto3)"
            ) from e
        self.bucket = bucket
        self._c = boto3.client("s3")

    def put(self, local_path, key):
        self._c.upload_file(local_path, self.bucket, key)

    def get(self, key, local_path):
        self._c.download_file(self.bucket, key, local_path)

    def exists(self, key):
        from botocore.exceptions import ClientError     # noqa: PLC0415
        try:
            self._c.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False


class ResumeStore:
    """Layer-granular checkpoint store: local directory + optional S3 mirror."""

    def __init__(self, local_dir, run_key, bucket=None, prefix="", backend=None):
        self.local_dir = local_dir
        self.run_key = run_key
        self.prefix = prefix.strip("/")
        os.makedirs(local_dir, exist_ok=True)
        if backend is not None:
            self.backend = backend
        elif bucket:
            self.backend = _S3Backend(bucket)
        else:
            self.backend = None

    # ---- paths -------------------------------------------------------------

    def _shard_name(self, idx):
        return f"layer_{idx:04d}.safetensors"

    def _meta_name(self, idx):
        return f"layer_{idx:04d}.json"

    def _local(self, name):
        return os.path.join(self.local_dir, name)

    def _key(self, name):
        return f"{self.prefix}/{name}" if self.prefix else name

    # ---- manifest ----------------------------------------------------------

    def _read_manifest(self):
        path = self._local(_MANIFEST)
        if not os.path.exists(path) and self.backend is not None:
            try:                                        # new box: pull the manifest down
                self.backend.get(self._key(_MANIFEST), path)
            except Exception:
                return {"run_key": self.run_key, "layers": {}}
        if not os.path.exists(path):
            return {"run_key": self.run_key, "layers": {}}
        with open(path) as f:
            m = json.load(f)
        if m.get("run_key") != self.run_key:
            raise ResumeKeyMismatch(
                f"manifest run_key {m.get('run_key')!r} != this run's {self.run_key!r}; "
                "refusing to resume a checkpoint produced with different settings "
                "(model/codebook/bpw/nsamples/seqlen/variant). Use a different "
                "--resume-prefix or start fresh.")
        return m

    def _write_manifest(self, m):
        path = self._local(_MANIFEST)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(m, f, indent=2, sort_keys=True)
        os.replace(tmp, path)                           # atomic locally
        if self.backend is not None:
            self.backend.put(path, self._key(_MANIFEST))

    # ---- public API --------------------------------------------------------

    def save_layer(self, idx, artifacts, metrics=None, proxy_losses=None):
        """Persist one layer. Raises if the upload fails, leaving the layer not-done."""
        from safetensors.torch import save_file          # noqa: PLC0415

        flat = {}
        for layer_prefix, tensors in artifacts.items():
            for key, t in tensors.items():
                flat[f"{layer_prefix}.{key}"] = t.detach().cpu().contiguous()
        shard = self._local(self._shard_name(idx))
        save_file(flat, shard)

        meta = {"prefixes": sorted(artifacts.keys()),
                "metrics": metrics or {}, "proxy_losses": proxy_losses or {}}
        meta_path = self._local(self._meta_name(idx))
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

        # Upload BEFORE recording. A failure here propagates and the layer stays not-done.
        if self.backend is not None:
            self.backend.put(shard, self._key(self._shard_name(idx)))
            self.backend.put(meta_path, self._key(self._meta_name(idx)))

        m = self._read_manifest()
        m["layers"][str(idx)] = {"sha256": _sha256(shard),
                                 "file": self._shard_name(idx)}
        self._write_manifest(m)

    def _ensure_local(self, idx):
        """Make the shard present locally, pulling from the bucket if needed."""
        shard = self._local(self._shard_name(idx))
        if not os.path.exists(shard) and self.backend is not None:
            self.backend.get(self._key(self._shard_name(idx)), shard)
        meta = self._local(self._meta_name(idx))
        if not os.path.exists(meta) and self.backend is not None:
            self.backend.get(self._key(self._meta_name(idx)), meta)
        return shard, meta

    def completed_layers(self):
        """Layers with a shard whose bytes still match the manifest's digest."""
        m = self._read_manifest()
        done = set()
        for k, rec in m.get("layers", {}).items():
            try:
                shard, _ = self._ensure_local(int(k))
            except Exception:
                continue                                # unreachable object => not done
            if os.path.exists(shard) and _sha256(shard) == rec.get("sha256"):
                done.add(int(k))
        return done

    def next_layer(self):
        """First layer not yet done. Stops at the first GAP: replay is sequential, so a
        hole means every later layer must be redone regardless of what is stored."""
        done = self.completed_layers()
        i = 0
        while i in done:
            i += 1
        return i

    def load_layer(self, idx):
        """Return ``(artifacts, metrics, proxy_losses)`` for a completed layer."""
        from safetensors.torch import load_file          # noqa: PLC0415
        shard, meta_path = self._ensure_local(idx)
        with open(meta_path) as f:
            meta = json.load(f)
        flat = load_file(shard)
        artifacts = {p: {} for p in meta["prefixes"]}
        for flat_key, t in flat.items():
            for p in meta["prefixes"]:
                if flat_key.startswith(p + "."):
                    artifacts[p][flat_key[len(p) + 1:]] = t
                    break
        return artifacts, meta.get("metrics", {}), meta.get("proxy_losses", {})


def dequantize_artifacts(arts, in_features, out_features, codebook_type,
                         block_diagonal=False, codebook=None):
    """Rebuild the dense ``W_hat`` a completed layer was written back with.

    Replay must put the *quantized* weights back into the layer before running the
    calibration forward — layer N's Hessians are gathered from activations that passed
    through the quantized layers 0..N-1 (``quantize_model.py:1798-1824``). Feeding the
    original bf16 weights instead would calibrate the remaining layers against a
    distribution the finished checkpoint never sees, and nothing would raise.

    Reuses ``E8RHTLinear`` (glq/quantized_linear.py:165) and its ``dequantize()`` rather
    than reimplementing per-codebook decode: the artifact keys are exactly that module's
    buffer names, because both come from the same checkpoint layout.
    """
    if codebook_type == "trellis":
        # Reuse the quantizer's OWN inverse RHT rather than recomposing it. An
        # independently-written FHT chain is mathematically equivalent but not bit-exact
        # (float addition is not associative — measured 1.8e-7 relative), and the trellis
        # encoder makes DISCRETE tile decisions, so a 1e-7 drift in the replayed
        # activations flips near-tie assignments and changes trellis_packed for every
        # later layer. Calling the same code path as
        # trellis.quantize_layer_trellis_rht (`W_hat = rht.inverse_transform_weights(
        # hatWr_norm * Wscale)`) keeps the op order identical, which is what byte-identical
        # resume actually requires.
        from .trellis import decode_layer, decode_layer_nstage   # noqa: PLC0415
        from .rht import RHT                             # noqa: PLC0415
        if codebook is None:
            raise ValueError("resume: trellis replay needs the codebook object")
        m, n = out_features, in_features
        dev = codebook.device
        packed = arts["trellis_packed"].to(dev)
        if "trellis_packed2" in arts:
            # Stacked RVQ (5-8 bpw): stage 1 K=4 plus a residual stage, summed with the
            # cumulative inverse residual scale — the same composition
            # _forward_trellis uses. The driver hangs the stage list off the primary
            # codebook, exactly as quantize_layer_trellis_rht resolves it.
            cbs = list(getattr(codebook, "rvq_stages", None) or [codebook])
            if len(cbs) < 2:
                raise ValueError(
                    "resume: shard has trellis_packed2 but the codebook carries no "
                    "second RVQ stage — codebook and checkpoint disagree on bpw.")
            W_rht = decode_layer_nstage(
                [cbs[0], cbs[1]],
                [packed, arts["trellis_packed2"].to(dev)],
                [1.0, float(arts["inv_resid_scale2"])],
                m, n, cbs[0].has_kernel)
        else:
            W_rht = decode_layer(codebook, packed, m, n, codebook.has_kernel)
        W_rht = W_rht * arts["Wscale"].to(dev).float()
        # Same RHT geometry the quantizer built; the stored signs replace the generated
        # ones so the transform is this layer's, not a fresh random draw.
        rht = RHT(m, n, device=dev, block_diagonal=True, apply_left=True, e8p=False)
        rht.su = arts["SU"].to(dev).float()
        rht.sv = arts["SV"].to(dev).float()
        return rht.inverse_transform_weights(W_rht)

    from .quantized_linear import E8RHTLinear     # noqa: PLC0415 — avoid import cycle
    mod = E8RHTLinear(in_features, out_features, bias=False,
                      block_diagonal=block_diagonal, codebook_type=codebook_type)
    missing, unexpected = mod.load_state_dict(
        {k: v for k, v in arts.items()}, strict=False)
    if unexpected:
        raise RuntimeError(
            f"resume: artifact keys {sorted(unexpected)} are not buffers of "
            f"E8RHTLinear(codebook_type={codebook_type!r}) — the shard was written by a "
            "different codebook or glq version than this run.")
    return mod.dequantize()


#: Leaf names that mark a tensor as a quantization ARTIFACT rather than a plain weight.
#: A prefix is quantized exactly when it owns at least one of these, which is what makes
#: `_is_artifact_leaf` — not `layer_metrics.json` — the authority on what to re-shard.
_ARTIFACT_LEAVES = frozenset({
    "trellis_packed", "tlut",            # QTIP trellis
    "Qidxs", "Qidxs2", "Qidxs3",         # shell / stacked RVQ indices
    "SU", "SV", "Wscale",                # RHT row/col signs, scale
})


#: Leaf names that are ordinary parameters, never quantization artifacts. Used as the
#: NEGATIVE half of the self-correcting rule below.
_PLAIN_LEAVES = frozenset({"weight", "bias"})


def _is_artifact_leaf(leaf: str) -> bool:
    """Is this final key segment a quantization artifact?

    Checked on the tensor keys because `layer_metrics.json` is **incomplete**, and silently
    so. Measured on xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4: 73,728
    quantized expert prefixes exist as tensors but only 24,576 appear in the metrics file —
    every ``experts.{e}.gate_proj`` and ``up_proj`` is absent, because gate and up are
    quantized *jointly* and the metrics land under the fused name while the artifacts are
    stored split.

    Keying off the metrics therefore dropped two thirds of the expert weights from the
    store, and replay would then leave them at bf16 through the calibration forward — which
    nothing reports. The only outward sign was the store being 16 GiB instead of ~43 GiB.
    """
    return (leaf in _ARTIFACT_LEAVES
            or leaf.startswith("trellis_packed")       # trellis_packed2, ...
            or leaf.startswith("inv_resid_scale")      # inv_resid_scale, ...2
            or leaf.endswith("_e8p") or leaf.endswith("_e81b"))


def quantized_prefix_leaves(keys) -> dict[str, set[str]]:
    """``{prefix: {leaf, ...}}`` for every quantized prefix among ``keys``.

    Two passes, because a whitelist of artifact names is not maintainable. Pass one finds
    prefixes that own a *recognised* artifact leaf; pass two then takes **all** of that
    prefix's non-plain leaves. So a layer type with an artifact nobody listed still travels
    complete, as long as it also has one familiar one.

    That is not hypothetical: the PLE table carries ``rht_blocks`` — the RHT block sizes the
    inverse transform needs for a non-power-of-two width — which no other layer type
    produces. A hand-kept list omitted it, and the table came back from the store
    undecodable, with nothing reporting a problem.

    The negative half matters equally. `ple_embedding.layer_multipliers` and
    `ple.norm_key.weight` sit right beside the table under *different* prefixes and own no
    artifact, so they stay out; sweeping them in would hand replay tensors with no module
    behind them.
    """
    owners = set()
    for key in keys:
        if "." not in key:
            continue
        prefix, leaf = key.rsplit(".", 1)
        if _is_artifact_leaf(leaf):
            owners.add(prefix)
    out: dict[str, set[str]] = {}
    for key in keys:
        if "." not in key:
            continue
        prefix, leaf = key.rsplit(".", 1)
        if prefix in owners and leaf not in _PLAIN_LEAVES:
            out.setdefault(prefix, set()).add(leaf)
    return out


def shard_from_checkpoint(*, checkpoint_dir, store, sd_prefix, layers=None,
                          verbose=True):
    """Rebuild resume shards from a finished GLQ checkpoint. ``-> {layer indices written}``.

    A 77 GiB quantize is hours of GPU time, and for a published model that work already
    exists on HuggingFace. Re-sharding it into the resume layout lets a new run replay the
    finished layers and quantize only what is new — adding the MTP draft head, say — rather
    than starting at layer 0.

    This is re-sharding, not conversion: the resume shard layout is deliberately the same
    flattening the final checkpoint uses (``f"{layer_prefix}.{key}"``), so the tensors go
    across untouched and ``load_layer`` reassembles them by the same prefix split.

    **The tensor keys are the authority on what is quantized** — see `_is_artifact_leaf`.
    ``layer_metrics.json`` looks like the natural source and is not: it is incomplete, and
    silently. On Flash-Next it records 24,576 expert prefixes while 73,728 exist, omitting
    every jointly-quantized ``gate_proj``/``up_proj``. Keying off it dropped two thirds of
    the expert weights. Metrics and ``proxy_loss`` are still read from it where present,
    best-effort, and the shortfall is reported rather than hidden.

    The run_key is the caller's: it must match the run that will resume, and `ResumeStore`
    refuses a mismatch rather than blending two configurations.
    """
    import re                                              # noqa: PLC0415
    from safetensors import safe_open                      # noqa: PLC0415

    # layer_metrics.json supplies metrics and proxy_loss, and nothing else. It is
    # explicitly NOT the authority on what is quantized — see _is_artifact_leaf.
    all_metrics = {}
    metrics_path = os.path.join(checkpoint_dir, "layer_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            all_metrics = json.load(f)
    elif verbose:
        print("  note: no layer_metrics.json; shards will carry no metrics or proxy_loss. "
              "Only mixed-precision allocation needs those, and that cannot be resumed "
              "anyway (see derive_run_key).")

    shard_files = sorted(f for f in os.listdir(checkpoint_dir)
                         if f.endswith(".safetensors"))
    if not shard_files:
        raise ValueError(f"{checkpoint_dir} contains no .safetensors files")

    want = None if layers is None else set(layers)
    pat = re.compile(rf"^{re.escape(sd_prefix)}\.(\d+)\.")
    # layer -> prefix -> {artifact key: tensor}
    by_layer: dict[int, dict] = {}
    for sf in shard_files:
        with safe_open(os.path.join(checkpoint_dir, sf), framework="pt") as fh:
            wanted = quantized_prefix_leaves(fh.keys())
            for prefix, leaves in wanted.items():
                m = pat.match(prefix)
                if m is None:
                    continue           # outside the decoder (the MTP head, ...)
                idx = int(m.group(1))
                if want is not None and idx not in want:
                    continue
                slot = by_layer.setdefault(idx, {}).setdefault(prefix, {})
                for leaf in leaves:
                    slot[leaf] = fh.get_tensor(f"{prefix}.{leaf}")

    if not by_layer:
        raise ValueError(
            f"no quantized tensor under sd_prefix {sd_prefix!r} in {checkpoint_dir}. A "
            "wrapped checkpoint uses 'model.language_model.layers', a plain one "
            "'model.layers'. Raising rather than writing an empty store, which would look "
            "like success and send the run back to layer 0 at full cost.")

    # The PLE table's rung, which `layer_metrics.json` does not record for it. Reuse needs
    # it: the published Flash-Next checkpoint is ple_bpw 4 while the descriptor currently
    # defaults to 3, and reusing 4 bpw artifacts under a 3 bpw declaration would size the
    # serving buffers wrongly. Injected as a metrics entry so it rides existing plumbing.
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            _qc = json.load(f).get("quantization_config", {})
        _ple_bpw = _qc.get("ple_bpw")
        if _ple_bpw is not None:
            for _p in [p for p in {k for v in by_layer.values() for k in v}
                       if p.endswith("ngram_embedding")
                       or p.endswith("embed_tokens_per_layer")]:
                all_metrics.setdefault(_p, {})["bpw"] = int(_ple_bpw)

    written = set()
    n_no_metrics = 0
    for idx in sorted(by_layer):
        arts = by_layer[idx]
        metrics = {p: all_metrics[p] for p in arts if p in all_metrics}
        losses = {p: m["proxy_loss"] for p, m in metrics.items() if "proxy_loss" in m}
        n_no_metrics += len(arts) - len(metrics)
        store.save_layer(idx, arts, metrics=metrics, proxy_losses=losses)
        written.add(idx)
        if verbose:
            print(f"  layer {idx}: {len(arts)} matrices re-sharded")
    if verbose:
        print(f"re-sharded {len(written)} layers into {store.local_dir}")
        if n_no_metrics:
            # Reported rather than silent: this is normal (jointly-quantized gate/up carry
            # no split-name metrics) but a large number is also how an actual problem would
            # look, and it is the count that distinguishes the two.
            print(f"  {n_no_metrics} matrices had artifacts but no layer_metrics entry — "
                  "sharded anyway; metrics are best-effort, artifacts are not")
    return written


def detect_sd_prefix(checkpoint_dir):
    """The checkpoint's own decoder-layer prefix, read off ``layer_metrics.json``.

    Asking the caller invites the one mistake that actually bites — ``model.layers`` against a
    wrapped ``model.language_model.layers`` checkpoint — and the answer is in the file.

    Splits on the **first** numeric segment, not the last: a per-expert prefix like
    ``model.language_model.layers.0.mlp.experts.0.gate_proj`` carries two, and the decoder
    index is the outer one. A greedy match returns ``...layers.0.mlp.experts``, which then
    groups every expert of every layer under "layer 0". Whichever prefix covers the most
    entries wins, so a stray non-layer key cannot outvote the decoder.
    """
    import re                                              # noqa: PLC0415
    from collections import Counter                        # noqa: PLC0415

    with open(os.path.join(checkpoint_dir, "layer_metrics.json")) as f:
        metrics = json.load(f)
    counts = Counter()
    for prefix in metrics:
        m = re.match(r"^(.*?)\.(\d+)\.", prefix)
        if m:
            counts[m.group(1)] += 1
    if not counts:
        raise ValueError(
            f"no prefix in {checkpoint_dir}/layer_metrics.json looks like "
            "'<something>.<N>.<sublayer>', so the decoder-layer prefix cannot be detected; "
            "pass sd_prefix explicitly.")
    return counts.most_common(1)[0][0]


def derive_run_key(checkpoint_dir, version=None, sd_prefix=None, mtp_bpw=None):
    """Run key for a run that will RESUME from this checkpoint.

    Every field `compute_run_key` hashes is recorded in the checkpoint — codebook, bpw,
    nsamples, seqlen and source_model in ``quantize_config.json``, variant in
    ``config.json``'s ``quantization_config`` — except the glq version, which must be the
    version of the resuming run rather than the older one that produced the published
    weights. Deriving beats asking: a mistyped ``nsamples`` yields a store the resuming run
    refuses, and it refuses it after the download.

    ``mtp_bpw`` declares that the resuming run will quantize the MTP draft head, because the
    key covers that too — without it, a store built here is refused by a run launched with
    ``--mtp-bpw``, which breaks the one workflow this tool exists for. Note what is being
    recorded: the shards hold decoder layers only, and the MTP post-pass runs after the loop
    and is never checkpointed, so this changes no stored byte. It declares intent so the two
    keys agree.
    """
    with open(os.path.join(checkpoint_dir, "quantize_config.json")) as f:
        qc = json.load(f)
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    variant = None
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            variant = json.load(f).get("quantization_config", {}).get("variant")

    bpw = qc["bpw"]
    if float(bpw) != int(bpw):
        raise ValueError(
            f"{checkpoint_dir} was quantized at a fractional (mixed-precision) bpw "
            f"{bpw}. The run key hashes the bpw_map handed IN to the allocator, and the "
            "checkpoint records only the per-layer assignment that came out, so the key "
            "cannot be reconstructed. Guessing would address the wrong manifest while "
            "looking correct. Pass run_key explicitly if you know it.")

    if version is None:
        from . import __version__ as version                # noqa: PLC0415
    return compute_run_key(
        model=qc["source_model"], codebook=qc["codebook"], bpw=int(bpw),
        nsamples=qc["nsamples"], seqlen=qc["seqlen"], variant=variant,
        sd_prefix=sd_prefix or detect_sd_prefix(checkpoint_dir),
        version=version, bpw_map=None, mtp_bpw=mtp_bpw)


def build_import_parser():
    """CLI for re-sharding a published checkpoint into a resume store.

    Only ``--checkpoint`` and ``--store-dir`` are required: everything the run key hashes is
    derived from the checkpoint's own config, because those values must match the resuming
    run exactly and hand-entering them is the failure this tool exists to avoid. The rest are
    escape hatches for the cases derivation cannot cover (a mixed-precision checkpoint, an
    unusual layer prefix).
    """
    import argparse                                        # noqa: PLC0415

    p = argparse.ArgumentParser(
        prog="python -m glq.resume",
        description="Rebuild a GLQ resume store from a finished checkpoint, so a new "
                    "quantize run replays the published layers instead of redoing them.")
    p.add_argument("--checkpoint", required=True,
                   help="HF repo id or local directory of the finished GLQ checkpoint")
    p.add_argument("--store-dir", required=True,
                   help="where to write the shards; point the resuming run's "
                        "<output>/_resume at this, or use --bucket to mirror it")
    p.add_argument("--bucket", default=None,
                   help="S3 bucket to mirror the shards into, for --resume-bucket")
    p.add_argument("--prefix", default="", help="key prefix within the bucket")
    p.add_argument("--run-key", default=None,
                   help="override the derived run key (needed for a mixed-precision "
                        "checkpoint, whose key cannot be reconstructed)")
    p.add_argument("--sd-prefix", default=None,
                   help="override the detected decoder-layer prefix")
    p.add_argument("--mtp-bpw", type=int, default=None,
                   help="the --mtp-bpw the resuming run will use. The run key covers it, so "
                        "without this a store built here is refused by a run that quantizes "
                        "the draft head — which is the main reason to build one. Declares "
                        "intent only: the shards are decoder layers and are unaffected")
    p.add_argument("--layers", default=None,
                   help="comma-separated subset, e.g. 0,1,2 — a cheap dry run before "
                        "committing to the full re-shard")
    return p


def main(argv=None):
    p = build_import_parser()
    args = p.parse_args(argv)

    path = args.checkpoint
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download       # noqa: PLC0415
        print(f"downloading {path} ...")
        # The weights are the bulk and they are all needed; no allow_patterns.
        path = snapshot_download(repo_id=path)

    sd_prefix = args.sd_prefix or detect_sd_prefix(path)
    run_key = args.run_key or derive_run_key(path, sd_prefix=sd_prefix,
                                             mtp_bpw=args.mtp_bpw)
    print(f"sd_prefix = {sd_prefix}")
    print(f"run_key   = {run_key}")
    print("  the resuming run must produce this same key, which means the same model, "
          "codebook, bpw, nsamples, seqlen, variant AND glq version. It prints its own "
          "key at startup; if they differ the store is refused rather than blended.")

    store = ResumeStore(local_dir=args.store_dir, run_key=run_key,
                        bucket=args.bucket, prefix=args.prefix)
    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    written = shard_from_checkpoint(checkpoint_dir=path, store=store,
                                    sd_prefix=sd_prefix, layers=layers)
    print(f"next_layer() = {store.next_layer()} "
          f"({len(written)} layer(s) now checkpointed)")


if __name__ == "__main__":
    main()
