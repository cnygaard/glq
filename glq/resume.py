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

__all__ = ["ResumeStore", "ResumeKeyMismatch", "compute_run_key",
           "dequantize_artifacts"]

_MANIFEST = "manifest.json"


class ResumeKeyMismatch(RuntimeError):
    """The manifest was written by a run with different settings."""


def compute_run_key(*, model, codebook, bpw, nsamples, seqlen, variant, sd_prefix,
                    version, bpw_map=None):
    """Stable digest of every setting that changes the produced weights.

    Resuming a 4 bpw manifest into a 3 bpw run would blend two configurations into one
    checkpoint without erroring anywhere, so the key must cover anything that alters
    output. Extra settings are cheap to add here and expensive to discover later.
    """
    payload = json.dumps({
        "model": str(model), "codebook": str(codebook), "bpw": bpw,
        "nsamples": nsamples, "seqlen": seqlen, "variant": str(variant),
        "sd_prefix": str(sd_prefix), "version": str(version),
        "bpw_map": bpw_map if bpw_map is None else dict(sorted(bpw_map.items())),
    }, sort_keys=True, separators=(",", ":"))
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
        import boto3                                   # noqa: PLC0415 — optional dep
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
