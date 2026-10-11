"""Pull Flash-Next base + the published GLQ checkpoint to the local SSD.

Token is read from a 0600 file, never from argv or the environment of a shell command,
so it cannot land in shell history or `ps`.
"""
import os, time
# HF_HUB_ENABLE_HF_TRANSFER is DEPRECATED — huggingface_hub 1.x no longer uses
# hf_transfer, so setting it only emits a FutureWarning and accelerates nothing.
# Xet is the transfer backend now (hf-xet), and it is already the default; this
# only raises its concurrency. Measured: 73.66 GiB in 3.7 min with hf_transfer not
# even installed.
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_HOME"] = "/opt/dlami/nvme/hf_cache"

with open("/opt/dlami/nvme/.hfread") as f:
    tok = f.read().strip()

from huggingface_hub import snapshot_download

REPOS = [
    # The published 3bpw checkpoint, for rebuilding the resume store. Smaller, and the
    # run would be pointless without it, so fetch it first.
    ("xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4", 77.56),
    # The base model: quantize streams layer weights from here, and the mtp.* tensors
    # the draft head needs live only here.
    ("Qwen/Qwen3.8-Flash-Next", 335.28),
]

for repo, gib in REPOS:
    t = time.perf_counter()
    print(f"=== {repo}  (~{gib:.1f} GiB) ===", flush=True)
    p = snapshot_download(repo_id=repo, token=tok, max_workers=16)
    dt = time.perf_counter() - t
    print(f"DONE {repo} in {dt/60:.1f} min  ({gib*1024/max(dt,1):.0f} MiB/s)  -> {p}",
          flush=True)
print("ALL_DOWNLOADS_DONE", flush=True)
