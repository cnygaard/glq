"""Pull the private -3mtp checkpoint to the L4's local SSD. Token from a 0600 file."""
import os, time
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"  # hf_transfer is gone in hub 1.x
os.environ["HF_HOME"] = "/opt/dlami/nvme/hf_cache"
with open("/opt/dlami/nvme/.hfread") as f:
    tok = f.read().strip()
from huggingface_hub import snapshot_download
REPO = "xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4-3mtp"
t = time.perf_counter()
p = snapshot_download(repo_id=REPO, token=tok, max_workers=16)
print(f"DL_DONE in {(time.perf_counter()-t)/60:.1f} min -> {p}", flush=True)
