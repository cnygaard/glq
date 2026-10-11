"""Private upload of the MTP-head checkpoint. Token read from a 0600 file, never argv.

`_resume/` is EXCLUDED and that is load-bearing: the resume store lives inside the output
directory (`<output>/_resume`), so a naive folder upload ships 69 GiB of working state
alongside the 74 GiB checkpoint — doubling the repo with shards nobody downloading it can use.
"""
import os, time
os.environ["HF_HOME"] = "/opt/dlami/nvme/hf_cache"
# HF_HUB_ENABLE_HF_TRANSFER is DEPRECATED — huggingface_hub 1.x no longer uses
# hf_transfer, so setting it only emits a FutureWarning and accelerates nothing.
# Xet is the transfer backend now (hf-xet), and it is already the default; this
# only raises its concurrency. Measured: 73.66 GiB in 3.7 min with hf_transfer not
# even installed.
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
with open("/opt/dlami/nvme/.hfwrite") as f:
    tok = f.read().strip()

from huggingface_hub import HfApi
REPO = "xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4-3mtp"
SRC = "/opt/dlami/nvme/fn_mtp3"
IGNORE = ["_resume/*", "_resume/**", "bpw_allocation.json"]

api = HfApi(token=tok)
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
print(f"repo ready (PRIVATE): {REPO}", flush=True)
print(f"ignoring: {IGNORE}", flush=True)
t = time.perf_counter()
api.upload_large_folder(repo_id=REPO, folder_path=SRC, repo_type="model",
                        ignore_patterns=IGNORE, print_report_every=60)
print(f"UPLOAD_DONE in {(time.perf_counter()-t)/60:.1f} min", flush=True)

files = api.list_repo_files(REPO)
print("repo files:", flush=True)
for f in sorted(files):
    print("   ", f, flush=True)
leaked = [f for f in files if f.startswith("_resume")]
print("LEAKED_RESUME_FILES:", len(leaked), flush=True)
