"""Write SPEED-Bench's coding prompts as the JSONL aiperf 0.11.0 reads.

`aiperf profile --custom-dataset-type speed-bench-coding` needs `--input-file`: it does not
download the dataset itself. The HF parquet stores each prompt as `turns` (a list of user
messages), while aiperf's loader expects `messages` (role/content dicts). 10 of the 80 coding
rows hold a placeholder instead of text ("... FETCHED FROM THE SOURCE USING SPECDEC_BENCH") and
are skipped, leaving 70 (9 of them two-turn).

Usage (aiperf venv, after `snapshot_download('nvidia/SPEED-Bench', repo_type='dataset',
allow_patterns=['qualitative/*'])`):

    python make_speed_bench_coding.py <qualitative/test-00000-of-00001.parquet> <out.jsonl>
"""
import json
import sys

import pandas as pd

PLACEHOLDER = "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"


def main(parquet: str, out: str) -> None:
    df = pd.read_parquet(parquet)
    kept = skipped = 0
    with open(out, "w") as f:
        for row in df[df.category == "coding"].to_dict("records"):
            turns = [str(t) for t in row["turns"]]
            if any(PLACEHOLDER in t or not t.strip() for t in turns):
                skipped += 1
                continue
            f.write(json.dumps({
                "question_id": row["question_id"],
                "category": "coding",
                "messages": [{"role": "user", "content": t} for t in turns],
            }) + "\n")
            kept += 1
    print(f"wrote {kept} rows, skipped {skipped} placeholder rows -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
