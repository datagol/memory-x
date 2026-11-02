"""Download LongMemEval-S (cleaned) into eval/longmemeval/data/."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
DEST = Path(__file__).parent / "data" / "longmemeval_s_cleaned.json"


def main() -> None:
    if DEST.exists():
        print(f"already present: {DEST}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".part")
    with httpx.stream("GET", URL, follow_redirects=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        done = 0
        with tmp.open("wb") as out:
            for chunk in resp.iter_bytes(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    sys.stderr.write(f"\r{done / total:6.1%} of {total / 1e6:.0f} MB")
    tmp.rename(DEST)
    print(f"\nsaved {DEST}")


if __name__ == "__main__":
    main()
