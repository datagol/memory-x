"""Fetch BEAM-100K (20 conversations) from Hugging Face and BEAM's rubric judge prompt from the official repo.

    python -m eval.beam.download
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
PARQUET = "https://huggingface.co/datasets/Mohammadta/BEAM/resolve/main/data/100K-00000-of-00001.parquet"
PROMPTS = "https://raw.githubusercontent.com/mohammadtavakoli78/BEAM/main/src/prompts.py"


def main() -> None:
    (HERE / "data").mkdir(exist_ok=True)
    urllib.request.urlretrieve(PARQUET, HERE / "data" / "100K.parquet")
    src = urllib.request.urlopen(PROMPTS).read().decode()
    judge = re.search(r'unified_llm_judge_base_prompt = """(.*?)"""', src, re.S).group(1)
    (HERE / "judge_prompt.txt").write_text(judge)
    print("BEAM-100K and judge prompt saved under", HERE)


if __name__ == "__main__":
    main()
