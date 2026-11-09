"""Fit decision floors from labelled decisions.

For each decision type, finds the lowest Jev probability at which acting on Jev's answer agrees with the labels at
least ``--target`` of the time, using the lower end of a Wilson 95% interval so a handful of lucky labels cannot
set a floor. Prints the suggested MEMX_* settings.

    python -m eval.calibration.fit                         # human labels (results/human_labels.jsonl)
    python -m eval.calibration.fit --labels reference      # the Claude Sonnet 5.5 reference labels instead

Re-run whenever ``jev_model`` or a question wording version changes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).parent / "results"
SETTING = {"candidate_replaces": "MEMX_VERDICT_REPLACE_MIN", "neighbor_covers": "MEMX_VERDICT_COVERS_MIN",
           "same": "MEMX_VERDICT_SAME_MIN", "entity": "MEMX_ENTITY_MERGE_MIN"}
GRID = [round(0.5 + 0.05 * i, 2) for i in range(10)]


def wilson_low(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den


def load(source: str) -> list[dict]:
    if source == "human":
        rows = [json.loads(x) for x in (HERE / "human_labels.jsonl").read_text().splitlines()]
        return [{"type": r["jev"] if r["kind"] == "verdict" else "entity", "p": float(r["jev_p"]),
                 "agree": r["human"] == r["jev"]} for r in rows]
    rows = [json.loads(x) for f in ("reference_destructive.jsonl", "reference_labels.jsonl")
            for x in (HERE / f).read_text().splitlines()]
    out, seen = [], set()
    for r in rows:
        if r["decision_id"] in seen or r.get("ref") is None:
            continue
        seen.add(r["decision_id"])
        if r["kind"] == "verdict" and r["jev"] in SETTING:
            out.append({"type": r["jev"], "p": float(r["jev_p"]), "agree": r["ref"] == r["jev"]})
        elif r["kind"] == "entity":
            p = float((r.get("probs") or {}).get("same", 0.0))
            out.append({"type": "entity", "p": p, "agree": r["ref"] == "same"})
    return out


def fit(rows: list[dict], target: float, min_n: int) -> dict:
    result = {}
    for kind in SETTING:
        sub = [r for r in rows if r["type"] == kind]
        table = []
        chosen = None
        for t in GRID:
            acted = [r for r in sub if r["p"] >= t]
            k = sum(r["agree"] for r in acted)
            low = wilson_low(k, len(acted))
            table.append({"floor": t, "acted": len(acted), "agree": k, "wilson_low": round(low, 3)})
            if chosen is None and len(acted) >= min_n and low >= target:
                chosen = t
        result[kind] = {"labelled": len(sub), "suggested_floor": chosen, "table": table}
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", choices=["human", "reference"], default="human")
    p.add_argument("--target", type=float, default=0.8, help="required agreement (Wilson lower bound)")
    p.add_argument("--min-n", type=int, default=10)
    args = p.parse_args()
    res = fit(load(args.labels), args.target, args.min_n)
    (HERE / f"fit_{args.labels}.json").write_text(json.dumps(res, indent=2))
    for kind, r in res.items():
        floor = r["suggested_floor"]
        print(f"{SETTING[kind]}={floor if floor is not None else 'no floor reaches the target'}"
              f"   # {r['labelled']} labelled")


if __name__ == "__main__":
    main()
