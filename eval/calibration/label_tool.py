"""Hand-label logged Jev decisions, so thresholds can be fitted to human judgement instead of a reference model.

Samples destructive reconcile verdicts (same / candidate replaces / neighbour covers) stratified by Jev's
probability, and entity merge decisions, from a benchmark database. Each item shows the two statements (or the two
names with their facts) and asks for the label a careful person would give. Labels are appended to
``results/human_labels.jsonl`` as you go, so a session can stop and resume.

    python -m eval.calibration.label_tool sample --n 300          # write results/label_queue.jsonl
    python -m eval.calibration.label_tool label                   # label in the terminal, resumable
    python -m eval.calibration.label_tool export > labels.csv     # or label in a spreadsheet ...
    python -m eval.calibration.label_tool import labels.csv       # ... and read it back

Then ``python -m eval.calibration.fit`` turns the labels into suggested floors.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

HERE = Path(__file__).parent / "results"
QUEUE = HERE / "label_queue.jsonl"
LABELS = HERE / "human_labels.jsonl"
DSN = os.environ.get("MEMX_BENCH_DSN", "postgresql://memx:memx@localhost:5433/memx_bench2")
BINS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
VERDICT_KEYS = {"s": "same", "r": "candidate_replaces", "c": "neighbor_covers", "l": "related", "u": "unrelated"}
ENTITY_KEYS = {"s": "same", "d": "different", "t": "cannot_tell"}


def sample(n: int, seed: str) -> list[dict]:
    per = max(1, n * 2 // 3 // len(BINS))
    ent_per = max(1, (n - per * len(BINS)) // len(BINS))
    out: list[dict] = []
    with psycopg.connect(DSN, row_factory=dict_row) as conn:
        for lo, hi in BINS:
            for r in conn.execute(
                """SELECT d.id, d.result FROM memx_decision d
                    WHERE d.step = 'verdict' AND d.result->>'label' IN ('same', 'candidate_replaces', 'neighbor_covers')
                      AND (d.result->>'probability')::float >= %s AND (d.result->>'probability')::float < %s
                    ORDER BY md5(d.id::text || %s) LIMIT %s""", (lo, hi, seed, per)).fetchall():
                res = r["result"]
                out.append({"kind": "verdict", "decision_id": r["id"], "jev": res["label"],
                            "jev_p": res["probability"], "a": res["candidate"], "b": res["neighbor"]})
            for r in conn.execute(
                """SELECT d.id, d.result FROM memx_decision d
                    WHERE d.step = 'entity' AND d.result ? 'with' AND (d.result->>'p_same')::float >= %s
                      AND (d.result->>'p_same')::float < %s
                    ORDER BY md5(d.id::text || %s) LIMIT %s""", (lo - 0.5 if lo == 0.5 else lo, hi, seed, ent_per)
            ).fetchall():
                res = r["result"]
                out.append({"kind": "entity", "decision_id": r["id"], "jev": "same", "jev_p": res["p_same"],
                            "how": res.get("how"), "a": res["name"], "b": res["with"]})
    return out


def done_ids() -> set[str]:
    if not LABELS.exists():
        return set()
    return {f"{x['kind']}:{x['decision_id']}" for x in map(json.loads, LABELS.read_text().splitlines())}


def label() -> None:
    queue = [json.loads(line) for line in QUEUE.read_text().splitlines()]
    seen = done_ids()
    todo = [q for q in queue if f"{q['kind']}:{q['decision_id']}" not in seen]
    print(f"{len(todo)} to label ({len(seen)} done). Ctrl-C to stop; progress is saved.")
    for i, q in enumerate(todo, 1):
        keys = VERDICT_KEYS if q["kind"] == "verdict" else ENTITY_KEYS
        print(f"\n[{i}/{len(todo)}] {q['kind']}  (Jev's answer is hidden)")
        print(f"  A: {q['a']}\n  B: {q['b']}")
        if q["kind"] == "verdict":
            print("  same | A replaces B (B outdated) | B covers A (A outdated/less complete) | related | unrelated")
        prompt = " / ".join(f"{k}={v}" for k, v in keys.items())
        while (ans := input(f"  {prompt}: ").strip().lower()) not in keys:
            pass
        with LABELS.open("a") as fh:
            fh.write(json.dumps({**q, "human": keys[ans]}) + "\n")


def export() -> None:
    w = csv.writer(sys.stdout)
    w.writerow(["kind", "decision_id", "a", "b", "human"])
    for q in map(json.loads, QUEUE.read_text().splitlines()):
        w.writerow([q["kind"], q["decision_id"], q["a"], q["b"], ""])


def import_csv(path: str) -> None:
    queue = {f"{q['kind']}:{q['decision_id']}": q for q in map(json.loads, QUEUE.read_text().splitlines())}
    seen = done_ids()
    added = 0
    with open(path) as fh, LABELS.open("a") as out:
        for row in csv.DictReader(fh):
            key = f"{row['kind']}:{row['decision_id']}"
            if row.get("human") and key in queue and key not in seen:
                out.write(json.dumps({**queue[key], "human": row["human"].strip()}) + "\n")
                added += 1
    print(f"imported {added} labels")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--seed", default="labels-1")
    sub.add_parser("label")
    sub.add_parser("export")
    i = sub.add_parser("import")
    i.add_argument("path")
    args = p.parse_args()
    if args.cmd == "sample":
        HERE.mkdir(exist_ok=True)
        rows = sample(args.n, args.seed)
        QUEUE.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"wrote {len(rows)} items to {QUEUE}")
    elif args.cmd == "label":
        label()
    elif args.cmd == "export":
        export()
    else:
        import_csv(args.path)


if __name__ == "__main__":
    main()
