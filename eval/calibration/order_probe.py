"""Does the reconcile verdict depend on the order its options are listed in?

Re-asks a sample of logged verdicts (one neighbor per request) twice: options in the shipped order and in a shuffled
order, and reports how often the chosen label changes and how far the chosen label's probability moves. A label that
flips with option order is a wording artefact, not a judgement.

    python -m eval.calibration.order_probe --n 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
from collections import Counter
from datetime import UTC
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from memx import MemxConfig
from memx.pipeline.reconcile import VERDICTS, verdict_question
from memx.providers.http import OpenRouterHTTP
from memx.providers.jev import Choice, envelope
from memx.timewords import recency_phrase

OUT = Path(__file__).parent / "results"
DSN = os.environ.get("MEMX_BENCH_DSN", "postgresql://memx:memx@localhost:5433/memx_bench2")
SCOPE_WORDS = {"same": "same scope", "ancestor": "broader (parent) scope", "descendant": "narrower (child) scope"}


def sample(n: int, seed: int) -> list[dict]:
    # Half destructive verdicts (the decisions that matter), half the rest.
    sql = """
        SELECT d.result, c.category AS c_cat, c.observed_at AS c_at, c.scope::text AS c_scope,
               nb.category AS n_cat, nb.observed_at AS n_at, nb.scope::text AS n_scope
          FROM memx_decision d
          JOIN memx_fact c ON c.id = d.subject_fact_id JOIN memx_fact nb ON nb.id = d.object_fact_id
         WHERE d.step = 'verdict' AND (d.result->>'label' IN ('candidate_replaces', 'neighbor_covers', 'same')) = %s
         ORDER BY md5(d.id::text || %s) LIMIT %s"""
    with psycopg.connect(DSN, row_factory=dict_row) as conn:
        destructive = conn.execute(sql, (True, str(seed), n // 2)).fetchall()
        other = conn.execute(sql, (False, str(seed), n - n // 2)).fetchall()
    return destructive + other


def state(r: dict) -> dict:
    rel = "same" if r["n_scope"] == r["c_scope"] else (
        "descendant" if r["n_scope"].startswith(r["c_scope"] + ".") else "ancestor")
    return envelope(
        {"candidate_category": r["c_cat"],
         "neighbors": [{"i": 0, "category": r["n_cat"], "scope_relation": SCOPE_WORDS[rel],
                        "recency": recency_phrase(r["c_at"].astimezone(UTC), r["n_at"].astimezone(UTC))}]},
        {"candidate": r["result"]["candidate"], "neighbors": [r["result"]["neighbor"]]},
    )


async def ask(http: OpenRouterHTTP, model: str, st: dict, q: Choice) -> dict:
    data = await http.post("/systemone", {"model": model, "state": st, "questions": {"n0": q.wire()}})
    return data["answers"]["n0"]


async def main_async(args) -> None:
    cfg = MemxConfig()
    http = OpenRouterHTTP(cfg.api_key(), cfg.openrouter_base_url, timeout_s=60)
    rng = random.Random(args.seed)
    rows = sample(args.n, args.seed)
    base = verdict_question(0)
    sem = asyncio.Semaphore(8)

    async def one(r: dict) -> dict:
        keys = list(VERDICTS)
        while keys == list(VERDICTS):
            rng.shuffle(keys)
        shuffled = Choice(instructions=base.instructions, criteria={k: VERDICTS[k] for k in keys})
        st = state(r)
        async with sem:
            a, b = await asyncio.gather(ask(http, cfg.jev_model, st, base), ask(http, cfg.jev_model, st, shuffled))
        pa, pb = a.get("probabilities") or {}, b.get("probabilities") or {}
        return {"logged": r["result"]["label"], "shipped": a["choice"], "shuffled": b["choice"], "order": keys,
                "p_shipped": pa.get(a["choice"]), "p_shuffled_same_label": pb.get(a["choice"]),
                "candidate": r["result"]["candidate"], "neighbor": r["result"]["neighbor"]}

    results = await asyncio.gather(*(one(r) for r in rows))
    await http.aclose()
    OUT.mkdir(exist_ok=True)
    (OUT / "order_probe.jsonl").write_text("".join(json.dumps(x, default=str) + "\n" for x in results))
    flips = [x for x in results if x["shipped"] != x["shuffled"]]
    destructive = ("candidate_replaces", "neighbor_covers", "same")
    d_rows = [x for x in results if x["shipped"] in destructive]
    d_flips = [x for x in d_rows if x["shipped"] != x["shuffled"]]
    shift = [abs(x["p_shipped"] - x["p_shuffled_same_label"]) for x in results
             if x["p_shipped"] is not None and x["p_shuffled_same_label"] is not None]
    summary = {
        "n": len(results), "flip_rate": round(len(flips) / len(results), 3),
        "destructive_n": len(d_rows), "destructive_flip_rate": round(len(d_flips) / max(1, len(d_rows)), 3),
        "median_abs_prob_shift": round(statistics.median(shift), 3) if shift else None,
        "p90_abs_prob_shift": round(sorted(shift)[int(0.9 * (len(shift) - 1))], 3) if shift else None,
        "flips": Counter(f"{x['shipped']}->{x['shuffled']}" for x in flips).most_common(),
    }
    (OUT / "order_probe.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=5)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
