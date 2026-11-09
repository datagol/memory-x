"""Does the narrower "can both be true at once?" question make borderline destructive verdicts safe to act on?

Takes the Sonnet-labelled destructive verdicts (reference_destructive.jsonl) whose Jev probability sits in the confirm
band (0.6 to just under the 0.9 floor), asks Jev the coexist question for each pair, and compares three policies
against the reference label:

- floor only (shipped): never act in the band;
- act on everything in the band (the old 0.5-floor behaviour);
- confirm: act only when P(coexist) <= the configured maximum.

    python -m eval.calibration.coexist_probe
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from memx import MemxConfig
from memx.pipeline.reconcile import coexist_question
from memx.providers.http import OpenRouterHTTP
from memx.providers.jev import envelope

HERE = Path(__file__).parent / "results"
DSN = os.environ.get("MEMX_BENCH_DSN", "postgresql://memx:memx@localhost:5433/memx_bench2")
DESTRUCTIVE = ("candidate_replaces", "neighbor_covers")


async def main_async() -> None:
    cfg = MemxConfig()
    lo, hi = cfg.verdict_confirm_band
    rows = [json.loads(line) for line in (HERE / "reference_destructive.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["jev"] in DESTRUCTIVE and lo <= r["jev_p"] < hi]
    with psycopg.connect(DSN, row_factory=dict_row) as conn:
        texts = {str(r["id"]): r["result"] for r in conn.execute(
            "SELECT id, result FROM memx_decision WHERE id = ANY(%s)",
            ([int(r["decision_id"]) for r in rows],)).fetchall()}
    http = OpenRouterHTTP(cfg.api_key(), cfg.openrouter_base_url, timeout_s=60)
    sem = asyncio.Semaphore(8)

    async def one(r: dict) -> dict:
        res = texts[r["decision_id"]]
        state = envelope({}, {"candidate": res["candidate"], "neighbors": [res["neighbor"]]})
        async with sem:
            data = await http.post("/systemone", {"model": cfg.jev_model, "state": state,
                                                  "questions": {"c0": coexist_question(0).wire()}})
        return {**r, "coexist": float(data["answers"]["c0"].get("noul", 1.0)),
                "candidate": res["candidate"], "neighbor": res["neighbor"]}

    out = await asyncio.gather(*(one(r) for r in rows))
    await http.aclose()
    (HERE / "coexist_probe.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))

    def agree(acted: list[dict]) -> dict:
        ok = sum(1 for x in acted if x["ref"] == x["jev"])
        return {"acted": len(acted), "reference_agrees": ok, "rate": round(ok / len(acted), 3) if acted else None}

    summary = {
        "band": [lo, hi], "n": len(out),
        "act_on_all": agree(out),
        "confirm": {str(m): agree([x for x in out if x["coexist"] <= m]) for m in (0.1, 0.2, 0.3, 0.5)},
        "reference_destructive_in_band": sum(1 for x in out if x["ref"] == x["jev"]),
    }
    (HERE / "coexist_probe.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
