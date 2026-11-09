"""Counterfactual recalls over the already-ingested LongMemEval scopes (no new extraction).

Two diagnostics for the paper, both in full-memory mode so retrieval is not a factor:

* ``full_history``: every fact, with replaced ones shown as earlier versions, to test whether hidden history explains
  the gap on "previous ..." and counting questions;
* ``full_active``: the control, active facts only, rendered the same simple way as the two variants;
* ``full_floor90``: facts replaced by a Jev verdict below 0.9 count as active, as if destructive verdicts needed 0.9.

    python -m eval.longmemeval.counterfactual
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict

import psycopg
from psycopg.rows import dict_row

from eval.longmemeval.run import (ANSWER_PROMPT, DATA, RESULTS, chat, judge_prompt, judged_yes, parse_date,
                                  scope_label, select, summarize)
from memx import MemxConfig
from memx.providers.http import OpenRouterHTTP
from memx.types import RENDER_ORDER, SECTION_TITLES

WHICH = os.environ.get("CF_CONFIGS", "full_active,full_history,full_floor90").split(",")
DSN = os.environ.get("MEMX_BENCH_DSN", "postgresql://memx:memx@localhost:5433/memx_bench2")


def facts_for(conn, scope: str, as_of) -> list[dict]:
    return conn.execute(
        "SELECT id, text, category, status, observed_at, event_at, flags FROM memx_fact WHERE scope = %s::ltree "
        "AND observed_at <= %s ORDER BY observed_at", (scope, as_of)).fetchall()


def replace_probs(conn, scope: str) -> dict:
    rows = conn.execute("""
        SELECT CASE WHEN result->>'label' = 'candidate_replaces' THEN object_fact_id ELSE subject_fact_id END AS fid,
               max((result->>'probability')::float) AS p
        FROM memx_decision WHERE step = 'verdict' AND scope = %s
          AND result->>'label' IN ('candidate_replaces', 'neighbor_covers') GROUP BY 1""", (scope,)).fetchall()
    return {r["fid"]: r["p"] for r in rows}


def render(scope: str, facts: list[dict], history: bool) -> str:
    lines = [f"# Memory: {scope}"]
    for cat in RENDER_ORDER:
        sec = [f for f in facts if f["category"] == cat.value and not (f["flags"] or {}).get("injection_suspect")]
        if not sec:
            continue
        lines += ["", f"## {SECTION_TITLES[cat]}"]
        for f in sec:
            text = f["text"]
            if cat.value == "event":
                text = f"({(f['event_at'] or f['observed_at']):%Y-%m-%d}) {text}"
            if history and f["status"] == "superseded":
                text = f"{text} (earlier; later replaced, said {f['observed_at']:%Y-%m-%d})"
            lines.append(f"- {text}")
    return "\n".join(lines) + "\n"


async def main() -> None:
    items = json.loads(DATA.read_text())
    chosen = select(items, 12, 10, 7)
    cfg = MemxConfig()
    http = OpenRouterHTTP(cfg.api_key(), cfg.openrouter_base_url, timeout_s=180)
    sem = asyncio.Semaphore(6)
    out = defaultdict(list)
    stats = defaultdict(int)

    async def one(item: dict, name: str, memory: str) -> None:
        async with sem:
            for attempt in range(3):
                try:
                    answer, _ = await chat(http, "anthropic/claude-haiku-5.5", ANSWER_PROMPT.format(
                        memory=memory, question_date=item.get("question_date", ""), question=item["question"]))
                    verdict, _ = await chat(http, "anthropic/claude-sonnet-5.5", judge_prompt(item, answer), 400)
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt == 2:
                        print("!", item["question_id"], name, exc)
                        return
            qt = "abstention" if item["question_id"].endswith("_abs") else item["question_type"]
            out[name].append({"config": name, "question_id": item["question_id"], "question_type": qt,
                              "correct": judged_yes(verdict), "response": answer, "memory_tokens": len(memory) // 4,
                              "facts": memory.count("\n- "), "evidence_recall": None, "recall_ms": 0})

    tasks = []
    with psycopg.connect(DSN, row_factory=dict_row) as conn:
        for item in chosen:
            scope = f"memory.lme.base.{scope_label(item['question_id'])}"
            as_of = parse_date(item.get("question_date", ""))
            facts = facts_for(conn, scope, as_of)
            probs = replace_probs(conn, scope)
            hist = facts
            floor = [f for f in facts if f["status"] == "active" or
                     (f["status"] == "superseded" and f["id"] in probs and probs[f["id"]] < 0.9)]
            stats["superseded"] += sum(f["status"] == "superseded" for f in facts)
            stats["restored_by_floor"] += sum(f["status"] == "superseded" for f in floor)
            active = [f for f in facts if f["status"] == "active"]
            variants = {"full_active": render(scope, active, False), "full_history": render(scope, hist, True),
                        "full_floor90": render(scope, floor, False)}
            for name in WHICH:
                tasks.append(one(item, name, variants[name]))
    await asyncio.gather(*tasks)
    for name, rows in out.items():
        (RESULTS / f"paper-{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        s = {"config": name, **summarize(rows)}
        (RESULTS / f"paper-{name}.summary.json").write_text(json.dumps(s, indent=2))
        print(name, s["questions"], s["accuracy"], s["by_type"])
    print(dict(stats))
    await http.aclose()


if __name__ == "__main__":
    asyncio.run(main())
