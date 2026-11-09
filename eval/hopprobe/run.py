"""Run HopProbe with the walk on and off (same stored memory, same answering model).

    python -m eval.hopprobe.run --dsn postgresql://memx:memx@localhost:5433/memx_hop

Results: eval/hopprobe/results/<run>-<config>.jsonl and a summary with the paired comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from datetime import timedelta
from pathlib import Path

from eval.hopprobe import fixtures as fx
from eval.longmemeval.run import ANSWER_PROMPT, DATA, chat
from memx import Memory, MemxConfig, Messages, Text, Turn
from memx.providers.http import OpenRouterHTTP

RESULTS = Path(__file__).parent / "results"
FILLER_FROM = 5
FILLER_SESSIONS = 16
CONFIGS = {"walk": {"walk_enabled": True}, "no_walk": {"walk_enabled": False}}


async def ingest(mem: Memory) -> None:
    await mem.insert(fx.SCOPE, Text(text=fx.handbook(), source="directory"), observed_at=fx.OBSERVED,
                     idempotency_key=f"{fx.SCOPE}:directory", mode="inline")
    item = json.loads(DATA.read_text())[FILLER_FROM]
    answer_ids = set(item.get("answer_session_ids") or [])
    sessions = [s for sid, s in zip(item["haystack_session_ids"], item["haystack_sessions"]) if sid not in answer_ids]
    for i, session in enumerate(sessions[:FILLER_SESSIONS]):
        turns = [Turn(role=t["role"], content=t["content"]) for t in session if t.get("content")]
        await mem.insert(fx.SCOPE, Messages(turns=turns), observed_at=fx.OBSERVED + timedelta(days=i + 1),
                         idempotency_key=f"{fx.SCOPE}:filler{i}", mode="inline")


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    b10 = sum(1 for x, y in zip(a, b) if x and not y)
    b01 = sum(1 for x, y in zip(a, b) if y and not x)
    n, k = b10 + b01, min(b10, b01)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"walk_only": b10, "no_walk_only": b01, "p": round(p, 4)}


async def main_async(args) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    base = MemxConfig(dsn=args.dsn, insert_mode="inline", extractor_model=args.extract_model)
    mem = await Memory.connect(base, start_runner=False)
    t = time.perf_counter()
    await ingest(mem)
    print(f"ingest {time.perf_counter() - t:.0f}s")
    await mem.close()
    http = OpenRouterHTTP(base.api_key(), base.openrouter_base_url, timeout_s=120)
    scored: dict[str, list[bool]] = {}
    for name, overrides in CONFIGS.items():
        m = await Memory.connect(MemxConfig(dsn=args.dsn, insert_mode="inline", **overrides), start_runner=False,
                                 run_migrations=False)
        rows = []
        for q in fx.questions():
            r = await m.recall(fx.SCOPE, q["q"], as_of=fx.QUESTION_DATE, force_mode="ranked",
                               budget_tokens=args.budget, explain=True)
            answer, _ = await chat(http, args.answer_model, ANSWER_PROMPT.format(
                memory=r.markdown or "(no relevant memories)", question_date=f"{fx.QUESTION_DATE:%Y/%m/%d}",
                question=q["q"]))
            walked = [c for c in (r.explain or {}).get("candidates", [])
                      if c.get("found_by") in ("entity_hop", "topic_hop") and c.get("status") == "picked"]
            ok = fx.correct(answer, q)
            rows.append({"config": name, "id": q["id"], "question": q["q"], "expected": q["answer"],
                         "response": answer, "correct": ok, "answer_in_memory": q["answer"].lower() in
                         (r.markdown or "").lower(), "tokens": r.token_count, "walked_picked": len(walked)})
            print(f"  {name:<8} {q['id']:<16} {'✓' if ok else '✗'} tokens={r.token_count} walked={len(walked)}")
        (RESULTS / f"{args.run}-{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
        scored[name] = [x["correct"] for x in rows]
        await m.close()
    await http.aclose()
    summary = {k: {"correct": sum(v), "n": len(v)} for k, v in scored.items()}
    summary["paired"] = mcnemar(scored["walk"], scored["no_walk"])
    (RESULTS / f"{args.run}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    p.add_argument("--run", default="paper")
    p.add_argument("--budget", type=int, default=2000)
    p.add_argument("--extract-model", default="anthropic/claude-haiku-5.5")
    p.add_argument("--answer-model", default="anthropic/claude-haiku-5.5")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
