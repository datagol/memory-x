"""Run the same LongMemEval subset against mem0 (open source), for comparison with memx.

Same questions (selection and seed), same extraction model (Haiku 5.5), same embeddings (text-embedding-3-small),
same answer model and judge. mem0 stores into pgvector in the same Postgres, one user_id per question.

    python -m eval.longmemeval.run_mem0 --per-type 12 --abstention 10 --top-k 30
"""

from __future__ import annotations

import os

os.environ.setdefault("MEM0_TELEMETRY", "False")  # keep benchmark data out of mem0's analytics

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mem0 import AsyncMemory  # noqa: E402

from eval.longmemeval.run import (
    ANSWER_PROMPT, BENCH_DSN, DATA, RESULTS, chat, judge_prompt, judged_yes, parse_date, scope_label, select, summarize,
)
from eval import bench_dsn
from memx import MemxConfig
from memx.providers.http import OpenRouterHTTP

PROGRESS = RESULTS / "mem0_progress"
USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


def _count_usage(_llm: Any, response: Any, _params: dict) -> None:
    usage = getattr(response, "usage", None)
    if usage is not None:
        USAGE["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        USAGE["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
    USAGE["calls"] += 1


PRICES = {"anthropic/claude-haiku-5.5": (0.10, 0.50)}
COLLECTION = "mem0_lme"
DB = "memx_bench"


def mem0_config(cfg: MemxConfig, model: str, collection: str = COLLECTION) -> dict[str, Any]:
    key = cfg.api_key()
    return {
        "llm": {"provider": "openai", "config": {
            "model": model, "api_key": key, "openai_base_url": cfg.openrouter_base_url, "temperature": 0,
            "max_tokens": 4000, "response_callback": _count_usage,
        }},
        "embedder": {"provider": "openai", "config": {
            "model": cfg.embedding_model, "api_key": key, "openai_base_url": cfg.openrouter_base_url,
            "embedding_dims": cfg.embedding_dim,
        }},
        "vector_store": {"provider": "pgvector", "config": {
            "connection_string": cfg.dsn, "collection_name": collection, "embedding_model_dims": cfg.embedding_dim,
            "hnsw": True,
        }},
        "history_db_path": str(RESULTS / "mem0_history.db"),
    }


async def ingest(mem: AsyncMemory, item: dict, uid: str) -> None:
    """One mem0.add per session, as mem0 is designed to be used. Progress is saved so reruns resume."""
    PROGRESS.mkdir(parents=True, exist_ok=True)
    # Keyed by database and collection: a marker must never outlive the data it describes (a dropped database).
    marker = PROGRESS / f"{DB}-{COLLECTION}-{uid}.json"
    done = set(json.loads(marker.read_text())) if marker.exists() else set()
    for sid, date, session in zip(item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"]):
        if sid in done:
            continue
        messages = [{"role": t["role"], "content": t["content"]} for t in session if t.get("content")]
        if messages:
            # OSS mem0 has no timestamp parameter (platform only), so the date goes into the conversation.
            day = parse_date(date).strftime("%Y-%m-%d (%A)")
            messages.insert(0, {"role": "system", "content": f"This conversation took place on {day}."})
            await mem.add(messages, user_id=uid, metadata={"session_id": sid, "session_date": day[:10]})
        done.add(sid)
        marker.write_text(json.dumps(sorted(done)))


def render(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["# Memory"]
    for r in results:
        when = (r.get("metadata") or {}).get("session_date", "")
        lines.append(f"- ({when}) {r['memory']}" if when else f"- {r['memory']}")
    return "\n".join(lines) + "\n"


async def run_one(mem: AsyncMemory, http: OpenRouterHTTP, item: dict, args) -> dict:
    uid = f"lme_{scope_label(item['question_id'])}"
    t0 = time.perf_counter()
    await ingest(mem, item, uid)
    t1 = time.perf_counter()
    found = await mem.search(item["question"], filters={"user_id": uid}, top_k=args.top_k)
    results = found.get("results", found) if isinstance(found, dict) else found
    t2 = time.perf_counter()
    memory = render(results)
    answer, _ = await chat(http, args.answer_model, ANSWER_PROMPT.format(
        memory=memory or "(no relevant memories)", question_date=item.get("question_date", ""),
        question=item["question"]))
    verdict, _ = await chat(http, args.judge_model, judge_prompt(item, answer), max_tokens=1000)
    sessions = {(r.get("metadata") or {}).get("session_id") for r in results}
    answer_sessions = set(item.get("answer_session_ids") or [])
    return {
        "config": f"mem0_top{args.top_k}", "question_id": item["question_id"],
        "question_type": "abstention" if item["question_id"].endswith("_abs") else item["question_type"],
        "question": item["question"], "answer": item["answer"], "response": answer, "judge": verdict[-200:],
        "correct": judged_yes(verdict), "mode": "search", "facts": len(results),
        "memory_tokens": len(memory) // 4 if memory else 0,
        "evidence_recall": len(sessions & answer_sessions) / len(answer_sessions) if answer_sessions else None,
        "ingest_s": round(t1 - t0, 2), "recall_ms": round((t2 - t1) * 1000),
    }


async def main_async(args) -> None:
    items = json.loads(DATA.read_text())
    chosen = select(items, args.per_type, args.abstention, args.seed)
    if args.ids:
        wanted = [line.strip() for line in Path(args.ids).read_text().splitlines() if line.strip()]
        by_id = {i["question_id"]: i for i in items}
        chosen = [by_id[q] for q in wanted]
    global COLLECTION, DB
    COLLECTION = args.collection
    DB = args.dsn.rsplit("/", 1)[-1]
    if args.limit:
        chosen = chosen[: args.limit]
    cfg = MemxConfig(dsn=bench_dsn(args.dsn))
    mem = AsyncMemory.from_config(mem0_config(cfg, args.extract_model, args.collection))
    http = OpenRouterHTTP(cfg.api_key(), cfg.openrouter_base_url, timeout_s=120)
    run_id = args.resume or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    name = f"mem0_top{args.top_k}"
    out = RESULTS / f"{run_id}-{name}.jsonl"
    previous = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    answered = {r["question_id"] for r in previous}
    chosen = [i for i in chosen if i["question_id"] not in answered]
    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(item: dict) -> dict | None:
        async with sem:
            try:
                row = await run_one(mem, http, item, args)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {item['question_id']}: {type(exc).__name__}: {exc}")
                return None
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"  {'✓' if row['correct'] else '✗'} {row['question_type']:<26} {row['question_id']}")
            return row

    print(f"== {name}: {len(chosen)} questions ({len(answered)} already answered)")
    rows = previous + [r for r in await asyncio.gather(*(guarded(i) for i in chosen)) if r]
    pin, pout = PRICES.get(args.extract_model, (1.0, 5.0))
    cost = USAGE["prompt_tokens"] / 1e6 * pin + USAGE["completion_tokens"] / 1e6 * pout
    summary = {"run_id": run_id, "config": name, **summarize(rows),
               "extract_usage": {**USAGE, "est_usd_this_run": round(cost, 2)}}
    (RESULTS / f"{run_id}-{name}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    await http.aclose()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-type", type=int, default=12)
    p.add_argument("--abstention", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--extract-model", default="anthropic/claude-haiku-5.5")
    p.add_argument("--answer-model", default="anthropic/claude-haiku-5.5")
    p.add_argument("--judge-model", default="anthropic/claude-sonnet-5.5")
    p.add_argument("--resume", default=None, help="run id to complete: skips questions already in its results")
    p.add_argument("--ids", default=None, help="file of question ids to run instead of the seeded sample")
    p.add_argument("--collection", default=COLLECTION, help="pgvector collection (keep extraction models apart)")
    p.add_argument("--dsn", default=BENCH_DSN, help="database for benchmark memory (default: memx_bench)")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
