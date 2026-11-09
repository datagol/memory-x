"""No-memory baselines on the same LongMemEval subset: full context and plain chunk retrieval.

* ``full_context``: the answering model reads every haystack session (dated, in order) — the whole ~115k-token history.
* ``chunk_rag``: sessions split into ~1,000-token chunks (dated), embedded with the same embedding model as memx and
  mem0; the chunks most similar to the question are given to the answering model, up to ``--rag-budget`` tokens,
  in chronological order.

No extraction, no memory system. Same questions, answering model, answer prompt and judge as ``run.py``.

    python -m eval.longmemeval.run_baselines --config full_context chunk_rag --per-type 12 --abstention 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

from eval.longmemeval.run import (
    ANSWER_PROMPT, DATA, RESULTS, chat, judge_prompt, judged_yes, select, summarize,
)
from memx import MemxConfig
from memx.providers.http import OpenRouterHTTP

CHARS_PER_TOKEN = 3.6
CACHE = RESULTS / "baseline_embeddings"


def tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def session_text(date: str, session: list[dict]) -> list[str]:
    return [f"{t['role']}: {t['content']}" for t in session if t.get("content")]


def full_history(item: dict) -> str:
    parts = []
    for i, (date, session) in enumerate(zip(item["haystack_dates"], item["haystack_sessions"]), 1):
        parts.append(f"### Session {i} ({date})\n" + "\n".join(session_text(date, session)))
    return "## Conversation history\n\n" + "\n\n".join(parts)


def chunks(item: dict, chunk_tokens: int) -> list[dict]:
    """~chunk_tokens-sized chunks of whole turns (a single long turn is cut), each tagged with its session date."""
    out: list[dict] = []
    limit = int(chunk_tokens * CHARS_PER_TOKEN)
    for si, (date, session) in enumerate(zip(item["haystack_dates"], item["haystack_sessions"])):
        buf: list[str] = []
        size = 0
        for line in session_text(date, session):
            pieces = [line[i:i + limit] for i in range(0, len(line), limit)] or [line]
            for piece in pieces:
                if buf and size + len(piece) > limit:
                    out.append({"session": si, "date": date, "text": "\n".join(buf)})
                    buf, size = [], 0
                buf.append(piece)
                size += len(piece)
        if buf:
            out.append({"session": si, "date": date, "text": "\n".join(buf)})
    return out


async def embed(http: OpenRouterHTTP, model: str, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), 96):
        data = await http.post("/embeddings", {"model": model, "input": texts[i:i + 96]})
        vectors += [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
    return vectors


def cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1.0)


async def rag_block(http: OpenRouterHTTP, cfg: MemxConfig, item: dict, args) -> tuple[str, list[dict]]:
    parts = chunks(item, args.chunk_tokens)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{item['question_id']}-{args.chunk_tokens}.json"
    if cache.exists():
        vectors = json.loads(cache.read_text())
    else:
        vectors = await embed(http, cfg.embedding_model, [f"({c['date']}) {c['text']}" for c in parts])
        cache.write_text(json.dumps(vectors))
    [q] = await embed(http, cfg.embedding_model, [item["question"]])
    ranked = sorted(range(len(parts)), key=lambda i: cos(q, vectors[i]), reverse=True)
    picked, used = [], 0
    for i in ranked:
        t = tokens(parts[i]["text"]) + 12
        if used + t > args.rag_budget:
            continue
        picked.append(i)
        used += t
    picked.sort(key=lambda i: (parts[i]["session"], i))  # chronological
    body = "\n\n".join(f"### Excerpt ({parts[i]['date']})\n{parts[i]['text']}" for i in picked)
    return "## Relevant excerpts from past conversations\n\n" + body, [parts[i] for i in picked]


async def run_one(http: OpenRouterHTTP, cfg: MemxConfig, item: dict, cfg_name: str, run_id: str, args) -> dict:
    t0 = time.perf_counter()
    evidence = None
    if cfg_name == "full_context":
        memory = full_history(item)
    else:
        memory, picked = await rag_block(http, cfg, item, args)
        answer_sessions = set(item.get("answer_session_ids") or [])
        if answer_sessions:
            hit = {item["haystack_session_ids"][c["session"]] for c in picked}
            evidence = len(hit & answer_sessions) / len(answer_sessions)
    t1 = time.perf_counter()
    answer, usage = await chat(http, args.answer_model, ANSWER_PROMPT.format(
        memory=memory, question_date=item.get("question_date", ""), question=item["question"]))
    verdict_text, _ = await chat(http, args.judge_model, judge_prompt(item, answer), max_tokens=400)
    return {
        "run_id": run_id, "config": cfg_name, "question_id": item["question_id"],
        "question_type": "abstention" if item["question_id"].endswith("_abs") else item["question_type"],
        "question": item["question"], "answer": item["answer"], "response": answer,
        "correct": judged_yes(verdict_text), "judge": verdict_text[-200:], "mode": cfg_name, "facts": 0,
        "memory_tokens": tokens(memory), "evidence_recall": evidence if cfg_name == "chunk_rag" else 1.0,
        "ingest_s": 0.0, "recall_ms": round((t1 - t0) * 1000),
        "usage": {"answer_prompt_tokens": usage.get("prompt_tokens"), "answer_cost": usage.get("cost")},
    }


async def main_async(args) -> None:
    items = json.loads(DATA.read_text())
    chosen = select(items, args.per_type, args.abstention, args.seed)
    if args.limit:
        chosen = chosen[: args.limit]
    run_id = args.resume or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    cfg = MemxConfig()
    http = OpenRouterHTTP(cfg.api_key(), cfg.openrouter_base_url, timeout_s=300)
    for cfg_name in args.config:
        out = RESULTS / f"{run_id}-{cfg_name}.jsonl"
        previous = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
        answered = {r["question_id"] for r in previous}
        todo = [i for i in chosen if i["question_id"] not in answered]
        sem = asyncio.Semaphore(args.concurrency)

        async def guarded(item: dict, cfg_name: str = cfg_name, out: Path = out) -> dict | None:
            async with sem:
                try:
                    row = await run_one(http, cfg, item, cfg_name, run_id, args)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! {item['question_id']}: {type(exc).__name__}: {exc}")
                    return None
                with out.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"  {'✓' if row['correct'] else '✗'} {row['question_type']:<26} {row['question_id']}")
                return row

        print(f"== {cfg_name}: {len(todo)} questions ({len(answered)} already answered)")
        rows = previous + [r for r in await asyncio.gather(*(guarded(i) for i in todo)) if r]
        summary = {"run_id": run_id, "config": cfg_name, **summarize(rows),
                   "answer_cost_usd": round(sum((r["usage"].get("answer_cost") or 0) for r in rows), 4)}
        (RESULTS / f"{run_id}-{cfg_name}.summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    await http.aclose()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", default=["chunk_rag"], choices=["full_context", "chunk_rag"])
    p.add_argument("--per-type", type=int, default=12)
    p.add_argument("--abstention", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--chunk-tokens", type=int, default=1000)
    p.add_argument("--rag-budget", type=int, default=4000, help="memory block budget, same as memx ranked_4k")
    p.add_argument("--answer-model", default="anthropic/claude-haiku-5.5")
    p.add_argument("--judge-model", default="anthropic/claude-sonnet-5.5")
    p.add_argument("--resume", default=None)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
