"""LoCoMo harness for memx.

LoCoMo (snap-research/locomo) has 10 long conversations between two named people, 19-32 sessions each, and 1,986
questions about them. Per conversation: ingest every session in order into its own scope (one ingest per session,
observed at the session's date, each speaker kept as a named participant), then ask each question with ranked
recall as of the day after the last session, answer from the memory block, and judge the answer.

The adversarial category (5: questions about things that never happened, with no gold answer) is skipped by
default, as in most published LoCoMo results; pass --categories 1 2 3 4 5 to include it.

    python -m eval.locomo.run                       # all 10 conversations, categories 1-4 (1,540 questions)
    python -m eval.locomo.run --config ranked_fill full     # other recall configurations, same ingest
    python -m eval.locomo.run --conversations 1 --limit 20      # smoke run
    python -m eval.locomo.run --resume <run id>     # finish an interrupted run

Benchmark memory lives in its own database (default memx_locomo). Ingests are keyed by session, so a resumed or
repeated run reuses them and only pays for recall, answering and judging.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import time
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from memx import Memory, MemxConfig, Messages, Turn
from memx.providers.http import OpenRouterHTTP

HERE = Path(__file__).parent
DATA = HERE / "data" / "locomo10.json"
RESULTS = HERE / "results"
SOURCE = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DSN = os.environ.get("MEMX_LOCOMO_DSN", "postgresql://memx:memx@localhost:5433/memx_locomo")

CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
# Recall configurations. "memx" holds MemxConfig overrides for the recall side only; every config reads the same
# ingested memory.
CONFIGS: dict[str, dict[str, Any]] = {
    "ranked_4k": {"recall": {"force_mode": "ranked", "budget_tokens": 4_000}},
    # Relevance cut lowered from 0.30 to 0.10: the block fills with the best-scoring facts up to the budget, and only
    # facts Jev rates clearly irrelevant are left out.
    "ranked_fill": {"recall": {"force_mode": "ranked", "budget_tokens": 4_000}, "memx": {"grade_floor": 0.1}},
    # Wider ranked recall: more search candidates, more graded, up to 80 facts (the 30-fact cap was binding).
    "ranked_wide": {"recall": {"force_mode": "ranked", "budget_tokens": 4_000},
                    "memx": {"grade_floor": 0.1, "recall_k_vector": 80, "recall_k_fts": 30, "grade_max": 90,
                             "recall_k_max": 80}},
    # The whole memory of the conversation: the ceiling when retrieval is not the problem.
    "full": {"recall": {"force_mode": "full", "budget_tokens": 200_000}},
}

ANSWER_PROMPT = """You have a long-term memory of past conversations between {a} and {b}.

{memory}

The conversations took place between {first} and {last}.

Answer the question from the memory above. Start with the answer itself, in a few words (a name, number, date or
short phrase); add a short reason only if the question asks for one. The memory may use different words than the
question ("held" a game when the question says "organized" it): use the fact that matches best. Draw reasonable
conclusions from what the memory says (someone who had a breakup and plans to adopt alone is single). Write dates as
absolute dates (for example "7 May 2023"). Answer "I don't know" only when nothing in the memory relates to the
question.

Question: {question}
Answer:"""

JUDGE_PROMPT = """You are grading an answer to a question about past conversations between two people.

Question: {question}
Gold answer: {gold}
Generated answer: {response}

Label the generated answer CORRECT if it gives the same information as the gold answer. Be generous about wording,
length and format: a longer answer that contains the gold information is CORRECT. For dates and times, the answer is
CORRECT when it refers to the same day, month or year as the gold answer, or names a date inside the period the gold
answer describes ("2 June 2023" for "the week before 9 June 2023"). Label it WRONG if it contradicts the gold answer,
misses it, or says it does not know.

Reply with one word: CORRECT or WRONG."""


def load() -> list[dict[str, Any]]:
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE, DATA)  # noqa: S310 - fixed public URL
    return json.loads(DATA.read_text())


def parse_when(text: str) -> datetime:
    """LoCoMo session dates look like "1:56 pm on 8 May, 2023"."""
    return datetime.strptime(text.strip(), "%I:%M %p on %d %B, %Y").replace(tzinfo=UTC)


def sessions(conv: dict[str, Any]) -> list[tuple[int, datetime, list[dict[str, Any]]]]:
    out = []
    n = 1
    while f"session_{n}" in conv:
        out.append((n, parse_when(conv[f"session_{n}_date_time"]), conv[f"session_{n}"]))
        n += 1
    return out


def turn_text(turn: dict[str, Any]) -> str:
    text = turn.get("text") or ""
    if turn.get("blip_caption"):  # the speaker shared a photo; the caption is all a text system can see of it
        text = f"{text} [shares a photo: {turn['blip_caption']}]".strip()
    return text


async def ingest_conversation(mem: Memory, scope: str, conv: dict[str, Any]) -> None:
    for n, when, turns in sessions(conv):
        msgs = [Turn(role=t["speaker"], content=turn_text(t)) for t in turns if turn_text(t)]
        if not msgs:
            continue
        for attempt in range(3):
            try:
                # Resending the same key resumes a failed ingest from its saved progress.
                await mem.insert(scope, Messages(turns=msgs), observed_at=when, idempotency_key=f"{scope}:session_{n}",
                                 mode="inline")
                break
            except Exception:  # noqa: BLE001 - a dropped connection should not end a long run
                if attempt == 2:
                    raise
                await asyncio.sleep(5 * (attempt + 1))


async def chat(http: OpenRouterHTTP, model: str, prompt: str, max_tokens: int = 2000) -> str:
    """One completion. Models that think before answering (Haiku 5.5, Sonnet 5.5) spend output tokens on reasoning
    first, so the budget must leave room for the answer; if it still comes back empty, ask again without thinking."""
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    data = await http.post("/chat/completions", body)
    text = (data["choices"][0]["message"].get("content") or "").strip()
    if not text:
        data = await http.post("/chat/completions", {**body, "reasoning": {"enabled": False}})
        text = (data["choices"][0]["message"].get("content") or "").strip()
    return text


def judged_correct(verdict: str) -> bool:
    """The judge may reason before its label; the last CORRECT or WRONG wins."""
    found = re.findall(r"\b(CORRECT|WRONG)\b", verdict.upper())
    return bool(found) and found[-1] == "CORRECT"


async def evidence_recall(mem: Memory, scope: str, evidence: list[str], fact_ids: list) -> float | None:
    wanted = {f"session_{e.split(':')[0].lstrip('D')}" for e in evidence if e.startswith("D") and ":" in e}
    if not wanted:
        return None
    async with mem.ctx.pool.connection() as conn:
        cur = await conn.execute(
            "SELECT DISTINCT i.idempotency_key FROM memx_fact f JOIN memx_ingest i ON i.id = f.ingest_id "
            "WHERE f.id = ANY(%s)", (fact_ids,))
        hit = {r["idempotency_key"].removeprefix(f"{scope}:") for r in await cur.fetchall()}
    return len(hit & wanted) / len(wanted)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    ms = sorted(r["recall_ms"] for r in rows)
    ev = [r["evidence_recall"] for r in rows if r["evidence_recall"] is not None]
    pick = lambda q: ms[min(len(ms) - 1, math.ceil(q * len(ms)) - 1)] if ms else None  # noqa: E731
    return {
        "questions": len(rows),
        "accuracy": round(sum(r["correct"] for r in rows) / max(1, len(rows)), 4),
        "by_category": {c: {"n": len(v), "accuracy": round(sum(r["correct"] for r in v) / len(v), 4)}
                        for c, v in sorted(by_cat.items())},
        "evidence_recall": round(statistics.mean(ev), 4) if ev else None,
        "memory_tokens_mean": round(statistics.mean(r["memory_tokens"] for r in rows), 1) if rows else 0,
        "recall_ms_p50": pick(0.5), "recall_ms_p95": pick(0.95),
    }


async def ingest_cost(mem: Memory, scope: str) -> dict[str, Any]:
    async with mem.ctx.pool.connection() as conn:
        cur = await conn.execute(
            "SELECT count(*) AS ingests, coalesce(sum(tokens_in), 0) AS tin, coalesce(sum(tokens_out), 0) AS tout, "
            "coalesce(sum((receipt->'usage'->>'jev_cost_usd')::float), 0) AS jev_usd, "
            "coalesce(sum((receipt->'usage'->>'jev_requests')::int), 0) AS jev_requests "
            "FROM memx_ingest WHERE scope = %s::ltree", (scope,))
        row = await cur.fetchone()
        cur = await conn.execute("SELECT count(*) AS n FROM memx_fact WHERE scope = %s::ltree AND status = 'active'",
                                 (scope,))
        facts = (await cur.fetchone())["n"]
    return {**{k: (float(v) if k == "jev_usd" else int(v)) for k, v in row.items()}, "active_facts": int(facts)}


async def main_async(args: argparse.Namespace) -> None:
    data = load()[: args.conversations or None]
    if args.samples:
        data = [d for d in data if d["sample_id"] in set(args.samples)]
    run_id = args.resume or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    RESULTS.mkdir(parents=True, exist_ok=True)
    for cfg_name in args.config:
        await run_config(args, data, run_id, cfg_name)


async def run_config(args: argparse.Namespace, data: list[dict], run_id: str, cfg_name: str) -> None:
    spec = CONFIGS[cfg_name]
    out = RESULTS / f"{run_id}-{cfg_name}.jsonl"
    previous = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    done = {(r["sample_id"], r["index"]) for r in previous}

    mem = await Memory.connect(MemxConfig(dsn=args.dsn, insert_mode="inline", **spec.get("memx", {})),
                               start_runner=False)
    http = OpenRouterHTTP(mem.ctx.config.api_key(), mem.ctx.config.openrouter_base_url, timeout_s=120)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    costs: dict[str, dict] = {}

    async def ask(sample: dict, scope: str, index: int, qa: dict, as_of: datetime, first: str, last: str) -> dict | None:
        conv = sample["conversation"]
        async with sem:
            try:
                t0 = time.perf_counter()
                recall = await mem.recall(scope, qa["question"], as_of=as_of, **spec["recall"])
                t1 = time.perf_counter()
                answer = await chat(http, args.answer_model, ANSWER_PROMPT.format(
                    a=conv["speaker_a"], b=conv["speaker_b"], memory=recall.markdown or "(no relevant memories)",
                    first=first, last=last, question=qa["question"]))
                gold = qa.get("answer", qa.get("adversarial_answer", ""))
                verdict = await chat(http, args.judge_model, JUDGE_PROMPT.format(
                    question=qa["question"], gold=gold, response=answer), max_tokens=400)
            except Exception as exc:  # noqa: BLE001 - one bad question should not sink the run
                print(f"  ! {sample['sample_id']}#{index}: {type(exc).__name__}: {exc}")
                return None
        row = {
            "run_id": run_id, "config": cfg_name, "sample_id": sample["sample_id"], "index": index,
            "category": CATEGORIES[qa["category"]], "question": qa["question"], "answer": gold, "response": answer,
            "correct": judged_correct(verdict), "judge": verdict[-80:],
            "facts": len(recall.facts), "memory_tokens": recall.token_count, "recall_ms": round((t1 - t0) * 1000),
            "evidence_recall": await evidence_recall(mem, scope, qa.get("evidence") or [], [f.id for f in recall.facts]),
        }
        async with lock:
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        return row

    async def conversation(sample: dict) -> list[dict]:
        conv = sample["conversation"]
        tag = f"{args.ingest_tag}." if args.ingest_tag else ""  # a new tag re-ingests into fresh scopes
        scope = f"memory.locomo.{tag}{sample['sample_id'].replace('-', '_')}"
        t0 = time.perf_counter()
        await ingest_conversation(mem, scope, conv)
        costs[sample["sample_id"]] = {**await ingest_cost(mem, scope), "ingest_s": round(time.perf_counter() - t0)}
        spans = sessions(conv)
        first, last = spans[0][1].strftime("%d %B %Y"), spans[-1][1].strftime("%d %B %Y")
        as_of = spans[-1][1] + timedelta(days=1)  # questions are asked after the last session
        todo = [(i, qa) for i, qa in enumerate(sample["qa"])
                if qa["category"] in args.categories and (sample["sample_id"], i) not in done]
        if args.limit:
            todo = todo[: args.limit]
        print(f"== {cfg_name} {sample['sample_id']}: ingested ({costs[sample['sample_id']]}), {len(todo)} questions")
        rows = await asyncio.gather(*(ask(sample, scope, i, qa, as_of, first, last) for i, qa in todo))
        return [r for r in rows if r]

    new = [r for rows in await asyncio.gather(*(conversation(s) for s in data)) for r in rows]
    rows = previous + new
    ingest = {k: sum(c[k] for c in costs.values()) for k in ("ingests", "tin", "tout", "jev_usd", "jev_requests",
                                                              "active_facts")} if costs else {}
    summary = {"run_id": run_id, "config": cfg_name, "conversations": len(data), "ingest": ingest,
               **summarize(rows)}
    (RESULTS / f"{run_id}-{cfg_name}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    await http.aclose()
    await mem.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", default=["ranked_4k"], choices=sorted(CONFIGS))
    p.add_argument("--conversations", type=int, default=0, help="only the first N conversations (default: all 10)")
    p.add_argument("--samples", nargs="*", default=None, help="only these sample ids (e.g. conv-44 conv-50)")
    p.add_argument("--categories", type=int, nargs="+", default=[1, 2, 3, 4], help="1-4 by default; 5 = adversarial")
    p.add_argument("--limit", type=int, default=0, help="at most N questions per conversation (smoke runs)")
    p.add_argument("--concurrency", type=int, default=12, help="questions answered at once")
    p.add_argument("--answer-model", default="anthropic/claude-haiku-5.5")
    p.add_argument("--judge-model", default="anthropic/claude-sonnet-5.5")
    p.add_argument("--resume", default=None, help="run id to complete: skips questions already in its results")
    p.add_argument("--dsn", default=DSN, help="database for benchmark memory (default: memx_locomo)")
    p.add_argument("--ingest-tag", default="", help="ingest into memory.locomo.<tag>.<conversation>, e.g. after an "
                   "extraction change (default: the original scopes)")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
