"""LongMemEval-S harness for memx.

Per question: ingest every haystack session into its own scope (one ingest per session, observed at the session
date), recall with the question, answer with the extractor model from the memory block, and judge the answer.

    python -m eval.longmemeval.download
    python -m eval.longmemeval.run --config ranked_4k --per-type 25 --abstention 20

Ingested scopes are kept between runs (keyed by question id + extraction settings), so comparing recall
configurations only pays for extraction once.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import statistics
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval import bench_dsn
from memx import Memory, Messages, MemxConfig, Turn
from memx.providers.http import OpenRouterHTTP

HERE = Path(__file__).parent
# Benchmark data lives in its own database so it never mixes with real memory.
BENCH_DSN = os.environ.get("MEMX_BENCH_DSN", "postgresql://memx:memx@localhost:5433/memx_bench")
DATA = HERE / "data" / "longmemeval_s_cleaned.json"
RESULTS = HERE / "results"

TYPES = ["single-session-user", "single-session-assistant", "single-session-preference", "multi-session",
         "knowledge-update", "temporal-reasoning"]

# Recall settings per configuration. Ingest-affecting settings (hint, assistant facts) change the scope key.
CONFIGS: dict[str, dict[str, Any]] = {
    "dump_all": {"recall": {"force_mode": "full", "budget_tokens": 200_000}},
    "ranked_2k": {"recall": {"force_mode": "ranked", "budget_tokens": 2_000}},
    "ranked_4k": {"recall": {"force_mode": "ranked", "budget_tokens": 4_000}},
    "auto_4k": {"recall": {"budget_tokens": 4_000}},
    "no_walk": {"recall": {"force_mode": "ranked", "budget_tokens": 4_000}, "memx": {"walk_enabled": False}},
    "no_jev_grade": {"recall": {"force_mode": "ranked", "budget_tokens": 4_000}, "memx": {"grade_enabled": False}},
    "no_decay": {"recall": {"force_mode": "ranked", "budget_tokens": 4_000},
                 "memx": {"half_lives": {k: None for k in ("identity", "rule", "preference", "goal", "relationship",
                                                           "work_context", "event", "procedure", "other")}}},
    "no_hint": {"recall": {"force_mode": "ranked", "budget_tokens": 4_000}, "memx": {"sort_use_hint": False},
                "ingest_tag": "nohint"},
    # Reconcile policy before the accuracy fixes: 0.5 floors, no guards. Ingests into its own scopes.
    "legacy_policy": {"recall": {"force_mode": "ranked", "budget_tokens": 4_000}, "ingest_tag": "legacy",
                      "memx": {"verdict_same_min": 0.5, "verdict_min_prob": 0.5, "verdict_guards": False}},
    "legacy_policy_full": {"recall": {"force_mode": "full", "budget_tokens": 200_000}, "ingest_tag": "legacy",
                           "memx": {"verdict_same_min": 0.5, "verdict_min_prob": 0.5, "verdict_guards": False}},
    # Current policy (floors 0.9 / 0.8 and guards), with and without the borderline coexist question.
    "policy": {"recall": {"force_mode": "ranked", "budget_tokens": 4_000}, "ingest_tag": "policy"},
    "policy_full": {"recall": {"force_mode": "full", "budget_tokens": 200_000}, "ingest_tag": "policy"},
    "policy_confirm": {"recall": {"force_mode": "ranked", "budget_tokens": 4_000}, "ingest_tag": "confirm",
                       "memx": {"verdict_confirm": True}},
    # Cost cuts: lists as one fact, no general assistant explanations, slim schema, continuation instead of
    # re-extraction, prompt caching, chit-chat gate; recall shows replaced versions when the question asks.
    "lean": {"recall": {"force_mode": "ranked", "budget_tokens": 4_000}, "ingest_tag": "lean"},
    "lean_full": {"recall": {"force_mode": "full", "budget_tokens": 200_000}, "ingest_tag": "lean"},
    # Relevance cut 0.10 instead of 0.30: fill the ranked block with the best-scoring facts (helped on LoCoMo).
    # The pre-2026-11-08 recall defaults, for paired comparisons against runs made with them.
    "lean_narrow": {"recall": {"force_mode": "ranked", "budget_tokens": 4_000}, "ingest_tag": "lean",
                    "memx": {"grade_floor": 0.3, "recall_k_vector": 40, "recall_k_fts": 20, "grade_max": 40,
                             "recall_k_max": 30}},
    # Wide ranked recall, spelled out so the comparison does not move when the defaults do.
    "lean_wide": {"recall": {"force_mode": "ranked", "budget_tokens": 4_000}, "ingest_tag": "lean",
                  "memx": {"grade_floor": 0.1, "recall_k_vector": 80, "recall_k_fts": 30, "grade_max": 90,
                           "recall_k_max": 80}},
    # Floor 0.1 with the old 30-fact cap: the small-block option (pinned so it does not follow the defaults).
    "lean_fill": {"recall": {"force_mode": "ranked", "budget_tokens": 4_000}, "ingest_tag": "lean",
                  "memx": {"grade_floor": 0.1, "recall_k_vector": 40, "recall_k_fts": 20, "grade_max": 40,
                           "recall_k_max": 30}},
    "lean_nowalk": {"recall": {"force_mode": "ranked", "budget_tokens": 4_000}, "ingest_tag": "lean",
                    "memx": {"walk_enabled": False}},
    "no_assistant_facts": {"recall": {"force_mode": "ranked", "budget_tokens": 4_000},
                           "ingest_tag": "noasst", "drop_assistant": True},
}

ANSWER_PROMPT = """You are a helpful assistant with a long-term memory of past conversations with the user.

{memory}

Current date: {question_date}

Answer the user's question using the memory above. If the memory does not contain the information needed, say you don't know. When the user asks for suggestions, recommendations, tips or ideas, give them: build on what the memory records about the user's interests, background, plans and past activities, and say which remembered details you are building on. Count or compute from the remembered facts when the question asks for a number or a duration.

Question: {question}
Answer:"""

# Adapted from LongMemEval's evaluate_qa.py judge prompts.
JUDGE_DEFAULT = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
                 "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the "
                 "correct answer or contains all the intermediate steps to get the correct answer, you should also "
                 "answer yes. If the response only contains a subset of the information required by the answer, "
                 "answer no.")
JUDGE_EXTRA = {
    "temporal-reasoning": " In addition, do not penalize off-by-one errors for the number of days. If the question "
                          "asks for the number of days/weeks/months, etc., and the model makes off-by-one errors "
                          "(e.g., predicting 19 days when the answer is 18), the model's response is still correct.",
    "knowledge-update": " If the response contains some previous information along with an updated answer, the "
                        "response should be considered as correct as long as the updated answer is the required "
                        "answer.",
}
JUDGE_PREFERENCE = ("I will give you a question, a rubric for desired personalized response, and a response from a "
                    "model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
                    "The model does not need to reflect all the points in the rubric. The response is correct as "
                    "long as it recalls and utilizes the user's personal information correctly.")
JUDGE_ABSTENTION = ("I will give you an unanswerable question, an explanation, and a response from a model. Please "
                    "answer yes if the model correctly identifies the question as unanswerable. The model could say "
                    "that the information is incomplete, or some other information is given but the asked "
                    "information is not.")


def parse_date(text: str) -> datetime:
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})(?:\s*\(\w+\))?\s*(\d{2}):(\d{2})?", text)
    if not m:
        return datetime.now(UTC)
    y, mo, d, h, mi = m.groups()
    return datetime(int(y), int(mo), int(d), int(h), int(mi or 0), tzinfo=UTC)


def scope_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", text)[:64]


def select(items: list[dict], per_type: int, abstention: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    abst: list[dict] = []
    for item in items:
        (abst if item["question_id"].endswith("_abs") else by_type[item["question_type"]]).append(item)
    chosen = []
    for t in TYPES:
        pool = sorted(by_type[t], key=lambda x: x["question_id"])
        chosen += rng.sample(pool, min(per_type, len(pool)))
    chosen += rng.sample(sorted(abst, key=lambda x: x["question_id"]), min(abstention, len(abst)))
    return chosen


async def chat(http: OpenRouterHTTP, model: str, prompt: str, max_tokens: int = 2000) -> tuple[str, dict]:
    """Thinking models (Haiku 5.5, Sonnet 5.5) spend output tokens on reasoning first; leave room, and if the answer
    still comes back empty, ask again without thinking."""
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    data = await http.post("/chat/completions", body)
    text = (data["choices"][0]["message"].get("content") or "").strip()
    if not text:
        data = await http.post("/chat/completions", {**body, "reasoning": {"enabled": False}})
        text = (data["choices"][0]["message"].get("content") or "").strip()
    return text, data.get("usage") or {}


def judge_prompt(item: dict, response: str) -> str:
    if item["question_id"].endswith("_abs"):
        head, label = JUDGE_ABSTENTION, "Explanation"
    elif item["question_type"] == "single-session-preference":
        head, label = JUDGE_PREFERENCE, "Rubric"
    else:
        head, label = JUDGE_DEFAULT + JUDGE_EXTRA.get(item["question_type"], ""), "Correct Answer"
    return (f"{head}\n\nQuestion: {item['question']}\n\n{label}: {item['answer']}\n\n"
            f"Model Response: {response}\n\nIs the model response correct? Answer yes or no only.")


def judged_yes(text: str) -> bool:
    """Judges sometimes reason before answering; the last standalone yes/no wins."""
    found = re.findall(r"\b(yes|no)\b", text.lower())
    return bool(found) and found[-1] == "yes"


async def ingest_question(mem: Memory, item: dict, scope: str, drop_assistant: bool) -> None:
    for sid, date, session in zip(item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"]):
        turns = [Turn(role=t["role"], content=t["content"]) for t in session
                 if t.get("content") and not (drop_assistant and t["role"] == "assistant")]
        if not turns:
            continue
        await mem.insert(scope, Messages(turns=turns), observed_at=parse_date(date), idempotency_key=f"{scope}:{sid}",
                         mode="inline")


async def evidence_recall(mem: Memory, scope: str, item: dict, fact_ids: list) -> float | None:
    answer_sessions = set(item.get("answer_session_ids") or [])
    if not answer_sessions:
        return None
    async with mem.ctx.pool.connection() as conn:
        cur = await conn.execute(
            "SELECT DISTINCT i.idempotency_key FROM memx_fact f JOIN memx_ingest i ON i.id = f.ingest_id "
            "WHERE f.id = ANY(%s)", (fact_ids,))
        hit = {r["idempotency_key"].removeprefix(f"{scope}:") for r in await cur.fetchall()}
    return len(hit & answer_sessions) / len(answer_sessions)


async def ingest_usage(mem: Memory, scope: str) -> dict:
    """Extraction tokens and Jev cost for every ingest of a question (from receipts; older ingests lack Jev cost)."""
    async with mem.ctx.pool.connection() as conn:
        cur = await conn.execute(
            "SELECT coalesce(sum(tokens_in), 0) AS tin, coalesce(sum(tokens_out), 0) AS tout, count(*) AS ingests, "
            "coalesce(sum((receipt->'usage'->>'jev_cost_usd')::float), 0) AS jev_usd, "
            "coalesce(sum((receipt->'usage'->>'jev_requests')::int), 0) AS jev_requests, "
            "coalesce(sum((receipt->'usage'->>'extract_splits')::int), 0) AS splits, "
            "coalesce(sum((receipt->'usage'->>'extract_wasted_tokens_out')::int), 0) AS wasted_out, "
            "coalesce(sum((receipt->'usage'->>'extract_cached_tokens_in')::int), 0) AS cached_in, "
            "coalesce(sum((receipt->'usage'->>'extract_continuations')::int), 0) AS continuations, "
            "coalesce(sum((receipt->'usage'->>'gate_turns_skipped')::int), 0) AS gate_skipped, "
            "max(extractor_model) AS model FROM memx_ingest WHERE scope = %s::ltree", (scope,))
        row = await cur.fetchone()
        cur = await conn.execute("SELECT count(*) AS n FROM memx_fact WHERE scope = %s::ltree", (scope,))
        facts = (await cur.fetchone())["n"]
    return {"extract_tokens_in": int(row["tin"]), "extract_tokens_out": int(row["tout"]), "ingests": row["ingests"],
            "jev_usd": round(float(row["jev_usd"]), 5), "jev_requests": int(row["jev_requests"]),
            "extract_splits": int(row["splits"]), "extract_wasted_tokens_out": int(row["wasted_out"]),
            "extract_cached_tokens_in": int(row["cached_in"]), "extract_continuations": int(row["continuations"]),
            "gate_turns_skipped": int(row["gate_skipped"]), "extractor_model": row["model"], "facts": facts}


async def run_one(mem: Memory, http: OpenRouterHTTP, item: dict, cfg_name: str, run_id: str, args) -> dict:
    spec = CONFIGS[cfg_name]
    scope = f"memory.lme.{spec.get('ingest_tag', 'base')}.{scope_label(item['question_id'])}"
    t0 = time.perf_counter()
    await ingest_question(mem, item, scope, spec.get("drop_assistant", False))
    t1 = time.perf_counter()
    recall = await mem.recall(scope, item["question"], as_of=parse_date(item.get("question_date", "")),
                              **spec["recall"])
    t2 = time.perf_counter()
    memory = recall.markdown or "(no relevant memories)"
    answer, _ = await chat(http, args.answer_model, ANSWER_PROMPT.format(
        memory=memory, question_date=item.get("question_date", ""), question=item["question"]))
    verdict_text, _ = await chat(http, args.judge_model, judge_prompt(item, answer), max_tokens=1000)
    return {
        "run_id": run_id, "config": cfg_name, "question_id": item["question_id"],
        "question_type": "abstention" if item["question_id"].endswith("_abs") else item["question_type"],
        "question": item["question"], "answer": item["answer"], "response": answer,
        "correct": judged_yes(verdict_text), "judge": verdict_text[-200:], "mode": recall.mode,
        "facts": len(recall.facts),
        "memory_tokens": recall.token_count, "evidence_recall": await evidence_recall(
            mem, scope, item, [f.id for f in recall.facts]),
        "ingest_s": round(t1 - t0, 2), "recall_ms": round((t2 - t1) * 1000),
        "usage": await ingest_usage(mem, scope),
    }


def summarize(rows: list[dict]) -> dict:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(r)
    recall_ms = sorted(r["recall_ms"] for r in rows)
    ev = [r["evidence_recall"] for r in rows if r["evidence_recall"] is not None]
    return {
        "questions": len(rows),
        "accuracy": round(sum(r["correct"] for r in rows) / max(1, len(rows)), 4),
        "by_type": {t: round(sum(r["correct"] for r in v) / len(v), 4) for t, v in sorted(by_type.items())},
        "evidence_recall": round(statistics.mean(ev), 4) if ev else None,
        "memory_tokens_mean": round(statistics.mean(r["memory_tokens"] for r in rows), 1) if rows else 0,
        "recall_ms_p50": recall_ms[min(len(recall_ms) - 1, math.ceil(0.50 * len(recall_ms)) - 1)] if rows else None,
        "recall_ms_p95": recall_ms[min(len(recall_ms) - 1, math.ceil(0.95 * len(recall_ms)) - 1)] if rows else None,
    }


async def main_async(args) -> None:
    items = json.loads(DATA.read_text())
    chosen = select(items, args.per_type, args.abstention, args.seed)
    if args.ids:
        wanted = [line.strip() for line in Path(args.ids).read_text().splitlines() if line.strip()]
        by_id = {i["question_id"]: i for i in items}
        chosen = [by_id[q] for q in wanted]
    if args.limit:
        chosen = chosen[: args.limit]
    run_id = args.resume or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    RESULTS.mkdir(parents=True, exist_ok=True)
    for cfg_name in args.config:
        out = RESULTS / f"{run_id}-{cfg_name}.jsonl"
        previous = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
        answered = {r["question_id"] for r in previous}
        todo = [i for i in chosen if i["question_id"] not in answered]
        spec = CONFIGS[cfg_name]
        overrides = {**({"extractor_model": args.extract_model} if args.extract_model else {}), **spec.get("memx", {})}
        mem = await Memory.connect(MemxConfig(dsn=bench_dsn(args.dsn), insert_mode="inline", **overrides), start_runner=False)
        http = OpenRouterHTTP(mem.ctx.config.api_key(), mem.ctx.config.openrouter_base_url, timeout_s=120)
        sem = asyncio.Semaphore(args.concurrency)

        async def guarded(item: dict) -> dict | None:
            async with sem:
                try:
                    row = await run_one(mem, http, item, cfg_name, run_id, args)
                except Exception as exc:  # noqa: BLE001 - one bad question should not sink the run
                    print(f"  ! {item['question_id']}: {type(exc).__name__}: {exc}")
                    return None
                with out.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"  {'✓' if row['correct'] else '✗'} {row['question_type']:<26} {row['question_id']}")
                return row

        print(f"== {cfg_name}: {len(todo)} questions ({len(answered)} already answered)")
        rows = previous + [r for r in await asyncio.gather(*(guarded(i) for i in todo)) if r]
        summary = {"run_id": run_id, "config": cfg_name, **summarize(rows)}
        (RESULTS / f"{run_id}-{cfg_name}.summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        await http.aclose()
        await mem.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", default=["ranked_4k"], choices=sorted(CONFIGS))
    p.add_argument("--per-type", type=int, default=25)
    p.add_argument("--abstention", type=int, default=20)
    p.add_argument("--limit", type=int, default=0, help="cap total questions (smoke runs)")
    p.add_argument("--ids", default=None, help="file of question ids to run instead of the seeded sample")
    p.add_argument("--extract-model", default=None, help="extractor model (default: MemxConfig / MEMX_EXTRACTOR_MODEL)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--answer-model", default="anthropic/claude-haiku-5.5")
    p.add_argument("--judge-model", default="anthropic/claude-sonnet-5.5")
    p.add_argument("--resume", default=None, help="run id to complete: skips questions already in its results")
    p.add_argument("--dsn", default=BENCH_DSN, help="database for benchmark memory (default: memx_bench)")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
