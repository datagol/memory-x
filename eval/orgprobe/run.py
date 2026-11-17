"""Run OrgProbe against memx and mem0 with the same extraction model, embeddings, answer model and judge.

    python -m eval.orgprobe.run --system memx mem0 --reps 3

Results: eval/orgprobe/results/<run>-<config>.jsonl and a summary JSON. Ingest is idempotent, so a rerun only pays
for answering and judging.
"""

from __future__ import annotations

import os

os.environ.setdefault("MEM0_TELEMETRY", "False")

import argparse
import asyncio
import json
import statistics
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from eval.longmemeval.run import ANSWER_PROMPT, BENCH_DSN, DATA, chat, judged_yes
from eval.orgprobe import fixtures as fx
from eval import bench_dsn
from memx import Memory, Messages, MemxConfig, Text, Turn
from memx.providers.http import OpenRouterHTTP

HERE = Path(__file__).parent
RESULTS = HERE / "results"
CHAIN = {  # memx scope -> the mem0 (user_id, agent_id) stores an agent there should read
    "memory.acme": [("acme_company", None)],
    "memory.acme.alex": [("acme_company", None), ("acme_alex", None)],
    "memory.acme.alex.agent1": [("acme_company", None), ("acme_alex", None)],
    "memory.acme.sre": [("acme_company", None), ("acme_sre", None)],
}
WRITE = {"memory.acme": ("acme_company", None), "memory.acme.alex": ("acme_alex", None),
         "memory.acme.alex.agent1": ("acme_alex", "agent1"), "memory.acme.sre": ("acme_sre", None)}

JUDGE = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
         "response contains the correct answer, or is equivalent to it. If the response only contains a subset of "
         "the information required by the answer, answer no. If the correct answer says the information is not "
         "known, answer yes only if the response says it does not know.")


def filler() -> list[tuple[datetime, list[tuple[str, str]]]]:
    items = json.loads(DATA.read_text())
    item = items[fx.FILLER_FROM_QUESTION_INDEX]
    answer_ids = set(item.get("answer_session_ids") or [])
    out = []
    picked = [s for sid, s in zip(item["haystack_session_ids"], item["haystack_sessions"]) if sid not in answer_ids]
    for i, session in enumerate(picked[: fx.FILLER_SESSIONS]):
        when = datetime(2026, 9, 2, 12, tzinfo=UTC) + timedelta(days=2 * i)
        out.append((when, [(t["role"], t["content"]) for t in session if t.get("content")]))
    return out


# ---------------------------------------------------------------- memx

async def memx_ingest(mem: Memory) -> None:
    for scope, when, source, text, trust in fx.DOCS:
        await mem.insert(scope, Text(text=text, source=source, trust=trust), observed_at=when,
                         idempotency_key=f"orgprobe:{source}", mode="inline")
    for i, (scope, when, turns) in enumerate(fx.SESSIONS):
        await mem.insert(scope, Messages(turns=[Turn(role=r, content=c) for r, c in turns]), observed_at=when,
                         idempotency_key=f"orgprobe:s{i}", mode="inline")
    for i, (when, turns) in enumerate(filler()):
        await mem.insert("memory.acme.alex", Messages(turns=[Turn(role=r, content=c) for r, c in turns]),
                         observed_at=when, idempotency_key=f"orgprobe:filler{i}", mode="inline")


async def memx_recall(mem: Memory, q: dict, cfg: dict) -> tuple[str, dict]:
    r = await mem.recall(q["scope"], q["q"], as_of=fx.QUESTION_DATE, explain=True, **cfg)
    walked = [c for c in (r.explain or {}).get("candidates", []) if c.get("found_by") in ("entity_hop", "topic_hop")
              and c.get("status") == "picked"]
    return r.markdown or "", {"mode": r.mode, "facts": len(r.facts), "tokens": r.token_count, "walked_picked": len(walked)}


# ---------------------------------------------------------------- mem0

def mem0_memory():
    from mem0 import AsyncMemory

    from eval.longmemeval.run_mem0 import mem0_config
    cfg = MemxConfig(dsn=bench_dsn(BENCH_DSN))
    conf = mem0_config(cfg, "anthropic/claude-haiku-5.5")
    conf["vector_store"]["config"]["collection_name"] = "mem0_orgprobe"
    conf["history_db_path"] = str(RESULTS / "mem0_orgprobe_history.db")
    return AsyncMemory.from_config(conf)


async def mem0_ingest(mem) -> None:
    marker = RESULTS / "mem0_ingested.json"
    done = set(json.loads(marker.read_text())) if marker.exists() else set()

    async def add(key: str, scope: str, when: datetime, messages: list[dict]) -> None:
        if key in done:
            return
        uid, aid = WRITE[scope]
        msgs = [{"role": "system", "content": f"This conversation took place on {when:%Y-%m-%d (%A)}."}] + messages
        kwargs = {"user_id": uid, **({"agent_id": aid} if aid else {})}
        await mem.add(msgs, metadata={"session_date": f"{when:%Y-%m-%d}", "source": key}, **kwargs)
        done.add(key)
        marker.write_text(json.dumps(sorted(done)))

    for scope, when, source, text, _trust in fx.DOCS:
        await add(f"doc:{source}", scope, when, [{"role": "user", "content": f"Document ({source}):\n{text}"}])
    for i, (scope, when, turns) in enumerate(fx.SESSIONS):
        await add(f"s{i}", scope, when, [{"role": r, "content": c} for r, c in turns])
    for i, (when, turns) in enumerate(filler()):
        await add(f"filler{i}", "memory.acme.alex", when, [{"role": r, "content": c} for r, c in turns])


async def mem0_recall(mem, q: dict, top_k: int) -> tuple[str, dict]:
    merged: dict[str, dict] = {}
    for uid, _aid in CHAIN[q["scope"]]:
        found = await mem.search(q["q"], filters={"user_id": uid}, top_k=top_k)
        for r in (found.get("results", found) if isinstance(found, dict) else found):
            if r["id"] not in merged or (r.get("score") or 0) > (merged[r["id"]].get("score") or 0):
                merged[r["id"]] = r
    results = sorted(merged.values(), key=lambda r: -(r.get("score") or 0))[:top_k]
    lines = ["# Memory"] + [f"- ({(r.get('metadata') or {}).get('session_date', '')}) {r['memory']}" for r in results]
    text = "\n".join(lines) + "\n" if results else ""
    return text, {"mode": "search", "facts": len(results), "tokens": len(text) // 4}


# ---------------------------------------------------------------- shared

def judge_prompt(q: dict, response: str) -> str:
    extra = f"\n\nThe response must not present this as current: {q['forbid_claim']}." if q.get("forbid_claim") else ""
    return (f"{JUDGE}{extra}\n\nQuestion: {q['q']}\n\nCorrect Answer: {q['answer']}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only.")


async def answer_and_judge(http: OpenRouterHTTP, q: dict, memory: str, args) -> dict:
    answer, _ = await chat(http, args.answer_model, ANSWER_PROMPT.format(
        memory=memory or "(no relevant memories)", question_date=f"{fx.QUESTION_DATE:%Y/%m/%d (%a) %H:%M}",
        question=q["q"]))
    verdict, _ = await chat(http, args.judge_model, judge_prompt(q, answer), max_tokens=300)
    leaked = [s for s in q.get("forbid", []) if s.lower() in answer.lower()]
    in_memory = [s for s in q.get("forbid", []) + ["hunter2", "DataDogX"] if s.lower() in memory.lower()]
    return {"response": answer, "judge": verdict[-160:], "correct": judged_yes(verdict) and not leaked,
            "leaked": leaked, "forbidden_in_memory": sorted(set(in_memory))}


async def main_async(args) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_id = args.resume or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    http = OpenRouterHTTP(MemxConfig().api_key(), MemxConfig().openrouter_base_url, timeout_s=120)
    configs: list[tuple[str, Any]] = []
    mem = m0 = None
    if "memx" in args.system:
        mem = await Memory.connect(MemxConfig(dsn=bench_dsn(args.dsn), insert_mode="inline"), start_runner=False)
        t = time.perf_counter()
        await memx_ingest(mem)
        print(f"memx ingest {time.perf_counter() - t:.0f}s")
        configs += [("memx_ranked_2k", {"force_mode": "ranked", "budget_tokens": 2000}),
                    ("memx_full", {"force_mode": "full", "budget_tokens": 200_000})]
        mem_nowalk = await Memory.connect(MemxConfig(dsn=bench_dsn(args.dsn), insert_mode="inline", walk_enabled=False),
                                          start_runner=False, run_migrations=False)
        configs.append(("memx_ranked_2k_nowalk", ("nowalk", {"force_mode": "ranked", "budget_tokens": 2000})))
    if "mem0" in args.system:
        m0 = mem0_memory()
        t = time.perf_counter()
        await mem0_ingest(m0)
        print(f"mem0 ingest {time.perf_counter() - t:.0f}s")
        configs += [("mem0_top30", 30), ("mem0_top10", 10)]

    for name, cfg in configs:
        out = RESULTS / f"{run_id}-{name}.jsonl"
        rows = []
        for q in fx.QUESTIONS:
            t = time.perf_counter()
            if name.startswith("mem0"):
                memory, meta = await mem0_recall(m0, q, cfg)
            elif isinstance(cfg, tuple):
                memory, meta = await memx_recall(mem_nowalk, q, cfg[1])
            else:
                memory, meta = await memx_recall(mem, q, cfg)
            recall_ms = int((time.perf_counter() - t) * 1000)
            reps = await asyncio.gather(*(answer_and_judge(http, q, memory, args) for _ in range(args.reps)))
            row = {"config": name, "id": q["id"], "kind": q["kind"], "scope": q["scope"], "question": q["q"],
                   "expected": q["answer"], "memory": memory, "recall_ms": recall_ms, **meta,
                   "reps": reps, "score": sum(r["correct"] for r in reps) / len(reps)}
            rows.append(row)
            print(f"  {name:<24} {q['id']:<3} {row['score']:.2f} {meta}")
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        by_kind: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            by_kind[r["kind"]].append(r["score"])
        summary = {"config": name, "questions": len(rows), "score": round(statistics.mean(r["score"] for r in rows), 4),
                   "by_kind": {k: round(statistics.mean(v), 4) for k, v in by_kind.items()},
                   "tokens_mean": round(statistics.mean(r["tokens"] for r in rows), 1),
                   "forbidden_in_memory": sum(bool(r["reps"][0]["forbidden_in_memory"]) for r in rows),
                   "leaks": sum(bool(x["leaked"]) for r in rows for x in r["reps"])}
        (RESULTS / f"{run_id}-{name}.summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary))
    await http.aclose()
    if mem:
        await mem.close()
        await mem_nowalk.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--system", nargs="+", default=["memx", "mem0"], choices=["memx", "mem0"])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--answer-model", default="anthropic/claude-haiku-5.5")
    p.add_argument("--judge-model", default="anthropic/claude-sonnet-5.5")
    p.add_argument("--resume", default=None)
    p.add_argument("--dsn", default=BENCH_DSN)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
