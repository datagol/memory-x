"""Run LoCoMo against mem0 (open source), for comparison with memx on the same harness.

Same conversations, questions (categories 1-4), answer model, answer prompt, judge and judge prompt as
``eval.locomo.run``; same extraction model (Haiku 5.5) and embeddings (text-embedding-3-small) as memx.

Ingest follows mem0's own LoCoMo evaluation: each conversation is added twice, once per speaker (that speaker's
turns as "user", the other's as "assistant"), so each person has their own memory. At question time both memories
are searched and shown to the answer model, each fact with its session date.

    python -m eval.locomo.run_mem0 --top-k 15        # 15 memories per speaker, up to 30 in all
"""

from __future__ import annotations

import os

os.environ.setdefault("MEM0_TELEMETRY", "False")  # keep benchmark data out of mem0's analytics

import argparse
import asyncio
import json
import re
import time
from datetime import UTC, datetime

from mem0 import AsyncMemory  # noqa: E402

from eval.locomo.run import (
    ANSWER_PROMPT,
    CATEGORIES,
    DSN,
    JUDGE_PROMPT,
    RESULTS,
    chat,
    judged_correct,
    load,
    sessions,
    summarize,
    turn_text,
)
from eval.longmemeval.run_mem0 import USAGE, mem0_config
from eval import bench_dsn
from memx import MemxConfig
from memx.providers.http import OpenRouterHTTP

PROGRESS = RESULTS / "mem0_progress"
COLLECTION = "mem0_locomo"
DB = DSN.rsplit("/", 1)[-1]
FAILED: list[dict] = []  # sessions mem0 could not add after retries
PRICES = {"anthropic/claude-haiku-5.5": (0.10, 0.50)}


async def ingest_speaker(mem: AsyncMemory, sample: dict, me: str, uid: str) -> None:
    """Every session, in order, from ``me``'s side. Progress is saved so a rerun resumes."""
    PROGRESS.mkdir(parents=True, exist_ok=True)
    marker = PROGRESS / f"{DB}-{COLLECTION}-{uid}.json"  # never outlives the database it describes
    done = set(json.loads(marker.read_text())) if marker.exists() else set()
    for n, when, turns in sessions(sample["conversation"]):
        if n in done:
            continue
        messages = [{"role": "user" if t["speaker"] == me else "assistant", "content": f"{t['speaker']}: {turn_text(t)}"}
                    for t in turns if turn_text(t)]
        if messages:
            # OSS mem0 has no timestamp parameter, so the date goes into the conversation, as in run_mem0 for LME.
            messages.insert(0, {"role": "system", "content": f"This conversation took place on {when:%d %B %Y}."})
            for attempt in range(3):
                try:
                    await mem.add(messages, user_id=uid, metadata={"session": n, "session_date": f"{when:%Y-%m-%d}"})
                    break
                except Exception as exc:  # noqa: BLE001 - mem0 raises when the model's JSON can't be parsed
                    if attempt == 2:
                        # As a mem0 user would have to: give up on this session, and count it.
                        FAILED.append({"user": uid, "session": n, "error": f"{type(exc).__name__}: {exc}"[:200]})
                        print(f"  ! {uid} session {n}: skipped after 3 attempts: {exc}")
        done.add(n)
        marker.write_text(json.dumps(sorted(done)))


def render(by_speaker: dict[str, list[dict]]) -> str:
    lines = []
    for speaker, results in by_speaker.items():
        if not results:
            continue
        lines += ["", f"## Memories of {speaker}"]
        for r in results:
            when = (r.get("metadata") or {}).get("session_date", "")
            lines.append(f"- ({when}) {r['memory']}" if when else f"- {r['memory']}")
    return ("# Memory" + "\n".join(lines) + "\n") if lines else ""


async def main_async(args: argparse.Namespace) -> None:
    data = load()[: args.conversations or None]
    cfg = MemxConfig(dsn=bench_dsn(args.dsn))
    mem = AsyncMemory.from_config({**mem0_config(cfg, args.extract_model, COLLECTION),
                                   "history_db_path": str(RESULTS / "mem0_locomo_history.db")})
    http = OpenRouterHTTP(cfg.api_key(), cfg.openrouter_base_url, timeout_s=120)
    run_id = args.resume or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    name = f"mem0_top{args.top_k}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{run_id}-{name}.jsonl"
    previous = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    done = {(r["sample_id"], r["index"]) for r in previous}
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def ask(sample: dict, uids: dict[str, str], index: int, qa: dict, first: str, last: str) -> dict | None:
        conv = sample["conversation"]
        async with sem:
            try:
                t0 = time.perf_counter()
                found = {}
                for speaker, uid in uids.items():
                    res = await mem.search(qa["question"], filters={"user_id": uid}, top_k=args.top_k)
                    found[speaker] = res.get("results", res) if isinstance(res, dict) else res
                t1 = time.perf_counter()
                memory = render(found)
                answer = await chat(http, args.answer_model, ANSWER_PROMPT.format(
                    a=conv["speaker_a"], b=conv["speaker_b"], memory=memory or "(no relevant memories)",
                    first=first, last=last, question=qa["question"]))
                gold = qa.get("answer", qa.get("adversarial_answer", ""))
                verdict = await chat(http, args.judge_model, JUDGE_PROMPT.format(
                    question=qa["question"], gold=gold, response=answer), max_tokens=400)
            except Exception as exc:  # noqa: BLE001 - one bad question should not sink the run
                print(f"  ! {sample['sample_id']}#{index}: {type(exc).__name__}: {exc}")
                return None
        hits = {(r.get("metadata") or {}).get("session") for rs in found.values() for r in rs}
        # Evidence ids look like "D3:12"; a few in the dataset are malformed ("D:11:26"), so parse leniently.
        wanted = {int(m.group(1)) for e in qa.get("evidence") or [] if (m := re.match(r"D(\d+):", e))}
        row = {
            "run_id": run_id, "config": name, "sample_id": sample["sample_id"], "index": index,
            "category": CATEGORIES[qa["category"]], "question": qa["question"], "answer": gold, "response": answer,
            "correct": judged_correct(verdict), "judge": verdict[-80:],
            "facts": sum(len(v) for v in found.values()), "memory_tokens": len(memory) // 4,
            "recall_ms": round((t1 - t0) * 1000),
            "evidence_recall": len(hits & wanted) / len(wanted) if wanted else None,
        }
        async with lock:
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        return row

    async def conversation(sample: dict) -> list[dict]:
        conv = sample["conversation"]
        sid = sample["sample_id"].replace("-", "_")
        uids = {conv["speaker_a"]: f"locomo_{sid}_a", conv["speaker_b"]: f"locomo_{sid}_b"}
        t0 = time.perf_counter()
        await asyncio.gather(*(ingest_speaker(mem, sample, speaker, uid) for speaker, uid in uids.items()))
        spans = sessions(conv)
        first, last = spans[0][1].strftime("%d %B %Y"), spans[-1][1].strftime("%d %B %Y")
        todo = [(i, qa) for i, qa in enumerate(sample["qa"])
                if qa["category"] in args.categories and (sample["sample_id"], i) not in done]
        print(f"== {sample['sample_id']}: ingested in {time.perf_counter() - t0:.0f}s, {len(todo)} questions")
        rows = await asyncio.gather(*(ask(sample, uids, i, qa, first, last) for i, qa in todo))
        return [r for r in rows if r]

    new = [r for rows in await asyncio.gather(*(conversation(s) for s in data)) for r in rows]
    rows = previous + new
    pin, pout = PRICES.get(args.extract_model, (1.0, 5.0))
    cost = USAGE["prompt_tokens"] / 1e6 * pin + USAGE["completion_tokens"] / 1e6 * pout
    summary = {"run_id": run_id, "config": name, "conversations": len(data), **summarize(rows),
               "extract_usage": {**USAGE, "est_usd_this_run": round(cost, 2)}, "sessions_skipped": FAILED}
    (RESULTS / f"{run_id}-{name}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    await http.aclose()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=15, help="memories retrieved per speaker")
    p.add_argument("--conversations", type=int, default=0)
    p.add_argument("--categories", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--extract-model", default="anthropic/claude-haiku-5.5")
    p.add_argument("--answer-model", default="anthropic/claude-haiku-5.5")
    p.add_argument("--judge-model", default="anthropic/claude-sonnet-5.5")
    p.add_argument("--resume", default=None)
    p.add_argument("--dsn", default=DSN)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
