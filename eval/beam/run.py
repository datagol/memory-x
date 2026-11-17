"""BEAM (ICLR 2026) contradiction-resolution and knowledge-update questions, every system on the same footing.

Data: the 20 BEAM-100K conversations (~150k tokens each); 2 contradiction-resolution and 2 knowledge-update probing
questions per conversation = 80 questions. Each conversation is cut into the same episodes for every system:
consecutive blocks of up to ``--episode-messages`` messages within one session, dated by the session's time anchor.

Systems (``--system``):
  memx        ingest each episode with Memory.insert; recall ranked, 4k budget
  mem0        mem0.add per episode (date in a system message, as in run_mem0); search top 30
  zep_llm     Graphiti, one episode per episode; LLM dedupe/contradiction       (needs graphiti-core + FalkorDB)
  zep_jev     Graphiti with node dedupe and edge resolution routed to Jev        (same)
  full_context  every message, oldest pruned to fit ``--context-budget`` tokens
  chunk_rag   ~800-token pieces of the episodes embedded; top pieces up to 4k tokens, chronological

Every system: extraction model Haiku 5.5, text-embedding-3-small, the same answer prompt and answering model, and
BEAM's own rubric judge prompt (``unified_llm_judge_base_prompt``: each rubric item scored 0 / 0.5 / 1; a question's
score is the mean over its rubric items).

    python -m eval.beam.download   # once
    python -m eval.beam.run --system chunk_rag full_context memx mem0 --resume paper
"""

from __future__ import annotations

import os

os.environ.setdefault("MEM0_TELEMETRY", "False")
os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

import argparse  # noqa: E402
import ast  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

from eval import bench_dsn
from memx import Memory, Messages, MemxConfig, Turn  # noqa: E402
from memx.providers.http import OpenRouterHTTP  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data" / "100K.parquet"
RESULTS = HERE / "results"
PROGRESS = RESULTS / "progress"
DSN = os.environ.get("MEMX_BEAM_DSN", "postgresql://memx:memx@localhost:5433/memx_beam")
TYPES = ("contradiction_resolution", "knowledge_update")
CHARS_PER_TOKEN = 3.6
SYSTEMS = ("memx", "mem0", "zep_llm", "zep_jev", "full_context", "chunk_rag")

ANSWER_PROMPT = """You are a helpful assistant with a long-term memory of past conversations with the user.

{memory}

Current date: {question_date}

Answer the user's question using the memory above. If the memory does not contain the information needed, say you don't know.

Question: {question}
Answer:"""

# BEAM src/prompts.py, unified_llm_judge_base_prompt, verbatim.
JUDGE = (HERE / "judge_prompt.txt").read_text() if (HERE / "judge_prompt.txt").exists() else ""

_MARKER = re.compile(r"\s*->->\s*\d+\s*,\s*\d+\s*$")


# ---------------------------------------------------------------- data


def load() -> list[dict]:
    import pyarrow.parquet as pq

    chats = []
    for row in pq.read_table(DATA).to_pylist():
        probing = ast.literal_eval(row["probing_questions"])
        questions = [{"qid": f"{row['conversation_id']}-{t}-{i}", "type": t, **q}
                     for t in TYPES for i, q in enumerate(probing.get(t, []))]
        chats.append({"id": row["conversation_id"], "sessions": row["chat"], "questions": questions})
    return chats


def anchor(text: str) -> datetime:
    for fmt in ("%B-%d-%Y", "%b-%d-%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text.strip(), fmt).replace(hour=12, tzinfo=UTC)
        except ValueError:
            continue
    return datetime(2024, 1, 1, 12, tzinfo=UTC)


def episodes(chat: dict, size: int) -> list[dict]:
    out = []
    for si, session in enumerate(chat["sessions"]):
        msgs = [{"role": m["role"], "content": _MARKER.sub("", m["content"] or "").strip()} for m in session]
        msgs = [m for m in msgs if m["content"]]
        when = anchor(session[0].get("time_anchor") or "") if session else datetime(2024, 1, 1, tzinfo=UTC)
        for start in range(0, len(msgs), size):
            out.append({"key": f"{chat['id']}:{si}:{start}", "session": si, "date": when,
                        "messages": msgs[start:start + size]})
    return out


def as_text(ep: dict) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in ep["messages"])


def tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


# ---------------------------------------------------------------- progress


def done_keys(system: str, chat_id: str) -> set[str]:
    f = PROGRESS / f"{system}-{chat_id}.json"
    return set(json.loads(f.read_text())) if f.exists() else set()


def mark(system: str, chat_id: str, keys: set[str]) -> None:
    PROGRESS.mkdir(parents=True, exist_ok=True)
    (PROGRESS / f"{system}-{chat_id}.json").write_text(json.dumps(sorted(keys)))


# ---------------------------------------------------------------- systems


class System:
    name = ""

    async def ingest(self, chat: dict, eps: list[dict]) -> None: ...

    async def memory(self, chat: dict, eps: list[dict], question: str) -> str: ...

    def usage(self, chat: dict) -> dict:
        return {}

    async def close(self) -> None: ...


class MemxSystem(System):
    name = "memx"

    def __init__(self, args):
        self.args = args
        self.mem: Memory | None = None
        self.lock = asyncio.Lock()

    async def _mem(self) -> Memory:
        async with self.lock:
            if self.mem is None:
                self.mem = await Memory.connect(MemxConfig(dsn=bench_dsn(DSN), insert_mode="inline",
                                                           extractor_model=self.args.extract_model),
                                                start_runner=False)
        return self.mem

    def scope(self, chat: dict) -> str:
        return f"memory.beam.c{chat['id']}"

    async def ingest(self, chat, eps):
        mem = await self._mem()
        for ep in eps:
            await mem.insert(self.scope(chat), Messages(turns=[Turn(**m) for m in ep["messages"]]),
                             observed_at=ep["date"], idempotency_key=ep["key"], mode="inline")

    async def memory(self, chat, eps, question):
        recall = await (await self._mem()).recall(self.scope(chat), question, force_mode="ranked", budget_tokens=4_000,
                                                  as_of=eps[-1]["date"])
        return recall.markdown or "(no relevant memories)"

    def usage(self, chat):
        return {}  # read from receipts afterwards (memx_ingest), see summarize_costs

    async def close(self):
        if self.mem:
            await self.mem.close()


class Mem0System(System):
    name = "mem0"

    def __init__(self, args):
        from mem0 import AsyncMemory

        from eval.longmemeval.run_mem0 import mem0_config

        cfg = MemxConfig(dsn=bench_dsn(DSN))
        conf = mem0_config(cfg, args.extract_model)
        conf["vector_store"]["config"]["collection_name"] = "mem0_beam"
        conf["history_db_path"] = str(RESULTS / "mem0_beam_history.db")
        self.mem = AsyncMemory.from_config(conf)
        self.cut = 0

    async def _add(self, messages: list[dict], uid: str, day: str) -> None:
        """mem0 embeds the batch it is given, and the embedding model takes at most 8,192 tokens: halve a batch
        that is too long, and as a last resort cut a single oversized message (logged in ``self.cut``)."""
        try:
            await self.mem.add([{"role": "system", "content": f"This conversation took place on {day}."}, *messages],
                               user_id=uid, metadata={"session_date": day[:10]})
        except Exception as exc:  # noqa: BLE001
            if "maximum input length" not in str(exc):
                raise
            if len(messages) > 1:
                mid = len(messages) // 2
                await self._add(messages[:mid], uid, day)
                await self._add(messages[mid:], uid, day)
            else:
                self.cut += 1
                m = messages[0]
                await self._add([{**m, "content": m["content"][: len(m["content"]) // 2]}], uid, day)

    async def ingest(self, chat, eps):
        uid = f"beam_{chat['id']}"
        done = done_keys(self.name, chat["id"])
        for ep in eps:
            if ep["key"] in done:
                continue
            await self._add(ep["messages"], uid, ep["date"].strftime("%Y-%m-%d (%A)"))
            done.add(ep["key"])
            mark(self.name, chat["id"], done)

    async def memory(self, chat, eps, question):
        found = await self.mem.search(question, filters={"user_id": f"beam_{chat['id']}"}, top_k=30)
        results = found.get("results", found) if isinstance(found, dict) else found
        lines = ["# Memory"] + [f"- ({(r.get('metadata') or {}).get('session_date', '')}) {r['memory']}"
                                for r in results]
        return "\n".join(lines) if results else "(no relevant memories)"


class ZepSystem(System):
    def __init__(self, args, name: str):
        self.name, self.args = name, args
        self.cfg = MemxConfig()
        self.spent: dict[str, dict] = {}

    def group(self, chat):
        return f"beam_{self.name}_{chat['id']}"

    def _graph(self, chat):
        from eval.longmemeval.run_graphiti import build

        return build(self.cfg, self.args, self.name, self.group(chat))

    async def ingest(self, chat, eps):
        from graphiti_core.nodes import EpisodeType

        from eval.longmemeval.run_graphiti import CountingJev, usage_of

        done = done_keys(self.name, chat["id"])
        graph = self._graph(chat)
        jev0 = (CountingJev.requests, CountingJev.cost)
        try:
            if not done:
                await graph.build_indices_and_constraints()
            for i, ep in enumerate(eps):
                if ep["key"] in done:
                    continue
                await graph.add_episode(name=f"episode-{i}", episode_body=as_text(ep), source=EpisodeType.message,
                                        source_description="chat", reference_time=ep["date"],
                                        group_id=self.group(chat))
                done.add(ep["key"])
                mark(self.name, chat["id"], done)
            u = usage_of(graph)
            u["jev_usd"] = round(CountingJev.cost - jev0[1], 5)
            u["jev_requests"] = CountingJev.requests - jev0[0]
            self.spent[chat["id"]] = u
            usage_file = PROGRESS / f"{self.name}-{chat['id']}.usage.json"
            prev = json.loads(usage_file.read_text()) if usage_file.exists() else {}
            usage_file.write_text(json.dumps({k: prev.get(k, 0) + v for k, v in u.items()}))
        finally:
            await graph.close()

    async def memory(self, chat, eps, question):
        from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

        from eval.longmemeval.run_graphiti import context

        graph = self._graph(chat)
        try:
            config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
            config.limit = 20
            return context(await graph.search_(question, config=config, group_ids=[self.group(chat)]))
        finally:
            await graph.close()


class FullContext(System):
    name = "full_context"

    def __init__(self, args):
        self.budget = args.context_budget

    async def memory(self, chat, eps, question):
        blocks = [f"### {ep['date']:%Y-%m-%d}\n{as_text(ep)}" for ep in eps]
        kept, used = [], 0
        for b in reversed(blocks):  # prune the oldest, as BEAM's long-context baseline does
            if used + tokens(b) > self.budget:
                break
            kept.append(b)
            used += tokens(b)
        return "## Conversation history\n\n" + "\n\n".join(reversed(kept))


class ChunkRag(System):
    """Plain retrieval: each episode's text cut into ~``chunk_tokens`` pieces, embedded; the pieces most similar
    to the question are given to the answering model up to 4,000 tokens, in chronological order."""

    name = "chunk_rag"

    def __init__(self, args, chunk_tokens: int = 800):
        self.cfg = MemxConfig()
        self.http = OpenRouterHTTP(self.cfg.api_key(), self.cfg.openrouter_base_url, timeout_s=120)
        self.limit = int(chunk_tokens * CHARS_PER_TOKEN)
        self.index: dict[str, tuple[list[dict], list[list[float]]]] = {}

    def pieces(self, eps: list[dict]) -> list[dict]:
        out = []
        for i, ep in enumerate(eps):
            text = as_text(ep)
            for start in range(0, len(text), self.limit):
                out.append({"ep": i, "date": ep["date"], "text": text[start:start + self.limit]})
        return out

    async def _embed(self, texts):
        out = []
        for i in range(0, len(texts), 64):
            data = await self.http.post("/embeddings", {"model": self.cfg.embedding_model, "input": texts[i:i + 64]})
            out += [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
        return out

    async def ingest(self, chat, eps):
        parts = self.pieces(eps)
        cache = PROGRESS / f"chunk_rag-{chat['id']}.vectors.json"
        if cache.exists():
            vectors = json.loads(cache.read_text())
        else:
            PROGRESS.mkdir(parents=True, exist_ok=True)
            vectors = await self._embed([f"({p['date']:%Y-%m-%d}) {p['text']}" for p in parts])
            cache.write_text(json.dumps(vectors))
        self.index[chat["id"]] = (parts, vectors)

    async def memory(self, chat, eps, question):
        parts, vecs = self.index[chat["id"]]
        [q] = await self._embed([question])

        def cos(a, b):
            return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

        ranked = sorted(range(len(parts)), key=lambda i: cos(q, vecs[i]), reverse=True)
        picked, used = [], 0
        for i in ranked:
            t = tokens(parts[i]["text"]) + 8
            if used + t <= 4_000:
                picked.append(i)
                used += t
        return "## Relevant excerpts\n\n" + "\n\n".join(
            f"### {parts[i]['date']:%Y-%m-%d}\n{parts[i]['text']}" for i in sorted(picked))

    async def close(self):
        await self.http.aclose()


def make(name: str, args) -> System:
    return {"memx": lambda: MemxSystem(args), "mem0": lambda: Mem0System(args),
            "zep_llm": lambda: ZepSystem(args, "zep_llm"), "zep_jev": lambda: ZepSystem(args, "zep_jev"),
            "full_context": lambda: FullContext(args), "chunk_rag": lambda: ChunkRag(args)}[name]()


# ---------------------------------------------------------------- answer + judge


async def chat_call(http: OpenRouterHTTP, model: str, prompt: str, max_tokens: int = 800) -> tuple[str, dict]:
    data = await http.post("/chat/completions", {"model": model, "temperature": 0, "max_tokens": max_tokens,
                                                 "messages": [{"role": "user", "content": prompt}]})
    return (data["choices"][0]["message"].get("content") or "").strip(), data.get("usage") or {}


def parse_score(text: str) -> float:
    m = re.search(r'"score"\s*:\s*"?([01](?:\.\d+)?)', text)
    return float(m.group(1)) if m else 0.0


async def judge(http: OpenRouterHTTP, model: str, rubric: list[str], response: str) -> tuple[float, list[dict]]:
    outs = []
    for item in rubric:
        prompt = JUDGE.replace("<rubric_item>", item).replace("<llm_response>", response)
        text, _ = await chat_call(http, model, prompt, max_tokens=600)
        outs.append({"rubric": item, "score": parse_score(text), "raw": text[-300:]})
    return (sum(o["score"] for o in outs) / len(outs) if outs else 0.0), outs


# ---------------------------------------------------------------- run


async def run_chat(system: System, http: OpenRouterHTTP, chat: dict, out: Path, answered: set[str], args) -> list[dict]:
    eps = episodes(chat, args.episode_messages)
    todo = [q for q in chat["questions"] if q["qid"] not in answered]
    if not todo:
        return []
    t0 = time.perf_counter()
    await system.ingest(chat, eps)
    ingest_s = time.perf_counter() - t0
    rows = []
    for q in todo:
        t1 = time.perf_counter()
        memory = await system.memory(chat, eps, q["question"])
        recall_ms = (time.perf_counter() - t1) * 1000
        answer, usage = await chat_call(http, args.answer_model, ANSWER_PROMPT.format(
            memory=memory, question_date=f"{eps[-1]['date']:%Y-%m-%d}", question=q["question"]))
        score, details = await judge(http, args.judge_model, q["rubric"], answer)
        row = {"system": system.name, "qid": q["qid"], "chat": chat["id"], "type": q["type"],
               "question": q["question"], "ideal": q.get("ideal_answer") or q.get("answer"), "response": answer,
               "score": score, "judge": details, "memory_tokens": tokens(memory), "ingest_s": round(ingest_s, 1),
               "recall_ms": round(recall_ms), "answer_prompt_tokens": usage.get("prompt_tokens")}
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"  {system.name:<12} {q['qid']:<32} score {score:.2f}", flush=True)
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r["score"])
    return {"questions": len(rows), "score": round(statistics.mean(r["score"] for r in rows), 4) if rows else None,
            "by_type": {t: round(statistics.mean(v), 4) for t, v in by.items()},
            "memory_tokens_mean": round(statistics.mean(r["memory_tokens"] for r in rows)) if rows else None}


async def main_async(args) -> None:
    global JUDGE
    if not JUDGE:
        raise SystemExit("eval/beam/judge_prompt.txt missing: run python -m eval.beam.download")
    chats = load()
    if args.chats:
        chats = [c for c in chats if c["id"] in set(args.chats)]
    run_id = args.resume or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    RESULTS.mkdir(parents=True, exist_ok=True)
    cfg = MemxConfig()
    http = OpenRouterHTTP(cfg.api_key(), cfg.openrouter_base_url, timeout_s=300)
    for name in args.system:
        system = make(name, args)
        out = RESULTS / f"{run_id}-{name}.jsonl"
        previous = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
        answered = {r["qid"] for r in previous}
        sem = asyncio.Semaphore(args.concurrency)

        async def guarded(chat: dict, system: System = system, out: Path = out) -> list[dict]:
            async with sem:
                try:
                    return await run_chat(system, http, chat, out, answered, args)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! {system.name} chat {chat['id']}: {type(exc).__name__}: {exc}", flush=True)
                    return []

        print(f"== {name}: {len(chats)} conversations", flush=True)
        rows = previous + [r for batch in await asyncio.gather(*(guarded(c) for c in chats)) for r in batch]
        summary = {"run_id": run_id, "system": name, "extract_model": args.extract_model,
                   "answer_model": args.answer_model, "judge_model": args.judge_model, **summarize(rows)}
        (RESULTS / f"{run_id}-{name}.summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        await system.close()
    await http.aclose()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--system", nargs="+", default=["chunk_rag"], choices=SYSTEMS)
    p.add_argument("--chats", nargs="*", default=None, help="conversation ids (default: all 20)")
    p.add_argument("--episode-messages", type=int, default=10)
    p.add_argument("--context-budget", type=int, default=150_000)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--extract-model", default="anthropic/claude-haiku-5.5")
    p.add_argument("--answer-model", default="anthropic/claude-haiku-5.5")
    p.add_argument("--judge-model", default="anthropic/claude-sonnet-5.5")
    p.add_argument("--max-coroutines", type=int, default=8)  # graphiti
    p.add_argument("--falkor-host", default="localhost")
    p.add_argument("--falkor-port", type=int, default=6390)
    p.add_argument("--resume", default=None)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
