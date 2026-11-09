"""Zep's open-source engine (Graphiti) on the same LongMemEval subset, with its LLM judge or with Jev.

* ``zep_llm``: Graphiti defaults — the extraction model also judges node dedupe and edge duplicate/contradiction.
* ``zep_jev``: the same, but node dedupe and edge resolution routed to Jev through Graphiti's own ``JevClient``
  (duplicate threshold 0.7, contradiction threshold 0.8, Graphiti's defaults), via OpenRouter's System One endpoint.

One Graphiti graph per question (FalkorDB), one episode per haystack session (dated). Recall uses Graphiti's
``COMBINED_HYBRID_SEARCH_RRF`` recipe (no cross-encoder); the context lists the top 20 facts with their validity
ranges and the top 20 entity summaries, as in the Zep paper. Answer prompt, answering model and judge as ``run.py``.

Requires (not a project dependency):
    uv pip install --python .venv/bin/python "graphiti-core[falkordb]==0.31.2"
    docker run -d --name memx-bench-falkordb -p 6390:6379 falkordb/falkordb:latest

    python -m eval.longmemeval.run_graphiti --config zep_llm --extract-model anthropic/claude-haiku-5.5 --limit 2
"""

from __future__ import annotations

import os

os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import ssl  # noqa: E402
import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from typing import Any  # noqa: E402

import httpx  # noqa: E402
from graphiti_core import Graphiti  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient  # noqa: E402
from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: E402
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig  # noqa: E402
from graphiti_core.llm_client import (  # noqa: E402
    JevClient,
    LLMRuntime,
    LLMTransport,
    PromptRoutes,
    jev_prompt_overrides,
)
from graphiti_core.llm_client.config import LLMConfig  # noqa: E402
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient  # noqa: E402
from graphiti_core.nodes import EpisodeType  # noqa: E402
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF  # noqa: E402

from eval.longmemeval.run import (  # noqa: E402
    ANSWER_PROMPT, DATA, RESULTS, chat, judge_prompt, judged_yes, parse_date, scope_label, select, summarize,
)
from memx import MemxConfig  # noqa: E402
from memx.providers.http import OpenRouterHTTP  # noqa: E402

PROGRESS = RESULTS / "graphiti_progress"
CHARS_PER_TOKEN = 3.6


class CountingJev(JevClient):
    """Graphiti's JevClient, also counting requests and OpenRouter's metered cost."""

    requests = 0
    cost = 0.0

    async def _post(self, state, questions, model):
        for attempt in range(4):  # Graphiti retries HTTP errors, not TLS resets under load
            try:
                data = await super()._post(state, questions, model)
                break
            except ssl.SSLError:
                if attempt == 3:
                    raise
                await asyncio.sleep(2 ** attempt)
        CountingJev.requests += 1
        CountingJev.cost += float((data.get("usage") or {}).get("cost") or 0.0)
        return data


class Meter:
    """Counts OpenRouter's metered cost and tokens for every chat completion one Graphiti instance makes."""

    def __init__(self) -> None:
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.usd = 0.0

    async def hook(self, response: httpx.Response) -> None:
        if not response.request.url.path.endswith("/chat/completions") or response.status_code != 200:
            return
        await response.aread()
        try:
            usage = response.json().get("usage") or {}
        except ValueError:
            return
        self.calls += 1
        self.tokens_in += int(usage.get("prompt_tokens") or 0)
        self.tokens_out += int(usage.get("completion_tokens") or 0)
        self.usd += float(usage.get("cost") or 0.0)


def build(cfg: MemxConfig, args, cfg_name: str, database: str) -> Graphiti:
    key, base = cfg.api_key(), cfg.openrouter_base_url
    llm_config = LLMConfig(api_key=key, base_url=base, model=args.extract_model, small_model=args.extract_model,
                           temperature=0)
    meter = Meter()
    client = AsyncOpenAI(api_key=key, base_url=base,
                         http_client=httpx.AsyncClient(timeout=180, event_hooks={"response": [meter.hook]}))
    llm = OpenAIGenericClient(config=llm_config, client=client, max_tokens=8000)
    embedder = OpenAIEmbedder(OpenAIEmbedderConfig(api_key=key, base_url=base, embedding_model=cfg.embedding_model,
                                                   embedding_dim=cfg.embedding_dim))
    reranker = OpenAIRerankerClient(config=llm_config)  # constructed but unused by the RRF recipe
    driver = FalkorDriver(host=args.falkor_host, port=args.falkor_port, database=database)
    runtime = None
    if cfg_name == "zep_jev":
        main = LLMTransport(llm, models=[args.extract_model])
        jev = LLMTransport(CountingJev(LLMConfig(api_key=key), base_url=f"{base}/systemone"), models=[cfg.jev_model])
        decide = jev.model(cfg.jev_model, prompt_overrides=jev_prompt_overrides())
        runtime = LLMRuntime(model=main.model(args.extract_model), routes=PromptRoutes(
            dedupe_nodes=PromptRoutes.DedupeNodes(nodes=decide),
            dedupe_edges=PromptRoutes.DedupeEdges(resolve_edge=decide),
        ))
    clients = {"llm_runtime": runtime} if runtime is not None else {"llm_client": llm}
    graph = Graphiti(graph_driver=driver, embedder=embedder, cross_encoder=reranker,
                     max_coroutines=args.max_coroutines, **clients)
    graph.meter = meter  # type: ignore[attr-defined]
    return graph


def episode_body(session: list[dict]) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in session if t.get("content"))


def fmt_date(d: datetime | None) -> str:
    return d.strftime("%Y-%m-%d") if d else "present"


def context(results: Any) -> str:
    facts = "\n".join(f"- {e.fact} (valid {fmt_date(e.valid_at) if e.valid_at else 'unknown'} – "
                      f"{fmt_date(e.invalid_at)})" for e in results.edges)
    entities = "\n".join(f"- {n.name}: {n.summary}" for n in results.nodes if n.summary)
    return (f"## Facts (with the dates they were valid)\n{facts or '(none)'}\n\n"
            f"## Entities\n{entities or '(none)'}")


async def ingest(graph: Graphiti, item: dict, group: str, cfg_name: str) -> int:
    PROGRESS.mkdir(parents=True, exist_ok=True)
    marker = PROGRESS / f"{cfg_name}-{group}.json"
    done = set(json.loads(marker.read_text())) if marker.exists() else set()
    if not done:
        await graph.build_indices_and_constraints()
    added = 0
    for i, (sid, date, session) in enumerate(zip(item["haystack_session_ids"], item["haystack_dates"],
                                                 item["haystack_sessions"])):
        if sid in done:
            continue
        body = episode_body(session)
        if body:
            await graph.add_episode(name=f"session-{i}", episode_body=body, source=EpisodeType.message,
                                    source_description="chat session", reference_time=parse_date(date),
                                    group_id=group)
            added += 1
        done.add(sid)
        marker.write_text(json.dumps(sorted(done)))
    return added


def usage_of(graph: Graphiti) -> dict:
    m: Meter = graph.meter  # type: ignore[attr-defined]
    return {"llm_calls": m.calls, "llm_tokens_in": m.tokens_in, "llm_tokens_out": m.tokens_out,
            "llm_usd_est": round(m.usd, 5)}


async def run_one(http: OpenRouterHTTP, cfg: MemxConfig, item: dict, cfg_name: str, run_id: str, args) -> dict:
    group = f"{cfg_name}_{scope_label(item['question_id'])}"
    graph = build(cfg, args, cfg_name, group)
    jev_before = (CountingJev.requests, CountingJev.cost)
    try:
        t0 = time.perf_counter()
        await ingest(graph, item, group, cfg_name)
        t1 = time.perf_counter()
        config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
        config.limit = args.top_k
        results = await graph.search_(item["question"], config=config, group_ids=[group])
        t2 = time.perf_counter()
        usage = usage_of(graph)
    finally:
        await graph.close()
    memory = context(results)
    answer, _ = await chat(http, args.answer_model, ANSWER_PROMPT.format(
        memory=memory, question_date=item.get("question_date", ""), question=item["question"]))
    verdict_text, _ = await chat(http, args.judge_model, judge_prompt(item, answer), max_tokens=400)
    return {
        "run_id": run_id, "config": cfg_name, "question_id": item["question_id"],
        "question_type": "abstention" if item["question_id"].endswith("_abs") else item["question_type"],
        "question": item["question"], "answer": item["answer"], "response": answer,
        "correct": judged_yes(verdict_text), "judge": verdict_text[-200:], "mode": "graphiti",
        "facts": len(results.edges), "memory_tokens": math.ceil(len(memory) / CHARS_PER_TOKEN),
        "evidence_recall": None, "ingest_s": round(t1 - t0, 2), "recall_ms": round((t2 - t1) * 1000),
        "usage": {**usage, "jev_requests": CountingJev.requests - jev_before[0],
                  "jev_usd": round(CountingJev.cost - jev_before[1], 5)},
    }


async def main_async(args) -> None:
    items = json.loads(DATA.read_text())
    chosen = select(items, args.per_type, args.abstention, args.seed)
    if args.ids:
        wanted = [line.strip() for line in open(args.ids).read().splitlines() if line.strip()]
        by_id = {i["question_id"]: i for i in items}
        chosen = [by_id[q] for q in wanted]
    if args.limit:
        chosen = chosen[: args.limit]
    run_id = args.resume or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    cfg = MemxConfig()
    http = OpenRouterHTTP(cfg.api_key(), cfg.openrouter_base_url, timeout_s=180)
    for cfg_name in args.config:
        out = RESULTS / f"{run_id}-{cfg_name}.jsonl"
        previous = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
        answered = {r["question_id"] for r in previous}
        todo = [i for i in chosen if i["question_id"] not in answered]
        sem = asyncio.Semaphore(args.concurrency)

        async def guarded(item: dict, cfg_name: str = cfg_name, out=out) -> dict | None:
            async with sem:
                try:
                    row = await run_one(http, cfg, item, cfg_name, run_id, args)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! {item['question_id']}: {type(exc).__name__}: {exc}", flush=True)
                    return None
                with out.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"  {'✓' if row['correct'] else '✗'} {row['question_type']:<26} {row['question_id']} "
                      f"ingest {row['ingest_s']:.0f}s ${row['usage']['llm_usd_est'] + row['usage']['jev_usd']:.3f}",
                      flush=True)
                return row

        print(f"== {cfg_name}: {len(todo)} questions ({len(answered)} already answered)", flush=True)
        rows = previous + [r for r in await asyncio.gather(*(guarded(i) for i in todo)) if r]
        summary = {"run_id": run_id, "config": cfg_name, "extract_model": args.extract_model, **summarize(rows),
                   "ingest_usd": round(sum(r["usage"]["llm_usd_est"] + r["usage"]["jev_usd"] for r in rows), 4)}
        (RESULTS / f"{run_id}-{cfg_name}.summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    await http.aclose()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", default=["zep_llm"], choices=["zep_llm", "zep_jev"])
    p.add_argument("--per-type", type=int, default=12)
    p.add_argument("--abstention", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ids", default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-coroutines", type=int, default=8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--extract-model", default="anthropic/claude-haiku-5.5")
    p.add_argument("--answer-model", default="anthropic/claude-haiku-5.5")
    p.add_argument("--judge-model", default="anthropic/claude-sonnet-5.5")
    p.add_argument("--falkor-host", default="localhost")
    p.add_argument("--falkor-port", type=int, default=6390)
    p.add_argument("--resume", default=None)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
