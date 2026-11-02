"""The ingest job: extract → sort → reconcile → commit. Resumable at every step."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import UUID

from memx.context import Context
from memx.db import repo
from memx.pipeline.commit import commit_candidate, db_now, reject_candidate
from memx.pipeline.entities import EntityPlan, plan_entities
from memx.pipeline.extract import ensure_candidates
from memx.pipeline.reconcile import Plan, judge_neighbors, plan_from
from memx.pipeline.sort import SortResult, sort_candidate
from memx.scopes import ScopePath
from memx.types import Candidate


def build_receipt(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    receipt = {"added": 0, "dropped": 0, "historical": 0, "superseded": 0, "shadowed": 0, "rejected": 0,
               "fact_ids": []}
    for c in candidates:
        outcome = c.get("outcome")
        if outcome == "added":
            receipt["added"] += 1
            receipt["fact_ids"].append(c["fact_id"])
        elif outcome == "dropped":
            receipt["dropped"] += 1
        elif outcome == "superseded_on_arrival":
            receipt["historical"] += 1
        elif outcome == "rejected":
            receipt["rejected"] += 1
        receipt["superseded"] += int(c.get("superseded") or 0)
        receipt["shadowed"] += int(c.get("shadowed") or 0)
    return receipt


def sort_trace(sort: SortResult | None) -> dict[str, Any] | None:
    if sort is None:
        return None
    return {"category": sort.category.value, "confidence": round(sort.confidence, 4),
            "probabilities": sort.probabilities, "injection": round(sort.injection, 4), "ms": sort.latency_ms,
            "topic": sort.topic, "topic_confidence": round(sort.topic_confidence or 0.0, 4),
            "topic_probabilities": dict(sorted(sort.topic_probabilities.items(), key=lambda kv: -kv[1])[:4])}


async def _process(ctx: Context, ingest: dict[str, Any], candidate: Candidate, embedding: list[float],
                   sort: SortResult | None, scope: str, chain: list[str], settings: dict[str, Any]) -> dict[str, Any]:
    cfg = ctx.config
    started = time.perf_counter()
    trace: dict[str, Any] = {"sort": sort_trace(sort), "neighbors": []}
    async with ctx.pool.connection() as conn:
        if sort is None:
            return await reject_candidate(conn, ingest=ingest, candidate=candidate, sort=None, scope=scope,
                                          reason="too_long", trace=trace)
        if sort.injection >= cfg.injection_reject:
            return await reject_candidate(conn, ingest=ingest, candidate=candidate, sort=sort, scope=scope,
                                          reason="injection", trace=trace)
        fetched_at = await db_now(conn)
        entities: list[EntityPlan] = []
        if sort.injection >= cfg.injection_flag:
            plan = Plan("insert")  # a suspect fact never retires or links to anything
            trace["note"] = "suspected injection: stored flagged, not reconciled"
        else:
            verdicts = await judge_neighbors(
                conn, ctx.judge, cfg, scope=scope, chain=chain, candidate=candidate, category=sort.category,
                embedding=embedding, observed_at=ingest["observed_at"], considered=trace["neighbors"],
            )
            plan = plan_from(verdicts, cfg, scope)
            entities = await plan_entities(conn, ctx.judge, cfg, scope=scope, candidate=candidate)
        plan.fetched_at = fetched_at
        trace["reconcile_ms"] = int((time.perf_counter() - started) * 1000)
        return await commit_candidate(
            conn, cfg, ingest=ingest, candidate=candidate, embedding=embedding,
            embedding_model=ctx.embedder.model, sort=sort, plan=plan, scope=scope, chain=chain, settings=settings,
            trace=trace, entities=entities,
        )


async def run_ingest(ctx: Context, ingest_id: UUID) -> dict[str, Any] | None:
    """Process one ingest. Returns the receipt, or None if the ingest was deleted meanwhile."""
    cfg = ctx.config
    async with ctx.pool.connection() as conn:
        ingest = await repo.get_ingest(conn, ingest_id)
    if ingest is None:
        return None
    if ingest["status"] == "committed":
        return ingest["receipt"]

    t0 = time.perf_counter()
    candidates, embeddings = await ensure_candidates(ctx, ingest)
    t_extract = time.perf_counter()
    scope = ingest["scope"]
    chain = ScopePath.parse(scope).chain_str()
    async with ctx.pool.connection() as conn:
        await repo.ensure_scope(conn, scope)
        settings = await repo.scope_settings(conn, scope)

    pending = [c for c in candidates if c.outcome is None]
    missing = [c for c in pending if c.index not in embeddings]
    if missing:
        for c, vec in zip(missing, await ctx.embedder.embed([c.text for c in missing])):
            embeddings[c.index] = vec

    sem = asyncio.Semaphore(cfg.jev_concurrency)

    async def sort_one(c: Candidate) -> SortResult | None:
        if len(c.text) > cfg.max_fact_chars:
            return None
        async with sem:
            return await sort_candidate(ctx.judge, c, cfg, use_hint=cfg.sort_use_hint)

    t_embed = time.perf_counter()
    sorts = await asyncio.gather(*(sort_one(c) for c in pending))
    t_sort = time.perf_counter()

    # Reconcile and commit one at a time so each candidate sees the ones committed before it.
    for c, s in zip(pending, sorts):
        await _process(ctx, ingest, c, embeddings[c.index], s, scope, chain, settings)

    async with ctx.pool.connection() as conn:
        final = await repo.get_ingest(conn, ingest_id)
        if final is None:
            return None
        receipt = build_receipt(final["candidates"] or [])
        t_end = time.perf_counter()
        receipt["timings"] = {
            "extract_ms": int((t_extract - t0) * 1000), "embed_ms": int((t_embed - t_extract) * 1000),
            "sort_ms": int((t_sort - t_embed) * 1000), "reconcile_commit_ms": int((t_end - t_sort) * 1000),
            "total_ms": int((t_end - t0) * 1000),
        }
        await repo.finish_ingest(conn, ingest_id, receipt)
    return receipt
