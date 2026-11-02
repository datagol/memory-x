"""Step 4: apply a reconcile plan atomically under a per-scope advisory lock."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection

from memx.config import MemxConfig
from memx.db import repo
from memx.pipeline.entities import EntityPlan, apply_entities, entity_decisions, tenant_of
from memx.pipeline.reconcile import VERDICT_VERSION, Plan, Verdict
from memx.pipeline.similarity import lexical_guard, text_hash
from memx.pipeline.sort import SORT_VERSION, SortResult
from memx.tokens import estimate_tokens
from memx.types import Candidate, Category


def speaker_trust(cfg: MemxConfig, ingest: dict[str, Any], speaker: str) -> float:
    if ingest["kind"] == "text":
        return float(ingest["trust"])
    if speaker == "assistant":
        return cfg.assistant_trust
    return cfg.user_trust


def event_times(cfg: MemxConfig, candidate: Candidate, category: Category, observed_at: datetime,
                settings: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
    """Upcoming events ("on PTO next week") expire after their date. Past and undated events are history:
    they never expire, and the event half-life fades them in ranking instead."""
    event_at = datetime.combine(candidate.event_date, time(12), tzinfo=UTC) if candidate.event_date else None
    if category != Category.event or event_at is None or event_at.date() < observed_at.date():
        return event_at, None
    return event_at, event_at + timedelta(days=settings.get("event_ttl_days") or cfg.event_ttl_days)


def sort_decision(ingest: dict[str, Any], candidate: Candidate, sort: SortResult | None, scope: str,
                  subject: UUID | None, note: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"text": candidate.text}
    if sort:
        result |= {"category": sort.category.value, "confidence": sort.confidence,
                   "probabilities": sort.probabilities, "injection": sort.injection,
                   "topic": sort.topic, "topic_confidence": sort.topic_confidence}
    if note:
        result["note"] = note
    return {
        "step": "sort", "question_version": SORT_VERSION, "ingest_id": ingest["id"],
        "candidate_index": candidate.index, "subject_fact_id": subject, "scope": scope,
        "result": result, "latency_ms": sort.latency_ms if sort else None,
    }


def verdict_decisions(ingest: dict[str, Any], candidate: Candidate, verdicts: list[Verdict], scope: str,
                      subject: UUID | None) -> list[dict[str, Any]]:
    return [
        {
            "step": "verdict" if v.source == "jev" else "auto",
            "question_version": VERDICT_VERSION if v.source == "jev" else f"band.{v.source}",
            "ingest_id": ingest["id"], "candidate_index": candidate.index, "subject_fact_id": subject,
            "object_fact_id": v.neighbor["id"], "scope": scope, "similarity": v.similarity,
            "result": {"label": v.label, "probability": v.probability, "probabilities": v.probabilities,
                       "candidate": candidate.text, "neighbor": v.neighbor["text"]},
            "latency_ms": v.latency_ms,
        }
        for v in verdicts
    ]


async def db_now(conn: AsyncConnection) -> datetime:
    cur = await conn.execute("SELECT clock_timestamp() AS t")
    return (await cur.fetchone())["t"]


async def _recheck(conn: AsyncConnection, cfg: MemxConfig, plan: Plan, *, chain: list[str], scope: str,
                   candidate: Candidate, embedding: list[float]) -> Plan:
    """Under the lock, catch duplicates written by a concurrent ingest since neighbors were fetched."""
    if plan.action != "insert" or plan.status != "active":
        return plan
    exact = await repo.find_exact(conn, chain, text_hash(candidate.text))
    if exact:
        return Plan("drop", reinforce=exact["id"], supersede_below=plan.supersede_below, verdicts=plan.verdicts)
    if plan.fetched_at is not None:
        late = await repo.find_neighbors(
            conn, chain, embedding, candidate.text, k_vector=5, k_fts=0,
            created_after=plan.fetched_at, scope_only=scope,
        )
        for row in late:
            if row["sim"] >= cfg.band_high and lexical_guard(candidate.text, row["text"], cfg.guard_min_jaccard):
                return Plan("drop", reinforce=row["id"], supersede_below=plan.supersede_below, verdicts=plan.verdicts)
    return plan


async def commit_candidate(
    conn: AsyncConnection, cfg: MemxConfig, *, ingest: dict[str, Any], candidate: Candidate, embedding: list[float],
    embedding_model: str, sort: SortResult, plan: Plan, scope: str, chain: list[str], settings: dict[str, Any],
    trace: dict[str, Any] | None = None, entities: list[EntityPlan] | None = None,
) -> dict[str, Any]:
    outcome: dict[str, Any] = {"outcome": None, "fact_id": None, "superseded": 0, "shadowed": 0, **(trace or {})}
    entities = entities or []
    tenant = tenant_of(scope)
    async with conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (scope,))
        plan = await _recheck(conn, cfg, plan, chain=chain, scope=scope, candidate=candidate, embedding=embedding)

        outcome["verdicts"] = [
            {"neighbor_id": str(v.neighbor["id"]), "label": v.label, "probability": round(v.probability, 4),
             "probabilities": v.probabilities, "source": v.source, "sim": round(v.similarity, 4)}
            for v in plan.verdicts
        ]
        if plan.action == "drop":
            await repo.reinforce(conn, plan.reinforce)
            active = await repo.still_active(conn, plan.supersede_below)
            retired = await repo.supersede_below(
                conn, [i for i in plan.supersede_below if i in active], plan.reinforce, scope)
            # A repeat still teaches us which names the existing fact mentions.
            linked = await apply_entities(conn, tenant=tenant, fact_id=plan.reinforce, plans=entities)
            outcome |= {"outcome": "dropped", "same_as": str(plan.reinforce), "superseded": len(retired),
                        "superseded_ids": [str(i) for i in retired],
                        "entities": [p.trace() for p in entities], "entity_ids": [str(i) for i in linked]}
            await repo.add_decisions(conn, [sort_decision(ingest, candidate, sort, scope, None)]
                                     + verdict_decisions(ingest, candidate, plan.verdicts, scope, None)
                                     + entity_decisions(ingest["id"], candidate, entities, scope, plan.reinforce))
        else:
            event_at, stale_after = event_times(cfg, candidate, sort.category, ingest["observed_at"], settings)
            flags: dict[str, Any] = {}
            if sort.injection >= cfg.injection_flag:
                flags = {"injection_suspect": True, "injection_p": sort.injection}
            fact_id = await repo.insert_fact(conn, {
                "scope": scope,
                "category": sort.category.value,
                "text": candidate.text,
                "text_hash": text_hash(candidate.text),
                "token_count": estimate_tokens(candidate.text, cfg.chars_per_token) + 2,
                "embedding": embedding,
                "embedding_model": embedding_model,
                "status": plan.status,
                "superseded_by": plan.superseded_by,
                "superseded_at": datetime.now(UTC) if plan.status == "superseded" else None,
                "ingest_id": ingest["id"],
                "candidate_index": candidate.index,
                "speaker": candidate.speaker,
                "source": ingest.get("source"),
                "trust": speaker_trust(cfg, ingest, candidate.speaker),
                "extract_confidence": candidate.confidence,
                "sort_confidence": sort.confidence,
                "sort_probs": sort.probabilities,
                "flags": flags,
                "observed_at": ingest["observed_at"],
                "event_at": event_at,
                "stale_after": stale_after,
                "topic": sort.topic,
                "topic_confidence": sort.topic_confidence,
                "attrs": {"time_text": candidate.time_text, "window": candidate.window,
                          "category_hint": candidate.category_hint.value if candidate.category_hint else None},
            })
            if fact_id is None:  # committed by an earlier attempt of this job
                fact_id = await repo.fact_id_for_candidate(conn, ingest["id"], candidate.index)
                outcome |= {"outcome": "added", "fact_id": str(fact_id)}
            else:
                linked = await apply_entities(conn, tenant=tenant, fact_id=fact_id, plans=entities)
                active = await repo.still_active(
                    conn, plan.supersede + plan.supersede_below + plan.shadow + [r for r, _ in plan.related])
                superseded = await repo.supersede(conn, [i for i in plan.supersede if i in active], fact_id, scope)
                superseded += await repo.supersede_below(
                    conn, [i for i in plan.supersede_below if i in active], fact_id, scope)
                shadows = [i for i in plan.shadow if i in active]
                links = [(fact_id, b, "shadows", None) for b in shadows]
                links += [(fact_id, b, "related", sim) for b, sim in plan.related if b in active]
                await repo.add_links(conn, links)
                await repo.add_decisions(conn, [sort_decision(ingest, candidate, sort, scope, fact_id)]
                                         + verdict_decisions(ingest, candidate, plan.verdicts, scope, fact_id)
                                         + entity_decisions(ingest["id"], candidate, entities, scope, fact_id))
                outcome |= {
                    "outcome": "added" if plan.status == "active" else "superseded_on_arrival",
                    "fact_id": str(fact_id), "superseded": len(superseded), "shadowed": len(shadows),
                    "category": sort.category.value, "topic": sort.topic,
                    "entities": [p.trace() for p in entities], "entity_ids": [str(i) for i in linked],
                    "superseded_ids": [str(i) for i in superseded], "shadowed_ids": [str(i) for i in shadows],
                    "related_ids": [str(b) for b, _ in plan.related if b in active],
                    "superseded_by": str(plan.superseded_by) if plan.superseded_by else None,
                }
        await repo.set_candidate_outcome(
            conn, ingest["id"], candidate.index, {**candidate.model_dump(mode="json"), **outcome}
        )
    return outcome


async def reject_candidate(conn: AsyncConnection, *, ingest: dict[str, Any], candidate: Candidate,
                           sort: SortResult | None, scope: str, reason: str,
                           trace: dict[str, Any] | None = None) -> dict[str, Any]:
    outcome = {"outcome": "rejected", "reason": reason, "fact_id": None, "superseded": 0, "shadowed": 0,
               **(trace or {})}
    async with conn.transaction():
        await repo.add_decisions(conn, [sort_decision(ingest, candidate, sort, scope, None, note=reason)])
        await repo.set_candidate_outcome(
            conn, ingest["id"], candidate.index, {**candidate.model_dump(mode="json"), **outcome}
        )
    return outcome
