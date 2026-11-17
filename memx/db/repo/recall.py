"""Recall: what a scope chain can see, candidate search, access marks and logs."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pgvector import Vector
from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from memx.types import Fact

from memx.db.repo._base import FACT_COLS, VISIBLE, or_tsquery, to_fact


async def chain_token_total(conn: AsyncConnection, chain: list[str], as_of: datetime | None = None) -> tuple[int, int]:
    cur = await conn.execute(
        f"SELECT coalesce(sum(f.token_count), 0) AS t, count(*) AS n FROM memx_fact f "
        f"WHERE f.scope = ANY(%(chain)s::ltree[]) AND {VISIBLE}",
        {"chain": chain, "as_of": as_of},
    )
    row = await cur.fetchone()
    return int(row["t"]), int(row["n"])


async def chain_facts(conn: AsyncConnection, chain: list[str], categories: list[str] | None = None,
                      as_of: datetime | None = None) -> list[Fact]:
    cat = "AND f.category = ANY(%(cats)s)" if categories else ""
    cur = await conn.execute(
        f"SELECT {FACT_COLS} FROM memx_fact f WHERE f.scope = ANY(%(chain)s::ltree[]) AND {VISIBLE} {cat}",
        {"chain": chain, "cats": categories, "as_of": as_of},
    )
    return [to_fact(r) for r in await cur.fetchall()]


async def shadowed_ids(conn: AsyncConnection, chain: list[str]) -> set[UUID]:
    """Ancestor facts hidden by an active child-scope fact within this chain."""
    cur = await conn.execute(
        """
        SELECT l.b FROM memx_fact_link l JOIN memx_fact f ON f.id = l.a
         WHERE l.kind = 'shadows' AND f.status = 'active' AND f.scope = ANY(%s::ltree[])
        """,
        (chain,),
    )
    return {r["b"] for r in await cur.fetchall()}


async def replaced_versions(conn: AsyncConnection, ids: list[UUID], chain: list[str],
                            as_of: datetime | None = None, per_fact: int = 2) -> dict[UUID, list[Fact]]:
    """Earlier versions directly replaced by each of ``ids`` (newest first), as of a moment.

    Only versions an agent at this scope chain could have seen: a parent-scope fact can retire facts in many child
    scopes, and a sibling's retired fact must not appear here. Flagged facts stay hidden, and a promotion (the same
    text copied to a broader scope) is not a change of value.
    """
    if not ids:
        return {}
    cur = await conn.execute(
        f"""
        SELECT {FACT_COLS} FROM memx_fact f JOIN memx_fact r ON r.id = f.superseded_by
         WHERE f.superseded_by = ANY(%(ids)s) AND f.status = 'superseded'
           AND f.scope = ANY(%(chain)s::ltree[])
           AND NOT (f.flags ? 'injection_suspect')
           AND f.text_hash <> r.text_hash
           AND coalesce(r.attrs->>'promoted_from', '') <> f.id::text
           AND (f.superseded_at IS NULL OR f.superseded_at <= coalesce(%(as_of)s::timestamptz, now()))
         ORDER BY f.observed_at DESC, f.id
        """,
        {"ids": ids, "chain": chain, "as_of": as_of},
    )
    out: dict[UUID, list[Fact]] = {}
    for row in await cur.fetchall():
        versions = out.setdefault(row["superseded_by"], [])
        if len(versions) < per_fact:
            versions.append(to_fact(row))
    return out


async def related_pairs(conn: AsyncConnection, ids: list[UUID]) -> list[tuple[UUID, UUID]]:
    if not ids:
        return []
    cur = await conn.execute(
        "SELECT a, b FROM memx_fact_link WHERE kind = 'related' AND a = ANY(%s) AND b = ANY(%s)", (ids, ids)
    )
    return [(r["a"], r["b"]) for r in await cur.fetchall()]


async def hybrid_candidates(
    conn: AsyncConnection, chain: list[str], embedding: list[float], text: str, *, k_vector: int, k_fts: int,
    categories: list[str] | None = None, as_of: datetime | None = None,
) -> list[dict[str, Any]]:
    """Visible facts for a query: vector and full-text results merged by reciprocal rank."""
    cat = "AND f.category = ANY(%(cats)s)" if categories else ""
    params: dict[str, Any] = {"chain": chain, "e": Vector(embedding), "kv": k_vector, "kf": k_fts,
                              "cats": categories, "as_of": as_of}
    tsq = or_tsquery(text)
    async with conn.transaction():
        await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        await conn.execute("SET LOCAL hnsw.ef_search = 100")
        cur = await conn.execute(
            f"""
            SELECT {FACT_COLS}, 1 - (f.embedding <=> %(e)s::vector) AS sim FROM memx_fact f
             WHERE f.scope = ANY(%(chain)s::ltree[]) AND {VISIBLE} {cat}
             ORDER BY f.embedding <=> %(e)s::vector LIMIT %(kv)s
            """,
            params,
        )
        vec = await cur.fetchall()
        fts: list[dict[str, Any]] = []
        if tsq and k_fts > 0:
            params["q"] = tsq
            cur = await conn.execute(
                f"""
                SELECT {FACT_COLS}, 1 - (f.embedding <=> %(e)s::vector) AS sim FROM memx_fact f
                 WHERE f.scope = ANY(%(chain)s::ltree[]) AND {VISIBLE} {cat}
                   AND f.tsv @@ to_tsquery('english', %(q)s)
                 ORDER BY ts_rank(f.tsv, to_tsquery('english', %(q)s)) DESC LIMIT %(kf)s
                """,
                params,
            )
            fts = await cur.fetchall()
    vec.sort(key=lambda r: -r["sim"])
    merged: dict[UUID, dict[str, Any]] = {}
    for source, ranked in (("vec_rank", vec), ("fts_rank", fts)):
        for rank, row in enumerate(ranked):
            entry = merged.setdefault(row["id"], {**row, "rrf": 0.0, "vec_rank": None, "fts_rank": None})
            entry["rrf"] += 1.0 / (60 + rank + 1)
            entry[source] = rank + 1
    return sorted(merged.values(), key=lambda r: -r["rrf"])


async def mark_accessed(conn: AsyncConnection, ids: list[UUID]) -> None:
    """Count a recall of each fact. Rows are locked in id order: concurrent recalls over overlapping facts (every
    full-memory recall of a scope touches the same rows) would otherwise lock them in different orders and deadlock."""
    if ids:
        await conn.execute(
            """
            UPDATE memx_fact f SET access_count = f.access_count + 1, last_accessed_at = now()
              FROM (SELECT id FROM memx_fact WHERE id = ANY(%s) ORDER BY id FOR UPDATE) locked
             WHERE f.id = locked.id
            """,
            (ids,),
        )


async def log_recall(
    conn: AsyncConnection, *, recall_id: UUID, scope: str, query: str | None, mode: str, fact_ids: list[UUID],
    scores: dict[str, float], token_count: int, latency_ms: int, explain: dict[str, Any] | None = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO memx_recall_log (id, scope, query, mode, fact_ids, scores, token_count, latency_ms, explain)
        VALUES (%s, %s::ltree, %s, %s, %s, %s, %s, %s, %s)
        """,
        (recall_id, scope, query, mode, fact_ids, Jsonb(scores), token_count, latency_ms,
         Jsonb(explain) if explain is not None else None),
    )
