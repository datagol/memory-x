"""Facts: inserts, neighbors for reconcile, links, reinforcement."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pgvector import Vector
from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from memx.types import Fact

from memx.db.repo._base import FACT_COLS, NEIGHBOR_COLS, or_tsquery, to_fact


async def get_fact(conn: AsyncConnection, fact_id: UUID) -> Fact | None:
    cur = await conn.execute(f"SELECT {FACT_COLS} FROM memx_fact f WHERE f.id = %s", (fact_id,))
    row = await cur.fetchone()
    return to_fact(row) if row else None


async def get_fact_embedding(conn: AsyncConnection, fact_id: UUID) -> list[float] | None:
    cur = await conn.execute("SELECT embedding FROM memx_fact WHERE id = %s", (fact_id,))
    row = await cur.fetchone()
    if not row:
        return None
    vec = row["embedding"]
    return vec.to_list() if hasattr(vec, "to_list") else [float(x) for x in vec]


async def facts_by_ids(conn: AsyncConnection, ids: list[UUID]) -> list[Fact]:
    if not ids:
        return []
    cur = await conn.execute(f"SELECT {FACT_COLS} FROM memx_fact f WHERE f.id = ANY(%s)", (ids,))
    by_id = {r["id"]: to_fact(r) for r in await cur.fetchall()}
    return [by_id[i] for i in ids if i in by_id]


async def find_exact(conn: AsyncConnection, chain: list[str], digest: bytes) -> dict[str, Any] | None:
    cur = await conn.execute(
        f"""
        SELECT {FACT_COLS} FROM memx_fact f
         WHERE f.scope = ANY(%s::ltree[]) AND f.text_hash = %s AND f.status = 'active'
         ORDER BY nlevel(f.scope) DESC LIMIT 1
        """,
        (chain, digest),
    )
    return await cur.fetchone()


async def find_neighbors(
    conn: AsyncConnection, chain: list[str], embedding: list[float], text: str, *, k_vector: int, k_fts: int,
    created_after: datetime | None = None, scope_only: str | None = None, below: str | None = None,
) -> list[dict[str, Any]]:
    """Active facts near a statement: vector top-k plus full-text top-k, each with cosine similarity.
    Searches the scope chain, plus every scope under ``below`` when given."""
    scopes = [scope_only] if scope_only else chain
    extra = "AND f.created_at > %(after)s" if created_after else ""
    params: dict[str, Any] = {"chain": scopes, "e": Vector(embedding), "kv": k_vector, "kf": k_fts,
                              "after": created_after, "below": below}
    where = ("(f.scope = ANY(%(chain)s::ltree[]) OR f.scope <@ %(below)s::ltree)" if below
             else "f.scope = ANY(%(chain)s::ltree[])")
    tsq = or_tsquery(text)
    fts = ""
    if tsq and k_fts > 0:
        params["q"] = tsq
        fts = f"""
        UNION ALL
        (SELECT {FACT_COLS}, {NEIGHBOR_COLS}, 1 - (f.embedding <=> %(e)s::vector) AS sim, 'fts' AS via
           FROM memx_fact f
          WHERE {where} AND f.status = 'active' {extra}
            AND f.tsv @@ to_tsquery('english', %(q)s)
          ORDER BY ts_rank(f.tsv, to_tsquery('english', %(q)s)) DESC
          LIMIT %(kf)s)
        """
    sql = f"""
        (SELECT {FACT_COLS}, {NEIGHBOR_COLS}, 1 - (f.embedding <=> %(e)s::vector) AS sim, 'vec' AS via
           FROM memx_fact f
          WHERE {where} AND f.status = 'active' {extra}
          ORDER BY f.embedding <=> %(e)s::vector
          LIMIT %(kv)s)
        {fts}
    """
    async with conn.transaction():
        await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        await conn.execute("SET LOCAL hnsw.ef_search = 100")
        cur = await conn.execute(sql, params)
        rows = await cur.fetchall()
    seen: dict[UUID, dict[str, Any]] = {}
    for row in rows:
        if row["id"] not in seen:
            seen[row["id"]] = row
    return sorted(seen.values(), key=lambda r: -r["sim"])


async def insert_fact(conn: AsyncConnection, values: dict[str, Any]) -> UUID | None:
    cols = list(values)
    placeholders = ", ".join(
        "%s::ltree" if c == "scope" else "%s::vector" if c == "embedding" else "%s" for c in cols
    )
    params = [
        Jsonb(v) if c in ("flags", "sort_probs", "attrs") else Vector(v) if c == "embedding" else v
        for c, v in values.items()
    ]
    cur = await conn.execute(
        f"""
        INSERT INTO memx_fact ({", ".join(cols)}) VALUES ({placeholders})
        ON CONFLICT (ingest_id, candidate_index) DO NOTHING
        RETURNING id
        """,
        params,
    )
    row = await cur.fetchone()
    return row["id"] if row else None


async def fact_id_for_candidate(conn: AsyncConnection, ingest_id: UUID, index: int) -> UUID | None:
    cur = await conn.execute(
        "SELECT id FROM memx_fact WHERE ingest_id = %s AND candidate_index = %s", (ingest_id, index)
    )
    row = await cur.fetchone()
    return row["id"] if row else None


async def reinforce(conn: AsyncConnection, fact_id: UUID, at: datetime) -> None:
    """Count a repeat. ``at`` is when the repeat was said (the ingest's observed_at), not when it was processed:
    a transcript backfilled today must not make a 2023 fact look new."""
    await conn.execute(
        """
        UPDATE memx_fact SET reinforced_count = reinforced_count + 1,
               last_reinforced_at = greatest(last_reinforced_at, %s)
         WHERE id = %s
        """,
        (at, fact_id),
    )


async def supersede(conn: AsyncConnection, ids: list[UUID], by: UUID, scope: str) -> list[UUID]:
    if not ids:
        return []
    cur = await conn.execute(
        """
        UPDATE memx_fact SET status = 'superseded', superseded_by = %s, superseded_at = now()
         WHERE id = ANY(%s) AND status = 'active' AND scope = %s::ltree
        RETURNING id
        """,
        (by, ids, scope),
    )
    return [r["id"] for r in await cur.fetchall()]


async def supersede_below(conn: AsyncConnection, ids: list[UUID], by: UUID, scope: str) -> list[UUID]:
    """Retire facts in scopes under ``scope`` that a newer, broader fact replaces."""
    if not ids:
        return []
    cur = await conn.execute(
        """
        UPDATE memx_fact SET status = 'superseded', superseded_by = %s, superseded_at = now()
         WHERE id = ANY(%s) AND status = 'active' AND scope <@ %s::ltree AND scope <> %s::ltree
        RETURNING id
        """,
        (by, ids, scope, scope),
    )
    return [r["id"] for r in await cur.fetchall()]


async def still_active(conn: AsyncConnection, ids: list[UUID]) -> set[UUID]:
    if not ids:
        return set()
    cur = await conn.execute("SELECT id FROM memx_fact WHERE id = ANY(%s) AND status = 'active'", (ids,))
    return {r["id"] for r in await cur.fetchall()}


async def add_links(conn: AsyncConnection, links: list[tuple[UUID, UUID, str, float | None]]) -> None:
    for a, b, kind, sim in links:
        await conn.execute(
            """
            INSERT INTO memx_fact_link (a, b, kind, similarity) VALUES (%s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (a, b, kind, sim),
        )
