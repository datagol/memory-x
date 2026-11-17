"""Read-only queries for the dashboard."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from psycopg import AsyncConnection


from memx.db.repo._base import FACT_COLS, to_fact
from memx.db.repo.entities import mentions_for


SUBTREE = "(%(scope)s::ltree IS NULL OR {col} <@ %(scope)s::ltree)"


async def list_scopes(conn: AsyncConnection, prefix: str | None, limit: int) -> list[dict[str, Any]]:
    cur = await conn.execute(
        f"""
        SELECT s.path::text AS path, nlevel(s.path) AS depth, s.created_at,
               count(f.id) FILTER (WHERE f.status = 'active') AS active,
               count(f.id) FILTER (WHERE f.status = 'superseded') AS superseded,
               count(f.id) FILTER (WHERE f.status = 'expired') AS expired,
               greatest(max(f.created_at), s.created_at) AS last_activity
          FROM memx_scope s LEFT JOIN memx_fact f ON f.scope = s.path
         WHERE {SUBTREE.format(col="s.path")}
         GROUP BY s.path, s.created_at
         ORDER BY s.path
         LIMIT %(limit)s
        """,
        {"scope": prefix, "limit": limit},
    )
    return await cur.fetchall()


async def stats(conn: AsyncConnection, scope: str | None) -> dict[str, Any]:
    params = {"scope": scope}
    cur = await conn.execute(
        f"SELECT category, status, count(*) AS n FROM memx_fact f WHERE {SUBTREE.format(col='f.scope')} "
        "GROUP BY 1, 2", params)
    by_cat = await cur.fetchall()
    cur = await conn.execute(
        f"SELECT status, count(*) AS n FROM memx_ingest i WHERE {SUBTREE.format(col='i.scope')} GROUP BY 1", params)
    ingests = {r["status"]: r["n"] for r in await cur.fetchall()}
    cur = await conn.execute(
        f"SELECT count(*) AS n, coalesce(avg(latency_ms), 0)::int AS avg_ms, coalesce(avg(token_count), 0)::int AS avg_tokens "
        f"FROM memx_recall_log r WHERE {SUBTREE.format(col='r.scope')}", params)
    recalls = await cur.fetchone()
    cur = await conn.execute(
        f"SELECT step, count(*) AS n FROM memx_decision d WHERE {SUBTREE.format(col='d.scope')} GROUP BY 1", params)
    decisions = {r["step"]: r["n"] for r in await cur.fetchall()}
    cur = await conn.execute(
        f"""
        SELECT kind, count(*) AS n FROM memx_fact_link l JOIN memx_fact f ON f.id = l.a
         WHERE {SUBTREE.format(col='f.scope')} GROUP BY 1
        """, params)
    links = {r["kind"]: r["n"] for r in await cur.fetchall()}
    cur = await conn.execute(
        f"""
        SELECT count(DISTINCT fe.entity_id) AS entities, count(*) AS mentions
          FROM memx_fact_entity fe JOIN memx_fact f ON f.id = fe.fact_id WHERE {SUBTREE.format(col='f.scope')}
        """, params)
    ent = await cur.fetchone()
    links["mentions"] = int(ent["mentions"])
    cur = await conn.execute(
        f"""
        SELECT to_char(date_trunc('day', created_at), 'YYYY-MM-DD') AS day, count(*) AS n
          FROM memx_fact f WHERE {SUBTREE.format(col='f.scope')} AND created_at > now() - interval '30 days'
         GROUP BY 1 ORDER BY 1
        """, params)
    per_day = await cur.fetchall()
    cur = await conn.execute(
        f"SELECT count(*) AS n FROM memx_scope s WHERE {SUBTREE.format(col='s.path')}", params)
    scopes = (await cur.fetchone())["n"]
    return {"facts": by_cat, "ingests": ingests, "recalls": recalls, "decisions": decisions, "links": links,
            "facts_per_day": per_day, "scopes": scopes, "entities": int(ent["entities"])}


async def list_ingests(conn: AsyncConnection, scope: str | None, limit: int, offset: int) -> list[dict[str, Any]]:
    cur = await conn.execute(
        f"""
        SELECT id, scope::text AS scope, kind, mode, status, source, trust, observed_at, created_at, committed_at,
               receipt, error, payload, jsonb_array_length(coalesce(candidates, '[]'::jsonb)) AS candidates
          FROM memx_ingest i WHERE {SUBTREE.format(col='i.scope')}
         ORDER BY created_at DESC LIMIT %(limit)s OFFSET %(offset)s
        """,
        {"scope": scope, "limit": limit, "offset": offset},
    )
    return await cur.fetchall()


async def list_recalls(conn: AsyncConnection, scope: str | None, limit: int, offset: int) -> list[dict[str, Any]]:
    cur = await conn.execute(
        f"""
        SELECT id, scope::text AS scope, query, mode, cardinality(fact_ids) AS facts, token_count, latency_ms,
               created_at
          FROM memx_recall_log r WHERE {SUBTREE.format(col='r.scope')}
         ORDER BY created_at DESC LIMIT %(limit)s OFFSET %(offset)s
        """,
        {"scope": scope, "limit": limit, "offset": offset},
    )
    return await cur.fetchall()


async def get_recall(conn: AsyncConnection, recall_id: UUID) -> dict[str, Any] | None:
    cur = await conn.execute(
        "SELECT *, scope::text AS scope FROM memx_recall_log WHERE id = %s", (recall_id,))
    return await cur.fetchone()


async def list_jobs(conn: AsyncConnection, status: str | None, limit: int) -> list[dict[str, Any]]:
    cur = await conn.execute(
        """
        SELECT id, kind, payload, status, attempts, max_attempts, run_after, locked_at, locked_by, last_error,
               created_at, finished_at
          FROM memx_job WHERE (%(status)s::text IS NULL OR status = %(status)s)
         ORDER BY id DESC LIMIT %(limit)s
        """,
        {"status": status, "limit": limit},
    )
    return await cur.fetchall()


async def graph(conn: AsyncConnection, scope: str, *, view: str, statuses: list[str], limit: int) -> dict[str, Any]:
    """Facts around a scope plus every link between them. view: 'chain' (scope and ancestors, what recall sees)
    or 'subtree' (scope and descendants)."""
    if view == "chain":
        labels = scope.split(".")
        where = "f.scope = ANY(%(scopes)s::ltree[])"
        params: dict[str, Any] = {"scopes": [".".join(labels[: i + 1]) for i in range(len(labels))]}
    else:
        where = "f.scope <@ %(scope)s::ltree"
        params = {"scope": scope}
    params |= {"statuses": statuses, "limit": limit}
    cur = await conn.execute(
        f"""
        SELECT {FACT_COLS} FROM memx_fact f
         WHERE {where} AND f.status = ANY(%(statuses)s)
         ORDER BY f.created_at DESC LIMIT %(limit)s
        """,
        params,
    )
    facts = [to_fact(r) for r in await cur.fetchall()]
    ids = [f.id for f in facts]
    cur = await conn.execute(
        "SELECT a, b, kind, similarity FROM memx_fact_link WHERE a = ANY(%s) AND b = ANY(%s)", (ids, ids))
    links = await cur.fetchall()
    cur = await conn.execute(f"SELECT count(*) AS n FROM memx_fact f WHERE {where} AND f.status = ANY(%(statuses)s)",
                             params)
    total = (await cur.fetchone())["n"]
    mentions = await mentions_for(conn, ids)
    return {"facts": facts, "links": links, "total": total, "mentions": mentions}
