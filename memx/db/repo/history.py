"""Listing, version history, decisions and forgetting."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from memx.types import Fact

from memx.db.repo._base import FACT_COLS, to_fact
from memx.db.repo.entities import prune_entities


async def list_facts(
    conn: AsyncConnection, scopes: list[str], *, status: str | None, category: str | None, limit: int, offset: int,
) -> list[Fact]:
    where = ["f.scope = ANY(%(scopes)s::ltree[])"]
    if status:
        where.append("f.status = %(status)s")
    if category:
        where.append("f.category = %(category)s")
    cur = await conn.execute(
        f"""
        SELECT {FACT_COLS} FROM memx_fact f WHERE {" AND ".join(where)}
         ORDER BY f.observed_at DESC, f.id LIMIT %(limit)s OFFSET %(offset)s
        """,
        {"scopes": scopes, "status": status, "category": category, "limit": limit, "offset": offset},
    )
    return [to_fact(r) for r in await cur.fetchall()]


async def subtree_facts(conn: AsyncConnection, scope: str) -> list[dict[str, Any]]:
    cur = await conn.execute(
        f"""
        SELECT {FACT_COLS}, f.extract_confidence, f.sort_probs
          FROM memx_fact f WHERE f.scope <@ %s::ltree ORDER BY f.scope, f.category, f.observed_at, f.id
        """,
        (scope,),
    )
    return await cur.fetchall()


async def links_for(conn: AsyncConnection, ids: list[UUID], within: str | None = None) -> list[dict[str, Any]]:
    """Links touching ``ids``. With ``within``, only links whose both ends sit at that scope or an ancestor of it."""
    if not ids:
        return []
    cur = await conn.execute(
        """
        SELECT l.a, l.b, l.kind, l.similarity, l.created_at
          FROM memx_fact_link l JOIN memx_fact fa ON fa.id = l.a JOIN memx_fact fb ON fb.id = l.b
         WHERE (l.a = ANY(%(ids)s) OR l.b = ANY(%(ids)s))
           AND (%(within)s::ltree IS NULL OR (fa.scope @> %(within)s::ltree AND fb.scope @> %(within)s::ltree))
        """,
        {"ids": ids, "within": within},
    )
    return await cur.fetchall()


async def version_chain(conn: AsyncConnection, fact_id: UUID) -> list[UUID]:
    """Every fact connected to this one through superseded_by, oldest first.

    Only versions at the fact's own scope or an ancestor of it: a parent-scope fact can retire facts in many child
    scopes, and its history must not show one child's facts to a reader of another."""
    cur = await conn.execute(
        """
        WITH RECURSIVE back AS (
            SELECT id, superseded_by FROM memx_fact WHERE id = %(id)s
            UNION
            SELECT f.id, f.superseded_by FROM memx_fact f JOIN back ON f.superseded_by = back.id
        ), fwd AS (
            SELECT id, superseded_by FROM memx_fact WHERE id = %(id)s
            UNION
            SELECT f.id, f.superseded_by FROM memx_fact f JOIN fwd ON f.id = fwd.superseded_by
        )
        SELECT f.id FROM memx_fact f
         WHERE f.id IN (SELECT id FROM back UNION SELECT id FROM fwd)
           AND f.scope @> (SELECT scope FROM memx_fact WHERE id = %(id)s)
         ORDER BY f.observed_at, f.created_at
        """,
        {"id": fact_id},
    )
    return [r["id"] for r in await cur.fetchall()]


async def decisions_for(conn: AsyncConnection, ids: list[UUID], within: str | None = None) -> list[dict[str, Any]]:
    """Decisions touching ``ids``. With ``within``, a decision whose other fact sits outside that scope's chain (a
    sibling or a descendant) is left out."""
    if not ids:
        return []
    cur = await conn.execute(
        """
        SELECT d.id, d.step, d.question_version, d.subject_fact_id, d.object_fact_id, d.similarity, d.result,
               d.latency_ms, d.created_at
          FROM memx_decision d
          LEFT JOIN memx_fact s ON s.id = d.subject_fact_id
          LEFT JOIN memx_fact o ON o.id = d.object_fact_id
         WHERE (d.subject_fact_id = ANY(%(ids)s) OR d.object_fact_id = ANY(%(ids)s))
           AND (%(within)s::ltree IS NULL
                OR ((s.id IS NULL OR s.scope @> %(within)s::ltree) AND (o.id IS NULL OR o.scope @> %(within)s::ltree)))
         ORDER BY d.created_at, d.id
        """,
        {"ids": ids, "within": within},
    )
    return await cur.fetchall()


async def forget_scope(conn: AsyncConnection, scope: str, descendants: bool) -> tuple[int, int, int]:
    op = "<@" if descendants else "="
    cur = await conn.execute(f"SELECT id FROM memx_fact WHERE scope {op} %s::ltree", (scope,))
    fact_ids = [r["id"] for r in await cur.fetchall()]
    cur = await conn.execute(f"SELECT id FROM memx_ingest WHERE scope {op} %s::ltree", (scope,))
    ingest_ids = [r["id"] for r in await cur.fetchall()]
    await conn.execute(
        f"""
        DELETE FROM memx_decision
         WHERE scope {op} %s::ltree OR ingest_id = ANY(%s) OR subject_fact_id = ANY(%s) OR object_fact_id = ANY(%s)
        """,
        (scope, ingest_ids, fact_ids, fact_ids),
    )
    await conn.execute(f"DELETE FROM memx_recall_log WHERE scope {op} %s::ltree", (scope,))
    await conn.execute(
        "DELETE FROM memx_job WHERE kind = 'ingest' AND payload->>'ingest_id' = ANY(%s)",
        ([str(i) for i in ingest_ids],),
    )
    await conn.execute("DELETE FROM memx_fact WHERE id = ANY(%s)", (fact_ids,))
    await conn.execute("DELETE FROM memx_ingest WHERE id = ANY(%s)", (ingest_ids,))
    cur = await conn.execute(f"DELETE FROM memx_scope WHERE path {op} %s::ltree RETURNING path", (scope,))
    scopes_removed = len(await cur.fetchall())
    await prune_entities(conn)  # entities left with no facts carry no content, but go too
    return len(fact_ids), len(ingest_ids), scopes_removed


FORGOTTEN = "[forgotten]"


async def forget_fact(conn: AsyncConnection, fact_id: UUID) -> int:
    """Delete one fact and every stored copy of its wording: the candidate it was extracted as and the recall logs
    that listed it. The raw input it came from stays (it may hold other facts); forgetting the scope removes that."""
    cur = await conn.execute("SELECT scope, text, ingest_id FROM memx_fact WHERE id = %s FOR UPDATE", (fact_id,))
    fact = await cur.fetchone()
    if fact is None:
        return 0
    await conn.execute(
        "DELETE FROM memx_decision WHERE subject_fact_id = %s OR object_fact_id = %s", (fact_id, fact_id)
    )
    fid, norm = str(fact_id), " ".join(fact["text"].split())

    def scrub(node: Any) -> None:
        """Blank the wording wherever this fact appears: as a candidate, or as a neighbor another candidate was
        compared with."""
        if isinstance(node, dict):
            if "text" in node and (fid in (node.get("id"), node.get("fact_id"))
                                   or " ".join(str(node["text"]).split()) == norm):
                node["text"] = FORGOTTEN
                if "entities" in node:
                    node["entities"] = []
            for value in node.values():
                scrub(value)
        elif isinstance(node, list):
            for value in node:
                scrub(value)

    # Its own ingest, plus any ingest at this scope or below that judged a candidate against it.
    cur = await conn.execute(
        """
        SELECT id, windows, candidates FROM memx_ingest
         WHERE id = %(ingest)s OR (scope <@ %(scope)s::ltree AND candidates::text LIKE %(like)s)
           FOR UPDATE
        """,
        {"ingest": fact["ingest_id"], "scope": fact["scope"], "like": f"%{fid}%"},
    )
    for ingest in await cur.fetchall():
        scrub(ingest["windows"])
        scrub(ingest["candidates"])
        await conn.execute(
            "UPDATE memx_ingest SET windows = %s, candidates = %s WHERE id = %s",
            (Jsonb(ingest["windows"]), Jsonb(ingest["candidates"]) if ingest["candidates"] is not None else None,
             ingest["id"]),
        )
    # A fact is visible to recalls at its own scope and below, so only those logs can hold it.
    await conn.execute(
        """
        UPDATE memx_recall_log r
           SET fact_ids = array_remove(r.fact_ids, %(id)s::uuid),
               scores = r.scores - %(fid)s::text,
               explain = CASE WHEN jsonb_typeof(r.explain->'candidates') = 'array' THEN
                   jsonb_set(r.explain, '{candidates}', coalesce(
                       (SELECT jsonb_agg(c ORDER BY ord)
                          FROM jsonb_array_elements(r.explain->'candidates') WITH ORDINALITY AS t(c, ord)
                         WHERE c->>'id' <> %(fid)s::text), '[]'::jsonb))
                   ELSE r.explain END
         WHERE r.scope <@ %(scope)s::ltree
           AND (r.fact_ids @> ARRAY[%(id)s::uuid]
                OR r.explain->'candidates' @> jsonb_build_array(jsonb_build_object('id', %(fid)s::text)))
        """,
        {"id": fact_id, "fid": fid, "scope": fact["scope"]},
    )
    cur = await conn.execute("DELETE FROM memx_fact WHERE id = %s RETURNING id", (fact_id,))
    removed = len(await cur.fetchall())
    await prune_entities(conn)
    return removed


async def log_forget(
    conn: AsyncConnection, *, scope: str | None, fact_id: UUID | None, descendants: bool, facts: int, ingests: int,
    requested_by: str | None,
) -> None:
    await conn.execute(
        """
        INSERT INTO memx_forget_log (scope, fact_id, descendants, facts_removed, ingests_removed, requested_by)
        VALUES (%s::ltree, %s, %s, %s, %s, %s)
        """,
        (scope, fact_id, descendants, facts, ingests, requested_by),
    )
