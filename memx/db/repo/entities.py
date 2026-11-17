"""Entities: the people, teams and systems facts name."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pgvector import Vector
from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from memx.types import EntityRef, Fact

from memx.db.repo._base import FACT_COLS, VISIBLE


ENTITY_COLS = "id, name, norm, type, aliases, known_as"


async def find_entities_exact(conn: AsyncConnection, tenant: str, norm: str) -> list[dict[str, Any]]:
    """Entities with exactly this name, including namesakes ("priya", "priya~2") and merged spellings."""
    cur = await conn.execute(
        f"""
        SELECT {ENTITY_COLS} FROM memx_entity
         WHERE tenant = %(t)s AND (split_part(norm, '~', 1) = %(n)s OR %(n)s = ANY(aliases))
         ORDER BY norm
        """,
        {"t": tenant, "n": norm},
    )
    return await cur.fetchall()


async def find_entity_partial(conn: AsyncConnection, tenant: str, words: list[str], limit: int = 5) -> list[dict[str, Any]]:
    """Entities in the tenant sharing at least one word with the mention, longest first."""
    words = [w for w in words if len(w) >= 3]
    if not words:
        return []
    cur = await conn.execute(
        f"""
        SELECT {ENTITY_COLS} FROM memx_entity
         WHERE tenant = %s AND string_to_array(split_part(norm, '~', 1), ' ') && %s::text[]
         ORDER BY length(norm) DESC LIMIT %s
        """,
        (tenant, words, limit),
    )
    return await cur.fetchall()


async def find_entities_known_as(conn: AsyncConnection, tenant: str, norm: str, akas: list[str]) -> list[dict[str, Any]]:
    """Entities linked to this name through suggested alternative names, in either direction: a stored "Blore"
    whose alternatives include "bangalore", or a new "Bangalore" whose alternatives include a stored name."""
    cur = await conn.execute(
        f"""
        SELECT {ENTITY_COLS} FROM memx_entity
         WHERE tenant = %(t)s AND (%(n)s = ANY(known_as) OR split_part(norm, '~', 1) = ANY(%(a)s::text[])
                                   OR aliases && %(a)s::text[])
        """,
        {"t": tenant, "n": norm, "a": akas},
    )
    return await cur.fetchall()


async def find_entities_similar(conn: AsyncConnection, tenant: str, vector: list[float], min_sim: float,
                                limit: int = 3) -> list[dict[str, Any]]:
    """Known entities whose name embedding is close to this one. Candidates for Jev only, never merged on this."""
    cur = await conn.execute(
        f"""
        SELECT * FROM (
          SELECT {ENTITY_COLS}, 1 - (embedding <=> %(v)s::vector) AS sim FROM memx_entity
           WHERE tenant = %(t)s AND embedding IS NOT NULL ORDER BY embedding <=> %(v)s::vector LIMIT %(k)s
        ) x WHERE sim >= %(m)s
        """,
        {"t": tenant, "v": Vector(vector), "k": limit, "m": min_sim},
    )
    return await cur.fetchall()


async def find_entities_similar_to(conn: AsyncConnection, tenant: str, entity_id: UUID, min_sim: float,
                                   limit: int = 3) -> list[dict[str, Any]]:
    cur = await conn.execute(
        f"""
        SELECT * FROM (
          SELECT {ENTITY_COLS}, 1 - (e.embedding <=> s.embedding) AS sim
            FROM memx_entity e, (SELECT embedding FROM memx_entity WHERE id = %(id)s) s
           WHERE e.tenant = %(t)s AND e.id <> %(id)s AND e.embedding IS NOT NULL AND s.embedding IS NOT NULL
           ORDER BY e.embedding <=> s.embedding LIMIT %(k)s
        ) x WHERE sim >= %(m)s
        """,
        {"t": tenant, "id": entity_id, "k": limit, "m": min_sim},
    )
    return await cur.fetchall()


async def entity_sample_facts(conn: AsyncConnection, entity_id: UUID, limit: int = 3) -> list[str]:
    """A few facts about an entity for the merge question: active first, shortest first (they read as definitions,
    "Priya Lindqvist leads the Security Team"), then newest."""
    cur = await conn.execute(
        """
        SELECT f.text FROM memx_fact_entity fe JOIN memx_fact f ON f.id = fe.fact_id
         WHERE fe.entity_id = %s ORDER BY (f.status = 'active') DESC, length(f.text), f.created_at DESC LIMIT %s
        """,
        (entity_id, limit),
    )
    return [r["text"] for r in await cur.fetchall()]


async def tenant_entities(conn: AsyncConnection, tenant: str) -> list[dict[str, Any]]:
    cur = await conn.execute(
        f"SELECT {ENTITY_COLS} FROM memx_entity WHERE tenant = %s ORDER BY length(norm), name", (tenant,))
    return await cur.fetchall()


async def entities_without_embedding(conn: AsyncConnection, tenant: str) -> list[dict[str, Any]]:
    cur = await conn.execute("SELECT id, name FROM memx_entity WHERE tenant = %s AND embedding IS NULL", (tenant,))
    return await cur.fetchall()


async def set_entity_embedding(conn: AsyncConnection, entity_id: UUID, vector: list[float]) -> None:
    await conn.execute("UPDATE memx_entity SET embedding = %s::vector WHERE id = %s", (Vector(vector), entity_id))


async def entities_in_ingest(conn: AsyncConnection, ids: list[UUID], ingest_id: UUID | None) -> set[UUID]:
    """Which of these entities are already named by a fact from this same insert."""
    if not ids or ingest_id is None:
        return set()
    cur = await conn.execute(
        "SELECT DISTINCT fe.entity_id FROM memx_fact_entity fe JOIN memx_fact f ON f.id = fe.fact_id "
        "WHERE fe.entity_id = ANY(%s) AND f.ingest_id = %s", (ids, ingest_id))
    return {r["entity_id"] for r in await cur.fetchall()}


async def entity_fact_counts(conn: AsyncConnection, ids: list[UUID]) -> dict[UUID, int]:
    cur = await conn.execute(
        "SELECT entity_id, count(*) AS n FROM memx_fact_entity WHERE entity_id = ANY(%s) GROUP BY 1", (ids,))
    return {r["entity_id"]: int(r["n"]) for r in await cur.fetchall()}


async def add_known_as(conn: AsyncConnection, entity_id: UUID, names: list[str]) -> None:
    if names:
        await conn.execute(
            "UPDATE memx_entity SET known_as = ARRAY(SELECT DISTINCT unnest(known_as || %s::text[])) WHERE id = %s",
            (names, entity_id),
        )


async def alias_entity(conn: AsyncConnection, entity_id: UUID, name: str, norm: str) -> None:
    """Record a new spelling. A longer spelling becomes the entity's name and key; the old key turns into an alias."""
    await conn.execute(
        """
        UPDATE memx_entity e SET
          name = CASE WHEN x.rename THEN %(name)s ELSE e.name END,
          aliases = CASE WHEN x.rename THEN array_append(e.aliases, e.norm)
                         WHEN %(norm)s = ANY(e.aliases) THEN e.aliases ELSE array_append(e.aliases, %(norm)s) END,
          norm = CASE WHEN x.rename THEN %(norm)s ELSE e.norm END
          FROM (SELECT length(%(norm)s) > length(m.norm)
                       -- the key can only move to the longer spelling while no other entity holds it
                       AND NOT EXISTS (SELECT 1 FROM memx_entity o
                                        WHERE o.tenant = m.tenant AND o.norm = %(norm)s AND o.id <> m.id) AS rename
                  FROM memx_entity m WHERE m.id = %(id)s) x
         WHERE e.id = %(id)s
        """,
        {"id": entity_id, "name": name, "norm": norm},
    )


async def merge_entity(conn: AsyncConnection, from_id: UUID, into_id: UUID) -> None:
    """Fold one entity into another: move its mentions, keep its spellings as aliases, delete it."""
    await conn.execute(
        """
        INSERT INTO memx_fact_entity (fact_id, entity_id, surface)
        SELECT fact_id, %(into)s, surface FROM memx_fact_entity WHERE entity_id = %(from)s
        ON CONFLICT DO NOTHING
        """,
        {"from": from_id, "into": into_id},
    )
    await conn.execute(
        """
        UPDATE memx_entity t SET
          aliases = (SELECT array_agg(DISTINCT a) FROM unnest(t.aliases || f.aliases || ARRAY[split_part(f.norm, '~', 1)]) a),
          known_as = ARRAY(SELECT DISTINCT unnest(t.known_as || f.known_as)),
          embedding = coalesce(t.embedding, f.embedding)
          FROM memx_entity f WHERE t.id = %(into)s AND f.id = %(from)s
        """,
        {"from": from_id, "into": into_id},
    )
    await conn.execute("DELETE FROM memx_entity WHERE id = %s", (from_id,))


async def upsert_entity(conn: AsyncConnection, tenant: str, name: str, norm: str, etype: str,
                        embedding: list[float] | None = None, known_as: list[str] | None = None) -> UUID:
    cur = await conn.execute(
        """
        INSERT INTO memx_entity (tenant, name, norm, type, embedding, known_as)
        VALUES (%s, %s, %s, %s, %s::vector, %s::text[])
        ON CONFLICT (tenant, norm) DO UPDATE SET
          type = CASE WHEN memx_entity.type = 'other' THEN excluded.type ELSE memx_entity.type END,
          embedding = coalesce(memx_entity.embedding, excluded.embedding),
          known_as = ARRAY(SELECT DISTINCT unnest(memx_entity.known_as || excluded.known_as))
        RETURNING id
        """,
        (tenant, name, norm, etype, Vector(embedding) if embedding else None, known_as or []),
    )
    return (await cur.fetchone())["id"]


async def create_entity_variant(conn: AsyncConnection, tenant: str, name: str, norm: str, etype: str,
                                embedding: list[float] | None = None, known_as: list[str] | None = None) -> UUID:
    """A second entity with a name that is already taken by someone else ("Priya" from the bakery). Stored under a
    numbered key, "priya~2", so the name stays the same and the key stays unique."""
    cur = await conn.execute(
        "SELECT norm FROM memx_entity WHERE tenant = %s AND split_part(norm, '~', 1) = %s", (tenant, norm))
    taken = {r["norm"] for r in await cur.fetchall()}
    if norm not in taken:
        return await upsert_entity(conn, tenant, name, norm, etype, embedding, known_as)
    # The next unused number: counting rows would reuse "priya~3" after "priya~2" was merged away.
    n = 2
    while f"{norm}~{n}" in taken:
        n += 1
    return await upsert_entity(conn, tenant, name, f"{norm}~{n}", etype, embedding, known_as)


async def link_entity(conn: AsyncConnection, fact_id: UUID, entity_id: UUID, surface: str | None) -> None:
    await conn.execute(
        "INSERT INTO memx_fact_entity (fact_id, entity_id, surface) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
        (fact_id, entity_id, surface),
    )


async def entities_for_facts(conn: AsyncConnection, ids: list[UUID]) -> dict[UUID, list[dict[str, Any]]]:
    out: dict[UUID, list[dict[str, Any]]] = {i: [] for i in ids}
    if not ids:
        return out
    cur = await conn.execute(
        """
        SELECT fe.fact_id, e.id, e.name, e.type, fe.surface FROM memx_fact_entity fe
          JOIN memx_entity e ON e.id = fe.entity_id WHERE fe.fact_id = ANY(%s) ORDER BY e.name
        """,
        (ids,),
    )
    for r in await cur.fetchall():
        out.setdefault(r["fact_id"], []).append({"id": r["id"], "name": r["name"], "type": r["type"],
                                                 "surface": r["surface"]})
    return out


async def attach_entities(conn: AsyncConnection, facts: list[Fact]) -> list[Fact]:
    found = await entities_for_facts(conn, [f.id for f in facts])
    for f in facts:
        f.entities = [EntityRef(**e) for e in found.get(f.id, [])]
    return facts


async def entity_edges(conn: AsyncConnection, chain: list[str], fact_ids: list[UUID],
                       as_of: datetime | None) -> list[dict[str, Any]]:
    """For the walk: (fact, entity) pairs for these facts, with each entity's degree over visible chain facts."""
    if not fact_ids:
        return []
    cur = await conn.execute(
        f"""
        WITH touched AS (
          SELECT DISTINCT fe.entity_id FROM memx_fact_entity fe WHERE fe.fact_id = ANY(%(ids)s)
        ), degree AS (
          SELECT fe.entity_id, count(*) AS n FROM memx_fact_entity fe JOIN memx_fact f ON f.id = fe.fact_id
           WHERE fe.entity_id IN (SELECT entity_id FROM touched) AND f.scope = ANY(%(chain)s::ltree[]) AND {VISIBLE}
           GROUP BY fe.entity_id
        )
        SELECT fe.fact_id, e.id AS entity_id, e.name, e.type, coalesce(d.n, 0) AS degree
          FROM memx_fact_entity fe JOIN memx_entity e ON e.id = fe.entity_id
          LEFT JOIN degree d ON d.entity_id = fe.entity_id
         WHERE fe.fact_id = ANY(%(ids)s)
        """,
        {"ids": fact_ids, "chain": chain, "as_of": as_of},
    )
    return await cur.fetchall()


async def facts_for_entities(conn: AsyncConnection, chain: list[str], entity_ids: list[UUID],
                             as_of: datetime | None) -> list[dict[str, Any]]:
    """Visible chain facts mentioning any of these entities, one row per (entity, fact)."""
    if not entity_ids:
        return []
    cur = await conn.execute(
        f"""
        SELECT fe.entity_id, {FACT_COLS} FROM memx_fact_entity fe JOIN memx_fact f ON f.id = fe.fact_id
         WHERE fe.entity_id = ANY(%(ids)s) AND f.scope = ANY(%(chain)s::ltree[]) AND {VISIBLE}
        """,
        {"ids": entity_ids, "chain": chain, "as_of": as_of},
    )
    return await cur.fetchall()


async def topic_degrees(conn: AsyncConnection, chain: list[str], topics: list[str],
                        as_of: datetime | None) -> dict[str, int]:
    if not topics:
        return {}
    cur = await conn.execute(
        f"""
        SELECT f.topic, count(*) AS n FROM memx_fact f
         WHERE f.topic = ANY(%(topics)s) AND f.scope = ANY(%(chain)s::ltree[]) AND {VISIBLE} GROUP BY f.topic
        """,
        {"topics": topics, "chain": chain, "as_of": as_of},
    )
    return {r["topic"]: int(r["n"]) for r in await cur.fetchall()}


async def facts_for_topics(conn: AsyncConnection, chain: list[str], topics: list[str],
                           as_of: datetime | None) -> list[dict[str, Any]]:
    if not topics:
        return []
    cur = await conn.execute(
        f"""
        SELECT {FACT_COLS} FROM memx_fact f
         WHERE f.topic = ANY(%(topics)s) AND f.scope = ANY(%(chain)s::ltree[]) AND {VISIBLE}
        """,
        {"topics": topics, "chain": chain, "as_of": as_of},
    )
    return await cur.fetchall()


async def prune_entities(conn: AsyncConnection) -> int:
    cur = await conn.execute(
        "DELETE FROM memx_entity e WHERE NOT EXISTS (SELECT 1 FROM memx_fact_entity fe WHERE fe.entity_id = e.id) "
        "RETURNING id"
    )
    return len(await cur.fetchall())


async def list_entities(conn: AsyncConnection, scope: str, *, view: str, limit: int) -> list[dict[str, Any]]:
    """Entities mentioned by facts around a scope, with how many of those facts mention each."""
    if view == "chain":
        labels = scope.split(".")
        where = "f.scope = ANY(%(scopes)s::ltree[])"
        params: dict[str, Any] = {"scopes": [".".join(labels[: i + 1]) for i in range(len(labels))]}
    else:
        where = "f.scope <@ %(scope)s::ltree"
        params = {"scope": scope}
    params["limit"] = limit
    cur = await conn.execute(
        f"""
        SELECT e.id, e.name, e.type, e.aliases, count(*) FILTER (WHERE f.status = 'active') AS active,
               count(*) AS facts
          FROM memx_entity e JOIN memx_fact_entity fe ON fe.entity_id = e.id JOIN memx_fact f ON f.id = fe.fact_id
         WHERE {where}
         GROUP BY e.id ORDER BY active DESC, facts DESC, e.name LIMIT %(limit)s
        """,
        params,
    )
    return await cur.fetchall()


async def mentions_for(conn: AsyncConnection, ids: list[UUID]) -> list[dict[str, Any]]:
    if not ids:
        return []
    cur = await conn.execute(
        """
        SELECT fe.fact_id, fe.entity_id, e.name, e.type, e.aliases FROM memx_fact_entity fe
          JOIN memx_entity e ON e.id = fe.entity_id WHERE fe.fact_id = ANY(%s)
        """,
        (ids,),
    )
    return await cur.fetchall()


async def add_decisions(conn: AsyncConnection, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        await conn.execute(
            """
            INSERT INTO memx_decision (step, question_version, ingest_id, candidate_index, subject_fact_id,
                                       object_fact_id, recall_id, scope, similarity, result, latency_ms)
            VALUES (%(step)s, %(question_version)s, %(ingest_id)s, %(candidate_index)s, %(subject_fact_id)s,
                    %(object_fact_id)s, %(recall_id)s, %(scope)s::ltree, %(similarity)s, %(result)s, %(latency_ms)s)
            """,
            {
                "ingest_id": None, "candidate_index": None, "subject_fact_id": None, "object_fact_id": None,
                "recall_id": None, "scope": None, "similarity": None, "latency_ms": None,
                **row, "result": Jsonb(row["result"]),
            },
        )
