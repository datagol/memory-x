"""All SQL lives here. Functions take an open connection (dict rows) and never commit on their own."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from uuid import UUID

from pgvector import Vector
from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from memx.types import EntityRef, Fact

FACT_COLS = """
  f.id, f.scope::text AS scope, f.category, f.text, f.status, f.superseded_by, f.superseded_at,
  f.speaker, f.source, f.trust, f.sort_confidence, f.observed_at, f.event_at, f.stale_after,
  f.reinforced_count, f.last_reinforced_at, f.access_count, f.last_accessed_at, f.flags,
  f.ingest_id, f.created_at, f.token_count, f.topic, f.topic_confidence
"""

# Facts recall may return. %(as_of)s lets callers evaluate expiry at a past moment (replays, evaluation).
VISIBLE = """
  f.status = 'active'
  AND (f.stale_after IS NULL OR f.stale_after > coalesce(%(as_of)s::timestamptz, now()))
  AND NOT (f.flags ? 'injection_suspect')
"""


def to_fact(row: dict[str, Any]) -> Fact:
    return Fact.model_validate({k: row[k] for k in Fact.model_fields if k in row})


def or_tsquery(text: str, limit: int = 32) -> str | None:
    """Build an OR full-text query so any shared term can surface a fact."""
    words = re.findall(r"[A-Za-z0-9]{2,}", text)[:limit]
    return " | ".join(dict.fromkeys(w.lower() for w in words)) or None


async def ensure_scope(conn: AsyncConnection, scope: str) -> None:
    await conn.execute("INSERT INTO memx_scope (path) VALUES (%s::ltree) ON CONFLICT DO NOTHING", (scope,))


async def scope_settings(conn: AsyncConnection, scope: str) -> dict[str, Any]:
    cur = await conn.execute("SELECT settings FROM memx_scope WHERE path = %s::ltree", (scope,))
    row = await cur.fetchone()
    return row["settings"] if row else {}


# ---- Ingest -----------------------------------------------------------------


async def create_ingest(
    conn: AsyncConnection, *, scope: str, kind: str, payload: dict, source: str | None, trust: float,
    observed_at: datetime, idempotency_key: str, mode: str,
) -> tuple[UUID, bool]:
    """Returns (ingest_id, created). An existing idempotency key returns the original ingest."""
    cur = await conn.execute(
        """
        INSERT INTO memx_ingest (scope, kind, payload, source, trust, observed_at, idempotency_key, mode)
        VALUES (%s::ltree, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (idempotency_key) DO NOTHING
        RETURNING id
        """,
        (scope, kind, Jsonb(payload), source, trust, observed_at, idempotency_key, mode),
    )
    row = await cur.fetchone()
    if row:
        return row["id"], True
    cur = await conn.execute("SELECT id FROM memx_ingest WHERE idempotency_key = %s", (idempotency_key,))
    return (await cur.fetchone())["id"], False


async def reopen_failed_ingest(conn: AsyncConnection, ingest_id: UUID, *, inline: bool) -> bool:
    """Put a failed ingest and its job back in play. Saved windows and candidate outcomes are kept."""
    cur = await conn.execute(
        """
        UPDATE memx_ingest
           SET status = CASE WHEN candidates IS NULL THEN 'received' ELSE 'extracted' END, error = NULL
         WHERE id = %s AND status = 'failed'
        RETURNING id
        """,
        (ingest_id,),
    )
    if not await cur.fetchone():
        return False
    await conn.execute(
        """
        UPDATE memx_job
           SET status = %s, attempts = CASE WHEN %s THEN 1 ELSE 0 END, run_after = now(),
               locked_at = CASE WHEN %s THEN now() END, locked_by = NULL, last_error = NULL, finished_at = NULL
         WHERE idempotency_key = %s
        """,
        ("running" if inline else "queued", inline, inline, f"ingest:{ingest_id}"),
    )
    return True


async def get_ingest(conn: AsyncConnection, ingest_id: UUID) -> dict[str, Any] | None:
    cur = await conn.execute(
        "SELECT *, scope::text AS scope FROM memx_ingest WHERE id = %s", (ingest_id,)
    )
    return await cur.fetchone()


async def save_window(conn: AsyncConnection, ingest_id: UUID, index: int, facts: list[dict], tin: int, tout: int, model: str) -> None:
    await conn.execute(
        """
        UPDATE memx_ingest
           SET windows = jsonb_set(windows, ARRAY[%s::text], %s),
               tokens_in = tokens_in + %s, tokens_out = tokens_out + %s, extractor_model = %s
         WHERE id = %s
        """,
        (str(index), Jsonb(facts), tin, tout, model, ingest_id),
    )


async def save_candidates(conn: AsyncConnection, ingest_id: UUID, candidates: list[dict]) -> None:
    await conn.execute(
        "UPDATE memx_ingest SET candidates = %s, status = 'extracted' WHERE id = %s",
        (Jsonb(candidates), ingest_id),
    )


async def set_candidate_outcome(conn: AsyncConnection, ingest_id: UUID, position: int, candidate: dict) -> None:
    await conn.execute(
        "UPDATE memx_ingest SET candidates = jsonb_set(candidates, ARRAY[%s::text], %s) WHERE id = %s",
        (str(position), Jsonb(candidate), ingest_id),
    )


async def finish_ingest(conn: AsyncConnection, ingest_id: UUID, receipt: dict) -> None:
    await conn.execute(
        """
        UPDATE memx_ingest SET status = 'committed', receipt = %s, committed_at = now(), error = NULL
         WHERE id = %s
        """,
        (Jsonb(receipt), ingest_id),
    )


async def fail_ingest(conn: AsyncConnection, ingest_id: UUID, error: str, final: bool) -> None:
    if final:
        await conn.execute("UPDATE memx_ingest SET status = 'failed', error = %s WHERE id = %s", (error, ingest_id))
    else:
        await conn.execute("UPDATE memx_ingest SET error = %s WHERE id = %s", (error, ingest_id))


# ---- Facts ------------------------------------------------------------------


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
        (SELECT {FACT_COLS}, 1 - (f.embedding <=> %(e)s::vector) AS sim, 'fts' AS via
           FROM memx_fact f
          WHERE {where} AND f.status = 'active' {extra}
            AND f.tsv @@ to_tsquery('english', %(q)s)
          ORDER BY ts_rank(f.tsv, to_tsquery('english', %(q)s)) DESC
          LIMIT %(kf)s)
        """
    sql = f"""
        (SELECT {FACT_COLS}, 1 - (f.embedding <=> %(e)s::vector) AS sim, 'vec' AS via
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


async def reinforce(conn: AsyncConnection, fact_id: UUID) -> None:
    await conn.execute(
        """
        UPDATE memx_fact SET reinforced_count = reinforced_count + 1, last_reinforced_at = now()
         WHERE id = %s
        """,
        (fact_id,),
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


# ---- Entities ---------------------------------------------------------------


async def find_entity(conn: AsyncConnection, tenant: str, norm: str) -> dict[str, Any] | None:
    cur = await conn.execute(
        "SELECT id, name, norm, type FROM memx_entity WHERE tenant = %s AND (norm = %s OR %s = ANY(aliases)) LIMIT 1",
        (tenant, norm, norm),
    )
    return await cur.fetchone()


async def find_entity_partial(conn: AsyncConnection, tenant: str, words: list[str], limit: int = 5) -> list[dict[str, Any]]:
    """Entities in the tenant sharing at least one word with the mention, longest shared prefix first."""
    words = [w for w in words if len(w) >= 3]
    if not words:
        return []
    cur = await conn.execute(
        """
        SELECT id, name, norm, type FROM memx_entity
         WHERE tenant = %s AND string_to_array(norm, ' ') && %s::text[]
         ORDER BY length(norm) DESC LIMIT %s
        """,
        (tenant, words, limit),
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
        "SELECT id, name, norm, type, aliases FROM memx_entity WHERE tenant = %s ORDER BY length(norm), name", (tenant,))
    return await cur.fetchall()


async def alias_entity(conn: AsyncConnection, entity_id: UUID, name: str, norm: str) -> None:
    """Record a new spelling. A longer spelling becomes the entity's name and key; the old key turns into an alias."""
    await conn.execute(
        """
        UPDATE memx_entity SET
          name = CASE WHEN length(%(norm)s) > length(norm) THEN %(name)s ELSE name END,
          aliases = CASE WHEN length(%(norm)s) > length(norm) THEN array_append(aliases, norm)
                         WHEN %(norm)s = ANY(aliases) THEN aliases ELSE array_append(aliases, %(norm)s) END,
          norm = CASE WHEN length(%(norm)s) > length(norm) THEN %(norm)s ELSE norm END
         WHERE id = %(id)s
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
        UPDATE memx_entity t SET aliases = (
          SELECT array_agg(DISTINCT a) FROM unnest(t.aliases || f.aliases || ARRAY[f.norm]) a)
          FROM memx_entity f WHERE t.id = %(into)s AND f.id = %(from)s
        """,
        {"from": from_id, "into": into_id},
    )
    await conn.execute("DELETE FROM memx_entity WHERE id = %s", (from_id,))


async def upsert_entity(conn: AsyncConnection, tenant: str, name: str, norm: str, etype: str) -> UUID:
    cur = await conn.execute(
        """
        INSERT INTO memx_entity (tenant, name, norm, type) VALUES (%s, %s, %s, %s)
        ON CONFLICT (tenant, norm) DO UPDATE SET type = CASE WHEN memx_entity.type = 'other' THEN excluded.type
                                                             ELSE memx_entity.type END
        RETURNING id
        """,
        (tenant, name, norm, etype),
    )
    return (await cur.fetchone())["id"]


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


# ---- Recall -----------------------------------------------------------------


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
    if ids:
        await conn.execute(
            "UPDATE memx_fact SET access_count = access_count + 1, last_accessed_at = now() WHERE id = ANY(%s)",
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


# ---- Listing, history, forget ------------------------------------------------


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


async def links_for(conn: AsyncConnection, ids: list[UUID]) -> list[dict[str, Any]]:
    if not ids:
        return []
    cur = await conn.execute(
        "SELECT a, b, kind, similarity, created_at FROM memx_fact_link WHERE a = ANY(%s) OR b = ANY(%s)",
        (ids, ids),
    )
    return await cur.fetchall()


async def version_chain(conn: AsyncConnection, fact_id: UUID) -> list[UUID]:
    """Every fact connected to this one through superseded_by, oldest first."""
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
         ORDER BY f.observed_at, f.created_at
        """,
        {"id": fact_id},
    )
    return [r["id"] for r in await cur.fetchall()]


async def decisions_for(conn: AsyncConnection, ids: list[UUID]) -> list[dict[str, Any]]:
    if not ids:
        return []
    cur = await conn.execute(
        """
        SELECT id, step, question_version, subject_fact_id, object_fact_id, similarity, result,
               latency_ms, created_at
          FROM memx_decision
         WHERE subject_fact_id = ANY(%s) OR object_fact_id = ANY(%s)
         ORDER BY created_at, id
        """,
        (ids, ids),
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


async def forget_fact(conn: AsyncConnection, fact_id: UUID) -> int:
    await conn.execute(
        "DELETE FROM memx_decision WHERE subject_fact_id = %s OR object_fact_id = %s", (fact_id, fact_id)
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


# ---- Sweep ------------------------------------------------------------------


async def sweep(conn: AsyncConnection, *, expired_retention_days: int, decision_retention_days: int,
                done_job_retention_days: int) -> dict[str, int]:
    cur = await conn.execute(
        "UPDATE memx_fact SET status = 'expired' WHERE status = 'active' AND stale_after <= now() RETURNING id"
    )
    expired = len(await cur.fetchall())
    cur = await conn.execute(
        """
        DELETE FROM memx_fact WHERE status = 'expired'
           AND stale_after < now() - make_interval(days => %s) RETURNING id
        """,
        (expired_retention_days,),
    )
    purged = len(await cur.fetchall())
    cur = await conn.execute(
        "DELETE FROM memx_decision WHERE created_at < now() - make_interval(days => %s) RETURNING id",
        (decision_retention_days,),
    )
    decisions = len(await cur.fetchall())
    cur = await conn.execute(
        """
        DELETE FROM memx_job WHERE status = 'done'
           AND finished_at < now() - make_interval(days => %s) RETURNING id
        """,
        (done_job_retention_days,),
    )
    jobs = len(await cur.fetchall())
    await prune_entities(conn)
    return {"expired": expired, "purged": purged, "decisions_pruned": decisions, "jobs_pruned": jobs}


# ---- Dashboard reads ----------------------------------------------------------

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
