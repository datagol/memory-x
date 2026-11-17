"""Shared pieces: column lists, row conversion, scope rows and locks."""

from __future__ import annotations

import re
from typing import Any

from psycopg import AsyncConnection

from memx.types import Fact



FACT_COLS = """
  f.id, f.scope::text AS scope, f.category, f.text, f.status, f.superseded_by, f.superseded_at,
  f.speaker, f.source, f.trust, f.sort_confidence, f.observed_at, f.event_at, f.stale_after,
  f.reinforced_count, f.last_reinforced_at, f.access_count, f.last_accessed_at, f.flags,
  f.ingest_id, f.created_at, f.token_count, f.topic, f.topic_confidence
"""

# Extra columns reconcile needs: the extractor's attribute label and the window a fact came from.
NEIGHBOR_COLS = "f.attrs->>'attribute' AS attribute, (f.attrs->>'window')::int AS extract_window"

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


async def lock_scope(conn: AsyncConnection, scope: str) -> None:
    """Serialize writers of one scope until the surrounding transaction ends."""
    await conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (scope,))


async def lock_entities(conn: AsyncConnection, tenant: str) -> None:
    """Serialize entity writes of one tenant. Entities are shared by every scope of a tenant, so the scope lock
    alone lets two sibling scopes create or rename the same name at once. Take it after the scope lock."""
    await conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"entities:{tenant}",))
