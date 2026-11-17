"""Maintenance: expiry, retention and the event-expiry reapply."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection


from memx.db.repo.entities import prune_entities


async def sweep(conn: AsyncConnection, *, expired_retention_days: int, decision_retention_days: int,
                done_job_retention_days: int, recall_log_retention_days: int = 0) -> dict[str, int]:
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
    # Old decisions go, except the verdicts behind a replacement or shadow still in effect: history() shows why a
    # fact was retired, and rereconcile re-reads exactly those verdicts when the policy changes.
    cur = await conn.execute(
        """
        DELETE FROM memx_decision d
         WHERE d.created_at < now() - make_interval(days => %s)
           AND NOT (d.step = 'verdict' AND (
                 EXISTS (SELECT 1 FROM memx_fact f
                          WHERE f.status = 'superseded'
                            AND ((f.id = d.object_fact_id AND f.superseded_by = d.subject_fact_id)
                              OR (f.id = d.subject_fact_id AND f.superseded_by = d.object_fact_id)))
              OR EXISTS (SELECT 1 FROM memx_fact_link l
                          WHERE l.kind = 'shadows' AND l.a = d.subject_fact_id AND l.b = d.object_fact_id)))
        RETURNING id
        """,
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
    recalls = 0
    if recall_log_retention_days > 0:
        cur = await conn.execute(
            "DELETE FROM memx_recall_log WHERE created_at < now() - make_interval(days => %s) RETURNING id",
            (recall_log_retention_days,),
        )
        recalls = len(await cur.fetchall())
    await prune_entities(conn)
    return {"expired": expired, "purged": purged, "decisions_pruned": decisions, "jobs_pruned": jobs,
            "recalls_pruned": recalls}


async def event_facts(conn: AsyncConnection, scope: str | None) -> list[dict[str, Any]]:
    """Dated event facts that are active or expired (not yet purged), for re-applying the expiry policy."""
    cur = await conn.execute(
        """
        SELECT id, scope::text AS scope, event_at, observed_at, stale_after, status FROM memx_fact
         WHERE category = 'event' AND event_at IS NOT NULL AND status IN ('active', 'expired')
           AND (%(scope)s::ltree IS NULL OR scope <@ %(scope)s::ltree)
        """,
        {"scope": scope},
    )
    return await cur.fetchall()


async def set_event_expiry(conn: AsyncConnection, fact_id: UUID, stale_after: datetime | None, revive: bool) -> bool:
    """Set a fact's expiry; ``revive`` also makes an expired fact active again unless that would duplicate an
    active fact with the same text in its scope. Returns False when the revive was skipped."""
    if revive:
        cur = await conn.execute(
            """
            UPDATE memx_fact f SET stale_after = %s, status = 'active'
             WHERE id = %s AND status = 'expired'
               AND NOT EXISTS (SELECT 1 FROM memx_fact o WHERE o.scope = f.scope AND o.text_hash = f.text_hash
                                                          AND o.status = 'active')
            RETURNING id
            """,
            (stale_after, fact_id),
        )
        return bool(await cur.fetchone())
    await conn.execute("UPDATE memx_fact SET stale_after = %s WHERE id = %s", (stale_after, fact_id))
    return True
