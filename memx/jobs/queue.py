"""Postgres-backed job queue: SKIP LOCKED dequeue, leases, exponential backoff, dead-lettering."""

from __future__ import annotations

import random
from datetime import datetime
from typing import Any

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

CHANNEL = "memx_jobs"


async def enqueue(conn: AsyncConnection, kind: str, payload: dict[str, Any], key: str, *,
                  max_attempts: int = 5, lease_seconds: int = 300, status: str = "queued",
                  run_after: datetime | None = None) -> int | None:
    """Insert a job unless one with the same key exists. Returns the new id, or None for a duplicate."""
    cur = await conn.execute(
        """
        INSERT INTO memx_job (kind, payload, idempotency_key, max_attempts, lease_seconds, status, run_after,
                              locked_at, attempts)
        VALUES (%s, %s, %s, %s, %s, %s, coalesce(%s::timestamptz, now()),
                CASE WHEN %s::text = 'running' THEN now() END, CASE WHEN %s::text = 'running' THEN 1 ELSE 0 END)
        ON CONFLICT (idempotency_key) DO NOTHING
        RETURNING id
        """,
        (kind, Jsonb(payload), key, max_attempts, lease_seconds, status, run_after, status, status),
    )
    row = await cur.fetchone()
    if row and status == "queued":
        await conn.execute(f"NOTIFY {CHANNEL}")
    return row["id"] if row else None


async def claim(conn: AsyncConnection, worker: str) -> dict[str, Any] | None:
    cur = await conn.execute(
        """
        UPDATE memx_job
           SET status = 'running', locked_at = now(), locked_by = %s, attempts = attempts + 1
         WHERE id = (SELECT id FROM memx_job
                      WHERE status = 'queued' AND run_after <= now()
                      ORDER BY run_after, id
                      FOR UPDATE SKIP LOCKED
                      LIMIT 1)
        RETURNING *
        """,
        (worker,),
    )
    return await cur.fetchone()


async def complete(conn: AsyncConnection, job_id: int) -> None:
    await conn.execute(
        "UPDATE memx_job SET status = 'done', finished_at = now(), last_error = NULL WHERE id = %s", (job_id,)
    )


async def fail(conn: AsyncConnection, job: dict[str, Any], error: str, *, retryable: bool) -> bool:
    """Record a failure. Returns True if the job is now dead (no more retries)."""
    dead = not retryable or job["attempts"] >= job["max_attempts"]
    if dead:
        await conn.execute(
            "UPDATE memx_job SET status = 'dead', finished_at = now(), last_error = %s WHERE id = %s",
            (error[:2000], job["id"]),
        )
    else:
        delay = 10 * 2 ** (job["attempts"] - 1) + random.uniform(0, 5)
        await conn.execute(
            """
            UPDATE memx_job SET status = 'queued', locked_at = NULL, locked_by = NULL, last_error = %s,
                   run_after = now() + make_interval(secs => %s)
             WHERE id = %s
            """,
            (error[:2000], delay, job["id"]),
        )
    return dead


async def reap(conn: AsyncConnection) -> int:
    """Requeue running jobs whose lease expired (worker crashed or was killed)."""
    cur = await conn.execute(
        """
        UPDATE memx_job SET status = 'queued', locked_at = NULL, locked_by = NULL, run_after = now(),
               last_error = coalesce(last_error, 'lease expired')
         WHERE status = 'running' AND locked_at + make_interval(secs => lease_seconds) < now()
        RETURNING id
        """
    )
    return len(await cur.fetchall())


async def stats(conn: AsyncConnection) -> dict[str, int]:
    cur = await conn.execute("SELECT status, count(*) AS n FROM memx_job GROUP BY status")
    return {r["status"]: int(r["n"]) for r in await cur.fetchall()}
