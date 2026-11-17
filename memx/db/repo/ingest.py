"""Ingests: the raw inputs, their saved windows and candidates."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb




async def create_ingest(
    conn: AsyncConnection, *, scope: str, kind: str, payload: dict, source: str | None, trust: float,
    observed_at: datetime, idempotency_key: str, mode: str,
) -> tuple[UUID, bool]:
    """Returns (ingest_id, created). An existing idempotency key returns the original ingest."""
    cur = await conn.execute(
        """
        INSERT INTO memx_ingest (scope, kind, payload, source, trust, observed_at, idempotency_key, mode)
        VALUES (%s::ltree, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (scope, idempotency_key) DO NOTHING
        RETURNING id
        """,
        (scope, kind, Jsonb(payload), source, trust, observed_at, idempotency_key, mode),
    )
    row = await cur.fetchone()
    if row:
        return row["id"], True
    cur = await conn.execute(
        "SELECT id FROM memx_ingest WHERE scope = %s::ltree AND idempotency_key = %s", (scope, idempotency_key)
    )
    return (await cur.fetchone())["id"], False


async def reopen_failed_ingest(conn: AsyncConnection, ingest_id: UUID, *, inline: bool) -> bool:
    """Put a failed or abandoned ingest and its job back in play. Saved windows and candidate outcomes are kept.

    An ingest is abandoned when it never reached ``committed`` and nothing is working on it any more: its job is
    dead or done, or was left ``running`` past its lease by a process that died mid-ingest (a benchmark run lost a
    whole session this way: the ingest sat in ``received`` with zero facts and every resend returned its receipt).
    A queued job, or a running one inside its lease, is left alone.
    """
    cur = await conn.execute(
        """
        UPDATE memx_ingest i
           SET status = CASE WHEN candidates IS NULL THEN 'received' ELSE 'extracted' END, error = NULL
         WHERE i.id = %s
           AND (i.status = 'failed'
                OR (i.status IN ('received', 'extracted') AND NOT EXISTS (
                        SELECT 1 FROM memx_job j
                         WHERE j.idempotency_key = 'ingest:' || i.id::text
                           AND (j.status = 'queued'
                                OR (j.status = 'running'
                                    AND j.locked_at + make_interval(secs => j.lease_seconds) > now())))))
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


async def set_extract_batch(conn: AsyncConnection, ingest_id: UUID, state: dict | None) -> None:
    await conn.execute("UPDATE memx_ingest SET extract_batch = %s WHERE id = %s",
                       (Jsonb(state) if state is not None else None, ingest_id))


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
