from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg.pq import TransactionStatus
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


async def _configure(conn: AsyncConnection) -> None:
    await register_vector_async(conn)
    conn.row_factory = dict_row


async def open_pool(dsn: str, min_size: int = 1, max_size: int = 10) -> AsyncConnectionPool:
    pool = AsyncConnectionPool(
        dsn, min_size=min_size, max_size=max_size, configure=_configure, open=False,
        kwargs={"autocommit": True},
    )
    await pool.open(wait=True)
    return pool


class Borrowed:
    """A pooled connection that can be handed back while its holder waits on a model call.

    A recall or an ingest alternates short queries with model calls that take seconds. Holding one connection for
    the whole of it means a handful of concurrent calls use up the pool while doing nothing with it. This stands in
    for the connection (``execute``, ``transaction`` and the rest pass through) and ``released()`` returns it to the
    pool for the length of a ``with`` block, taking another afterwards. Inside a transaction nothing is released.
    """

    def __init__(self, pool: AsyncConnectionPool):
        self._pool = pool
        self._conn: AsyncConnection | None = None

    async def __aenter__(self) -> Borrowed:
        self._conn = await self._pool.getconn()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._conn is not None:
            conn, self._conn = self._conn, None
            await self._pool.putconn(conn)

    def __getattr__(self, name: str) -> Any:
        if self._conn is None:
            raise RuntimeError("the connection is released; database calls must wait until the model call returns")
        return getattr(self._conn, name)

    @asynccontextmanager
    async def released(self) -> AsyncIterator[None]:
        conn = self._conn
        if conn is None or conn.info.transaction_status != TransactionStatus.IDLE:
            yield  # already released by an outer block, or mid-transaction: keep what we have
            return
        self._conn = None
        await self._pool.putconn(conn)
        try:
            yield
        finally:
            self._conn = await self._pool.getconn()

    def judge(self, judge: Any) -> Any:
        """The same judge, releasing this connection for the length of each question."""
        return _ReleasingJudge(judge, self)


class _ReleasingJudge:
    def __init__(self, judge: Any, db: Borrowed):
        self._judge, self._db = judge, db

    def __getattr__(self, name: str) -> Any:
        return getattr(self._judge, name)

    async def ask(self, *args: Any, **kwargs: Any) -> Any:
        async with self._db.released():
            return await self._judge.ask(*args, **kwargs)
