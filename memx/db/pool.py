from __future__ import annotations

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
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
