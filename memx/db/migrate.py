from __future__ import annotations

from importlib import resources

import psycopg

_TABLE = """
CREATE TABLE IF NOT EXISTS memx_schema_version (
  version     int PRIMARY KEY,
  name        text NOT NULL,
  applied_at  timestamptz NOT NULL DEFAULT now()
)
"""


def _migration_files() -> list[tuple[int, str, str]]:
    root = resources.files("memx.db") / "migrations"
    out = []
    for entry in sorted(root.iterdir(), key=lambda e: e.name):
        if entry.name.endswith(".sql"):
            out.append((int(entry.name.split("_", 1)[0]), entry.name, entry.read_text()))
    return out


async def migrate(dsn: str, embedding_dim: int) -> list[str]:
    """Apply pending migrations. Safe to run repeatedly and from several pods at once."""
    applied: list[str] = []
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=False) as conn:
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext('memx_migrate'))")
            await conn.execute(_TABLE)
            cur = await conn.execute("SELECT version FROM memx_schema_version")
            done = {row[0] for row in await cur.fetchall()}
            for version, name, sql in _migration_files():
                if version in done:
                    continue
                await conn.execute(sql.replace("{{EMBED_DIM}}", str(int(embedding_dim))))
                await conn.execute(
                    "INSERT INTO memx_schema_version (version, name) VALUES (%s, %s)", (version, name)
                )
                applied.append(name)
    return applied
