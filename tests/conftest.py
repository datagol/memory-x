from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from typing import Any

import psycopg
import pytest

from memx import Memory, MemxConfig
from memx.testing import FakeJudge, HashEmbedder, SentenceExtractor

if not os.environ.get("MEMX_LIVE"):
    # Tests must behave the same on every machine: do not pick up settings from the developer's .env. The opt-in
    # live tests are the exception, since that is where their API key lives.
    MemxConfig.model_config["env_file"] = None

ADMIN_DSN = os.environ.get("MEMX_TEST_ADMIN_DSN", "postgresql://memx:memx@localhost:5433/memx")
TEST_DB = "memx_test"
TABLES = ("memx_fact_entity, memx_entity, memx_fact_link, memx_fact, memx_ingest, memx_scope, memx_decision, "
          "memx_recall_log, memx_job, memx_forget_log")


def _test_dsn() -> str:
    base, _, _ = ADMIN_DSN.rpartition("/")
    return f"{base}/{TEST_DB}"


@pytest.fixture(scope="session")
async def test_dsn() -> AsyncIterator[str]:
    try:
        async with await psycopg.AsyncConnection.connect(ADMIN_DSN, autocommit=True, connect_timeout=3) as conn:
            await conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
            await conn.execute(f"CREATE DATABASE {TEST_DB}")
    except psycopg.OperationalError as exc:
        if os.environ.get("CI"):  # a CI run without its database must not pass by skipping most of the suite
            pytest.fail(f"Postgres not available at {ADMIN_DSN}: {exc}")
        pytest.skip(f"Postgres not available at {ADMIN_DSN}: {exc}")
    yield _test_dsn()


@pytest.fixture
async def make_memory(test_dsn: str) -> AsyncIterator[Callable[..., Any]]:
    """Factory: ``await make_memory(judge=..., extractor=..., **config_overrides)``."""
    opened: list[Memory] = []

    async def factory(*, extractor=None, embedder=None, judge=None, start_runner=False, **overrides) -> Memory:
        overrides.setdefault("api_token", None)  # ignore any MEMX_API_TOKEN in the developer's .env
        cfg = MemxConfig(dsn=test_dsn, insert_mode=overrides.pop("insert_mode", "inline"), **overrides)
        mem = await Memory.connect(
            cfg,
            extractor=extractor or SentenceExtractor(),
            embedder=embedder or HashEmbedder(cfg.embedding_dim),
            judge=judge or FakeJudge(),
            start_runner=start_runner,
        )
        async with mem.ctx.pool.connection() as conn:
            await conn.execute(f"TRUNCATE {TABLES} CASCADE")
        opened.append(mem)
        return mem

    yield factory
    for mem in opened:
        await mem.close()


@pytest.fixture
async def mem(make_memory) -> Memory:
    return await make_memory()
