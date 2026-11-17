"""Benchmark harnesses. They ingest thousands of synthetic conversations, so they must never run against the
database an application uses."""

from __future__ import annotations

import os
from urllib.parse import urlparse


def bench_dsn(dsn: str) -> str:
    """Return ``dsn`` if it is safe for benchmark data; refuse the main memx database."""
    name = urlparse(dsn).path.lstrip("/")
    main = os.environ.get("MEMX_DSN", "")
    if name == "memx" or (main and dsn == main):
        raise SystemExit(
            f"refusing to write benchmark data to {name!r}: that is the application's memory database. "
            "Create a separate one (createdb memx_bench) and pass it with --dsn or MEMX_BENCH_DSN."
        )
    return dsn
