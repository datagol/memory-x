from __future__ import annotations

from dataclasses import dataclass

from psycopg_pool import AsyncConnectionPool

from memx.config import MemxConfig
from memx.providers.embedder import Embedder
from memx.providers.extractor import Extractor
from memx.providers.jev import Judge


@dataclass
class Context:
    """Everything a pipeline step needs. Built once by ``Memory.connect``."""

    config: MemxConfig
    pool: AsyncConnectionPool
    extractor: Extractor
    embedder: Embedder
    judge: Judge
