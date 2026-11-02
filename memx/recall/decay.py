from __future__ import annotations

import math
from datetime import datetime

from memx.config import MemxConfig
from memx.types import Fact


def freshness(fact: Fact, now: datetime, half_lives: dict[str, float | None]) -> float:
    half_life = half_lives.get(fact.category.value)
    if not half_life:
        return 1.0
    anchor = max(fact.observed_at, fact.last_reinforced_at or fact.observed_at)
    age_days = max(0.0, (now - anchor).total_seconds() / 86400)
    return 2 ** (-age_days / half_life)


def access_factor(access_count: int, cap: float) -> float:
    return min(cap, 1 + 0.05 * math.log1p(access_count))


def final_score(relevance: float, fact: Fact, now: datetime, cfg: MemxConfig) -> float:
    fresh = freshness(fact, now, cfg.half_lives)
    return (relevance * (0.6 + 0.4 * fresh) * (0.5 + 0.5 * fact.trust)
            * access_factor(fact.access_count, cfg.access_boost_cap))


def priority(fact: Fact, now: datetime, cfg: MemxConfig) -> float:
    """Query-independent importance, used to order facts within a section and to trim when there is no query."""
    return fact.trust * freshness(fact, now, cfg.half_lives)
