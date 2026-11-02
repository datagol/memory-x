from __future__ import annotations

import math

DEFAULT_CHARS_PER_TOKEN = 3.6


def estimate_tokens(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Cheap token estimate. Budgets apply a safety factor on top, so this errs small on purpose."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / chars_per_token))
