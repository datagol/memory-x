"""Per-job usage accounting: Jev requests and cost, extraction continuations, prompt-cache hits, and
extraction work thrown away by window splits.

A job opens ``track()``; provider code calls ``current()`` and adds to it when one is open. Tasks started with
``asyncio.gather`` inside the job share the same ``Usage`` object, so concurrent requests all count.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass


@dataclass
class Usage:
    jev_requests: int = 0
    jev_input_tokens: int = 0
    jev_cost_usd: float = 0.0
    extract_splits: int = 0  # windows re-extracted in halves after hitting the fact cap
    extract_wasted_tokens_in: int = 0  # tokens of the discarded first attempts
    extract_wasted_tokens_out: int = 0
    extract_continuations: int = 0  # follow-up calls for the facts after a capped answer
    extract_incomplete_windows: int = 0  # windows that could be neither finished nor split; facts may be missing
    extract_cached_tokens_in: int = 0  # prompt tokens served from the provider's prompt cache
    gate_turns_skipped: int = 0  # turns the chit-chat gate kept out of extraction
    extract_batched_windows: int = 0  # windows extracted through the provider batch API (half price)
    extract_batch_tokens_in: int = 0  # tokens billed at the batch rate; included in the receipt's extract totals
    extract_batch_tokens_out: int = 0
    extract_cache_write_tokens_in: int = 0  # prompt tokens written to the provider cache (billed above base)

    def add_jev(self, usage: dict | None) -> None:
        self.jev_requests += 1
        if usage:
            self.jev_input_tokens += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            self.jev_cost_usd += float(usage.get("cost") or 0.0)

    def merge(self, saved: dict | None) -> None:
        """Add counts saved by an earlier run of the same job (one deferred while a batch ran)."""
        for key, value in (saved or {}).items():
            if hasattr(self, key) and isinstance(value, (int, float)):
                setattr(self, key, getattr(self, key) + value)

    def as_dict(self) -> dict:
        out = asdict(self)
        out["jev_cost_usd"] = round(self.jev_cost_usd, 6)
        return out


_current: ContextVar[Usage | None] = ContextVar("memx_usage", default=None)


def current() -> Usage | None:
    return _current.get()


@contextmanager
def track() -> Iterator[Usage]:
    usage = Usage()
    token = _current.set(usage)
    try:
        yield usage
    finally:
        _current.reset(token)
