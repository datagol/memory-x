"""Turn timestamps into words for Jev, which is weak at date arithmetic."""

from __future__ import annotations

from datetime import datetime


def _magnitude(days: float) -> str:
    days = abs(days)
    if days < 14:
        n = max(1, round(days))
        return f"{n} day{'s' if n != 1 else ''}"
    if days < 60:
        return f"about {round(days / 7)} weeks"
    if days < 365:
        n = max(2, round(days / 30))
        return f"about {n} months"
    return "over a year"


def recency_phrase(candidate_at: datetime, neighbor_at: datetime) -> str:
    days = (candidate_at - neighbor_at).total_seconds() / 86400
    if abs(days) < 1:
        return "both observed the same day"
    if days > 0:
        return f"candidate is newer by {_magnitude(days)}"
    return f"candidate is OLDER by {_magnitude(days)}"


def age_phrase(observed_at: datetime, now: datetime) -> str:
    days = (now - observed_at).total_seconds() / 86400
    if days < 1:
        return "today"
    if days < 7:
        return "this week"
    if days < 31:
        return "this month"
    if days < 183:
        return "1-6 months ago"
    if days < 365:
        return "6-12 months ago"
    return "over a year ago"


def event_phrase(event_at: datetime, now: datetime) -> str:
    days = (event_at - now).total_seconds() / 86400
    if abs(days) < 1:
        return "event is today"
    if days > 0:
        return f"event is upcoming in {_magnitude(days)}"
    return f"event happened {_magnitude(days)} ago"
