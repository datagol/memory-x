from __future__ import annotations
from datetime import UTC, datetime, timedelta
import pytest
from memx.pipeline.similarity import lexical_guard, normalize, text_hash
from memx.timewords import age_phrase, event_phrase, recency_phrase


NOW = datetime(2026, 10, 30, 12, tzinfo=UTC)


def test_normalize_and_hash():
    assert normalize("  The user  likes Coffee. ") == "the user likes coffee"
    assert text_hash("The user likes coffee.") == text_hash("the user likes coffee")


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("The user likes coffee", "The user likes coffee.", True),
        ("The user likes coffee", "The user does not like coffee", False),
        ("The user likes coffee", "The user doesn't like coffee", False),
        ("The user likes coffee", "The user dislikes coffee", False),
        ("The team has 5 engineers", "The team has 6 engineers", False),
        ("The user lives in Pune", "The user's manager lives in Pune with family", False),
        ("The user lives in Pune, Maharashtra", "The user lives in Pune", False),
        ("The user lives in Pune", "The user lives in Pune, Maharashtra", False),  # low overlap: Jev decides
        ("The user drinks a lot of espresso", "The user drinks espresso", True),
    ],
)
def test_lexical_guard(a, b, same):
    assert lexical_guard(a, b) is same


def test_time_words():
    assert recency_phrase(NOW, NOW - timedelta(days=42)) == "candidate is newer by about 6 weeks"
    assert recency_phrase(NOW - timedelta(days=3), NOW) == "candidate is OLDER by 3 days"
    assert recency_phrase(NOW, NOW - timedelta(hours=2)) == "both observed the same day"
    assert age_phrase(NOW - timedelta(days=100), NOW) == "1-6 months ago"
    assert event_phrase(NOW + timedelta(days=5), NOW) == "event is upcoming in 5 days"
