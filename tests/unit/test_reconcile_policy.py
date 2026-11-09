"""Reconcile policy: destructive floors, the coexist check, code guards, and usage accounting."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from memx import usage
from memx.config import MemxConfig
from memx.pipeline.extract import to_candidates
from memx.pipeline.reconcile import Verdict, _confirm, effective_label, plan_from
from memx.pipeline.similarity import attributes_compatible, is_past_statement
from memx.providers.jev import JevClient
from memx.testing import FakeJudge
from memx.types import Candidate

CFG = MemxConfig(_env_file=None)
NOW = datetime(2026, 10, 30, 12, tzinfo=UTC)
INGEST = uuid4()


def _n(text="x", scope="memory.a", **kw):
    return {"id": uuid4(), "scope": scope, "text": text, **kw}


def _v(label, p, neighbor=None, sim=0.8):
    return Verdict(neighbor or _n(), label, p, sim, "jev")


def _c(text="The user is on page 250 of The Nightingale.", attribute=None, window=0):
    return Candidate(text=text, speaker="user", attribute=attribute, window=window)


@pytest.mark.parametrize(("label", "p", "acted"), [
    ("candidate_replaces", 0.89, "related"),
    ("candidate_replaces", 0.91, "candidate_replaces"),
    ("neighbor_covers", 0.6, "related"),
    ("neighbor_covers", 0.95, "neighbor_covers"),
    ("same", 0.7, "related"),
    ("same", 0.85, "same"),
    ("related", 0.4, "related"),
])
def test_destructive_verdicts_need_their_floor(label, p, acted):
    assert effective_label(_v(label, p), CFG)[0] == acted


def test_held_back_replacements_are_linked_and_listed():
    v = _v("candidate_replaces", 0.7)
    plan = plan_from([v], CFG, "memory.a")
    assert plan.supersede == [] and plan.related and plan.possible_replace == [v.neighbor["id"]]
    assert v.acted == "related" and v.reason == "below_floor"


def test_deprecated_min_prob_still_overrides_both_floors():
    cfg = MemxConfig(_env_file=None, verdict_min_prob=0.5)
    assert effective_label(_v("candidate_replaces", 0.6), cfg)[0] == "candidate_replaces"


def test_different_attributes_never_replace():
    old = _n("The Nightingale has 440 pages.", attribute="page count")
    v = _v("candidate_replaces", 0.97, old)
    plan = plan_from([v], CFG, "memory.a", _c(attribute="reading progress"), INGEST)
    assert not plan.supersede and v.reason == "different_attribute"
    same_attr = _v("candidate_replaces", 0.97, _n("x", attribute="frequent flyer status"))
    assert plan_from([same_attr], CFG, "memory.a", _c(attribute="status"), INGEST).supersede


def test_facts_from_the_same_window_never_replace_each_other():
    sibling = _n("The alert channel is #sre-pager.", ingest_id=INGEST, extract_window=3)
    v = _v("candidate_replaces", 0.95, sibling)
    assert effective_label(v, CFG, _c("Alerts went to #ops-alerts.", window=3), INGEST) == ("related", "same_window")
    # another window of the same insert, or another insert, may still replace
    assert effective_label(v, CFG, _c("Alerts went to #ops-alerts.", window=4), INGEST)[0] == "candidate_replaces"
    assert effective_label(v, CFG, _c("Alerts went to #ops-alerts.", window=3), uuid4())[0] == "candidate_replaces"


def test_a_statement_about_the_past_does_not_replace_the_present():
    current = _n("The alert channel is #sre-pager.")
    v = _v("candidate_replaces", 0.95, current)
    assert effective_label(v, CFG, _c("The previous alert channel was #ops-alerts."), INGEST) == (
        "related", "past_statement")
    transition = _c("The alert channel changed from #ops-alerts to #sre-pager.")
    assert effective_label(v, CFG, transition, INGEST)[0] == "candidate_replaces"


@pytest.mark.parametrize(("text", "past"), [
    ("The previous alert channel was #ops-alerts.", True),
    ("The user used to play tennis.", True),
    ("The user previously lived in Porto but now lives in Lisbon.", False),
    ("The alert channel changed from #ops-alerts to #sre-pager.", False),
    ("Marcus Oyelaran no longer leads the Data Platform Team.", False),
    ("The user is currently on page 250.", False),
])
def test_is_past_statement(text, past):
    assert is_past_statement(text) is past


@pytest.mark.parametrize(("a", "b", "ok"), [
    ("page count", "reading progress", False),
    ("status", "frequent flyer status", True),
    ("home city", "city", True),
    (None, "anything", True),
    ("previous status", "status", True),
])
def test_attributes_compatible(a, b, ok):
    assert attributes_compatible(a, b) is ok


async def test_confirm_question_lets_a_borderline_replace_through_only_when_both_cannot_be_true():
    cfg = MemxConfig(_env_file=None, verdict_confirm=True)
    conflict, compatible = _v("candidate_replaces", 0.7, _n("lives in Porto")), _v("candidate_replaces", 0.7, _n("has a cat"))
    judge = FakeJudge(coexist=lambda c, n: 0.05 if "Porto" in n else 0.9)
    await _confirm(judge, cfg, _c("The user lives in Lisbon."), [conflict, compatible])
    assert len(judge.requests) == 1  # one request for every borderline verdict
    assert effective_label(conflict, cfg)[0] == "candidate_replaces"
    assert effective_label(compatible, cfg)[0] == "related"


async def test_confirm_is_skipped_when_off_or_not_borderline():
    judge = FakeJudge()
    await _confirm(judge, CFG, _c(), [_v("candidate_replaces", 0.7)])
    await _confirm(judge, MemxConfig(_env_file=None, verdict_confirm=True), _c(),
                   [_v("candidate_replaces", 0.95), _v("candidate_replaces", 0.5), _v("related", 0.7)])
    assert judge.requests == []


def test_to_candidates_keeps_the_attribute():
    [c] = to_candidates([{"text": "x", "attribute": " reading progress "}])
    assert c.attribute == "reading progress"


async def test_jev_client_reports_cost_to_the_open_tracker():
    class Http:
        async def post(self, path, body):
            return {"answers": {"q": {"noul": 0.1}}, "usage": {"input_tokens": 120, "cost": 0.00007}}

    from memx.providers.jev import YesNo

    client = JevClient(Http(), "jev-test")
    res = await client.ask({}, {"q": YesNo("?")})  # no tracker open: still fine
    assert res.cost_usd == pytest.approx(0.00007)
    with usage.track() as u:
        await client.ask({}, {"q": YesNo("?")})
        await client.ask({}, {"q": YesNo("?")})
    assert (u.jev_requests, u.jev_input_tokens) == (2, 240) and u.jev_cost_usd == pytest.approx(0.00014)


async def test_split_windows_count_the_discarded_attempt():
    from memx.pipeline import extract as px
    from memx.providers.extractor import ExtractResult
    from memx.types import Text

    class Extractor:
        model = "fake"

        async def extract(self, window, *, observed_at, total_windows):
            if len(window.units) > 1:
                return ExtractResult(facts=[], truncated=True, tokens_in=100, tokens_out=8000)
            return ExtractResult(facts=[{"text": window.units[0].text}], tokens_in=50, tokens_out=40)

    ctx = SimpleNamespace(config=CFG, extractor=Extractor())
    [window] = px.windows_for(ctx, Text(text="First paragraph.\n\nSecond paragraph.").model_dump())
    with usage.track() as u:
        facts, _, tout = await px._extract_window(ctx, window, NOW, 1)
    assert len(facts) == 2 and tout == 8080
    assert (u.extract_splits, u.extract_wasted_tokens_in, u.extract_wasted_tokens_out) == (1, 100, 8000)
