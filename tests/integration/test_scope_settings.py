from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pydantic import ValidationError

from memx.testing import SentenceExtractor
from memx.types import Messages, Text, Turn
from memx_api.app import app


async def test_settings_inherit_down_the_chain_and_clear(mem):
    base = await mem.scope_settings("memory.acme.alex")
    assert {k: v["from"] for k, v in base["effective"].items()} == dict.fromkeys(base["effective"], "default")
    assert base["effective"]["budget_tokens"]["env"] == "MEMX_BUDGET_TOKENS"

    await mem.set_scope_settings("memory.acme", budget_tokens=900, walk_enabled=False)
    await mem.set_scope_settings("memory.acme.alex", budget_tokens=1500)
    leaf = (await mem.scope_settings("memory.acme.alex.agent1"))["effective"]
    assert leaf["budget_tokens"] == {**leaf["budget_tokens"], "value": 1500, "from": "memory.acme.alex"}
    assert leaf["walk_enabled"]["value"] is False and leaf["walk_enabled"]["from"] == "memory.acme"
    assert leaf["text_trust"]["from"] == "default"

    sibling = (await mem.scope_settings("memory.acme.bo"))["effective"]
    assert sibling["budget_tokens"]["value"] == 900  # siblings see the parent, not alex

    cleared = await mem.set_scope_settings("memory.acme.alex", budget_tokens=None)
    assert cleared["own"] == {} and cleared["effective"]["budget_tokens"]["from"] == "memory.acme"


async def test_bad_settings_are_refused(mem):
    with pytest.raises(ValidationError):
        await mem.set_scope_settings("memory.acme", budget_tokens=10)
    with pytest.raises(ValidationError):
        await mem.set_scope_settings("memory.acme", band_low=0.1)  # server-wide only


async def test_settings_change_recall_insert_and_event_expiry(mem):
    scope = "memory.acme.alex"
    await mem.set_scope_settings("memory.acme", budget_tokens=700, text_trust=0.4, event_ttl_days=3)
    receipt = await mem.insert(scope, Text(text="The team prefers short standups."), mode="inline")
    [fact] = receipt.facts
    assert fact.trust == pytest.approx(0.4)

    result = await mem.recall(scope, "standups", explain=True)
    assert result.explain["budget_tokens"] == 700
    assert result.explain["scope_settings"]["budget_tokens"] == {"value": 700, "from": "memory.acme"}
    assert (await mem.recall(scope, "standups", budget_tokens=2000, explain=True)).explain["budget_tokens"] == 2000


async def test_event_ttl_comes_from_the_scope(make_memory):
    mem = await make_memory(extractor=SentenceExtractor(hints={"pto": "event"}, event_dates={"pto": "2026-11-09"}))
    await mem.set_scope_settings("memory.acme", event_ttl_days=3)
    receipt = await mem.insert("memory.acme.alex", Messages(turns=[Turn(role="user", content="I am on PTO next week.")]),
                               observed_at=datetime(2026, 11, 2, 12, tzinfo=UTC), mode="inline")
    [fact] = receipt.facts
    assert fact.stale_after == fact.event_at + timedelta(days=3)


async def test_settings_api(make_memory):
    app.state.memory = await make_memory()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as c:
        r = await c.patch("/v1/scopes/memory.acme/settings", json={"walk_enabled": False, "budget_tokens": 1200})
        assert r.status_code == 200 and r.json()["own"] == {"walk_enabled": False, "budget_tokens": 1200}
        r = await c.get("/v1/scopes/memory.acme.alex/settings")
        assert r.json()["effective"]["walk_enabled"] == {"value": False, "from": "memory.acme", "default": True,
                                                         "env": "MEMX_WALK_ENABLED"}
        r = await c.patch("/v1/scopes/memory.acme/settings", json={"budget_tokens": 5})
        assert r.status_code == 422 and "budget_tokens" in r.json()["detail"]
        r = await c.patch("/v1/scopes/memory.acme/settings", json={"walk_enabled": None})
        assert r.json()["own"] == {"budget_tokens": 1200}
