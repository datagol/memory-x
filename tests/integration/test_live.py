"""End-to-end against real models via OpenRouter. Run with MEMX_LIVE=1 (costs a few cents)."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from memx import Memory, Messages, MemxConfig, Turn

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("MEMX_LIVE") != "1", reason="set MEMX_LIVE=1 to call real models"),
]

PARENT = "memory.acme.jyotish"
CHILD = "memory.acme.jyotish.agent1"


@pytest.fixture
async def live(test_dsn):
    mem = await Memory.connect(MemxConfig(dsn=test_dsn, insert_mode="inline"), start_runner=False)
    async with mem.ctx.pool.connection() as conn:
        await conn.execute("TRUNCATE memx_fact_link, memx_fact, memx_ingest, memx_scope, memx_decision, "
                           "memx_recall_log, memx_job, memx_forget_log CASCADE")
    yield mem
    await mem.close()


async def test_walkthrough_example(live: Memory):
    first = await live.insert(PARENT, Messages(turns=[
        Turn(role="user", content="I always want dashboards in dark mode. I drink a lot of espresso. "
                                  "These days I'm mostly working on the memory-x project."),
        Turn(role="assistant", content="Noted! I'll keep dashboards dark."),
    ]), observed_at=datetime(2026, 8, 1, tzinfo=UTC))
    assert first.status == "committed" and first.added >= 3, first
    by_text = {f.text: f for f in first.facts}
    print("\nfirst ingest:", [(f.category, f.text) for f in first.facts])
    assert any(f.category == "preference" and "dark" in f.text.lower() for f in first.facts)
    assert any(f.category == "work_context" and "memory-x" in f.text.lower() for f in first.facts)

    second = await live.insert(CHILD, Messages(turns=[
        Turn(role="user", content="btw I switched to light mode, the dark one hurts my eyes. "
                                  "Also I'm on PTO next week"),
    ]), observed_at=datetime(2026, 10, 30, tzinfo=UTC))
    print("second ingest:", [(f.category, f.text) for f in second.facts or []], second)
    assert second.shadowed >= 1
    child = await live.render(CHILD)
    parent = await live.render(PARENT)
    print("\n--- child ---\n" + child + "\n--- parent ---\n" + parent)
    assert "light" in child.lower() and "dark mode in dashboards" not in child.lower()
    assert "dark" in parent.lower()
    assert any(f.category == "event" and "2026-11-02" in f.text for f in second.facts)

    ranked = await live.recall(CHILD, "Which colour theme should the new dashboard use?", force_mode="ranked")
    print("\n--- ranked ---\n" + ranked.markdown, [(round(f.relevance, 2), f.text) for f in ranked.facts])
    assert ranked.facts and "light" in ranked.facts[0].text.lower()
    del by_text
