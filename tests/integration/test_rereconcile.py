from __future__ import annotations

from memx import Messages, Turn
from memx.config import MemxConfig
from memx.pipeline.rereconcile import rereconcile
from memx.testing import FakeJudge, SentenceExtractor

SCOPE = "memory.acme.alex"
LEGACY = {"verdict_same_min": 0.5, "verdict_min_prob": 0.5, "verdict_guards": False}


def say(text: str) -> Messages:
    return Messages(turns=[Turn(role="user", content=text)])


def verdict(candidate: str, neighbor: str) -> tuple[str, float]:
    if "page 250" in candidate and "440 pages" in neighbor:
        return "candidate_replaces", 0.62  # the weak verdict that hid a compatible fact
    if "Lisbon" in candidate and "Porto" in neighbor:
        return "candidate_replaces", 0.96  # a real move
    if "oat milk" in candidate and "tea" in neighbor:
        return "neighbor_covers", 0.7
    return "unrelated", 0.9


async def _seeded(make_memory):
    mem = await make_memory(judge=FakeJudge(verdict=verdict), extractor=SentenceExtractor(), **LEGACY)
    await mem.insert(SCOPE, say("The Nightingale has 440 pages."), idempotency_key="a")
    await mem.insert(SCOPE, say("The user lives in Porto."), idempotency_key="b")
    await mem.insert(SCOPE, say("The user prefers tea."), idempotency_key="c")
    await mem.insert(SCOPE, say("The user is on page 250 of The Nightingale."), idempotency_key="d")
    await mem.insert(SCOPE, say("The user lives in Lisbon."), idempotency_key="e")
    await mem.insert(SCOPE, say("The user prefers tea with oat milk."), idempotency_key="f")
    return mem


def _visible(memory: str) -> set[str]:
    return {k for k in ("440 pages", "page 250", "Porto", "Lisbon", "oat milk") if k in memory}


async def test_rereconcile_undoes_only_what_the_current_policy_would_not_do(make_memory):
    mem = await _seeded(make_memory)
    assert _visible(await mem.render(SCOPE)) == {"page 250", "Lisbon"}  # legacy policy hid 440 pages and oat milk

    current = MemxConfig(_env_file=None)
    async with mem.ctx.pool.connection() as conn:
        dry = await rereconcile(conn, current, scope=SCOPE, dry_run=True)
    assert (dry.checked, dry.restored, dry.kept) == (3, 2, 1)
    assert _visible(await mem.render(SCOPE)) == {"page 250", "Lisbon"}  # dry run changed nothing

    async with mem.ctx.pool.connection() as conn:
        report = await rereconcile(conn, current, scope=SCOPE)
        cur = await conn.execute("SELECT result FROM memx_decision WHERE step = 'rereconcile'")
        logged = [r["result"] for r in await cur.fetchall()]
    assert report.restored == 2 and report.reasons == {"below_floor": 2}
    assert _visible(await mem.render(SCOPE)) == {"440 pages", "page 250", "Lisbon", "oat milk"}  # Porto stays retired
    assert {r["action"] for r in logged} == {"restore"} and all(r["verdict_decision_id"] for r in logged)

    async with mem.ctx.pool.connection() as conn:
        again = await rereconcile(conn, current, scope=SCOPE)
    assert again.restored == 0 and again.checked == 1  # idempotent


async def test_rereconcile_does_not_restore_a_second_identical_copy(make_memory):
    mem = await _seeded(make_memory)
    await mem.insert(SCOPE, say("The Nightingale has 440 pages."), idempotency_key="again")  # stored again, active
    async with mem.ctx.pool.connection() as conn:
        report = await rereconcile(conn, MemxConfig(_env_file=None), scope=SCOPE)
    assert report.restored == 1 and [c["fact"] for c in report.changes] == ["The user prefers tea with oat milk."]
    assert (await mem.render(SCOPE)).count("440 pages") == 1


async def test_rereconcile_removes_weak_shadows(make_memory):
    def weak(candidate: str, neighbor: str) -> tuple[str, float]:
        return ("candidate_replaces", 0.55) if "light" in candidate and "dark" in neighbor else ("unrelated", 0.9)

    mem = await make_memory(judge=FakeJudge(verdict=weak), extractor=SentenceExtractor(), **LEGACY)
    await mem.insert(SCOPE, say("The user prefers dark mode."))
    await mem.insert(f"{SCOPE}.agent1", say("The user prefers light mode."))
    assert "dark mode" not in await mem.render(f"{SCOPE}.agent1")
    assert (await mem.rereconcile(SCOPE)).unshadowed == 0  # under its own (legacy) policy nothing changes
    async with mem.ctx.pool.connection() as conn:
        report = await rereconcile(conn, MemxConfig(_env_file=None), scope=SCOPE)
    assert report.unshadowed == 1
    assert "dark mode" in await mem.render(f"{SCOPE}.agent1")
