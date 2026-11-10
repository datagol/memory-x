from __future__ import annotations

import httpx
import pytest

from memx import Messages, Turn
from memx.jobs import queue
from memx.providers.http import ProviderError
from memx.testing import SentenceExtractor
from memx_api.app import app

SCOPE = "memory.acme.alex"


def say(text: str) -> Messages:
    return Messages(turns=[Turn(role="user", content=text)])


class BrokenExtractor(SentenceExtractor):
    def __init__(self, retryable: bool):
        super().__init__()
        self.retryable = retryable
        self.calls = 0

    async def extract(self, window, **kw):
        self.calls += 1
        raise ProviderError("model unavailable", retryable=self.retryable)


async def test_background_mode_commits(make_memory):
    mem = await make_memory(insert_mode="background", start_runner=True, job_poll_seconds=0.1)
    receipt = await mem.insert(SCOPE, say("The user likes tea."))
    assert receipt.status == "received"
    done = await mem.wait(receipt.ingest_id, timeout=10)
    assert done.status == "committed" and done.added == 1 and done.facts[0].text == "The user likes tea."


async def test_queued_mode_with_separate_runner(make_memory):
    producer = await make_memory(insert_mode="queued")
    receipt = await producer.insert(SCOPE, say("The user likes chess."))
    assert (await producer.receipt(receipt.ingest_id)).status == "received"
    await producer.start_runner(concurrency=2)
    done = await producer.wait(receipt.ingest_id, timeout=10)
    assert done.added == 1
    async with producer.ctx.pool.connection() as conn:
        assert await queue.enqueue(conn, "ingest", {"ingest_id": str(receipt.ingest_id)},
                                   f"ingest:{receipt.ingest_id}") is None  # duplicate enqueue ignored


async def test_non_retryable_failure_dead_letters(make_memory):
    mem = await make_memory(insert_mode="queued", extractor=BrokenExtractor(retryable=False))
    receipt = await mem.insert(SCOPE, say("The user likes tea."))
    runner = await mem.start_runner(concurrency=1)
    async with mem.ctx.pool.connection() as conn:
        job = await queue.claim(conn, "test")
    if job:  # the runner may already have taken it
        await runner.run_job(job)
    failed = await mem.wait(receipt.ingest_id, timeout=10)
    assert failed.status == "failed" and "model unavailable" in failed.error
    async with mem.ctx.pool.connection() as conn:
        assert (await queue.stats(conn)).get("dead") == 1


async def test_retryable_failure_requeues_with_backoff(make_memory):
    mem = await make_memory(insert_mode="queued", extractor=BrokenExtractor(retryable=True))
    receipt = await mem.insert(SCOPE, say("The user likes tea."))
    runner = await mem.start_runner(concurrency=1)
    await runner.stop()
    async with mem.ctx.pool.connection() as conn:
        await conn.execute("UPDATE memx_job SET status='queued', run_after=now(), attempts=0")
        job = await queue.claim(conn, "test")
    await runner.run_job(job)
    async with mem.ctx.pool.connection() as conn:
        cur = await conn.execute("SELECT status, attempts, run_after > now() AS later FROM memx_job")
        row = await cur.fetchone()
    assert row["status"] == "queued" and row["attempts"] == 1 and row["later"]
    assert (await mem.receipt(receipt.ingest_id)).status == "received"


class FlakyExtractor(SentenceExtractor):
    def __init__(self):
        super().__init__()
        self.fail = True

    async def extract(self, window, **kw):
        if self.fail:
            raise ProviderError("temporarily down", retryable=False)
        return await super().extract(window, **kw)


async def test_resending_failed_ingest_retries_it(make_memory):
    extractor = FlakyExtractor()
    mem = await make_memory(extractor=extractor)
    with pytest.raises(ProviderError):
        await mem.insert(SCOPE, say("The user likes tea."), idempotency_key="once")
    assert (await mem.facts(SCOPE)) == []
    extractor.fail = False
    receipt = await mem.insert(SCOPE, say("The user likes tea."), idempotency_key="once")
    assert receipt.status == "committed" and receipt.added == 1
    assert receipt.scope == SCOPE and (await mem.receipt(receipt.ingest_id)).scope == SCOPE
    async with mem.ctx.pool.connection() as conn:
        assert (await queue.stats(conn)) == {"done": 1}


async def test_reaper_requeues_expired_lease(make_memory):
    mem = await make_memory(insert_mode="queued")
    await mem.insert(SCOPE, say("The user likes tea."))
    async with mem.ctx.pool.connection() as conn:
        await conn.execute(
            "UPDATE memx_job SET status='running', locked_at = now() - interval '1 hour', locked_by='dead-pod'"
        )
        assert await queue.reap(conn) == 1
        assert (await queue.stats(conn)).get("queued") == 1


async def test_input_too_large(make_memory):
    from memx import InputTooLarge

    mem = await make_memory(max_input_chars=100)
    with pytest.raises(InputTooLarge):
        await mem.insert(SCOPE, say("x" * 101))


@pytest.fixture
async def client(make_memory):
    mem = await make_memory(extractor=SentenceExtractor(hints={"likes": "preference"}), api_token="secret")
    app.state.memory = mem
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test",
                                 headers={"Authorization": "Bearer secret"}) as c:
        yield c


async def test_api_roundtrip(client):
    body = {"input": {"kind": "messages", "turns": [{"role": "user", "content": "The user likes tea."}]},
            "mode": "inline"}
    r = await client.post(f"/v1/scopes/{SCOPE}/ingests", json=body, headers={"Idempotency-Key": "abc"})
    assert r.status_code == 200, r.text
    fact_id = r.json()["facts"][0]["id"]
    again = await client.post(f"/v1/scopes/{SCOPE}/ingests", json=body, headers={"Idempotency-Key": "abc"})
    assert again.json()["ingest_id"] == r.json()["ingest_id"]

    r = await client.post(f"/v1/scopes/{SCOPE}.agent1/recall", json={"query": "tea?"})
    assert r.status_code == 200 and "The user likes tea." in r.json()["markdown"]
    assert "## Preferences" in (await client.get(f"/v1/scopes/{SCOPE}/render")).text
    assert len((await client.get(f"/v1/scopes/{SCOPE}/facts")).json()) == 1
    assert (await client.get(f"/v1/facts/{fact_id}/history")).json()["versions"][0]["fact"]["id"] == fact_id
    export = await client.get(f"/v1/scopes/{SCOPE}/export.okf")
    assert export.headers["content-type"] == "application/zip"
    assert (await client.delete(f"/v1/facts/{fact_id}")).json()["facts_removed"] == 1
    assert (await client.delete(f"/v1/facts/{fact_id}")).status_code == 404
    assert (await client.get("/readyz")).json()["database"] == "ok"


async def test_api_auth_and_validation(client):
    unauth = await client.get(f"/v1/scopes/{SCOPE}/render", headers={"Authorization": "Bearer nope"})
    assert unauth.status_code == 401
    bad = await client.get("/v1/scopes/memory..x/render")
    assert bad.status_code == 422
    big = {"input": {"kind": "text", "text": "x" * 500_000}}
    assert (await client.post(f"/v1/scopes/{SCOPE}/ingests", json=big)).status_code == 413


async def test_api_reports_provider_refusal_clearly(make_memory):
    class Refusing(SentenceExtractor):
        async def extract(self, window, **kw):
            raise ProviderError('/chat/completions: HTTP 403: {"error":{"message":"Key limit exceeded (daily limit).","code":403}}',
                                retryable=False, status=403)

    app.state.memory = await make_memory(extractor=Refusing())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(f"/v1/scopes/{SCOPE}/ingests",
                         json={"input": {"kind": "text", "text": "The user likes tea."}, "mode": "inline"})
    assert r.status_code == 502
    assert r.json()["detail"] == "The model provider refused the request: Key limit exceeded (daily limit)."


async def test_resending_abandoned_ingest_resumes_it(make_memory):
    """A process that dies mid-ingest leaves the ingest in ``received`` and its job ``running``. Once the lease has
    expired, resending the same key must run it; while the lease is live, the resend must leave it alone."""
    extractor = FlakyExtractor()
    mem = await make_memory(extractor=extractor)
    with pytest.raises(ProviderError):
        await mem.insert(SCOPE, say("The user likes tea."), idempotency_key="lost")
    async with mem.ctx.pool.connection() as conn:
        await conn.execute("UPDATE memx_ingest SET status = 'received', error = NULL, candidates = NULL")
        await conn.execute("UPDATE memx_job SET status = 'running', locked_at = now(), locked_by = 'pod-a'")
    extractor.fail = False
    receipt = await mem.insert(SCOPE, say("The user likes tea."), idempotency_key="lost")
    assert receipt.status == "received" and (await mem.facts(SCOPE)) == []  # someone is still working on it
    async with mem.ctx.pool.connection() as conn:
        await conn.execute("UPDATE memx_job SET locked_at = now() - interval '1 hour'")
    receipt = await mem.insert(SCOPE, say("The user likes tea."), idempotency_key="lost")
    assert receipt.status == "committed" and receipt.added == 1
