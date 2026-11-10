"""What memx logs for real operations, and what it keeps out of the logs."""

import logging

import httpx
import pytest

from memx import Messages, Turn
from memx.jobs import queue
from memx.jobs.runner import JobRunner
from memx.providers.http import ProviderError
from memx_api.app import app

SCOPE = "memory.acme.alex"
SECRET = "The user's salary is 91000 dollars."


def events(caplog, name):
    return [r for r in caplog.records if getattr(r, "event", None) == name]


def info_text(caplog) -> str:
    """Everything an operator sees at INFO and above: messages and every attached field."""
    return " ".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records if r.levelno >= logging.INFO)


async def test_insert_recall_and_forget_are_logged_without_content(make_memory, caplog):
    caplog.set_level(logging.INFO, logger="memx")
    mem = await make_memory()
    receipt = await mem.insert(SCOPE, Messages(turns=[Turn(role="user", content=SECRET)]))
    received, = events(caplog, "ingest.received")
    committed, = events(caplog, "ingest.committed")
    assert received.ingest_id == committed.ingest_id == str(receipt.ingest_id)
    assert committed.scope == SCOPE and committed.added == 1 and committed.ms >= 0

    result = await mem.recall(SCOPE, "how much do I earn?", force_mode="ranked")
    rec, = events(caplog, "recall")
    assert rec.recall_id == str(result.recall_id) and rec.scope == SCOPE and rec.facts == len(result.facts)
    assert rec.query_chars == len("how much do I earn?")

    await mem.forget(fact_id=receipt.facts[0].id, requested_by="test")
    gone, = events(caplog, "forget")
    assert gone.facts_removed == 1 and gone.requested_by == "test"

    shown = info_text(caplog)
    assert "91000" not in shown and "salary" not in shown and "how much do I earn" not in shown


async def test_failed_jobs_are_logged_with_their_ids(make_memory, caplog):
    caplog.set_level(logging.INFO, logger="memx")
    mem = await make_memory(insert_mode="queued")
    receipt = await mem.insert(SCOPE, Messages(turns=[Turn(role="user", content="The user likes tea.")]))

    async def failing(_payload):
        raise ProviderError("HTTP 502 from upstream", retryable=True, status=502)

    runner = JobRunner(mem.ctx, {"ingest": failing}, schedule_sweep=False)
    async with mem.ctx.pool.connection() as conn:
        job = await queue.claim(conn, "test-worker")
    await runner.run_job(job)
    failed, = events(caplog, "job.failed")
    assert failed.levelno == logging.WARNING and failed.retryable is True
    assert failed.job_id == job["id"] and failed.ingest_id == str(receipt.ingest_id)
    assert "HTTP 502" in failed.error


async def test_api_requests_get_an_id_that_follows_the_work(make_memory, caplog):
    caplog.set_level(logging.INFO, logger="memx")
    mem = await make_memory(api_token="secret")
    app.state.memory = mem
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test",
                                 headers={"Authorization": "Bearer secret"}) as client:
        body = {"input": {"kind": "messages", "turns": [{"role": "user", "content": "The user likes tea."}]},
                "mode": "inline"}
        r = await client.post(f"/v1/scopes/{SCOPE}/ingests", json=body, headers={"X-Request-ID": "gw-42"})
        assert r.headers["x-request-id"] == "gw-42"
        generated = (await client.get(f"/v1/scopes/{SCOPE}/facts")).headers["x-request-id"]
        assert len(generated) == 16
        await client.get("/healthz")
    requests = events(caplog, "http.request")
    assert [(e.method, e.status) for e in requests] == [("POST", 200), ("GET", 200)]
    assert requests[0].request_id == "gw-42" and requests[0].path == f"/v1/scopes/{SCOPE}/ingests"
    committed, = events(caplog, "ingest.committed")
    assert committed.request_id == "gw-42"  # the work the request caused carries its id


def test_mcp_tool_calls_are_logged(monkeypatch, caplog):
    from starlette.testclient import TestClient

    from memx_mcp import app as mcp_app
    from memx_mcp import server

    caplog.set_level(logging.INFO, logger="memx")
    monkeypatch.setenv("MEMX_API_TOKEN", "")
    monkeypatch.setenv("MEMX_PROVIDERS", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("MEMX_OPENROUTER_API_KEY", "")
    monkeypatch.setattr(server._State, "memory", None)
    call = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "recall", "arguments": {"query": SECRET, "scope": "memory.acme"}}}
    headers = {"accept": "application/json, text/event-stream", "content-type": "application/json",
               "mcp-protocol-version": "2025-11-25"}
    with TestClient(mcp_app.build(), base_url="http://localhost:8100") as client:
        assert client.post("/mcp", json=call, headers=headers).status_code == 200
    tool, = events(caplog, "mcp.tool")
    assert (tool.tool, tool.scope, tool.ok) == ("recall", "memory.acme", False)
    assert "OPENROUTER_API_KEY" in tool.error
    assert "91000" not in info_text(caplog)
    assert events(caplog, "http.request")[0].path == "/mcp"
