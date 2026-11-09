"""The MCP server: tools over a real memx (fake providers), and the HTTP layer (token, health, host checks)."""

from __future__ import annotations

import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from starlette.testclient import TestClient

from memx_mcp import server
from memx_mcp.server import mcp

SCOPE = "memory.acme.alex"


@pytest.fixture
async def tools(make_memory):
    mem = await make_memory()
    server._State.memory = mem
    yield mem
    server._State.memory = None


async def call(name: str, **args):
    result = await mcp.call_tool(name, args)
    assert not getattr(result, "is_error", False) and not getattr(result, "isError", False), result
    structured = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if structured is not None:
        return structured.get("result", structured)
    return json.loads(result.content[0].text)


async def call_error(name: str, **args) -> str:
    """Called directly, the server raises the tool's error; over the wire the client gets it as an error result."""
    with pytest.raises(ToolError) as caught:
        await mcp.call_tool(name, args)
    return str(caught.value)


async def test_lists_the_tools_with_safety_hints():
    listed = {t.name: t for t in await mcp.list_tools()}
    assert set(listed) == {"recall", "remember", "remember_conversation", "remember_document", "ingest_status",
                           "list_facts", "fact_history", "list_scopes", "forget_fact"}
    assert listed["recall"].annotations.read_only_hint is True
    assert listed["forget_fact"].annotations.destructive_hint is True


async def test_remember_then_recall(tools):
    stored = await call("remember", text="I lead the data platform team. We never deploy on Fridays.", scope=SCOPE)
    assert stored["status"] == "committed" and stored["added"] == 2
    assert "We never deploy on Fridays." in " ".join(stored["facts"])

    recalled = await call("recall", query="can I ship on Friday?", scope=SCOPE)
    assert "Fridays" in recalled["memory"]
    assert {f["text"] for f in recalled["facts"]} >= {"We never deploy on Fridays."}

    listed = await call("list_facts", scope=SCOPE)
    assert listed["count"] == 2
    fid = listed["facts"][0]["id"]
    history = await call("fact_history", fact_id=fid)
    assert history["versions"][0]["id"] == fid

    scopes = await call("list_scopes", prefix="memory.acme")
    assert scopes["scopes"] == [{"scope": SCOPE, "active_facts": 2}]

    assert (await call("forget_fact", fact_id=fid))["facts_removed"] == 1
    assert (await call("list_facts", scope=SCOPE))["count"] == 1


async def test_conversation_document_and_default_scope(tools):
    conv = await call("remember_conversation", turns=[
        {"role": "user", "content": "My manager is Priya."}, {"role": "assistant", "content": "Noted."}])
    assert conv["status"] == "committed" and conv["added"] >= 1
    doc = await call("remember_document", text="The office is in Pune.", source="handbook.md")
    assert doc["status"] == "committed"
    facts = await call("list_facts")
    assert facts["scope"] == server.DEFAULT_SCOPE and facts["count"] >= 2
    status = await call("ingest_status", ingest_id=doc["ingest_id"])
    assert status["status"] == "committed" and status["facts"]


async def test_bad_input_is_a_tool_error_not_a_crash(tools):
    assert "invalid scope segment" in await call_error("recall", query="x", scope="memory.bad scope")
    assert "not a valid id" in await call_error("fact_history", fact_id="nope")
    assert "not found" in await call_error("fact_history", fact_id="00000000-0000-0000-0000-000000000000")


async def test_without_a_key_tools_say_what_to_set(monkeypatch):
    monkeypatch.setenv("MEMX_PROVIDERS", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("MEMX_OPENROUTER_API_KEY", "")
    server._State.memory = None
    assert "OPENROUTER_API_KEY is not set" in await call_error("recall", query="anything")


def _client(monkeypatch, **env) -> TestClient:
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    from memx_mcp import app as app_module

    return TestClient(app_module.build(), base_url="http://localhost:8100")


INIT = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json",
           "mcp-protocol-version": "2025-11-25"}


def test_http_token_health_and_hosts(monkeypatch):
    with _client(monkeypatch, MEMX_API_TOKEN="s3cret") as client:
        assert client.get("/healthz").status_code == 200
        assert client.post("/mcp", json=INIT, headers=HEADERS).status_code == 401
        assert client.post("/mcp", json=INIT, headers={**HEADERS, "authorization": "Bearer wrong"}).status_code == 401
        ok = client.post("/mcp", json=INIT, headers={**HEADERS, "authorization": "Bearer s3cret"})
        assert ok.status_code == 200, ok.text
        assert "recall" in ok.text
        evil = client.post("/mcp", json=INIT, headers={**HEADERS, "authorization": "Bearer s3cret",
                                                        "host": "attacker.example"})
        assert evil.status_code == 421


def test_http_open_when_no_token(monkeypatch):
    with _client(monkeypatch, MEMX_API_TOKEN="") as client:
        assert client.post("/mcp", json=INIT, headers=HEADERS).status_code == 200
