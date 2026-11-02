from __future__ import annotations

import httpx
import pytest

from memx.testing import FakeJudge, SentenceExtractor
from memx_api.app import app

SCOPE = "memory.acme.alex"


def theme_verdict(candidate: str, neighbor: str) -> tuple[str, float]:
    if "light mode" in candidate and "dark mode" in neighbor:
        return "candidate_replaces", 0.9
    return "related", 0.8


@pytest.fixture
async def client(make_memory):
    mem = await make_memory(extractor=SentenceExtractor(hints={"prefers": "preference", "likes": "preference"}),
                            judge=FakeJudge(verdict=theme_verdict), budget_tokens=120)
    app.state.memory = mem
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def ingest(client, scope: str, text: str, key: str) -> dict:
    body = {"input": {"kind": "messages", "turns": [{"role": "user", "content": text}]}, "mode": "inline",
            "idempotency_key": key}
    r = await client.post(f"/v1/scopes/{scope}/ingests", json=body)
    assert r.status_code == 200, r.text
    return r.json()


async def test_ingest_trace_shows_bands_verdicts_and_actions(client):
    await ingest(client, SCOPE, "The user prefers dark mode in dashboards. The user likes espresso.", "a")
    second = await ingest(client, f"{SCOPE}.agent1", "The user prefers light mode in dashboards.", "b")
    trace = (await client.get(f"/v1/ingests/{second['ingest_id']}/trace")).json()
    assert trace["ingest"]["status"] == "committed" and trace["windows"][0]["units"] == 1
    [cand] = trace["candidates"]
    assert cand["outcome"] == "added" and cand["sort"]["category"] == "preference"
    bands = {n["text"]: n["band"] for n in cand["neighbors"]}
    assert bands["The user prefers dark mode in dashboards."] == "judge"
    assert cand["verdicts"][0]["label"] == "candidate_replaces"
    assert len(cand["shadowed_ids"]) == 1 and cand["shadowed_ids"][0] in trace["facts"]
    assert trace["ingest"]["receipt"]["timings"]["total_ms"] >= 0


async def test_recall_explain_is_returned_and_stored(client):
    for i, topic in enumerate(["espresso", "Postgres", "hiking", "jazz", "tennis", "Kubernetes", "sushi", "Rust"]):
        await ingest(client, SCOPE, f"The user likes {topic} a lot and mentions it in most conversations.", f"k{i}")
    r = (await client.post(f"/v1/scopes/{SCOPE}/recall", json={"query": "espresso coffee?", "explain": True})).json()
    ex = r["explain"]
    assert ex["mode"] == "ranked" and "larger than the budget" in ex["mode_reason"]
    statuses = {c["status"] for c in ex["candidates"]}
    assert "picked" in statuses and statuses & {"below_floor", "over_budget"}
    top = next(c for c in ex["candidates"] if "espresso" in c["text"])
    assert top["status"] == "picked" and top["relevance"] > 0.3 and top["vec_rank"] >= 1
    stored = (await client.get(f"/v1/recalls/{r['recall_id']}")).json()
    assert stored["explain"]["candidates"] == ex["candidates"] and "espresso" in stored["markdown"]
    assert (await client.get(f"/v1/recalls?scope={SCOPE}")).json()[0]["id"] == r["recall_id"]


async def test_graph_scopes_stats_and_lists(client):
    await ingest(client, SCOPE, "The user prefers dark mode in dashboards.", "a")
    await ingest(client, f"{SCOPE}.agent1", "The user prefers light mode in dashboards.", "b")
    g = (await client.get(f"/v1/graph?scope={SCOPE}.agent1")).json()
    assert len(g["facts"]) == 2 and g["scopes"][:3] == ["memory", "memory.acme", SCOPE]
    assert [e["kind"] for e in g["edges"]] == ["shadows"]
    assert sum(f["shadowed_here"] for f in g["facts"]) == 1
    scopes = (await client.get("/v1/scopes")).json()
    assert {s["path"] for s in scopes} == {SCOPE, f"{SCOPE}.agent1"}
    stats = (await client.get(f"/v1/stats?scope={SCOPE}")).json()
    assert stats["links"] == {"shadows": 1, "mentions": 0} and stats["scopes"] == 2
    ingests = (await client.get(f"/v1/ingests?scope={SCOPE}")).json()
    assert len(ingests) == 2 and ingests[0]["preview"].startswith("user: The user prefers light")
    assert (await client.get("/v1/config")).json()["bands"]["low"] == 0.45
    assert len((await client.get("/v1/jobs")).json()) == 2
