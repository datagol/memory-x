"""Entities, topics and the two-hop walk in ranked recall."""

from __future__ import annotations

from memx import Text
from memx.pipeline.entities import normalize, tokens_subset
from memx.testing import FakeJudge, SentenceExtractor, demo_grade, guess_entities

ORG = "memory.halvorsen"
# Ranked mode normally runs over hundreds of facts with search returning 40 of them. On a 12-fact test memory
# search would return everything, leaving the walk nothing to add, so these settings keep search narrow.
SMALL = {"recall_k_vector": 3, "recall_k_fts": 3, "walk_hub_min_degree": 4, "walk_topic_weight": 0.0,
         "walk_grade_floor": 0.3}  # the fake grader is crude; the stricter walked-fact floor gets its own test

HALVORSEN = (
    "Halvorsen Bioworks prefers Azure as its cloud provider. "
    "Legacy data remains on ARCHIVE-02. "
    "Marcus Oyelaran is the CTO of Halvorsen Bioworks. "
    "Marcus Oyelaran leads the Data Platform Team. "
    "The Data Platform Team owns all database migrations. "
    "Marcus Oyelaran approved retiring ARCHIVE-02 by Q2. "
    "The Azure enterprise contract renews in March. "
    "Priya Lindqvist is the CEO of Halvorsen Bioworks. "
    "Halvorsen Bioworks was founded in Oslo. "
    "Halvorsen Bioworks sells enzyme kits to laboratories. "
    "Halvorsen Bioworks runs its finance on Workday. "
    "Halvorsen Bioworks holds an all-hands every Friday."
)


def test_normalize_and_subset():
    assert normalize("Dr. Marcus Oyelaran's") == "marcus oyelaran"
    assert normalize("the Data Platform Team") == "data platform team"
    assert normalize("ARCHIVE-02") == "archive-02"
    assert tokens_subset("marcus", "marcus oyelaran")
    assert not tokens_subset("marcus oyelaran", "marcus oyelaran")
    assert not tokens_subset("ab", "ab team")


def test_guess_entities_skips_sentence_starts_and_the_user():
    names = {e["name"] for e in guess_entities("The user asked Marcus Oyelaran about ARCHIVE-02 and Azure.")}
    assert names == {"Marcus Oyelaran", "ARCHIVE-02", "Azure"}
    assert guess_entities("The user likes chicken.") == []


async def test_extraction_stores_entities_shared_within_tenant(make_memory):
    mem = await make_memory(extractor=SentenceExtractor(hints={"prefers": "preference", "cto": "identity"}))
    receipt = await mem.insert(ORG, Text(text=HALVORSEN, source="onboarding"))
    assert receipt.added == 12
    by_text = {f.text: f for f in receipt.facts}
    cto = await mem.get_fact(by_text["Marcus Oyelaran is the CTO of Halvorsen Bioworks."].id)
    assert {e.name for e in cto.entities} == {"Marcus Oyelaran", "Halvorsen Bioworks"}
    hubs = await mem.entities(ORG)
    top = hubs[0]
    assert top["name"] == "Halvorsen Bioworks" and top["active"] == 7
    marcus = next(e for e in hubs if e["name"] == "Marcus Oyelaran")
    assert marcus["active"] == 3 and marcus["type"] == "person"


async def test_partial_name_merges_as_alias_after_judge_agrees(make_memory):
    mem = await make_memory(extractor=SentenceExtractor(entities={"marcus approved": ("Marcus", "person")}))
    await mem.insert(ORG, Text(text="Marcus Oyelaran is the CTO of Halvorsen Bioworks."), idempotency_key="a")
    receipt = await mem.insert(ORG, Text(text="Marcus approved retiring ARCHIVE-02 by Q2."), idempotency_key="b")
    [fact] = receipt.facts
    fact = await mem.get_fact(fact.id)
    assert {e.name for e in fact.entities} == {"Marcus Oyelaran", "ARCHIVE-02"}
    async with mem.ctx.pool.connection() as conn:
        cur = await conn.execute("SELECT name, aliases FROM memx_entity WHERE name = 'Marcus Oyelaran'")
        row = await cur.fetchone()
        assert row["aliases"] == ["marcus"]
        cur = await conn.execute("SELECT result FROM memx_decision WHERE step = 'entity'")
        [decision] = await cur.fetchall()
        assert decision["result"]["action"] == "alias" and decision["result"]["p_same"] >= 0.7
    history = await mem.history(fact.id)
    assert any(d["step"] == "entity" for d in history.versions[0].decisions)


async def test_partial_name_stays_separate_when_judge_disagrees(make_memory):
    mem = await make_memory(extractor=SentenceExtractor(entities={"priya joined": ("Priya", "person")}),
                            judge=FakeJudge(merge=lambda a, b: 0.2))
    await mem.insert(ORG, Text(text="Priya Lindqvist is the CEO of Halvorsen Bioworks."), idempotency_key="a")
    await mem.insert(ORG, Text(text="Priya joined the Data Platform Team."), idempotency_key="b")
    names = {e["name"] for e in await mem.entities(ORG)}
    assert {"Priya Lindqvist", "Priya"} <= names


async def test_topic_comes_from_the_sort_request(make_memory):
    mem = await make_memory(extractor=SentenceExtractor(hints={"likes": "preference"}))
    receipt = await mem.insert(ORG, Text(text="The user likes chicken and seafood for dinner."))
    [fact] = receipt.facts
    assert fact.topic == "food_and_drink" and fact.topic_confidence and fact.topic_confidence >= 0.45
    judge: FakeJudge = mem.ctx.judge  # type: ignore[assignment]
    sort_requests = [q for _, q in judge.requests if "topic" in q and "category" in q]
    assert sort_requests, "topic and category are asked in the same request"


async def test_walk_reaches_facts_two_hops_from_the_question(make_memory):
    mem = await make_memory(extractor=SentenceExtractor(), judge=FakeJudge(grade=demo_grade), **SMALL)
    await mem.insert(ORG, Text(text=HALVORSEN, source="onboarding"))
    recall = await mem.recall(ORG, "Who should I talk to about moving ARCHIVE-02 to Azure?",
                              force_mode="ranked", explain=True)
    texts = {f.text: f for f in recall.facts}
    owns = texts["The Data Platform Team owns all database migrations."]
    leads = texts["Marcus Oyelaran leads the Data Platform Team."]
    assert leads.found_by == "entity_hop" and [s.via for s in leads.path] == ["Marcus Oyelaran"]
    assert owns.found_by == "entity_hop" and [s.via for s in owns.path] == ["Marcus Oyelaran", "Data Platform Team"]
    assert owns.path[1].from_fact == leads.id
    ex = recall.explain
    assert ex["walk"]["hops"] == 2 and ex["walk"]["expanded"] >= 2
    assert [h["name"] for h in ex["walk"]["hubs_cut"]] == ["Halvorsen Bioworks"]
    walked = [c for c in ex["candidates"] if c["found_by"] == "entity_hop"]
    assert all(c["path"] and c["walk_score"] > 0 for c in walked)
    # A walked fact carries the 0.85 factor, so at equal relevance a direct hit wins.
    for c in walked:
        if c["score"] is not None:
            expected = c["relevance"] * c["freshness_factor"] * c["trust_factor"] * c["access_factor"] * 0.85
            assert abs(c["score"] - expected) < 0.002


async def test_walked_facts_need_a_higher_grade_than_search_hits(make_memory):
    mem = await make_memory(extractor=SentenceExtractor(), judge=FakeJudge(grade=demo_grade),
                            **{**SMALL, "walk_grade_floor": 0.5})
    await mem.insert(ORG, Text(text=HALVORSEN))
    recall = await mem.recall(ORG, "Who should I talk to about moving ARCHIVE-02 to Azure?", force_mode="ranked",
                              explain=True)
    by_text = {c["text"]: c for c in recall.explain["candidates"]}
    leads = by_text["Marcus Oyelaran leads the Data Platform Team."]
    assert leads["found_by"] == "entity_hop" and 0.3 <= leads["relevance"] < 0.5
    assert leads["status"] == "below_floor"  # would have passed the 0.3 floor as a search hit
    assert recall.explain["walk_grade_floor"] == 0.5


async def test_grade_request_carries_the_path(make_memory):
    judge = FakeJudge(grade=demo_grade)
    mem = await make_memory(extractor=SentenceExtractor(), judge=judge, **SMALL)
    await mem.insert(ORG, Text(text=HALVORSEN))
    await mem.recall(ORG, "Who should I talk to about moving ARCHIVE-02 to Azure?", force_mode="ranked")
    state, questions = next((s, q) for s, q in judge.requests if any(k.startswith("f") for k in q))
    walked = [f for f in state["judge_context"]["facts"] if f.get("found_by") == "entity link"]
    assert walked and all("via" in f and f["from_fact"] is not None for f in walked)
    i = walked[0]["i"]
    linked_q = questions.get(f"f{i}_l") or questions[f"f{i}"]  # search hits get both, walked facts only the linked one
    assert "linked to the fact at index" in linked_q.instructions


async def test_search_hits_the_walk_reaches_are_graded_with_their_link(make_memory):
    judge = FakeJudge(grade=demo_grade)
    mem = await make_memory(extractor=SentenceExtractor(), judge=judge, walk_hub_min_degree=4, walk_seeds=2)
    await mem.insert(ORG, Text(text=HALVORSEN))  # 12 facts: default search returns all of them
    recall = await mem.recall(ORG, "Who should I talk to about moving ARCHIVE-02 to Azure?", force_mode="ranked",
                              explain=True)
    linked = [c for c in recall.explain["candidates"] if c["found_by"] == "search" and c.get("path")]
    assert linked and recall.explain["walk"]["bridged"] == len(linked)
    assert all(c["path"][-1]["via"] for c in linked)
    # Linked search hits are asked about alone and with their link; the higher grade wins.
    state, questions = next((s, q) for s, q in judge.requests if any(k.endswith("_l") for k in q))
    assert all(k[:-2] in questions for k in questions if k.endswith("_l"))
    alone = [c for c in linked if not c["graded_via_link"]]
    assert all(c["relevance"] >= 0 for c in alone)


async def test_reresolve_merges_short_names_into_the_full_entity(make_memory):
    mem = await make_memory(extractor=SentenceExtractor(entities={"jonas joined": ("Priya", "person")}),
                            judge=FakeJudge(merge=lambda a, b: 0.1))
    await mem.insert(ORG, Text(text="Priya Lindqvist leads the Security Team."), idempotency_key="a")
    await mem.insert(ORG, Text(text="Jonas joined after a call with the Security Team."), idempotency_key="b")
    assert {"Priya", "Priya Lindqvist"} <= {e["name"] for e in await mem.entities(ORG)}
    mem.ctx.judge.merge = lambda a, b: 0.9  # type: ignore[attr-defined]
    report = await mem.reresolve_entities()
    assert report == [{"tenant": "halvorsen", "short": "Priya", "into": "Priya Lindqvist", "p_same": 0.9, "merged": True}]
    ents = {e["name"]: e for e in await mem.entities(ORG)}
    assert "Priya" not in ents and ents["Priya Lindqvist"]["active"] == 2 and "priya" in ents["Priya Lindqvist"]["aliases"]


async def test_full_name_seen_later_renames_the_short_entity(make_memory):
    mem = await make_memory(extractor=SentenceExtractor(entities={"jonas met": ("Marcus", "person")}))
    await mem.insert(ORG, Text(text="Jonas met with the team about Snowflake."), idempotency_key="a")
    await mem.insert(ORG, Text(text="Marcus Oyelaran is the CTO."), idempotency_key="b")
    ents = {e["name"]: e for e in await mem.entities(ORG)}
    assert "Marcus" not in ents and ents["Marcus Oyelaran"]["active"] == 2 and ents["Marcus Oyelaran"]["aliases"] == ["marcus"]


async def test_walk_never_leaves_the_scope_chain(make_memory):
    mem = await make_memory(extractor=SentenceExtractor(), judge=FakeJudge(grade=demo_grade),
                            **{**SMALL, "recall_k_vector": 1, "recall_k_fts": 1})
    await mem.insert("memory.co", Text(text="Marcus Oyelaran leads the Data Platform Team. "
                                            "The Data Platform Team owns all database migrations."), idempotency_key="co")
    await mem.insert("memory.co.user_3", Text(text="Asked Marcus Oyelaran for sign-off on the ARCHIVE-02 move."),
                     idempotency_key="u3")
    await mem.insert("memory.co.user_7", Text(text="Marcus Oyelaran rejected the Q2 budget for the Data Platform Team. "
                                                   "Prefers Marcus Oyelaran to be copied on everything."),
                     idempotency_key="u7")
    recall = await mem.recall("memory.co.user_3", "Who signs off on the ARCHIVE-02 move?", force_mode="ranked",
                              explain=True)
    scopes = {c["scope"] for c in recall.explain["candidates"]}
    assert "memory.co.user_7" not in scopes
    assert any(c["found_by"] == "entity_hop" and c["scope"] == "memory.co" for c in recall.explain["candidates"])
    # Degree is counted over visible facts only: Marcus has 2 visible facts here, not 4.
    assert recall.explain["walk"]["hubs_cut"] == []


async def test_superseded_fact_is_not_reached_through_an_entity(make_memory):
    def verdict(candidate: str, neighbor: str) -> tuple[str, float]:
        return ("candidate_replaces", 0.9) if "no longer" in candidate and "leads" in neighbor else ("unrelated", 0.9)

    mem = await make_memory(extractor=SentenceExtractor(), judge=FakeJudge(verdict=verdict, grade=demo_grade),
                            **SMALL)
    await mem.insert(ORG, Text(text="Marcus Oyelaran leads the Data Platform Team. "
                                    "Marcus Oyelaran approved retiring ARCHIVE-02 by Q2."), idempotency_key="a")
    await mem.insert(ORG, Text(text="Marcus Oyelaran no longer leads the Data Platform Team."), idempotency_key="b")
    recall = await mem.recall(ORG, "What is happening with ARCHIVE-02?", force_mode="ranked", explain=True)
    texts = [c["text"] for c in recall.explain["candidates"]]
    assert "Marcus Oyelaran leads the Data Platform Team." not in texts
    assert "Marcus Oyelaran no longer leads the Data Platform Team." in texts


async def test_topic_hop_reaches_a_fact_sharing_only_a_theme(make_memory):
    mem = await make_memory(extractor=SentenceExtractor(hints={"likes": "preference"}),
                            judge=FakeJudge(grade=demo_grade),
                            recall_k_vector=1, recall_k_fts=1, walk_hub_min_degree=4)
    await mem.insert(ORG, Text(text="The user likes seafood for dinner. "
                                    "The user drinks espresso every morning. "
                                    "The user runs a marathon every spring. "
                                    "The user lives in Oslo."))
    recall = await mem.recall(ORG, "What does the user like for dinner?", force_mode="ranked", explain=True)
    walked = [c for c in recall.explain["candidates"] if c["found_by"] == "topic_hop"]
    assert [c["text"] for c in walked] == ["The user drinks espresso every morning."]
    assert walked[0]["path"][0]["via"] == "food_and_drink" and walked[0]["path"][0]["kind"] == "topic"


async def test_walk_can_be_switched_off(make_memory):
    mem = await make_memory(extractor=SentenceExtractor(), judge=FakeJudge(grade=demo_grade), walk_enabled=False)
    await mem.insert(ORG, Text(text=HALVORSEN))
    recall = await mem.recall(ORG, "Who should I talk to about moving ARCHIVE-02 to Azure?", force_mode="ranked",
                              explain=True)
    assert recall.explain["walk"] is None
    assert all(c["found_by"] == "search" for c in recall.explain["candidates"])


async def test_forget_removes_orphan_entities(make_memory):
    mem = await make_memory(extractor=SentenceExtractor())
    await mem.insert(ORG, Text(text="Marcus Oyelaran is the CTO of Halvorsen Bioworks."), idempotency_key="a")
    await mem.insert("memory.other", Text(text="Priya Lindqvist runs Lindqvist Labs."), idempotency_key="b")
    await mem.forget(scope=ORG)
    async with mem.ctx.pool.connection() as conn:
        cur = await conn.execute("SELECT name FROM memx_entity ORDER BY name")
        assert [r["name"] for r in await cur.fetchall()] == ["Lindqvist Labs", "Priya Lindqvist"]


async def test_dashboard_graph_and_entities_endpoints(make_memory):
    import httpx

    from memx_api.app import app

    app.state.memory = await make_memory(extractor=SentenceExtractor())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(f"/v1/scopes/{ORG}/ingests", json={"input": {"kind": "text", "text": HALVORSEN}, "mode": "inline"})
        assert r.status_code == 200, r.text
        g = (await c.get(f"/v1/graph?scope={ORG}")).json()
        assert g["entities"][0]["name"] == "Halvorsen Bioworks" and len(g["mentions"]) >= 12
        ents = (await c.get(f"/v1/entities?scope={ORG}")).json()
        assert {e["name"] for e in ents} >= {"Marcus Oyelaran", "Data Platform Team", "ARCHIVE-02"}
        cfg = (await c.get("/v1/config")).json()
        assert cfg["walk"]["hops"] == 2 and len(cfg["topics"]) == 30
        trace = (await c.get(f"/v1/ingests/{r.json()['ingest_id']}/trace")).json()
        cand = next(x for x in trace["candidates"] if x["text"].startswith("Marcus Oyelaran is the CTO"))
        assert {e["name"] for e in cand["entities"]} == {"Marcus Oyelaran", "Halvorsen Bioworks"}


async def test_tenants_never_share_entities(make_memory):
    mem = await make_memory(extractor=SentenceExtractor(entities={"priya joined": ("Priya", "person")}),
                            judge=FakeJudge(merge=lambda a, b: 0.95))
    await mem.insert("memory.company_a", Text(text="Priya Lindqvist leads the Security Team."), idempotency_key="a")
    await mem.insert("memory.company_b.user_1", Text(text="Priya joined the call."), idempotency_key="b")
    async with mem.ctx.pool.connection() as conn:
        cur = await conn.execute("SELECT tenant, name FROM memx_entity ORDER BY tenant, name")
        rows = [(r["tenant"], r["name"]) for r in await cur.fetchall()]
    assert ("company_b", "Priya") in rows and ("company_a", "Priya Lindqvist") in rows
    assert not any(t == "company_b" and n == "Priya Lindqvist" for t, n in rows)
    judge: FakeJudge = mem.ctx.judge  # type: ignore[assignment]
    assert not any("pairs" in s.get("untrusted", {}) for s, _ in judge.requests), "no merge question across tenants"
