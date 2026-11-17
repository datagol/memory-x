from __future__ import annotations
from uuid import uuid4
import pytest
from memx.config import MemxConfig
from memx.pipeline.reconcile import Verdict, plan_from
from memx.pipeline.sort import resolve
from memx.types import Category


CFG = MemxConfig(_env_file=None)


def _v(label, p=0.9, scope="memory.a", sim=0.8, source="jev"):
    return Verdict({"id": uuid4(), "scope": scope, "text": "x"}, label, p, sim, source)


def test_sort_resolve_thresholds():
    probs = {c.value: 0.0 for c in Category}
    ok = {"category": {"choice": "preference", "probabilities": {**probs, "preference": 0.9}, "confidence": 0.9},
          "injection": {"noul": 0.01}}
    assert resolve(ok, CFG, 1).category == Category.preference
    weak = {"category": {"choice": "goal", "probabilities": {**probs, "goal": 0.3}, "confidence": 0.3},
            "injection": {"noul": 0.01}}
    # an unsure answer keeps its best real category instead of falling into "other"
    assert resolve(weak, CFG, 1).category == Category.goal
    # "other" as the pick gives way to the next real category
    vague = {"category": {"choice": "other", "confidence": 0.44,
                          "probabilities": {**probs, "other": 0.5, "rule": 0.28, "work_context": 0.12}},
             "injection": {"noul": 0.01}}
    assert resolve(vague, CFG, 1).category == Category.work_context
    # a rule is always given to agents, so it needs a confident answer and is never the fallback
    weak_rule = {"category": {"choice": "rule", "probabilities": {**probs, "rule": 0.4, "goal": 0.3}, "confidence": 0.4},
                 "injection": {"noul": 0.01}}
    assert resolve(weak_rule, CFG, 1).category == Category.goal
    nothing = {"category": {"choice": "other", "probabilities": {**probs, "other": 1.0}, "confidence": 0.9},
               "injection": {"noul": 0.01}}
    assert resolve(nothing, CFG, 1).category == Category.other
    injected = {**ok, "injection": {"noul": 0.7}}
    assert resolve(injected, CFG, 1).category == Category.other


def test_plan_matrix():
    assert plan_from([_v("same")], CFG, "memory.a").action == "drop"
    assert plan_from([_v("same", p=0.4)], CFG, "memory.a").related  # weak same falls back to related
    covers = plan_from([_v("neighbor_covers")], CFG, "memory.a")
    assert covers.action == "insert" and covers.status == "superseded" and covers.superseded_by
    replace_same = plan_from([_v("candidate_replaces")], CFG, "memory.a")
    assert replace_same.supersede and not replace_same.shadow
    replace_parent = plan_from([_v("candidate_replaces", scope="memory")], CFG, "memory.a")
    assert replace_parent.shadow and not replace_parent.supersede
    assert plan_from([_v("unrelated")], CFG, "memory.a").related == []
    mixed = plan_from([_v("related"), _v("candidate_replaces"), _v("unrelated")], CFG, "memory.a")
    assert len(mixed.related) == 1 and len(mixed.supersede) == 1
    assert plan_from([_v("same", source="auto"), _v("candidate_replaces")], CFG, "memory.a").action == "drop"


async def test_gate_drops_chitchat_turns():
    from memx.pipeline.gate import gate_window
    from memx.pipeline.windowing import Unit, Window
    from memx.testing import FakeJudge

    w = Window(0, "messages", [Unit("user", "My sister Ana lives in Lima."), Unit("assistant", "Thanks!"),
                               Unit("user", "ok")])
    kept, record = await gate_window(FakeJudge(), w, 0.1)
    assert kept is not None and [u.text for u in kept.units] == ["My sister Ana lives in Lima."]
    assert record["skipped"] == 2
    empty, _ = await gate_window(FakeJudge(), Window(1, "messages", [Unit("user", "hi"), Unit("assistant", "Hello!")]),
                                 0.1)
    assert empty is None


async def test_jev_client_splits_requests_over_100_questions():
    from memx.providers.jev import JevClient, YesNo

    class Http:
        def __init__(self):
            self.sizes = []

        async def post(self, path, body):
            self.sizes.append(len(body["questions"]))
            return {"answers": {k: {"type": "noul", "noul": 0.5} for k in body["questions"]},
                    "usage": {"cost": 0.001}, "model": "jev"}

    http = Http()
    res = await JevClient(http, "jev").ask({}, {f"q{i}": YesNo("x") for i in range(230)})
    assert sorted(http.sizes) == [30, 100, 100] and len(res.answers) == 230
    assert res.cost_usd == pytest.approx(0.003)
