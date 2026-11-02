"""Step 3: decide how a candidate relates to what is already stored.

Neighbors come from vector and full-text search over the scope chain. Similarity bands settle the obvious
cases; Jev judges the ambiguous middle. The result is a ``Plan`` that ``commit`` applies under a scope lock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from psycopg import AsyncConnection

from memx.config import MemxConfig
from memx.db import repo
from memx.pipeline.similarity import lexical_guard, text_hash
from memx.providers.jev import UNTRUSTED_NOTE, Choice, Judge, clip, envelope
from memx.timewords import recency_phrase
from memx.types import Candidate, Category

VERDICT_VERSION = "verdict.v2"

# Wording matters: v1 said "newer, corrects, or is more specific" and Jev put a real preference flip at 0.38.
# Naming the conflict explicitly ("both cannot be true now") moved it to 0.76 with no regressions on the probe set.
VERDICTS = {
    "same": "Both statements give the same information; the candidate adds nothing new.",
    "candidate_replaces": (
        "The two statements conflict, so both cannot be true now (a changed preference, a corrected value, a new "
        "status), and the candidate is the newer one. Also choose this when the candidate is a more specific "
        "version of the neighbor."
    ),
    "neighbor_covers": (
        "The neighbor already says everything the candidate says, more specifically or more completely, or the "
        "neighbor is the newer of two conflicting statements."
    ),
    "related": (
        "Both are about the same subject and both can be true at the same time; neither contradicts nor replaces "
        "the other."
    ),
    "unrelated": "Different subjects or different attributes.",
}


@dataclass
class Verdict:
    neighbor: dict[str, Any]
    label: str
    probability: float
    similarity: float
    source: Literal["hash", "auto", "jev"]
    probabilities: dict[str, float] = field(default_factory=dict)
    latency_ms: int | None = None


def relation(neighbor_scope: str, scope: str) -> str:
    """How a neighbor's scope relates to the scope being written: same, ancestor (broader) or descendant (narrower)."""
    if neighbor_scope == scope:
        return "same"
    return "descendant" if neighbor_scope.startswith(scope + ".") else "ancestor"


@dataclass
class Plan:
    action: Literal["insert", "drop"]
    status: Literal["active", "superseded"] = "active"
    reinforce: UUID | None = None
    superseded_by: UUID | None = None
    supersede: list[UUID] = field(default_factory=list)
    supersede_below: list[UUID] = field(default_factory=list)  # outdated facts in narrower scopes
    shadow: list[UUID] = field(default_factory=list)
    related: list[tuple[UUID, float]] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
    fetched_at: datetime | None = None


def verdict_question(i: int) -> Choice:
    return Choice(
        instructions=(
            f"Compare the statement in `untrusted.candidate` with `untrusted.neighbors[{i}]`. "
            f"`judge_context.neighbors[{i}]` gives the neighbor's category, whether it belongs to the same scope "
            f"a broader (parent) one, or a narrower (child) one, and how the candidate's age compares with it. " + UNTRUSTED_NOTE
        ),
        criteria=VERDICTS,
    )


def verdict_state(candidate: Candidate, category: Category, observed_at: datetime,
                  neighbors: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    return envelope(
        {
            "candidate_category": category.value,
            "neighbors": [
                {
                    "i": i,
                    "category": n["category"],
                    "scope_relation": {"same": "same scope", "ancestor": "broader (parent) scope",
                                       "descendant": "narrower (child) scope"}[relation(n["scope"], scope)],
                    "recency": recency_phrase(observed_at, n["observed_at"]),
                }
                for i, n in enumerate(neighbors)
            ],
        },
        {"candidate": clip(candidate.text, 600), "neighbors": [clip(n["text"], 600) for n in neighbors]},
    )


def _considered(row: dict[str, Any], band: str) -> dict[str, Any]:
    return {"id": str(row["id"]), "text": row["text"], "scope": row["scope"], "category": row["category"],
            "sim": round(float(row["sim"]), 4), "via": row.get("via", "exact"), "band": band}


async def judge_neighbors(
    conn: AsyncConnection, judge: Judge, cfg: MemxConfig, *, scope: str, chain: list[str], candidate: Candidate,
    category: Category, embedding: list[float], observed_at: datetime, considered: list[dict[str, Any]] | None = None,
) -> list[Verdict]:
    """Find neighbors and judge the ambiguous ones.

    Two groups are checked. Facts in this scope and its parents (what agents here already read) decide whether the
    candidate is new. Facts in narrower child scopes (which will inherit the candidate) are checked for conflicts
    even when the candidate turns out to be a repeat. Every neighbor looked at is appended to ``considered`` with
    its band: exact, auto, judge, below (under band_low), over_cap (beyond neighbor_max) or not_needed.
    """
    seen = considered if considered is not None else []
    auto_same = lambda r: r["sim"] >= cfg.band_high and lexical_guard(candidate.text, r["text"], cfg.guard_min_jaccard)  # noqa: E731
    exact = await repo.find_exact(conn, chain, text_hash(candidate.text))
    found = await repo.find_neighbors(
        conn, chain, embedding, candidate.text, k_vector=cfg.neighbor_k_vector, k_fts=cfg.neighbor_k_fts, below=scope,
    )
    if exact:
        found = [r for r in found if r["id"] != exact["id"]]
    below = [r for r in found if r["sim"] < cfg.band_low]
    above = [r for r in found if r["sim"] >= cfg.band_low]
    upper = [r for r in above if relation(r["scope"], scope) != "descendant"]
    lower = [r for r in above if relation(r["scope"], scope) == "descendant"]
    upper, upper_over = upper[: cfg.neighbor_max], upper[cfg.neighbor_max:]
    lower, lower_over = lower[: cfg.neighbor_max], lower[cfg.neighbor_max:]

    verdicts: list[Verdict] = []
    to_judge: list[dict[str, Any]] = []
    # 1) Is the candidate already known to agents here?
    if exact:
        seen.append(_considered({**exact, "sim": 1.0}, "exact"))
        verdicts.append(Verdict(exact, "same", 1.0, 1.0, "hash"))
        seen.extend(_considered(r, "not_needed") for r in upper)
    else:
        hit = next((r for r in upper if auto_same(r)), None)
        if hit:
            seen.append(_considered(hit, "auto"))
            seen.extend(_considered(r, "not_needed") for r in upper if r is not hit)
            verdicts.append(Verdict(hit, "same", 1.0, float(hit["sim"]), "auto"))
        else:
            to_judge += upper
            seen.extend(_considered(r, "judge") for r in upper)
    # 2) Does it make anything in a narrower scope outdated or redundant?
    for r in lower:
        if auto_same(r):
            seen.append(_considered(r, "auto"))
            verdicts.append(Verdict(r, "same", 1.0, float(r["sim"]), "auto"))
        else:
            to_judge.append(r)
            seen.append(_considered(r, "judge"))
    seen.extend(_considered(r, "over_cap") for r in upper_over + lower_over)
    seen.extend(_considered(r, "below") for r in below)
    if not to_judge:
        return verdicts
    questions = {f"n{i}": verdict_question(i) for i in range(len(to_judge))}
    res = await judge.ask(verdict_state(candidate, category, observed_at, to_judge, scope), questions)
    for i, row in enumerate(to_judge):
        ans = res.answers[f"n{i}"]
        probs = {k: float(v) for k, v in (ans.get("probabilities") or {}).items()}
        label = ans.get("choice") or "related"
        verdicts.append(Verdict(row, label, probs.get(label, 0.0), float(row["sim"]), "jev", probs, res.latency_ms))
    return verdicts


def plan_from(verdicts: list[Verdict], cfg: MemxConfig, scope: str) -> Plan:
    """Turn per-neighbor verdicts into one action. Low-confidence verdicts fall back to 'related'.

    Neighbors in this scope or a parent decide whether the candidate is new at all. Neighbors in narrower (child)
    scopes never block it: a newer broad fact retires outdated or duplicate child facts, and anything else is linked.
    """

    def effective(v: Verdict) -> str:
        if v.source != "jev":
            return v.label
        if v.label == "same" and v.probability < cfg.verdict_same_min:
            return "related"
        if v.label in ("candidate_replaces", "neighbor_covers") and v.probability < cfg.verdict_min_prob:
            return "related"
        return v.label

    labelled = [(v, effective(v), relation(v.neighbor["scope"], scope)) for v in verdicts]
    outdated_below = [v.neighbor["id"] for v, label, rel in labelled
                      if rel == "descendant" and label in ("candidate_replaces", "same")]
    same = [v for v, label, rel in labelled if label == "same" and rel != "descendant"]
    if same:
        # Already known here. The existing fact still retires outdated child facts.
        best = max(same, key=lambda v: v.similarity)
        return Plan("drop", reinforce=best.neighbor["id"], supersede_below=outdated_below, verdicts=verdicts)
    covers = [v for v, label, rel in labelled if label == "neighbor_covers" and rel != "descendant"]
    if covers:
        best = max(covers, key=lambda v: v.similarity)
        return Plan("insert", status="superseded", superseded_by=best.neighbor["id"], verdicts=verdicts)
    plan = Plan("insert", verdicts=verdicts)
    for v, label, rel in labelled:
        nid = v.neighbor["id"]
        if rel == "descendant":
            if label in ("candidate_replaces", "same"):
                plan.supersede_below.append(nid)
            elif label in ("related", "neighbor_covers"):
                plan.related.append((nid, v.similarity))
        elif label == "candidate_replaces":
            (plan.supersede if rel == "same" else plan.shadow).append(nid)
        elif label == "related":
            plan.related.append((nid, v.similarity))
    return plan
