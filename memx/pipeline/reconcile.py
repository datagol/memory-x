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
from memx.pipeline.similarity import (
    attributes_compatible,
    is_past_statement,
    lexical_guard,
    text_hash,
)
from memx.providers.jev import UNTRUSTED_NOTE, Choice, Judge, YesNo, clip, envelope
from memx.timewords import recency_phrase
from memx.types import Candidate, Category

VERDICT_VERSION = "verdict.v3"  # v3: judge_context carries each fact's attribute label
COEXIST_VERSION = "coexist.v1"
SAME_ATTRIBUTE_VERSION = "same_attribute.v1"
DESTRUCTIVE = ("candidate_replaces", "neighbor_covers")

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
    coexist: float | None = None  # P(both can be true at once), when the confirm question was asked
    same_attribute: float | None = None  # P(the two labels name one attribute), asked when they share no word
    acted: str | None = None  # the label plan_from acted on, after floors and guards
    reason: str | None = None  # why ``acted`` differs from ``label``


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
    possible_replace: list[UUID] = field(default_factory=list)  # destructive verdicts held back as "related"
    verdicts: list[Verdict] = field(default_factory=list)
    fetched_at: datetime | None = None


def verdict_question(i: int) -> Choice:
    return Choice(
        instructions=(
            f"Compare the statement in `untrusted.candidate` with `untrusted.neighbors[{i}]`. "
            f"`judge_context.neighbors[{i}]` gives the neighbor's category, whether it belongs to the same scope "
            f"a broader (parent) one, or a narrower (child) one, how the candidate's age compares with it, and "
            f"which attribute each statement gives when known. " + UNTRUSTED_NOTE
        ),
        criteria=VERDICTS,
    )


def verdict_state(candidate: Candidate, category: Category, observed_at: datetime,
                  neighbors: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    return envelope(
        {
            "candidate_category": category.value,
            **({"candidate_attribute": clip(candidate.attribute, 60)} if candidate.attribute else {}),
            "neighbors": [
                {
                    "i": i,
                    "category": n["category"],
                    **({"attribute": clip(n.get("attribute"), 60)} if n.get("attribute") else {}),
                    "scope_relation": {"same": "same scope", "ancestor": "broader (parent) scope",
                                       "descendant": "narrower (child) scope"}[relation(n["scope"], scope)],
                    "recency": recency_phrase(observed_at, n["observed_at"]),
                }
                for i, n in enumerate(neighbors)
            ],
        },
        {"candidate": clip(candidate.text, 600), "neighbors": [clip(n["text"], 600) for n in neighbors]},
    )


def coexist_question(i: int) -> YesNo:
    return YesNo(instructions=(
        f"Can the statement in `untrusted.candidate` and the statement in `untrusted.neighbors[{i}]` both be true at "
        f"the same time? Answer yes when they describe different attributes, different moments, or details that "
        f"fit together; answer no only when one must be false if the other is true now. " + UNTRUSTED_NOTE
    ))


def same_attribute_question(i: int) -> YesNo:
    return YesNo(instructions=(
        f"`untrusted.candidate` and `untrusted.neighbors[{i}]` are statements about a subject; "
        f"`untrusted.candidate_attribute` and `untrusted.neighbor_attributes[{i}]` name the attribute each one gives, "
        f"written independently, so the same attribute may be worded differently (\"location\" and \"city of "
        f"residence\"; \"role\" and \"job title\"). Do both statements give a value for the same attribute of the "
        f"same subject, so that one could be a newer value of the other? " + UNTRUSTED_NOTE
    ))


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
    judged: list[Verdict] = []
    for i, row in enumerate(to_judge):
        ans = res.answers[f"n{i}"]
        probs = {k: float(v) for k, v in (ans.get("probabilities") or {}).items()}
        label = ans.get("choice") or "related"
        judged.append(Verdict(row, label, probs.get(label, 0.0), float(row["sim"]), "jev", probs, res.latency_ms))
    await _confirm(judge, cfg, candidate, judged)
    await _check_attributes(judge, cfg, candidate, judged)
    return verdicts + judged


def _floor(cfg: MemxConfig, label: str) -> float:
    if label == "same":
        return cfg.verdict_same_min
    if cfg.verdict_min_prob is not None:
        return cfg.verdict_min_prob
    return cfg.verdict_replace_min if label == "candidate_replaces" else cfg.verdict_covers_min


async def _confirm(judge: Judge, cfg: MemxConfig, candidate: Candidate, judged: list[Verdict]) -> None:
    """Ask one narrower question for destructive verdicts just under their floor: can both be true now?"""
    if not cfg.verdict_confirm:
        return
    lo, _ = cfg.verdict_confirm_band
    borderline = [v for v in judged if v.label in DESTRUCTIVE and lo <= v.probability < _floor(cfg, v.label)]
    if not borderline:
        return
    state = envelope({}, {"candidate": clip(candidate.text, 600),
                          "neighbors": [clip(v.neighbor["text"], 600) for v in borderline]})
    res = await judge.ask(state, {f"c{i}": coexist_question(i) for i in range(len(borderline))})
    for i, v in enumerate(borderline):
        v.coexist = float(res.answers[f"c{i}"].get("noul", 1.0))


async def _check_attributes(judge: Judge, cfg: MemxConfig, candidate: Candidate, judged: list[Verdict]) -> None:
    """Labels that share no word may still name one attribute. Ask, but only where the answer changes the outcome:
    a replace or covers verdict that would otherwise act."""
    if not cfg.verdict_guards or not candidate.attribute:
        return
    ask = [v for v in judged
           if v.label in ("candidate_replaces", "neighbor_covers") and v.neighbor.get("attribute")
           and not attributes_compatible(candidate.attribute, v.neighbor["attribute"])
           and (v.probability >= _floor(cfg, v.label)
                or (v.coexist is not None and v.coexist <= cfg.verdict_confirm_max_coexist))]
    if not ask:
        return
    state = envelope({}, {"candidate": clip(candidate.text, 600), "candidate_attribute": clip(candidate.attribute, 60),
                          "neighbors": [clip(v.neighbor["text"], 600) for v in ask],
                          "neighbor_attributes": [clip(v.neighbor["attribute"], 60) for v in ask]})
    res = await judge.ask(state, {f"a{i}": same_attribute_question(i) for i in range(len(ask))})
    for i, v in enumerate(ask):
        value = (res.answers.get(f"a{i}") or {}).get("noul")
        v.same_attribute = float(value) if value is not None else None


def effective_label(v: Verdict, cfg: MemxConfig, candidate: Candidate | None = None,
                    ingest_id: UUID | None = None) -> tuple[str, str | None]:
    """The label to act on, and why it differs from Jev's when it does.

    Destructive outcomes (replace, covers, same) must clear their own floor, or be confirmed by the narrower
    coexist question. Three code rules then hold back replace/covers regardless of probability: the facts name
    different attributes, they were extracted from the same window of the same insert (one statement split in
    two), or exactly one of the two only describes an earlier state.
    """
    if v.source != "jev":
        return v.label, None
    if v.label == "same":
        return ("same", None) if v.probability >= cfg.verdict_same_min else ("related", "below_floor")
    if v.label not in DESTRUCTIVE:
        return v.label, None
    if v.probability < _floor(cfg, v.label):
        confirmed = v.coexist is not None and v.coexist <= cfg.verdict_confirm_max_coexist
        if not confirmed:
            return "related", "below_floor"
    n = v.neighbor
    if candidate is not None and cfg.verdict_guards:
        if not attributes_compatible(candidate.attribute, n.get("attribute")) and not (
                v.same_attribute is not None and v.same_attribute >= cfg.same_attribute_min):
            return "related", "different_attribute"
        if (ingest_id is not None and n.get("ingest_id") == ingest_id
                and n.get("extract_window") is not None and n.get("extract_window") == candidate.window):
            return "related", "same_window"
        # A statement about an earlier state and one about the current state can both be true. Neither may retire
        # the other: not a past candidate replacing a current fact ("the previous channel was #ops-alerts"), not a
        # past neighbor covering a current candidate, and not a current fact retiring a history fact
        # ("works at Meta" vs "the previous employer was Google").
        cand_past, neighbor_past = is_past_statement(candidate.text), is_past_statement(n["text"])
        if cand_past != neighbor_past:
            return "related", "past_statement"
    return v.label, None


def plan_from(verdicts: list[Verdict], cfg: MemxConfig, scope: str, candidate: Candidate | None = None,
              ingest_id: UUID | None = None) -> Plan:
    """Turn per-neighbor verdicts into one action. Verdicts held back by ``effective_label`` become 'related'.

    Neighbors in this scope or a parent decide whether the candidate is new at all. Neighbors in narrower (child)
    scopes never block it: a newer broad fact retires outdated or duplicate child facts, and anything else is linked.
    """
    for v in verdicts:
        v.acted, v.reason = effective_label(v, cfg, candidate, ingest_id)
    labelled = [(v, v.acted, relation(v.neighbor["scope"], scope)) for v in verdicts]
    held = [v.neighbor["id"] for v in verdicts if v.label in DESTRUCTIVE and v.acted == "related"]
    outdated_below = [v.neighbor["id"] for v, label, rel in labelled
                      if rel == "descendant" and label in ("candidate_replaces", "same")]
    same = [v for v, label, rel in labelled if label == "same" and rel != "descendant"]
    if same:
        # Already known here. The existing fact still retires outdated child facts.
        best = max(same, key=lambda v: v.similarity)
        return Plan("drop", reinforce=best.neighbor["id"], supersede_below=outdated_below, verdicts=verdicts,
                    possible_replace=held)
    covers = [v for v, label, rel in labelled if label == "neighbor_covers" and rel != "descendant"]
    if covers:
        best = max(covers, key=lambda v: v.similarity)
        return Plan("insert", status="superseded", superseded_by=best.neighbor["id"], verdicts=verdicts,
                    possible_replace=held)
    plan = Plan("insert", verdicts=verdicts, possible_replace=held)
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
