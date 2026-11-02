"""Entity resolution: turn the names a candidate mentions into stored entity rows.

Entities are keyed by tenant (the first scope label) and a normalized spelling. Exact normalized matches are
one entity. A partial match ("Marcus" against "Marcus Oyelaran") is settled by one Jev question and, if merged,
remembered as an alias so the question is never asked twice. Anything else becomes a new entity. Resolution is
conservative on purpose: two Priyas staying apart costs one missed hop; merging them puts wrong facts in answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection

from memx.config import MemxConfig
from memx.db import repo
from memx.providers.jev import UNTRUSTED_NOTE, Choice, Judge, clip, envelope
from memx.types import Candidate, EntityMention

# v2: "does the short form refer to the known entity", with up to three facts about the known one and how many
# known entities share a word. v1 ("do these name the same thing") put Priya vs Priya Lindqvist at 0.09 and
# Northwind vs Northwind Analytics at 0.01 on real data; v2 gets 0.90 and 0.98 and still keeps an unrelated
# Priya (0.03) and "azure blue" vs Azure DevOps (0.01) apart.
ENTITY_VERSION = "entity.v2"

_TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof", "sir", "madam", "mx"}
_GENERIC = {
    "user", "the user", "assistant", "the assistant", "ai", "i", "me", "my", "we", "us", "you", "they", "them",
    "he", "she", "it", "someone", "people", "team", "the team", "company", "the company", "manager",
    "the manager", "colleague", "a colleague", "friend", "a friend", "family", "work", "home", "today",
    "tomorrow", "yesterday",
}
_WORD = re.compile(r"[^\w\s-]+", re.UNICODE)

MERGE_OPTIONS = {
    "same": ("The shorter name is a first name, short form or nickname for the longer one, and the facts are "
             "plausibly about the same one (same organization, team, project or circle of people)."),
    "different": ("Something in the facts shows they are different: a different organization, role, place or kind "
                  "of thing, or the shorter word is used in an unrelated sense."),
    "cannot_tell": "The facts point in neither direction.",
}


def normalize(name: str) -> str:
    """Lowercase, drop titles and punctuation (hyphens stay), collapse spaces, drop a trailing possessive."""
    text = re.sub(r"['’]s\b", "", name.lower())
    text = _WORD.sub(" ", text).strip()
    text = re.sub(r"\s+", " ", text)
    words = [w for w in text.split(" ") if w and w not in _TITLES]
    if words and words[0] in ("the", "a", "an") and len(words) > 1:
        words = words[1:]
    return " ".join(words)


def is_generic(norm: str) -> bool:
    return not norm or norm in _GENERIC or len(norm) < 2 or norm.isdigit()


def tokens_subset(a: str, b: str) -> bool:
    """True when every word of the shorter name appears in the longer one and they share a real word."""
    ta, tb = set(a.split()), set(b.split())
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return bool(small) and small <= big and small != big and any(len(w) >= 3 for w in small)


def tenant_of(scope: str) -> str:
    """The label under the root (`company_2` in `memory.company_2.user_3`). Entities are shared within a tenant and
    never across tenants. Every scope starts with the same root, so the root itself cannot be the tenant."""
    labels = scope.split(".")
    return labels[1] if len(labels) > 1 else labels[0]


@dataclass
class EntityPlan:
    """What commit should do for one mention: link to an existing entity, merge as an alias, or create."""

    mention: EntityMention
    norm: str
    action: str  # link | alias | create | skip
    entity_id: UUID | None = None
    canonical: str | None = None
    merge: dict[str, Any] | None = None  # the judge decision behind an alias/create choice, for the trace

    def trace(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.mention.name, "type": self.mention.type, "action": self.action}
        if self.canonical and self.canonical != self.mention.name:
            out["canonical"] = self.canonical
        if self.entity_id:
            out["entity_id"] = str(self.entity_id)
        if self.merge:
            out["merge"] = self.merge
        return out


def merge_question(i: int) -> Choice:
    return Choice(
        instructions=(
            f"Entity resolution in one organization's memory. `untrusted.pairs[{i}].a` appears in `a_fact`; "
            f"`untrusted.pairs[{i}].b` is an already known entity, shown with facts that mention it in `b_facts`. "
            f"`judge_context.pairs[{i}]` says how many known entities share a word with `a`. People usually write "
            f"a short form after the full name is known. Does `a` refer to `b`? Judge only that pair. "
            + UNTRUSTED_NOTE
        ),
        criteria=MERGE_OPTIONS,
    )


@dataclass
class MergeAsk:
    short: str          # the name as written in the new fact
    short_type: str
    fact: str           # the fact it appeared in
    other: dict[str, Any]  # the known entity row: id, name, norm, type
    sharing: int        # how many known entities share a word with the short name


async def ask_merges(conn: AsyncConnection, judge: Judge, asks: list[MergeAsk]) -> tuple[list[dict[str, Any]], int]:
    """One Jev request for every pending pair. Returns per pair {p_same, probabilities} and the latency."""
    if not asks:
        return [], 0
    pairs = []
    for a in asks:
        samples = await repo.entity_sample_facts(conn, a.other["id"], limit=3)
        pairs.append({"a": clip(a.short, 80), "a_fact": clip(a.fact, 400), "b": clip(a.other["name"], 80),
                      "b_facts": [clip(t, 300) for t in samples]})
    state = envelope(
        {"pairs": [{"a_type": a.short_type, "b_type": a.other["type"], "known_entities_sharing_a_word": a.sharing}
                   for a in asks]},
        {"pairs": pairs},
    )
    res = await judge.ask(state, {f"m{i}": merge_question(i) for i in range(len(asks))})
    out = []
    for i in range(len(asks)):
        ans = res.answers[f"m{i}"]
        probs = {k: float(v) for k, v in (ans.get("probabilities") or {}).items()}
        out.append({"p_same": probs.get("same", 1.0 if ans.get("choice") == "same" else 0.0), "probabilities": probs})
    return out, res.latency_ms


async def plan_entities(
    conn: AsyncConnection, judge: Judge, cfg: MemxConfig, *, scope: str, candidate: Candidate,
) -> list[EntityPlan]:
    """Resolve a candidate's mentions. Runs before commit, outside the scope lock; creation happens in commit."""
    if not cfg.entities_enabled or not candidate.entities:
        return []
    tenant = tenant_of(scope)
    plans: list[EntityPlan] = []
    pending: list[tuple[EntityPlan, dict[str, Any]]] = []  # partial matches that need the judge
    seen: set[str] = set()
    for mention in candidate.entities[: cfg.entity_max_per_fact]:
        norm = normalize(mention.name)
        if is_generic(norm) or norm in seen:
            continue
        seen.add(norm)
        plan = EntityPlan(mention=mention, norm=norm, action="create")
        exact = await repo.find_entity(conn, tenant, norm)
        if exact:
            plan.action, plan.entity_id, plan.canonical = "link", exact["id"], exact["name"]
            plans.append(plan)
            continue
        found = await repo.find_entity_partial(conn, tenant, norm.split())
        partial = [p for p in found if tokens_subset(norm, p["norm"])]
        if partial:
            pending.append((plan, MergeAsk(mention.name, mention.type, candidate.text, partial[0], len(found))))
        plans.append(plan)

    answers, latency = await ask_merges(conn, judge, [a for _, a in pending])
    for (plan, ask), ans in zip(pending, answers):
        other = ask.other
        plan.merge = {"with": other["name"], "entity_id": str(other["id"]), "p_same": round(ans["p_same"], 4),
                      "probabilities": ans["probabilities"], "latency_ms": latency}
        if ans["p_same"] >= cfg.entity_merge_min:
            plan.action, plan.entity_id = "alias", other["id"]
            # The longer spelling becomes the display name ("Marcus" seen first, "Marcus Oyelaran" later).
            plan.canonical = plan.mention.name if len(plan.norm) > len(other["norm"]) else other["name"]
    return plans


def entity_decisions(ingest_id: Any, candidate: Candidate, plans: list[EntityPlan], scope: str,
                     subject: UUID | None) -> list[dict[str, Any]]:
    return [
        {
            "step": "entity", "question_version": ENTITY_VERSION, "ingest_id": ingest_id,
            "candidate_index": candidate.index, "subject_fact_id": subject, "scope": scope,
            "result": {"name": p.mention.name, "action": p.action, **(p.merge or {})},
            "latency_ms": (p.merge or {}).get("latency_ms"),
        }
        for p in plans if p.merge
    ]


async def apply_entities(conn: AsyncConnection, *, tenant: str, fact_id: UUID, plans: list[EntityPlan]) -> list[UUID]:
    """Inside the commit transaction: create or alias entities as planned, then link the fact to each."""
    linked: list[UUID] = []
    for plan in plans:
        if plan.action == "skip":
            continue
        if plan.action == "create":
            plan.entity_id = await repo.upsert_entity(conn, tenant, plan.mention.name, plan.norm, plan.mention.type)
            plan.canonical = plan.mention.name
        elif plan.action == "alias" and plan.entity_id is not None:
            await repo.alias_entity(conn, plan.entity_id, plan.mention.name, plan.norm)
        if plan.entity_id is None:
            continue
        await repo.link_entity(conn, fact_id, plan.entity_id, plan.mention.name)
        linked.append(plan.entity_id)
    return linked


async def reresolve(conn: AsyncConnection, judge: Judge, cfg: MemxConfig, tenant: str) -> list[dict[str, Any]]:
    """Repair pass over stored entities: ask about every short name that is a word-subset of exactly one longer
    entity in the tenant, and merge the pairs Jev agrees on. Used after the merge question changed, or for data
    ingested in an order the planner could not resolve. Returns what was asked and done."""
    rows = await repo.tenant_entities(conn, tenant)
    asks: list[MergeAsk] = []
    for short in rows:
        longer = [r for r in rows if r["id"] != short["id"] and tokens_subset(short["norm"], r["norm"])
                  and len(r["norm"]) > len(short["norm"])]
        if len(longer) != 1:
            continue  # nothing to merge into, or ambiguous ("Priya" with two Priya surnames): leave it
        fact = await repo.entity_sample_facts(conn, short["id"], limit=1)
        sharing = sum(1 for r in rows if r["id"] != short["id"] and set(short["norm"].split()) & set(r["norm"].split()))
        asks.append(MergeAsk(short["name"], short["type"], fact[0] if fact else short["name"], longer[0], sharing))
    answers, latency = await ask_merges(conn, judge, asks)
    report = []
    for ask, ans in zip(asks, answers):
        merged = ans["p_same"] >= cfg.entity_merge_min
        short_id = next(r["id"] for r in rows if r["name"] == ask.short)
        if merged:
            await repo.merge_entity(conn, short_id, ask.other["id"])
        report.append({"short": ask.short, "into": ask.other["name"], "p_same": round(ans["p_same"], 4),
                       "merged": merged})
    if report:
        await repo.add_decisions(conn, [{
            "step": "entity", "question_version": ENTITY_VERSION, "scope": tenant, "latency_ms": latency,
            "result": {"name": r["short"], "with": r["into"], "p_same": r["p_same"],
                       "action": "alias" if r["merged"] else "create", "reresolve": True},
        } for r in report])
    return report


__all__ = ["reresolve", "ask_merges", "MergeAsk", "EntityPlan", "plan_entities", "apply_entities", "entity_decisions", "normalize", "tokens_subset",
           "tenant_of", "is_generic"]
