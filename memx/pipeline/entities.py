"""Entity resolution: turn the names a candidate mentions into stored entity rows.

Entities are keyed by tenant (the label under `memory`) and a normalized spelling. For each name a candidate
mentions, cheap code finds up to three known entities it might be, and Jev decides. Code never merges two
different spellings on its own, and Jev never has to search.

How candidates are found, and which Jev wording judges them:

| Pairing reason            | Example                              | Wording        |
|---------------------------|--------------------------------------|----------------|
| identical, but ambiguous  | "Priya" (work) vs "Priya" (bakery)   | identical      |
| one name inside the other | "Priya" vs "Priya Lindqvist"         | short form     |
| alternative name          | "Bangalore" vs "Blore" (extractor)   | variant        |
| similar name embedding    | "Kubernetes" vs "K8s"                | variant        |

An identical name is linked without asking unless it is ambiguous: a single-word person name, or a single-word name
whose type differs ("Mercury" the bank vs "Mercury" the planet). A full name such as "Marcus Oyelaran", a code
such as "ARCHIVE-02", or a name already used earlier in the same insert is linked directly.

The two kinds of pair start from opposite defaults. Different spellings stay apart unless Jev is sure they are the
same (`entity_merge_min`, 0.7): two Priyas staying apart costs one missed hop, merging them puts wrong facts in
answers. Identical names stay together unless Jev is sure they differ (`entity_split_min`, 0.7): only then is a
second entity with the same name created (stored key "priya~2"), and later mentions are compared with both. On a
live run, requiring "same" for identical names split one Mahesh into two at 0.3; Jev's "different" was never close
to sure for those, while it was 0.93 for "Priya from the bakery".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection

from memx.scopes import ScopePath
from memx.config import MemxConfig
from memx.db import repo
from memx.providers.jev import UNTRUSTED_NOTE, Choice, Judge, clip, envelope
from memx.types import Candidate, EntityMention

# v3: three wordings chosen by why a pair was formed. On a 20-pair probe set against jev-1.14 it got 18 right;
# the two misses were an identical full team name (now never asked) and "GCP" vs "Google Cloud Platform" (0.58,
# kept apart, a safe miss). v2 short-form wording is unchanged inside v3.
ENTITY_VERSION = "entity.v3"

_TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof", "sir", "madam", "mx"}
_GENERIC = {
    "user", "the user", "assistant", "the assistant", "ai", "i", "me", "my", "we", "us", "you", "they", "them",
    "he", "she", "it", "someone", "people", "team", "the team", "company", "the company", "manager",
    "the manager", "colleague", "a colleague", "friend", "a friend", "family", "work", "home", "today",
    "tomorrow", "yesterday",
}
_WORD = re.compile(r"[^\w\s-]+", re.UNICODE)

HOW_IDENTICAL = "identical name"
HOW_SUBSET = "shares a word"
HOW_AKA = "suggested alternative name"
HOW_SIMILAR = "similar name"
HOW_ALIAS = "known spelling"  # the name is a spelling already merged into the entity ("Marcus" for Marcus Oyelaran)
SETTLED = (HOW_IDENTICAL, HOW_ALIAS)  # names already decided to mean this entity: linked unless Jev is sure otherwise

# One option set per wording. The keys are the same in all three, so a decision always reads the same way.
MERGE_OPTIONS = {
    "same": ("The shorter name is a first name, short form or nickname for the longer one, and the facts are "
             "plausibly about the same one (same organization, team, project or circle of people)."),
    "different": ("Something in the facts shows they are different: a different organization, role, place or kind "
                  "of thing, or the shorter word is used in an unrelated sense."),
    "cannot_tell": "The facts point in neither direction.",
}
VARIANT_OPTIONS = {
    "same": ("Both names refer to the same person, place, organization, team, system, project or product: a short "
             "form, nickname, abbreviation, former name or alternative spelling, and the facts are plausibly about "
             "the same one."),
    "different": ("Something in the facts shows they are different things: a different organization, role, place "
                  "or kind of thing, or the name is used in an unrelated sense."),
    "cannot_tell": "The facts point in neither direction.",
}
IDENTICAL_OPTIONS = {
    "same": "The facts are plausibly about the same one: the same organization, team, project, circle of people or subject.",
    "different": ("The facts point to different ones: a different context, role, organization or kind of thing, or "
                  "the name is used as an ordinary word."),
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


def base(norm: str) -> str:
    """A stored key without its same-name suffix: "priya~2" → "priya"."""
    return norm.split("~", 1)[0]


_GROUP_WORDS = ("team", "group", "department", "squad", "committee", "office", "unit", "board", "guild", "crew")


def is_generic(norm: str) -> bool:
    return not norm or norm in _GENERIC or len(norm) < 2 or norm.isdigit()


def is_description(name: str) -> bool:
    """A lowercase phrase such as "the bakery down the street" is a description, not a name. Descriptive team names
    ("the data platform team") stay, because facts about who owns what connect through them."""
    words = name.split()
    return (name == name.lower() and len(words) >= 3 and words[0] in ("the", "a", "an", "our", "my", "their")
            and words[-1].rstrip("s") not in _GROUP_WORDS)


def tokens_subset(a: str, b: str) -> bool:
    """True when every word of the shorter name appears in the longer one and they share a real word."""
    ta, tb = set(base(a).split()), set(base(b).split())
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return bool(small) and small <= big and small != big and any(len(w) >= 3 for w in small)


def tenant_of(scope: str) -> str:
    """The entity namespace a scope belongs to; see ``ScopePath.tenant``."""
    return ScopePath.parse(scope).tenant


def ambiguous_identical(norm: str, mention_type: str, known: dict[str, Any]) -> bool:
    """Does an identical name still need Jev? Single-word person names, and single-word names whose type differs.
    Full names and codes ("Marcus Oyelaran", "ARCHIVE-02") are trusted."""
    if " " in base(norm):
        return False
    if "person" in (mention_type, known["type"]):
        return True
    return mention_type != known["type"] and "other" not in (mention_type, known["type"])


def how_paired(norm: str, known: dict[str, Any]) -> str:
    if base(known["norm"]) == norm:
        return HOW_IDENTICAL
    if norm in (known.get("aliases") or []):
        return HOW_ALIAS
    if tokens_subset(norm, known["norm"]) or any(tokens_subset(norm, a) for a in known.get("aliases") or []):
        return HOW_SUBSET
    return HOW_SIMILAR


@dataclass
class MergeAsk:
    short: str             # the name as written in the new fact
    short_type: str
    fact: str              # the fact it appeared in
    other: dict[str, Any]  # the known entity row: id, name, norm, type
    sharing: int = 1       # how many known entities were paired with the name
    how: str = HOW_SUBSET
    short_id: Any = None   # the entity the name belongs to, in the repair pass (namesakes share a display name)


@dataclass
class EntityPlan:
    """What commit should do for one mention: link to an existing entity, merge as an alias, or create one."""

    mention: EntityMention
    norm: str
    action: str  # link | alias | create | skip
    entity_id: UUID | None = None
    canonical: str | None = None
    merge: dict[str, Any] | None = None  # the best judged candidate, for the trace and the decision log
    candidates: list[dict[str, Any]] = field(default_factory=list)  # every judged candidate
    variant: bool = False  # create a second entity with an existing name ("priya~2")
    akas: list[str] = field(default_factory=list)  # normalized alternative names from the extractor
    embedding: list[float] | None = None

    def trace(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.mention.name, "type": self.mention.type, "action": self.action}
        if self.canonical and self.canonical != self.mention.name:
            out["canonical"] = self.canonical
        if self.entity_id:
            out["entity_id"] = str(self.entity_id)
        if self.merge:
            out["merge"] = self.merge
        if len(self.candidates) > 1:
            out["candidates"] = self.candidates
        if self.mention.aka:
            out["aka"] = self.mention.aka
        if self.variant:
            out["variant"] = True
        return out


def merge_question(i: int, how: str = HOW_SUBSET) -> Choice:
    if how == HOW_IDENTICAL:
        return Choice(
            instructions=(
                f"Disambiguation in one organization's memory. `untrusted.pairs[{i}].a`, as written in `a_fact`, has "
                f"exactly the same name as the known entity `untrusted.pairs[{i}].b`, shown with facts in `b_facts`. "
                f"Common first names and ordinary words are often shared by different people or things. Does `a` "
                f"refer to the same one as `b`? Judge only that pair. " + UNTRUSTED_NOTE
            ),
            criteria=IDENTICAL_OPTIONS,
        )
    if how == HOW_ALIAS:
        return Choice(
            instructions=(
                f"Disambiguation in one organization's memory. `untrusted.pairs[{i}].a`, as written in `a_fact`, is a "
                f"spelling already known to mean the entity `untrusted.pairs[{i}].b` (shown with facts in `b_facts`): "
                f"earlier mentions written this way were judged to be `b`. Common first names are often shared by "
                f"different people. Does this mention also refer to `b`? Judge only that pair. " + UNTRUSTED_NOTE
            ),
            criteria=IDENTICAL_OPTIONS,
        )
    if how == HOW_SUBSET:
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
    return Choice(
        instructions=(
            f"Entity resolution in one organization's memory. `untrusted.pairs[{i}].a` is a name as written in "
            f"`a_fact`; `untrusted.pairs[{i}].b` is an already known entity, shown with facts that mention it in "
            f"`b_facts`. `judge_context.pairs[{i}]` says why the two were paired. Names that look alike can be "
            f"different things, and different-looking names can be the same thing. Does `a` refer to the same thing "
            f"as `b`? Judge only that pair. " + UNTRUSTED_NOTE
        ),
        criteria=VARIANT_OPTIONS,
    )


async def ask_merges(conn: AsyncConnection, judge: Judge, asks: list[MergeAsk]) -> tuple[list[dict[str, Any]], int]:
    """One Jev request for every pending pair. Returns per pair {p_same, probabilities} and the latency."""
    if not asks:
        return [], 0
    pairs, contexts = [], []
    for a in asks:
        samples = await repo.entity_sample_facts(conn, a.other["id"], limit=3)
        pairs.append({"a": clip(a.short, 80), "a_fact": clip(a.fact, 400), "b": clip(a.other["name"], 80),
                      "b_facts": [clip(t, 300) for t in samples]})
        ctx: dict[str, Any] = {"a_type": a.short_type, "b_type": a.other["type"], "how_found": a.how}
        ctx["known_entities_sharing_a_word" if a.how == HOW_SUBSET else "candidates_for_a"] = a.sharing
        contexts.append(ctx)
    res = await judge.ask(envelope({"pairs": contexts}, {"pairs": pairs}),
                          {f"m{i}": merge_question(i, a.how) for i, a in enumerate(asks)})
    out = []
    for i in range(len(asks)):
        ans = res.answers[f"m{i}"]
        probs = {k: float(v) for k, v in (ans.get("probabilities") or {}).items()}
        out.append({"p_same": probs.get("same", 1.0 if ans.get("choice") == "same" else 0.0), "probabilities": probs})
    return out, res.latency_ms


async def _candidates(conn: AsyncConnection, cfg: MemxConfig, tenant: str, norm: str, mention: EntityMention,
                      akas: list[str], vector: list[float] | None) -> tuple[list[tuple[dict[str, Any], str]], int]:
    """Known entities this name might be, each with the reason it was paired. Also returns how many words-sharing
    entities exist, which the short-form wording reports to Jev."""
    found: dict[Any, tuple[dict[str, Any], str]] = {}

    def add(row: dict[str, Any], how: str) -> None:
        if row["id"] not in found and len(found) < cfg.entity_candidates_max:
            found[row["id"]] = (row, how)

    for row in await repo.find_entities_exact(conn, tenant, norm):
        add(row, how_paired(norm, row))
    words = await repo.find_entity_partial(conn, tenant, norm.split())
    for row in words:
        if tokens_subset(norm, row["norm"]):
            add(row, HOW_SUBSET)
    for row in await repo.find_entities_known_as(conn, tenant, norm, akas):
        add(row, HOW_AKA)
    if vector is not None:
        for row in await repo.find_entities_similar(conn, tenant, vector, cfg.entity_name_sim_min, limit=3):
            add(row, HOW_SIMILAR)
    return list(found.values()), len(words)


async def plan_entities(
    conn: AsyncConnection, judge: Judge, cfg: MemxConfig, *, scope: str, candidate: Candidate,
    vectors: dict[str, list[float]] | None = None, ingest_id: Any = None,
) -> list[EntityPlan]:
    """Resolve a candidate's mentions. Runs before commit, outside the scope lock; creation happens in commit.
    ``vectors`` maps a mention's name to its embedding, computed once per ingest."""
    if not cfg.entities_enabled or not candidate.entities:
        return []
    tenant = tenant_of(scope)
    vectors = vectors or {}
    plans: list[EntityPlan] = []
    pending: list[tuple[EntityPlan, MergeAsk]] = []
    seen: set[str] = set()
    for mention in candidate.entities[: cfg.entity_max_per_fact]:
        norm = normalize(mention.name)
        if is_generic(norm) or is_description(mention.name) or norm in seen:
            continue
        seen.add(norm)
        akas = [a for a in dict.fromkeys(normalize(x) for x in mention.aka) if a and a != norm and not is_generic(a)]
        plan = EntityPlan(mention=mention, norm=norm, action="create", akas=akas, embedding=vectors.get(mention.name))
        plans.append(plan)
        cands, sharing = await _candidates(conn, cfg, tenant, norm, mention, akas, plan.embedding)
        identical = [row for row, how in cands if how in SETTLED]
        same_insert = await repo.entities_in_ingest(conn, [r["id"] for r in identical], ingest_id)
        trusted = [row for row in identical
                   if row["id"] in same_insert or not ambiguous_identical(norm, mention.type, row)]
        if trusted:  # a full name or a code, already known: no question needed
            plan.action, plan.entity_id, plan.canonical = "link", trusted[0]["id"], trusted[0]["name"]
            continue
        if identical:
            plan.variant = True  # if Jev says every identical one is someone else, this is a new namesake
        for row, how in cands:
            pending.append((plan, MergeAsk(mention.name, mention.type, candidate.text, row,
                                           sharing if how == HOW_SUBSET else len(cands), how)))

    answers, latency = await ask_merges(conn, judge, [a for _, a in pending])
    for (plan, ask), ans in zip(pending, answers):
        p_diff = round(ans["probabilities"].get("different", 0.0), 4)
        plan.candidates.append({"with": ask.other["name"], "entity_id": str(ask.other["id"]), "how": ask.how,
                                "p_same": round(ans["p_same"], 4), "p_different": p_diff})
        if plan.merge is None or ans["p_same"] > plan.merge["p_same"]:
            plan.merge = {"with": ask.other["name"], "entity_id": str(ask.other["id"]), "how": ask.how,
                          "p_same": round(ans["p_same"], 4), "p_different": p_diff,
                          "probabilities": ans["probabilities"], "latency_ms": latency, "_row": ask.other}
    for plan in plans:
        if not plan.merge:
            continue
        other = plan.merge.pop("_row")
        if plan.merge["p_same"] < cfg.entity_merge_min:
            # Identical names stay together unless Jev is sure every one of them is someone else.
            kept = [c for c in plan.candidates if c["how"] in SETTLED and c["p_different"] < cfg.entity_split_min]
            if not kept:
                continue  # stays "create"; a namesake if identical names were all judged to be someone else
            best = max(kept, key=lambda c: c["p_same"])
            plan.variant, plan.action, plan.canonical = False, "link", best["with"]
            plan.entity_id = UUID(best["entity_id"])
            plan.merge = {**plan.merge, "with": best["with"], "entity_id": best["entity_id"], "how": HOW_IDENTICAL,
                          "p_same": best["p_same"], "p_different": best["p_different"], "kept_by_default": True}
            continue
        plan.variant = False
        plan.entity_id = other["id"]
        if plan.merge["how"] in SETTLED:
            plan.action, plan.canonical = "link", other["name"]
        else:
            plan.action = "alias"
            # The longer spelling becomes the display name ("Marcus" seen first, "Marcus Oyelaran" later).
            plan.canonical = plan.mention.name if len(plan.norm) > len(base(other["norm"])) else other["name"]
    return plans


def entity_decisions(ingest_id: Any, candidate: Candidate, plans: list[EntityPlan], scope: str,
                     subject: UUID | None) -> list[dict[str, Any]]:
    return [
        {
            "step": "entity", "question_version": ENTITY_VERSION, "ingest_id": ingest_id,
            "candidate_index": candidate.index, "subject_fact_id": subject, "scope": scope,
            "result": {"name": p.mention.name, "action": p.action, "variant": p.variant, **(p.merge or {}),
                       "candidates": p.candidates},
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
            if plan.variant:
                plan.entity_id = await repo.create_entity_variant(
                    conn, tenant, plan.mention.name, plan.norm, plan.mention.type, plan.embedding, plan.akas)
            else:
                plan.entity_id = await repo.upsert_entity(
                    conn, tenant, plan.mention.name, plan.norm, plan.mention.type, plan.embedding, plan.akas)
            plan.canonical = plan.mention.name
        elif plan.action == "alias" and plan.entity_id is not None:
            await repo.alias_entity(conn, plan.entity_id, plan.mention.name, plan.norm)
        if plan.entity_id is None:
            continue
        if plan.action != "create" and plan.akas:
            await repo.add_known_as(conn, plan.entity_id, plan.akas)
        await repo.link_entity(conn, fact_id, plan.entity_id, plan.mention.name)
        linked.append(plan.entity_id)
    return linked


async def reresolve(conn: AsyncConnection, judge: Judge, cfg: MemxConfig, tenant: str,
                    embed: Any = None) -> list[dict[str, Any]]:
    """Repair pass over stored entities. Each entity is paired with known entities it might be the same as (one name
    inside the other, a suggested alternative name, or a similar name embedding), Jev judges the pairs, and agreed
    pairs are merged: the entity with fewer facts folds into the one with more. ``embed`` (an async function from
    names to vectors) fills in missing name embeddings first. Returns what was asked and done."""
    if embed is not None:
        missing = await repo.entities_without_embedding(conn, tenant)
        if missing:
            for row, vec in zip(missing, await embed([r["name"] for r in missing])):
                await repo.set_entity_embedding(conn, row["id"], vec)
    rows = await repo.tenant_entities(conn, tenant)
    by_id = {r["id"]: r for r in rows}
    asks: list[MergeAsk] = []
    paired: set[frozenset] = set()
    for row in rows:
        options: list[tuple[dict[str, Any], str]] = []
        for other in rows:
            if other["id"] == row["id"] or base(other["norm"]) == base(row["norm"]):
                continue  # namesakes ("priya" and "priya~2") were already judged to be different
            if tokens_subset(row["norm"], other["norm"]) and len(base(other["norm"])) > len(base(row["norm"])):
                options.append((other, HOW_SUBSET))
            elif base(row["norm"]) in (other["known_as"] or []) or base(other["norm"]) in (row["known_as"] or []):
                options.append((other, HOW_AKA))
        for other in await repo.find_entities_similar_to(conn, tenant, row["id"], cfg.entity_name_sim_min, limit=3):
            if base(other["norm"]) != base(row["norm"]) and other["id"] not in {o["id"] for o, _ in options}:
                options.append((by_id.get(other["id"], other), HOW_SIMILAR))
        if sum(1 for _, how in options if how == HOW_SUBSET) > 1:
            options = [(o, how) for o, how in options if how != HOW_SUBSET]  # "Priya" with two Priya surnames
        for other, how in options[: cfg.entity_candidates_max]:
            key = frozenset((row["id"], other["id"]))
            if key in paired:
                continue
            paired.add(key)
            fact = await repo.entity_sample_facts(conn, row["id"], limit=1)
            asks.append(MergeAsk(row["name"], row["type"], fact[0] if fact else row["name"], other, len(options), how,
                                 short_id=row["id"]))
    answers, latency = await ask_merges(conn, judge, asks)
    report = []
    gone: set[Any] = set()
    async with conn.transaction():  # the merges of one pass land together or not at all
        await repo.lock_entities(conn, tenant)
        for ask, ans in zip(asks, answers):
            merged = ans["p_same"] >= cfg.entity_merge_min
            a = by_id[ask.short_id]
            b = ask.other
            if merged and a["id"] not in gone and b["id"] not in gone:
                counts = await repo.entity_fact_counts(conn, [a["id"], b["id"]])
                keep, fold = (b, a) if (counts.get(b["id"], 0), len(b["name"])) >= (counts.get(a["id"], 0), len(a["name"])) else (a, b)
                await repo.merge_entity(conn, fold["id"], keep["id"])
                gone.add(fold["id"])
                report.append({"short": fold["name"], "into": keep["name"], "how": ask.how,
                               "p_same": round(ans["p_same"], 4), "merged": True})
            else:
                report.append({"short": ask.short, "into": b["name"], "how": ask.how,
                               "p_same": round(ans["p_same"], 4), "merged": False})
        if report:
            await repo.add_decisions(conn, [{
                "step": "entity", "question_version": ENTITY_VERSION, "scope": tenant, "latency_ms": latency,
                "result": {"name": r["short"], "with": r["into"], "p_same": r["p_same"], "how": r["how"],
                           "action": "alias" if r["merged"] else "create", "reresolve": True},
            } for r in report])
    return report


__all__ = ["reresolve", "ask_merges", "MergeAsk", "EntityPlan", "plan_entities", "apply_entities",
           "entity_decisions", "normalize", "base", "tokens_subset", "tenant_of", "is_generic", "is_description"]
