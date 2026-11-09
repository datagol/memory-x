"""Deterministic stand-ins for the model providers, for tests and local development without API keys."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

from memx.pipeline.entities import MERGE_OPTIONS, tokens_subset
from memx.pipeline.reconcile import VERDICTS
from memx.pipeline.windowing import Window
from memx.providers.extractor import ExtractResult
from memx.providers.jev import Choice, JudgeResult, Question, Score, YesNo
from memx.types import CATEGORY_DEFINITIONS, TOPIC_DEFINITIONS

_WORD = re.compile(r"[a-z0-9]+")
# Capitalized runs ("Marcus Oyelaran", "Data Platform Team") and code-like names ("ARCHIVE-02", "memory-x").
_NAME = re.compile(r"(?:[A-Z][\w-]*(?:\s+[A-Z][\w-]*)*)|(?:\b[A-Z]{2,}[\w-]*\d[\w-]*\b)")
_NAME_STOP = frozenset("The A An I It They We You He She My Our Their This That These Those User Item Saw Asked "
                       "Prefers Told Met Likes Uses Works Legacy Moved".split())
_ORG_WORDS = ("inc", "ltd", "works", "corp", "bio", "labs", "group")


def guess_entities(sentence: str, overrides: dict[str, tuple] | None = None) -> list[dict[str, Any]]:
    """Deterministic stand-in for the extractor's entity list: capitalized runs that are not sentence starts.
    ``overrides`` maps a phrase to (name, type) or (name, type, [alternative names])."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key, spec in (overrides or {}).items():
        name, etype, aka = spec[0], spec[1], list(spec[2]) if len(spec) > 2 else []
        if key.lower() in sentence.lower() and name.lower() not in seen:
            seen.add(name.lower())
            out.append({"name": name, "type": etype, "aka": aka})
    for m in _NAME.finditer(sentence):
        name = m.group(0).strip()
        words = name.split()
        if m.start() == 0:
            if len(words) == 1:  # a lone capitalized sentence start is just a sentence start
                continue
            if words[0] in _NAME_STOP:
                words = words[1:]
                name = " ".join(words)
        if not name or words[0] in _NAME_STOP or name.lower() in seen:
            continue
        if len(words) == 1 and (len(name) < 3 or (name.isupper() and not any(ch.isdigit() for ch in name))):
            continue  # too short, or an acronym such as CTO
        seen.add(name.lower())
        lowered = name.lower()
        etype = ("team" if "team" in lowered else "organization" if any(w in lowered for w in _ORG_WORDS)
                 else "system" if any(ch.isdigit() for ch in name) or "-" in name else "person" if len(words) == 2 else "other")
        out.append({"name": name, "type": etype, "aka": []})
    return out[:6]


def _words(text: str) -> list[str]:
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _WORD.findall(text.lower())]


class HashEmbedder:
    """Bag-of-words hashed into a fixed-size unit vector. Shared words mean higher cosine similarity."""

    model = "hash-test"

    def __init__(self, dim: int = 1536):
        self.dim = dim
        self.calls = 0

    def vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in _words(text):
            digest = hashlib.sha1(word.encode()).digest()
            vec[int.from_bytes(digest[:4], "big") % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out: list[float] = [v / norm for v in vec]
        out[0] = 0 if out[0] == 0 else out[0]  # mimic providers that send integer zeros
        return out

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self.vector(t) for t in texts]


class SentenceExtractor:
    """Treats every sentence of every unit as one fact. Optional ``hints`` map a substring to a category."""

    model = "sentence-test"

    def __init__(self, hints: dict[str, str] | None = None, event_dates: dict[str, str] | None = None,
                 entities: dict[str, tuple[str, str]] | None = None, attributes: dict[str, str] | None = None):
        self.hints = hints or {}
        self.attributes = attributes or {}
        self.event_dates = event_dates or {}
        self.entities = entities or {}
        self.windows_seen: list[Window] = []

    async def extract(self, window: Window, *, observed_at: datetime, total_windows: int) -> ExtractResult:
        self.windows_seen.append(window)
        facts = []
        for unit in window.units:
            for sentence in re.split(r"(?<=[.!?])\s+", unit.text.strip()):
                sentence = sentence.strip()
                if not sentence:
                    continue
                hint = next((c for key, c in self.hints.items() if key.lower() in sentence.lower()), None)
                date = next((d for key, d in self.event_dates.items() if key.lower() in sentence.lower()), None)
                attribute = next((a for key, a in self.attributes.items() if key.lower() in sentence.lower()), None)
                facts.append({
                    "attribute": attribute,
                    "text": sentence, "speaker": unit.role, "category_hint": hint, "is_event": bool(date),
                    "time_text": None, "event_date": date, "confidence": 1.0,
                    "entities": guess_entities(sentence, self.entities),
                })
        return ExtractResult(facts=facts[:25], tokens_in=len(window.units), tokens_out=len(facts))


CHITCHAT = {"thanks", "thank you", "ok", "okay", "great", "hi", "hello", "sure", "you're welcome", "bye"}
HISTORY_WORDS = ("previous", "earlier", "before", "used to", "originally", "first", "changed")

VerdictFn = Callable[[str, str], tuple[str, float]]
GradeFn = Callable[[str, str], float]
TopicFn = Callable[[str], tuple[str, float]]
MergeFn = Callable[..., float]


def overlap_grade(query: str, fact: str) -> float:
    q, f = set(_words(query)), set(_words(fact))
    return 4.0 * len(q & f) / max(1, len(q))


DEMO_TOPIC_HINTS: dict[str, str] = {
    "food_and_drink": "food eat chicken seafood coffee tea espresso restaurant dinner lunch vegetarian wine beer",
    "health_and_fitness": "gym run marathon doctor sleep health yoga weight",
    "family_and_home": "wife husband partner kid child son daughter dog cat house apartment family",
    "travel": "trip flight travel hotel visit commute holiday vacation",
    "hobbies_and_leisure": "hobby chess guitar hiking climbing football game paint",
    "arts_and_media": "book film movie music podcast album novel",
    "finance": "budget salary invoice price cost contract renew pay dollar",
    "career_and_role": "cto ceo engineer manager role title hired promoted job lead",
    "people_and_relationships": "reports colleague friend client works with married",
    "communication_style": "answer reply tone concise formal spanish english format",
    "scheduling_and_availability": "meeting pto timezone available deadline schedule week",
    "leadership_and_decisions": "approved decided owns ownership decision strategy sign-off",
    "software_development": "repo code python typescript deploy test release bug",
    "data_and_analytics": "data dataset database dashboard pipeline metric migration",
    "cloud_and_infrastructure": "azure aws gcp cloud server cluster kubernetes archive on-prem hosting",
    "security_and_compliance": "access permission policy gdpr audit security",
    "product_and_design": "feature roadmap design ux theme dark mode light mode",
    "documents_and_knowledge": "document file notes wiki archive",
    "location_and_environment": "lives office city country moved",
    "goals_and_plans": "want plan goal trying intend",
}


def demo_topic(text: str) -> tuple[str, float]:
    words = set(_words(text))
    best, hits = "general", 0
    for topic, keys in DEMO_TOPIC_HINTS.items():
        n = len(words & set(keys.split()))
        if n > hits:
            best, hits = topic, n
    return (best, 0.7) if hits else ("general", 0.3)


def subset_merge(a: str, b: str) -> float:
    """Default merge rule: the same name, or a name that is a word-subset of the other, reads as the same thing."""
    from memx.pipeline.entities import normalize
    if normalize(a) == normalize(b):
        return 0.9
    return 0.85 if tokens_subset(normalize(a), normalize(b)) else 0.1


class FakeJudge:
    """Answers Jev questions with simple rules. Override ``verdict``, ``grade``, ``topic`` or ``merge`` per test.

    Grading is path-aware like the real question: a fact reached through an entity hop is judged against the
    query together with the fact it was reached from."""

    def __init__(self, verdict: VerdictFn | None = None, grade: GradeFn | None = None,
                 injection_markers: tuple[str, ...] = ("ignore previous", "ignore all", "system prompt"),
                 topic: TopicFn | None = None, merge: MergeFn | None = None,
                 coexist: Callable[[str, str], float] | None = None,
                 same_attribute: Callable[[str, str], float] | None = None):
        self.verdict = verdict or (lambda candidate, neighbor: ("related", 0.9))
        self.grade = grade or overlap_grade
        self.topic = topic or demo_topic
        self.merge = merge or subset_merge
        self.coexist = coexist or (lambda candidate, neighbor: 0.9)
        # (candidate label, neighbor label) -> P(same attribute); by default differently worded labels differ
        self.same_attribute = same_attribute or (lambda a, b: 0.1)
        self.injection_markers = injection_markers
        self.requests: list[tuple[dict[str, Any], dict[str, Question]]] = []

    async def ask(self, state: dict[str, Any], questions: dict[str, Question]) -> JudgeResult:
        self.requests.append((state, questions))
        ctx, untrusted = state.get("judge_context", {}), state.get("untrusted", {})
        answers: dict[str, dict[str, Any]] = {}
        for name, q in questions.items():
            if isinstance(q, Choice) and set(q.criteria) == set(CATEGORY_DEFINITIONS):
                choice = ctx.get("extractor_hint") or "other"
                answers[name] = {"type": "choice", "choice": choice, "confidence": 0.95,
                                 "probabilities": {k: (0.95 if k == choice else 0.05 / 8) for k in q.criteria}}
            elif isinstance(q, Choice) and set(q.criteria) == set(TOPIC_DEFINITIONS):
                choice, p = self.topic(str(untrusted.get("candidate", "")))
                rest = (1 - p) / max(1, len(q.criteria) - 1)
                answers[name] = {"type": "choice", "choice": choice, "confidence": p,
                                 "probabilities": {k: (p if k == choice else rest) for k in q.criteria}}
            elif isinstance(q, Choice) and set(q.criteria) == set(MERGE_OPTIONS):
                i = int(name[1:])
                pair = untrusted["pairs"][i]
                # A merge rule may take (a, b) or (a, b, pair) to look at the facts as well as the names.
                p = self.merge(pair["a"], pair["b"]) if self.merge.__code__.co_argcount == 2 else self.merge(
                    pair["a"], pair["b"], pair)
                choice = "same" if p >= 0.5 else "different"
                answers[name] = {"type": "choice", "choice": choice, "confidence": max(p, 1 - p),
                                 "probabilities": {"same": p, "different": 1 - p - 0.05, "cannot_tell": 0.05}}
            elif isinstance(q, Choice) and set(q.criteria) == set(VERDICTS):
                i = int(name[1:])
                label, p = self.verdict(untrusted["candidate"], untrusted["neighbors"][i])
                answers[name] = {"type": "choice", "choice": label, "confidence": p,
                                 "probabilities": {k: (p if k == label else (1 - p) / 4) for k in q.criteria}}
            elif isinstance(q, YesNo) and "worth remembering" in q.instructions:
                i = int(name[1:])
                turn = str(untrusted["turns"][i]).strip().lower().rstrip("!.")
                answers[name] = {"type": "noul", "noul": 0.02 if turn in CHITCHAT else 0.95}
            elif isinstance(q, YesNo) and "earlier" in q.instructions and "query" in q.instructions:
                query = str(untrusted.get("query", "")).lower()
                answers[name] = {"type": "noul", "noul": 0.95 if any(w in query for w in HISTORY_WORDS) else 0.03}
            elif isinstance(q, YesNo) and "name the attribute each one gives" in q.instructions:
                i = int(name[1:])
                answers[name] = {"type": "noul", "noul": self.same_attribute(
                    untrusted["candidate_attribute"], untrusted["neighbor_attributes"][i])}
            elif isinstance(q, YesNo) and "both be true" in q.instructions:
                i = int(name[1:])
                answers[name] = {"type": "noul", "noul": self.coexist(untrusted["candidate"], untrusted["neighbors"][i])}
            elif isinstance(q, YesNo):
                text = str(untrusted.get("candidate", "")).lower()
                answers[name] = {"type": "noul", "noul": 0.95 if any(m in text for m in self.injection_markers) else 0.02}
            elif isinstance(q, Score):
                i = int(name[1:].split("_")[0])
                fact = untrusted["facts"][i]
                score = self.grade(untrusted.get("query", ""), fact)
                meta = (ctx.get("facts") or [{}] * (i + 1))[i] if i < len(ctx.get("facts") or []) else {}
                if meta.get("from_fact") is not None and "linked to the fact" in q.instructions:
                    # judged with its link in view: as relevant as the linked fact, at most
                    bridge = untrusted["facts"][int(meta["from_fact"])]
                    score = max(score, self.grade(bridge, fact))
                answers[name] = {"type": "score", "score": min(score, len(q.criteria) - 1)}
            else:
                raise NotImplementedError(f"FakeJudge cannot answer {name}")
        return JudgeResult(answers=answers, latency_ms=1, model="fake")


DEMO_HINTS = {
    "prefer": "preference", "likes": "preference", "love": "preference", "hate": "preference", "enjoy": "preference",
    "never": "rule", "always": "rule", "must": "rule", "not allowed": "rule",
    "manager": "relationship", "reports to": "relationship", "works with": "relationship", "client": "relationship",
    "pto": "event", "meeting": "event", "yesterday": "event", "tomorrow": "event", "next week": "event",
    "project": "work_context", "repo": "work_context", "uses": "work_context", "working on": "work_context",
    "deploy": "work_context", "postgres": "work_context",
    "my name": "identity", "lives in": "identity", "i am": "identity", "i'm a": "identity", "lead": "identity",
    "want to": "goal", "goal": "goal", "plan to": "goal", "trying to": "goal",
    "steps": "procedure", "first,": "procedure", "then ": "procedure",
}


def demo_verdict(candidate: str, neighbor: str) -> tuple[str, float]:
    """Rough stand-in for Jev's verdict: heavy overlap with a changed word reads as a replacement."""
    a, b = set(_words(candidate)), set(_words(neighbor))
    overlap = len(a & b) / max(1, len(a | b))
    if overlap >= 0.95:
        return "same", 0.9
    if overlap >= 0.55:
        return "candidate_replaces", 0.78
    if overlap >= 0.25:
        return "related", 0.7
    return "unrelated", 0.85


_STOP = frozenset("a an the of to in on at for and or is are was be it its their they this that with as by do does "
                  "did what which who whom how when where why should would could can will i me my you your user".split())


def demo_grade(query: str, fact: str) -> float:
    """Content-word overlap on the 0-4 grade scale, ignoring stopwords."""
    q = {w for w in _words(query) if w not in _STOP}
    f = {w for w in _words(fact) if w not in _STOP}
    return min(4.0, 6.0 * len(q & f) / max(1, len(q)))


def demo_providers(dim: int = 1536) -> tuple[SentenceExtractor, HashEmbedder, FakeJudge]:
    return (SentenceExtractor(hints=DEMO_HINTS), HashEmbedder(dim),
            FakeJudge(verdict=demo_verdict, grade=demo_grade, topic=demo_topic))
