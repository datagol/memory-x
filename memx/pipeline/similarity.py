from __future__ import annotations

import hashlib
import math
import re

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_NUMBER = re.compile(r"\d+(?:[.,:/-]\d+)*")
NEGATIONS = frozenset({
    "not", "never", "no", "without", "stopped", "former", "formerly", "none", "nobody", "nothing",
    "neither", "nor", "cannot", "dislikes", "dislike", "hates", "hate", "longer", "quit", "ex",
})


def normalize(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .!?;:,")


def text_hash(text: str) -> bytes:
    return hashlib.sha256(normalize(text).encode()).digest()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("’", "'"))


def _negated(text: str) -> bool:
    lowered = text.lower().replace("’", "'")
    if "n't" in lowered or re.search(r"\bex-", lowered):
        return True
    return any(t in NEGATIONS for t in _tokens(lowered))


STOPWORDS = frozenset({
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or", "is", "are", "was", "be", "it", "its",
    "their", "they", "them", "this", "that", "with", "as", "by", "very", "really", "lot", "lots", "much",
})


def lexical_guard(candidate: str, existing: str, min_jaccard: float = 0.8) -> bool:
    """True when a candidate is safe to treat as a restatement of an existing fact without asking the judge.

    Embeddings score "likes coffee" and "does not like coffee" as near-identical, and "lives in Pune, Maharashtra"
    as near-identical to "lives in Pune". So a match also needs high word overlap, no new content words in the
    candidate, the same negation polarity and the same numbers.
    """
    ta, tb = set(_tokens(candidate)) - STOPWORDS, set(_tokens(existing)) - STOPWORDS
    if not ta or not tb:
        return False
    if ta - tb:
        return False
    if len(ta & tb) / len(ta | tb) < min_jaccard:
        return False
    if _negated(candidate) != _negated(existing):
        return False
    if set(_NUMBER.findall(candidate)) != set(_NUMBER.findall(existing)):
        return False
    return True


# A statement about an earlier state ("the previous channel was #ops") must not retire the current one.
_PAST = re.compile(
    r"\b(previous(ly)?|formerly|used to|originally|before (that|switching|moving|the change)"
    r"|old (value|channel|address|number|email|role|status))\b"
)
# "is used to cold weather" (accustomed) and "is used to deploy" (purpose) are present tense.
_NOT_PAST = re.compile(r"\b(is|are|am|be|been|being|was|were|get|gets|got|getting)\s+used to\b")
# ...unless it also states the new value: "moved from Porto to Lisbon", "previously X, now Y".
_PRESENT = re.compile(r"\b(now|currently|instead|switched to|moved to|changed to|replaced (by|with))\b|\bfrom\b.+\bto\b")


def is_past_statement(text: str) -> bool:
    """True when a statement is only about an earlier state, not the current one."""
    lowered = _NOT_PAST.sub(" ", text.lower().replace("’", "'"))
    return bool(_PAST.search(lowered)) and not _PRESENT.search(lowered)


_ATTR_STOP = STOPWORDS | {"user", "users", "current", "new", "old", "previous", "personal", "detail", "details",
                          "info", "information", "general"}


def _forms(word: str) -> set[str]:
    """The word plus its possible singulars, so "classes"/"class", "statuses"/"status" and "houses"/"house" meet."""
    forms = {word}
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        forms.add(word[:-1])
        if word.endswith("es"):
            forms.add(word[:-2])
        if word.endswith("ies"):
            forms.add(word[:-3] + "y")
    return forms


def _attr_words(label: str) -> set[str]:
    return {form for t in _tokens(label) if t not in _ATTR_STOP for form in _forms(t)}


def attributes_compatible(a: str | None, b: str | None) -> bool:
    """False only when both facts name an attribute and the labels share no content word.

    Different words can still name one attribute ("location" / "city of residence"), so reconcile asks Jev before
    acting on a False here; see ``reconcile.same_attribute_question``.

    The extractor names one attribute per fact ("frequent flyer status", "reading progress"). Two facts about
    different attributes ("page count" vs "reading progress") can both be true, so neither may replace the
    other. Labels are free text, so any shared word ("status" / "frequent flyer status") counts as compatible,
    and a missing label never blocks anything.
    """
    if not a or not b:
        return True
    ta, tb = _attr_words(a), _attr_words(b)
    if not ta or not tb:
        return True
    return bool(ta & tb)
