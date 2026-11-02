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
