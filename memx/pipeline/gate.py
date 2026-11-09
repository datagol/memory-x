"""Keep chit-chat out of extraction.

One Jev request per window asks, for each turn, whether it says anything worth remembering. Turns Jev is sure say
nothing ("thanks!", "Sure, happy to help.") are left out of the extraction call, which is billed per token in and
out. A turn right before a kept turn is always kept, because it may give that turn its subject ("What's your guest's
name?" / "Marisol."). A window with nothing left is not extracted.
"""

from __future__ import annotations

from memx.pipeline.windowing import Window
from memx.providers.jev import UNTRUSTED_NOTE, Judge, Question, YesNo, clip, envelope

GATE_VERSION = "gate.v1"
GATE_QUESTION = (
    "Does `untrusted.turns[{i}]` state anything worth remembering about the user, other people, their work, plans, "
    "preferences, events, or advice, recommendations or answers given to the user? Greetings, thanks, "
    "acknowledgements and questions that reveal nothing are not worth remembering. " + UNTRUSTED_NOTE
)
MIN_UNITS = 2  # a window this small is not worth a gate request


def gate_questions(n: int) -> dict[str, Question]:
    return {f"t{i}": YesNo(GATE_QUESTION.format(i=i)) for i in range(n)}


async def gate_window(judge: Judge, window: Window, skip_below: float) -> tuple[Window | None, dict]:
    """Return the window with chit-chat turns removed (None when nothing is left) and a record for the log."""
    if window.kind != "messages" or len(window.units) < MIN_UNITS or len(window.units) > 100:
        return window, {}
    state = envelope({"roles": [u.role for u in window.units]},
                     {"turns": [clip(u.text, 1500) for u in window.units]})
    res = await judge.ask(state, gate_questions(len(window.units)))
    keep_p = [_prob(res.answers.get(f"t{i}")) for i in range(len(window.units))]
    keep = [p >= skip_below for p in keep_p]
    for i in range(len(keep) - 1):  # the turn before a kept turn stays: it may hold that turn's subject
        if keep[i + 1]:
            keep[i] = True
    kept = [u for u, k in zip(window.units, keep) if k]
    record = {"window": window.index, "keep": [round(p, 3) for p in keep_p], "skipped": len(window.units) - len(kept),
              "latency_ms": res.latency_ms}
    if not kept:
        return None, record
    if len(kept) == len(window.units):
        return window, record
    return Window(window.index, window.kind, kept, window.context, window.source), record


def _prob(answer: dict | None) -> float:
    """Jev's yes-probability; a missing or empty answer keeps the turn (fail open: never lose a fact to the gate)."""
    value = (answer or {}).get("noul")
    try:
        return 1.0 if value is None else float(value)
    except (TypeError, ValueError):
        return 1.0
