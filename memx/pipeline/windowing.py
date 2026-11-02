"""Split ingest input into extraction windows.

Single-call extraction is reliable only up to a few thousand tokens, so long inputs are cut into windows of
roughly ``window_tokens``. Conversations split on turn boundaries; documents on paragraph boundaries. Each
window carries a little read-only context from before it so pronouns and references resolve.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from memx.types import Messages, Text

CONTEXT_TURN_CHARS = 500
TEXT_CONTEXT_CHARS = 540  # ~150 tokens


@dataclass
class Unit:
    role: str
    text: str
    at: datetime | None = None


@dataclass
class Window:
    index: int
    kind: str  # "messages" | "text"
    units: list[Unit]
    context: list[Unit] = field(default_factory=list)
    source: str | None = None


def _split_long(text: str, max_chars: int) -> list[str]:
    """Break text into pieces no longer than max_chars, preferring paragraph then sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            pieces.append(para)
            continue
        buf = ""
        for sentence in re.split(r"(?<=[.!?])\s+", para):
            while len(sentence) > max_chars:
                if buf:
                    pieces.append(buf)
                    buf = ""
                pieces.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            if len(buf) + len(sentence) + 1 > max_chars and buf:
                pieces.append(buf)
                buf = sentence
            else:
                buf = f"{buf} {sentence}".strip()
        if buf:
            pieces.append(buf)
    # Merge tiny neighbours back together so we do not create needless windows.
    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) + len(piece) + 2 <= max_chars:
            merged[-1] = f"{merged[-1]}\n\n{piece}"
        else:
            merged.append(piece)
    return merged


def build_units(inp: Messages | Text, max_unit_chars: int) -> list[Unit]:
    units: list[Unit] = []
    if isinstance(inp, Messages):
        for turn in inp.turns:
            role = turn.name or turn.role
            for piece in _split_long(turn.content.strip(), max_unit_chars):
                if piece:
                    units.append(Unit(role=role, text=piece, at=turn.at))
    else:
        for para in re.split(r"\n\s*\n", inp.text):
            para = para.strip()
            if para:
                for piece in _split_long(para, max_unit_chars):
                    units.append(Unit(role="document", text=piece))
    return units


def _context_for(kind: str, previous: list[Unit], context_turns: int) -> list[Unit]:
    if not previous:
        return []
    if kind == "messages":
        return [
            Unit(role=u.role, text=u.text[-CONTEXT_TURN_CHARS:], at=u.at)
            for u in previous[-context_turns:]
        ] if context_turns > 0 else []
    tail = previous[-1].text[-TEXT_CONTEXT_CHARS:]
    return [Unit(role="document", text=tail)]


def make_windows(
    inp: Messages | Text,
    *,
    window_tokens: int,
    context_turns: int,
    chars_per_token: float,
) -> list[Window]:
    kind = inp.kind
    max_chars = int(window_tokens * chars_per_token)
    units = build_units(inp, max_chars)
    source = inp.source if isinstance(inp, Text) else None
    windows: list[Window] = []
    current: list[Unit] = []
    size = 0
    consumed: list[Unit] = []
    for unit in units:
        if current and size + len(unit.text) > max_chars:
            windows.append(Window(len(windows), kind, current, _context_for(kind, consumed, context_turns), source))
            consumed.extend(current)
            current, size = [], 0
        current.append(unit)
        size += len(unit.text)
    if current:
        windows.append(Window(len(windows), kind, current, _context_for(kind, consumed, context_turns), source))
    return windows


def split_window(window: Window, context_turns: int) -> tuple[Window, Window]:
    """Halve a window whose extraction hit the candidate cap."""
    mid = len(window.units) // 2
    first = Window(window.index, window.kind, window.units[:mid], window.context, window.source)
    second = Window(
        window.index, window.kind, window.units[mid:],
        _context_for(window.kind, window.context + window.units[:mid], context_turns), window.source,
    )
    return first, second
