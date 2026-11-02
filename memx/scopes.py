from __future__ import annotations

import re
from dataclasses import dataclass

_SEGMENT = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_DEPTH = 16


class InvalidScope(ValueError):
    pass


@dataclass(frozen=True, order=True)
class ScopePath:
    """A dotted memory scope such as ``memory.acme.jyotish.agent1``.

    Recall at a scope sees facts at that scope and every ancestor; writes go to the scope itself.
    """

    labels: tuple[str, ...]

    @classmethod
    def parse(cls, value: str | ScopePath) -> ScopePath:
        if isinstance(value, ScopePath):
            return value
        if not isinstance(value, str) or not value:
            raise InvalidScope("scope must be a non-empty string")
        labels = tuple(value.split("."))
        if len(labels) > MAX_DEPTH:
            raise InvalidScope(f"scope deeper than {MAX_DEPTH} levels")
        for label in labels:
            if not _SEGMENT.match(label):
                raise InvalidScope(
                    f"invalid scope segment {label!r}: use 1-64 of A-Z a-z 0-9 _ -"
                )
        return cls(labels)

    def __str__(self) -> str:
        return ".".join(self.labels)

    @property
    def depth(self) -> int:
        return len(self.labels)

    @property
    def tenant(self) -> str:
        """The label under the root: `company_2` in `memory.company_2.user_3`. A bare root scope is its own tenant."""
        return self.labels[1] if len(self.labels) > 1 else self.labels[0]

    def chain(self) -> list[ScopePath]:
        """Ancestors first, self last."""
        return [ScopePath(self.labels[:i]) for i in range(1, len(self.labels) + 1)]

    def chain_str(self) -> list[str]:
        return [str(s) for s in self.chain()]

    def is_ancestor_of(self, other: ScopePath) -> bool:
        return len(self.labels) < len(other.labels) and other.labels[: len(self.labels)] == self.labels
