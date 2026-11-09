"""Per-scope settings: a few server defaults that a scope may override for itself and everything below it.

Only settings that are safe to change while memory is in use are here. A scope inherits from its nearest
ancestor that sets a value, and falls back to the server default (``MEMX_*``) when none does, the same
downward-only inheritance facts use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field

from memx.config import MemxConfig
from memx.scopes import ScopePath


class ScopeSettings(BaseModel):
    """Overrides for one scope. ``None`` means "not set here; inherit"."""

    model_config = ConfigDict(extra="forbid")

    budget_tokens: int | None = Field(default=None, ge=200, le=200_000)
    event_ttl_days: int | None = Field(default=None, ge=0, le=3650)  # 0: never expire (None inherits)
    text_trust: float | None = Field(default=None, ge=0, le=1)
    walk_enabled: bool | None = None


# setting -> the MemxConfig field that supplies its server default
DEFAULTS_FROM = {
    "budget_tokens": "budget_tokens",
    "event_ttl_days": "event_ttl_days",
    "text_trust": "default_text_trust",
    "walk_enabled": "walk_enabled",
}
KEYS = tuple(DEFAULTS_FROM)


@dataclass(frozen=True)
class Resolved:
    scope: str
    values: dict[str, Any]  # effective value of every setting
    sources: dict[str, str]  # setting -> the scope that set it, or "default"
    own: dict[str, Any]  # what this scope sets itself

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def overridden(self) -> dict[str, dict[str, Any]]:
        return {k: {"value": self.values[k], "from": s} for k, s in self.sources.items() if s != "default"}

    def as_dict(self, cfg: MemxConfig) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "own": self.own,
            "effective": {k: {"value": self.values[k], "from": self.sources[k],
                              "default": getattr(cfg, DEFAULTS_FROM[k]),
                              "env": f"MEMX_{DEFAULTS_FROM[k].upper()}"} for k in KEYS},
        }


def _known(settings: dict[str, Any] | None) -> dict[str, Any]:
    return {k: v for k, v in (settings or {}).items() if k in DEFAULTS_FROM and v is not None}


async def resolve(conn: AsyncConnection, cfg: MemxConfig, scope: str) -> Resolved:
    path = ScopePath.parse(scope)
    chain = path.chain_str()
    cur = await conn.execute(
        "SELECT path::text AS path, settings FROM memx_scope WHERE path = ANY(%s::ltree[])", (chain,))
    by_path = {r["path"]: _known(r["settings"]) for r in await cur.fetchall()}
    values = {k: getattr(cfg, DEFAULTS_FROM[k]) for k in KEYS}
    sources = {k: "default" for k in KEYS}
    for p in chain:  # root first, so the nearest scope wins
        for k, v in by_path.get(p, {}).items():
            values[k], sources[k] = v, p
    return Resolved(scope=str(path), values=values, sources=sources, own=by_path.get(str(path), {}))


async def update(conn: AsyncConnection, scope: str, changes: dict[str, Any]) -> None:
    """Apply a partial update: a value sets the override, ``None`` removes it (inherit again).

    Unknown keys or out-of-range values raise ``pydantic.ValidationError``.
    """
    checked = ScopeSettings.model_validate(changes)
    sets = {k: getattr(checked, k) for k in checked.model_fields_set if getattr(checked, k) is not None}
    clears = [k for k in checked.model_fields_set if getattr(checked, k) is None]
    path = str(ScopePath.parse(scope))
    await conn.execute(
        """INSERT INTO memx_scope (path, settings) VALUES (%s::ltree, %s)
           ON CONFLICT (path) DO UPDATE SET settings = (memx_scope.settings - %s::text[]) || EXCLUDED.settings""",
        (path, Jsonb(sets), clears),
    )
