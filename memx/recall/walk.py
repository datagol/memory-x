"""The walk: expand ranked recall through shared entities and topics, at most two hops from the search hits.

Spreading activation with path tracking. A seed's search score flows along each of its entities to the other
visible facts that mention the entity, split evenly between them and halved per hop. Hubs (entities mentioned
by a large share of the visible facts) are not walked at all. Each reached fact keeps its strongest path, so
the result can be explained ("F5 → Marcus Oyelaran → F4") as well as scored. Seeds are never re-added, cycles
end at the hop limit, and only facts visible to the asking scope are nodes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection

from memx.config import MemxConfig
from memx.db import repo
from memx.types import Fact, PathStep


@dataclass
class Reached:
    fact: Fact
    score: float = 0.0
    best: float = 0.0
    path: list[PathStep] = field(default_factory=list)
    hop: int = 0
    kinds: set[str] = field(default_factory=set)
    pure: bool = False  # the kept path uses entities only; preferred over a stronger path through a theme

    @property
    def found_by(self) -> str:
        return "topic_hop" if self.kinds == {"topic"} else "entity_hop"


@dataclass
class WalkResult:
    expanded: list[Reached]  # strongest reached facts, capped
    info: dict[str, Any]
    ranked: list[Reached] = field(default_factory=list)  # every reached fact, strongest first


def hub_threshold(cfg: MemxConfig, visible: int) -> int:
    return max(cfg.walk_hub_min_degree, math.ceil(cfg.walk_hub_share * visible))


async def walk(
    conn: AsyncConnection, cfg: MemxConfig, *, chain: list[str], seeds: dict[UUID, float], visible: int,
    as_of: datetime | None, shadowed: set[UUID], topics_of: dict[UUID, str | None],
) -> WalkResult:
    """``seeds`` maps search-hit fact ids to scores in (0, 1]. ``topics_of`` gives each seed's topic.
    ``visible`` is the number of visible facts in the chain, used for the hub cut-off."""
    started = time.perf_counter()
    threshold = hub_threshold(cfg, visible)
    reached: dict[UUID, Reached] = {}
    hubs: dict[str, dict[str, Any]] = {}
    frontier: dict[UUID, float] = dict(seeds)
    frontier_paths: dict[UUID, list[PathStep]] = {s: [] for s in seeds}
    topic_of: dict[UUID, str | None] = dict(topics_of)
    hops_run = 0

    for hop in range(1, cfg.walk_hops + 1):
        if not frontier:
            break
        hops_run = hop
        ids = list(frontier)
        edges = await repo.entity_edges(conn, chain, ids, as_of)
        kept_entities: dict[UUID, dict[str, Any]] = {}
        by_fact: dict[UUID, list[dict[str, Any]]] = {}
        for e in edges:
            if int(e["degree"]) > threshold:
                hubs.setdefault(str(e["entity_id"]), {"name": e["name"], "type": e["type"], "degree": int(e["degree"])})
                continue
            kept_entities[e["entity_id"]] = e
            by_fact.setdefault(e["fact_id"], []).append(e)
        targets = await repo.facts_for_entities(conn, chain, list(kept_entities), as_of)
        by_entity: dict[UUID, list[dict[str, Any]]] = {}
        for row in targets:
            by_entity.setdefault(row["entity_id"], []).append(row)

        # Topic edges: the frontier's topics, weighted down and subject to the same hub cut.
        topics = sorted({t for f in ids for t in [topic_of.get(f)] if t})
        degrees = await repo.topic_degrees(conn, chain, topics, as_of)
        walk_topics = [t for t in topics if 0 < degrees.get(t, 0) <= threshold]
        for t in topics:
            if degrees.get(t, 0) > threshold:
                hubs.setdefault(f"topic:{t}", {"name": t, "type": "topic", "degree": degrees[t]})
        topic_rows = await repo.facts_for_topics(conn, chain, walk_topics, as_of)
        by_topic: dict[str, list[dict[str, Any]]] = {}
        for row in topic_rows:
            by_topic.setdefault(row["topic"], []).append(row)

        next_frontier: dict[UUID, float] = {}
        next_paths: dict[UUID, list[PathStep]] = {}

        def arrive(row: dict[str, Any], weight: float, step: PathStep, from_id: UUID) -> None:
            fid = row["id"]
            if fid in seeds or fid == from_id or fid in shadowed or weight <= 0:
                return
            node = reached.get(fid)
            if node is None:
                node = reached[fid] = Reached(fact=repo.to_fact(row), hop=hop)
                topic_of[fid] = row.get("topic")
            node.score += weight
            node.kinds.add(step.kind)
            path = frontier_paths.get(from_id, []) + [step]
            pure = all(s.kind == "entity" for s in path)
            # Keep the strongest path, but an all-entity path beats a stronger one that goes through a theme:
            # "via Marcus Oyelaran" explains more than "via career_and_role".
            if (pure and not node.pure) or (pure == node.pure and weight > node.best):
                node.best = weight
                node.path = path
                node.pure = pure
            if node.hop == hop and fid not in frontier:
                if weight > next_frontier.get(fid, 0.0):
                    next_frontier[fid] = weight
                    next_paths[fid] = node.path

        for from_id, act in frontier.items():
            for e in by_fact.get(from_id, []):
                others = max(1, int(e["degree"]) - 1)
                w = act * (1.0 / others) * cfg.walk_decay
                step = PathStep(from_fact=from_id, via=e["name"], kind="entity")
                for row in by_entity.get(e["entity_id"], []):
                    arrive(row, w, step, from_id)
            t = topic_of.get(from_id)
            if t and t in by_topic:
                others = max(1, degrees.get(t, 1) - 1)
                w = act * (1.0 / others) * cfg.walk_decay * cfg.walk_topic_weight
                step = PathStep(from_fact=from_id, via=t, kind="topic")
                for row in by_topic[t]:
                    arrive(row, w, step, from_id)

        frontier = next_frontier
        frontier_paths = next_paths

    ranked = sorted(reached.values(), key=lambda r: (-r.score, str(r.fact.id)))
    expanded = ranked[: cfg.walk_cap]
    info = {
        "enabled": True, "seeds": len(seeds), "hops": hops_run, "reached": len(reached), "expanded": len(expanded),
        "hub_threshold": threshold, "hubs_cut": sorted(hubs.values(), key=lambda h: -h["degree"]),
        "over_cap": len(ranked) - len(expanded), "ms": int((time.perf_counter() - started) * 1000),
    }
    return WalkResult(expanded=expanded, info=info, ranked=ranked)
