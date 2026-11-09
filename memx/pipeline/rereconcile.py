"""Re-apply the current reconcile policy to replacements already made.

Every replacement and shadow was decided by a logged Jev verdict. When the policy changes (higher floors, new guards),
those verdicts can be re-read under the new rules without calling Jev or re-extracting anything: a replacement the
new policy would not make is undone (the retired fact becomes active again and is linked as related), and a shadow it
would not make is removed. Each undo is logged as a ``rereconcile`` decision pointing at the original verdict.

Replacements with no logged verdict between the two facts (exact or near-exact repeats, or a repeat that retired a
child fact without being stored itself) are left alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from psycopg import AsyncConnection

from memx.config import MemxConfig
from memx.db import repo
from memx.pipeline.reconcile import VERDICT_VERSION, Verdict, effective_label
from memx.types import Candidate

REREC_VERSION = "rereconcile.v1"

_FACT = """{p}.id AS {p}_id, {p}.text AS {p}_text, {p}.scope::text AS {p}_scope, {p}.speaker AS {p}_speaker,
           {p}.ingest_id AS {p}_ingest, (({p}.attrs)->>'window')::int AS {p}_window,
           ({p}.attrs)->>'attribute' AS {p}_attribute"""

# Retired facts and the logged verdict that retired them. "r" replaced "f" (candidate_replaces, or same for a child
# fact), or "f" arrived already covered by "r" (neighbor_covers).
_SUPERSEDED = f"""
    SELECT {_FACT.format(p='f')}, {_FACT.format(p='r')}, d.id AS decision_id, d.result, d.similarity,
           d.subject_fact_id = f.id AS arrived_covered
      FROM memx_fact f
      JOIN memx_fact r ON r.id = f.superseded_by
      JOIN LATERAL (
            SELECT * FROM memx_decision d
             WHERE d.step = 'verdict'
               AND ((d.subject_fact_id = r.id AND d.object_fact_id = f.id
                     AND d.result->>'label' IN ('candidate_replaces', 'same'))
                 OR (d.subject_fact_id = f.id AND d.object_fact_id = r.id
                     AND d.result->>'label' = 'neighbor_covers'))
             ORDER BY d.created_at DESC LIMIT 1) d ON true
     WHERE f.status = 'superseded' AND (%(scope)s::ltree IS NULL OR f.scope <@ %(scope)s::ltree)
     ORDER BY f.superseded_at DESC, f.id
"""
# Newest replacements first: in a chain Austin -> Boston -> Chicago, Boston's case is settled before Austin's, and
# Austin may only come back while the fact that replaced it is still current.

# The other verdicts logged when a fact arrived already covered. They were never applied (a covered fact is stored
# as history), so restoring it must not leave it active beside a fact it would now replace.
_OTHER_VERDICTS = """
    SELECT d.result, d.similarity, o.id, o.text, o.scope::text AS scope, o.ingest_id,
           (o.attrs->>'window')::int AS extract_window, o.attrs->>'attribute' AS attribute
      FROM memx_decision d JOIN memx_fact o ON o.id = d.object_fact_id
     WHERE d.step = 'verdict' AND d.subject_fact_id = %(f)s AND d.object_fact_id <> %(r)s AND o.status = 'active'
"""

# An identical active copy anywhere this fact is visible from (its scope or a parent) makes a restore a duplicate.
_DUPLICATE = """
    SELECT 1 FROM memx_fact f JOIN memx_fact o ON o.text_hash = f.text_hash AND o.status = 'active'
                                            AND o.scope @> f.scope AND o.id <> f.id
     WHERE f.id = %s LIMIT 1
"""

# Shadow links ("r" hides parent fact "f" below r's scope) and the verdict behind each.
_SHADOWS = f"""
    SELECT {_FACT.format(p='f')}, {_FACT.format(p='r')}, d.id AS decision_id, d.result, d.similarity,
           false AS arrived_covered
      FROM memx_fact_link l
      JOIN memx_fact r ON r.id = l.a
      JOIN memx_fact f ON f.id = l.b
      JOIN LATERAL (
            SELECT * FROM memx_decision d
             WHERE d.step = 'verdict' AND d.subject_fact_id = r.id AND d.object_fact_id = f.id
               AND d.result->>'label' = 'candidate_replaces'
             ORDER BY d.created_at DESC LIMIT 1) d ON true
     WHERE l.kind = 'shadows' AND (%(scope)s::ltree IS NULL OR r.scope <@ %(scope)s::ltree)
"""


@dataclass
class RereconcileReport:
    checked: int = 0
    restored: int = 0
    unshadowed: int = 0
    kept: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)  # undos the policy allows but that would be wrong now
    changes: list[dict[str, Any]] = field(default_factory=list)


def _fact(row: dict[str, Any], p: str) -> dict[str, Any]:
    return {"id": row[f"{p}_id"], "text": row[f"{p}_text"], "scope": row[f"{p}_scope"],
            "speaker": row[f"{p}_speaker"], "ingest_id": row[f"{p}_ingest"],
            "extract_window": row[f"{p}_window"], "attribute": row[f"{p}_attribute"]}


def recheck(row: dict[str, Any], cfg: MemxConfig) -> tuple[str, str | None, Verdict]:
    """Re-read one logged verdict under the current policy. Returns (label it had, reason to undo or None, verdict)."""
    replaced, replacer = _fact(row, "f"), _fact(row, "r")
    # The candidate is whichever fact was being inserted when the verdict was asked.
    cand, neighbor = (replaced, replacer) if row["arrived_covered"] else (replacer, replaced)
    result = row["result"]
    # Follow-up answers logged with the verdict count too, or a confirmed replacement would be undone as below_floor.
    verdict = _verdict_from(result, neighbor, row["similarity"])
    candidate = Candidate(text=cand["text"], speaker=cand["speaker"] or "user", attribute=cand["attribute"],
                          window=cand["extract_window"] if cand["extract_window"] is not None else -1)
    acted, reason = effective_label(verdict, cfg, candidate, cand["ingest_id"])
    return result["label"], (reason if acted != result["label"] else None), verdict


def _verdict_from(result: dict[str, Any], neighbor: dict[str, Any], similarity: float) -> Verdict:
    verdict = Verdict(neighbor, result["label"], float(result.get("probability") or 0.0), float(similarity or 0.0),
                      "jev", {k: float(v) for k, v in (result.get("probabilities") or {}).items()})
    if result.get("coexist") is not None:
        verdict.coexist = float(result["coexist"])
    if result.get("same_attribute") is not None:
        verdict.same_attribute = float(result["same_attribute"])
    return verdict


async def _blocked(conn: AsyncConnection, cfg: MemxConfig, row: dict[str, Any], kind: str,
                   planned: set[tuple[str, str]]) -> str | None:
    """Why this undo must not happen now, or None. Checked in dry runs too, so they report what a real run does;
    ``planned`` holds (scope, text) of restores a dry run has counted but not made."""
    if kind == "supersede":
        cur = await conn.execute("SELECT status FROM memx_fact WHERE id = %s", (row["r_id"],))
        replacer = await cur.fetchone()
        if replacer is None or replacer["status"] != "active":
            return "replacer_not_current"  # restoring would bring back a value two changes old
        cur = await conn.execute(_DUPLICATE, (row["f_id"],))
        labels = row["f_scope"].split(".")
        if await cur.fetchone() or any((".".join(labels[:i]), row["f_text"]) in planned
                                       for i in range(1, len(labels) + 1)):
            return "duplicate"
    if row["arrived_covered"]:
        cand = _fact(row, "f")
        candidate = Candidate(text=cand["text"], speaker=cand["speaker"] or "user", attribute=cand["attribute"],
                              window=cand["extract_window"] if cand["extract_window"] is not None else -1)
        cur = await conn.execute(_OTHER_VERDICTS, {"f": row["f_id"], "r": row["r_id"]})
        for other in await cur.fetchall():
            neighbor = {k: other[k] for k in ("id", "text", "scope", "ingest_id", "extract_window", "attribute")}
            acted, _ = effective_label(_verdict_from(other["result"], neighbor, other["similarity"]), cfg,
                                       candidate, cand["ingest_id"])
            if acted in ("candidate_replaces", "same"):
                return "would_replace_other"
    return None


def _decision(row: dict[str, Any], action: str, label: str, reason: str, verdict: Verdict,
              cfg: MemxConfig) -> dict[str, Any]:
    return {
        "step": "rereconcile", "question_version": REREC_VERSION, "subject_fact_id": row["r_id"],
        "object_fact_id": row["f_id"], "scope": row["f_scope"], "similarity": verdict.similarity,
        "result": {"action": action, "verdict_decision_id": str(row["decision_id"]), "label": label,
                   "probability": verdict.probability, "reason": reason, "verdict_version": VERDICT_VERSION,
                   "policy": {"same_min": cfg.verdict_same_min, "replace_min": cfg.verdict_replace_min,
                              "covers_min": cfg.verdict_covers_min, "min_prob": cfg.verdict_min_prob,
                              "guards": cfg.verdict_guards},
                   "fact": row["f_text"], "by": row["r_text"]},
    }


async def rereconcile(conn: AsyncConnection, cfg: MemxConfig, *, scope: str | None = None,
                      dry_run: bool = False) -> RereconcileReport:
    report = RereconcileReport()
    planned: set[tuple[str, str]] = set()
    for kind, sql in (("supersede", _SUPERSEDED), ("shadow", _SHADOWS)):
        cur = await conn.execute(sql, {"scope": scope})
        for row in await cur.fetchall():
            report.checked += 1
            label, reason, verdict = recheck(row, cfg)
            if reason is None:
                report.kept += 1
                continue
            change = {"action": "restore" if kind == "supersede" else "unshadow", "fact_id": str(row["f_id"]),
                      "fact": row["f_text"], "by_id": str(row["r_id"]), "by": row["r_text"], "label": label,
                      "probability": round(verdict.probability, 4), "reason": reason}
            blocked = await _blocked(conn, cfg, row, kind, planned)
            if blocked:
                report.kept += 1
                report.skipped[blocked] = report.skipped.get(blocked, 0) + 1
                continue
            report.reasons[reason] = report.reasons.get(reason, 0) + 1
            report.changes.append(change)
            if dry_run:
                if kind == "supersede":
                    planned.add((row["f_scope"], row["f_text"]))
                report.restored += kind == "supersede"
                report.unshadowed += kind == "shadow"
                continue
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (row["f_scope"],))
                if kind == "supersede":
                    cur2 = await conn.execute(
                        "UPDATE memx_fact f SET status = 'active', superseded_by = NULL, superseded_at = NULL "
                        "WHERE id = %s AND status = 'superseded' AND superseded_by = %s "
                        # the same text may have been stored again since; one active copy is enough
                        "AND NOT EXISTS (SELECT 1 FROM memx_fact o WHERE o.scope @> f.scope "
                        "                AND o.text_hash = f.text_hash AND o.status = 'active') "
                        "RETURNING id",
                        (row["f_id"], row["r_id"]))
                    if not await cur2.fetchone():
                        report.changes.pop()
                        report.reasons[reason] -= 1
                        report.kept += 1
                        continue  # changed meanwhile, or an identical active fact exists
                    report.restored += 1
                else:
                    await conn.execute("DELETE FROM memx_fact_link WHERE a = %s AND b = %s AND kind = 'shadows'",
                                       (row["r_id"], row["f_id"]))
                    report.unshadowed += 1
                await repo.add_links(conn, [(row["r_id"], row["f_id"], "related", verdict.similarity)])
                await repo.add_decisions(conn, [_decision(row, change["action"], label, reason, verdict, cfg)])
    return report
