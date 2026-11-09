"""Classify every question memx got wrong by the pipeline stage that lost the answer.

Read-only. Works from three things that already exist after a benchmark run:
  * the per-question results jsonl (``correct``, ``response``, ``answer``, evidence),
  * ``memx_fact`` joined to ``memx_ingest`` (which session each fact came from, and whether it is still active),
  * ``memx_recall_log.explain`` (every candidate with its search ranks, grade and why it was kept or dropped).

Every miss gets exactly one label, the weakest link among the evidence it needed:

  abstention_failed   the question has no answer in memory and the model answered anyway
  session_dropped     an evidence session produced no facts at all
  turn_not_extracted  (LoCoMo) the evidence turn produced no fact
  answer_not_in_facts facts exist for the evidence but none carries the answer
  reconciled_away     the fact carrying the answer was superseded or expired
  not_retrieved       the fact is active but neither search nor the walk proposed it
  not_graded          proposed, but beyond the grading cap
  below_floor         graded, but Jev rated it under the relevance floor
  cut_by_k            graded above the floor, dropped by the fact cap or the token budget
  judge_disputed      the fact was in the block and the response contains the answer; the judge said wrong
  rendered_but_wrong  the fact was in the block and the model still answered wrong

Usage:
  python -m eval.analysis.misses --bench locomo --results eval/locomo/results/<run>-<config>.jsonl
  python -m eval.analysis.misses --bench lme --results eval/longmemeval/results/<run>-<config>.jsonl
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

HERE = Path(__file__).parent
DSNS = {"locomo": "postgresql://memx:memx@localhost:5433/memx_locomo",
        "lme": "postgresql://memx:memx@localhost:5433/memx_lme"}

# Stage order: a lower index is an earlier loss. The label of a miss is the earliest stage among its evidence.
STAGES = ["session_dropped", "turn_not_extracted", "answer_not_in_facts", "wrong_date_extracted", "reconciled_away",
          "not_retrieved", "not_graded", "below_floor", "cut_by_k", "picked"]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
RELATIVE = re.compile(r"(?:the\s+)?(" + "|".join(WEEKDAYS) + r"|week|weekend|day|month|year)\s+(before|after)\s+(.+)", re.I)
STATUS_TO_STAGE = {"picked": "picked", "over_k": "cut_by_k", "over_budget": "cut_by_k", "below_floor": "below_floor",
                   "not_graded": "not_graded", "shadowed": "not_retrieved"}
GENERALIZING = re.compile(r"\b(would (likely |probably )?(prefer|enjoy|appreciate|want|be)|likely|probably|might|"
                          r"tend to|generally|typically)\b", re.I)
STOP = set("the a an of to in on at for and or is are was were be been by with from as that this it its their his her "
           "they them he she i my me you your we our us do does did has have had not no yes what which who when where "
           "how about into than then so if".split())
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
DATE_PATTERNS = [
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
    re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\b"),
    re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b"),
    re.compile(r"\b([A-Za-z]{3,9})\.?,?\s+(\d{4})\b"),
    re.compile(r"\b(\d{4})\b"),
]


def dates_in(text: str) -> set[str]:
    """Dates mentioned in a text, as ISO prefixes (``2023-05-07``, ``2023-05``, ``2023``), most specific per span."""
    out: set[str] = set()
    taken: list[tuple[int, int]] = []
    for i, pat in enumerate(DATE_PATTERNS):
        for m in pat.finditer(text):
            if any(a <= m.start() < b or a < m.end() <= b for a, b in taken):
                continue  # a less specific pattern inside a date already matched ("2023" inside "7 May 2023")
            g = m.groups()
            try:
                if i == 0:
                    val = f"{g[0]}-{g[1]}-{g[2]}"
                elif i == 1 and g[1].lower() in MONTHS:
                    val = f"{g[2]}-{MONTHS[g[1].lower()]:02d}-{int(g[0]):02d}"
                elif i == 2 and g[0].lower() in MONTHS:
                    val = f"{g[2]}-{MONTHS[g[0].lower()]:02d}-{int(g[1]):02d}"
                elif i == 3 and g[0].lower() in MONTHS:
                    val = f"{g[1]}-{MONTHS[g[0].lower()]:02d}"
                elif i == 4 and 1900 < int(g[0]) < 2100:
                    val = g[0]
                else:
                    continue
            except (ValueError, KeyError):
                continue
            out.add(val)
            taken.append((m.start(), m.end()))
    return out


def date_window(answer: str) -> tuple[date, date] | None:
    """The calendar window a gold answer names: an exact date, a month, a year, or a relative phrase such as
    "The Friday before 15 July 2023" (the weekday strictly before that date) or "The week before 9 June 2023"."""
    m = RELATIVE.search(answer)
    if m:
        unit, direction, rest = m.group(1).lower(), m.group(2).lower(), m.group(3)
        anchor = date_window(rest)
        if not anchor:
            return None
        a = anchor[0]
        sign = -1 if direction == "before" else 1
        if unit in WEEKDAYS:
            want = WEEKDAYS.index(unit)
            d = a + timedelta(days=sign)
            while d.weekday() != want:
                d += timedelta(days=sign)
            return (d, d)
        if unit == "day":
            d = a + timedelta(days=sign); return (d, d)
        if unit == "weekend":
            lo, hi = sorted((a + timedelta(days=sign), a + timedelta(days=8 * sign))); return (lo, hi)
        if unit == "week":
            lo, hi = sorted((a + timedelta(days=sign), a + timedelta(days=14 * sign))); return (lo, hi)
        if unit == "month":
            lo, hi = sorted((a + timedelta(days=sign), a + timedelta(days=45 * sign))); return (lo, hi)
        if unit == "year":
            lo, hi = sorted((a + timedelta(days=sign), a + timedelta(days=400 * sign))); return (lo, hi)
    found = sorted(dates_in(answer), key=len, reverse=True)
    if not found:
        return None
    v = found[0]
    try:
        if len(v) == 10:
            d = date.fromisoformat(v); return (d, d)
        if len(v) == 7:
            y, mo = int(v[:4]), int(v[5:]); last = (date(y + (mo == 12), mo % 12 + 1, 1) - timedelta(days=1))
            return (date(y, mo, 1), last)
        y = int(v); return (date(y, 1, 1), date(y, 12, 31))
    except ValueError:
        return None


def dates_match(answer: str, text: str) -> bool | None:
    """True/False when the answer names a date window and the text names dates; None when there is nothing to compare."""
    win = date_window(answer)
    if not win:
        return None
    found = dates_in(text)
    if not found:
        return False
    for v in found:
        lo = date.fromisoformat(v) if len(v) == 10 else date(int(v[:4]), int(v[5:7]) if len(v) == 7 else 1, 1)
        hi = lo if len(v) == 10 else (date(int(v[:4]), int(v[5:7]), 28) if len(v) == 7 else date(int(v[:4]), 12, 31))
        if lo <= win[1] and hi >= win[0]:
            return True
    return False


def tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())
    out = set()
    for w in words:
        w = w.split("'")[0]
        if w in STOP or (len(w) < 2 and not w.isdigit()):
            continue
        # light stemming so "classes" matches "class", "painted" matches "paint"
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.add(w)
    return out


def coverage(needle: str, hay: str) -> float:
    """Share of the needle's content tokens present in the hay (dates compared as ISO prefixes)."""
    nt, ht = tokens(needle), tokens(hay)
    dm = dates_match(needle, hay)
    if dm is not None:
        # a date answer is matched by its date; the words around it ("the week before") rarely appear in a fact
        rest = nt - {w for w in nt if w.isdigit()} - set(MONTHS) - set(WEEKDAYS) - {"before", "after", "week", "weekend"}
        word_cov = len(rest & ht) / len(rest) if rest else 1.0
        return 1.0 if dm else min(0.5, word_cov * 0.5)
    if not nt:
        return 1.0
    return len(nt & ht) / len(nt)


# ---------------------------------------------------------------------------------------------------------------------
# Benchmark adapters: where a question's scope is, and which ingest keys / turns hold the evidence.


class Locomo:
    CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}

    def __init__(self, tag: str = "v2"):
        self.data = {s["sample_id"]: s for s in json.load(open(HERE.parent / "locomo" / "data" / "locomo10.json"))}
        self.tag = tag + "." if tag else ""

    def scope(self, row: dict) -> str:
        return f"memory.locomo.{self.tag}{row['sample_id'].replace('-', '_')}"

    def evidence(self, row: dict) -> list[dict]:
        """One item per evidence turn: ingest key suffix, turn text, dia_id."""
        sample = self.data[row["sample_id"]]
        qa = sample["qa"][row["index"]]
        conv = sample["conversation"]
        out = []
        for e in qa.get("evidence") or []:
            m = re.match(r"D:?(\d+):(\d+)", str(e))
            if not m:
                continue
            n = m.group(1)
            turn = next((t for t in conv.get(f"session_{n}", []) if t.get("dia_id") in (e, f"D{n}:{m.group(2)}")), {})
            text = turn.get("text", "")
            if turn.get("blip_caption"):
                text = f"{text} {turn['blip_caption']}"
            out.append({"key": f"session_{n}", "dia_id": e, "text": text, "image": bool(turn.get("blip_caption")),
                        "image_query": turn.get("query")})
        return out

    @staticmethod
    def group(row: dict) -> str:
        return row["category"]

    @staticmethod
    def qid(row: dict) -> str:
        return f"{row['sample_id']}#{row['index']}"


class LongMemEval:
    def __init__(self, tag: str = "lean"):
        self.data = {q["question_id"]: q for q in json.load(open(HERE.parent / "longmemeval" / "data" /
                                                                   "longmemeval_s_cleaned.json"))}
        self.tag = tag

    def scope(self, row: dict) -> str:
        return f"memory.lme.{self.tag}.{re.sub(r'[^A-Za-z0-9_]', '_', row['question_id'])[:64]}"

    def evidence(self, row: dict) -> list[dict]:
        item = self.data[row["question_id"]]
        return [{"key": sid, "dia_id": None, "text": "", "image": False} for sid in item.get("answer_session_ids") or []]

    @staticmethod
    def group(row: dict) -> str:
        return row["question_type"]

    @staticmethod
    def qid(row: dict) -> str:
        return row["question_id"]


# ---------------------------------------------------------------------------------------------------------------------


def load_results(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def scope_facts(conn, scope: str) -> list[dict]:
    cur = conn.execute(
        "SELECT f.id::text AS id, f.text, f.status, f.superseded_by::text AS superseded_by, f.category, "
        "f.attrs->>'attribute' AS attribute, f.flags, i.idempotency_key AS key "
        "FROM memx_fact f JOIN memx_ingest i ON i.id = f.ingest_id WHERE f.scope = %s::ltree", (scope,))
    rows = cur.fetchall()
    for r in rows:
        r["key"] = r["key"].removeprefix(f"{scope}:")
    return rows


def recall_row(conn, scope: str, query: str, mode: str, k_max: int | None, floor: float | None) -> dict | None:
    clauses, params = ["scope = %s::ltree", "query = %s", "mode = %s"], [scope, query, mode]
    if k_max is not None:
        clauses.append("(explain->>'k_max')::int = %s"); params.append(k_max)
    if floor is not None:
        clauses.append("(explain->>'grade_floor')::float = %s"); params.append(floor)
    cur = conn.execute(f"SELECT fact_ids::text[] AS fact_ids, explain, created_at FROM memx_recall_log "
                       f"WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT 1", params)
    return cur.fetchone()


def classify(row: dict, bench, conn, facts_cache: dict, mode: str, k_max: int | None, floor: float | None,
             turn_min: float, answer_min: float) -> dict:
    scope = bench.scope(row)
    if scope not in facts_cache:
        facts_cache[scope] = scope_facts(conn, scope)
    facts = facts_cache[scope]
    by_id = {f["id"]: f for f in facts}
    answer = str(row["answer"])
    response = str(row.get("response") or "")
    group = bench.group(row)
    out: dict[str, Any] = {"qid": bench.qid(row), "group": group, "question": row["question"], "answer": answer,
                           "response": response[:300], "scope": scope}

    if group == "abstention":
        out["label"] = "abstention_failed"
        return out

    rec = recall_row(conn, scope, row["question"], mode, k_max, floor)
    cands = {c["id"]: c for c in (rec or {}).get("explain", {}).get("candidates", [])} if rec else {}
    rendered = set((rec or {}).get("fact_ids") or [])
    out["recall_log_found"] = rec is not None

    evidence = bench.evidence(row)
    if not evidence:
        out["label"] = "no_evidence_given"
        return out

    per_evidence = []
    for ev in evidence:
        ev_facts = [f for f in facts if f["key"] == ev["key"]]
        item: dict[str, Any] = {"key": ev["key"], "dia_id": ev["dia_id"], "session_facts": len(ev_facts),
                                "image": ev.get("image", False), "image_query": ev.get("image_query")}
        if not ev_facts:
            item["stage"] = "session_dropped"
            per_evidence.append(item)
            continue
        # facts that came from this evidence turn (LoCoMo) or from the whole session (LongMemEval)
        if ev["text"]:
            # a fact "comes from" the turn when a third of either side's content words are shared, or four words are:
            # extracted facts paraphrase ("Last Friday I went to a council meeting" -> "attended a council meeting on
            # 2023-07-07"), so exact coverage runs 0.35-0.45 for true matches and under 0.3 with 1-3 words for others
            turn_facts = [f for f in ev_facts
                          if max(coverage(f["text"], ev["text"]), coverage(ev["text"], f["text"])) >= turn_min
                          or len(tokens(f["text"]) & tokens(ev["text"])) >= 4]
            item["turn_facts"] = len(turn_facts)
            if ev.get("image_query"):
                item["answer_in_image_query"] = coverage(answer, ev["image_query"]) >= 0.6
            if not turn_facts:
                item["stage"] = "turn_not_extracted"
                item["session_sample"] = [f["text"] for f in ev_facts[:3]]
                per_evidence.append(item)
                continue
        else:
            turn_facts = ev_facts
        if group == "temporal" and date_window(answer) and len(evidence) == 1:
            dated = [f for f in turn_facts if dates_in(f["text"])]
            if dated and not any(dates_match(answer, f["text"]) for f in dated):
                item["stage"] = "wrong_date_extracted"
                item["fact_dates"] = sorted({d for f in dated for d in dates_in(f["text"])})
                item["answer_window"] = [str(d) for d in date_window(answer)]
                item["gold"] = [f["text"] for f in dated[:3]]
                per_evidence.append(item)
                continue
            if not dated:
                item["undated_turn_facts"] = True
        scored = sorted(((coverage(answer, f["text"]), f) for f in turn_facts), key=lambda x: -x[0])
        gold = [f for c, f in scored if c >= answer_min]
        item["best_answer_coverage"] = round(scored[0][0], 2) if scored else 0.0
        item["best_fact"] = scored[0][1]["text"] if scored else None
        if not gold:
            # multi-evidence questions: a bridge turn need not carry the answer text itself
            if len(evidence) > 1 and turn_facts:
                gold = [f for _, f in scored[:3]]
                item["bridge_only"] = True
            else:
                item["stage"] = "answer_not_in_facts"
                item["answer_in_turn_text"] = coverage(answer, ev["text"]) >= 0.6 if ev["text"] else None
                item["turn_sample"] = [f["text"] for f, in [(x[1],) for x in scored[:3]]]
                per_evidence.append(item)
                continue
        active = [f for f in gold if f["status"] == "active"]
        if not active:
            item["stage"] = "reconciled_away"
            item["replaced_by"] = [by_id.get(f["superseded_by"], {}).get("text") for f in gold if f["superseded_by"]]
            item["gold"] = [f["text"] for f in gold]
            per_evidence.append(item)
            continue
        # the best outcome any gold fact reached in recall
        best, best_fact = "not_retrieved", None
        for f in active:
            c = cands.get(f["id"])
            stage = "picked" if f["id"] in rendered else STATUS_TO_STAGE.get((c or {}).get("status"), "not_retrieved")
            if STAGES.index(stage) > STAGES.index(best):
                best, best_fact = stage, (f, c)
        item["stage"] = best
        if best_fact:
            f, c = best_fact
            item["gold_fact"] = f["text"]
            item["gold_attribute"] = f["attribute"]
            if c:
                item["gold_relevance"] = c.get("relevance")
                item["gold_vec_rank"] = c.get("vec_rank")
                item["gold_fts_rank"] = c.get("fts_rank")
                item["gold_found_by"] = c.get("found_by")
                item["gold_score"] = c.get("score")
        else:
            item["gold_fact"] = active[0]["text"]
        per_evidence.append(item)

    out["evidence"] = per_evidence
    weakest = min(per_evidence, key=lambda e: STAGES.index(e["stage"]))
    label = weakest["stage"]
    if label == "picked":
        dm = dates_match(answer, response)
        label = "judge_disputed" if (dm if dm is not None else coverage(answer, response) >= 0.8) else "rendered_but_wrong"
        out["generalizing"] = bool(GENERALIZING.search(response))
        out["response_kind"] = ("refused" if re.search(r"^\W*(i don'?t know|i do not know|the memory does(n't| not)|"
                                                      r"not (named|stated|mentioned) in|my memory does(n't| not)|"
                                                      r"i don'?t have)", response, re.I) else "wrong_answer")
        out["rendered_facts"] = len(rendered)
    out["label"] = label
    out["weakest"] = weakest
    out["image_evidence"] = bool(weakest.get("image"))
    out["grade_floor"] = (rec or {}).get("explain", {}).get("grade_floor") if rec else None
    return out


def summarize(items: list[dict]) -> dict:
    labels = Counter(i["label"] for i in items)
    by_group: dict[str, Counter] = defaultdict(Counter)
    for i in items:
        by_group[i["group"]][i["label"]] += 1
    rel = [i["weakest"]["gold_relevance"] for i in items if i["label"] == "below_floor"
           and i.get("weakest", {}).get("gold_relevance") is not None]
    gen = [i for i in items if i["label"] == "rendered_but_wrong"]
    return {
        "misses": len(items), "labels": dict(labels.most_common()),
        "by_group": {g: dict(c.most_common()) for g, c in sorted(by_group.items())},
        "below_floor_relevance": {"n": len(rel), "median": round(statistics.median(rel), 3) if rel else None,
                                  "p75": round(sorted(rel)[int(0.75 * (len(rel) - 1))], 3) if rel else None,
                                  "over_0.2": sum(1 for r in rel if r >= 0.2)},
        "rendered_but_wrong": {"n": len(gen), "generalizing": sum(1 for i in gen if i.get("generalizing")),
                               "refused": sum(1 for i in gen if i.get("response_kind") == "refused")},
        "extraction_misses_with_image_evidence": sum(
            1 for i in items if i["label"] in ("turn_not_extracted", "answer_not_in_facts") and i.get("image_evidence")),
        "of_which_answer_in_image_query": sum(
            1 for i in items if i["label"] in ("turn_not_extracted", "answer_not_in_facts")
            and i.get("weakest", {}).get("answer_in_image_query")),
        "answer_not_in_facts_but_in_turn_text": sum(
            1 for i in items if i["label"] == "answer_not_in_facts" and i.get("weakest", {}).get("answer_in_turn_text")),
        "recall_log_missing": sum(1 for i in items if i.get("recall_log_found") is False),
        "image_evidence_by_label": {lab: sum(1 for i in items if i["label"] == lab and i.get("image_evidence"))
                                    for lab in labels},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", choices=["locomo", "lme"], required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--dsn")
    p.add_argument("--tag", help="ingest tag in the scope (locomo default v2, lme default lean)")
    p.add_argument("--mode", default=None, help="recall mode to match in the log (default: from the results rows)")
    p.add_argument("--k-max", type=int, default=None)
    p.add_argument("--floor", type=float, default=None)
    p.add_argument("--turn-min", type=float, default=0.35, help="fact→turn coverage to count a fact as from that turn")
    p.add_argument("--answer-min", type=float, default=0.6, help="answer→fact coverage to count a fact as gold")
    p.add_argument("--also-correct", action="store_true", help="classify every row, not only misses")
    p.add_argument("--wrong-here-right-in", type=Path, help="restrict to misses that another results file got right")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    bench = Locomo(args.tag if args.tag is not None else "v2") if args.bench == "locomo" else \
        LongMemEval(args.tag if args.tag is not None else "lean")
    rows = load_results(args.results)
    if args.wrong_here_right_in:
        other = {bench.qid(r): r for r in load_results(args.wrong_here_right_in)}
        rows = [r for r in rows if bench.qid(r) in other and other[bench.qid(r)]["correct"]]
    misses = rows if args.also_correct else [r for r in rows if not r["correct"]]
    mode = args.mode or (rows[0].get("mode") if rows and rows[0].get("mode") else None)
    if mode is None:
        mode = "full" if "full" in args.results.stem else "ranked"
    out_path = args.out or (HERE / "results" / f"{args.results.stem}.misses.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(args.dsn or DSNS[args.bench], row_factory=dict_row) as conn:
        conn.execute("SET default_transaction_read_only = on")
        cache: dict = {}
        items = [classify(r, bench, conn, cache, mode, args.k_max, args.floor, args.turn_min, args.answer_min)
                 for r in misses]
    with open(out_path, "w") as f:
        for i in items:
            f.write(json.dumps(i, default=str) + "\n")
    summary = summarize(items)
    summary.update({"results": str(args.results), "mode": mode, "k_max": args.k_max, "floor": args.floor,
                    "rows": len(rows)})
    with open(out_path.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
