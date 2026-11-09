"""Reference labels for a sample of logged Jev decisions, to check how Jev's probabilities track agreement.

Samples reconcile verdicts and entity-merge decisions from the benchmark database, stratified by Jev's top
probability, and asks a strong generative model (default Claude Sonnet 5.5) the same question with the same
definitions. Agreement with that model is not ground truth; it shows whether Jev's confidence orders its decisions
the way a much larger model would.

    python -m eval.calibration.reference --verdicts 240 --entities 160
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from memx import MemxConfig
from memx.pipeline.entities import IDENTICAL_OPTIONS, MERGE_OPTIONS, VARIANT_OPTIONS
from memx.pipeline.reconcile import VERDICTS, relation
from memx.providers.http import OpenRouterHTTP
from memx.timewords import recency_phrase

HERE = Path(__file__).parent
OUT = HERE / "results"
DSN = os.environ.get("MEMX_BENCH_DSN", "postgresql://memx:memx@localhost:5433/memx_bench2")
BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]


def stratified(rows: list[dict], key, n: int, rng: random.Random) -> list[dict]:
    per = max(1, n // len(BINS))
    out = []
    for lo, hi in BINS:
        pool = [r for r in rows if lo <= key(r) < hi]
        out += rng.sample(pool, min(per, len(pool)))
    return out


def verdict_rows(conn, n: int, rng, labels: tuple[str, ...] | None = None) -> list[dict]:
    rows = conn.execute("""
        SELECT d.id, d.scope, d.similarity, d.result, c.observed_at AS c_at, c.scope AS c_scope,
               nb.observed_at AS n_at, nb.scope AS n_scope, nb.category AS n_cat, c.category AS c_cat
        FROM memx_decision d JOIN memx_fact c ON c.id = d.subject_fact_id JOIN memx_fact nb ON nb.id = d.object_fact_id
        WHERE d.step = 'verdict'""").fetchall()
    if labels:
        rows = [r for r in rows if r["result"]["label"] in labels]
    return stratified(rows, lambda r: float(r["result"]["probability"]), n, rng)


def entity_rows(conn, n: int, rng) -> list[dict]:
    rows = conn.execute("""
        SELECT d.id, d.result, f.text AS a_fact, cand AS c
        FROM memx_decision d JOIN memx_fact f ON f.id = d.subject_fact_id,
             jsonb_array_elements(d.result->'candidates') cand
        WHERE d.step = 'entity'""").fetchall()
    picked = stratified(rows, lambda r: max(float(r["c"]["p_same"]), float(r["c"]["p_different"])), n, rng)
    for r in picked:
        r["b_facts"] = [x["text"] for x in conn.execute(
            "SELECT f.text FROM memx_fact_entity fe JOIN memx_fact f ON f.id = fe.fact_id WHERE fe.entity_id = %s "
            "ORDER BY f.observed_at LIMIT 3", (r["c"]["entity_id"],)).fetchall()]
    return picked


def verdict_prompt(r: dict) -> str:
    res = r["result"]
    rel = {"same": "the same scope", "ancestor": "a broader (parent) scope",
           "descendant": "a narrower (child) scope"}[relation(r["n_scope"], r["c_scope"])]
    opts = "\n".join(f"- {k}: {v}" for k, v in VERDICTS.items())
    return (f"You are judging how a new candidate memory relates to an existing stored memory.\n\n"
            f"Candidate (category {r['c_cat']}): {res['candidate']}\n"
            f"Stored neighbor (category {r['n_cat']}, in {rel}): {res['neighbor']}\n"
            f"Timing: {recency_phrase(r['c_at'], r['n_at'])}\n\nOptions:\n{opts}\n\n"
            "Think briefly, then end with a final line of the form `LABEL: <option>` using one option name exactly.")


def entity_prompt(r: dict) -> str:
    how = r["c"]["how"]
    opts = {"identical name": IDENTICAL_OPTIONS, "shares a word": MERGE_OPTIONS}.get(how, VARIANT_OPTIONS)
    text = "\n".join(f"- {k}: {v}" for k, v in opts.items())
    b = "\n".join(f"  - {t}" for t in r["b_facts"]) or "  (none)"
    return (f"Entity resolution in one organization's memory. Name A, as written in a new fact, and known entity B.\n\n"
            f"A: {r['result']['name']}\nFact mentioning A: {r['a_fact']}\nB: {r['c']['with']}\nFacts about B:\n{b}\n"
            f"They were paired because: {how}.\n\nDoes A refer to the same thing as B? Options:\n{text}\n\n"
            "Think briefly, then end with a final line of the form `LABEL: <option>` using one option name exactly.")


def parse(text: str, options) -> str | None:
    m = re.findall(r"LABEL:\s*`?([a-z_]+)", text)
    return m[-1] if m and m[-1] in options else None


async def label(http: OpenRouterHTTP, model: str, prompt: str) -> str:
    data = await http.post("/chat/completions", {"model": model, "temperature": 0, "max_tokens": 400,
                                                 "messages": [{"role": "user", "content": prompt}]})
    return data["choices"][0]["message"].get("content") or ""


async def main_async(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    with psycopg.connect(DSN, row_factory=dict_row) as conn:
        vrows = verdict_rows(conn, args.verdicts, rng,
                             ("same", "candidate_replaces", "neighbor_covers") if args.destructive else None)
        erows = entity_rows(conn, args.entities, rng) if args.entities else []
    cfg = MemxConfig()
    http = OpenRouterHTTP(cfg.api_key(), cfg.openrouter_base_url, timeout_s=120)
    sem = asyncio.Semaphore(8)

    async def one(kind: str, r: dict) -> dict:
        async with sem:
            prompt = verdict_prompt(r) if kind == "verdict" else entity_prompt(r)
            opts = VERDICTS if kind == "verdict" else ("same", "different", "cannot_tell")
            text = await label(http, args.model, prompt)
            if kind == "verdict":
                jev, probs = r["result"]["label"], r["result"]["probabilities"]
            else:
                probs = {"same": r["c"]["p_same"], "different": r["c"]["p_different"]}
                probs["cannot_tell"] = round(max(0.0, 1 - sum(probs.values())), 4)
                jev = max(probs, key=probs.get)
            return {"kind": kind, "decision_id": str(r["id"]), "jev": jev, "jev_p": probs[jev], "probs": probs,
                    "how": r["c"]["how"] if kind == "entity" else None,
                    "similarity": float(r["similarity"]) if kind == "verdict" and r["similarity"] else None,
                    "ref": parse(text, opts), "ref_text": text[-300:]}

    out = await asyncio.gather(*[one("verdict", r) for r in vrows] + [one("entity", r) for r in erows])
    (OUT / ("reference_destructive.jsonl" if args.destructive else "reference_labels.jsonl")).write_text("".join(json.dumps(o, default=str) + "\n" for o in out))
    for kind in ("verdict", "entity"):
        sub = [o for o in out if o["kind"] == kind and o["ref"]]
        print(kind, len(sub), "agree", sum(o["ref"] == o["jev"] for o in sub))
    await http.aclose()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--verdicts", type=int, default=240)
    p.add_argument("--entities", type=int, default=160)
    p.add_argument("--model", default="anthropic/claude-sonnet-5.5")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--destructive", action="store_true", help="sample only same / replaces / covers verdicts")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
