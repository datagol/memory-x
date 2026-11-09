"""Figures for the Memory-X technical report (docs/paper/tech-report).

Reads the benchmark database (memx_bench2: every insert trace, Jev decision and recall explanation from the runs in
the paper) and the result files under eval/. Run from the repository root:

    uv run --no-project --with matplotlib --with "psycopg[binary]" --with numpy \
        python docs/paper/tech-report/make_figures.py
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import psycopg  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
DSN = os.environ.get("MEMX_BENCH_DSN", "postgresql://memx:memx@localhost:5433/memx_bench2")
LME = ROOT / "eval" / "longmemeval" / "results"
ORG = ROOT / "eval" / "orgprobe" / "results"
RUN = os.environ.get("PAPER_RUN", "paper")

for f in (HERE / "fonts").glob("Manrope-*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({
    "font.family": "Manrope", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": "#9aa0a6", "axes.linewidth": 0.6, "xtick.color": "#4a4f55", "ytick.color": "#4a4f55",
    "axes.labelcolor": "#2b2f33", "legend.frameon": False, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
INK, MUTED, GRID = "#1f2328", "#6b7280", "#eceef0"
BLUE, TEAL, ORANGE, PURPLE, RED, GOLD, GREEN = "#3b5bdb", "#0d9488", "#ea580c", "#9333ea", "#dc2626", "#ca8a04", "#16a34a"
VERDICT_COLOR = {"same": BLUE, "candidate_replaces": RED, "neighbor_covers": ORANGE, "related": TEAL,
                 "unrelated": "#b8bec6"}
VERDICT_LABEL = {"same": "same", "candidate_replaces": "candidate replaces", "neighbor_covers": "neighbor covers",
                 "related": "related", "unrelated": "unrelated"}
CATS = ["identity", "rule", "preference", "goal", "relationship", "work_context", "procedure", "event", "other"]
CAT_LABEL = {"work_context": "work context"}
HALF_LIFE = {"identity": None, "rule": None, "preference": 365, "goal": 180, "relationship": 365,
             "work_context": 120, "procedure": 365, "event": 30, "other": 90}


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", name)


def q(sql: str, *args):
    with psycopg.connect(DSN, row_factory=dict_row) as conn:
        return conn.execute(sql, args or None).fetchall()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def style(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- 1. classification

def fig_sort() -> dict:
    rows = q("SELECT c->'sort' AS s, c->>'outcome' AS outcome FROM memx_ingest i, "
             "jsonb_array_elements(i.candidates) c WHERE i.scope::text LIKE 'memory.lme%' AND c ? 'sort'")
    sorts = [r["s"] for r in rows if r["s"]]
    fig = plt.figure(figsize=(6.9, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.55], wspace=0.55)

    # (a) category: what Jev chose vs what was stored after the 0.45 floor
    ax = fig.add_subplot(gs[0])
    chosen = Counter(max(s["probabilities"], key=s["probabilities"].get) for s in sorts if s.get("probabilities"))
    stored = Counter(s["category"] for s in sorts)
    y = np.arange(len(CATS))
    ax.barh(y + 0.2, [chosen[c] / len(sorts) * 100 for c in CATS], 0.38, color="#c7d2fe", label="Jev's top choice")
    ax.barh(y - 0.2, [stored[c] / len(sorts) * 100 for c in CATS], 0.38, color=BLUE, label="stored (after floor)")
    ax.set_yticks(y, [CAT_LABEL.get(c, c) for c in CATS], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("% of candidates")
    ax.legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.45, -0.22), handlelength=1, ncol=1)
    ax.set_title("a  Category", loc="left", fontweight="bold")
    style(ax, "x")

    # (b) how sure: top probability for category and theme
    ax = fig.add_subplot(gs[1])
    cat_top = [max(s["probabilities"].values()) for s in sorts if s.get("probabilities")]
    top_t = [s.get("topic_confidence") for s in sorts if s.get("topic_confidence") is not None]
    bins = np.linspace(0, 1, 21)
    ax.hist(cat_top, bins=bins, color=BLUE, alpha=0.85, label="category", density=True)
    ax.hist(top_t, bins=bins, histtype="step", color=ORANGE, linewidth=1.3, label="theme", density=True)
    ax.axvline(0.45, color=INK, linestyle="--", linewidth=0.8)
    ax.text(0.43, ax.get_ylim()[1] * 0.55, "floor\n0.45", ha="right", fontsize=6.3, color=INK)
    ax.set_xlabel("Jev top probability")
    ax.set_ylabel("density")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("b  How sure", loc="left", fontweight="bold")
    style(ax)

    # (c) category x theme heatmap (row-normalised)
    ax = fig.add_subplot(gs[2])
    pairs = Counter((s["category"], s.get("topic") or "none") for s in sorts)
    themes = [t for t, _ in Counter(s.get("topic") or "none" for s in sorts).most_common(14)]
    row_tot = Counter(s["category"] for s in sorts)
    mat = np.array([[pairs[(c, t)] for t in themes] for c in CATS], dtype=float)
    norm = mat / np.array([[max(1, row_tot[c])] for c in CATS])
    cmap = LinearSegmentedColormap.from_list("m", ["#ffffff", "#c7d2fe", BLUE, "#1e2a78"])
    ax.imshow(norm, aspect="auto", cmap=cmap, vmin=0, vmax=0.45)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            if norm[i, j] >= 0.12:
                ax.text(j, i, f"{norm[i, j] * 100:.0f}", ha="center", va="center", fontsize=5.2,
                        color="white" if norm[i, j] > 0.3 else INK)
    ax.set_xticks(range(len(themes)), [t.replace("_and_", " & ").replace("_", " ") for t in themes], rotation=55,
                  ha="right", fontsize=5.8)
    ax.set_yticks(range(len(CATS)), [CAT_LABEL.get(c, c) for c in CATS], fontsize=6.5)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("c  Category x theme (row share)", loc="left", fontweight="bold")
    save(fig, "fig_sort")
    return {"candidates": len(sorts), "other_share": stored["other"] / len(sorts),
            "other_chosen_share": chosen["other"] / len(sorts),
            "cat_median_top": float(np.median(cat_top)), "theme_median_top": float(np.median(top_t)),
            "injection_flagged": sum(1 for s in sorts if (s.get("injection") or 0) >= 0.5)}


# ---------------------------------------------------------------- 2. reconciliation

def fig_reconcile() -> dict:
    rows = q("SELECT c->'neighbors' AS n, c->'verdicts' AS v, c->>'outcome' AS outcome FROM memx_ingest i, "
             "jsonb_array_elements(i.candidates) c WHERE i.scope::text LIKE 'memory.lme%'")
    bands = Counter()
    for r in rows:
        for n in r["n"] or []:
            bands[n.get("band")] += 1
    outcomes = Counter(r["outcome"] for r in rows)
    ver = q("SELECT similarity AS s, result->>'label' AS label, (result->>'probability')::float AS p "
            "FROM memx_decision WHERE step='verdict'")
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.5), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axs[0]
    bins = np.linspace(0.40, 1.0, 31)
    order = ["unrelated", "related", "neighbor_covers", "candidate_replaces", "same"]
    data = [[v["s"] for v in ver if v["label"] == lab and v["s"] is not None] for lab in order]
    ax.hist(data, bins=bins, stacked=True, color=[VERDICT_COLOR[o] for o in order],
            label=[VERDICT_LABEL[o] for o in order], edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.axvspan(0.40, 0.45, color="#f1f2f4", zorder=0)
    ax.axvspan(0.95, 1.0, color="#f1f2f4", zorder=0)
    for x in (0.45, 0.95):
        ax.axvline(x, color=INK, linestyle="--", linewidth=0.8)
    ax.set_ylim(0.7, 2.5e4)
    ax.text(0.425, 3, "below:\nno call", ha="center", fontsize=6, color=MUTED)
    ax.text(0.975, 2.0e3, "auto-same\nif lexical\nguard", ha="center", fontsize=6, color=MUTED)
    ax.text(0.70, 6e3, "band where Jev decides", ha="center", fontsize=7.5, color=INK, fontweight="bold")
    ax.set_xlabel("cosine similarity of candidate and stored neighbor")
    ax.set_ylabel("Jev verdicts (log)")
    ax.legend(fontsize=6.3, ncol=5, loc="upper center", bbox_to_anchor=(0.62, -0.33), handlelength=1,
              columnspacing=0.8)
    ax.set_title("a  Where Jev is asked, and what it says", loc="left", fontweight="bold")
    style(ax)

    ax = axs[1]
    labs = [b for b in ["exact", "auto", "judge", "below", "not_needed", "over_cap"] if bands[b]]
    names = {"exact": "hash match", "auto": "auto (>0.95)", "judge": "sent to Jev", "below": "below 0.45",
             "not_needed": "not needed", "over_cap": "over cap"}
    vals = [bands[b] for b in labs]
    tot = sum(vals) or 1
    ax.barh(range(len(labs)), [v / tot * 100 for v in vals],
            color=[{"exact": BLUE, "auto": BLUE, "judge": ORANGE, "below": "#b8bec6"}.get(b, "#d5d8dc") for b in labs])
    for i, v in enumerate(vals):
        ax.text(v / tot * 100 + 1, i, f"{v:,}", va="center", fontsize=6.5)
    ax.set_yticks(range(len(labs)), [names[b] for b in labs], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("% of neighbors examined")
    ax.set_title("b  How neighbors were settled", loc="left", fontweight="bold")
    ax.set_xlim(0, max(v / tot * 100 for v in vals) * 1.3)
    style(ax, "x")
    fig.tight_layout()
    save(fig, "fig_reconcile")
    lab = Counter(v["label"] for v in ver)
    destructive = [v for v in ver if v["label"] in ("candidate_replaces", "neighbor_covers")]
    return {"neighbors": tot, "bands": dict(bands), "verdicts": len(ver), "labels": dict(lab),
            "outcomes": dict(outcomes),
            "destructive": len(destructive),
            "destructive_05_06": sum(1 for v in destructive if 0.5 <= v["p"] < 0.6),
            "destructive_below_05": sum(1 for v in destructive if v["p"] < 0.5)}


def fig_verdict_prob() -> None:
    ver = q("SELECT result->>'label' AS label, (result->>'probability')::float AS p FROM memx_decision "
            "WHERE step='verdict'")
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    bins = np.linspace(0.2, 1.0, 33)
    for lab in ["candidate_replaces", "neighbor_covers", "same"]:
        ps = [v["p"] for v in ver if v["label"] == lab]
        ax.hist(ps, bins=bins, histtype="step", linewidth=1.4, color=VERDICT_COLOR[lab],
                label=f"{VERDICT_LABEL[lab]} (n={len(ps)})")
    ax.axvspan(0.2, 0.5, color="#f1f2f4", zorder=0)
    ax.axvline(0.5, color=INK, linestyle="--", linewidth=0.8)
    ax.text(0.35, ax.get_ylim()[1] * 0.85, "falls back\nto related", ha="center", fontsize=6.5, color=MUTED)
    ax.set_xlabel("probability of the chosen verdict")
    ax.set_ylabel("verdicts")
    ax.legend(fontsize=6.2, loc="upper left", bbox_to_anchor=(0.28, 1.0))
    style(ax)
    save(fig, "fig_verdict_prob")


# ---------------------------------------------------------------- 3. entities

def fig_entities() -> dict:
    rows = q("SELECT c->>'how' AS how, (c->>'p_same')::float AS ps, (c->>'p_different')::float AS pd "
             "FROM memx_decision d, jsonb_array_elements(d.result->'candidates') c WHERE d.step='entity'")
    hows = [("identical name", BLUE), ("shares a word", TEAL), ("suggested alternative name", ORANGE),
            ("similar name", "#9aa0a6")]
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.4), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axs[0]
    bins = np.linspace(0, 1, 21)
    for how, col in hows:
        ps = [r["ps"] for r in rows if r["how"] == how]
        if ps:
            ax.hist(ps, bins=bins, histtype="step", linewidth=1.4, color=col, label=f"{how} (n={len(ps)})",
                    density=False)
    ax.set_yscale("log")
    ax.axvline(0.7, color=INK, linestyle="--", linewidth=0.8)
    ax.text(0.71, 3, "merge\nat 0.7", fontsize=6.5, color=INK)
    ax.set_xlabel("Jev P(same entity)")
    ax.set_ylabel("pairs (log)")
    ax.legend(fontsize=6.2, loc="upper center")
    ax.set_title("a  P(same) by why the pair was formed", loc="left", fontweight="bold")
    style(ax)

    # (b) the asymmetric decision rule as regions over (p_same, p_different)
    ax = axs[1]
    g = np.linspace(0, 1, 201)
    P, D = np.meshgrid(g, g)
    valid = P + D <= 1.0001
    diff_name = np.where(P >= 0.7, 1, 0)  # different spellings: merge only when sure
    ident = np.where(P >= 0.7, 1, np.where(D >= 0.7, 0, 1))  # identical names: split only when sure
    img = np.where(valid, ident * 2 + diff_name, np.nan)
    cmap = matplotlib.colors.ListedColormap(["#fde2e2", "#ffffff", "#e0e7ff", "#c7f0e8"])
    ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.plot([0, 1], [1, 0], color=MUTED, linewidth=0.6)
    ax.axvline(0.7, color=INK, linestyle="--", linewidth=0.7)
    ax.axhline(0.7, xmax=0.3, color=RED, linestyle="--", linewidth=0.7)
    ax.text(0.86, 0.035, "merge (both)", ha="center", fontsize=6.0)
    ax.text(0.15, 0.82, "identical: split\nspellings: apart", ha="left", fontsize=6.2, color=RED)
    ax.text(0.33, 0.25, "identical names: kept together\ndifferent spellings: kept apart", ha="center",
            fontsize=6.2, color=INK)
    ax.set_xlabel("P(same)")
    ax.set_ylabel("P(different)")
    ax.set_title("b  Asymmetric defaults", loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_entities")
    ents = q("SELECT count(*) AS n, count(*) FILTER (WHERE array_length(aliases,1) > 0) AS aliased FROM memx_entity")
    return {"pairs": len(rows), "by_how": {h: sum(1 for r in rows if r["how"] == h) for h, _ in hows},
            "merged": sum(1 for r in rows if r["ps"] >= 0.7), "entities": ents[0]}


# ---------------------------------------------------------------- 4. recall

CURRENT_DSN = os.environ.get("MEMX_CURRENT_DSN", "postgresql://memx:memx@localhost:5433/memx_bench3")


def memx_ingest_usd(u: dict) -> float:
    """Haiku 5.5 list prices; cached prompt tokens bill at the cache-read price. Plus metered Jev cost."""
    cached = u.get("extract_cached_tokens_in", 0)
    return (u["extract_tokens_in"] - cached) * 0.1e-6 + cached * 0.01e-6 + u["extract_tokens_out"] * 0.5e-6 \
        + u["jev_usd"]


def fig_recall() -> dict:
    # ranked recalls of the current system: the fresh Haiku 5.5 LongMemEval run (scopes memory.lme.lean.*)
    with psycopg.connect(CURRENT_DSN, row_factory=dict_row) as conn:
        rows = conn.execute(
            "SELECT explain FROM memx_recall_log WHERE mode='ranked' AND scope::text LIKE 'memory.lme.lean.%' "
            "AND explain ? 'candidates'").fetchall()
    stats = defaultdict(list)
    grades = Counter()
    found = Counter()
    for r in rows:
        ex = r["explain"]
        cands = ex.get("candidates") or []
        stats["candidates"].append(len(cands))
        stats["search"].append(sum(1 for c in cands if c.get("found_by") == "search"))
        stats["walk"].append(sum(1 for c in cands if c.get("found_by") in ("entity_hop", "topic_hop")))
        stats["graded"].append(sum(1 for c in cands if c.get("status") != "not_graded"))
        stats["above"].append(sum(1 for c in cands if c.get("status") in ("picked", "over_budget", "over_k")))
        stats["picked"].append(sum(1 for c in cands if c.get("status") == "picked"))
        for c in cands:
            if c.get("relevance") is not None:
                grades[min(4, int(math.floor(float(c["relevance"]) * 4 + 0.5)))] += 1
            if c.get("status") == "picked":
                found[c.get("found_by")] += 1
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.3), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    steps = [("candidates", "candidates\n(search + walk)"), ("graded", "graded by Jev"),
             ("above", "above relevance\nfloor"), ("picked", "fit the budget\n(picked)")]
    vals = [np.mean(stats[k]) if stats[k] else 0 for k, _ in steps]
    cols = ["#c7d2fe", "#93a5f5", BLUE, "#1e2a78"]
    for i, (v, c) in enumerate(zip(vals, cols)):
        ax.barh(i, v, color=c, height=0.62, left=(max(vals) - v) / 2)
        ax.text(max(vals) / 2, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                color="white" if i >= 2 else INK, fontweight="bold")
    ax.set_yticks(range(len(steps)), [s[1] for s in steps], fontsize=7)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.spines[["bottom", "left"]].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"a  Ranked recall funnel (mean of {len(rows)} recalls)", loc="left", fontweight="bold")
    ax = axs[1]
    levels = ["irrelevant", "background", "useful", "important", "critical"]
    tot = sum(grades.values()) or 1
    ax.bar(range(5), [grades[i] / tot * 100 for i in range(5)],
           color=["#d5d8dc", "#b8bec6", TEAL, BLUE, "#1e2a78"])
    ax.set_xticks(range(5), levels, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("% of graded facts")
    ax.axvline(1.2, color=INK, linestyle="--", linewidth=0.8)
    ax.text(1.27, ax.get_ylim()[1] * 0.9, "search floor 0.3", fontsize=6.3)
    ax.axvline(2.0, color=TEAL, linestyle=":", linewidth=0.8)
    ax.text(2.07, ax.get_ylim()[1] * 0.75, "walk floor 0.5", fontsize=6.3, color=TEAL)
    ax.set_title("b  Jev relevance grades", loc="left", fontweight="bold")
    style(ax)
    fig.tight_layout()
    save(fig, "fig_recall")
    return {"recalls": len(rows), **{k: float(np.mean(v)) for k, v in stats.items() if v},
            "picked_by": dict(found), "grades": dict(grades)}


def fig_decay() -> None:
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    days = np.linspace(0, 400, 400)
    cols = {"event": RED, "work_context": ORANGE, "other": GOLD, "goal": PURPLE, "preference": TEAL,
            "identity": BLUE}
    for c, col in cols.items():
        hl = HALF_LIFE[c]
        fresh = np.ones_like(days) if hl is None else 2 ** (-days / hl)
        label = f"{CAT_LABEL.get(c, c)} ({'never' if hl is None else str(hl) + ' d'})"
        if c == "identity":
            label = "identity, rule (never)"
        if c == "preference":
            label = "pref., relationship, proc. (365 d)"
        ax.plot(days, 0.6 + 0.4 * fresh, color=col, linewidth=1.3, label=label)
    ax.set_ylim(0.55, 1.03)
    ax.set_xlabel("days since observed or last repeated")
    ax.set_ylabel("freshness factor")
    ax.legend(fontsize=5.8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.24), handlelength=1.2)
    style(ax)
    save(fig, "fig_decay")


def fig_latency() -> dict:
    rows = q("SELECT step, latency_ms FROM memx_decision WHERE latency_ms IS NOT NULL")
    by = defaultdict(list)
    for r in rows:
        by[r["step"]].append(r["latency_ms"])
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    cols = {"sort": BLUE, "verdict": RED, "entity": TEAL, "grade": ORANGE}
    out = {}
    for s, col in cols.items():
        v = np.sort(np.array(by.get(s, [])))
        if not len(v):
            continue
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, linewidth=1.3,
                label=f"{s} (n={len(v):,}, p50 {int(np.median(v))} ms)")
        out[s] = {"n": len(v), "p50": float(np.median(v)), "p95": float(np.percentile(v, 95))}
    ax.set_xscale("log")
    ax.set_xlabel("Jev request latency (ms, log)")
    ax.set_ylabel("cumulative share")
    ax.legend(fontsize=6, loc="lower right")
    style(ax, "both")
    save(fig, "fig_latency")
    return out


# ---------------------------------------------------------------- 5. benchmark

def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


LME_TYPES = ["single-session-user", "single-session-assistant", "single-session-preference", "multi-session",
             "knowledge-update", "temporal-reasoning", "abstention"]
TYPE_LABEL = {"single-session-user": "SS-user", "single-session-assistant": "SS-assistant",
              "single-session-preference": "SS-preference", "multi-session": "multi-session",
              "knowledge-update": "knowledge update", "temporal-reasoning": "temporal", "abstention": "abstention"}
# name -> (result file stem, label, colour). Memory-X and mem0: fresh ingest with Haiku 5.5 (full55-*). The two
# baselines do no extraction; they share the answering model (Haiku 4.5) and judge.
LME_FILES = {
    "v2_ranked": ("final-lean", "Memory-X ranked", BLUE),
    "v2_full": ("final-lean_full", "Memory-X full memory", "#93a5f5"),
    "mem0": ("full55-mem0_top30", "mem0 top-30", ORANGE),
    "chunk_rag": (f"{RUN}-chunk_rag", "chunk RAG (4k)", TEAL),
    "full_context": (f"{RUN}-full_context", "full context", "#9ca3af"),
}
BARS = ["v2_ranked", "v2_full", "mem0", "chunk_rag", "full_context"]


def mcnemar_rows(a: dict, b: dict, common) -> dict:
    b10 = sum(1 for q in common if a[q]["correct"] and not b[q]["correct"])
    b01 = sum(1 for q in common if not a[q]["correct"] and b[q]["correct"])
    n, k = b10 + b01, min(b10, b01)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"a_only": b10, "b_only": b01, "p": round(p, 4)}


def fig_lme() -> dict:
    runs = {n: load_jsonl(LME / f"{stem}.jsonl") for n, (stem, _, _) in LME_FILES.items()}
    common = set.intersection(*[{r["question_id"] for r in v} for v in runs.values() if v])
    by = {n: {r["question_id"]: r for r in v if r["question_id"] in common} for n, v in runs.items() if v}
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.8), gridspec_kw={"width_ratios": [2.1, 1]})
    ax = axs[0]
    w = 0.16
    for j, name in enumerate(BARS):
        _, label, col = LME_FILES[name]
        rows = list(by.get(name, {}).values())
        for i, t in enumerate(LME_TYPES):
            sub = [r for r in rows if r["question_type"] == t]
            k, n = sum(r["correct"] for r in sub), len(sub)
            if not n:
                continue
            lo, hi = wilson(k, n)
            x = i + (j - 2) * w
            ax.bar(x, k / n * 100, w * 0.92, color=col, label=label if i == 0 else None)
            ax.errorbar(x, k / n * 100, yerr=[[k / n * 100 - lo * 100], [hi * 100 - k / n * 100]], color=INK,
                        elinewidth=0.5, capsize=1.0, fmt="none")
    ax.set_xticks(range(len(LME_TYPES)), [TYPE_LABEL[t] for t in LME_TYPES], rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 112)
    ax.legend(fontsize=6.2, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.14), handlelength=1.2, columnspacing=0.8)
    style(ax)
    ax = axs[1]
    out = {"common": len(common)}
    for name, (_, label, col) in LME_FILES.items():
        rows = list(by.get(name, {}).values())
        if not rows:
            continue
        acc = sum(r["correct"] for r in rows) / len(rows) * 100
        tok = np.mean([(r.get("usage") or {}).get("answer_prompt_tokens") or r["memory_tokens"] for r in rows]
                      if name == "full_context" else [r["memory_tokens"] for r in rows])
        ax.scatter(tok, acc, color=col, s=30, zorder=3, edgecolor=INK, linewidth=0.4)
        ax.annotate(label, (tok, acc), textcoords="offset points", xytext=(4, 3 if name != "v2_full" else -8),
                    fontsize=6.0)
        out[name] = {"acc": acc, "tokens": float(tok), "n": len(rows), "k": sum(r["correct"] for r in rows)}
    out["by_type"] = {n: {t: [sum(r["correct"] for r in v.values() if r["question_type"] == t),
                              sum(1 for r in v.values() if r["question_type"] == t)] for t in LME_TYPES}
                      for n, v in by.items()}
    out["recall_ms"] = {n: float(np.median([r["recall_ms"] for r in v.values()])) for n, v in by.items()}
    out["evidence_recall"] = {n: float(np.mean([r["evidence_recall"] for r in v.values()
                                                if r.get("evidence_recall") is not None] or [float("nan")]))
                              for n, v in by.items()}
    pairs = [("v2_ranked", "mem0"), ("v2_full", "mem0"), ("v2_ranked", "v2_full"), ("v2_ranked", "chunk_rag"),
             ("v2_ranked", "full_context"), ("v2_full", "chunk_rag"), ("v2_full", "full_context"),
             ("mem0", "chunk_rag"), ("mem0", "full_context")]
    out["paired"] = {f"{a}_vs_{b}": mcnemar_rows(by[a], by[b], common) for a, b in pairs if a in by and b in by}
    ax.set_xscale("log")
    ax.set_xlabel("memory tokens given to the answer model (log)")
    ax.set_ylabel("accuracy (%)")
    style(ax, "both")
    fig.tight_layout()
    save(fig, "fig_lme")
    return out


BEAM = ROOT / "eval" / "beam" / "results"
BEAM_SYSTEMS = [("memx", "Memory-X ranked", BLUE), ("mem0", "mem0 top-30", ORANGE),
                ("full_context", "full context", "#9ca3af"), ("chunk_rag", "chunk RAG (4k)", TEAL)]


BEAM_RUN = {"memx": "lean_noexp"}  # Memory-X re-run with the current system; the others are unchanged


def fig_beam() -> dict:
    rows = {n: {r["qid"]: r for r in load_jsonl(BEAM / f"{BEAM_RUN.get(n, RUN)}-{n}.jsonl")} for n, _, _ in BEAM_SYSTEMS}
    common = set.intersection(*[set(v) for v in rows.values() if v])
    rng = np.random.default_rng(0)
    types = [("contradiction_resolution", "contradiction resolution"), ("knowledge_update", "knowledge update"),
             (None, "both")]
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.5), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axs[0]
    out: dict = {"common": len(common)}
    w = 0.2
    for j, (n, label, col) in enumerate(BEAM_SYSTEMS):
        out[n] = {}
        for i, (t, tl) in enumerate(types):
            v = np.array([rows[n][q]["score"] for q in sorted(common) if t is None or rows[n][q]["type"] == t])
            boots = [rng.choice(v, len(v)).mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            x = i + (j - 1.5) * w
            ax.bar(x, v.mean() * 100, w * 0.92, color=col, label=label if i == 0 else None)
            ax.errorbar(x, v.mean() * 100, yerr=[[v.mean() * 100 - lo * 100], [hi * 100 - v.mean() * 100]],
                        color=INK, elinewidth=0.5, capsize=1.2, fmt="none")
            out[n][tl] = {"mean": round(float(v.mean()), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                          "n": int(len(v))}
        out[n]["tokens"] = float(np.mean([rows[n][q]["memory_tokens"] for q in common]))
    ax.set_xticks(range(3), [tl.replace(" ", "\n", 1) for _, tl in types], fontsize=7.5)
    ax.set_ylabel("rubric score (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6.3, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), handlelength=1.2)
    style(ax)
    ax = axs[1]
    for n, label, col in BEAM_SYSTEMS:
        ax.scatter(out[n]["tokens"], out[n]["both"]["mean"] * 100, color=col, s=30, edgecolor=INK, linewidth=0.4,
                   zorder=3)
        ax.annotate(label, (out[n]["tokens"], out[n]["both"]["mean"] * 100), textcoords="offset points",
                    xytext=(4, 3), fontsize=6.2)
    ax.set_xscale("log")
    ax.set_xlabel("memory tokens given to the answer model (log)")
    ax.set_ylabel("rubric score (%)")
    style(ax, "both")
    fig.tight_layout()
    save(fig, "fig_beam")

    def sign(a: str, b: str) -> dict:
        wins = sum(rows[a][q]["score"] > rows[b][q]["score"] for q in common)
        losses = sum(rows[a][q]["score"] < rows[b][q]["score"] for q in common)
        n, k = wins + losses, min(wins, losses)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
        diff = np.array([rows[a][q]["score"] - rows[b][q]["score"] for q in sorted(common)])
        boots = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
        return {"wins": wins, "losses": losses, "p_sign": round(p, 4), "mean_diff": round(float(diff.mean()), 4),
                "ci": [round(float(x), 4) for x in np.percentile(boots, [2.5, 97.5])]}
    out["paired"] = {f"memx_vs_{b}": sign("memx", b) for b in ("mem0", "full_context", "chunk_rag")}
    out["paired"]["mem0_vs_full_context"] = sign("mem0", "full_context")
    return out


ZEP = [("smokefinal-lean", "Memory-X ranked", BLUE), ("smokefinal-lean_full", "Memory-X full", "#93a5f5"),
       ("smoke55-mem0_top30", "mem0 top-30", ORANGE), ("zeppilot-zep_llm", "Zep (LLM judge)", PURPLE),
       ("zeppilot-zep_jev", "Zep + Jev", RED)]


def fig_zep() -> dict:
    """10 LongMemEval questions, every system extracting with Haiku 5.5. Ingest cost per question: memx from its
    receipts (extraction tokens at list price + metered Jev cost); mem0 from its usage callback; Zep from metered
    OpenRouter cost of every call (the Jev share of zep_jev from the one question that ran alone, because concurrent
    questions shared the request counter)."""
    ids = [line.strip() for line in (LME.parent / "smoke_ids.txt").read_text().splitlines() if line.strip()]
    out: dict = {}
    for stem, label, col in ZEP:
        rows = {r["question_id"]: r for r in load_jsonl(LME / f"{stem}.jsonl")}
        rows = [rows[q] for q in ids if q in rows]
        out[stem] = {"label": label, "k": sum(r["correct"] for r in rows), "n": len(rows),
                     "tokens": float(np.mean([r["memory_tokens"] for r in rows]))}
    ids_set = set(ids)
    memx_usage = [r["usage"] for r in load_jsonl(LME / "smokefinal-lean.jsonl") if r["question_id"] in ids_set]
    memx_cost = np.mean([memx_ingest_usd(u) for u in memx_usage])
    for stem in ("smokefinal-lean", "smokefinal-lean_full"):
        out[stem]["ingest_usd"] = float(memx_cost)
    m0 = json.loads((LME / "smoke55-mem0_top30.summary.json").read_text())["extract_usage"]
    out["smoke55-mem0_top30"]["ingest_usd"] = m0["est_usd_this_run"] / max(1, out["smoke55-mem0_top30"]["n"])
    zl = [r for r in load_jsonl(LME / "zeppilot-zep_llm.jsonl") if r["usage"].get("llm_calls")]
    out["zeppilot-zep_llm"]["ingest_usd"] = float(np.mean([r["usage"]["llm_usd_est"] for r in zl]))
    out["zeppilot-zep_llm"]["llm_calls"] = float(np.mean([r["usage"]["llm_calls"] for r in zl]))
    zj = [r for r in load_jsonl(LME / "zeppilot-zep_jev.jsonl") if r["ingest_s"] > 500]  # complete single-pass ingests
    solo = next(r for r in zj if r["question_id"] == "5d3d2817")
    out["zeppilot-zep_jev"]["ingest_usd"] = float(np.mean([r["usage"]["llm_usd_est"] for r in zj])) + \
        solo["usage"]["jev_usd"]
    out["zeppilot-zep_jev"]["llm_calls"] = float(np.mean([r["usage"]["llm_calls"] for r in zj]))
    out["zep_same_question"] = {"llm": next(r["usage"] for r in zl if r["question_id"] == "5d3d2817"),
                                "jev": solo["usage"]}
    out["zep_ingest_s"] = {"llm": float(np.mean([r["ingest_s"] for r in zl])),
                           "jev": float(np.mean([r["ingest_s"] for r in zj]))}
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for stem, label, col in ZEP:
        d = out[stem]
        ax.scatter(d["ingest_usd"], d["k"], color=col, s=34, edgecolor=INK, linewidth=0.4, zorder=3)
        ax.annotate(label, (d["ingest_usd"], d["k"]), textcoords="offset points",
                    xytext=(5, -9 if stem == "smokefinal-lean" else 3), fontsize=6.2)
    ax.set_xscale("log")
    ax.set_xlim(0.08, 1.5)
    ax.set_ylim(3.5, 10.8)
    ax.set_xlabel("ingest cost per question, USD (log)")
    ax.set_ylabel("correct of 10")
    style(ax, "both")
    fig.tight_layout()
    save(fig, "fig_zep")
    return out


def fig_calibration() -> dict:
    rows = load_jsonl(ROOT / "eval" / "calibration" / "results" / "reference_labels.jsonl")
    rows = [r for r in rows if r.get("ref")]
    dest = [dict(r, kind="destructive") for r in
            load_jsonl(ROOT / "eval" / "calibration" / "results" / "reference_destructive.jsonl") if r.get("ref")]
    rows = rows + dest
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.6), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axs[0]
    edges = [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    out: dict = {}
    for kind, col, lab in [("verdict", BLUE, "all reconcile verdicts"), ("destructive", RED, "same / replaces / covers"),
                           ("entity", TEAL, "entity decisions")]:
        sub = [r for r in rows if r["kind"] == kind]
        xs, ys, los, his, ns = [], [], [], [], []
        for lo, hi in zip(edges, edges[1:]):
            b = [r for r in sub if lo <= r["jev_p"] < hi]
            if len(b) < 5:
                continue
            k = sum(r["ref"] == r["jev"] for r in b)
            l, h = wilson(k, len(b))
            xs.append(np.mean([r["jev_p"] for r in b])); ys.append(k / len(b)); los.append(l); his.append(h)
            ns.append(len(b))
        ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(los), np.array(his) - np.array(ys)], fmt="o-", color=col,
                    ms=4, capsize=2, elinewidth=0.8, linewidth=1.1, label=f"{lab} (n={len(sub)})")
        out[kind] = {"n": len(sub), "agree": sum(r["ref"] == r["jev"] for r in sub),
                     "bins": [{"p": round(x, 3), "agree": round(y, 3), "n": n} for x, y, n in zip(xs, ys, ns)]}
    ax.plot([0.25, 1], [0.25, 1], color=MUTED, linestyle=":", linewidth=0.8)
    ax.set_xlabel("Jev probability of its chosen answer")
    ax.set_ylabel("agreement with Sonnet 5.5")
    ax.axvspan(0.25, 0.5, color="#f1f2f4", zorder=0)
    ax.text(0.375, 0.05, "replace / covers\nnot acted on", ha="center", fontsize=6, color=MUTED)
    ax.set_xlim(0.25, 1.02); ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=6.3, loc="lower right")
    ax.set_title("a  Confidence vs agreement", loc="left", fontweight="bold")
    style(ax, "both")
    ax = axs[1]
    labs = ["same", "candidate_replaces", "neighbor_covers", "related", "unrelated"]
    short = ["same", "replaces", "covers", "related", "unrel."]
    sub = [r for r in rows if r["kind"] in ("verdict", "destructive")]
    m = np.array([[sum(1 for r in sub if r["jev"] == a and r["ref"] == b) for b in labs] for a in labs], dtype=float)
    cmap = LinearSegmentedColormap.from_list("m", ["#ffffff", "#fde2e2", RED])
    ax.imshow(m, cmap=cmap)
    for i in range(5):
        for j in range(5):
            if m[i, j]:
                ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if m[i, j] > m.max() * 0.6 else INK)
    ax.set_xticks(range(5), short, fontsize=6.5, rotation=25, ha="right"); ax.set_yticks(range(5), short, fontsize=7)
    ax.set_xlabel("Sonnet 5.5 label"); ax.set_ylabel("Jev verdict")
    ax.spines[:].set_visible(False); ax.tick_params(length=0)
    ax.set_title(f"b  Reconcile verdicts, Jev vs Sonnet (n={len(sub)})", loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_calibration")
    out["confusion"] = m.tolist()
    return out


def costs() -> dict:
    """Extraction spend per question. memx: tokens recorded on each ingest. mem0: the usage callback in run_mem0,
    summed over every run that ingested questions into the shared mem0 store (Haiku 4.5 at $1 / $5 per Mtok)."""
    rows = q("SELECT split_part(scope::text, '.', 4) AS qid, sum(tokens_in) AS tin, sum(tokens_out) AS tout, "
             "count(*) AS sessions FROM memx_ingest WHERE scope::text LIKE 'memory.lme.base.%%' GROUP BY 1")
    per_q = [(r["tin"] or 0) / 1e6 * 1.0 + (r["tout"] or 0) / 1e6 * 5.0 for r in rows]
    jev = q("SELECT d.step, count(*) AS n FROM memx_decision d JOIN memx_ingest i ON i.id = d.ingest_id "
            "WHERE i.scope::text LIKE 'memory.lme.base.%%' GROUP BY 1")
    m0_usd, m0_q = 0.0, 0
    for p in LME.glob("*mem0_top30.summary.json"):
        s = json.loads(p.read_text())
        u = s.get("extract_usage") or {}
        m0_usd += u.get("est_usd_this_run", 0.0)
    m0_q = len(list((LME / "mem0_progress").glob("*.json")))
    return {"memx_questions": len(rows), "memx_usd_per_q": float(np.mean(per_q)) if per_q else None,
            "memx_tokens_in_per_q": float(np.mean([r["tin"] or 0 for r in rows])) if rows else None,
            "memx_jev_requests": {r["step"]: r["n"] for r in jev},
            "memx_jev_requests_per_q": sum(r["n"] for r in jev) / max(1, len(rows)),
            "mem0_usd_total": round(m0_usd, 2), "mem0_questions_ingested": m0_q,
            "mem0_usd_per_q": round(m0_usd / max(1, m0_q), 3)}


ORG = ROOT / "eval" / "orgprobe" / "results"


def fig_efficiency() -> dict:
    """Memory tokens handed to the answering model per question, Memory-X ranked vs mem0 top-30, with each
    system's score, on every benchmark in the report."""
    def lme(stem):
        r = load_jsonl(LME / f"{stem}.jsonl")
        return float(np.mean([x["memory_tokens"] for x in r])), 100 * sum(x["correct"] for x in r) / len(r)

    def beam(sys):
        r = load_jsonl(BEAM / f"{BEAM_RUN.get(sys, RUN)}-{sys}.jsonl")
        return float(np.mean([x["memory_tokens"] for x in r])), 100 * float(np.mean([x["score"] for x in r]))

    def org(name):
        r = load_jsonl(ORG / f"{name}.jsonl")
        return float(np.mean([x["tokens"] for x in r])), 100 * sum(x["score"] for x in r) / len(r)

    ids = {line.strip() for line in (LME.parent / "smoke_ids.txt").read_text().splitlines() if line.strip()}

    def smoke(stem):
        r = [x for x in load_jsonl(LME / f"{stem}.jsonl") if x["question_id"] in ids]
        return float(np.mean([x["memory_tokens"] for x in r])), 100 * sum(x["correct"] for x in r) / len(r)

    org_memx = "final-memx_ranked_2k"
    bench = [("LongMemEval\n(82)", lme("final-lean"), lme("full55-mem0_top30")),
             ("BEAM\n(80)", beam("memx"), beam("mem0")),
             ("Zep set\n(10)", smoke("smokefinal-lean"), smoke("smoke55-mem0_top30")),
             ("OrgProbe\n(17)", org(org_memx), org("paper-mem0_top30"))]
    fig, ax = plt.subplots(figsize=(6.9, 2.3))
    w = 0.36
    out = {}
    for i, (name, (mt, ma), (zt, za)) in enumerate(bench):
        ax.bar(i - w / 2, mt, w * 0.95, color=BLUE, label="Memory-X ranked" if i == 0 else None)
        ax.bar(i + w / 2, zt, w * 0.95, color=ORANGE, label="mem0 top-30" if i == 0 else None)
        ax.text(i - w / 2, mt * 1.08, f"{mt:,.0f}\n{ma:.0f}%", ha="center", va="bottom", fontsize=6.3)
        ax.text(i + w / 2, zt * 1.08, f"{zt:,.0f}\n{za:.0f}%", ha="center", va="bottom", fontsize=6.3)
        ax.text(i, max(mt, zt) * 3.4, f"{zt / mt:.1f}x fewer tokens", ha="center", fontsize=7, fontweight="bold",
                color=BLUE)
        out[name.replace("\n", " ")] = {"memx_tokens": mt, "memx_score": ma, "mem0_tokens": zt, "mem0_score": za,
                                         "ratio": zt / mt}
    ax.set_yscale("log")
    ax.set_ylim(15, 20000)
    ax.set_xticks(range(len(bench)), [b[0] for b in bench], fontsize=7)
    ax.set_ylabel("memory tokens per answer (log)")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False)
    style(ax)
    fig.tight_layout()
    save(fig, "fig_efficiency")
    return out


def main() -> None:
    summary = {}
    for fn in (fig_sort, fig_reconcile, fig_verdict_prob, fig_entities, fig_recall, fig_decay, fig_latency, fig_lme,
               fig_beam, fig_zep, fig_calibration, costs, fig_efficiency):
        try:
            res = fn()
            if res:
                summary[fn.__name__] = res
        except Exception as exc:  # noqa: BLE001 - one missing dataset should not stop the rest
            print(f"! {fn.__name__}: {type(exc).__name__}: {exc}")
    (OUT / "numbers.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
