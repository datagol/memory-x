"""Suggest similarity bands from logged Jev verdicts.

band_high: lowest similarity where P(same) >= 0.98, so auto-dedupe above it is safe.
band_low:  highest similarity where P(unrelated) >= 0.98, so skipping the judge below it is safe.

    python -m eval.longmemeval.calibrate [--bin 0.02]
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict

import psycopg
from psycopg.rows import dict_row

from eval.longmemeval.run import BENCH_DSN
from memx import MemxConfig


async def main_async(bin_width: float, target: float) -> None:
    cfg = MemxConfig()
    async with await psycopg.AsyncConnection.connect(BENCH_DSN, row_factory=dict_row) as conn:
        cur = await conn.execute(
            "SELECT similarity, result->>'label' AS label FROM memx_decision "
            "WHERE step = 'verdict' AND similarity IS NOT NULL"
        )
        rows = await cur.fetchall()
    if not rows:
        print("no verdicts logged yet; run some ingests first")
        return
    bins: dict[int, list[str]] = defaultdict(list)
    for r in rows:
        bins[int(r["similarity"] / bin_width)].append(r["label"])
    print(f"{len(rows)} verdicts\n  sim      n   same  unrel")
    stats = []
    for b in sorted(bins):
        labels = bins[b]
        stats.append((b * bin_width, len(labels), labels.count("same") / len(labels),
                      labels.count("unrelated") / len(labels)))
        print(f"  {stats[-1][0]:4.2f}  {len(labels):5d}  {stats[-1][2]:5.2f}  {stats[-1][3]:5.2f}")
    # band_low: top of the contiguous run of bins, from the bottom, where nearly everything is unrelated.
    low = None
    for lo, n, _, p_unrel in stats:
        if n < 5 or p_unrel < target:
            break
        low = lo + bin_width
    # band_high: bottom of the contiguous run of bins, from the top, where nearly everything is the same.
    high = None
    for lo, n, p_same, _ in reversed(stats):
        if n < 5 or p_same < target:
            break
        high = lo
    print(f"\nsuggested band_low={low if low is not None else cfg.band_low:.2f} "
          f"band_high={high if high is not None else cfg.band_high:.2f} (current {cfg.band_low}/{cfg.band_high})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bin", type=float, default=0.02)
    p.add_argument("--target", type=float, default=0.98)
    a = p.parse_args()
    asyncio.run(main_async(a.bin, a.target))


if __name__ == "__main__":
    main()
