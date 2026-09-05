#!/usr/bin/env python3
"""Concentration statistics for a captured corpus.

A clip count says almost nothing. The Milestone 5.5 training corpus was
"12 clips", which sounds like twelve independent observations and was closer to
eight: half of it was rail footage, three clips came from one operator in one
German city, and two more were consecutive rolls from a single afternoon. The
model saw far less variety than the number implied.

So this reports the quantities that actually bound what a corpus can teach:
how much of it one creator, one shoot or one category accounts for. The headline
numbers are the largest shares, because a corpus is only as diverse as its
most dominant group.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_schema import CATEGORIES, normalise_creator  # noqa: E402


def herfindahl(counts: list[int]) -> float:
    """Sum of squared shares: 1.0 is one group, 1/n is perfectly even.

    Reported alongside the largest share because the two fail differently. A
    corpus can have a modest largest share and still be concentrated across a
    handful of groups, which the maximum alone would not reveal.
    """
    total = sum(counts)
    return sum((c / total) ** 2 for c in counts) if total else 0.0


def effective_n(counts: list[int]) -> float:
    """1/HHI — how many *equally sized* groups the corpus behaves like."""
    h = herfindahl(counts)
    return 1.0 / h if h > 0 else 0.0


def report(manifest: dict) -> dict:
    clips = manifest.get("clips", [])
    n = len(clips)
    by_creator = collections.Counter(normalise_creator(c.get("creator")) for c in clips)
    by_shoot = collections.Counter(c.get("shootId") or "?" for c in clips)
    by_cat = collections.Counter(c.get("category") or "unlabelled" for c in clips)
    by_res = collections.Counter(f"{c.get('source_width')}x{c.get('source_height')}" for c in clips)
    by_fps = collections.Counter(str(round(float(c.get("fps") or 0), 2)) for c in clips)
    by_lic = collections.Counter(c.get("licence") or "?" for c in clips)

    creators_per_cat = {
        cat: len({normalise_creator(c.get("creator")) for c in clips if c.get("category") == cat})
        for cat in sorted(by_cat)
    }
    duration = sum(float(c.get("duration") or 0) for c in clips)

    def share(counter: collections.Counter) -> dict:
        if not counter:
            return {"largestShare": 0.0, "largestGroup": None, "groups": 0, "effectiveN": 0.0}
        top, cnt = counter.most_common(1)[0]
        return {
            "largestShare": cnt / n,
            "largestGroup": top,
            "largestCount": cnt,
            "groups": len(counter),
            "effectiveN": effective_n(list(counter.values())),
        }

    return {
        "schema": "aethervsr.corpus-report/1",
        "role": manifest.get("role"),
        "clips": n,
        "creators": len(by_creator),
        "shoots": len(by_shoot),
        "categories": len(by_cat),
        "totalDurationSeconds": duration,
        "concentration": {
            "creator": share(by_creator),
            "shoot": share(by_shoot),
            "category": share(by_cat),
        },
        "clipsPerCategory": dict(by_cat.most_common()),
        "creatorsPerCategory": creators_per_cat,
        "clipsPerCreator": dict(by_creator.most_common()),
        "clipsPerShoot": dict(by_shoot.most_common()),
        "resolutionDistribution": dict(by_res.most_common()),
        "frameRateDistribution": dict(by_fps.most_common()),
        "licenceDistribution": dict(by_lic.most_common()),
        "missingCategories": [c for c in CATEGORIES if c not in by_cat],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Corpus concentration report.")
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    reports = {}
    for path in args.manifests:
        with open(path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        r = report(manifest)
        reports[path] = r
        c = r["concentration"]
        print(f"\n{path}  [{r['role']}]", file=sys.stderr)
        print(f"  {r['clips']} clips · {r['creators']} creators · {r['shoots']} shoots · "
              f"{r['categories']} categories · {r['totalDurationSeconds']/60:.1f} min", file=sys.stderr)
        for level in ("creator", "shoot", "category"):
            s = c[level]
            print(f"  largest {level:<9} {s['largestShare']:>6.1%}  "
                  f"effective-n {s['effectiveN']:>5.1f} of {s['groups']:<3}  "
                  f"({str(s['largestGroup'])[:38]})", file=sys.stderr)
        if r["clipsPerCategory"]:
            print(f"  categories: {r['clipsPerCategory']}", file=sys.stderr)
        if r["missingCategories"]:
            print(f"  MISSING: {r['missingCategories']}", file=sys.stderr)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(reports, fh, indent=1)
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
