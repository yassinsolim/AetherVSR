#!/usr/bin/env python3
"""Paired clip-level statistics for a held-out corpus.

Consumes the per-cell output of tools/m6-validate.py and produces the numbers
the confirmation-set acceptance criteria are written against: per compression
tier, per content class, per clip, with paired bootstrap intervals and exact
permutation tests.

The statistical unit is the clip. Eight frames of one video are eight views of
one scene, and treating them as independent observations would shrink every
interval by a factor of roughly the square root of eight while adding no
information. Frames are averaged within a clip before anything is compared.

Comparisons are paired throughout. The same decoded frames are scored by every
method, so pairing removes all between-clip variance - which dominates - and
leaves the difference the question is actually about. Two overlapping intervals
against a shared baseline are not evidence of no difference between two models;
the paired difference is computed directly instead.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import random
import statistics as st
import sys

BOOTSTRAP = 20000
SEED = 12345


def paired(values: list[float]) -> dict:
    """Mean, bootstrap CI and exact permutation p for one paired sample."""
    n = len(values)
    mean = st.fmean(values)
    rnd = random.Random(SEED)
    boots = sorted(st.fmean(rnd.choices(values, k=n)) for _ in range(BOOTSTRAP))
    lo, hi = boots[int(0.025 * BOOTSTRAP)], boots[int(0.975 * BOOTSTRAP)]
    if n <= 20:
        # Exact sign-flip enumeration. At n=17 that is 131,072 assignments,
        # which is cheap and avoids quoting a sampled approximation.
        hits = sum(
            1 for signs in itertools.product((1, -1), repeat=n)
            if abs(st.fmean(v * s for v, s in zip(values, signs))) >= abs(mean) - 1e-12
        )
        p = hits / 2 ** n
        exact = True
    else:
        rnd2 = random.Random(SEED)
        hits = sum(
            1 for _ in range(BOOTSTRAP)
            if abs(st.fmean(v * rnd2.choice((1, -1)) for v in values)) >= abs(mean) - 1e-12
        )
        p = hits / BOOTSTRAP
        exact = False
    return {
        "clips": n, "mean": mean, "median": st.median(values),
        "ci95": [lo, hi], "wins": sum(1 for v in values if v > 0),
        "permutationP": p, "permutationExact": exact,
        "smallestAttainableP": 2 ** (1 - n) if exact else 1 / BOOTSTRAP,
    }


def per_clip(cells: list[dict], key: str = "delta") -> dict[str, float]:
    by = collections.defaultdict(list)
    for c in cells:
        by[c["clip"]].append(c[key])
    return {k: st.fmean(v) for k, v in by.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired clip-level statistics.")
    ap.add_argument("--validation", required=True, help="m6-validate.py output")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", default=None,
                    help="second model to compare against, e.g. the shipped default")
    ap.add_argument("--tiers", default="18,26,34")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.validation, encoding="utf-8") as fh:
        data = json.load(fh)
    models = data["models"]
    if args.candidate not in models:
        raise SystemExit(f"{args.candidate!r} not in {sorted(models)}")
    cand = models[args.candidate]["perCell"]
    tiers = [int(t) for t in args.tiers.split(",") if t.strip()]

    report = {
        "schema": "aethervsr.m6-confirm-stats/1",
        "candidate": args.candidate,
        "baseline": args.baseline,
        "statisticalUnit": "clip; frames averaged within a clip before comparison",
        "comparison": "paired: every method scores the same decoded frames",
        "vsCatmullRom": {}, "vsBaseline": {},
        "byCategory": {}, "byCategoryVsBaseline": {},
        "perClip": {},
    }

    for tier in tiers:
        sel = [c for c in cand if c["crf"] == tier]
        report["vsCatmullRom"][f"crf{tier}"] = paired(list(per_clip(sel).values()))
    report["vsCatmullRom"]["allTiers"] = paired(list(per_clip(cand).values()))

    cats = sorted({c["category"] for c in cand})
    for cat in cats:
        sel = [c for c in cand if c["category"] == cat]
        report["byCategory"][cat] = paired(list(per_clip(sel).values()))

    if args.baseline and args.baseline in models:
        base = {(c["clip"], c["crf"]): c["delta"] for c in models[args.baseline]["perCell"]}
        diffs = [{**c, "delta": c["delta"] - base[(c["clip"], c["crf"])]}
                 for c in cand if (c["clip"], c["crf"]) in base]
        for tier in tiers:
            sel = [c for c in diffs if c["crf"] == tier]
            report["vsBaseline"][f"crf{tier}"] = paired(list(per_clip(sel).values()))
        report["vsBaseline"]["allTiers"] = paired(list(per_clip(diffs).values()))
        for cat in cats:
            sel = [c for c in diffs if c["category"] == cat]
            report["byCategoryVsBaseline"][cat] = paired(list(per_clip(sel).values()))

    report["perClip"] = {k: round(v, 4) for k, v in sorted(per_clip(cand).items())}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    print(f"  {args.candidate}  vs production Catmull-Rom\n", file=sys.stderr)
    print(f"  {'tier':<10}{'mean dB':>10}{'CI95':>22}{'wins':>9}{'p':>9}", file=sys.stderr)
    for k, v in report["vsCatmullRom"].items():
        print(f"  {k:<10}{v['mean']:>+10.4f}  [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]"
              f"{str(v['wins'])+'/'+str(v['clips']):>9}{v['permutationP']:>9.4f}", file=sys.stderr)
    print(f"\n  {'category':<12}{'mean dB':>10}{'wins':>9}", file=sys.stderr)
    for k, v in report["byCategory"].items():
        print(f"  {k:<12}{v['mean']:>+10.4f}{str(v['wins'])+'/'+str(v['clips']):>9}", file=sys.stderr)
    if report["vsBaseline"]:
        print(f"\n  vs {args.baseline}\n", file=sys.stderr)
        for k, v in report["vsBaseline"].items():
            print(f"  {k:<10}{v['mean']:>+10.4f}  [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]"
                  f"{str(v['wins'])+'/'+str(v['clips']):>9}{v['permutationP']:>9.4f}", file=sys.stderr)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
