#!/usr/bin/env python3
"""Apply the pre-registered Milestone 7 selection rule to a screening run.

Written before the replacement models exist, so the metric, the ranking and the
tie-break cannot be chosen after seeing the numbers. It implements exactly what
`docs/M7-PREREGISTRATION.md` says and nothing else:

  * statistical unit is the clip, frames averaged within a clip first
  * the metric is per-clip mean delta against the *frozen production model*,
    paired on identical clip x CRF cells
  * highest mean wins; rungs within 0.01 dB are a tie, and a tie resolves to the
    simpler rung, with R0 beating everything

It also reports what the rule does not use but the milestone must disclose:
per-CRF and per-category breakdowns, seed sd, the weak cells Milestone 6 left
negative, and an exact permutation test with its own floor stated - at three
seeds per arm the smallest two-sided p an unpaired permutation can return is
0.100, so a "non-significant" result there carries almost no information and
saying so is part of the report.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import statistics as st
import sys

TIE_BAND_DB = 0.01
SIMPLICITY_ORDER = ["R0", "R1", "R2", "R3", "R4"]
# Milestone 6 left motion@CRF34 (-0.0907) and texture@CRF34 (-0.0069) negative.
# Those numbers came from the *confirmation* corpus, and none of its clips
# appear in the validation corpus screened here, so what follows is the
# same-named category cell measured on different footage - a weather report for
# a neighbouring town, not the same reading. It is tracked because a rung that
# hurts these categories on validation is worth seeing early, but it neither
# confirms nor refutes the Milestone 6 cells, and only a confirmation run can.
WEAK_CELLS = [("motion", 34), ("texture", 34)]


def per_clip_delta_vs(model: dict, baseline_psnr: dict[tuple[str, int], float]) -> dict[str, float]:
    """Mean over CRFs within a clip, of candidate-minus-baseline PSNR."""
    by_clip: dict[str, list[float]] = collections.defaultdict(list)
    for cell in model["perCell"]:
        key = (cell["clip"], cell["crf"])
        if key not in baseline_psnr:
            raise KeyError(f"baseline has no cell {key}; the runs are not paired")
        by_clip[cell["clip"]].append(cell["psnr"] - baseline_psnr[key])
    return {clip: st.fmean(v) for clip, v in by_clip.items()}


def exact_permutation_p(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-sided p over all label assignments, with the attainable floor."""
    pool = a + b
    n = len(a)
    obs = abs(st.fmean(b) - st.fmean(a))
    hits = total = 0
    for combo in itertools.combinations(range(len(pool)), n):
        x = [pool[i] for i in combo]
        y = [pool[i] for i in range(len(pool)) if i not in combo]
        total += 1
        if abs(st.fmean(y) - st.fmean(x)) >= obs - 1e-12:
            hits += 1
    return hits / total, 2.0 / total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screening", required=True)
    ap.add_argument("--baseline-key", default="aethersr-c16d2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.screening, encoding="utf-8") as fh:
        data = json.load(fh)
    models = data["models"]
    if args.baseline_key not in models:
        print(f"  frozen production model {args.baseline_key!r} is not in the run; "
              f"the pre-registered metric cannot be computed", file=sys.stderr)
        return 1

    baseline_psnr = {(c["clip"], c["crf"]): c["psnr"] for c in models[args.baseline_key]["perCell"]}

    arms: dict[str, list[dict]] = collections.defaultdict(list)
    for name, model in models.items():
        if name == args.baseline_key:
            continue
        rung = name.split("-")[0]
        deltas = per_clip_delta_vs(model, baseline_psnr)
        by_crf = collections.defaultdict(list)
        by_cat = collections.defaultdict(list)
        for cell in model["perCell"]:
            d = cell["psnr"] - baseline_psnr[(cell["clip"], cell["crf"])]
            by_crf[cell["crf"]].append(d)
            by_cat[(cell["category"], cell["crf"])].append(d)
        arms[rung].append({
            "model": name,
            "meanDelta": st.fmean(deltas.values()),
            "byCrf": {str(k): st.fmean(v) for k, v in sorted(by_crf.items())},
            "byCategoryCrf": {f"{c}@{q}": st.fmean(v) for (c, q), v in sorted(by_cat.items())},
        })

    summary = {}
    for rung, runs in arms.items():
        vals = [r["meanDelta"] for r in runs]
        summary[rung] = {
            "seeds": len(vals),
            "mean": st.fmean(vals),
            "sd": st.stdev(vals) if len(vals) > 1 else 0.0,
            "perSeed": vals,
            "byCrf": {k: st.fmean([r["byCrf"][k] for r in runs]) for k in runs[0]["byCrf"]},
            "weakCellsOnValidation": {
                f"{cat}@{crf}": st.fmean(
                    [r["byCategoryCrf"].get(f"{cat}@{crf}", float("nan")) for r in runs]
                )
                for cat, crf in WEAK_CELLS
            },
        }

    ranked = sorted(summary, key=lambda r: summary[r]["mean"], reverse=True)
    top = ranked[0]
    tied = [r for r in ranked if summary[top]["mean"] - summary[r]["mean"] <= TIE_BAND_DB]
    winner = min(tied, key=lambda r: SIMPLICITY_ORDER.index(r) if r in SIMPLICITY_ORDER else 99)

    stats = {}
    if "R0" in summary:
        for rung in ranked:
            if rung == "R0":
                continue
            p, floor = exact_permutation_p(summary["R0"]["perSeed"], summary[rung]["perSeed"])
            stats[f"{rung}_vs_R0"] = {
                "deltaDb": summary[rung]["mean"] - summary["R0"]["mean"],
                "p": p,
                "attainableFloor": floor,
            }

    report = {
        "schema": "aethervsr.m7-screen/1",
        "metric": "per-clip mean PSNR delta vs the frozen production model, paired on clip x CRF",
        "statisticalUnit": data.get("statisticalUnit"),
        "tieBandDb": TIE_BAND_DB,
        "ranking": ranked,
        "argmax": top,
        "tiedWithArgmax": tied,
        "winner": winner,
        "winnerRationale": (
            f"{top} has the highest mean; {sorted(tied)} lie within {TIE_BAND_DB} dB of it, "
            f"and the pre-registered tie-break takes the simplest of those, {winner}."
            if len(tied) > 1 else f"{top} wins outright, no rung within {TIE_BAND_DB} dB."
        ),
        "nullResult": winner == "R0",
        "whatANullDoesNotMean": (
            "With 3 seeds per arm the exact unpaired permutation test cannot return below "
            f"{2.0 / 20:.3f}, so it can never reach p<0.05 no matter how large the effect. "
            "Selecting R0 therefore means 'no gain was detected at this budget under this "
            "rule'. It is not evidence that the rungs are optimization-equivalent, and it "
            "does not bound how large an undetected effect could be."
        ),
        "budgetCaveat": (
            "Screening runs the pre-registered 16,200-step budget, which is about a fifth of "
            "the 81,180 steps the shipped model was trained for. A reparameterization benefit "
            "that only appears late in a schedule would not be visible here. Raising the "
            "budget would be a different experiment and must be declared before it is run."
        ),
        "arms": summary,
        "significance": stats,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    print(f"  {'rung':<6}{'n':>2}{'mean dB':>10}{'sd':>8}   per-seed", file=sys.stderr)
    for rung in ranked:
        s = summary[rung]
        print(f"  {rung:<6}{s['seeds']:>2}{s['mean']:>10.4f}{s['sd']:>8.4f}   "
              f"{' '.join(f'{x:+.4f}' for x in s['perSeed'])}", file=sys.stderr)
    for name, s in stats.items():
        print(f"  {name}: {s['deltaDb']:+.4f} dB  p={s['p']:.3f} (floor {s['attainableFloor']:.3f})",
              file=sys.stderr)
    print(f"\n  winner: {winner}"
          f"{'  -> NULL RESULT, current architecture retained' if winner == 'R0' else ''}",
          file=sys.stderr)
    print(f"  {report['winnerRationale']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
