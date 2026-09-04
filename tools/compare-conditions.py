#!/usr/bin/env python3
"""Compare two training conditions with statistics that n=5 can actually support.

Milestone 4.5 first reported an effect as "8.9 sigma", computed by dividing a
difference in means by a standard deviation estimated from five runs. That is
not a defensible statement: with n=5 per arm the standard deviation is itself
uncertain by roughly a third, and a sigma count implies a tail probability the
data cannot resolve. Independent review put the honest figure near 2.6-3.9
sigma depending on the test.

So this reports, for each condition pair:

  * both group means and sample standard deviations, with every run listed
  * Welch's t-test, which does not assume equal variances
  * an exact two-sided permutation test over all C(10,5) = 252 relabellings

The permutation test is the one to trust. It assumes only exchangeability under
the null, and with five per arm its smallest attainable two-sided p-value is
2/252 = 0.0079 - so **no comparison of five against five can report better than
about p = 0.008**, whatever the means look like. Any claim beyond that is an
artefact of the parametric model, not evidence.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import statistics as st
import subprocess
import sys

PY = sys.executable
EVAL = os.path.join(os.path.dirname(__file__), "evaluate.py")


def deltas(models: list[str], which: str, profile: str, limit: int, seed: int) -> list[float]:
    out = []
    for model in models:
        r = subprocess.run(
            [PY, EVAL, "--model", model, "--set", which, "--profile", profile,
             "--limit", str(limit), "--seed", str(seed), "--device", "cpu", "--json"],
            capture_output=True, text=True, check=False,
        )
        if r.returncode != 0:
            raise SystemExit(f"evaluate.py failed for {model}:\n{r.stderr[-1200:]}")
        out.append(json.loads(r.stdout)["neuralOverCatmullRom"]["psnrDb"])
    return out


def welch(a: list[float], b: list[float]) -> tuple[float, float, float]:
    na, nb = len(a), len(b)
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return float("inf"), 0.0, float("inf")
    t = (st.fmean(a) - st.fmean(b)) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    # Two-sided p from the t distribution via the regularised incomplete beta.
    x = df / (df + t * t)
    p = _betainc(df / 2, 0.5, x)
    return t, p, df


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta, continued fraction. Enough for a p-value."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 200):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + numerator / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-10:
            break
    return front * (f - 1.0)


def permutation(a: list[float], b: list[float]) -> tuple[float, int, float]:
    """Exact two-sided permutation test. Returns p, number of relabellings, floor."""
    pooled = a + b
    observed = abs(st.fmean(a) - st.fmean(b))
    n = len(a)
    count = total = 0
    for combo in itertools.combinations(range(len(pooled)), n):
        left = [pooled[i] for i in combo]
        right = [pooled[i] for i in range(len(pooled)) if i not in combo]
        total += 1
        if abs(st.fmean(left) - st.fmean(right)) >= observed - 1e-12:
            count += 1
    return count / total, total, 2.0 / total


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two training conditions honestly.")
    ap.add_argument("--a", default="models/box/seed*.json")
    ap.add_argument("--b", default="models/realistic/seed*.json")
    ap.add_argument("--labels", default="box,realistic")
    ap.add_argument("--set", default="independent")
    ap.add_argument("--profiles", default="box,h264_high,h264_typical,h264_poor")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/condition-comparison.json")
    args = ap.parse_args()

    la, lb = [s.strip() for s in args.labels.split(",")]
    ma, mb = sorted(glob.glob(args.a)), sorted(glob.glob(args.b))
    if not ma or not mb:
        raise SystemExit("no models matched")

    report = {
        "schema": "aethervsr.condition-comparison/1",
        "set": args.set,
        "images": args.limit,
        "arms": {la: [os.path.basename(m) for m in ma], lb: [os.path.basename(m) for m in mb]},
        "note": (
            f"Exact permutation floor for {len(ma)} vs {len(mb)} is "
            f"2/C({len(ma)+len(mb)},{len(ma)}) - no comparison at this sample size can "
            "report a smaller two-sided p, however large the difference in means looks."
        ),
        "profiles": {},
    }

    print(f"{'profile':<15}{la+' mean':>13}{lb+' mean':>15}{'diff':>9}{'Welch p':>11}{'perm p':>10}")
    for profile in [p.strip() for p in args.profiles.split(",") if p.strip()]:
        da = deltas(ma, args.set, profile, args.limit, args.seed)
        db = deltas(mb, args.set, profile, args.limit, args.seed)
        t, pw, df = welch(da, db)
        pp, total, floor = permutation(da, db)
        report["profiles"][profile] = {
            la: {"runs": da, "mean": st.fmean(da), "std": st.stdev(da)},
            lb: {"runs": db, "mean": st.fmean(db), "std": st.stdev(db)},
            "difference": st.fmean(db) - st.fmean(da),
            "welch": {"t": t, "p": pw, "df": df},
            "permutation": {"p": pp, "relabellings": total, "smallestAttainableP": floor},
        }
        print(
            f"{profile:<15}{st.fmean(da):>13.3f}{st.fmean(db):>15.3f}"
            f"{st.fmean(db)-st.fmean(da):>+9.3f}{pw:>11.4f}{pp:>10.4f}"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nsmallest attainable two-sided p at this n: {report['profiles'][list(report['profiles'])[0]]['permutation']['smallestAttainableP']:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
