#!/usr/bin/env python3
"""Compare all-I against GOP at matched delivered input quality.

At equal CRF the two compression contexts do not deliver equal input quality -
GOP is up to 1.8 dB better per frame - so comparing them at equal CRF measures
the encoder as much as the model. This re-compares them as a function of
delivered input PSNR instead, which is the quantity the model actually faces.

The first version of this analysis clamped: two of its six points sat below the
GOP curve's measured range and silently reused the endpoint value, which is a
flat extrapolation. Independent review showed the clamp was load-bearing -
extrapolating those points linearly instead flips the sign at the lowest one,
which is precisely the heavy-compression regime the milestone is about.

So this only reports the overlapping range where both curves were actually
measured, and refuses to state a comparison outside it. An honest four points
beat six with two invented.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def curve(summary: dict, structure: str, crfs: list[int]) -> list[tuple[float, float]]:
    pts = []
    for c in crfs:
        key = f"crf{c}_{structure}"
        if key in summary:
            pts.append((summary[key]["meanInputPsnr"], summary[key]["meanDelta"]))
    return sorted(pts)


def interpolate(pts: list[tuple[float, float]], x: float) -> float | None:
    """Linear interpolation, or None outside the measured range. Never clamps."""
    if x < pts[0][0] or x > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            return y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Matched-input-quality all-I vs GOP comparison.")
    ap.add_argument("--diagnostic", default="results/gop-diagnostic.json")
    ap.add_argument("--crfs", default="18,22,26,30,34,38")
    ap.add_argument("--out", default="results/gop-matched-quality.json")
    args = ap.parse_args()

    with open(args.diagnostic, encoding="utf-8") as fh:
        summary = json.load(fh)["summary"]
    crfs = [int(c) for c in args.crfs.split(",") if c.strip()]
    all_i = curve(summary, "all_i", crfs)
    gop = curve(summary, "gop", crfs)

    lo = max(all_i[0][0], gop[0][0])
    hi = min(all_i[-1][0], gop[-1][0])

    matched, excluded = [], []
    for x, y in all_i:
        g = interpolate(gop, x)
        if g is None or not (lo <= x <= hi):
            excluded.append({"inputPsnr": x, "allIDelta": y,
                             "reason": "outside the range where both curves were measured"})
            continue
        matched.append({"inputPsnr": x, "allIDelta": y, "gopDelta": g, "gopMinusAllI": g - y})

    easier = sum(1 for m in matched if m["gopMinusAllI"] > 0)
    report = {
        "schema": "aethervsr.gop-matched-quality/2",
        "question": "At equal delivered input quality, is GOP-compressed input harder for the model than all-I?",
        "method": ("Linear interpolation of the GOP delta-vs-input-PSNR curve, evaluated only at "
                   "all-I points inside the overlapping measured range. Points outside are "
                   "reported as excluded rather than extrapolated."),
        "measuredRange": {"allI": [all_i[0][0], all_i[-1][0]], "gop": [gop[0][0], gop[-1][0]],
                          "overlap": [lo, hi]},
        "answer": (f"No, over the measured overlap: GOP input is easier at {easier}/{len(matched)} "
                   f"matched points between {lo:.2f} and {hi:.2f} dB input PSNR."),
        "limitation": (f"{len(excluded)} all-I point(s) fall below the GOP curve's measured floor "
                       f"({gop[0][0]:.2f} dB) and are excluded. The comparison therefore says "
                       f"nothing about the very lowest input qualities, which is where a "
                       f"CRF 38-and-beyond regime would sit."),
        "matched": matched,
        "excluded": excluded,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    print(f"  overlap {lo:.2f}..{hi:.2f} dB input PSNR\n", file=sys.stderr)
    print(f"  {'input dB':>9}{'all-I':>11}{'GOP':>11}{'GOP-allI':>11}", file=sys.stderr)
    for m in matched:
        print(f"  {m['inputPsnr']:>9.2f}{m['allIDelta']:>+11.4f}{m['gopDelta']:>+11.4f}"
              f"{m['gopMinusAllI']:>+11.4f}", file=sys.stderr)
    for e in excluded:
        print(f"  {e['inputPsnr']:>9.2f}{e['allIDelta']:>+11.4f}{'excluded':>11}{'-':>11}",
              file=sys.stderr)
    print(f"\n  GOP easier at {easier}/{len(matched)} measured matched points", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
