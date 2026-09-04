#!/usr/bin/env python3
"""Prove the captured benchmark is aligned, rather than asserting it.

A one-frame or one-pixel offset, or a limited/full range mismatch, would
dominate PSNR and could manufacture either a win or a loss. Four checks, each
designed so that the failure it looks for would actually show up:

1. **Geometry** — every master is 2560x1440, every decoded frame 1280x720, and
   the counts match. Cheap, and catches a truncated encode.

2. **Frame correspondence** — decoded frame *i* must match master frame *i*
   better than it matches its neighbours. This is the off-by-one detector: a
   benchmark with a one-frame shift still produces plausible-looking PSNR, and
   nothing else here would notice.

3. **Range and colour** — regress decoded luma against master luma. A
   limited/full range error shows up as a slope near 0.86 or 1.16 rather than
   1.0, and a black-level error as a non-zero intercept. Both would otherwise
   look like a uniform quality loss applied equally to every method.

4. **Shuffle control** — deliberately misalign the sequence and confirm the
   score collapses. If shuffling frames does *not* hurt, the benchmark is not
   measuring temporal correspondence at all and check 2 proves nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

import numpy as np
from PIL import Image


def luma(path: str) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def psnr_arr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse <= 0 else 10.0 * np.log10(1.0 / mse)


def box2(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    return a[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def main() -> int:
    ap = argparse.ArgumentParser(description="Alignment and colour proof for captured clips.")
    ap.add_argument("--root", default="data/captured")
    ap.add_argument("--condition", default="h264_high")
    ap.add_argument("--out", default="results/captured-alignment.json")
    args = ap.parse_args()

    with open(os.path.join(args.root, "prepared.json"), encoding="utf-8") as fh:
        prepared = json.load(fh)

    report = {
        "schema": "aethervsr.captured-alignment/1",
        "condition": args.condition,
        "checks": {
            "geometry": "master 2560x1440, decoded 1280x720, equal counts",
            "frameCorrespondence": "decoded[i] must match master[i] better than master[i+-1]",
            "colourRange": "luma regression slope ~1.0, intercept ~0; catches tv/pc mismatch",
            "shuffleControl": "misaligning the sequence must collapse the score",
        },
        "clips": {},
    }
    failures = []

    for clip in prepared["clips"]:
        cid = clip["id"]
        root = os.path.join(args.root, "clips", cid)
        mdir, ddir = os.path.join(root, "master"), os.path.join(root, "decoded", args.condition)
        if not os.path.isdir(ddir):
            continue
        names = sorted(os.listdir(mdir))
        masters = [luma(os.path.join(mdir, n)) for n in names]
        decoded = [luma(os.path.join(ddir, n)) for n in names if os.path.exists(os.path.join(ddir, n))]

        geom_ok = (
            all(m.shape == (1440, 2560) for m in masters)
            and all(d.shape == (720, 1280) for d in decoded)
            and len(masters) == len(decoded)
        )

        # Correspondence on the box-downsampled master, which is the geometry
        # the decoded frame actually lives in.
        small = [box2(m) for m in masters]
        aligned, off_by_one = [], []
        for i in range(1, len(decoded) - 1):
            aligned.append(psnr_arr(decoded[i], small[i]))
            off_by_one.append(max(psnr_arr(decoded[i], small[i - 1]), psnr_arr(decoded[i], small[i + 1])))
        margin = st.fmean(a - b for a, b in zip(aligned, off_by_one)) if aligned else float("nan")

        # Range/colour: least squares decoded ~ a*master + b over a subsample.
        flat_d = np.concatenate([d[::8, ::8].ravel() for d in decoded])
        flat_m = np.concatenate([s[::8, ::8].ravel() for s in small])
        A = np.vstack([flat_m, np.ones_like(flat_m)]).T
        slope, intercept = np.linalg.lstsq(A, flat_d, rcond=None)[0]

        # Shuffle control: reverse the decoded order. On genuinely moving
        # footage this must be much worse; if it is not, the clip has too
        # little motion for check 2 to mean anything, and that is worth knowing.
        shuffled = st.fmean(psnr_arr(decoded[len(decoded) - 1 - i], small[i]) for i in range(len(decoded)))
        straight = st.fmean(psnr_arr(decoded[i], small[i]) for i in range(len(decoded)))

        entry = {
            "geometryOk": geom_ok,
            "frames": len(decoded),
            "alignedPsnr": st.fmean(aligned) if aligned else float("nan"),
            "offByOnePsnr": st.fmean(off_by_one) if off_by_one else float("nan"),
            "alignmentMarginDb": margin,
            "lumaSlope": float(slope),
            "lumaIntercept": float(intercept),
            "straightPsnr": straight,
            "reversedPsnr": shuffled,
            "shuffleCollapseDb": straight - shuffled,
        }
        problems = []
        if not geom_ok:
            problems.append("geometry")
        if not (margin > 1.0):
            problems.append(f"alignment margin only {margin:.2f} dB")
        if not (0.97 <= slope <= 1.03):
            problems.append(f"luma slope {slope:.4f}")
        if abs(intercept) > 0.02:
            problems.append(f"luma intercept {intercept:.4f}")
        if not (straight - shuffled > 1.0):
            problems.append(f"shuffle collapse only {straight - shuffled:.2f} dB (static clip?)")
        entry["problems"] = problems
        if problems:
            failures.append((cid, problems))
        report["clips"][cid] = entry
        print(
            f"  {cid[:40]:<40} margin {margin:+6.2f} dB  slope {slope:.4f}  "
            f"intercept {intercept:+.4f}  shuffle {straight - shuffled:+6.2f} dB  "
            f"{'OK' if not problems else 'CHECK: ' + '; '.join(problems)}",
            file=sys.stderr,
        )

    report["failures"] = [{"clip": c, "problems": p} for c, p in failures]
    report["allClipsAligned"] = not failures
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"\n{len(report['clips'])} clips checked, {len(failures)} with problems", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
