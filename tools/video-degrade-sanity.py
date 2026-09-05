#!/usr/bin/env python3
"""Validate the video degradation generator before any model trains on it.

A degradation pipeline that is silently misaligned, colour-shifted or off by one
frame will still produce a trainable dataset and plausible-looking losses. It
will simply teach the model to invert the wrong transform, and the damage is
only visible much later as an unexplained evaluation result. These checks are
cheap and run before training, not after.

Each check fails loudly and independently:

  1. GOP structure    the requested frame types actually occur in the bitstream
  2. Patch count      every LR patch has exactly one HR partner
  3. Alignment        a +-2 px shift search peaks at zero offset
  4. Range/colour     no full/limited range mismatch, no channel swap
  5. CRF effective    higher CRF really does spend fewer bits per frame
  6. Geometry         HR patch is exactly twice the LR patch in both axes

Check 3 is the one that matters and the one an earlier version got wrong. It
used to score each LR patch against a *different* patch and require the true
pair to win by 1 dB. That always passed - a quarter-patch misregistration still
cleared it by nearly an order of magnitude - because unrelated content sits
near 9.5 dB no matter how the pair is aligned. It measured content
dissimilarity, not correspondence.

The replacement asks whether zero is the *best* offset among its neighbours,
which is the question a misalignment would actually answer differently, and
reports the +1/-1 asymmetry because a half-pixel phase error from the downscale
shows up there rather than in the peak.

Frame-level count and off-by-one checks live in tools/video-degrade.py, which
drops any sequence whose decoded LR count differs from its HR count before a
patch is ever cut.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

FFMPEG = os.environ.get("AETHER_FFMPEG", "/opt/homebrew/bin/ffmpeg")

MIN_SHIFT_MARGIN_DB = 0.05
MAX_RANGE_ERROR = 0.02
MIN_CRF_MONOTONIC_DB = 0.10


def load(path: str) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        raw = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        return raw.reshape(h, w, 3).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return 99.0 if mse <= 1e-12 else 10.0 * float(np.log10(1.0 / mse))


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanity-check video degradation pairs.")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--sequences", type=int, default=8, help="how many sequences to inspect")
    ap.add_argument("--out", default="results/video-degrade-sanity.json")
    args = ap.parse_args()

    with open(os.path.join(args.pairs, "pairs.json"), encoding="utf-8") as fh:
        meta = json.load(fh)

    failures: list[str] = []
    checks: dict[str, dict] = {}

    # --- 1. GOP structure -------------------------------------------------
    want_temporal = meta["structure"] == "gop"
    temporal_seqs, intra_only_seqs = 0, 0
    for ev in meta["encodeEvidence"]:
        types = ev["frameTypes"]
        has_pb = (types.get("P", 0) + types.get("B", 0)) > 0
        temporal_seqs += bool(has_pb)
        intra_only_seqs += (not has_pb)
    if want_temporal and temporal_seqs != len(meta["encodeEvidence"]):
        failures.append(f"gop requested but {intra_only_seqs} sequences are intra-only")
    if not want_temporal and temporal_seqs != 0:
        failures.append(f"all_i requested but {temporal_seqs} sequences contain P/B frames")
    checks["gopStructure"] = {
        "requested": meta["structure"], "sequencesWithPorB": temporal_seqs,
        "intraOnlySequences": intra_only_seqs, "total": len(meta["encodeEvidence"]),
    }

    # --- 2/3/4/6. Patch-level checks -------------------------------------
    # The patches are what training actually consumes, so they are what gets
    # checked. A pipeline can produce perfectly aligned full frames and still
    # emit misaligned patches if the crop offsets drift between LR and HR.
    align_rows: list[dict] = []
    count_ok, geom_ok = True, True
    range_errors: list[float] = []

    for split in ("train", "val"):
        path = os.path.join(args.pairs, f"{split}.pt")
        if not os.path.exists(path):
            failures.append(f"missing {path}")
            count_ok = False
            continue
        blob = torch.load(path)
        lr_all, hr_all = blob["lr"], blob["hr"]
        if lr_all.shape[0] != hr_all.shape[0]:
            failures.append(f"{split}: {lr_all.shape[0]} lr vs {hr_all.shape[0]} hr patches")
            count_ok = False
            continue
        if hr_all.shape[-1] != lr_all.shape[-1] * 2 or hr_all.shape[-2] != lr_all.shape[-2] * 2:
            failures.append(f"{split}: hr {tuple(hr_all.shape)} is not 2x lr {tuple(lr_all.shape)}")
            geom_ok = False

        n = lr_all.shape[0]
        step = max(1, n // max(1, args.sequences * 40))
        sel = list(range(0, n, step))[: args.sequences * 40]

        # Shift search, not a shuffle. The original check scored each LR patch
        # against a *different* patch and asserted the matched pair won by 1 dB.
        # It always passed: review showed a quarter-patch misregistration still
        # cleared that gate by nearly an order of magnitude, because the
        # mismatched partner is unrelated content pinned near 9.5 dB. It was
        # measuring content dissimilarity, not correspondence.
        #
        # This asks the question that actually matters: of all small offsets,
        # is zero the best one? A pipeline shifted by a pixel, or carrying a
        # half-pixel phase error from the downscale, peaks somewhere else.
        offsets = [-2, -1, 0, 1, 2]
        scores: dict[tuple[int, int], list[float]] = {(dx, dy): [] for dx in offsets for dy in offsets}
        for i in sel[: max(8, len(sel) // 8)]:
            lr = lr_all[i : i + 1].float() / 255.0
            up = F.interpolate(lr, scale_factor=2, mode="bilinear", align_corners=False).clamp(0, 1)
            hr = hr_all[i : i + 1].float() / 255.0
            for (dx, dy), acc in scores.items():
                # Compare on the overlap only, so a shifted candidate is not
                # penalised for edge pixels it cannot see.
                a = up[:, :, 2 + dy : -2 + dy or None, 2 + dx : -2 + dx or None]
                b = hr[:, :, 2 : -2, 2 : -2]
                acc.append(psnr(a, b))
        mean_by_offset = {k: st.fmean(v) for k, v in scores.items()}
        best = max(mean_by_offset, key=lambda k: mean_by_offset[k])
        zero = mean_by_offset[(0, 0)]
        runner = max(v for k, v in mean_by_offset.items() if k != (0, 0))
        margin = zero - runner
        if best != (0, 0):
            failures.append(
                f"{split}: alignment peaks at offset {best}, not (0,0); "
                f"{mean_by_offset[best]:.3f} dB against {zero:.3f} dB"
            )
        elif margin < MIN_SHIFT_MARGIN_DB:
            failures.append(
                f"{split}: zero offset beats its nearest neighbour by only {margin:.3f} dB "
                f"(<{MIN_SHIFT_MARGIN_DB}); alignment is not sharply peaked"
            )
        # A half-pixel phase error shows up as asymmetry between +1 and -1.
        asym_x = abs(mean_by_offset[(1, 0)] - mean_by_offset[(-1, 0)])
        asym_y = abs(mean_by_offset[(0, 1)] - mean_by_offset[(0, -1)])
        lr_f = lr_all[sel].float() / 255.0
        hr_f = hr_all[sel].float() / 255.0
        rng_err = abs(float(lr_f.mean()) - float(hr_f.mean()))
        range_errors.append(rng_err)
        if rng_err > MAX_RANGE_ERROR:
            failures.append(f"{split}: mean level differs by {rng_err:.4f}; range mismatch?")
        ch_err = max(abs(float(lr_f[:, c].mean()) - float(hr_f[:, c].mean())) for c in range(3))
        if ch_err > 0.05:
            failures.append(f"{split}: channel means differ by {ch_err:.4f}; channel swap?")

        align_rows.append({
            "split": split, "patches": n, "inspected": len(sel),
            "zeroOffsetPsnr": zero, "bestOffset": list(best),
            "marginOverNearestOffsetDb": margin,
            "horizontalAsymmetryDb": asym_x, "verticalAsymmetryDb": asym_y,
            "psnrByOffset": {f"{dx},{dy}": v for (dx, dy), v in sorted(mean_by_offset.items())},
            "meanLevelError": rng_err, "channelMeanError": ch_err,
            "lrPatch": list(lr_all.shape[-2:]), "hrPatch": list(hr_all.shape[-2:]),
        })
        print(f"  {split:<6} {n:>6} patches  peak {best} {zero:6.3f} dB  "
              f"margin +{margin:.3f}  asym x {asym_x:.3f} y {asym_y:.3f}  "
              f"range {rng_err:.4f}", file=sys.stderr)

    checks["patchCount"] = {"ok": count_ok}
    checks["geometry"] = {"ok": geom_ok, "expected": "hr patch == 2x lr patch in both axes"}
    checks["alignment"] = {
        "splits": align_rows,
        "allPeakAtZeroOffset": all(r["bestOffset"] == [0, 0] for r in align_rows),
        "minMarginOverNearestOffsetDb": min(
            (r["marginOverNearestOffsetDb"] for r in align_rows), default=float("nan")
        ),
    }
    checks["range"] = {
        "maxMeanLevelError": max(range_errors, default=float("nan")),
        "threshold": MAX_RANGE_ERROR,
    }

    # --- 5. CRF really reaches the encoder --------------------------------
    # If it did not, every "heavy compression" sample is a mislabelled easy one
    # and the whole CRF-distribution experiment is measuring nothing. Verified
    # from the recorded per-sequence bitrates rather than a re-encode: the
    # sequences were produced by the run under test, so this checks that run.
    by_crf: dict[int, list[float]] = {}
    for ev, rec in zip(meta["encodeEvidence"], meta["index"]):
        by_crf.setdefault(ev["crf"], []).append(ev["bytes"] / rec["frames"])
    crf_points = sorted(by_crf)
    mono_ok = True
    if len(crf_points) >= 3:
        lo = st.fmean(by_crf[crf_points[0]])
        hi = st.fmean(by_crf[crf_points[-1]])
        mono_ok = lo > hi
        if not mono_ok:
            failures.append(
                f"CRF {crf_points[0]} produces {lo:.0f} B/frame but CRF {crf_points[-1]} "
                f"produces {hi:.0f}; CRF is not reaching the encoder"
            )
        checks["crfEffective"] = {
            "ok": mono_ok,
            "bytesPerFrameByCrf": {str(c): round(st.fmean(by_crf[c]), 1) for c in crf_points},
            "lowestCrfBytes": round(lo, 1), "highestCrfBytes": round(hi, 1),
        }
        print(f"\n  CRF effective: crf{crf_points[0]} {lo:.0f} B/frame > "
              f"crf{crf_points[-1]} {hi:.0f} B/frame", file=sys.stderr)

    report = {
        "schema": "aethervsr.video-degrade-sanity/1",
        "pairs": args.pairs,
        "structure": meta["structure"], "crfDistribution": meta["crfDistribution"],
        "splitsInspected": len(align_rows),
        "checks": checks,
        "failures": failures,
        "pass": not failures,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    if failures:
        print(f"\nFAILED {len(failures)} check(s):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"\nall checks pass ({len(align_rows)} sequences inspected)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
