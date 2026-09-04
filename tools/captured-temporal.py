#!/usr/bin/env python3
"""Temporal behaviour on captured motion.

The synthetic harness measured roughly 3x Catmull-Rom sub-pixel response and
called it architectural. That was measured on procedural translation with
known, exact displacement. Real footage has parallax, rolling shutter, motion
blur and compression noise, so the question here is whether that response
shows up as measurable instability on actual camera motion.

Two metrics, both relative to a baseline measured on identical frames:

**Motion-compensated residual.** Warp frame t-1 to frame t using dense optical
flow estimated on the *reference* footage, then measure how much each
upscaler's output changes beyond what the motion explains. Flow is estimated
once per frame pair from the ground-truth HR reference, never from the
upscaled output, so a method cannot influence its own compensation. Flow error
still inflates every method equally, which is why only the *ratio* between
methods is interpreted, never the absolute value.

**Static-region variance.** Find the pixels the flow says did not move, and
measure temporal variance there. On a locked-off region, any variance an
upscaler adds beyond the reference's own is shimmer it invented.

Both are validated on controls first: Catmull-Rom and bilinear must order
sensibly, and a still region must score near the reference floor.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from evaluate import catmull_rom_2x, load_model

try:
    import cv2  # type: ignore
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


def load_png(path: str) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        raw = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        return raw.reshape(h, w, 3).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def to_luma(t: torch.Tensor) -> np.ndarray:
    a = t.squeeze(0).permute(1, 2, 0).numpy()
    return (0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]).astype(np.float32)


def warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = img.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def main() -> int:
    ap = argparse.ArgumentParser(description="Temporal metrics on captured footage.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", default="data/captured")
    ap.add_argument("--condition", default="h264_typical")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--out", default="results/captured-temporal.json")
    args = ap.parse_args()

    if not HAVE_CV2:
        print("opencv not available; install opencv-python-headless to run this", file=sys.stderr)
        return 2

    model, payload = load_model(args.model)
    model.eval()
    with open(os.path.join(args.root, "prepared.json"), encoding="utf-8") as fh:
        prepared = json.load(fh)

    per_clip = {}
    for clip in prepared["clips"]:
        cid = clip["id"]
        root = os.path.join(args.root, "clips", cid)
        mdir, ddir = os.path.join(root, "master"), os.path.join(root, "decoded", args.condition)
        if not os.path.isdir(ddir):
            continue
        names = sorted(os.listdir(mdir))[: args.frames]

        methods = ("reference", "neural", "catmull_rom", "bilinear")
        residual = {m: [] for m in methods}
        static_var = {m: [] for m in methods}

        prev = None
        for name in names:
            ref = load_png(os.path.join(mdir, name))
            lr = load_png(os.path.join(ddir, name))
            with torch.no_grad():
                frames = {
                    "reference": ref,
                    "neural": model(lr).clamp(0, 1),
                    "catmull_rom": catmull_rom_2x(lr),
                    "bilinear": F.interpolate(lr, scale_factor=2, mode="bilinear", align_corners=False).clamp(0, 1),
                }
            cur = {m: to_luma(v) for m, v in frames.items()}
            if prev is not None:
                # Flow from the reference only, so no method compensates itself.
                flow = cv2.calcOpticalFlowFarneback(
                    (prev["reference"] * 255).astype(np.uint8),
                    (cur["reference"] * 255).astype(np.uint8),
                    None, 0.5, 3, 21, 3, 5, 1.2, 0,
                )
                mag = np.linalg.norm(flow, axis=2)
                still = mag < 0.25
                for m in methods:
                    warped = warp(prev[m], flow)
                    residual[m].append(float(np.mean(np.abs(cur[m] - warped)) * 255.0))
                    if still.sum() > 1000:
                        static_var[m].append(float(np.mean((cur[m][still] - prev[m][still]) ** 2) * 255.0 * 255.0))
            prev = cur

        if not residual["neural"]:
            continue
        entry = {m: {"mcResidual": st.fmean(residual[m])} for m in methods}
        for m in methods:
            if static_var[m]:
                entry[m]["staticRegionVariance"] = st.fmean(static_var[m])
        entry["neuralOverCatmullResidual"] = entry["neural"]["mcResidual"] / entry["catmull_rom"]["mcResidual"]
        entry["neuralOverReferenceResidual"] = entry["neural"]["mcResidual"] / entry["reference"]["mcResidual"]
        entry["catmullOverReferenceResidual"] = entry["catmull_rom"]["mcResidual"] / entry["reference"]["mcResidual"]
        per_clip[cid] = entry
        print(
            f"  {cid[:38]:<38} residual ref {entry['reference']['mcResidual']:6.3f}  "
            f"catmull {entry['catmull_rom']['mcResidual']:6.3f}  neural {entry['neural']['mcResidual']:6.3f}  "
            f"neural/catmull {entry['neuralOverCatmullResidual']:.3f}",
            file=sys.stderr,
        )

    ratios = [v["neuralOverCatmullResidual"] for v in per_clip.values()]
    cat_ratio = [v["catmullOverReferenceResidual"] for v in per_clip.values()]
    report = {
        "schema": "aethervsr.captured-temporal/1",
        "model": os.path.basename(args.model),
        "condition": args.condition,
        "method": (
            "Dense Farneback flow estimated on the HR reference only; each method's own output "
            "is warped by that flow and the residual measured. Flow error inflates every method "
            "equally, so only ratios between methods are interpreted."
        ),
        "control": (
            "catmullOverReferenceResidual shows how much of the residual is flow error rather "
            "than upscaler behaviour; a ratio near 1 would mean the metric cannot resolve anything."
        ),
        "perClip": per_clip,
        "aggregate": {
            "neuralOverCatmullResidual": {
                "mean": st.fmean(ratios), "median": st.median(ratios),
                "min": min(ratios), "max": max(ratios),
                "clipsWorseThanCatmull": sum(1 for r in ratios if r > 1.0), "clips": len(ratios),
            },
            "catmullOverReferenceResidual": {"mean": st.fmean(cat_ratio)},
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nneural/catmull residual ratio: mean {st.fmean(ratios):.3f}, "
          f"worse on {report['aggregate']['neuralOverCatmullResidual']['clipsWorseThanCatmull']}/{len(ratios)} clips",
          file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
