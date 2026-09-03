#!/usr/bin/env python3
"""Score the neural upscaler on the ground-truth video benchmark.

Deliberately reuses the exact frames `tools/videobench.py` wrote: the same
1440p master references and the same decoded 720p frames the conventional
baselines were scored against. A separate decode path here would risk scoring
the neural stage against slightly different pixels than the baselines saw,
which is the sort of mismatch that quietly manufactures a win.

Reports per-frame distributions, not just a mean: a model that is good on
average and bad on the hardest frames is a different product from one that is
uniformly slightly better, and the mean hides the difference.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F
from PIL import Image

from evaluate import load_model, psnr, ssim


def load_png(path: str) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        raw = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        return raw.reshape(h, w, 3).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def distribution(values: list[float]) -> dict:
    ordered = sorted(v for v in values if v != float("inf"))
    if not ordered:
        return {}
    return {
        "mean": statistics.fmean(ordered),
        "p5": ordered[max(0, int(0.05 * len(ordered)) - 1)],
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "min": ordered[0],
        "max": ordered[-1],
        "n": len(ordered),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Score the neural model on the video benchmark.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", default="data/video")
    ap.add_argument("--categories", default="natural,texture,motion,text,animation")
    ap.add_argument("--tiers", default="h264-high,h264-medium,h264-low,vp9-medium,av1-medium")
    ap.add_argument("--out", default="")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    model, payload = load_model(args.model)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    with open(os.path.join(args.root, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    results: dict[str, dict] = {}

    for category in categories:
        entry = manifest["categories"].get(category)
        if entry is None:
            print(f"  no such category {category!r}", file=sys.stderr)
            continue
        frames = entry["frames"]
        results[category] = {}

        for tier in tiers:
            decoded_dir = os.path.join(args.root, category, "decoded", tier)
            if not os.path.isdir(decoded_dir):
                continue
            neural_psnr, neural_ssim, cat_psnr, cat_ssim = [], [], [], []
            worst: list[tuple[float, int]] = []

            for frame in frames:
                idx = frame["index"]
                ref_path = os.path.join(args.root, frame["master_file"])
                lr_path = os.path.join(decoded_dir, f"frame_{idx:04d}.png")
                if not (os.path.exists(ref_path) and os.path.exists(lr_path)):
                    continue
                ref = load_png(ref_path).to(device)
                lr = load_png(lr_path).to(device)
                with torch.no_grad():
                    out = model(lr).clamp(0, 1)
                # Catmull-Rom on the identical decoded frame, so the comparison
                # cannot be moved by a decode or geometry difference.
                cat = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)
                if out.shape != ref.shape:
                    raise SystemExit(
                        f"geometry mismatch {category}/{tier} frame {idx}: "
                        f"neural {tuple(out.shape)} against reference {tuple(ref.shape)}"
                    )
                p = psnr(out, ref)
                neural_psnr.append(p)
                neural_ssim.append(ssim(out, ref))
                cat_psnr.append(psnr(cat, ref))
                cat_ssim.append(ssim(cat, ref))
                worst.append((p - cat_psnr[-1], idx))

            if not neural_psnr:
                continue
            worst.sort()
            results[category][tier] = {
                "neural": {"psnr": distribution(neural_psnr), "ssim": distribution(neural_ssim)},
                "catmull_rom": {"psnr": distribution(cat_psnr), "ssim": distribution(cat_ssim)},
                "deltaPsnrDbMean": statistics.fmean(n - c for n, c in zip(neural_psnr, cat_psnr)),
                "deltaSsimMean": statistics.fmean(n - c for n, c in zip(neural_ssim, cat_ssim)),
                "framesNeuralWins": sum(1 for n, c in zip(neural_psnr, cat_psnr) if n > c),
                "frames": len(neural_psnr),
                "worstFramesForNeural": [{"frame": i, "deltaDb": d} for d, i in worst[:3]],
            }
            r = results[category][tier]
            print(
                f"  {category:<10} {tier:<13} neural {r['neural']['psnr']['mean']:6.3f}  "
                f"catmull {r['catmull_rom']['psnr']['mean']:6.3f}  "
                f"delta {r['deltaPsnrDbMean']:+6.3f}  wins {r['framesNeuralWins']}/{r['frames']}",
                file=sys.stderr,
            )

    report = {
        "schema": "aethervsr.videobench-neural/1",
        "model": os.path.basename(args.model),
        "modelSeed": payload.get("training", {}).get("seed"),
        "modelDegradation": payload.get("training", {}).get("degradationProfile"),
        "reference": "1440p master frames written by tools/videobench.py",
        "input": "the same decoded 720p frames the conventional baselines were scored on",
        "results": results,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        print(f"wrote {args.out}", file=sys.stderr)
    if args.json:
        print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
