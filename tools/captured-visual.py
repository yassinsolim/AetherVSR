#!/usr/bin/env python3
"""Deterministic comparison crops from captured footage.

Crop placement is a fixed rule applied before any score is consulted: the
128px window with the highest gradient energy in the reference. Choosing crops
after seeing results, or by eye, turns a comparison into an argument.

Clip selection is also rule-based. The milestone requires a representative
win, tie and loss, so clips are ranked by their measured delta and the
extremes plus the median are taken. That is deliberately not cherry-picking:
the rule names which clips appear before anyone looks at the images.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from evaluate import catmull_rom_2x, load_model, psnr


def load_png(path: str) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        raw = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        return raw.reshape(h, w, 3).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.squeeze(0).clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr, "RGB")


def busiest(ref: torch.Tensor, size: int) -> tuple[int, int]:
    grey = ref.mean(1, keepdim=True)
    gx = (grey[:, :, :, 1:] - grey[:, :, :, :-1]).abs()
    gy = (grey[:, :, 1:, :] - grey[:, :, :-1, :]).abs()
    energy = F.pad(gx, (0, 1)) + F.pad(gy, (0, 0, 0, 1))
    pooled = F.avg_pool2d(energy, kernel_size=size, stride=size // 2)
    idx = int(torch.argmax(pooled))
    row, col = divmod(idx, pooled.shape[3])
    return (max(0, min(row * (size // 2), ref.shape[2] - size)),
            max(0, min(col * (size // 2), ref.shape[3] - size)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Captured-footage comparison crops.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", default="results/captured-realistic.json")
    ap.add_argument("--root", default="data/captured")
    ap.add_argument("--condition", default="h264_typical")
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--zoom", type=int, default=3)
    ap.add_argument("--out", default="docs/captured")
    args = ap.parse_args()

    model, _ = load_model(args.model)
    with open(args.results, encoding="utf-8") as fh:
        res = json.load(fh)
    # The first version of this sheet was rendered with the candidate's pixels
    # and the production model's scores, which inverted the roles: the clip
    # labelled a loss was in fact the candidate's best. Refuse the mismatch.
    named = os.path.basename(res.get("model", ""))
    if named and named != os.path.basename(args.model):
        raise SystemExit(
            f"--results describes {named!r} but --model is "
            f"{os.path.basename(args.model)!r}; the roles would describe the wrong model"
        )
    os.makedirs(args.out, exist_ok=True)

    scored = [
        (c["conditions"][args.condition]["clipMeanDiffVsCatmull"]["psnr"], c["id"], c["category"])
        for c in res["perClip"] if args.condition in c["conditions"]
    ]
    scored.sort()
    # Rank-based names, not outcome names. Calling the worst clip a "loss" when
    # it is in fact a small win invents a failure case the evidence does not
    # contain - and reading a "LOSS" banner over a winning panel is worse than
    # no banner at all. The labels describe rank; whether the worst clip is
    # actually a loss is stated, not assumed.
    picks = {
        "worst": scored[0],
        "median": scored[len(scored) // 2],
        "best": scored[-1],
    }
    if scored[0][0] > 0:
        print(f"  note: no losing clip at {args.condition}; worst is {scored[0][0]:+.4f} dB",
              file=sys.stderr)

    index = []
    for role, (delta, cid, category) in picks.items():
        root = os.path.join(args.root, "clips", cid)
        mdir = os.path.join(root, "master")
        ddir = os.path.join(root, "decoded", args.condition)
        name = sorted(os.listdir(mdir))[len(os.listdir(mdir)) // 2]
        ref = load_png(os.path.join(mdir, name))
        lr = load_png(os.path.join(ddir, name))
        with torch.no_grad():
            neural = model(lr).clamp(0, 1)
        cat = catmull_rom_2x(lr)
        near = F.interpolate(lr, scale_factor=2, mode="nearest").clamp(0, 1)

        y, x = busiest(ref, args.crop)
        panels = [("720p input", near), ("Catmull-Rom", cat), ("AetherVSR", neural), ("reference", ref)]
        tile = args.crop * args.zoom
        sheet = Image.new("RGB", (tile * len(panels), tile + 34), (16, 16, 20))
        draw = ImageDraw.Draw(sheet)
        draw.text((6, 4), f"{role.upper()}  {cid[:52]}  [{category}]  {args.condition}  "
                          f"neural-catmull {delta:+.3f} dB", fill=(235, 235, 245))
        for i, (label, t) in enumerate(panels):
            crop = to_pil(t[:, :, y:y + args.crop, x:x + args.crop]).resize((tile, tile), Image.Resampling.NEAREST)
            sheet.paste(crop, (i * tile, 34))
            txt = label if label == "reference" else f"{label}  {psnr(t, ref):.2f} dB"
            draw.text((i * tile + 6, 20), txt, fill=(205, 205, 220))
        fname = f"{role}-{cid[:40]}-{args.condition}.png"
        sheet.save(os.path.join(args.out, fname))
        index.append({"role": role, "file": fname, "clip": cid, "category": category,
                      "deltaPsnrDb": delta, "frame": name, "crop": [x, y, args.crop, args.crop]})
        print(f"  {role:<5} {delta:+7.3f} dB  {cid[:44]}")

    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "model": os.path.basename(args.model),
            "condition": args.condition,
            "cropRule": "highest gradient energy in the reference, 128 px, 3x nearest zoom",
            "clipRule": "measured delta ranked; worst, median and best clip selected before viewing",
        "resultsFile": args.results,
        "hasLosingClip": bool(scored[0][0] <= 0),
        "worstClipDeltaDb": scored[0][0],
            "panelOrder": ["720p input (nearest)", "Catmull-Rom", "AetherVSR", "reference"],
            "images": index,
        }, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
