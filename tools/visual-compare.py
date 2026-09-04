#!/usr/bin/env python3
"""Deterministic side-by-side crops for visual diagnosis.

Metrics say whether a number moved. They do not say whether the model
hallucinated a texture that was never there, rang around an edge, or shifted a
colour. These crops exist to answer that, and they are chosen by a fixed rule
rather than by eye: for each source the crop is placed at the location of
highest local gradient energy, so the comparison lands on the detail the
upscaler is supposed to be reconstructing instead of on flat sky.

Every panel is the same crop through a different stage, at the same scale, in
the same order, so a difference in the image is a difference in the stage.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from evaluate import catmull_rom_2x, load_model, psnr

CROP = 128  # HR pixels


def to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.squeeze(0).clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, "RGB")


def busiest_crop(hr: torch.Tensor, size: int) -> tuple[int, int]:
    """Top-left of the `size` square with the most gradient energy.

    A fixed, content-driven rule. Choosing crops by hand is how a comparison
    becomes an argument.
    """
    grey = hr.mean(1, keepdim=True)
    gx = (grey[:, :, :, 1:] - grey[:, :, :, :-1]).abs()
    gy = (grey[:, :, 1:, :] - grey[:, :, :-1, :]).abs()
    energy = F.pad(gx, (0, 1)) + F.pad(gy, (0, 0, 0, 1))
    pooled = F.avg_pool2d(energy, kernel_size=size, stride=size // 2)
    idx = int(torch.argmax(pooled))
    row, col = divmod(idx, pooled.shape[3])
    y = min(row * (size // 2), hr.shape[2] - size)
    x = min(col * (size // 2), hr.shape[3] - size)
    return max(0, y), max(0, x)


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministic visual comparison crops.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--sources", required=True, help="comma-separated image paths")
    ap.add_argument("--profile", default="h264_typical")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="docs/visual")
    args = ap.parse_args()

    model, payload = load_model(args.model)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    os.makedirs(args.out, exist_ok=True)

    from degrade import degrade_tensor

    index = []
    for path in [p.strip() for p in args.sources.split(",") if p.strip()]:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = 1280 / max(w, h)
            if scale < 1:
                im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            w, h = im.size
            w -= w % 4
            h -= h % 4
            im = im.crop((0, 0, w, h))
            raw = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
            hr = raw.reshape(h, w, 3).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)

        lr = (degrade_tensor(hr, args.profile, args.seed) if args.profile != "box"
              else F.avg_pool2d(hr, 2))
        hr_d, lr_d = hr.to(device), lr.to(device)
        with torch.no_grad():
            neural = model(lr_d).clamp(0, 1).cpu()
        cat = catmull_rom_2x(lr_d).cpu()
        near = F.interpolate(lr_d, scale_factor=2, mode="nearest").clamp(0, 1).cpu()

        y, x = busiest_crop(hr, CROP)
        panels = [
            ("nearest", near), ("Catmull-Rom", cat),
            ("neural", neural), ("reference", hr),
        ]
        zoom = 3
        tile = CROP * zoom
        sheet = Image.new("RGB", (tile * len(panels), tile + 18), (16, 16, 20))
        draw = ImageDraw.Draw(sheet)
        for i, (label, t) in enumerate(panels):
            crop = to_pil(t[:, :, y : y + CROP, x : x + CROP]).resize((tile, tile), Image.Resampling.NEAREST)
            sheet.paste(crop, (i * tile, 18))
            d = psnr(t.to(device), hr_d) if label != "reference" else float("inf")
            text = label if label == "reference" else f"{label}  {d:.2f} dB"
            draw.text((i * tile + 6, 4), text, fill=(220, 220, 230))

        name = f"{os.path.splitext(os.path.basename(path))[0]}-{args.profile}.png"
        sheet.save(os.path.join(args.out, name))
        index.append({"file": name, "source": path, "crop": [x, y, CROP, CROP], "profile": args.profile})
        print(f"  wrote {name}  crop at ({x},{y})")

    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": os.path.basename(args.model),
                "modelSeed": payload.get("training", {}).get("seed"),
                "cropRule": "highest gradient energy, 128 HR px, nearest-neighbour 3x zoom for inspection",
                "panelOrder": ["nearest", "Catmull-Rom", "neural", "reference"],
                "images": index,
            },
            fh,
            indent=1,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
