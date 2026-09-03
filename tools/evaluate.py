#!/usr/bin/env python3
"""Score a frozen AetherVSR model on a named split or corpus.

Kept separate from `tools/train.py` on purpose. Training never loads the test
split - not to print a number, not to plot a curve - so the only way a test
score can influence a checkpoint, a seed or a hyperparameter is if a human
deliberately runs this and then goes back. That is a decision someone has to
make and record, rather than a gradient that leaks in by default.

The evidentiary categories are kept apart and never averaged together:

  REGRESSION        the deterministic synthetic reference and the eight-image
                    natural set from Milestone 4. Same corpus as training, and
                    the natural set overlaps it at source level. Useful for
                    detecting change; not evidence about unseen content.

  SOURCE-DISJOINT   the `test` split of the training corpus. No source image
                    is shared with training, but the corpus, the curation and
                    the encoding pipeline are the same.

  INDEPENDENT       `data/eval-independent/`, a different institution entirely,
                    checked for exact, URL and perceptual overlap. This is the
                    one that answers the generalisation question.
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

from aethersr import AetherSR, box_downsample2
from dataset import load_split, split_files

CATEGORY = {
    "regression": "REGRESSION / DEVELOPMENT",
    "val": "SOURCE-DISJOINT VALIDATION",
    "test": "SOURCE-DISJOINT TEST",
    "independent": "HELD-OUT INDEPENDENT TEST",
}


def load_model(path: str) -> tuple[AetherSR, dict]:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    channels = payload.get("features", 16)
    depth = payload.get("depth", 2)
    model = AetherSR(channels=channels, depth=depth)
    state = {}
    flat = payload["weights"]
    for key, tensor in model.state_dict().items():
        if key not in flat:
            raise SystemExit(f"model file is missing weight {key!r}")
        state[key] = torch.tensor(flat[key], dtype=torch.float32).reshape(tensor.shape)
    model.load_state_dict(state)
    model.eval()
    return model, payload


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a.clamp(0, 1), b.clamp(0, 1)).item()
    return float("inf") if mse <= 0 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=a.device).view(1, 3, 1, 1)
    x = (a.clamp(0, 1) * weights).sum(1, keepdim=True)
    y = (b.clamp(0, 1) * weights).sum(1, keepdim=True)
    coords = torch.arange(11, dtype=torch.float32, device=a.device) - 5.0
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = (g / g.sum()).view(1, 1, -1)
    kernel = (g.transpose(2, 1) @ g).view(1, 1, 11, 11)
    blur = lambda t: F.conv2d(t, kernel, padding=0)  # noqa: E731
    mu_x, mu_y = blur(x), blur(y)
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sxx, syy, sxy = blur(x * x) - mu_xx, blur(y * y) - mu_yy, blur(x * y) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    return float((((2 * mu_xy + c1) * (2 * sxy + c2)) / ((mu_xx + mu_yy + c1) * (sxx + syy + c2))).mean())


def _keys_weights(t: torch.Tensor, a: float) -> torch.Tensor:
    """Keys cubic taps for fractional offsets t in [0,1), shape (..., 4)."""
    def far(x: torch.Tensor) -> torch.Tensor:
        return a * x**3 - 5 * a * x**2 + 8 * a * x - 4 * a

    def near(x: torch.Tensor) -> torch.Tensor:
        return (a + 2) * x**3 - (a + 3) * x**2 + 1

    return torch.stack([far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t)], dim=-1)


def catmull_rom_2x(lr: torch.Tensor, a: float = -0.5) -> torch.Tensor:
    """True Catmull-Rom 2x, matching the production WGSL baseline.

    `F.interpolate(mode="bicubic")` is Keys with **a = -0.75**, not -0.5, so it
    is a visibly sharper filter than the one AetherVSR ships. Scoring the neural
    stage against it compares against a baseline that is not in the product -
    exactly the hidden resampling mismatch a reviewer should hunt for.
    `src/core/upscale/baseline.wgsl.ts` documents the shipped kernel as
    "B = 0, C = 0.5", which is Keys a = -0.5.
    """
    n, c, h, w = lr.shape
    device, dtype = lr.device, lr.dtype

    def axis(size_out: int, size_in: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Half-pixel centres: what align_corners=False and the WGSL sampler use.
        pos = (torch.arange(size_out, device=device, dtype=dtype) + 0.5) * (size_in / size_out) - 0.5
        base = torch.floor(pos)
        return base.long(), pos - base

    yb, yt = axis(h * 2, h)
    xb, xt = axis(w * 2, w)
    wy, wx = _keys_weights(yt, a), _keys_weights(xt, a)

    padded = F.pad(lr, (2, 2, 2, 2), mode="replicate")
    out = torch.zeros(n, c, h * 2, w * 2, device=device, dtype=dtype)
    for dy in range(4):
        band = padded.index_select(2, (yb + dy - 1 + 2).clamp(0, h + 3))
        acc = torch.zeros(n, c, h * 2, w * 2, device=device, dtype=dtype)
        for dx in range(4):
            acc = acc + band.index_select(3, (xb + dx - 1 + 2).clamp(0, w + 3)) * wx[:, dx].view(1, 1, 1, -1)
        out = out + acc * wy[:, dy].view(1, 1, -1, 1)
    return out.clamp(0, 1)


def load_image(path: str, max_side: int = 1280) -> torch.Tensor | None:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            # Bound the work and, more importantly, make every image contribute
            # comparably rather than letting one 4000px photograph dominate.
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                im = im.resize((max(2, int(w * scale)), max(2, int(h * scale))), Image.Resampling.LANCZOS)
                w, h = im.size
            # Even dimensions, so a 2x downsample and re-upsample align exactly.
            w -= w % 2
            h -= h % 2
            im = im.crop((0, 0, w, h))
            raw = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
            return raw.reshape(h, w, 3).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
    except Exception as exc:  # noqa: BLE001
        print(f"  skip {os.path.basename(path)}: {type(exc).__name__} {exc}", file=sys.stderr)
        return None


def degrade(hr: torch.Tensor, profile: str, seed: int) -> torch.Tensor:
    if profile == "box":
        return box_downsample2(hr)
    try:
        from degrade import degrade_tensor
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit(f"degradation profile {profile!r} needs tools/degrade.py: {exc}")
    return degrade_tensor(hr, profile, seed)


def evaluate(model: AetherSR, files: list[str], root: str, profile: str, seed: int, device: str) -> dict:
    rows: list[dict] = []
    model = model.to(device)
    for name in files:
        hr = load_image(os.path.join(root, name))
        if hr is None:
            continue
        hr = hr.to(device)
        lr = degrade(hr, profile, seed).to(device)
        with torch.no_grad():
            neural = model(lr).clamp(0, 1)
        cat = catmull_rom_2x(lr)
        bil = F.interpolate(lr, scale_factor=2, mode="bilinear", align_corners=False).clamp(0, 1)
        near = F.interpolate(lr, scale_factor=2, mode="nearest").clamp(0, 1)
        # Every stage is scored against the same reference on the same crop, so
        # the comparison cannot be moved by a geometry difference.
        rows.append(
            {
                "file": name,
                "neural": {"psnr": psnr(neural, hr), "ssim": ssim(neural, hr)},
                "catmull_rom": {"psnr": psnr(cat, hr), "ssim": ssim(cat, hr)},
                "bilinear": {"psnr": psnr(bil, hr), "ssim": ssim(bil, hr)},
                "nearest": {"psnr": psnr(near, hr), "ssim": ssim(near, hr)},
            }
        )
    if not rows:
        raise SystemExit("no images scored")

    def agg(stage: str, metric: str) -> dict:
        vals = [r[stage][metric] for r in rows if r[stage][metric] != float("inf")]
        vals.sort()
        return {
            "mean": statistics.fmean(vals),
            "median": statistics.median(vals),
            "p5": vals[max(0, int(0.05 * len(vals)) - 1)],
            "p95": vals[min(len(vals) - 1, int(0.95 * len(vals)))],
            "min": vals[0],
            "max": vals[-1],
            "n": len(vals),
        }

    stages = ("neural", "catmull_rom", "bilinear", "nearest")
    wins = sum(1 for r in rows if r["neural"]["psnr"] > r["catmull_rom"]["psnr"])
    return {
        "images": len(rows),
        "profile": profile,
        "summary": {s: {m: agg(s, m) for m in ("psnr", "ssim")} for s in stages},
        "neuralOverCatmullRom": {
            "psnrDb": statistics.fmean(r["neural"]["psnr"] - r["catmull_rom"]["psnr"] for r in rows),
            "ssim": statistics.fmean(r["neural"]["ssim"] - r["catmull_rom"]["ssim"] for r in rows),
            "winsOf": f"{wins}/{len(rows)}",
        },
        "perImage": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Score a frozen model on a named evaluation set.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", required=True, choices=sorted(CATEGORY), help="which evidentiary category")
    ap.add_argument("--split", default="data/splits/corpus-v1.json")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--independent", default="data/eval-independent")
    ap.add_argument("--profile", default="box", help="degradation profile applied to make LR")
    ap.add_argument("--seed", type=int, default=7, help="seed for randomised degradation profiles")
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    model, payload = load_model(args.model)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if args.set == "independent":
        with open(os.path.join(args.independent, "manifest.json"), encoding="utf-8") as fh:
            files = [e["file"] for e in json.load(fh)["images"]][: args.limit]
        root = args.independent
    elif args.set == "regression":
        with open(os.path.join(args.corpus, "manifest.json"), encoding="utf-8") as fh:
            files = [e["file"] for e in json.load(fh)["images"]][: args.limit]
        root = args.corpus
    else:
        files = split_files(load_split(args.split), args.set)[: args.limit]
        root = args.corpus

    result = evaluate(model, files, root, args.profile, args.seed, device)
    result["category"] = CATEGORY[args.set]
    result["set"] = args.set
    result["model"] = os.path.basename(args.model)
    result["modelSeed"] = payload.get("training", {}).get("seed")
    result["modelSplitDigest"] = payload.get("training", {}).get("splitDigest")

    if args.json:
        print(json.dumps(result, indent=1))
        return 0

    s = result["summary"]
    print(f"{result['category']}  set={args.set}  profile={args.profile}  images={result['images']}")
    print(f"  {'stage':<14}{'PSNR mean':>10}{'p5':>8}{'p95':>8}{'SSIM mean':>11}")
    for stage in ("nearest", "bilinear", "catmull_rom", "neural"):
        p, q = s[stage]["psnr"], s[stage]["ssim"]
        print(f"  {stage:<14}{p['mean']:>10.3f}{p['p5']:>8.2f}{p['p95']:>8.2f}{q['mean']:>11.4f}")
    d = result["neuralOverCatmullRom"]
    print(f"  neural - Catmull-Rom: {d['psnrDb']:+.3f} dB  {d['ssim']:+.4f} SSIM  wins {d['winsOf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
