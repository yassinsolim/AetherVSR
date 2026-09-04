#!/usr/bin/env python3
"""Score upscalers on captured footage, with the clip as the statistical unit.

Two decisions here matter more than the metric code.

**The statistical unit is the clip, not the frame.** Twenty-four frames from one
drone shot are not twenty-four independent observations; they are one scene
sampled repeatedly. Treating frames as independent would let a 15-clip corpus
report p-values derived from 360 correlated samples, which is how a benchmark
manufactures certainty it has not earned. Frames are averaged within a clip
first, and every inferential statement is computed across clips.

**Comparisons are paired.** Every method sees the identical decoded frame, so
the meaningful quantity is the per-clip difference, not two independent group
means. Paired analysis also removes between-clip variance, which is enormous
here: a snowboard POV and a studio interview differ by more in absolute PSNR
than any upscaler does.

CPU only. Milestone 4.5 established that MPS on this machine returns different
answers for identical model-free computations, by up to 2.5 dB per image.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics as st
import sys

import torch
import torch.nn.functional as F
from PIL import Image

from evaluate import catmull_rom_2x, load_model, psnr, ssim


def load_png(path: str) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        raw = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        return raw.reshape(h, w, 3).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def lanczos_2x(lr: torch.Tensor) -> torch.Tensor:
    """Offline Lanczos, via PIL. Not implementable at the browser's per-frame cost.

    Included because a benchmark that only beats weak baselines has proved
    nothing, and because the gap between this and Catmull-Rom bounds what any
    conventional filter could contribute.
    """
    arr = (lr.squeeze(0).clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(arr, "RGB").resize(
        (lr.shape[3] * 2, lr.shape[2] * 2), Image.Resampling.LANCZOS
    )
    out = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
    return out.reshape(im.size[1], im.size[0], 3).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def frame_distribution(values: list[float]) -> dict:
    ordered = sorted(v for v in values if v != float("inf"))
    if not ordered:
        return {}
    return {
        "mean": st.fmean(ordered),
        "median": st.median(ordered),
        "p5": ordered[max(0, int(0.05 * len(ordered)) - 1)],
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "min": ordered[0],
        "max": ordered[-1],
        "n": len(ordered),
    }


def paired_stats(diffs: list[float], bootstrap: int = 20_000, seed: int = 12345) -> dict:
    """Paired clip-level summary: median, bootstrap CI, sign test, permutation."""
    n = len(diffs)
    if n == 0:
        return {}
    mean = st.fmean(diffs)
    wins = sum(1 for d in diffs if d > 0)

    # Bootstrap CI over clips, which is the resampling unit that matches the
    # design. Resampling frames would understate the interval badly.
    rng = torch.Generator().manual_seed(seed)
    tensor = torch.tensor(diffs, dtype=torch.float64)
    idx = torch.randint(0, n, (bootstrap, n), generator=rng)
    means = tensor[idx].mean(dim=1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()

    # Exact sign-flip permutation: under the null a clip's difference is equally
    # likely to have either sign. 2^n relabellings, enumerated when n is small.
    if n <= 20:
        count = 0
        for signs in itertools.product((1, -1), repeat=n):
            flipped = st.fmean(s * d for s, d in zip(signs, diffs))
            if abs(flipped) >= abs(mean) - 1e-12:
                count += 1
        perm_p = count / (2**n)
        exact = True
    else:
        perm_p, exact = float("nan"), False

    # Two-sided exact binomial sign test.
    def comb(a: int, b: int) -> int:
        return math_comb(a, b)

    from math import comb as math_comb

    tail = sum(comb(n, k) for k in range(0, min(wins, n - wins) + 1))
    sign_p = min(1.0, 2 * tail / (2**n))

    return {
        "clips": n,
        "meanDiff": mean,
        "medianDiff": st.median(diffs),
        "bootstrapCI95": [lo, hi],
        "wins": wins,
        "losses": n - wins,
        "signTestP": sign_p,
        "permutationP": perm_p,
        "permutationExact": exact,
        "smallestAttainableP": 2 / (2**n) if n <= 20 else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate upscalers on captured footage.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", default="data/captured")
    ap.add_argument("--conditions", default="h264_high,h264_typical,h264_poor")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    ap.add_argument("--limit-frames", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    model, payload = load_model(args.model)
    model = model.to(args.device)
    with open(os.path.join(args.root, "prepared.json"), encoding="utf-8") as fh:
        prepared = json.load(fh)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    per_clip: list[dict] = []

    for clip in prepared["clips"]:
        clip_root = os.path.join(args.root, "clips", clip["id"])
        master_dir = os.path.join(clip_root, "master")
        masters = sorted(os.listdir(master_dir))
        if args.limit_frames:
            masters = masters[: args.limit_frames]

        row = {"id": clip["id"], "category": clip["category"], "conditions": {}}
        for condition in conditions:
            decoded_dir = os.path.join(clip_root, "decoded", condition)
            if not os.path.isdir(decoded_dir):
                continue
            acc: dict[str, dict[str, list[float]]] = {
                m: {"psnr": [], "ssim": []} for m in ("neural", "catmull_rom", "bilinear", "lanczos", "nearest")
            }
            for name in masters:
                ref_path = os.path.join(master_dir, name)
                lr_path = os.path.join(decoded_dir, name)
                if not os.path.exists(lr_path):
                    continue
                ref = load_png(ref_path).to(args.device)
                lr = load_png(lr_path).to(args.device)
                with torch.no_grad():
                    outs = {
                        "neural": model(lr).clamp(0, 1),
                        "catmull_rom": catmull_rom_2x(lr),
                        "bilinear": F.interpolate(lr, scale_factor=2, mode="bilinear", align_corners=False).clamp(0, 1),
                        "lanczos": lanczos_2x(lr.cpu()).to(args.device),
                        "nearest": F.interpolate(lr, scale_factor=2, mode="nearest").clamp(0, 1),
                    }
                for key, out in outs.items():
                    if out.shape != ref.shape:
                        raise SystemExit(
                            f"geometry mismatch {clip['id']}/{condition}/{key}: "
                            f"{tuple(out.shape)} against {tuple(ref.shape)}"
                        )
                    acc[key]["psnr"].append(psnr(out, ref))
                    acc[key]["ssim"].append(ssim(out, ref))

            if not acc["neural"]["psnr"]:
                continue
            worst = min(range(len(acc["neural"]["psnr"])),
                        key=lambda i: acc["neural"]["psnr"][i] - acc["catmull_rom"]["psnr"][i])
            best = max(range(len(acc["neural"]["psnr"])),
                       key=lambda i: acc["neural"]["psnr"][i] - acc["catmull_rom"]["psnr"][i])
            row["conditions"][condition] = {
                m: {k: frame_distribution(v) for k, v in acc[m].items()} for m in acc
            }
            row["conditions"][condition]["clipMeanDiffVsCatmull"] = {
                "psnr": st.fmean(a - b for a, b in zip(acc["neural"]["psnr"], acc["catmull_rom"]["psnr"])),
                "ssim": st.fmean(a - b for a, b in zip(acc["neural"]["ssim"], acc["catmull_rom"]["ssim"])),
            }
            row["conditions"][condition]["worstFrame"] = worst
            row["conditions"][condition]["bestFrame"] = best
            print(
                f"  {clip['id'][:34]:<34} {condition:<13} "
                f"neural {row['conditions'][condition]['neural']['psnr']['mean']:6.3f}  "
                f"catmull {row['conditions'][condition]['catmull_rom']['psnr']['mean']:6.3f}  "
                f"delta {row['conditions'][condition]['clipMeanDiffVsCatmull']['psnr']:+6.3f}",
                file=sys.stderr,
            )
        per_clip.append(row)

    aggregate = {}
    for condition in conditions:
        diffs = [c["conditions"][condition]["clipMeanDiffVsCatmull"]["psnr"]
                 for c in per_clip if condition in c["conditions"]]
        ssim_diffs = [c["conditions"][condition]["clipMeanDiffVsCatmull"]["ssim"]
                      for c in per_clip if condition in c["conditions"]]
        aggregate[condition] = {
            "psnr": paired_stats(diffs),
            "ssimMeanDiff": st.fmean(ssim_diffs) if ssim_diffs else float("nan"),
        }

    report = {
        "schema": "aethervsr.captured-eval/1",
        "model": os.path.basename(args.model),
        "modelSeed": payload.get("training", {}).get("seed"),
        "modelDegradation": payload.get("training", {}).get("degradationProfile"),
        "device": args.device,
        "statisticalUnit": (
            "clip. Frames within a clip are averaged first; all inference is across clips, "
            "because frames from one scene are not independent observations."
        ),
        "comparison": "paired: every method sees the identical decoded frame",
        "perClip": per_clip,
        "aggregate": aggregate,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
