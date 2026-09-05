#!/usr/bin/env python3
"""Score the Milestone 5.5 candidates on one common validation set.

Each training condition ships its own validation pairs, degraded the way that
condition degrades things. Those are the right sets for choosing a checkpoint
*within* a run and the wrong sets for comparing runs: a model trained on the
heavy-CRF distribution is validated on heavily compressed patches, whose PSNR
ceiling is simply lower. Comparing conditions on their own validation sets
would rank them by how hard their own data is.

So this builds one common set instead, from the same three held-out validation
videos every condition shares, encoded at fixed CRFs with the GOP structure a
real browser receives. Every model sees identical input.

This is validation, not test. The frozen captured corpus is not touched here;
model selection finishes before it is read.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys

import torch
import torch.nn.functional as F
from PIL import Image

from evaluate import catmull_rom_2x, load_model, psnr, ssim

FFMPEG = os.environ.get("AETHER_FFMPEG", "/opt/homebrew/bin/ffmpeg")
HR_W, HR_H = 2560, 1440
LR_W, LR_H = 1280, 720
GOP_PARAMS = "keyint=48:min-keyint=24:scenecut=40:bframes=3:ref=3"
SEQ_FRAMES = 12


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def load_png(path: str) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        raw = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        return raw.reshape(h, w, 3).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def build_common_set(pairs_meta: dict, sources: str, manifest: str, out: str,
                     crfs: list[int], fps: int) -> dict:
    """One fixed-CRF GOP validation set, shared by every candidate."""
    with open(manifest, encoding="utf-8") as fh:
        clips = {c["id"]: c for c in json.load(fh)["clips"]}
    index: list[dict] = []
    for cid in sorted(pairs_meta["valClips"]):
        src = os.path.join(sources, clips[cid]["source_file"])
        hr_dir = os.path.join(out, cid, "hr")
        if not os.path.isdir(hr_dir) or len(os.listdir(hr_dir)) < SEQ_FRAMES:
            os.makedirs(hr_dir, exist_ok=True)
            run([
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-ss", "3.0", "-i", src, "-frames:v", str(SEQ_FRAMES),
                "-fps_mode", "passthrough",
                "-vf", f"scale={HR_W}:{HR_H}:flags=area,format=rgb24",
                os.path.join(hr_dir, "hr_%04d.png"),
            ])
        for crf in crfs:
            lr_dir = os.path.join(out, cid, f"crf{crf}")
            if not os.path.isdir(lr_dir) or len(os.listdir(lr_dir)) < SEQ_FRAMES:
                os.makedirs(lr_dir, exist_ok=True)
                clip = os.path.join(lr_dir, "c.mp4")
                run([
                    FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                    "-framerate", str(fps), "-start_number", "1",
                    "-i", os.path.join(hr_dir, "hr_%04d.png"),
                    "-vf", f"scale={LR_W}:{LR_H}:flags=bicubic,format=yuv420p",
                    "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
                    "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                    "-x264-params", GOP_PARAMS, "-threads", "1", clip,
                ])
                run([
                    FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", clip,
                    "-vf", "scale=in_range=tv:out_range=pc,format=rgb24",
                    "-fps_mode", "passthrough", os.path.join(lr_dir, "lr_%04d.png"),
                ])
                os.remove(clip)
            index.append({"clip": cid, "crf": crf, "hr": hr_dir, "lr": lr_dir})
    return {"index": index, "crfs": crfs, "clips": sorted(pairs_meta["valClips"])}


def score(model_path: str | None, common: dict) -> dict:
    """PSNR/SSIM per (clip, CRF). `None` scores production Catmull-Rom."""
    model = None
    if model_path is not None:
        model, _ = load_model(model_path)
        model.eval()
    rows: list[dict] = []
    for entry in common["index"]:
        hr_names = sorted(n for n in os.listdir(entry["hr"]) if n.endswith(".png"))
        lr_names = sorted(n for n in os.listdir(entry["lr"]) if n.endswith(".png"))
        ps, ss = [], []
        for hn, ln in zip(hr_names, lr_names):
            hr = load_png(os.path.join(entry["hr"], hn))
            lr = load_png(os.path.join(entry["lr"], ln))
            with torch.no_grad():
                out = model(lr).clamp(0, 1) if model is not None else catmull_rom_2x(lr)
            ps.append(psnr(out, hr))
            ss.append(ssim(out, hr))
        rows.append({"clip": entry["clip"], "crf": entry["crf"],
                     "psnr": st.fmean(ps), "ssim": st.fmean(ss), "frames": len(ps)})
    return {"perCell": rows,
            "byCrf": {str(c): st.fmean(r["psnr"] for r in rows if r["crf"] == c)
                      for c in common["crfs"]},
            "mean": st.fmean(r["psnr"] for r in rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Common-set validation for Milestone 5.5.")
    ap.add_argument("--models", default="models/m55")
    ap.add_argument("--pairs", default="/tmp/pairs/gop_poor",
                    help="any condition; only its val clip list is used")
    ap.add_argument("--sources", default="data/captured-train/sources")
    ap.add_argument("--manifest", default="data/captured-train/manifest.json")
    ap.add_argument("--work", default="data/captured-train/commonval")
    ap.add_argument("--production", default="public/models/aethersr-c16d2-realistic.json")
    ap.add_argument("--crfs", default="18,26,34")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", default="results/m55-validation.json")
    args = ap.parse_args()

    with open(os.path.join(args.pairs, "pairs.json"), encoding="utf-8") as fh:
        pairs_meta = json.load(fh)
    crfs = [int(c) for c in args.crfs.split(",") if c.strip()]

    print("building common validation set...", file=sys.stderr)
    common = build_common_set(pairs_meta, args.sources, args.manifest, args.work, crfs, args.fps)
    print(f"  {len(common['clips'])} clips x {len(crfs)} CRFs = {len(common['index'])} cells\n",
          file=sys.stderr)

    entries: list[tuple[str, str | None]] = [("catmull_rom", None)]
    if os.path.exists(args.production):
        entries.append(("production-c16d2-realistic", args.production))
    for name in sorted(os.listdir(args.models)):
        if name.endswith(".json"):
            entries.append((name[:-5], os.path.join(args.models, name)))

    scored: dict[str, dict] = {}
    for name, path in entries:
        scored[name] = score(path, common)
        by = scored[name]["byCrf"]
        print(f"  {name:<34} " + "  ".join(f"crf{c} {by[str(c)]:6.3f}" for c in crfs)
              + f"   mean {scored[name]['mean']:6.3f}", file=sys.stderr)

    # Aggregate the 2x2 by condition, averaging its seeds. One seed is a draw;
    # the condition is the thing under test.
    base = scored["catmull_rom"]
    conditions: dict[str, dict] = {}
    for name in scored:
        if "-seed" not in name:
            continue
        cond = name.rsplit("-seed", 1)[0]
        conditions.setdefault(cond, {"seeds": []})["seeds"].append(name)
    for cond, info in conditions.items():
        seeds = info["seeds"]
        info["n"] = len(seeds)
        info["byCrf"] = {}
        for c in crfs:
            vals = [scored[s]["byCrf"][str(c)] for s in seeds]
            info["byCrf"][str(c)] = {
                "mean": st.fmean(vals),
                "sd": st.stdev(vals) if len(vals) > 1 else 0.0,
                "deltaVsCatmull": st.fmean(vals) - base["byCrf"][str(c)],
            }
        vals = [scored[s]["mean"] for s in seeds]
        info["mean"] = st.fmean(vals)
        info["sd"] = st.stdev(vals) if len(vals) > 1 else 0.0
        info["deltaVsCatmull"] = info["mean"] - base["mean"]

    report = {
        "schema": "aethervsr.m55-validation/1",
        "note": ("Common fixed-CRF GOP validation set built from the three held-out "
                 "validation videos every condition shares. This is validation; the "
                 "frozen captured test set is not read here."),
        "commonSet": {"clips": common["clips"], "crfs": crfs, "frames": SEQ_FRAMES,
                      "gopParams": GOP_PARAMS},
        "models": scored,
        "conditions": conditions,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    if conditions:
        print("\nBY CONDITION (mean over seeds, delta vs Catmull-Rom):", file=sys.stderr)
        print(f"  {'condition':<18}{'n':>3}" + "".join(f"{'crf'+str(c):>12}" for c in crfs)
              + f"{'mean':>10}", file=sys.stderr)
        for cond in sorted(conditions, key=lambda k: -conditions[k]["deltaVsCatmull"]):
            info = conditions[cond]
            print(f"  {cond:<18}{info['n']:>3}"
                  + "".join(f"{info['byCrf'][str(c)]['deltaVsCatmull']:>+12.4f}" for c in crfs)
                  + f"{info['deltaVsCatmull']:>+10.4f}", file=sys.stderr)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
