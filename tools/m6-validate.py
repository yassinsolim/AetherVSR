#!/usr/bin/env python3
"""Score models on the v2 validation corpus: CRF curve and content classes.

Model selection happens here and nowhere else. The frozen ten-clip set is a
regression benchmark now, and the confirmation set is closed until every
decision is made, so this is the only instrument allowed to choose anything.

Two properties the Milestone 5.5 validation lacked:

  breadth   sixteen clips from fifteen creators across eight content classes,
            against three clips from three creators with no labels. The old set
            was being asked to resolve 0.005 dB seed differences on three
            correlated videos, which it could not do.
  classes   every number is reported per content class as well as pooled,
            because a model that wins overall by getting very good at the
            dominant category is not a better model, it is a narrower one.

The evaluation cache is built once from the streamed masters and reused across
every candidate, so comparing twenty models costs one preparation.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import subprocess
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import catmull_rom_2x, load_model, psnr, ssim  # noqa: E402
from video_degrade_lib import extract_hr  # noqa: E402

FFMPEG = os.environ.get("AETHER_FFMPEG", "/opt/homebrew/bin/ffmpeg")
UA = "AetherVSR/0.1 (https://github.com/yassinsolim/AetherVSR) corpus-eval"
HR_W, HR_H = 2560, 1440
LR_W, LR_H = 1280, 720
GOP_PARAMS = "keyint=48:min-keyint=24:scenecut=40:bframes=3:ref=3"


def run(cmd: list[str], timeout: int = 900) -> bool:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def load_png(path: str) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        raw = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        return raw.reshape(h, w, 3).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def build_cache(manifest: dict, cache: str, crfs: list[int], frames: int, fps: int,
                start: float) -> list[dict]:
    cells = []
    for clip in manifest["clips"]:
        cid = clip["id"]
        hr_dir = os.path.join(cache, cid, "hr")
        have = len(os.listdir(hr_dir)) if os.path.isdir(hr_dir) else 0
        if have < frames:
            # Seek late enough to miss the training sequences' windows, but back
            # off toward the start for clips too short to allow it rather than
            # dropping them: several validation clips are under 15 seconds and
            # a fixed 12 s offset silently lost thirteen of sixteen.
            dur = float(clip.get("duration") or 0)
            at = start if dur <= 0 or dur > start + frames / fps + 1 else max(0.5, dur * 0.3)
            if extract_hr(clip["source_url"], at, hr_dir, frames) < frames:
                print(f"  SKIP {cid}: extraction failed", file=sys.stderr)
                continue
        for crf in crfs:
            lr_dir = os.path.join(cache, cid, f"crf{crf}")
            if os.path.isdir(lr_dir) and len(os.listdir(lr_dir)) >= frames:
                cells.append({"clip": cid, "category": clip["category"], "crf": crf,
                              "hr": hr_dir, "lr": lr_dir})
                continue
            os.makedirs(lr_dir, exist_ok=True)
            mp4 = os.path.join(lr_dir, "c.mp4")
            ok = run([
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", str(fps), "-start_number", "1",
                "-i", os.path.join(hr_dir, "hr_%04d.png"),
                "-vf", f"scale={LR_W}:{LR_H}:flags=bicubic,format=yuv420p",
                "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
                "-color_primaries", "bt709", "-color_trc", "bt709",
                "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                "-x264-params", GOP_PARAMS, "-threads", "1", mp4,
            ]) and run([
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", mp4,
                "-vf", "scale=in_range=tv:out_range=pc,format=rgb24",
                "-fps_mode", "passthrough", os.path.join(lr_dir, "lr_%04d.png"),
            ])
            if os.path.exists(mp4):
                os.remove(mp4)
            if ok:
                cells.append({"clip": cid, "category": clip["category"], "crf": crf,
                              "hr": hr_dir, "lr": lr_dir})
        print(f"  cached {cid[:46]:<46} {len(crfs)} CRFs", file=sys.stderr)
    return cells


def score(model_path: str | None, cells: list[dict]) -> dict:
    model = None
    if model_path is not None:
        model, _ = load_model(model_path)
        model.eval()
    rows = []
    for cell in cells:
        hr_names = sorted(n for n in os.listdir(cell["hr"]) if n.endswith(".png"))
        lr_names = sorted(n for n in os.listdir(cell["lr"]) if n.endswith(".png"))
        ps, ss = [], []
        for hn, ln in zip(hr_names, lr_names):
            hr = load_png(os.path.join(cell["hr"], hn))
            lr = load_png(os.path.join(cell["lr"], ln))
            with torch.no_grad():
                out = model(lr).clamp(0, 1) if model is not None else catmull_rom_2x(lr)
            ps.append(psnr(out, hr))
            ss.append(ssim(out, hr))
        rows.append({"clip": cell["clip"], "category": cell["category"], "crf": cell["crf"],
                     "psnr": st.fmean(ps), "ssim": st.fmean(ss)})
    return {"perCell": rows}


def summarise(model_rows: list[dict], base_rows: list[dict], crfs: list[int]) -> dict:
    base = {(r["clip"], r["crf"]): r for r in base_rows}
    deltas = [{**r, "delta": r["psnr"] - base[(r["clip"], r["crf"])]["psnr"],
               "deltaSsim": r["ssim"] - base[(r["clip"], r["crf"])]["ssim"]}
              for r in model_rows]
    by_crf = {str(c): st.fmean(d["delta"] for d in deltas if d["crf"] == c) for c in crfs}
    cats = sorted({d["category"] for d in deltas})
    by_cat = {c: st.fmean(d["delta"] for d in deltas if d["category"] == c) for c in cats}
    # Per clip first, then across clips: 12 frames of one video are not 12
    # independent observations, and pooling them would shrink every interval.
    per_clip = collections.defaultdict(list)
    for d in deltas:
        per_clip[d["clip"]].append(d["delta"])
    clip_means = [st.fmean(v) for v in per_clip.values()]
    return {
        "meanDelta": st.fmean(d["delta"] for d in deltas),
        "clipMeanDelta": st.fmean(clip_means),
        "clipWins": sum(1 for m in clip_means if m > 0),
        "clips": len(clip_means),
        "byCrf": by_crf,
        "byCategory": by_cat,
        "worstCategory": min(by_cat, key=lambda k: by_cat[k]) if by_cat else None,
        "worstCategoryDelta": min(by_cat.values()) if by_cat else None,
        "meanDeltaSsim": st.fmean(d["deltaSsim"] for d in deltas),
        "perCell": deltas,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Validation CRF curve and content-class analysis.")
    ap.add_argument("--manifest", default="data/captured-val/manifest.json")
    ap.add_argument("--cache", default="/tmp/m6valcache")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--crfs", default="18,22,26,30,32,34,36")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--start", type=float, default=12.0,
                    help="seek offset; kept away from the training sequences' windows")
    ap.add_argument("--out", default="results/m6-validation.json")
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    crfs = [int(c) for c in args.crfs.split(",") if c.strip()]

    cells = build_cache(manifest, args.cache, crfs, args.frames, args.fps, args.start)
    print(f"\n  {len({c['clip'] for c in cells})} clips x {len(crfs)} CRFs = {len(cells)} cells\n",
          file=sys.stderr)

    paths = list(args.models)
    if args.model_dir:
        paths += [os.path.join(args.model_dir, n) for n in sorted(os.listdir(args.model_dir))
                  if n.endswith(".json")]

    base = score(None, cells)["perCell"]
    out = {"schema": "aethervsr.m6-validation/1",
           "manifest": args.manifest, "crfs": crfs, "frames": args.frames,
           "clips": sorted({c["clip"] for c in cells}),
           "categories": dict(collections.Counter(c["category"] for c in cells)),
           "statisticalUnit": "clip; frames averaged within a clip first",
           "models": {}}
    for p in paths:
        name = os.path.basename(p)[:-5]
        rows = score(p, cells)["perCell"]
        out["models"][name] = summarise(rows, base, crfs)
        s = out["models"][name]
        print(f"  {name:<38} mean {s['clipMeanDelta']:+.4f}  wins {s['clipWins']}/{s['clips']}  "
              f"worst-cat {s['worstCategory']} {s['worstCategoryDelta']:+.4f}", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
