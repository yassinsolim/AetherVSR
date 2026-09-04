#!/usr/bin/env python3
"""Validate metrics on controls before trusting them on the model.

Milestone 4.5 published VMAF for 25 cells and then found it ranked
nearest-neighbour above Lanczos in 20 of them. A metric that prefers
nearest-neighbour cannot be used to choose an upscaler, and the failure was
only caught because a reviewer checked. So the check runs first now.

The control is an ordering that is not in dispute. On real footage under mild
compression, reconstruction quality should go:

    nearest  <  bilinear  <  Catmull-Rom  <=  Lanczos

Any metric that inverts this on the majority of clips is recorded as
diagnostic-only and no conclusion is allowed to rest on it.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F
from PIL import Image

from evaluate import catmull_rom_2x, psnr, ssim

FFMPEG = os.environ.get("AETHER_FFMPEG", "/opt/homebrew/bin/ffmpeg")
EXPECTED_ORDER = ["nearest", "bilinear", "catmull_rom", "lanczos"]


def load_png(path: str) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        raw = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        return raw.reshape(h, w, 3).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def save_png(t: torch.Tensor, path: str) -> None:
    arr = (t.squeeze(0).clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(arr, "RGB").save(path)


def lanczos_2x(lr: torch.Tensor) -> torch.Tensor:
    arr = (lr.squeeze(0).clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(arr, "RGB").resize((lr.shape[3] * 2, lr.shape[2] * 2), Image.Resampling.LANCZOS)
    out = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
    return out.reshape(im.size[1], im.size[0], 3).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def upscales(lr: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "nearest": F.interpolate(lr, scale_factor=2, mode="nearest").clamp(0, 1),
        "bilinear": F.interpolate(lr, scale_factor=2, mode="bilinear", align_corners=False).clamp(0, 1),
        "catmull_rom": catmull_rom_2x(lr),
        "lanczos": lanczos_2x(lr),
    }


def vmaf_score(ref_dir: str, dist_dir: str, frames: int) -> float | None:
    """VMAF over PNG sequences, both converted identically to BT.709 limited yuv."""
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "v.json")
        conv = "scale=in_range=pc:out_range=tv,format=yuv420p,setsar=1"
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-framerate", "24", "-i", os.path.join(dist_dir, "frame_%04d.png"),
            "-framerate", "24", "-i", os.path.join(ref_dir, "frame_%04d.png"),
            "-frames:v", str(frames),
            "-filter_complex",
            f"[0:v]{conv}[d];[1:v]{conv}[r];[d][r]libvmaf=log_path={log}:log_fmt=json",
            "-f", "null", "-",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(log):
            return None
        with open(log, encoding="utf-8") as fh:
            data = json.load(fh)
        vals = [f["metrics"]["vmaf"] for f in data.get("frames", [])]
        return st.fmean(vals) if vals else None


def kendall_ok(scores: dict[str, float]) -> bool:
    """True when the observed ranking respects the undisputed ordering."""
    ordered = [scores[k] for k in EXPECTED_ORDER if k in scores]
    return all(a <= b + 1e-9 for a, b in zip(ordered, ordered[1:]))


def main() -> int:
    ap = argparse.ArgumentParser(description="Metric sanity controls on captured footage.")
    ap.add_argument("--root", default="data/captured")
    ap.add_argument("--condition", default="h264_high")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--with-vmaf", action="store_true")
    ap.add_argument("--out", default="results/captured-metric-sanity.json")
    args = ap.parse_args()

    with open(os.path.join(args.root, "prepared.json"), encoding="utf-8") as fh:
        prepared = json.load(fh)

    per_clip, verdicts = {}, {"psnr": [], "ssim": [], "vmaf": []}
    for clip in prepared["clips"]:
        cid = clip["id"]
        root = os.path.join(args.root, "clips", cid)
        mdir, ddir = os.path.join(root, "master"), os.path.join(root, "decoded", args.condition)
        if not os.path.isdir(ddir):
            continue
        names = sorted(os.listdir(mdir))[: args.frames]
        acc = {m: {"psnr": [], "ssim": []} for m in EXPECTED_ORDER}
        for n in names:
            ref = load_png(os.path.join(mdir, n))
            lr = load_png(os.path.join(ddir, n))
            for key, out in upscales(lr).items():
                acc[key]["psnr"].append(psnr(out, ref))
                acc[key]["ssim"].append(ssim(out, ref))

        entry = {m: {k: st.fmean(v) for k, v in acc[m].items()} for m in acc}
        p_ok = kendall_ok({m: entry[m]["psnr"] for m in EXPECTED_ORDER})
        s_ok = kendall_ok({m: entry[m]["ssim"] for m in EXPECTED_ORDER})
        verdicts["psnr"].append(p_ok)
        verdicts["ssim"].append(s_ok)

        if args.with_vmaf:
            vm = {}
            with tempfile.TemporaryDirectory() as tmp:
                for key in EXPECTED_ORDER:
                    d = os.path.join(tmp, key)
                    os.makedirs(d)
                    for i, n in enumerate(names, 1):
                        lr = load_png(os.path.join(ddir, n))
                        save_png(upscales(lr)[key], os.path.join(d, f"frame_{i:04d}.png"))
                    score = vmaf_score(mdir, d, len(names))
                    if score is not None:
                        vm[key] = score
            if len(vm) == len(EXPECTED_ORDER):
                entry["vmaf"] = vm
                verdicts["vmaf"].append(kendall_ok(vm))

        per_clip[cid] = entry
        v = f"psnr {'OK' if p_ok else 'INVERTED'}  ssim {'OK' if s_ok else 'INVERTED'}"
        if "vmaf" in entry:
            v += f"  vmaf {'OK' if kendall_ok(entry['vmaf']) else 'INVERTED'}"
        print(f"  {cid[:40]:<40} {v}", file=sys.stderr)

    summary = {}
    for metric, results in verdicts.items():
        if not results:
            continue
        ok = sum(results)
        summary[metric] = {
            "clipsChecked": len(results),
            "clipsRespectingOrder": ok,
            "verdict": "USABLE" if ok >= 0.8 * len(results) else "DIAGNOSTIC ONLY - ordering inverted",
        }

    report = {
        "schema": "aethervsr.captured-metric-sanity/1",
        "condition": args.condition,
        "control": "nearest <= bilinear <= Catmull-Rom <= Lanczos on real footage",
        "why": ("Milestone 4.5 published VMAF for 25 cells then found it ranked nearest above "
                "Lanczos in 20. A metric that prefers nearest cannot choose an upscaler."),
        "summary": summary,
        "perClip": per_clip,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print("\n" + json.dumps(summary, indent=1), file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
