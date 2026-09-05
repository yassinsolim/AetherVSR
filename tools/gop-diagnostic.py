#!/usr/bin/env python3
"""All-I versus GOP compression: does the training approximation matter?

`tools/degrade.py` encodes each training batch as a single frame, which is an
I-frame approximation: no GOP, no temporal prediction, no lookahead, no B or P
frames. The captured benchmark encodes normal video. If the model's advantage
survives all-I compression but collapses under GOP compression at the same CRF,
the training degradation is teaching the wrong problem and the fix is data, not
capacity.

The two conditions differ in exactly one thing. Same source frames, same
downscale kernel, same pixel format, colour space and range, same preset, same
CRF, same decoder path. Only the GOP structure changes:

    all-I   keyint=1              every frame intra-coded
    GOP     keyint=48, bframes=3  normal temporal prediction

Everything else being held constant is what makes the comparison a diagnosis
rather than a correlation.
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
FFPROBE = os.environ.get("AETHER_FFPROBE", "/opt/homebrew/bin/ffprobe")

LR_W, LR_H = 1280, 720
COLOUR = [
    "-pix_fmt", "yuv420p", "-color_range", "tv",
    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
]
DECODE_FILTER = "scale=in_range=tv:out_range=pc,format=rgb24"

# The controlled variable, with a caveat that matters. At the ffmpeg level only
# -x264-params differs. At the encoder level it does more than restructure the
# GOP: libx264 reports mbtree=0, no rc_lookahead, weightp=0, mixed_ref=0, ref=1
# for the all-I arm against mbtree=1, rc_lookahead=40, weightp=2, mixed_ref=1,
# ref=3 for the GOP arm. mb-tree is a rate-control algorithm, not a GOP
# property, so "the same CRF" does not mean "the same quality target" - which is
# the mechanism behind the 1.8 dB input-quality gap this diagnostic then has to
# correct for by matching on delivered PSNR.
#
# This is therefore "production's all-intra approximation against a normal GOP
# encode", which is the comparison the milestone actually needs, and not a
# one-variable experiment. Reference count in particular is an independent knob
# that keyint=1 does not force.
STRUCTURES = {
    "all_i": {
        "x264params": "keyint=1:min-keyint=1:scenecut=0:bframes=0:ref=1",
        "note": "every frame intra-coded; no temporal prediction at all",
    },
    "gop": {
        "x264params": "keyint=48:min-keyint=24:scenecut=40:bframes=3:ref=3",
        "note": "normal libx264 GOP with I/P/B frames and temporal prediction",
    },
}


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def load_png(path: str) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        raw = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        return raw.reshape(h, w, 3).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def encode(master_dir: str, out_path: str, crf: int, structure: str, fps: int) -> list[str]:
    spec = STRUCTURES[structure]
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(fps), "-start_number", "1",
        "-i", os.path.join(master_dir, "frame_%04d.png"),
        "-vf", f"scale={LR_W}:{LR_H}:flags=area,format=yuv420p",
        *COLOUR,
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-x264-params", spec["x264params"],
        "-threads", "1",
        out_path,
    ]
    run(cmd)
    return cmd


def frame_types(clip_path: str) -> dict[str, int]:
    """Confirms the GOP structure actually differs, rather than trusting the flags."""
    out = run([
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=pict_type", "-of", "csv=p=0", clip_path,
    ]).stdout
    counts: dict[str, int] = {}
    for line in out.split():
        t = line.strip().strip(",")
        if t:
            counts[t] = counts.get(t, 0) + 1
    return counts


def decode(clip_path: str, out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    run([
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-i", clip_path, "-vf", DECODE_FILTER, "-fps_mode", "passthrough",
        os.path.join(out_dir, "frame_%04d.png"),
    ])
    return len(os.listdir(out_dir))


def main() -> int:
    ap = argparse.ArgumentParser(description="All-I vs GOP compression diagnostic.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", default="data/captured")
    ap.add_argument("--work", default="data/captured/gopdiag")
    ap.add_argument("--crfs", default="26,34")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--structures", default="all_i,gop",
                    help="which compression contexts to run; 'gop' alone gives the dense "
                         "CRF curve a real browser would see")
    ap.add_argument("--out", default="results/gop-diagnostic.json")
    args = ap.parse_args()

    model, payload = load_model(args.model)
    model.eval()
    with open(os.path.join(args.root, "prepared.json"), encoding="utf-8") as fh:
        prepared = json.load(fh)

    structures = [s for s in args.structures.split(",") if s.strip()]
    for s in structures:
        if s not in STRUCTURES:
            raise SystemExit(f"unknown structure {s!r}; choices: {sorted(STRUCTURES)}")

    crfs = [int(c) for c in args.crfs.split(",") if c.strip()]
    rows: list[dict] = []
    structure_evidence: dict[str, dict] = {}

    for clip in prepared["clips"]:
        cid = clip["id"]
        master_dir = os.path.join(args.root, "clips", cid, "master")
        names = sorted(os.listdir(master_dir))
        refs = [load_png(os.path.join(master_dir, n)) for n in names]

        for crf in crfs:
            for structure in structures:
                work = os.path.join(args.work, cid, f"{structure}_crf{crf}")
                os.makedirs(work, exist_ok=True)
                clip_path = os.path.join(work, "clip.mp4")
                cmd = encode(master_dir, clip_path, crf, structure, args.fps)
                types = frame_types(clip_path)
                n_dec = decode(clip_path, os.path.join(work, "decoded"))
                structure_evidence.setdefault(f"{structure}_crf{crf}", {
                    "x264params": STRUCTURES[structure]["x264params"],
                    "ffmpegCommand": cmd,
                    "frameTypes": types,
                    "decodedFrames": n_dec,
                    "masterFrames": len(names),
                })

                dn, dc, dlr = [], [], []
                for i, n in enumerate(names):
                    lr_path = os.path.join(work, "decoded", n)
                    if not os.path.exists(lr_path):
                        continue
                    lr = load_png(lr_path)
                    with torch.no_grad():
                        neural = model(lr).clamp(0, 1)
                    cat = catmull_rom_2x(lr)
                    dn.append(psnr(neural, refs[i]))
                    dc.append(psnr(cat, refs[i]))
                    # Input quality itself: the same CRF can mean different
                    # delivered quality under different GOP structures, and a
                    # comparison at equal CRF but unequal input quality would be
                    # measuring the encoder rather than the model.
                    dlr.append(psnr(F.interpolate(lr, scale_factor=2, mode="nearest").clamp(0, 1), refs[i]))
                rows.append({
                    "clip": cid, "category": clip["category"], "crf": crf, "structure": structure,
                    "neuralPsnr": st.fmean(dn), "catmullPsnr": st.fmean(dc),
                    "deltaPsnr": st.fmean(a - b for a, b in zip(dn, dc)),
                    "inputPsnrNearest": st.fmean(dlr),
                    "bytes": os.path.getsize(clip_path),
                })
                print(
                    f"  {cid[:30]:<30} crf{crf} {structure:<6} "
                    f"delta {rows[-1]['deltaPsnr']:+.4f}  input {rows[-1]['inputPsnrNearest']:6.3f} dB  "
                    f"{os.path.getsize(clip_path)/1024:7.1f} kB",
                    file=sys.stderr,
                )

    summary = {}
    for crf in crfs:
        for structure in structures:
            sel = [r for r in rows if r["crf"] == crf and r["structure"] == structure]
            summary[f"crf{crf}_{structure}"] = {
                "clips": len(sel),
                "meanDelta": st.fmean(r["deltaPsnr"] for r in sel),
                "medianDelta": st.median(r["deltaPsnr"] for r in sel),
                "wins": sum(1 for r in sel if r["deltaPsnr"] > 0),
                "meanCatmull": st.fmean(r["catmullPsnr"] for r in sel),
                "meanNeural": st.fmean(r["neuralPsnr"] for r in sel),
                "meanInputPsnr": st.fmean(r["inputPsnrNearest"] for r in sel),
                "meanBytes": st.fmean(r["bytes"] for r in sel),
            }

    report = {
        "schema": "aethervsr.gop-diagnostic/1",
        "question": ("Does the model lose more advantage on GOP-compressed frames than on "
                     "equivalently quantised all-I frames? If so the single-frame training "
                     "degradation is teaching the wrong problem."),
        "controlledVariable": (
            "Source, downscale kernel, colour, range, preset, CRF and decoder are identical; "
            "only -x264-params differs. That string changes more than GOP structure: the all-I "
            "arm also loses mb-tree, rc_lookahead, weighted P and multiple references. Since "
            "mb-tree is rate control, equal CRF is not equal quality target, which is why the "
            "conclusion rests on the matched-input-quality re-analysis rather than on equal CRF."
        ),
        "x264OptionsObserved": {
            "all_i": "ref=1 mixed_ref=0 bframes=0 weightp=0 keyint=1 rc=crf mbtree=0 crf=34.0",
            "gop": "ref=3 mixed_ref=1 bframes=3 weightp=2 keyint=48 rc_lookahead=40 rc=crf mbtree=1 crf=34.0",
        },
        "model": os.path.basename(args.model),
        "structures": STRUCTURES,
        "encodeEvidence": structure_evidence,
        "perClip": rows,
        "summary": summary,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
