#!/usr/bin/env python3
"""Generate training pairs from real video under real compression context.

The production model is trained on still photographs degraded one frame at a
time: each patch mosaic is encoded as a single H.264 I-frame. That is a
compression approximation with no GOP, no temporal prediction and no B-frames.
This tool produces the alternative - genuine video encodes of temporally
coherent sequences - so that the two can be compared with the architecture and
everything else held fixed.

Both conditions are produced by the same code path, differing only in the
x264 GOP parameters, so a difference in the trained models cannot come from an
incidental difference in the pipeline:

    all_i   keyint=1               every frame intra-coded (matches training today)
    gop     keyint=48, bframes=3   normal temporal prediction

The model itself remains single-frame: pairs are emitted per frame and carry no
temporal information into the network. Only the *degradation* gains temporal
context, which is what a real decoder delivers.

CRF is sampled per sequence, not per frame, because a real encoder holds a
rate-control setting across a GOP. Two distributions are supported:

    uniform  CRF ~ U[18, 36]           what the shipped model was trained on
    poor     CRF ~ U[18, 36] biased    two-thirds of draws in [28, 36]

Everything is seeded and recorded, so a given (condition, seed) reproduces the
same clips, the same crops, the same CRFs and the same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

FFMPEG = os.environ.get("AETHER_FFMPEG", "/opt/homebrew/bin/ffmpeg")
FFPROBE = os.environ.get("AETHER_FFPROBE", "/opt/homebrew/bin/ffprobe")


def read_u8(path: str) -> torch.Tensor:
    """PNG -> uint8 CHW. Stays in uint8: patches are stored, not computed on."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        raw = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        return raw.reshape(h, w, 3).permute(2, 0, 1)

HR_W, HR_H = 2560, 1440
LR_W, LR_H = 1280, 720

# Colour handling is pinned end-to-end and matches the captured benchmark, so a
# model trained here and evaluated there does not see a range or matrix shift.
ENCODE_COLOUR = [
    "-pix_fmt", "yuv420p", "-color_range", "tv",
    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
]
DECODE_FILTER = "scale=in_range=tv:out_range=pc,format=rgb24"

STRUCTURES: dict[str, dict] = {
    "all_i": {
        "x264params": "keyint=1:min-keyint=1:scenecut=0:bframes=0:ref=1",
        "description": "every frame intra-coded; no temporal prediction (matches tools/degrade.py)",
    },
    "gop": {
        "x264params": "keyint=48:min-keyint=24:scenecut=40:bframes=3:ref=3",
        "description": "normal libx264 GOP with I/P/B frames and temporal prediction",
    },
}

CRF_DISTRIBUTIONS: dict[str, dict] = {
    "uniform": {
        "range": [18, 36],
        "description": "CRF uniform on [18, 36]; the distribution the shipped model was trained on",
    },
    "poor": {
        "range": [18, 36],
        "heavy_range": [28, 36],
        "heavy_prob": 2 / 3,
        "description": (
            "CRF drawn from [28, 36] with probability 2/3, otherwise from [18, 36]; "
            "shifts capacity toward the compression levels where the model currently fails"
        ),
    },
}

X264_PRESET = "medium"
SEQ_FRAMES = 24


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def rng_for(*parts: object) -> np.random.Generator:
    digest = hashlib.sha256("/".join(str(p) for p in parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def probe(path: str) -> dict:
    out = run([
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration,r_frame_rate,nb_frames",
        "-show_entries", "format=duration", "-of", "json", path,
    ]).stdout
    d = json.loads(out)
    s = d["streams"][0]
    num, _, den = s["r_frame_rate"].partition("/")
    fps = float(num) / float(den or 1)
    dur = float(s.get("duration") or d.get("format", {}).get("duration") or 0.0)
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": fps, "duration": dur}


def sample_crf(dist: str, gen: np.random.Generator) -> int:
    spec = CRF_DISTRIBUTIONS[dist]
    if dist == "poor" and gen.random() < spec["heavy_prob"]:
        lo, hi = spec["heavy_range"]
    else:
        lo, hi = spec["range"]
    return int(gen.integers(lo, hi + 1))


def extract_hr(src: str, start: float, out_dir: str, fps: int) -> int:
    """Decode one sequence to 2560x1440 masters.

    `-ss` before `-i` seeks on keyframes, which is fast and exact enough here
    because the master is whatever the decoder yields; the LR stream is derived
    from these same frames, so the pair is aligned by construction.
    """
    os.makedirs(out_dir, exist_ok=True)
    run([
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", src,
        "-frames:v", str(SEQ_FRAMES), "-fps_mode", "passthrough",
        "-vf", f"scale={HR_W}:{HR_H}:flags=area,format=rgb24",
        os.path.join(out_dir, "hr_%04d.png"),
    ])
    return len([n for n in os.listdir(out_dir) if n.startswith("hr_")])


def degrade_sequence(hr_dir: str, lr_dir: str, crf: int, structure: str, fps: int) -> dict:
    """HR masters -> 720p H.264 under the chosen GOP structure -> decoded LR frames."""
    os.makedirs(lr_dir, exist_ok=True)
    clip = os.path.join(lr_dir, "seq.mp4")
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(fps), "-start_number", "1",
        "-i", os.path.join(hr_dir, "hr_%04d.png"),
        "-vf", f"scale={LR_W}:{LR_H}:flags=bicubic,format=yuv420p",
        *ENCODE_COLOUR,
        "-c:v", "libx264", "-preset", X264_PRESET, "-crf", str(crf),
        "-x264-params", STRUCTURES[structure]["x264params"],
        "-threads", "1", clip,
    ]
    run(cmd)
    types_out = run([
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=pict_type", "-of", "csv=p=0", clip,
    ]).stdout
    counts: dict[str, int] = {}
    for tok in types_out.split():
        t = tok.strip().strip(",")
        if t:
            counts[t] = counts.get(t, 0) + 1
    run([
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-i", clip, "-vf", DECODE_FILTER, "-fps_mode", "passthrough",
        os.path.join(lr_dir, "lr_%04d.png"),
    ])
    n_lr = len([n for n in os.listdir(lr_dir) if n.startswith("lr_")])
    size = os.path.getsize(clip)
    os.remove(clip)
    return {"frameTypes": counts, "lrFrames": n_lr, "bytes": size, "ffmpegCommand": cmd}


def harvest_patches(
    hr_dir: str, lr_dir: str, sink: dict[str, list[torch.Tensor]],
    hr_patch: int, per_frame: int, gen: np.random.Generator, split: str,
) -> int:
    """Cut aligned (LR, HR) patch pairs out of a decoded sequence.

    The compression already happened at full frame resolution, which is the
    whole point: a patch carries the artifacts a real 720p decode produced in
    its neighbourhood, including blocking that straddles macroblock edges. The
    LR patch is taken at (x, y) and the HR patch at exactly (2x, 2y), so the
    pair keeps the phase relationship the model must learn.
    """
    lr_patch = hr_patch // 2
    hr_names = sorted(n for n in os.listdir(hr_dir) if n.endswith(".png"))
    lr_names = sorted(n for n in os.listdir(lr_dir) if n.endswith(".png"))
    written = 0
    for hn, ln in zip(hr_names, lr_names):
        hr = read_u8(os.path.join(hr_dir, hn))
        lr = read_u8(os.path.join(lr_dir, ln))
        _, lh, lw = lr.shape
        if lh < lr_patch or lw < lr_patch:
            continue
        for _ in range(per_frame):
            # Even LR offsets only: an odd offset would pair an LR patch with
            # an HR patch at a half-pixel phase the 2x geometry cannot express.
            x = int(gen.integers(0, (lw - lr_patch) // 2 + 1)) * 2
            y = int(gen.integers(0, (lh - lr_patch) // 2 + 1)) * 2
            sink[f"{split}_lr"].append(lr[:, y : y + lr_patch, x : x + lr_patch].clone())
            sink[f"{split}_hr"].append(
                hr[:, 2 * y : 2 * y + hr_patch, 2 * x : 2 * x + hr_patch].clone()
            )
            written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Build GOP-aware or all-I video training pairs.")
    ap.add_argument("--manifest", default="data/captured-train/manifest.json")
    ap.add_argument("--sources", default="data/captured-train/sources")
    ap.add_argument("--out", required=True, help="output pair directory")
    ap.add_argument("--structure", required=True, choices=sorted(STRUCTURES))
    ap.add_argument("--crf-dist", required=True, choices=sorted(CRF_DISTRIBUTIONS))
    ap.add_argument("--sequences-per-clip", type=int, default=6)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--val-clips", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--patch", type=int, default=128, help="HR patch size; LR is half")
    ap.add_argument("--patches-per-frame", type=int, default=3)
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    clips = sorted(manifest["clips"], key=lambda c: c["id"])

    # The split is by SOURCE VIDEO, decided before a single frame is written, so
    # no validation frame can come from a video that also appears in training.
    # Two sequences from one clip share a scene, a camera and a compression
    # history; splitting by sequence would leak all three.
    order = rng_for("split", args.seed).permutation(len(clips))
    val_ids = {clips[i]["id"] for i in order[: args.val_clips]}

    os.makedirs(args.out, exist_ok=True)
    index: list[dict] = []
    evidence: list[dict] = []
    patches: dict[str, list[torch.Tensor]] = {
        "train_lr": [], "train_hr": [], "val_lr": [], "val_hr": []
    }

    for clip in clips:
        cid = clip["id"]
        src = os.path.join(args.sources, clip["source_file"])
        if not os.path.exists(src):
            print(f"  MISSING {cid}", file=sys.stderr)
            continue
        info = probe(src)
        split = "val" if cid in val_ids else "train"
        gen = rng_for("seq", args.seed, cid)

        # Sequences are spread across the clip rather than taken from its start,
        # so a corpus is not dominated by title cards and lead-in shots.
        usable = max(0.0, info["duration"] - (SEQ_FRAMES / args.fps) - 1.0)
        if usable <= 0:
            print(f"  TOO SHORT {cid} ({info['duration']:.1f}s)", file=sys.stderr)
            continue
        starts = sorted(float(x) for x in gen.uniform(0.5, 0.5 + usable, args.sequences_per_clip))

        for si, start in enumerate(starts):
            tag = f"{cid}__{si:02d}"
            hr_dir = os.path.join(args.out, split, tag, "hr")
            lr_dir = os.path.join(args.out, split, tag, "lr")
            n_hr = extract_hr(src, start, hr_dir, args.fps)
            if n_hr < SEQ_FRAMES:
                # A short tail cannot form a full GOP, and a partial sequence
                # would quietly change the compression context. Drop it loudly.
                print(f"  SKIP {tag}: {n_hr}/{SEQ_FRAMES} frames", file=sys.stderr)
                shutil.rmtree(os.path.dirname(hr_dir), ignore_errors=True)
                continue
            crf = sample_crf(args.crf_dist, gen)
            info_d = degrade_sequence(hr_dir, lr_dir, crf, args.structure, args.fps)
            if info_d["lrFrames"] != n_hr:
                print(f"  SKIP {tag}: {info_d['lrFrames']} lr vs {n_hr} hr", file=sys.stderr)
                shutil.rmtree(os.path.dirname(hr_dir), ignore_errors=True)
                continue
            # Full frames are the *reason* the compression is realistic, but
            # they are not what training consumes and 24 GB of 1440p PNGs per
            # condition is not worth keeping. Cut patch pairs out of the
            # decoded sequence, then drop the frames.
            n_patches = harvest_patches(
                hr_dir, lr_dir, patches, args.patch, args.patches_per_frame,
                rng_for("patch", args.seed, tag), split,
            )
            shutil.rmtree(os.path.dirname(hr_dir), ignore_errors=True)
            index.append({
                "tag": tag, "clip": cid, "split": split, "start": start,
                "crf": crf, "frames": n_hr, "patches": n_patches,
            })
            evidence.append({"tag": tag, "crf": crf, **{k: info_d[k] for k in ("frameTypes", "bytes")}})
            print(f"  {tag:<46} crf{crf:<3} {info_d['frameTypes']} {info_d['bytes']/1024:7.1f} kB "
                  f"{n_patches} patches", file=sys.stderr)

    train_clips = sorted({r["clip"] for r in index if r["split"] == "train"})
    val_clips = sorted({r["clip"] for r in index if r["split"] == "val"})
    assert not (set(train_clips) & set(val_clips)), "source-level split leaked"

    meta = {
        "schema": "aethervsr.video-pairs/1",
        "structure": args.structure,
        "structureSpec": STRUCTURES[args.structure],
        "crfDistribution": args.crf_dist,
        "crfSpec": CRF_DISTRIBUTIONS[args.crf_dist],
        "preset": X264_PRESET,
        "seed": args.seed,
        "fps": args.fps,
        "sequenceFrames": SEQ_FRAMES,
        "hrSize": [HR_W, HR_H], "lrSize": [LR_W, LR_H],
        "sourceManifest": args.manifest,
        "splitUnit": "source video",
        "trainClips": train_clips, "valClips": val_clips,
        "sequences": len(index),
        "crfHistogram": {str(c): sum(1 for r in index if r["crf"] == c)
                         for c in sorted({r["crf"] for r in index})},
        "index": index,
        "encodeEvidence": evidence,
        "patchHr": args.patch, "patchLr": args.patch // 2,
        "patchesPerFrame": args.patches_per_frame,
        "trainPatches": len(patches["train_hr"]), "valPatches": len(patches["val_hr"]),
    }
    with open(os.path.join(args.out, "pairs.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    # uint8 keeps the four conditions to a few hundred MB each and costs
    # nothing: the frames came off a decoder as 8-bit and converting to float
    # here would only store rounding noise at four times the size.
    for split in ("train", "val"):
        if not patches[f"{split}_hr"]:
            continue
        torch.save(
            {"lr": torch.stack(patches[f"{split}_lr"]), "hr": torch.stack(patches[f"{split}_hr"])},
            os.path.join(args.out, f"{split}.pt"),
        )
    print(f"\n{len(index)} sequences ({len(train_clips)} train / {len(val_clips)} val clips), "
          f"structure={args.structure} crf={args.crf_dist}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
