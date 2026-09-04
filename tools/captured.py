#!/usr/bin/env python3
"""Prepare and evaluate genuinely captured camera footage.

Milestone 4.5's video benchmark used procedurally generated content, and its
only positive result came from a near-Nyquist zone plate that is adversarial by
construction. Remove that one category and the model was neutral; remove
synthetic hard-edged text too and it was negative. So no video claim survived,
and none could, because the corpus contained no camera capture.

This module builds the same controlled pipeline over real footage:

    captured HR master
        ├──────────────────────────────→ HR reference frames
        ↓
    controlled downscale to 720p
        ↓
    controlled H.264/VP9/AV1 encode
        ↓
    decode
        ↓
    2x upscale (neural or conventional)
        ↓
    compare against the aligned HR reference

Alignment is the part that silently breaks video benchmarks, so every stage
pins colour explicitly and the preparation records evidence that frames line up
rather than asserting it. A one-frame or one-pixel offset would dominate PSNR
and could manufacture either a win or a loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

FFMPEG = os.environ.get("AETHER_FFMPEG", "/opt/homebrew/bin/ffmpeg")
FFPROBE = os.environ.get("AETHER_FFPROBE", "/opt/homebrew/bin/ffprobe")

# One reference geometry for everything: 2560x1440 master, 1280x720 degraded.
HR_W, HR_H = 2560, 1440
LR_W, LR_H = 1280, 720

# Explicit end to end. Browsers deliver BT.709 limited-range 4:2:0, and letting
# ffmpeg infer any part of that is how a benchmark acquires a silent bias.
COLOUR_ENCODE = [
    "-pix_fmt", "yuv420p",
    "-color_range", "tv",
    "-colorspace", "bt709",
    "-color_primaries", "bt709",
    "-color_trc", "bt709",
]
# Decoding back to RGB has to undo exactly that, or every metric inherits a
# range error that looks like a quality difference.
DECODE_FILTER = "scale=in_range=tv:out_range=pc,format=rgb24"

# Milestone 4.5's measured tiers, kept identical so results stay comparable:
# CRF 18 -> 39.28 dB, 26 -> 33.60 dB, 34 -> 29.14 dB on real 720p crops.
CONDITIONS: dict[str, dict] = {
    "h264_high": {"codec": "libx264", "crf": 18, "ext": "mp4", "extra": ["-preset", "medium"]},
    "h264_typical": {"codec": "libx264", "crf": 26, "ext": "mp4", "extra": ["-preset", "medium"]},
    "h264_poor": {"codec": "libx264", "crf": 34, "ext": "mp4", "extra": ["-preset", "medium"]},
    "vp9_typical": {"codec": "libvpx-vp9", "crf": 33, "ext": "mp4", "extra": ["-b:v", "0", "-row-mt", "1"]},
    "av1_typical": {"codec": "libsvtav1", "crf": 35, "ext": "mp4", "extra": ["-preset", "8"]},
}


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=True, **kw)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ffmpeg_version() -> str:
    return run([FFMPEG, "-hide_banner", "-version"]).stdout.splitlines()[0]


def probe(path: str) -> dict:
    out = run([
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,codec_name,pix_fmt,color_range,color_space,nb_read_packets",
        "-show_entries", "format=duration,size",
        "-count_packets", "-of", "json", path,
    ]).stdout
    return json.loads(out)


def extract_master_frames(source: str, out_dir: str, start: float, frames: int, fps: int) -> list[str]:
    """HR reference frames: deterministic crop-to-fit at exactly HR_W x HR_H.

    `increase` then centre-crop rather than `decrease` plus padding, so no black
    bars ever enter the reference. Padding would be scored as perfectly
    reconstructable flat area and would inflate every method's PSNR equally,
    which quietly compresses the differences the benchmark exists to measure.
    """
    os.makedirs(out_dir, exist_ok=True)
    vf = (
        f"scale={HR_W}:{HR_H}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={HR_W}:{HR_H},{DECODE_FILTER}"
    )
    run([
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-ss", f"{start}", "-i", source,
        "-frames:v", str(frames), "-r", str(fps),
        "-vf", vf, "-fps_mode", "cfr",
        os.path.join(out_dir, "frame_%04d.png"),
    ])
    return sorted(os.listdir(out_dir))


def encode_lr(master_dir: str, out_path: str, condition: str, fps: int) -> list[str]:
    """Downscale the HR reference to 720p and encode it, one documented command."""
    spec = CONDITIONS[condition]
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(fps),
        "-start_number", "1",
        "-i", os.path.join(master_dir, "frame_%04d.png"),
        # Area/box downscale: the same relationship the model was trained to
        # invert, and the one a browser applies when it fits a large frame to a
        # smaller surface.
        "-vf", f"scale={LR_W}:{LR_H}:flags=area,{'format=yuv420p'}",
        *COLOUR_ENCODE,
        "-c:v", spec["codec"], "-crf", str(spec["crf"]), *spec["extra"],
        "-g", "48", "-threads", "1",
        out_path,
    ]
    run(cmd)
    return cmd


def decode_lr(clip_path: str, out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    run([
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-i", clip_path, "-vf", DECODE_FILTER, "-fps_mode", "passthrough",
        os.path.join(out_dir, "frame_%04d.png"),
    ])
    return len(os.listdir(out_dir))


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare captured-footage benchmark clips.")
    ap.add_argument("--manifest", default="data/captured/manifest.json")
    ap.add_argument("--root", default="data/captured")
    ap.add_argument("--conditions", default="h264_high,h264_typical,h264_poor")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--only", default="", help="comma-separated clip ids")
    args = ap.parse_args()

    if not shutil.which(FFMPEG) and not os.path.exists(FFMPEG):
        raise SystemExit(f"ffmpeg not found at {FFMPEG}")

    with open(args.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)

    wanted = {c.strip() for c in args.only.split(",") if c.strip()}
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    prepared: list[dict] = []
    skipped: list[dict] = []

    for clip in manifest["clips"]:
        if wanted and clip["id"] not in wanted:
            continue
        source = os.path.join(args.root, "sources", clip["source_file"])
        if not os.path.exists(source):
            print(f"  missing source for {clip['id']}: {source}", file=sys.stderr)
            continue

        clip_root = os.path.join(args.root, "clips", clip["id"])
        master_dir = os.path.join(clip_root, "master")
        shutil.rmtree(clip_root, ignore_errors=True)
        extract_master_frames(source, master_dir, clip["start_seconds"], args.frames, args.fps)
        n_master = len(os.listdir(master_dir))

        # Fail this clip loudly instead of aborting the run, and never emit a
        # partial clip: a short source seeked past its end silently produced
        # zero frames, and the encode then failed several steps later with a
        # message that named the wrong problem.
        if n_master != args.frames:
            print(
                f"  SKIP {clip['id']}: extracted {n_master} of {args.frames} frames "
                f"(source {clip.get('source_duration_s', '?')}s, start {clip['start_seconds']}s)",
                file=sys.stderr,
            )
            shutil.rmtree(clip_root, ignore_errors=True)
            skipped.append({"id": clip["id"], "reason": f"extracted {n_master}/{args.frames} frames"})
            continue

        entry = {
            "id": clip["id"],
            "category": clip["category"],
            "source_file": clip["source_file"],
            "start_seconds": clip["start_seconds"],
            "frames": n_master,
            "fps": args.fps,
            "master_sha256": [sha256_file(os.path.join(master_dir, f)) for f in sorted(os.listdir(master_dir))],
            "conditions": {},
        }

        for condition in conditions:
            clip_path = os.path.join(clip_root, f"{condition}.{CONDITIONS[condition]['ext']}")
            cmd = encode_lr(master_dir, clip_path, condition, args.fps)
            decoded_dir = os.path.join(clip_root, "decoded", condition)
            n_decoded = decode_lr(clip_path, decoded_dir)
            info = probe(clip_path)
            stream = info["stream" if "stream" in info else "streams"][0] if "streams" in info else {}
            entry["conditions"][condition] = {
                "clip": os.path.relpath(clip_path, args.root),
                "decodedFrames": n_decoded,
                "framesMatchMaster": n_decoded == n_master,
                "bytes": int(info["format"]["size"]),
                "bitrateKbps": round(int(info["format"]["size"]) * 8 / float(info["format"]["duration"]) / 1000, 1),
                "pixFmt": stream.get("pix_fmt"),
                "colorRange": stream.get("color_range"),
                "colorSpace": stream.get("color_space"),
                "codec": stream.get("codec_name"),
                "ffmpegCommand": cmd,
            }
            if n_decoded != n_master:
                print(
                    f"  ALIGNMENT: {clip['id']}/{condition} decoded {n_decoded} against {n_master} master",
                    file=sys.stderr,
                )

        prepared.append(entry)
        print(f"  prepared {clip['id']:<22} {n_master} frames x {len(conditions)} conditions", file=sys.stderr)

    out = {
        "schema": "aethervsr.captured-prepared/1",
        "ffmpeg": ffmpeg_version(),
        "geometry": {"master": [HR_W, HR_H], "degraded": [LR_W, LR_H]},
        "colourContract": {
            "encode": COLOUR_ENCODE,
            "decodeFilter": DECODE_FILTER,
            "note": "BT.709 limited-range 4:2:0 on the wire, full-range RGB in memory. Nothing inferred.",
        },
        "downscaleFilter": "scale=area (box), matching the degradation the model inverts",
        "conditions": {k: CONDITIONS[k] for k in conditions},
        "clips": prepared,
        "skipped": skipped,
    }
    out_path = os.path.join(args.root, "prepared.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {out_path} ({len(prepared)} clips prepared, {len(skipped)} skipped)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
