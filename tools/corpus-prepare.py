#!/usr/bin/env python3
"""Prepare captured-video training pairs by streaming, never by downloading.

The v2 corpus is 186 clips totalling ~52 GB, of which this milestone uses about
three seconds each. Downloading all of it to keep 0.2% would take an hour of
bandwidth and 52 GB of disk for no scientific gain.

ffmpeg can seek into these files over HTTP and decode a short run of frames
directly, provided it identifies itself — Wikimedia returns 429 to ffmpeg's
default User-Agent. Pulling three frames from a 158 MB source takes about two
seconds, so the whole corpus is a bandwidth rounding error.

What that costs is the obvious provenance story. A SHA-256 of the whole file is
not available without transferring the whole file, so identity is pinned three
ways instead, none of which requires it:

  commonsSha1   the authoritative digest from the Commons API, computed by
                Wikimedia over the stored bytes
  prefixSha256  our own hash of a fixed byte prefix, which detects silent
                replacement of the file behind a stable URL
  byteLength    exact size, also from the API

Together those pin the file as firmly as a full hash for our purpose, which is
detecting that the thing we trained on changed. `tools/fetch-captured.py`
remains the full-download path and is still used for the small benchmark
corpora, where the media is small enough that a real SHA-256 is cheap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

FFMPEG = os.environ.get("AETHER_FFMPEG", "/opt/homebrew/bin/ffmpeg")
API = "https://commons.wikimedia.org/w/api.php"
UA = "AetherVSR/0.1 (https://github.com/yassinsolim/AetherVSR) corpus-prep"
PREFIX_BYTES = 2 << 20  # 2 MiB is plenty to catch a re-encode or a swap

HR_W, HR_H = 2560, 1440
LR_W, LR_H = 1280, 720
GOP_PARAMS = "keyint=48:min-keyint=24:scenecut=40:bframes=3:ref=3"
X264_PRESET = "medium"
ENCODE_COLOUR = [
    "-pix_fmt", "yuv420p", "-color_range", "tv",
    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
]
DECODE_FILTER = "scale=in_range=tv:out_range=pc,format=rgb24"


def api_info(titles: list[str], pause: float) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(titles), 20):
        chunk = [t for t in titles[i:i + 20] if t]
        if not chunk:
            continue
        q = {"action": "query", "format": "json", "prop": "imageinfo",
             "iiprop": "sha1|size|url|timestamp|mediatype", "titles": "|".join(chunk)}
        req = urllib.request.Request(API + "?" + urllib.parse.urlencode(q), headers={"User-Agent": UA})
        for attempt in range(6):
            try:
                data = json.load(urllib.request.urlopen(req, timeout=60))
                break
            except Exception:
                time.sleep(3 * (attempt + 1))
        else:
            continue
        for page in data.get("query", {}).get("pages", {}).values():
            ii = (page.get("imageinfo") or [{}])[0]
            if ii:
                out[page["title"]] = ii
        time.sleep(pause)
    return out


def prefix_hash(url: str, pause: float) -> tuple[str | None, int | None]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": f"bytes=0-{PREFIX_BYTES - 1}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                blob = resp.read()
            return hashlib.sha256(blob).hexdigest(), len(blob)
        except Exception:
            time.sleep(4 * (attempt + 1))
    return None, None


def run(cmd: list[str], timeout: int = 600) -> bool:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def extract_hr(url: str, start: float, out_dir: str, frames: int, pause: float = 1.5) -> int:
    """Decode one sequence straight from the URL into 2560x1440 masters."""
    os.makedirs(out_dir, exist_ok=True)
    vf = (f"scale={HR_W}:{HR_H}:force_original_aspect_ratio=increase:flags=lanczos,"
          f"crop={HR_W}:{HR_H},format=rgb24")
    pattern = os.path.join(out_dir, "hr_%04d.png")
    base = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-user_agent", UA]
    tail = ["-frames:v", str(frames), "-fps_mode", "passthrough", "-vf", vf, pattern]

    # Ogg/Theora will not seek on input over HTTP: ffmpeg exits 0 and writes
    # nothing, so the failure is silent rather than loud. Pick the strategy from
    # the container instead of discovering it per clip, which also avoids
    # spending a request to learn what the extension already says - and those
    # wasted requests were themselves tripping Wikimedia's rate limiter.
    ogg = url.lower().endswith((".ogv", ".ogg", ".oga"))
    orders = [["-i", url, "-ss", f"{start:.3f}"]] if ogg else [
        ["-ss", f"{start:.3f}", "-i", url], ["-i", url, "-ss", f"{start:.3f}"]
    ]

    for attempt, pre in enumerate(orders):
        for retry in range(3):
            for f in os.listdir(out_dir):
                os.remove(os.path.join(out_dir, f))
            run(base + pre + tail, timeout=900)
            n = len([x for x in os.listdir(out_dir) if x.startswith("hr_")])
            if n >= frames:
                return n
            # A 429 is transient and the only way through it is to wait; a
            # genuinely unusable file fails the same way three times and costs
            # ten seconds to establish.
            time.sleep(pause * (2 ** retry))
    return len([x for x in os.listdir(out_dir) if x.startswith("hr_")])

def degrade(hr_dir: str, lr_dir: str, crf: int, fps: int) -> dict | None:
    os.makedirs(lr_dir, exist_ok=True)
    clip = os.path.join(lr_dir, "seq.mp4")
    ok = run([
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(fps), "-start_number", "1",
        "-i", os.path.join(hr_dir, "hr_%04d.png"),
        "-vf", f"scale={LR_W}:{LR_H}:flags=bicubic,format=yuv420p",
        *ENCODE_COLOUR,
        "-c:v", "libx264", "-preset", X264_PRESET, "-crf", str(crf),
        "-x264-params", GOP_PARAMS, "-threads", "1", clip,
    ])
    if not ok:
        return None
    ok = run([
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", clip,
        "-vf", DECODE_FILTER, "-fps_mode", "passthrough",
        os.path.join(lr_dir, "lr_%04d.png"),
    ])
    if not ok:
        return None
    n = len([x for x in os.listdir(lr_dir) if x.startswith("lr_")])
    size = os.path.getsize(clip)
    os.remove(clip)
    return {"lrFrames": n, "bytes": size, "crf": crf}


def main() -> int:
    ap = argparse.ArgumentParser(description="Stream-prepare a captured corpus.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="pair output directory")
    ap.add_argument("--sequences", type=int, default=3)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--patches-per-frame", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--pause", type=float, default=1.5, help="seconds between network calls")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np
    import torch
    from video_degrade_lib import harvest_patches, rng_for  # noqa: F401

    with open(args.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    clips = manifest["clips"][: args.limit] if args.limit else manifest["clips"]

    print(f"  pinning {len(clips)} clips via the Commons API...", file=sys.stderr)
    info = api_info([c.get("title") for c in clips], args.pause)

    os.makedirs(args.out, exist_ok=True)
    patches: dict[str, list] = {"lr": [], "hr": []}
    index, evidence, failures = [], [], []

    for n, clip in enumerate(clips, 1):
        cid = clip["id"]
        url = clip["source_url"]
        ii = info.get(clip.get("title") or "", {})
        ph, plen = prefix_hash(url, args.pause)
        clip["commonsSha1"] = ii.get("sha1")
        clip["byteLength"] = ii.get("size")
        clip["prefixSha256"] = ph
        clip["prefixBytes"] = plen
        clip["pinnedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        gen = rng_for("seq", args.seed, cid)
        dur = float(clip.get("duration") or 0) or 60.0
        usable = max(1.0, dur - (args.frames / args.fps) - 1.0)
        starts = sorted(float(x) for x in gen.uniform(0.5, min(0.5 + usable, dur - 2), args.sequences))

        got = 0
        for si, start in enumerate(starts):
            tag = f"{cid}__{si:02d}"
            hr_dir = os.path.join(args.out, "_work", tag, "hr")
            lr_dir = os.path.join(args.out, "_work", tag, "lr")
            n_hr = extract_hr(url, start, hr_dir, args.frames, args.pause)
            if n_hr < args.frames:
                continue
            crf = int(gen.integers(18, 37))
            d = degrade(hr_dir, lr_dir, crf, args.fps)
            if d is None or d["lrFrames"] != n_hr:
                continue
            npatch = harvest_patches(hr_dir, lr_dir, patches, args.patch,
                                     args.patches_per_frame,
                                     rng_for("patch", args.seed, tag))
            index.append({"tag": tag, "clip": cid, "category": clip["category"],
                          "creator": clip.get("creator"), "shootId": clip.get("shootId"),
                          "start": start, "crf": crf, "frames": n_hr, "patches": npatch})
            evidence.append({"tag": tag, "crf": crf, "bytes": d["bytes"]})
            got += 1
            import shutil
            shutil.rmtree(os.path.join(args.out, "_work", tag), ignore_errors=True)
        if got == 0:
            failures.append({"clip": cid, "url": url, "reason": "no sequence extracted"})
        print(f"  [{n:>3}/{len(clips)}] {cid[:44]:<44} {got}/{args.sequences} seq  "
              f"{len(patches['hr']):>6} patches", file=sys.stderr)
        time.sleep(args.pause)

    import shutil
    shutil.rmtree(os.path.join(args.out, "_work"), ignore_errors=True)

    if patches["hr"]:
        torch.save({"lr": torch.stack(patches["lr"]), "hr": torch.stack(patches["hr"])},
                   os.path.join(args.out, "all.pt"))
    meta = {
        "schema": "aethervsr.video-pairs/2",
        "sourceManifest": args.manifest,
        "role": manifest.get("role"),
        "acquisition": "streamed over HTTP; whole files never downloaded",
        "pinning": {"commonsSha1": "authoritative digest from the Commons API",
                    "prefixSha256": f"sha256 of the first {PREFIX_BYTES} bytes",
                    "byteLength": "exact size from the API"},
        "structure": "gop", "x264params": GOP_PARAMS, "preset": X264_PRESET,
        "crfDistribution": "uniform [18,36] per sequence",
        "hrSize": [HR_W, HR_H], "lrSize": [LR_W, LR_H],
        "patchHr": args.patch, "patchLr": args.patch // 2,
        "sequencesPerClip": args.sequences, "sequenceFrames": args.frames,
        "seed": args.seed,
        "clipsPrepared": len({r["clip"] for r in index}),
        "clipsFailed": failures,
        "sequences": len(index), "patches": len(patches["hr"]),
        "index": index, "encodeEvidence": evidence,
    }
    with open(os.path.join(args.out, "pairs.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    # The manifest gains its pins; identity now travels with the corpus.
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    print(f"\n  {meta['clipsPrepared']}/{len(clips)} clips, {len(index)} sequences, "
          f"{len(patches['hr'])} patches, {len(failures)} failed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
