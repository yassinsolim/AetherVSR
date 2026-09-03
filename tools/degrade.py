#!/usr/bin/env python3
"""Reproducible web-video degradation pipeline for AetherVSR (Milestone 4.5).

`tools/train.py` trains the C16D2 upscaler on one clean inverse problem: HR ->
exact 2x box downsample -> LR (`aethersr.box_downsample2`). Real browser video
never arrives that way. It goes master -> resize -> 4:2:0 chroma subsampling
-> H.264 quantisation (ringing, blocking, deblocking-filter softening) ->
decode. If the model's quality gain is real, it must survive being trained
and/or evaluated against that actual pipeline, not just the box downsample it
was tuned on. This module is that pipeline, exposed both as a library
(`degrade_tensor`, called per training batch or per evaluation frame) and as a
`--selftest` CLI that proves the profiles behave as documented.

Profiles
--------
  box          exact 2x box downsample (avg_pool2d k2 s2). Re-exports
               `aethersr.box_downsample2` directly rather than reimplementing
               it, so "numerically identical" is true by construction, not by
               coincidence; `--selftest` still asserts it.
  bicubic      clean 2x antialiased bicubic downscale (no compression).
  lanczos      clean 2x Lanczos downscale (no compression).
  h264_high    resize (bicubic) + 4:2:0 + H.264 crf=18  (near-transparent)
  h264_typical resize (bicubic) + 4:2:0 + H.264 crf=26  (typical web quality)
  h264_poor    resize (bicubic) + 4:2:0 + H.264 crf=34  (poor connection)
  realistic    randomised training distribution: a random resize kernel
               (box/bicubic/lanczos, uniform), an optional small pre-resize
               blur, and a random H.264 crf in [18, 36]. Seeded, reproducible.

How the three H.264 CRF tiers were chosen
------------------------------------------
Not picked as round numbers. Measured: three 1280x720 corpus crops
(data/corpus/00001_ef00de100b.jpg native, plus center-crops of 00000 and
00002) were each encoded as a single H.264 I-frame at crf in
{18,22,24,26,28,30,32,34,36,38} with the exact flags `_ffmpeg_encode` below
uses (libx264, preset medium, 4:2:0, bt709 limited range, threads=1), decoded
back, and PSNR'd against the uncompressed source
(`/tmp/degrade_proto/proto*.py`, run 2026-09-03, ffmpeg 9.0.1):

    crf   mean PSNR   mean bytes/frame
    18    39.28 dB     88315
    22    36.30 dB     56290
    24    34.88 dB     43891
    26    33.60 dB     33978   <- h264_typical
    28    32.40 dB     26334
    30    31.29 dB     20614
    32    30.21 dB     16084
    34    29.14 dB     12496   <- h264_poor
    36    28.16 dB      9690
    38    27.21 dB      7536
   (18 is also h264_high; the curve above 18 flattens quickly toward lossless)

The three tiers (18 / 26 / 34) sit at roughly even 8-CRF spacing across this
measured curve and give an even ~2.6x-2.7x byte-size drop and ~4-5 dB PSNR
drop tier to tier: "high" is the near-transparent end (>39 dB mean, the point
where visual differences from source all but disappear), "typical" is the
curve's midpoint (mid-30s dB, matching the "good but visibly re-encoded"
quality most delivered web video sits at), "poor" is where blocking becomes
unmistakable (<30 dB mean, one image in the three-image sample already below
27 dB). `realistic` samples the same axis (crf in [18, 36]) uniformly rather
than only ever hitting these three points, because real delivered video does
not clump into three discrete tiers.

Bitrate is not reported in kbps: a single encoded I-frame with no B/P-frame
temporal prediction is not comparable to a real GOP's average bitrate (it
systematically overstates it), so only relative byte size and PSNR, which are
valid single-frame proxies for encoder quality, are used to justify the CRF
choice. Byte size and PSNR are exactly what the selftest also reports on the
actual corpus, per profile.

Chroma and range, made explicit rather than left to ffmpeg defaults
---------------------------------------------------------------------
Input/output tensors are full-range RGB in [0,1] (what the model and
`box_downsample2` use). The H.264 profiles convert explicitly to 4:2:0
(`format=yuv420p`, the universal web-video subsampling) and to studio/limited
range (`-color_range tv`, 16-235 luma / 16-240 chroma), tagged `bt709`
(matrix, primaries and transfer), because that is what browsers actually tag
delivered HD video with. Decoding converts back to full-range RGB
(`scale=in_range=tv:out_range=full`) because that is what a browser's video
decoder + compositor hands back to the page (canvas/WebGL readback of a
<video> element is full-range RGB). Every flag is passed explicitly on both
the encode and decode command lines; nothing here relies on an ffmpeg
default.

Tiling patches into a mosaic frame before encoding
---------------------------------------------------
A single 128x128 HR patch downsamples to a 64x64 LR patch, which is a tiny
frame for a video encoder: too few macroblocks for x264's rate control and
partition search to behave anything like it does on a real 720p frame, and a
single-frame encode has no GOP/lookahead context at all. `degrade_tensor`
handles this by tiling the whole (N, C, h, w) batch into one
(C, rows*h, cols*w) mosaic "frame" (`_tile_grid`, roughly square:
rows=ceil(N/cols), cols=ceil(sqrt(N))) and running exactly one real
encode+decode round trip on that mosaic, then slicing the N patches back out
(`_untile_grid`). This is a real libx264 encode of a real (if synthetic)
image, not a statistical approximation of compression - CRF, quantisation,
in-loop deblocking and 4:2:0 subsampling all really happen. The only thing
that's approximated is *context*: a patch's outer edge, in a mosaic, borders
an unrelated patch (or, if N does not divide the grid evenly, a repeated copy
of the last patch used as filler - not black, since an abrupt black/real edge
would inject spurious high-frequency content into the neighbouring real
patch's compression) instead of true spatially-continuous video content. That
is a documented approximation of *what borders the patch*, made explicit here
and in the profile whose "kind" is "h264" or "realistic"; the encode itself is
never faked.

This also means the same code path serves both training and evaluation
without a mode flag: called with a batch of small training patches (N > 1),
it produces the mosaic above; called with N=1 (one full evaluation frame,
which is what `tools/videobench.py` / whole-image evaluation passes) the grid
is trivially 1x1 - the mosaic degenerates to the frame itself, with no
synthetic neighbour at all, i.e. the *true*, fully faithful single-frame
encode `evaluate.py`/`videobench.py` needs. One CRF and one resize kernel are
chosen per call (per training batch, or per evaluation frame), never per
patch, because a real encoded frame only has one quantiser and there would be
no way to give two patches sharing one mosaic two different CRFs without
lying about how H.264 works.

Throughput: because the whole batch is one subprocess round trip rather than
one per patch, this is fast enough to run on the training loop, not just
offline evaluation - `--selftest` measures and prints live patches/sec at a
few batch sizes; see the report accompanying this file for the numbers from
one measured run.

ffmpeg: version and every flag are also queried/printed live by `--selftest`
(`ffmpeg -version`; commands built by `_ffmpeg_encode`/`_ffmpeg_decode`
below), so this comment cannot silently go stale relative to what actually
runs.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import math
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from aethersr import box_downsample2

FFMPEG = "/opt/homebrew/bin/ffmpeg"

# Measured CRF tiers; see the module docstring for the sweep that justifies
# these exact numbers (not round-number guesses).
H264_HIGH_CRF = 18
H264_TYPICAL_CRF = 26
H264_POOR_CRF = 34
REALISTIC_CRF_RANGE = (H264_HIGH_CRF, 36)  # spans measured high..just-past-poor
REALISTIC_KERNELS = ("box", "bicubic", "lanczos")
REALISTIC_BLUR_PROB = 0.5
# Small: at HR resolution this is a fraction of a pixel of softening, meant to
# model pre-resize camera/mezzanine softening, not to be a dominant
# degradation in its own right (the CRF is that).
REALISTIC_BLUR_SIGMA_RANGE = (0.15, 0.6)

X264_PRESET = "medium"

PROFILES: dict[str, dict] = {
    "box": {
        "kind": "box",
        "description": "exact 2x box downsample (avg_pool2d k2 s2); identical to aethersr.box_downsample2",
    },
    "bicubic": {
        "kind": "resize",
        "kernel": "bicubic",
        "description": "clean 2x antialiased bicubic downscale (torch F.interpolate, antialias=True); no compression",
    },
    "lanczos": {
        "kind": "resize",
        "kernel": "lanczos",
        "description": "clean 2x Lanczos downscale (PIL LANCZOS resample); no compression",
    },
    "h264_high": {
        "kind": "h264",
        "kernel": "bicubic",
        "crf": H264_HIGH_CRF,
        "description": (
            f"2x bicubic resize + 4:2:0 + H.264 crf={H264_HIGH_CRF}, preset {X264_PRESET} "
            "(measured mean PSNR 39.3 dB on a 3-image 720p sweep; near-transparent web quality)"
        ),
    },
    "h264_typical": {
        "kind": "h264",
        "kernel": "bicubic",
        "crf": H264_TYPICAL_CRF,
        "description": (
            f"2x bicubic resize + 4:2:0 + H.264 crf={H264_TYPICAL_CRF}, preset {X264_PRESET} "
            "(measured mean PSNR 33.6 dB; the sweep's midpoint, typical delivered web quality)"
        ),
    },
    "h264_poor": {
        "kind": "h264",
        "kernel": "bicubic",
        "crf": H264_POOR_CRF,
        "description": (
            f"2x bicubic resize + 4:2:0 + H.264 crf={H264_POOR_CRF}, preset {X264_PRESET} "
            "(measured mean PSNR 29.1 dB; visible blocking, poor-connection quality)"
        ),
    },
    "realistic": {
        "kind": "realistic",
        "kernels": REALISTIC_KERNELS,
        "crf_range": REALISTIC_CRF_RANGE,
        "blur_prob": REALISTIC_BLUR_PROB,
        "blur_sigma_range": REALISTIC_BLUR_SIGMA_RANGE,
        "description": (
            f"randomised training distribution: resize kernel uniform over {REALISTIC_KERNELS}, "
            f"pre-resize gaussian blur with probability {REALISTIC_BLUR_PROB} "
            f"(sigma in {REALISTIC_BLUR_SIGMA_RANGE}), H.264 crf uniform in {REALISTIC_CRF_RANGE} "
            "(720p web range spanning the measured high..poor tiers), preset "
            f"{X264_PRESET}, seeded and reproducible"
        ),
    },
}


def describe(profile: str) -> str:
    if profile not in PROFILES:
        raise SystemExit(f"unknown degradation profile {profile!r}; choices: {sorted(PROFILES)}")
    return PROFILES[profile]["description"]


def ffmpeg_version() -> str:
    out = subprocess.run(
        [FFMPEG, "-version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, text=True
    ).stdout
    return out.splitlines()[0]


# ---------------------------------------------------------------------------
# Resize kernels
# ---------------------------------------------------------------------------


def _bicubic_downsample2(hr: torch.Tensor) -> torch.Tensor:
    return F.interpolate(hr, scale_factor=0.5, mode="bicubic", align_corners=False, antialias=True).clamp(0.0, 1.0)


def _lanczos_downsample2(hr: torch.Tensor) -> torch.Tensor:
    """PIL LANCZOS, looped per item: torch has no native Lanczos kernel."""
    n, c, h, w = hr.shape
    if c != 3:
        raise SystemExit(f"lanczos resize expects 3-channel RGB, got {c} channels")
    u8 = (hr.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    out = torch.empty((n, c, h // 2, w // 2), dtype=torch.float32)
    for i in range(n):
        resized = Image.fromarray(u8[i], mode="RGB").resize((w // 2, h // 2), resample=Image.LANCZOS)
        out[i] = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return out


def _resize_downsample2(hr: torch.Tensor, kernel: str) -> torch.Tensor:
    if kernel == "box":
        return box_downsample2(hr)
    if kernel == "bicubic":
        return _bicubic_downsample2(hr)
    if kernel == "lanczos":
        return _lanczos_downsample2(hr)
    raise SystemExit(f"unknown resize kernel {kernel!r}")


def _gaussian_blur(hr: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable gaussian blur, reflect-padded. Pure torch (no torchvision/scipy)."""
    if sigma <= 0.0:
        return hr
    radius = max(1, int(math.ceil(sigma * 3.0)))
    # Build the kernel on the input's own device: a CPU kernel against an MPS
    # tensor raises, and this path is reached from training on the GPU.
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32, device=hr.device)
    k1d = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
    k1d = k1d / k1d.sum()
    c = hr.shape[1]
    kx = k1d.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = k1d.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    out = F.conv2d(F.pad(hr, (radius, radius, 0, 0), mode="reflect"), kx, groups=c)
    out = F.conv2d(F.pad(out, (0, 0, radius, radius), mode="reflect"), ky, groups=c)
    return out


# ---------------------------------------------------------------------------
# Mosaic tiling (see module docstring: "Tiling patches into a mosaic frame")
# ---------------------------------------------------------------------------


def _grid_layout(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def _tile_grid(lr: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    n, c, h, w = lr.shape
    rows, cols = _grid_layout(n)
    cells = rows * cols
    padded = lr
    if cells > n:
        filler = lr[-1:].expand(cells - n, c, h, w)
        padded = torch.cat([lr, filler], dim=0)
    grid = padded.view(rows, cols, c, h, w).permute(2, 0, 3, 1, 4).reshape(c, rows * h, cols * w)
    return grid, rows, cols, n


def _untile_grid(frame: torch.Tensor, rows: int, cols: int, h: int, w: int, n: int) -> torch.Tensor:
    c = frame.shape[0]
    grid = frame.view(c, rows, h, cols, w).permute(1, 3, 0, 2, 4).reshape(rows * cols, c, h, w)
    return grid[:n]


# ---------------------------------------------------------------------------
# H.264 round trip
# ---------------------------------------------------------------------------


def _ffmpeg_encode(frame_u8: np.ndarray, crf: int) -> bytes:
    """rgb24 raw HWC uint8 -> one H.264 IDR frame, 4:2:0, bt709 limited range."""
    h, w, c = frame_u8.shape
    if c != 3:
        raise SystemExit("h264 round trip expects RGB24 input")
    proc = subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            "1",  # arbitrary: exactly one frame is encoded, no real frame rate exists
            "-i",
            "pipe:0",
            "-vf",
            "scale=out_range=tv,format=yuv420p",  # explicit 4:2:0 + full->limited range
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-c:v",
            "libx264",
            "-preset",
            X264_PRESET,
            "-crf",
            str(crf),
            "-threads",
            "1",
            "-x264-params",
            "threads=1",  # both flags pinned: deterministic bitstream regardless of host core count
            "-frames:v",
            "1",
            "-f",
            "h264",
            "pipe:1",
        ],
        input=frame_u8.tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout


def _ffmpeg_decode(bitstream: bytes, h: int, w: int) -> np.ndarray:
    """One H.264 IDR frame -> rgb24 raw HWC uint8, converted back to full range."""
    proc = subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-vf",
            "scale=in_range=tv:out_range=full,format=rgb24",  # explicit limited->full range
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        input=bitstream,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(h, w, 3)


def _tensor_to_u8_hwc(frame: torch.Tensor) -> np.ndarray:
    return (frame.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def _u8_hwc_to_tensor(u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(u8.astype(np.float32) / 255.0).permute(2, 0, 1)


def _h264_round_trip(hr: torch.Tensor, kernel: str, crf: int) -> torch.Tensor:
    lr = _resize_downsample2(hr, kernel)
    grid, rows, cols, n = _tile_grid(lr)

    # 4:2:0 subsamples chroma by two in both axes, so an odd-sided frame is not
    # representable and libx264 refuses it outright. A single evaluation image
    # of 1274x1280 tiles 1x1 into a 637x640 mosaic and aborted the run.
    # Replicate-pad to even, encode, then crop the padding away.
    _, gh, gw = grid.shape
    pad_h, pad_w = gh % 2, gw % 2
    if pad_h or pad_w:
        grid = F.pad(grid.unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate").squeeze(0)

    frame_u8 = _tensor_to_u8_hwc(grid)
    bitstream = _ffmpeg_encode(frame_u8, crf)
    dec_u8 = _ffmpeg_decode(bitstream, frame_u8.shape[0], frame_u8.shape[1])
    dec = _u8_hwc_to_tensor(dec_u8)
    if pad_h or pad_w:
        dec = dec[:, :gh, :gw]
    return _untile_grid(dec, rows, cols, lr.shape[2], lr.shape[3], n)


# ---------------------------------------------------------------------------
# "realistic": seeded randomised training distribution
# ---------------------------------------------------------------------------


def _rng(profile: str, seed: int) -> np.random.Generator:
    """Deterministic, profile-namespaced RNG: same (profile, seed) -> same stream."""
    digest = hashlib.sha256(f"aethervsr/degrade/{profile}:{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _pick_realistic_params(seed: int) -> dict:
    rng = _rng("realistic", seed)
    kernel = rng.choice(REALISTIC_KERNELS)
    blur_sigma = 0.0
    if rng.random() < REALISTIC_BLUR_PROB:
        lo, hi = REALISTIC_BLUR_SIGMA_RANGE
        blur_sigma = float(rng.uniform(lo, hi))
    lo, hi = REALISTIC_CRF_RANGE
    crf = int(rng.integers(lo, hi + 1))
    return {"kernel": str(kernel), "blur_sigma": blur_sigma, "crf": crf}


def _realistic(hr: torch.Tensor, seed: int) -> torch.Tensor:
    params = _pick_realistic_params(seed)
    blurred = _gaussian_blur(hr, params["blur_sigma"]) if params["blur_sigma"] > 0.0 else hr
    return _h264_round_trip(blurred, params["kernel"], params["crf"])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def degrade_tensor(hr: torch.Tensor, profile: str, seed: int) -> torch.Tensor:
    """HR (float32 NCHW, [0,1], even H/W) -> half-resolution LR (float32 NCHW, [0,1]).

    `seed` controls only the "realistic" profile's random choices; every other
    profile is deterministic on its own and ignores it. Deterministic for a
    given (profile, seed, shape): re-running with the same three produces a
    bit-identical tensor (`--selftest` proves this for every profile).
    """
    if profile not in PROFILES:
        raise SystemExit(f"unknown degradation profile {profile!r}; choices: {sorted(PROFILES)}")
    if hr.ndim != 4 or hr.shape[1] != 3:
        raise SystemExit(f"expected NCHW RGB tensor, got shape {tuple(hr.shape)}")
    if hr.shape[2] % 2 or hr.shape[3] % 2:
        raise SystemExit(f"H and W must be even, got {tuple(hr.shape[2:])}")
    spec = PROFILES[profile]
    kind = spec["kind"]
    if kind == "box":
        return box_downsample2(hr)
    if kind == "resize":
        return _resize_downsample2(hr, spec["kernel"])
    if kind == "h264":
        return _h264_round_trip(hr, spec["kernel"], spec["crf"])
    if kind == "realistic":
        return _realistic(hr, seed)
    raise SystemExit(f"unhandled profile kind {kind!r}")


# ---------------------------------------------------------------------------
# --selftest
# ---------------------------------------------------------------------------


def _load_test_image(patch: int = 512) -> torch.Tensor:
    files = sorted(glob.glob("data/corpus/*.jpg"))
    if not files:
        raise SystemExit("no corpus images found under data/corpus; run from the repo root")
    im = Image.open(files[0]).convert("RGB")
    w, h = im.size
    side = min(patch, w - (w % 2), h - (h % 2))
    side -= side % 2
    left, top = (w - side) // 2, (h - side) // 2
    im = im.crop((left, top, left + side, top + side))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0), files[0]


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    if mse <= 0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _selftest() -> int:
    print(f"ffmpeg: {ffmpeg_version()}  (binary: {FFMPEG})")
    print()

    hr, src_path = _load_test_image()
    print(f"test image: {src_path}  HR shape {tuple(hr.shape)}")
    clean_lr = box_downsample2(hr)

    print()
    print("box vs aethersr.box_downsample2 numerical identity:")
    box_out = degrade_tensor(hr, "box", seed=0)
    identical = torch.equal(box_out, box_downsample2(hr))
    print(f"  torch.equal(degrade_tensor(hr,'box',0), box_downsample2(hr)) = {identical}")
    if not identical:
        raise SystemExit("box profile is not numerically identical to box_downsample2")

    print()
    print("per-profile: description, shape, PSNR vs clean box downsample, determinism")
    for name in sorted(PROFILES):
        out1 = degrade_tensor(hr, name, seed=1234)
        out2 = degrade_tensor(hr, name, seed=1234)
        deterministic = torch.equal(out1, out2)
        psnr_vs_clean = _psnr(out1.clamp(0, 1), clean_lr)
        print(f"- {name}")
        print(f"    params: {describe(name)}")
        print(f"    output shape: {tuple(out1.shape)}")
        print(f"    PSNR vs clean box downsample: {psnr_vs_clean:.2f} dB")
        print(f"    deterministic (same seed twice -> identical tensor): {deterministic}")
        if not deterministic:
            raise SystemExit(f"profile {name!r} is not deterministic for a fixed seed")
        if name == "realistic":
            params = _pick_realistic_params(1234)
            print(f"    realistic draw for seed=1234: {params}")

    print()
    print("training-time throughput (batched mosaic h264 round trip, profile=h264_typical):")
    for batch_n in (8, 32, 64, 128):
        patch = 128
        big, _ = _load_test_image(patch=max(512, patch * 4))
        _, _, bh, bw = big.shape
        rng = np.random.default_rng(batch_n)
        tiles = []
        for _ in range(batch_n):
            y = int(rng.integers(0, bh - patch + 1))
            x = int(rng.integers(0, bw - patch + 1))
            tiles.append(big[:, :, y : y + patch, x : x + patch])
        batch = torch.cat(tiles, dim=0)
        reps = 8
        times = []
        for r in range(reps):
            t0 = time.time()
            degrade_tensor(batch, "h264_typical", seed=r)
            times.append(time.time() - t0)
        times = times[2:]  # drop warmup
        avg = sum(times) / len(times)
        print(f"  batch={batch_n:4d}  avg={avg * 1000:6.1f} ms  patches/sec={batch_n / avg:8.1f}")

    print()
    print("OK: all profiles present, box verified identical, all deterministic.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true", help="run the full profile/PSNR/determinism/throughput report")
    ap.add_argument("--profile", choices=sorted(PROFILES), help="degrade a single --input image and write --output")
    ap.add_argument("--input", help="image file to degrade (with --profile)")
    ap.add_argument("--output", help="PNG path to write the degraded LR image to (with --profile)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the 'realistic' profile")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    if args.profile:
        if not args.input or not args.output:
            raise SystemExit("--profile requires both --input and --output")
        im = Image.open(args.input).convert("RGB")
        w, h = im.size
        if w % 2 or h % 2:
            im = im.crop((0, 0, w - (w % 2), h - (h % 2)))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        hr = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        lr = degrade_tensor(hr, args.profile, args.seed)
        out_u8 = (lr[0].clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).permute(1, 2, 0).numpy()
        Image.fromarray(out_u8, mode="RGB").save(args.output)
        print(f"{args.profile}: {tuple(hr.shape)} -> {tuple(lr.shape)}, wrote {args.output}")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
