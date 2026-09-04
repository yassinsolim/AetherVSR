#!/usr/bin/env python3
"""Milestone 4.5 Part K/L: a ground-truth video benchmark for AetherVSR.

Every prior AetherVSR evaluation (Milestones 1-4) has scored the network on
*still images*: a high-resolution photograph, downsampled once by an exact
box filter, upscaled, compared. That pipeline has no decoder in it. Real
playback does: capture -> encode -> transmit -> decode -> upscale, and the
codec stage throws away exactly the kind of high-frequency detail a neural
upscaler is supposed to reconstruct. A model that only ever saw clean
box-downsampled crops may be fitting compression-free degradation and could
fall apart on a genuine H.264/VP9/AV1 decode. This module is the harness that
answers that question.

Pipeline
--------

    2560x1440 master (generated, seeded RGB PNG)
           | exact 2x box mean; deterministic nearest-even uint8 rounding
           | for PNG storage (same block-mean filter AetherSR is trained on)
    1280x720 pristine LR  ---------------------------------------------+
           | RGB/full -> BT.709 limited yuv420p -> H.264 (3 CRF tiers) |
           | / VP9 / AV1 encode -> BT.709 limited -> RGB/full decode   |
    1280x720 codec-degraded LR                                         |
           | upscaler under test (baselines here; the neural model      |
           |  is scored separately by the same reference frames)        |
    2560x1440 reconstruction                                            |
           | PSNR / SSIM / VMAF, all via ffmpeg's own filters            |
    scores, compared against the untouched 2560x1440 master  <----------+

The master frames are *generated*, not sourced from an existing compressed
file, for the same reason `src/bench/quality.ts` generates its still-image
reference: comparing against a reference that was itself produced by an
unknown resize/compression chain conflates "the codec's artefacts" with "the
reference's artefacts", and the two are then impossible to separate. Content
is synthetic and this is stated plainly in the manifest - it buys an exact,
license-clean, fully reproducible ground truth at the cost of not being real
camera footage. It is *not* smooth/easy synthetic content: every category
below carries genuine high-frequency detail (fractal noise down to per-pixel
grain, near-Nyquist gratings, hard geometric edges, single-pixel UI
gridlines) specifically because smooth gradients are trivially invertible and
would not test anything.

Five content categories are generated, each stressing a different failure
mode:

  natural   - camera-like: layered fractal ("1/f") noise with a lighting
              vignette and independent per-pixel sensor grain, panned and
              zoomed gently by a sub-pixel camera transform.
  texture   - high-frequency: a zone-plate chirp plus a near-Nyquist
              sinusoidal grating in a central resolution-chart region. The
              surrounding field is exactly flat so storage remains practical;
              the chart region itself is deliberately difficult.
  motion    - the same style of cluttered background as `natural` but with
              a much larger pan/zoom transform, plus a discrete
              high-contrast disc flown across the frame - real object motion
              layered on top of real camera motion.
  text      - synthetic UI/screen content: a title bar, sidebar, scrolling
              data table with single-pixel gridlines, a live seven-segment
              clock, and a hard-blinking cursor. Deliberately drawn with
              flat rectangles rather than a system font: TrueType hinting and
              anti-aliasing differ across FreeType/Pillow builds, and a
              benchmark whose reference frames are not bit-reproducible on
              another machine is not reproducible. The single-pixel
              gridlines and hard edges reproduce the failure mode that
              matters (upscalers visibly smear sharp screen content)
              without depending on any installed font.
  animation - flat-colour/cel-shaded: solid polygons with no anti-aliasing,
              hard black outlines, diagonal cut edges (the worst case for 2x
              box subsampling), and independently bouncing foreground shapes.

`natural`/`texture`/`motion` share one mechanism: a single procedurally
generated "world" canvas somewhat larger than the frame, sampled through an
explicit per-frame affine transform (translate + scale-about-centre) with a
hand-written bilinear resampler - the same technique, for the same
determinism reason, as `tools/temporal-sequences.ts`'s `sampleBilinear`/
`renderFrame` (content is generated once; only the transform, and therefore
the motion, varies by category). `text`/`animation` redraw a fresh vector
frame every time instead, because integer-pixel UI scrolling and cel motion
are the realistic way those two ever move, and it keeps their edges exactly
as sharp as intended.

Usage
-----
    /tmp/aethertrain/bin/python tools/videobench.py --prepare
    /tmp/aethertrain/bin/python tools/videobench.py --score-baselines --json

`--prepare` is deterministic in (seed, frame count, fps): it removes stale
selected-category outputs before regenerating them, and re-running it with
the same ffmpeg build reproduces byte-identical PNGs, video clips and
manifest (hashes make that checkable). `--score-baselines` reads that
manifest and requires nothing but ffmpeg and PIL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

FFMPEG = os.environ.get("AETHERVSR_FFMPEG", "/opt/homebrew/bin/ffmpeg")
FFPROBE = os.environ.get(
    "AETHERVSR_FFPROBE",
    os.path.join(os.path.dirname(FFMPEG), "ffprobe"),
)

MASTER_W, MASTER_H = 2560, 1440
LR_W, LR_H = 1280, 720
DEFAULT_FPS = 24
DEFAULT_FRAMES = 12  # 0.5 second at 24 fps; measured corpus size is 227 MB.
DEFAULT_SEED = 424242  # matches the seed style already used by tools/prepare-eval-set.py.
MARGIN = 160  # world-canvas padding for the sub-pixel pan/zoom categories.

CATEGORIES = ("natural", "texture", "motion", "text", "animation")

CATEGORY_DESCRIPTIONS = {
    "natural": (
        "Synthetic camera-like content: multi-octave fractal noise, a soft "
        "lighting vignette, independent per-pixel sensor grain, sampled "
        "through a gentle sub-pixel camera pan/zoom."
    ),
    "texture": (
        "Synthetic high-frequency content: a zone-plate chirp crossed with "
        "a near-Nyquist sinusoidal grating in a central resolution-chart "
        "region, sampled through a small sub-pixel pan/zoom."
    ),
    "motion": (
        "Synthetic motion-stress content: cluttered fractal-noise "
        "background under a large sub-pixel pan/zoom, plus a discrete "
        "high-contrast disc flown across the frame independently of the "
        "camera motion."
    ),
    "text": (
        "Synthetic UI/screen content: title bar, sidebar, a scrolling data "
        "table with single-pixel gridlines, a live seven-segment clock and "
        "a hard-blinking cursor. Drawn with flat rectangles, not a system "
        "font, so it is bit-reproducible across machines."
    ),
    "animation": (
        "Synthetic flat-colour/cel content: solid non-anti-aliased "
        "polygons with hard black outlines and diagonal cut edges, plus "
        "independently bouncing foreground shapes."
    ),
}

# Quality tiers encoded for every category. libx264/libvpx-vp9/libsvtav1 are
# all present in this machine's ffmpeg build (see manifest["ffmpeg"]).
TIERS = [
    {
        "name": "h264-high",
        "codec": "libx264",
        "stream_codec": "h264",
        "container": "mp4",
        "args": ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"],
    },
    {
        "name": "h264-medium",
        "codec": "libx264",
        "stream_codec": "h264",
        "container": "mp4",
        "args": ["-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p"],
    },
    {
        "name": "h264-low",
        "codec": "libx264",
        "stream_codec": "h264",
        "container": "mp4",
        "args": ["-c:v", "libx264", "-crf", "28", "-preset", "medium", "-pix_fmt", "yuv420p"],
    },
    {
        # MP4 is used rather than WebM because this ffmpeg/libvpx build's
        # WebM muxer writes non-repeatable container metadata even when decoded
        # frames are identical; VP9-in-MP4 remains a standards-supported,
        # lossless-to-container choice and is byte-repeatable here.
        "name": "vp9-medium",
        "codec": "libvpx-vp9",
        "stream_codec": "vp9",
        "container": "mp4",
        "args": [
            "-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0",
            "-cpu-used", "4", "-row-mt", "1", "-pix_fmt", "yuv420p",
        ],
    },
    {
        "name": "av1-medium",
        "codec": "libsvtav1",
        "stream_codec": "av1",
        "container": "mp4",
        "args": ["-c:v", "libsvtav1", "-crf", "32", "-preset", "8", "-pix_fmt", "yuv420p"],
    },
]

SCALERS = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}

# Source PNGs are RGB full-range. Each encoded clip is explicitly converted to
# 8-bit 4:2:0 BT.709 *limited* range (the normal HD delivery representation),
# carries matching stream metadata, then is explicitly converted back to RGB
# full range during decode. This is both the controlled codec degradation and
# the range contract independently tested by `colour_calibration`.
RGB_PC_TO_YUV709_TV_FILTER = (
    "scale=in_range=pc:out_range=tv:in_color_matrix=bt709:"
    "out_color_matrix=bt709:in_primaries=bt709:out_primaries=bt709:"
    "in_transfer=bt709:out_transfer=bt709,format=yuv420p,"
    "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709"
)
YUV709_TV_TO_RGB_PC_FILTER = (
    "scale=in_range=tv:out_range=pc:in_color_matrix=bt709:"
    "out_color_matrix=bt709:in_primaries=bt709:out_primaries=bt709:"
    "in_transfer=bt709:out_transfer=bt709,format=rgb24,"
    "setparams=range=pc:colorspace=gbr:color_primaries=bt709:color_trc=bt709"
)

# VMAF accepts a pair of equal-geometry videos. It is luma-led, but its
# RGB->YUV conversion must still be explicit: automatic conversion left
# colours/range "unknown" in ffmpeg's graph. We convert both images identically
# to BT.709 limited-range yuv444p (no metric-side chroma downsample) before
# handing them to libvmaf.
VMAF_RGB_PC_TO_YUV709_TV_FILTER = (
    "scale=in_range=pc:out_range=tv:in_color_matrix=bt709:"
    "out_color_matrix=bt709:in_primaries=bt709:out_primaries=bt709:"
    "in_transfer=bt709:out_transfer=bt709,format=yuv444p,"
    "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709"
)
COLOUR_OUTPUT_METADATA = [
    "-color_range", "tv",
    "-colorspace", "bt709",
    "-color_primaries", "bt709",
    "-color_trc", "bt709",
]
EXPECTED_VIDEO_COLOUR = {
    "pix_fmt": "yuv420p",
    "color_range": "tv",
    "color_space": "bt709",
    "color_transfer": "bt709",
    "color_primaries": "bt709",
}


# --------------------------------------------------------------------------
# Determinism plumbing
# --------------------------------------------------------------------------

def derive_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256(f"{seed}:{'/'.join(parts)}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def rng_for(seed: int, *parts: str) -> np.random.Generator:
    return np.random.default_rng(derive_seed(seed, *parts))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Procedural content: natural / texture / motion (world + affine sampler)
# --------------------------------------------------------------------------

def fractal_noise(rng: np.random.Generator, h: int, w: int, octaves: int = 7, persistence: float = 0.6, base: int = 3) -> np.ndarray:
    """1/f-ish value noise: sums `octaves` random grids, each upsampled to
    (h, w), with geometrically decaying amplitude. Returns float64 in [0, 1].
    """
    acc = np.zeros((h, w), dtype=np.float64)
    amp = 1.0
    amp_sum = 0.0
    for o in range(octaves):
        gh = max(2, base * (2 ** o))
        gw = max(2, round(gh * w / h))
        grid = (rng.random((gh, gw)) * 255.0).astype(np.uint8)
        layer = np.asarray(
            Image.fromarray(grid, mode="L").resize((w, h), resample=Image.BILINEAR),
            dtype=np.float64,
        ) / 255.0
        acc += layer * amp
        amp_sum += amp
        amp *= persistence
    return acc / amp_sum


def build_natural_world(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    luma = fractal_noise(rng, h, w, octaves=7, persistence=0.62)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2.0, h / 2.0
    vignette = 1.0 - 0.25 * ((xx - cx) ** 2 + (yy - cy) ** 2) / (cx ** 2 + cy ** 2)
    chroma_r = fractal_noise(rng, h, w, octaves=4, persistence=0.5)
    chroma_b = fractal_noise(rng, h, w, octaves=4, persistence=0.5)
    r = luma * vignette + 0.06 * (chroma_r - 0.5)
    g = luma * vignette
    b = luma * vignette + 0.06 * (chroma_b - 0.5)
    color = np.stack([r, g, b], axis=-1)
    # A little independent per-pixel grain (real sensor noise) so the finest
    # octave isn't the only source of near-Nyquist energy. Kept small: true
    # i.i.d. noise is incompressible in PNG, and the fractal pyramid above
    # already reaches near-pixel-scale detail through its highest octave
    # (base=3, octaves=7 -> a 384-cell grid over ~2900px, ~7.5px period)
    # without paying that cost on every pixel.
    grain = rng.standard_normal((h, w, 3)) * 0.014
    return np.clip(color + grain, 0.0, 1.0).astype(np.float32)


def radial_mask(h: int, w: int, inner: float, outer: float) -> np.ndarray:
    """Elliptical radial mask: 1.0 within `inner` of the half-extent from
    centre, 0.0 beyond `outer`, smoothstepped in between. `inner`/`outer`
    are fractions of the half-width/half-height, so 1.0 reaches the frame
    edge along each axis.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2.0, h / 2.0
    rn = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    t = np.clip((rn - inner) / max(outer - inner, 1e-9), 0.0, 1.0)
    smooth = 1.0 - (3 * t ** 2 - 2 * t ** 3)
    return smooth


def build_texture_world(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """A resolution-chart-style pattern: a zone-plate chirp crossed with a
    sinusoidal grating near Nyquist, confined to a small central chart
    region by a tight radial mask (like a real MTF test chart's wedge
    sitting on a plain field) rather than spread over the full frame. The
    chart region is still genuinely near-Nyquist and genuinely hard; the
    (exactly flat, zero-variance) plain border exists so this category
    doesn't cost several hundred MB of near-incompressible PNG for no
    measurement benefit - PSNR/SSIM/VMAF over the frame still score
    reconstruction of the hard region, which is what this category is for.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2.0, h / 2.0
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    k = rng.uniform(0.0006, 0.001)
    theta = rng.uniform(0, math.pi)
    freq = rng.uniform(1.1, 1.6)
    zone = np.sin(k * r2)
    bars = np.sin(freq * (xx * math.cos(theta) + yy * math.sin(theta)))
    base = 0.55 * zone + 0.45 * bars
    mask = radial_mask(h, w, inner=0.22, outer=0.34)
    r = np.sin(k * r2 * 1.00 + 0.0)
    g = np.sin(k * r2 * 1.05 + 0.8)
    b = np.sin(k * r2 * 0.95 + 1.6)
    deviation = (0.5 * np.stack([r, g, b], axis=-1) + 0.5 * base[..., None]) * mask[..., None]
    return np.clip(0.5 + 0.42 * deviation, 0.0, 1.0).astype(np.float32)


def build_motion_world(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    luma = fractal_noise(rng, h, w, octaves=8, persistence=0.68)
    grain = rng.standard_normal((h, w, 3)) * 0.018
    color = np.repeat(luma[..., None], 3, axis=2) + grain
    return np.clip(color, 0.0, 1.0).astype(np.float32)

def bilinear_sample(world: np.ndarray, out_h: int, out_w: int, dx: float, dy: float, scale: float) -> np.ndarray:
    """Samples `world` (float32 [0,1] HxWx3) through translate-then-scale
    about the world's centre, matching temporal-sequences.ts's transform
    convention. Returns float64 [0,1] out_h x out_w x 3.
    """
    wh, ww = world.shape[0], world.shape[1]
    wcx, wcy = ww / 2.0, wh / 2.0
    ocx, ocy = out_w / 2.0, out_h / 2.0
    ys, xs = np.mgrid[0:out_h, 0:out_w].astype(np.float64)
    wx = wcx + (xs - ocx) / scale + dx
    wy = wcy + (ys - ocy) / scale + dy
    wx = np.clip(wx, 0.0, ww - 1.001)
    wy = np.clip(wy, 0.0, wh - 1.001)
    x0 = np.floor(wx).astype(np.int64)
    y0 = np.floor(wy).astype(np.int64)
    x1 = np.minimum(x0 + 1, ww - 1)
    y1 = np.minimum(y0 + 1, wh - 1)
    fx = (wx - x0)[..., None]
    fy = (wy - y0)[..., None]
    top = world[y0, x0] * (1 - fx) + world[y0, x1] * fx
    bot = world[y1, x0] * (1 - fx) + world[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def natural_motion(t: float, margin: float) -> tuple[float, float, float]:
    dx = 0.5 * margin * math.sin(2 * math.pi * t)
    dy = 0.35 * margin * math.sin(2 * math.pi * t * 0.5 + 1.1)
    scale = 1.0 + 0.03 * (0.5 - 0.5 * math.cos(2 * math.pi * t))
    return dx, dy, scale


def texture_motion(t: float, margin: float) -> tuple[float, float, float]:
    dx = 0.3 * margin * math.sin(2 * math.pi * t * 1.5)
    dy = 0.25 * margin * math.cos(2 * math.pi * t * 1.5)
    scale = 1.0 + 0.06 * (0.5 - 0.5 * math.cos(2 * math.pi * t * 2))
    return dx, dy, scale


def motion_motion(t: float, margin: float) -> tuple[float, float, float]:
    dx = 0.75 * margin * math.sin(2 * math.pi * t + 0.3)
    dy = 0.5 * margin * math.cos(2 * math.pi * t * 1.3)
    scale = 1.0 + 0.15 * (0.5 - 0.5 * math.cos(2 * math.pi * t * 1.7))
    return dx, dy, scale


def render_world_sequence(world: np.ndarray, motion_fn, frames: int, margin: float) -> list[np.ndarray]:
    out = []
    for i in range(frames):
        t = i / frames
        dx, dy, scale = motion_fn(t, margin)
        frame = bilinear_sample(world, MASTER_H, MASTER_W, dx, dy, scale)
        out.append(np.clip(frame * 255.0 + 0.5, 0, 255).astype(np.uint8))
    return out


def add_moving_disc(frame_u8: np.ndarray, t: float) -> np.ndarray:
    h, w = frame_u8.shape[:2]
    cx = w * (0.12 + 0.72 * t)
    cy = h * (0.5 + 0.28 * math.sin(2 * math.pi * t * 2.0))
    radius = 58
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    fill = d2 <= (radius - 6) ** 2
    ring = (d2 <= radius ** 2) & ~fill
    out = frame_u8.copy()
    out[fill] = np.array([255, 150, 20], dtype=np.uint8)
    out[ring] = np.array([20, 20, 20], dtype=np.uint8)
    return out


def render_motion_sequence(world: np.ndarray, frames: int, margin: float) -> list[np.ndarray]:
    out = []
    for i in range(frames):
        t = i / frames
        dx, dy, scale = motion_motion(t, margin)
        frame = bilinear_sample(world, MASTER_H, MASTER_W, dx, dy, scale)
        frame_u8 = np.clip(frame * 255.0 + 0.5, 0, 255).astype(np.uint8)
        out.append(add_moving_disc(frame_u8, t))
    return out


# --------------------------------------------------------------------------
# Procedural content: text/UI (own seven-segment font, no system font)
# --------------------------------------------------------------------------

SEVEN_SEG = {
    "0": set("abcdef"), "1": set("bc"), "2": set("abged"), "3": set("abgcd"),
    "4": set("fgbc"), "5": set("afgcd"), "6": set("afgecd"), "7": set("abc"),
    "8": set("abcdefg"), "9": set("abcdfg"),
}


def draw_seven_seg_digit(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, digit: str, color, thickness: int) -> None:
    segs = SEVEN_SEG[digit]
    t = thickness
    if "a" in segs:
        draw.rectangle([x + t, y, x + w - t, y + t], fill=color)
    if "g" in segs:
        draw.rectangle([x + t, y + h // 2 - t // 2, x + w - t, y + h // 2 + t // 2], fill=color)
    if "d" in segs:
        draw.rectangle([x + t, y + h - t, x + w - t, y + h], fill=color)
    if "f" in segs:
        draw.rectangle([x, y, x + t, y + h // 2 + t // 2], fill=color)
    if "b" in segs:
        draw.rectangle([x + w - t, y, x + w, y + h // 2 + t // 2], fill=color)
    if "e" in segs:
        draw.rectangle([x, y + h // 2 - t // 2, x + t, y + h], fill=color)
    if "c" in segs:
        draw.rectangle([x + w - t, y + h // 2 - t // 2, x + w, y + h], fill=color)


def draw_seven_seg_number(draw: ImageDraw.ImageDraw, x: int, y: int, digit_w: int, digit_h: int, gap: int, text: str, color, thickness: int) -> None:
    cx = x
    for ch in text:
        if ch == ":":
            cw = digit_w // 3
            draw.rectangle([cx, y + int(digit_h * 0.28), cx + cw, y + int(digit_h * 0.28) + cw], fill=color)
            draw.rectangle([cx, y + int(digit_h * 0.62), cx + cw, y + int(digit_h * 0.62) + cw], fill=color)
            cx += cw + gap
        else:
            draw_seven_seg_digit(draw, cx, y, digit_w, digit_h, ch, color, thickness)
            cx += digit_w + gap


def build_text_content(rng: np.random.Generator) -> dict:
    n_sidebar_items = 14
    n_rows = 16
    return {
        "sidebar_widths": [int(v) for v in rng.integers(60, 180, size=n_sidebar_items)],
        "table_values": [int(v) for v in rng.integers(0, 1000, size=n_rows)],
        "name_bar_widths": [int(v) for v in rng.integers(80, 300, size=n_rows)],
    }


def render_text_frame(content: dict, frame_idx: int, total_frames: int) -> np.ndarray:
    img = Image.new("RGB", (MASTER_W, MASTER_H), (235, 236, 239))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, MASTER_W, 72], fill=(38, 40, 46))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        cx = 36 + i * 34
        draw.ellipse([cx - 12, 36 - 12, cx + 12, 36 + 12], fill=c)

    counter = frame_idx
    clock_str = f"{(counter // 3600) % 24:02d}:{(counter // 60) % 60:02d}:{counter % 60:02d}"
    draw_seven_seg_number(draw, MASTER_W - 420, 16, 34, 40, 10, clock_str, (210, 215, 225), 6)

    sidebar_w = 300
    draw.rectangle([0, 72, sidebar_w, MASTER_H], fill=(46, 48, 56))
    widths = content["sidebar_widths"]
    for i, w in enumerate(widths):
        y = 100 + i * 54
        if y + 28 > MASTER_H:
            break
        draw.rectangle([28, y, 28 + 22, y + 22], fill=(120, 170, 255))
        draw.rectangle([64, y + 4, 64 + w, y + 4 + 14], fill=(150, 153, 165))
    hi_index = frame_idx % len(widths)
    hy = 100 + hi_index * 54
    if hy + 28 <= MASTER_H:
        draw.rectangle([0, hy - 6, sidebar_w, hy + 34], outline=(120, 170, 255), width=3)

    mx = sidebar_w + 40
    draw.rectangle([mx, 100, MASTER_W - 40, 140], fill=(255, 255, 255), outline=(200, 202, 208), width=2)
    draw.rectangle([mx + 20, 112, mx + 220, 128], fill=(190, 192, 198))
    draw.rectangle([mx + 900, 112, mx + 1100, 128], fill=(190, 192, 198))

    table_top = 170
    row_h = 48
    table_values = content["table_values"]
    name_widths = content["name_bar_widths"]
    col1_x = mx + 140
    table_bottom = table_top
    for r, val in enumerate(table_values):
        y0 = table_top + r * row_h
        y1 = y0 + row_h
        if y1 > MASTER_H - 40:
            break
        table_bottom = y1
        fill = (255, 255, 255) if r % 2 == 0 else (244, 245, 248)
        draw.rectangle([mx, y0, MASTER_W - 40, y1], fill=fill)
        draw_seven_seg_number(draw, mx + 16, y0 + 8, 18, 30, 6, f"{val:03d}", (60, 64, 74), 4)
        draw.rectangle([col1_x + 16, y0 + 16, col1_x + 16 + name_widths[r], y0 + 16 + 16], fill=(170, 173, 182))
    n_rows_drawn = max(1, round((table_bottom - table_top) / row_h))
    for r in range(n_rows_drawn + 1):
        y = table_top + r * row_h
        if y > table_bottom:
            break
        draw.line([mx, y, MASTER_W - 40, y], fill=(220, 222, 228), width=1)
    for x in (mx, col1_x, MASTER_W - 40):
        draw.line([x, table_top, x, table_bottom], fill=(220, 222, 228), width=1)

    thumb_h = 120
    track_h = MASTER_H - 72 - 20
    thumb_y = 72 + 10 + int((track_h - thumb_h) * (frame_idx / max(1, total_frames - 1)))
    draw.rectangle([MASTER_W - 14, 72 + 10, MASTER_W - 4, MASTER_H - 10], fill=(225, 226, 230))
    draw.rectangle([MASTER_W - 14, thumb_y, MASTER_W - 4, thumb_y + thumb_h], fill=(150, 153, 165))

    if frame_idx % 2 == 0:
        draw.rectangle([mx + 230, 112, mx + 236, 128], fill=(30, 30, 34))

    return np.asarray(img, dtype=np.uint8)


# --------------------------------------------------------------------------
# Procedural content: animation (flat colour, hard edges)
# --------------------------------------------------------------------------

PALETTE = [
    (230, 57, 70), (69, 123, 157), (42, 157, 143), (233, 196, 106),
    (244, 162, 97), (38, 70, 83), (255, 209, 102), (6, 214, 160),
]


def build_animation_content(rng: np.random.Generator) -> dict:
    bg_color = PALETTE[int(rng.integers(0, len(PALETTE)))]
    cuts = []
    for _ in range(5):
        x0 = int(rng.integers(0, MASTER_W))
        y0 = int(rng.integers(0, MASTER_H))
        w = int(rng.integers(300, 900))
        h = int(rng.integers(200, 700))
        color = PALETTE[int(rng.integers(0, len(PALETTE)))]
        style = int(rng.integers(0, 2))
        cuts.append((x0, y0, w, h, color, style))
    shapes = []
    kinds = ["circle", "triangle", "square"]
    for _ in range(4):
        shapes.append({
            "kind": kinds[int(rng.integers(0, len(kinds)))],
            "color": PALETTE[int(rng.integers(0, len(PALETTE)))],
            "size": int(rng.integers(70, 160)),
            "x0": float(rng.uniform(0, MASTER_W)),
            "y0": float(rng.uniform(0, MASTER_H)),
            "vx": float(rng.uniform(-90, 90)),
            "vy": float(rng.uniform(-60, 60)),
        })
    return {"bg_color": bg_color, "cuts": cuts, "shapes": shapes}


def bounced_position(p0: float, v: float, t: int, size: float, extent: float) -> float:
    span = extent - 2 * size
    if span <= 0:
        return extent / 2.0
    raw = p0 + v * t
    period = 2 * span
    m = raw % period
    return size + (m if m <= span else period - m)


def render_animation_frame(content: dict, frame_idx: int, total_frames: int) -> np.ndarray:
    img = Image.new("RGB", (MASTER_W, MASTER_H), content["bg_color"])
    draw = ImageDraw.Draw(img)
    for (x0, y0, w, h, color, style) in content["cuts"]:
        if style == 0:
            draw.polygon([(x0, y0), (x0 + w, y0), (x0 + int(w * 0.4), y0 + h)], fill=color)
        else:
            draw.polygon(
                [(x0, y0), (x0 + w, y0 + int(h * 0.3)), (x0 + int(w * 0.6), y0 + h), (x0 - int(w * 0.2), y0 + h)],
                fill=color,
            )
    for shape in content["shapes"]:
        cx = bounced_position(shape["x0"], shape["vx"], frame_idx, shape["size"], MASTER_W)
        cy = bounced_position(shape["y0"], shape["vy"], frame_idx, shape["size"], MASTER_H)
        s = shape["size"]
        if shape["kind"] == "circle":
            draw.ellipse([cx - s, cy - s, cx + s, cy + s], fill=shape["color"], outline=(20, 20, 20), width=4)
        elif shape["kind"] == "square":
            draw.rectangle([cx - s, cy - s, cx + s, cy + s], fill=shape["color"], outline=(20, 20, 20), width=4)
        else:
            draw.polygon([(cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)], fill=shape["color"], outline=(20, 20, 20), width=4)
    return np.asarray(img, dtype=np.uint8)


def generate_category_frames(category: str, seed: int, frames: int) -> list[np.ndarray]:
    if category == "natural":
        world = build_natural_world(rng_for(seed, "natural", "world"), MASTER_H + 2 * MARGIN, MASTER_W + 2 * MARGIN)
        return render_world_sequence(world, natural_motion, frames, MARGIN)
    if category == "texture":
        world = build_texture_world(rng_for(seed, "texture", "world"), MASTER_H + 2 * MARGIN, MASTER_W + 2 * MARGIN)
        return render_world_sequence(world, texture_motion, frames, MARGIN)
    if category == "motion":
        world = build_motion_world(rng_for(seed, "motion", "world"), MASTER_H + 2 * MARGIN, MASTER_W + 2 * MARGIN)
        return render_motion_sequence(world, frames, MARGIN)
    if category == "text":
        content = build_text_content(rng_for(seed, "text", "content"))
        return [render_text_frame(content, i, frames) for i in range(frames)]
    if category == "animation":
        content = build_animation_content(rng_for(seed, "animation", "content"))
        return [render_animation_frame(content, i, frames) for i in range(frames)]
    raise ValueError(f"unknown category {category!r}")


# --------------------------------------------------------------------------
# Exact 2x box downsample - matches AetherSR's training/eval contract
# --------------------------------------------------------------------------

def box_downsample2(hr_u8: np.ndarray) -> np.ndarray:
    """Returns PNG-storable LR from an RGB8 master via the exact 2x2 mean.

    The spatial filter is exactly `AetherSR.box_downsample2` /
    `F.avg_pool2d(kernel_size=2, stride=2)`: every output sample is the mean
    of one non-overlapping 2x2 input block. A PNG cannot represent a
    half-integer result, so the literal mean is then rounded with NumPy's
    deterministic nearest-even rule before it becomes RGB8. The manifest
    names both operations; no implicit image-resize kernel is involved.
    """
    h, w, c = hr_u8.shape
    hr = hr_u8.astype(np.float32).reshape(h // 2, 2, w // 2, 2, c).mean(axis=(1, 3))
    return np.clip(np.round(hr), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# ffmpeg plumbing and metric input validation
# --------------------------------------------------------------------------

def ffmpeg_version_output() -> list[str]:
    proc = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg -version failed: {proc.stderr}")
    return proc.stdout.strip().splitlines()


def ffmpeg_run(args: list[str]) -> subprocess.CompletedProcess:
    cmd = [FFMPEG, "-y", "-loglevel", "error"] + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({' '.join(cmd)}):\n{proc.stderr}")
    return proc


def ffprobe_stream(path: str) -> dict:
    """Reads exactly the stream properties that bind the color contract."""
    cmd = [
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,color_range,color_space,"
        "color_transfer,color_primaries",
        "-of", "json", path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed ({' '.join(cmd)}):\n{proc.stderr}")
    streams = json.loads(proc.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"ffprobe expected one video stream in {path}, got {len(streams)}")
    return streams[0]


def stream_contract(tier: dict, stream: dict) -> dict:
    expected = {
        "codec_name": tier["stream_codec"],
        "width": LR_W,
        "height": LR_H,
        **EXPECTED_VIDEO_COLOUR,
    }
    mismatches = {
        field: {"expected": value, "observed": stream.get(field)}
        for field, value in expected.items()
        if stream.get(field) != value
    }
    return {"ok": not mismatches, "expected": expected, "observed": stream, "mismatches": mismatches}


def encode_sequence_args(source_dir: str, tier: dict, fps: int, video_path: str) -> list[str]:
    return [
        "-start_number", "0", "-framerate", str(fps),
        "-i", os.path.join(source_dir, "frame_%04d.png"),
        "-vf", RGB_PC_TO_YUV709_TV_FILTER,
        *tier["args"],
        *COLOUR_OUTPUT_METADATA,
        video_path,
    ]


def decode_sequence_args(video_path: str, decoded_dir: str) -> list[str]:
    return [
        "-i", video_path,
        "-vf", YUV709_TV_TO_RGB_PC_FILTER,
        "-start_number", "0",
        os.path.join(decoded_dir, "frame_%04d.png"),
    ]


def expected_frame_names(count: int) -> list[str]:
    return [f"frame_{i:04d}.png" for i in range(count)]


def inspect_png_sequence(dir_path: str, count: int) -> dict:
    """Checks exact frame names/order, RGB8 mode, and immutable geometry."""
    expected = expected_frame_names(count)
    actual = sorted(f for f in os.listdir(dir_path) if f.endswith(".png"))
    dimensions: set[tuple[int, int]] = set()
    modes: set[str] = set()
    for name in actual:
        with Image.open(os.path.join(dir_path, name)) as img:
            dimensions.add(img.size)
            modes.add(img.mode)
    return {
        "expected_frame_count": count,
        "actual_frame_count": len(actual),
        "exact_filenames_and_order": actual == expected,
        "dimensions": [list(size) for size in sorted(dimensions)],
        "modes": sorted(modes),
    }


def sequence_contract(inspection: dict, size: tuple[int, int]) -> bool:
    return (
        inspection["actual_frame_count"] == inspection["expected_frame_count"]
        and inspection["exact_filenames_and_order"]
        and inspection["dimensions"] == [[size[0], size[1]]]
        and inspection["modes"] == ["RGB"]
    )


def frame_range_stats(dir_path: str, count: int) -> tuple[int, int]:
    gmin, gmax = 255, 0
    for name in expected_frame_names(count):
        with Image.open(os.path.join(dir_path, name)) as img:
            arr = np.asarray(img)
        gmin = min(gmin, int(arr.min()))
        gmax = max(gmax, int(arr.max()))
    return gmin, gmax


def parse_psnr_log(path: str) -> list[float]:
    vals = []
    with open(path) as fh:
        for line in fh:
            m = re.search(r"psnr_avg:(\S+)", line)
            if m:
                value = m.group(1)
                vals.append(float("inf") if value.lower() == "inf" else float(value))
    return vals


def parse_ssim_log(path: str) -> list[float]:
    vals = []
    with open(path) as fh:
        for line in fh:
            m = re.search(r"All:(\S+)", line)
            if m:
                vals.append(float(m.group(1)))
    return vals


def assert_metric_count(metric: str, values: list[float], count: int) -> None:
    if len(values) != count:
        raise RuntimeError(f"{metric} produced {len(values)} frames; expected exactly {count}")


def compare_sequences_psnr_ssim(
    distorted_dir: str,
    reference_dir: str,
    count: int,
    fps: int,
) -> tuple[list[float], list[float]]:
    tmp = tempfile.mkdtemp(prefix="vb_metrics_")
    try:
        psnr_log = os.path.join(tmp, "psnr.log")
        ssim_log = os.path.join(tmp, "ssim.log")
        ffmpeg_run([
            "-start_number", "0", "-framerate", str(fps), "-i", os.path.join(distorted_dir, "frame_%04d.png"),
            "-start_number", "0", "-framerate", str(fps), "-i", os.path.join(reference_dir, "frame_%04d.png"),
            "-lavfi", f"[0:v][1:v]psnr=stats_file={psnr_log};[0:v][1:v]ssim=stats_file={ssim_log}",
            "-f", "null", "-",
        ])
        psnr = parse_psnr_log(psnr_log)
        ssim = parse_ssim_log(ssim_log)
        assert_metric_count("PSNR", psnr, count)
        assert_metric_count("SSIM", ssim, count)
        return psnr, ssim
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def compute_vmaf(distorted_dir: str, reference_dir: str, count: int, fps: int) -> list[float]:
    """Computes VMAF only after the caller has verified sequence geometry and
    the RGB/full <-> BT.709/limited color contract. Both inputs undergo the
    same explicit conversion, so libvmaf never has to infer colorspace/range.
    """
    tmp = tempfile.mkdtemp(prefix="vb_vmaf_")
    try:
        log = os.path.join(tmp, "vmaf.json")
        graph = (
            f"[0:v]{VMAF_RGB_PC_TO_YUV709_TV_FILTER}[dist];"
            f"[1:v]{VMAF_RGB_PC_TO_YUV709_TV_FILTER}[ref];"
            f"[dist][ref]libvmaf=log_path={log}:log_fmt=json"
        )
        ffmpeg_run([
            "-start_number", "0", "-framerate", str(fps), "-i", os.path.join(distorted_dir, "frame_%04d.png"),
            "-start_number", "0", "-framerate", str(fps), "-i", os.path.join(reference_dir, "frame_%04d.png"),
            "-lavfi", graph,
            "-f", "null", "-",
        ])
        with open(log) as fh:
            data = json.load(fh)
        values = [float(frame["metrics"]["vmaf"]) for frame in data["frames"]]
        assert_metric_count("VMAF", values, count)
        return values
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def distribution(values: list[float]) -> dict:
    finite = [(index, value) for index, value in enumerate(values) if math.isfinite(value)]
    nonfinite_count = len(values) - len(finite)
    if not finite:
        return {
            "mean": None, "p5": None, "median": None, "p95": None,
            "min": None, "max": None, "worst_frames": [],
            "nonfinite_count": nonfinite_count, "count": len(values),
        }
    finite_values = np.asarray([value for _, value in finite], dtype=np.float64)
    worst = sorted(finite, key=lambda pair: pair[1])[:3]
    return {
        "mean": float(finite_values.mean()),
        "p5": float(np.percentile(finite_values, 5)),
        "median": float(np.percentile(finite_values, 50)),
        "p95": float(np.percentile(finite_values, 95)),
        "min": float(finite_values.min()),
        "max": float(finite_values.max()),
        "worst_frames": [{"frame": index, "value": float(value)} for index, value in worst],
        "nonfinite_count": nonfinite_count,
        "count": len(values),
    }


def json_safe(value):
    """Turns non-standard float values into valid JSON without hiding them."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
    return value


def write_json(path: str, value: dict) -> None:
    with open(path, "w") as fh:
        json.dump(json_safe(value), fh, indent=1, allow_nan=False)


def json_text(value: dict) -> str:
    return json.dumps(json_safe(value), indent=1, allow_nan=False)

# An actual frame's extrema are not enough to prove range handling: a natural
# clip might never contain black or white. This independent probe uses broad,
# flat RGB patches where codec quantisation cannot hide a 16↔0 / 235↔255
# range error. It is run once per codec/tier with exactly the production
# encode/decode filters and metadata flags.
COLOUR_CALIBRATION_TILES = [
    ("black", (0, 0, 0)),
    ("white", (255, 255, 255)),
    ("mid_gray", (128, 128, 128)),
    ("near_black", (16, 16, 16)),
    ("near_white", (235, 235, 235)),
    ("red", (255, 0, 0)),
    ("green", (0, 255, 0)),
    ("blue", (0, 0, 255)),
]


def write_colour_calibration(path: str) -> str:
    image = np.empty((LR_H, LR_W, 3), dtype=np.uint8)
    tile_w, tile_h = LR_W // 4, LR_H // 2
    for index, (_, colour) in enumerate(COLOUR_CALIBRATION_TILES):
        x0 = (index % 4) * tile_w
        y0 = (index // 4) * tile_h
        image[y0:y0 + tile_h, x0:x0 + tile_w] = colour
    Image.fromarray(image, "RGB").save(path, "PNG", compress_level=9)
    return sha256_file(path)


def sampled_calibration_colours(path: str) -> dict[str, list[float]]:
    with Image.open(path) as img:
        image = np.asarray(img.convert("RGB"), dtype=np.float64)
    tile_w, tile_h = LR_W // 4, LR_H // 2
    sample_w, sample_h = 120, 120
    observed = {}
    for index, (name, _) in enumerate(COLOUR_CALIBRATION_TILES):
        x0 = (index % 4) * tile_w + (tile_w - sample_w) // 2
        y0 = (index // 4) * tile_h + (tile_h - sample_h) // 2
        mean = image[y0:y0 + sample_h, x0:x0 + sample_w].mean(axis=(0, 1))
        observed[name] = [float(channel) for channel in mean]
    return observed


def colour_calibration(tier: dict, fps: int) -> dict:
    """Proves RGB full-range round-trips through this tier's colour contract."""
    tmp = tempfile.mkdtemp(prefix="vb_colour_")
    try:
        source_dir = os.path.join(tmp, "source")
        decoded_dir = os.path.join(tmp, "decoded")
        os.makedirs(source_dir)
        os.makedirs(decoded_dir)
        source_path = os.path.join(source_dir, "frame_0000.png")
        source_sha256 = write_colour_calibration(source_path)
        video_path = os.path.join(tmp, f"probe.{tier['container']}")
        encode_args = encode_sequence_args(source_dir, tier, fps, video_path)
        ffmpeg_run(encode_args)
        stream = ffprobe_stream(video_path)
        stream_check = stream_contract(tier, stream)
        ffmpeg_run(decode_sequence_args(video_path, decoded_dir))
        decoded_path = os.path.join(decoded_dir, "frame_0000.png")
        observed = sampled_calibration_colours(decoded_path)
        expected = {name: list(colour) for name, colour in COLOUR_CALIBRATION_TILES}
        max_error = {
            name: max(abs(observed[name][channel] - expected[name][channel]) for channel in range(3))
            for name in expected
        }
        black_max = max(observed["black"])
        white_min = min(observed["white"])
        range_round_trip_ok = black_max <= 8.0 and white_min >= 247.0
        colour_error_ok = max(max_error.values()) <= 8.0
        return {
            "description": (
                "One 1280x720 RGB/full calibration frame: eight 320x360 flat "
                "tiles (black, white, mid-gray, near-black, near-white, red, "
                "green, blue); each result is the mean of a central 120x120 "
                "patch after this tier's real encode/decode. It catches a "
                "limited/full range mismatch independently of corpus content."
            ),
            "source_sha256": source_sha256,
            "encode_flags": ["-vf", RGB_PC_TO_YUV709_TV_FILTER, *tier["args"], *COLOUR_OUTPUT_METADATA],
            "stream_contract": stream_check,
            "expected_rgb": expected,
            "decoded_patch_mean_rgb": observed,
            "max_abs_error_by_patch": max_error,
            "black_patch_max": black_max,
            "white_patch_min": white_min,
            "range_round_trip_ok": range_round_trip_ok,
            "colour_error_ok": colour_error_ok,
            "passed": bool(stream_check["ok"] and range_round_trip_ok and colour_error_ok),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# --prepare
# --------------------------------------------------------------------------

def build_alignment_proof(manifest: dict, out_root: str, fps: int) -> dict:
    """Produces the evidence consumed by `--score-baselines`.

    Geometry is checked before metric invocation, self-PSNR must be infinite,
    and the known-filter reconstruction must be finite. Color range evidence
    is deliberately independent of image-content extrema: each real encoded
    stream must advertise the BT.709/TV contract, and the one-frame color
    calibration for that tier must pass.
    """
    proof = {}
    for category, cat in manifest["categories"].items():
        master_dir = os.path.join(out_root, category, "master")
        lr_dir = os.path.join(out_root, category, "lr_source")
        frame_count = len(cat["frames"])
        master_sequence = inspect_png_sequence(master_dir, frame_count)
        lr_sequence = inspect_png_sequence(lr_dir, frame_count)
        master_geometry_ok = sequence_contract(master_sequence, (MASTER_W, MASTER_H))
        lr_geometry_ok = sequence_contract(lr_sequence, (LR_W, LR_H))
        if not master_geometry_ok or not lr_geometry_ok:
            raise RuntimeError(f"{category}: master/LR sequence geometry or order failed validation")

        self_psnr, _ = compare_sequences_psnr_ssim(master_dir, master_dir, frame_count, fps)
        self_ok = len(self_psnr) == frame_count and all(value == float("inf") for value in self_psnr)
        if not self_ok:
            raise RuntimeError(f"{category}: reference compared to itself did not yield infinite PSNR")

        tmp_recon = tempfile.mkdtemp(prefix="vb_align_")
        try:
            for i in range(frame_count):
                with Image.open(os.path.join(lr_dir, f"frame_{i:04d}.png")) as lr_file:
                    reconstruction = lr_file.convert("RGB").resize((MASTER_W, MASTER_H), Image.BICUBIC)
                reconstruction.save(os.path.join(tmp_recon, f"frame_{i:04d}.png"), "PNG")
            recon_sequence = inspect_png_sequence(tmp_recon, frame_count)
            recon_geometry_ok = sequence_contract(recon_sequence, (MASTER_W, MASTER_H))
            if not recon_geometry_ok:
                raise RuntimeError(f"{category}: known-filter reconstruction geometry failed validation")
            known_psnr, known_ssim = compare_sequences_psnr_ssim(tmp_recon, master_dir, frame_count, fps)
        finally:
            shutil.rmtree(tmp_recon, ignore_errors=True)

        known_filter_ok = all(math.isfinite(value) and value > 0.0 for value in known_psnr + known_ssim)
        if not known_filter_ok:
            raise RuntimeError(f"{category}: known bicubic filter check was not finite and positive")

        master_min, master_max = frame_range_stats(master_dir, frame_count)
        per_tier = {}
        for tier_name, enc in cat["encodes"].items():
            dec_dir = os.path.join(out_root, category, "decoded", tier_name)
            decoded_sequence = inspect_png_sequence(dec_dir, frame_count)
            decoded_geometry_ok = sequence_contract(decoded_sequence, (LR_W, LR_H))
            dmin, dmax = frame_range_stats(dec_dir, frame_count)
            calibration = manifest["colour_calibration"][tier_name]
            colour_range_ok = bool(enc["stream_contract"]["ok"] and calibration["passed"])
            per_tier[tier_name] = {
                "geometry_ok": bool(master_geometry_ok and lr_geometry_ok and decoded_geometry_ok),
                "frame_count_ok": decoded_sequence["actual_frame_count"] == frame_count,
                "decoded_sequence": decoded_sequence,
                "decoded_pixel_range": {"min": dmin, "max": dmax},
                "encoded_stream_contract_ok": enc["stream_contract"]["ok"],
                "colour_calibration_tier": tier_name,
                "colour_range_ok": colour_range_ok,
            }

        proof[category] = {
            "sequence_geometry": {
                "master": master_sequence,
                "lr_source": lr_sequence,
                "geometry_ok": bool(master_geometry_ok and lr_geometry_ok),
            },
            "self_comparison": {
                "description": (
                    "The reference master sequence compared against itself. "
                    "Every frame must be infinite PSNR; anything else is an "
                    "order, geometry or pixel-conversion error."
                ),
                "all_infinite": True,
                "psnr_db_sample": ["inf" for _ in self_psnr[:3]],
            },
            "known_filter_check": {
                "description": (
                    "Pristine pre-codec 720p upscaled 2x with PIL bicubic and "
                    "compared to its 1440p master. Exact frame naming and "
                    "geometry are checked first; the finite positive score is "
                    "a separate sanity check that the comparison is aligned."
                ),
                "filter": "PIL.Image.BICUBIC",
                "reconstruction_sequence": recon_sequence,
                "geometry_ok": recon_geometry_ok,
                "finite_positive_metrics": known_filter_ok,
                "psnr_db": distribution(known_psnr),
                "ssim": distribution(known_ssim),
            },
            "master_pixel_range": {"min": master_min, "max": master_max},
            "per_tier": per_tier,
        }
    return proof


def prepare(args: argparse.Namespace) -> dict:
    if args.frames <= 0:
        raise SystemExit("--frames must be positive")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")

    out_root = args.out
    os.makedirs(out_root, exist_ok=True)
    baseline_path = os.path.join(out_root, "baseline_scores.json")
    if os.path.exists(baseline_path):
        os.remove(baseline_path)

    print("verifying explicit RGB/full <-> BT.709/limited codec color contract ...", file=sys.stderr)
    calibrations = {}
    for tier in TIERS:
        calibrations[tier["name"]] = colour_calibration(tier, args.fps)

    manifest = {
        "format_version": 1,
        "seed": args.seed,
        "master": {"width": MASTER_W, "height": MASTER_H, "format": "RGB PNG, full range"},
        "lr": {"width": LR_W, "height": LR_H, "format": "RGB PNG, full range"},
        "fps": args.fps,
        "frame_count": args.frames,
        "downsample_filter": (
            "exact 2x2 block mean (the spatial operation in "
            "tools/aethersr.py's box_downsample2 / torch "
            "avg_pool2d(kernel_size=2,stride=2)), followed only by "
            "deterministic NumPy nearest-even rounding to RGB8 because PNG "
            "cannot store fractional channel values."
        ),
        "colour_contract": {
            "source_and_decoded_png": "RGB24, full/PC range",
            "encoded_video": "yuv420p, BT.709, limited/TV range",
            "encode_filter": RGB_PC_TO_YUV709_TV_FILTER,
            "decode_filter": YUV709_TV_TO_RGB_PC_FILTER,
            "vmaf_input_filter": VMAF_RGB_PC_TO_YUV709_TV_FILTER,
            "expected_stream_metadata": EXPECTED_VIDEO_COLOUR,
        },
        "ffmpeg": {
            "path": FFMPEG,
            "ffprobe_path": FFPROBE,
            "version_output": ffmpeg_version_output(),
        },
        "tiers": [
            {
                "name": tier["name"],
                "codec": tier["codec"],
                "stream_codec": tier["stream_codec"],
                "container": tier["container"],
                "args": tier["args"],
            }
            for tier in TIERS
        ],
        "colour_calibration": calibrations,
        "categories": {},
    }

    for category in dict.fromkeys(args.categories):
        cat_dir = os.path.join(out_root, category)
        # Removing only the category we are about to regenerate prevents stale
        # higher frame indices from silently entering an image2 sequence.
        shutil.rmtree(cat_dir, ignore_errors=True)
        master_dir = os.path.join(cat_dir, "master")
        lr_dir = os.path.join(cat_dir, "lr_source")
        os.makedirs(master_dir)
        os.makedirs(lr_dir)

        print(f"[{category}] generating {args.frames} frames ...", file=sys.stderr)
        frames = generate_category_frames(category, args.seed, args.frames)

        frame_entries = []
        for i, frame in enumerate(frames):
            mpath = os.path.join(master_dir, f"frame_{i:04d}.png")
            Image.fromarray(frame, "RGB").save(mpath, "PNG", compress_level=9)
            lr = box_downsample2(frame)
            lpath = os.path.join(lr_dir, f"frame_{i:04d}.png")
            Image.fromarray(lr, "RGB").save(lpath, "PNG", compress_level=9)
            frame_entries.append({
                "index": i,
                "master_file": os.path.relpath(mpath, out_root),
                "master_sha256": sha256_file(mpath),
                "lr_source_file": os.path.relpath(lpath, out_root),
                "lr_source_sha256": sha256_file(lpath),
            })

        master_sequence = inspect_png_sequence(master_dir, args.frames)
        lr_sequence = inspect_png_sequence(lr_dir, args.frames)
        if not sequence_contract(master_sequence, (MASTER_W, MASTER_H)):
            raise RuntimeError(f"{category}: generated master sequence failed geometry/order validation")
        if not sequence_contract(lr_sequence, (LR_W, LR_H)):
            raise RuntimeError(f"{category}: generated LR sequence failed geometry/order validation")

        enc_dir = os.path.join(cat_dir, "encoded")
        os.makedirs(enc_dir)
        encodes = {}
        for tier in TIERS:
            video_path = os.path.join(enc_dir, f"{tier['name']}.{tier['container']}")
            encode_args = encode_sequence_args(lr_dir, tier, args.fps, video_path)
            print(f"[{category}] encoding {tier['name']} ({tier['codec']}) ...", file=sys.stderr)
            ffmpeg_run(encode_args)

            dec_dir = os.path.join(cat_dir, "decoded", tier["name"])
            os.makedirs(dec_dir)
            ffmpeg_run(decode_sequence_args(video_path, dec_dir))
            decoded_sequence = inspect_png_sequence(dec_dir, args.frames)
            if not sequence_contract(decoded_sequence, (LR_W, LR_H)):
                raise RuntimeError(f"{category}/{tier['name']}: decoded sequence failed geometry/order validation")
            stream = ffprobe_stream(video_path)
            dec_entries = [
                {
                    "file": os.path.relpath(os.path.join(dec_dir, name), out_root),
                    "sha256": sha256_file(os.path.join(dec_dir, name)),
                }
                for name in expected_frame_names(args.frames)
            ]
            encodes[tier["name"]] = {
                "codec": tier["codec"],
                "container": tier["container"],
                "ffmpeg_args": encode_args,
                "video_file": os.path.relpath(video_path, out_root),
                "video_sha256": sha256_file(video_path),
                "video_size_bytes": os.path.getsize(video_path),
                "stream": stream,
                "stream_contract": stream_contract(tier, stream),
                "decoded_sequence": decoded_sequence,
                "decoded_frames": dec_entries,
            }

        manifest["categories"][category] = {
            "description": CATEGORY_DESCRIPTIONS[category],
            "synthetic": True,
            "master_sequence": master_sequence,
            "lr_source_sequence": lr_sequence,
            "frames": frame_entries,
            "encodes": encodes,
        }

    print(
        "verifying alignment (exact sequence geometry, self-PSNR, known-filter check, "
        "encoded stream metadata and calibrated colour range) ...",
        file=sys.stderr,
    )
    manifest["alignment_proof"] = build_alignment_proof(manifest, out_root, args.fps)

    manifest_path = os.path.join(out_root, "manifest.json")
    write_json(manifest_path, manifest)
    print(f"prepared {len(manifest['categories'])} categories -> {manifest_path}", file=sys.stderr)
    return manifest


# --------------------------------------------------------------------------
# --score-baselines
# --------------------------------------------------------------------------

def score_baselines(args: argparse.Namespace) -> dict:
    out_root = args.out
    manifest_path = os.path.join(out_root, "manifest.json")
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    fps = manifest["fps"]
    alignment = manifest["alignment_proof"]

    result = {
        "format_version": manifest["format_version"],
        "ffmpeg": manifest["ffmpeg"],
        "seed": manifest["seed"],
        "master": manifest["master"],
        "lr": manifest["lr"],
        "fps": fps,
        "frame_count": manifest["frame_count"],
        "colour_contract": manifest["colour_contract"],
        "colour_calibration": manifest["colour_calibration"],
        "alignment_proof": alignment,
        "scalers": list(SCALERS.keys()),
        "categories": {},
    }

    for category, cat in manifest["categories"].items():
        frame_count = len(cat["frames"])
        master_dir = os.path.join(out_root, category, "master")
        cat_result = {}
        for tier_name in cat["encodes"]:
            dec_dir = os.path.join(out_root, category, "decoded", tier_name)
            tier_proof = alignment[category]["per_tier"][tier_name]
            vmaf_reasons = []
            if not alignment[category]["sequence_geometry"]["geometry_ok"]:
                vmaf_reasons.append("master/LR source geometry or frame order failed alignment proof")
            if not tier_proof["geometry_ok"]:
                vmaf_reasons.append("decoded LR geometry or frame order failed alignment proof")
            if not tier_proof["colour_range_ok"]:
                vmaf_reasons.append(
                    "encoded-stream BT.709/limited metadata or independent calibrated range round-trip failed"
                )
            tier_result = {}
            for scaler_name, resample in SCALERS.items():
                tmp = tempfile.mkdtemp(prefix="vb_recon_")
                try:
                    for i in range(frame_count):
                        with Image.open(os.path.join(dec_dir, f"frame_{i:04d}.png")) as lr_file:
                            reconstruction = lr_file.convert("RGB").resize((MASTER_W, MASTER_H), resample)
                        reconstruction.save(os.path.join(tmp, f"frame_{i:04d}.png"), "PNG")
                    reconstruction_sequence = inspect_png_sequence(tmp, frame_count)
                    reconstruction_geometry_ok = sequence_contract(
                        reconstruction_sequence,
                        (MASTER_W, MASTER_H),
                    )
                    if not reconstruction_geometry_ok:
                        raise RuntimeError(
                            f"{category}/{tier_name}/{scaler_name}: reconstruction geometry/order validation failed"
                        )
                    psnr_vals, ssim_vals = compare_sequences_psnr_ssim(tmp, master_dir, frame_count, fps)
                    entry = {
                        "reconstruction_sequence": reconstruction_sequence,
                        "psnr_db": distribution(psnr_vals),
                        "ssim": distribution(ssim_vals),
                    }
                    if not vmaf_reasons:
                        entry["vmaf"] = distribution(compute_vmaf(tmp, master_dir, frame_count, fps))
                    else:
                        entry["vmaf"] = {
                            "unusable": True,
                            "reason": "; ".join(vmaf_reasons) + ". VMAF is intentionally withheld.",
                        }
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
                tier_result[scaler_name] = entry
                print(f"[{category}/{tier_name}/{scaler_name}] psnr_mean={entry['psnr_db']['mean']}", file=sys.stderr)
            cat_result[tier_name] = tier_result
        result["categories"][category] = cat_result

    out_path = os.path.join(out_root, "baseline_scores.json")
    write_json(out_path, result)

    if args.json:
        print(json_text(result))
    else:
        print_human_summary(result)
    return result


def print_human_summary(result: dict) -> None:
    for category, tiers in result["categories"].items():
        for tier_name, scalers in tiers.items():
            for scaler_name, entry in scalers.items():
                psnr = entry["psnr_db"]
                ssim = entry["ssim"]
                vmaf = entry.get("vmaf", {})
                vmaf_str = "unusable" if vmaf.get("unusable") else f"{vmaf.get('mean'):.2f}"
                print(
                    f"{category:10s} {tier_name:12s} {scaler_name:9s} "
                    f"psnr={psnr['mean']:.2f}dB (p5={psnr['p5']:.2f} p95={psnr['p95']:.2f}) "
                    f"ssim={ssim['mean']:.4f} vmaf={vmaf_str}"
                )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true", help="generate the corpus (masters, LR source, encodes, decodes, manifest)")
    ap.add_argument("--score-baselines", action="store_true", help="score conventional upscalers against the prepared corpus")
    ap.add_argument("--json", action="store_true", help="print --score-baselines output as JSON")
    ap.add_argument("--out", default="data/video", help="corpus root (default data/video)")
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=list(CATEGORIES))
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    if not args.prepare and not args.score_baselines:
        ap.error("specify --prepare and/or --score-baselines")

    if args.prepare:
        prepare(args)
    if args.score_baselines:
        score_baselines(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
