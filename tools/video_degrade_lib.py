"""Shared patch-harvesting helpers.

Extracted from tools/video-degrade.py so tools/corpus-prepare.py can reuse the
exact same cropping and seeding rules. Two pipelines cutting patches two
slightly different ways would make v1 and v2 corpora incomparable, which is the
whole point of the scaling curve.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import torch
from PIL import Image


def rng_for(*parts: object) -> np.random.Generator:
    digest = hashlib.sha256("/".join(str(p) for p in parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def read_u8(path: str) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        raw = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        return raw.reshape(h, w, 3).permute(2, 0, 1)


def harvest_patches(
    hr_dir: str, lr_dir: str, sink: dict[str, list[torch.Tensor]],
    hr_patch: int, per_frame: int, gen: np.random.Generator,
) -> int:
    """Cut aligned (LR, HR) patch pairs from a decoded sequence.

    LR is cropped at even offsets only: an odd offset pairs an LR patch with an
    HR patch at a half-pixel phase the 2x geometry cannot express.
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
            x = int(gen.integers(0, (lw - lr_patch) // 2 + 1)) * 2
            y = int(gen.integers(0, (lh - lr_patch) // 2 + 1)) * 2
            sink["lr"].append(lr[:, y:y + lr_patch, x:x + lr_patch].clone())
            sink["hr"].append(hr[:, 2 * y:2 * y + hr_patch, 2 * x:2 * x + hr_patch].clone())
            written += 1
    return written
