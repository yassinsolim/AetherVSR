#!/usr/bin/env python3
"""Reference implementation of the AetherVSR Milestone 4 network.

This is the trusted definition. The WGSL graph is checked against *this*, and
where they disagree this one is right until proven otherwise.

Architecture (ADR-0022), 2x, RGB:

    5x5 conv  3 -> C, tanh          padding 2
    D x (3x3 conv C -> C, tanh)     padding 1
    nearest 2x upsample
    3x3 conv  C -> 3                padding 1, evaluated at high resolution
    + nearest 2x upsample of the input          <- global residual
    clamp to [0,1]

The global residual makes the network learn the difference from a
nearest-upsampled image rather than the image itself, which is standard
super-resolution practice and, measured here, worth far more than its cost: the
head already runs at high resolution so it is one extra fetch per output pixel.
Nearest rather than bilinear because it is exactly reproducible on both sides
and exactly shift-equivariant in whole low-resolution pixels.

The head is a resize-convolution rather than a sub-pixel convolution: no
`r^2 * C` channel group is ever formed and no periodic shuffle happens. See
ADR-0022 for why, including the freedom-to-operate question that motivates it.

The training-to-inference contract, frozen before training:

  * HR is the source image at native resolution, cropped to even dimensions.
  * LR is produced by an exact 2x box downsample, matching `boxDownsample2` in
    `src/bench/quality.ts`. A different downsampler produces a model that is
    numerically fine and quality-wrong on our own evaluation.
  * RGB in [0,1]. No dataset mean is subtracted; the shader's mean/scale are
    identity.
  * Zero padding sized to keep every layer same-resolution.
  * Output clamped to [0,1], matching the shader's write.
  * Channel order RGB throughout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AetherSR(nn.Module):
    """The network exactly as the WGSL graph implements it."""

    def __init__(self, channels: int = 16, depth: int = 2, scale: int = 2) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError(f"scale 2 only, got {scale}")
        self.channels = channels
        self.depth = depth
        self.scale = scale

        self.stem = nn.Conv2d(3, channels, kernel_size=5, padding=2)
        self.body = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, padding=1) for _ in range(depth)]
        )
        # Evaluated after a nearest upsample, so it sees high-resolution input.
        self.head = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        y = torch.tanh(self.stem(x))
        for conv in self.body:
            y = torch.tanh(conv(y))
        y = F.interpolate(y, scale_factor=self.scale, mode="nearest")
        return (self.head(y) + skip).clamp(0.0, 1.0)

    def macs(self, width: int, height: int) -> int:
        """Multiply-accumulates for one forward pass at this input size."""
        c = self.channels
        lr = width * height
        hr = lr * self.scale * self.scale
        return lr * (25 * 3 * c) + self.depth * lr * (9 * c * c) + hr * (9 * c * 3)


def box_downsample2(hr: torch.Tensor) -> torch.Tensor:
    """Exact 2x box downsample.

    Must match `boxDownsample2` in `src/bench/quality.ts`, which is what the
    project's own quality evaluation uses to build its low-resolution input. If
    these disagree, the model is trained to invert a degradation that the
    evaluation never applies.
    """
    return F.avg_pool2d(hr, kernel_size=2, stride=2)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def export_weights(model: AetherSR) -> dict[str, list[float]]:
    """Flatten to the planar layouts the WGSL packers expect.

    * stem: `[oc][ic][tap]`, matching `packStemWeights`
    * body: `[oc][ic][tap]`, matching `packWeights` then `toTapMajorWeights`
    * head: `[colour][ic][tap]`, matching `packUpsampleHeadWeights`

    Packing into the GPU's vec4 / tap-major order happens once at model load in
    TypeScript, never per frame, and never here — this file only has to agree on
    what "planar" means.
    """
    out: dict[str, list[float]] = {}
    out["stem.weight"] = model.stem.weight.detach().cpu().reshape(-1).tolist()
    out["stem.bias"] = model.stem.bias.detach().cpu().reshape(-1).tolist()
    for i, conv in enumerate(model.body):
        out[f"body.{i}.weight"] = conv.weight.detach().cpu().reshape(-1).tolist()
        out[f"body.{i}.bias"] = conv.bias.detach().cpu().reshape(-1).tolist()
    out["head.weight"] = model.head.weight.detach().cpu().reshape(-1).tolist()
    out["head.bias"] = model.head.bias.detach().cpu().reshape(-1).tolist()
    return out
