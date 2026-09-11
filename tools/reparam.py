#!/usr/bin/env python3
"""Training-only structural reparameterization for the AetherVSR body layers.

The deployed graph never changes. Every block here is a sum of *linear* branches
evaluated before the existing `tanh`, so the whole sum collapses algebraically
into the single `nn.Conv2d` the WGSL shader already implements. Training sees a
wider parameterization; inference sees the identical 3x3 convolution it always
did, with the identical tensor names, shapes and parameter count.

What this can and cannot do
---------------------------
A sum of linear branches before a nonlinearity *is* one convolution. The fused
function class is therefore exactly what a plain 3x3 could already express, and
nothing here adds inference capacity. The hypothesis under test is purely about
optimization: whether an over-parameterized but algebraically equivalent
training graph lands on better weights in the same space. Claiming anything
stronger would be false.

Fusion mathematics
------------------
All branches share stride 1, `padding = kernel // 2` and zero padding, so each
one is a 3x3 convolution with most taps pinned to zero. Summing convolutions
that share input, stride and padding is the same as convolving with the summed
kernel, so for a 3x3 target:

    W_fused = W_3x3
            + embed(W_1x1)          zeros except the centre tap
            + embed(W_1x3)          zeros except the centre row
            + embed(W_3x1)          zeros except the centre column
            + delta                 identity: 1 at [c, c, 1, 1]

    b_fused = b_3x3 + b_1x1 + b_1x3 + b_3x1        (identity carries no bias)

`embed` places a smaller kernel in the centre of the 3x3 window, which is exact
because each branch is padded to preserve resolution: a 1x1 with padding 0, a
1x3 with padding (0,1) and a 3x1 with padding (1,0) all produce the same output
grid as a 3x3 with padding 1, and each output pixel is the same linear
combination the embedded kernel describes. Border pixels included: the zero
padding the small branches see is the same zero padding the fused 3x3 sees at
the taps where the embedded kernel is non-zero.

Deliberately excluded
---------------------
A *sequential* 1x1 -> 3x3 branch is not offered. Composing them looks like a
kernel product, but the 3x3 reads zero-padded borders while the true composition
would need the first branch's bias there instead. The interior matches and the
one-pixel border does not, so it is not exact reparameterization and it is not
implemented. BatchNorm is likewise absent from the primary ladder; folding it is
exact only once its statistics are frozen, and it belongs to a separate,
isolated experiment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Branch sets for each rung of the pre-registered ladder (docs/M7-PREREGISTRATION.md).
LADDERS: dict[str, tuple[str, ...]] = {
    "R0": ("k3",),
    "R1": ("k3", "k1"),
    "R2": ("k3", "k1", "id"),
    "R3": ("k3", "k1", "id", "k1x3", "k3x1"),
}


def embed_into_3x3(kernel: torch.Tensor) -> torch.Tensor:
    """Place a 1x1, 1x3, 3x1 or 3x3 kernel in the centre of a 3x3 window."""
    out_c, in_c, kh, kw = kernel.shape
    if (kh, kw) == (3, 3):
        return kernel
    if kh not in (1, 3) or kw not in (1, 3):
        raise ValueError(f"cannot embed kernel of shape {tuple(kernel.shape)} into 3x3")
    padded = kernel.new_zeros((out_c, in_c, 3, 3))
    row = slice(1, 2) if kh == 1 else slice(0, 3)
    col = slice(1, 2) if kw == 1 else slice(0, 3)
    padded[:, :, row, col] = kernel
    return padded


def identity_delta(channels: int, ref: torch.Tensor) -> torch.Tensor:
    """The 3x3 kernel that reproduces its input: 1 at the centre tap of each diagonal."""
    delta = ref.new_zeros((channels, channels, 3, 3))
    idx = torch.arange(channels, device=ref.device)
    delta[idx, idx, 1, 1] = 1.0
    return delta


class RepBody(nn.Module):
    """A body layer whose training graph is a sum of linear branches.

    Fuses to exactly one 3x3 `nn.Conv2d`. The activation is applied by the
    caller, after the sum, never inside a branch.
    """

    def __init__(self, channels: int, branches: tuple[str, ...] = LADDERS["R2"]) -> None:
        super().__init__()
        unknown = set(branches) - {"k3", "k1", "id", "k1x3", "k3x1"}
        if unknown:
            raise ValueError(f"unknown branches {sorted(unknown)}")
        if "k3" not in branches:
            raise ValueError("the 3x3 branch is required; it carries the receptive field")
        self.channels = channels
        self.branches = tuple(branches)

        self.k3 = nn.Conv2d(channels, channels, 3, padding=1)
        # Each optional branch keeps resolution, so its padding is kernel // 2 per axis.
        self.k1 = nn.Conv2d(channels, channels, 1, padding=0) if "k1" in branches else None
        self.k1x3 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1)) if "k1x3" in branches else None
        self.k3x1 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0)) if "k3x1" in branches else None
        self.use_identity = "id" in branches

        # Start every added branch at zero so the block begins numerically identical
        # to the plain convolution it replaces. Without this the extra branches
        # perturb the effective initialisation, and a quality difference could be
        # an initialisation difference rather than an optimization one.
        for extra in (self.k1, self.k1x3, self.k3x1):
            if extra is not None:
                nn.init.zeros_(extra.weight)
                nn.init.zeros_(extra.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.k3(x)
        if self.k1 is not None:
            y = y + self.k1(x)
        if self.k1x3 is not None:
            y = y + self.k1x3(x)
        if self.k3x1 is not None:
            y = y + self.k3x1(x)
        if self.use_identity:
            y = y + x
        return y

    @torch.no_grad()
    def fused_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The single 3x3 weight and bias equivalent to the whole branch sum."""
        weight = self.k3.weight.detach().clone()
        bias = self.k3.bias.detach().clone()
        for extra in (self.k1, self.k1x3, self.k3x1):
            if extra is not None:
                weight = weight + embed_into_3x3(extra.weight.detach())
                bias = bias + extra.bias.detach()
        if self.use_identity:
            weight = weight + identity_delta(self.channels, weight)
        return weight, bias

    @torch.no_grad()
    def to_conv(self) -> nn.Conv2d:
        """Materialise the fused convolution, ready to drop into `AetherSR.body`.

        Built in the block's own dtype and device. Hard-coding float32 here would
        silently round the fused weights when the block is in float64, which
        makes an exact-arithmetic equivalence test measure float32 rounding
        instead of the algebra it is supposed to be checking.
        """
        weight, bias = self.fused_parameters()
        conv = nn.Conv2d(self.channels, self.channels, 3, padding=1).to(
            dtype=weight.dtype, device=weight.device
        )
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)
        return conv

    def training_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def fused_parameter_count(self) -> int:
        c = self.channels
        return c * c * 9 + c


class RepAetherSR(nn.Module):
    """`AetherSR` with reparameterized body layers and an unchanged stem and head.

    `fuse()` returns an ordinary `AetherSR` whose weights are the algebraic sum
    of the branches, so training-only structure never reaches the exported model
    or the runtime.
    """

    def __init__(
        self,
        channels: int = 16,
        depth: int = 2,
        scale: int = 2,
        branches: tuple[str, ...] = LADDERS["R2"],
    ) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError(f"scale 2 only, got {scale}")
        self.channels = channels
        self.depth = depth
        self.scale = scale
        self.branches = tuple(branches)

        self.stem = nn.Conv2d(3, channels, kernel_size=5, padding=2)
        self.body = nn.ModuleList([RepBody(channels, branches) for _ in range(depth)])
        self.head = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        y = torch.tanh(self.stem(x))
        for block in self.body:
            y = torch.tanh(block(y))
        y = F.interpolate(y, scale_factor=self.scale, mode="nearest")
        return (self.head(y) + skip).clamp(0.0, 1.0)

    @torch.no_grad()
    def fuse(self):  # -> AetherSR
        """Collapse to the deployed architecture. Import is local to avoid a cycle."""
        from aethersr import AetherSR

        # Build the destination on the source's device and dtype. `copy_` casts
        # the values it writes but never moves the module it writes into, so a
        # default-constructed AetherSR would hand back a CPU float32 network no
        # matter what was fused - silently downcasting a float64 equivalence
        # check, or dragging an MPS model back to the CPU mid-run.
        ref = next(self.parameters())
        fused = AetherSR(channels=self.channels, depth=self.depth, scale=self.scale).to(
            device=ref.device, dtype=ref.dtype
        )
        fused.stem.weight.copy_(self.stem.weight)
        fused.stem.bias.copy_(self.stem.bias)
        for block, conv in zip(self.body, fused.body):
            weight, bias = block.fused_parameters()
            conv.weight.copy_(weight)
            conv.bias.copy_(bias)
        fused.head.weight.copy_(self.head.weight)
        fused.head.bias.copy_(self.head.bias)
        return fused.eval()

    def training_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def fused_parameter_count(self) -> int:
        c = self.channels
        return (3 * c * 25 + c) + self.depth * (c * c * 9 + c) + (3 * c * 9 + 3)
