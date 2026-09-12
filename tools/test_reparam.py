#!/usr/bin/env python3
"""Fusion equivalence gate for the training-only reparameterized body.

These tests are the whole safety argument of Milestone 7. If a branch sum does
not collapse *exactly* into the deployed convolution, then the model we measure
is not the model we ship, and every quality number afterwards is meaningless.

They are permanent, not scaffolding: a future change to the block, the branch
set or the export path can silently break deploy equivalence, and this is what
catches it. Border pixels are never cropped, because the border is precisely
where an incorrect embedding or a mishandled padding shows up first.

Run: /tmp/aethertrain-new/bin/python -m pytest tools/test_reparam.py -q
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aethersr import AetherSR, count_parameters, export_weights  # noqa: E402
from reparam import (  # noqa: E402
    LADDERS,
    RepAetherSR,
    RepBody,
    embed_into_3x3,
    identity_delta,
)

# Fusion is an algebraic identity, so it should hold *exactly* in exact
# arithmetic. Testing that separately from float32 rounding is what
# distinguishes "the maths is right" from "the numbers look close":
#
#   FP64_TOL   the algebra itself. Any failure here is a real bug.
#   TOL        float32 at realistic trained-weight scale (std ~0.1), the
#              pre-registered gate from docs/M7-PREREGISTRATION.md.
#   REL_TOL    float32 at absurd weight scale (std 1.0), where a 16-channel
#              3x3 accumulates ~144 products of order 1 and absolute error
#              reaches a few times 1e-5 purely from rounding. Only the
#              relative error is meaningful there.
FP64_TOL = 1e-12
TOL = 1e-5
REL_TOL = 1e-6
SEEDS = (0, 1, 2, 3, 4)
SHAPES = ((1, 16, 16, 16), (2, 16, 9, 13), (1, 16, 5, 5), (3, 16, 32, 24))

# Trained body weights in this project sit near this scale; see the shipped
# model's tensors. Tests use it so the tolerance describes the real regime.
REALISTIC_SCALE = 0.1


def _randomise(block: RepBody, generator: torch.Generator, scale: float = REALISTIC_SCALE) -> None:
    """Give every branch non-trivial weights.

    The block deliberately initialises its added branches to zero so training
    starts as close to a plain convolution as the branch set allows. That would
    make fusion trivially correct, so the tests overwrite them: fusion has to
    hold for arbitrary weights, not just the ones we happen to start from.
    """
    for param in block.parameters():
        param.data = torch.randn(param.shape, generator=generator) * scale




@pytest.mark.parametrize("rung", sorted(LADDERS))
@pytest.mark.parametrize("seed", SEEDS)
def test_block_fusion_is_algebraically_exact_in_float64(rung: str, seed: int) -> None:
    """The identity itself, free of float32 rounding."""
    gen = torch.Generator().manual_seed(seed)
    block = RepBody(16, LADDERS[rung]).eval()
    _randomise(block, gen, scale=1.0)
    # Cast BEFORE fusing: fusing first would sum the branch kernels in float32
    # and the comparison would measure that rounding rather than the algebra.
    block = block.double()
    conv = block.to_conv().eval()

    for shape in SHAPES:
        x = torch.randn(shape, generator=gen, dtype=torch.float64)
        with torch.no_grad():
            err = (block(x) - conv(x)).abs().max().item()
        assert err <= FP64_TOL, f"{rung} seed {seed} shape {shape}: fp64 err {err:.3e}"


@pytest.mark.parametrize("rung", sorted(LADDERS))
@pytest.mark.parametrize("seed", SEEDS)
def test_block_fusion_matches_before_activation(rung: str, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    block = RepBody(16, LADDERS[rung]).eval()
    _randomise(block, gen)
    conv = block.to_conv().eval()

    for shape in SHAPES:
        x = torch.randn(shape, generator=gen)
        with torch.no_grad():
            multi, fused = block(x), conv(x)
        err = (multi - fused).abs().max().item()
        assert err <= TOL, f"{rung} seed {seed} shape {shape}: pre-tanh max err {err:.3e}"


def test_border_pixels_are_included_and_correct() -> None:
    """The interior can match while the border does not; check the border alone."""
    gen = torch.Generator().manual_seed(5)
    block = RepBody(16, LADDERS["R3"]).eval()
    _randomise(block, gen)
    conv = block.to_conv().eval()
    x = torch.randn((1, 16, 12, 12), generator=gen)
    with torch.no_grad():
        out = conv(x)
        diff = (block(x) - out).abs()
    border = torch.cat(
        [diff[:, :, 0, :].reshape(-1), diff[:, :, -1, :].reshape(-1),
         diff[:, :, :, 0].reshape(-1), diff[:, :, :, -1].reshape(-1)]
    )
    rel = (border.max() / out.abs().max().clamp_min(1e-12)).item()
    assert border.max().item() <= TOL, f"border abs err {border.max().item():.3e}"
    assert rel <= REL_TOL, f"border rel err {rel:.3e}"


def test_one_by_one_embeds_at_the_centre_tap_only() -> None:
    k = torch.randn(4, 4, 1, 1)
    e = embed_into_3x3(k)
    assert e.shape == (4, 4, 3, 3)
    assert torch.equal(e[:, :, 1:2, 1:2], k)
    mask = torch.ones(3, 3, dtype=torch.bool)
    mask[1, 1] = False
    assert torch.count_nonzero(e[:, :, mask]) == 0


def test_asymmetric_kernels_embed_on_centre_row_and_column() -> None:
    row = embed_into_3x3(torch.randn(4, 4, 1, 3))
    col = embed_into_3x3(torch.randn(4, 4, 3, 1))
    assert torch.count_nonzero(row[:, :, 0, :]) == 0 and torch.count_nonzero(row[:, :, 2, :]) == 0
    assert torch.count_nonzero(col[:, :, :, 0]) == 0 and torch.count_nonzero(col[:, :, :, 2]) == 0


def test_identity_branch_is_a_centred_delta_kernel() -> None:
    delta = identity_delta(6, torch.zeros(1))
    assert delta.shape == (6, 6, 3, 3)
    assert torch.equal(delta[:, :, 1, 1], torch.eye(6))
    assert delta.sum().item() == pytest.approx(6.0)
    x = torch.randn(2, 6, 7, 7)
    assert torch.allclose(torch.nn.functional.conv2d(x, delta, padding=1), x, atol=TOL)


def test_branch_biases_sum_and_identity_contributes_none() -> None:
    gen = torch.Generator().manual_seed(7)
    block = RepBody(8, LADDERS["R3"]).eval()
    _randomise(block, gen)
    _, fused_bias = block.fused_parameters()
    expected = block.k3.bias + block.k1.bias + block.k1x3.bias + block.k3x1.bias
    assert torch.allclose(fused_bias, expected, atol=TOL)


def test_hostile_channel_counts_still_fuse() -> None:
    gen = torch.Generator().manual_seed(9)
    for channels in (1, 2, 3, 7, 33):
        block = RepBody(channels, LADDERS["R3"]).eval()
        _randomise(block, gen)
        conv = block.to_conv().eval()
        x = torch.randn((1, channels, 6, 6), generator=gen)
        with torch.no_grad():
            out = conv(x)
            err = (block(x) - out).abs().max().item()
        rel = err / max(out.abs().max().item(), 1e-12)
        assert err <= TOL, f"channels {channels}: abs err {err:.3e}"
        assert rel <= REL_TOL, f"channels {channels}: rel err {rel:.3e}"


@pytest.mark.parametrize("rung", sorted(LADDERS))
@pytest.mark.parametrize("seed", (0, 1, 2))
def test_full_network_fusion_is_exact(rung: str, seed: int) -> None:
    torch.manual_seed(seed)
    rep = RepAetherSR(16, 2, branches=LADDERS[rung]).eval()
    for p in rep.parameters():
        p.data = torch.randn(p.shape) * 0.1
    fused = rep.fuse()
    x = torch.rand(2, 3, 24, 32)
    with torch.no_grad():
        err = (rep(x) - fused(x)).abs().max().item()
    assert err <= TOL, f"{rung} seed {seed}: network max err {err:.3e}"


@pytest.mark.parametrize("rung", sorted(LADDERS))
def test_fused_model_matches_the_deployed_contract(rung: str) -> None:
    """Whatever the training graph, the shipped artefact must be indistinguishable."""
    torch.manual_seed(3)
    rep = RepAetherSR(16, 2, branches=LADDERS[rung])
    fused = rep.fuse()
    reference = AetherSR(16, 2)

    assert count_parameters(fused) == count_parameters(reference) == 6291
    assert rep.fused_parameter_count() == 6291
    exported, expected = export_weights(fused), export_weights(reference)
    assert sorted(exported) == sorted(expected)
    for name in expected:
        assert len(exported[name]) == len(expected[name]), name


def test_training_graph_is_wider_than_the_fused_graph() -> None:
    """The point of the milestone: training parameters exceed inference parameters."""
    counts = {r: RepAetherSR(16, 2, branches=LADDERS[r]).training_parameter_count() for r in LADDERS}
    assert counts["R0"] == 6291
    assert counts["R1"] > counts["R0"] and counts["R2"] == counts["R1"]
    assert counts["R3"] > counts["R2"]
    for rung in LADDERS:
        assert RepAetherSR(16, 2, branches=LADDERS[rung]).fused_parameter_count() == 6291


def test_added_branches_start_as_intended() -> None:
    """Learnable added branches start at zero, so they perturb nothing initially.

    The identity branch is the deliberate exception: it is a fixed `+x` with no
    parameters, so an identity-bearing rung starts at `k3(x) + x` rather than at
    `k3(x)`. That is inherent to having an identity branch at all, not a defect,
    and it is asserted explicitly here so the difference between "R1 starts as
    R0" and "R2/R3 start shifted by the identity" stays documented in a test
    rather than in a comment nobody re-reads.
    """
    torch.manual_seed(21)
    x = torch.randn(1, 16, 10, 10)

    r1 = RepAetherSR(16, 2, branches=LADDERS["R1"]).body[0]
    with torch.no_grad():
        assert torch.allclose(r1(x), r1.k3(x), atol=TOL)

    for rung in ("R2", "R3"):
        block = RepAetherSR(16, 2, branches=LADDERS[rung]).body[0]
        with torch.no_grad():
            assert torch.allclose(block(x), block.k3(x) + x, atol=TOL)
            assert not torch.allclose(block(x), block.k3(x), atol=TOL)


@pytest.mark.parametrize("rung", sorted(LADDERS))
def test_network_fuse_preserves_dtype_and_device(rung: str) -> None:
    """`fuse()` must hand back a network in the same dtype and device it fused.

    `Tensor.copy_` casts the values it writes but never moves the module it
    writes into, so a default-constructed destination silently returns CPU
    float32. That would quietly downcast the float64 exactness gate above -
    turning a 1e-12 algebra check into a 1e-7 float32 rounding check that still
    passes - and would drag an MPS model back to the CPU mid-run.
    """
    rep = RepAetherSR(16, 2, branches=LADDERS[rung]).double().eval()
    fused = rep.fuse()
    for name, p in fused.named_parameters():
        assert p.dtype is torch.float64, f"{name} came back {p.dtype}"
        assert p.device == next(rep.parameters()).device, f"{name} moved device"

    # And the exactness the dtype exists to protect still holds through fuse().
    x = torch.randn(1, 3, 12, 16, dtype=torch.float64)
    with torch.no_grad():
        assert (rep(x) - fused(x)).abs().max().item() <= 1e-12


@pytest.mark.parametrize("rung", [r for r in sorted(LADDERS) if r != "R0"])
def test_rung_choice_does_not_disturb_initialisation(rung: str) -> None:
    """Every rung must start from the same weights and the same RNG stream as R0.

    The ladder is supposed to vary one thing: the training-time block structure.
    Constructing the optional branches used to consume draws from the global
    generator, so `body.1` and `head` were initialised differently and the data
    order diverged - every arm differed by structure *plus* initialisation
    *plus* shuffling, and a 0.02 dB reading could have been any of the three.

    Checks both halves, because either alone is passable while the experiment is
    still confounded: the shared tensors must be bit-identical, and the stream
    left behind must be untouched so everything drawn later agrees.
    """

    def build(name: str) -> tuple[dict[str, torch.Tensor], list[float]]:
        torch.manual_seed(1)
        model = RepAetherSR(16, 2, branches=LADDERS[name])
        params = {k: v.detach().clone() for k, v in model.named_parameters()}
        # Stands in for every draw the training loop makes after construction:
        # the epoch shuffle, augmentation, dropout masks.
        return params, torch.randn(4).tolist()

    base, base_stream = build("R0")
    other, other_stream = build(rung)

    for name, tensor in base.items():
        assert name in other, f"{name} missing from {rung}"
        assert torch.equal(tensor, other[name]), f"{rung} changed the init of {name}"

    assert base_stream == other_stream, f"{rung} left the RNG stream shifted"

    # The added branches exist and are zeroed, so the extra tensors are real
    # but contribute nothing at step 0.
    extra = sorted(set(other) - set(base))
    assert extra, f"{rung} added no branch tensors"
    for name in extra:
        assert torch.count_nonzero(other[name]) == 0, f"{name} did not start at zero"


@pytest.mark.parametrize("seed", (8099, 8101, 8102, 8103))
def test_m8_paired_effective_initialization(seed: int) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        plain = AetherSR(16, 2).eval()
        plain_next_rng = torch.get_rng_state().clone()
        torch.manual_seed(seed)
        reparameterized = RepAetherSR(
            16, 2, branches=LADDERS["R3"], matched_init=True,
        ).eval()
        assert torch.equal(torch.get_rng_state(), plain_next_rng)

    for name in ("stem", "head"):
        for parameter in ("weight", "bias"):
            assert torch.equal(
                getattr(getattr(plain, name), parameter),
                getattr(getattr(reparameterized, name), parameter),
            )

    generator = torch.Generator().manual_seed(20260911)
    for plain_block, rep_block in zip(plain.body, reparameterized.body):
        fused_weight, fused_bias = rep_block.fused_parameters()
        torch.testing.assert_close(fused_weight, plain_block.weight, atol=TOL, rtol=0)
        torch.testing.assert_close(fused_bias, plain_block.bias, atol=TOL, rtol=0)
        for shape in SHAPES:
            input_tensor = torch.randn(shape, generator=generator)
            with torch.no_grad():
                plain_output = plain_block(input_tensor)
                rep_output = rep_block(input_tensor)
            torch.testing.assert_close(rep_output, plain_output, atol=TOL, rtol=0)
            torch.testing.assert_close(
                rep_output.tanh(), plain_output.tanh(), atol=TOL, rtol=0,
            )

    for height, width in ((1, 1), (9, 13), (24, 32), (64, 64)):
        input_tensor = torch.rand((2, 3, height, width), generator=generator)
        with torch.no_grad():
            torch.testing.assert_close(
                reparameterized(input_tensor), plain(input_tensor), atol=TOL, rtol=0,
            )
