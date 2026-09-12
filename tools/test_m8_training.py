from __future__ import annotations

import math
import json
import sys

import pytest
import torch

import train
from train import PairedStream, build_model, patch_batch, psnr, update_lr, validate_patches, validation_updates


def test_paired_stream_is_independent_of_model_and_global_rng() -> None:
    records = []
    for rung in ("R0", "R3"):
        stream = PairedStream(8101)
        before = torch.get_rng_state().clone()
        build_model(rung, 16, 2, 8101)
        assert torch.equal(before, torch.get_rng_state())
        torch.rand(73 if rung == "R3" else 13)
        permutation = stream.permutation(43488)
        records.append([
            (permutation[batch * 32:(batch + 1) * 32].tolist(), stream.transforms())
            for batch in range(8)
        ])
    assert records[0] == records[1]
    assert len({sample for samples, _ in records[0] for sample in samples}) == 256


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_model_construction_preserves_mps_rng() -> None:
    before = torch.mps.get_rng_state().clone()
    for rung in ("R0", "R3"):
        build_model(rung, 16, 2, 8101)
        assert torch.equal(before, torch.mps.get_rng_state())


@pytest.mark.parametrize("update", (0, 20295, 40590, 60885, 81180))
def test_registered_update_lr_matches_both_arms(update: int) -> None:
    rates = []
    for rung in ("R0", "R3"):
        optimizer = torch.optim.Adam(build_model(rung, 16, 2, 8101).parameters(), lr=0.002)
        optimizer.param_groups[0]["lr"] = update_lr(0.002, update, 81180)
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[0] == rates[1]
    assert rates[0] == 0.001 * (1 + math.cos(math.pi * update / 81180))


def test_registered_checkpoint_count_and_final_budget() -> None:
    updates = validation_updates(81180, 1353)
    assert updates == [1353 * draw for draw in range(1, 61)]
    assert len(updates) == 60
    assert updates[-1] == 81180
    assert 16200 not in updates
    assert 43488 // 32 * 60 > 81180


@pytest.mark.parametrize("budget,every", ((0, 1), (10, 0), (-1, 2)))
def test_invalid_schedule_is_rejected(budget: int, every: int) -> None:
    with pytest.raises(ValueError):
        validation_updates(budget, every)


def test_uint8_and_float_patch_conversion_are_identical() -> None:
    patches = torch.randint(0, 256, (8, 3, 16, 16), dtype=torch.uint8)
    selected = torch.tensor([3, 7, 1])
    torch.testing.assert_close(
        patch_batch(patches, selected, "cpu"),
        patch_batch(patches.float() / 255, selected, "cpu"), atol=0, rtol=0,
    )


@pytest.mark.parametrize("batch", (1, 3, 7))
def test_batched_validation_preserves_global_rgb_psnr(batch: int) -> None:
    generator = torch.Generator().manual_seed(31)
    inputs = torch.rand((7, 3, 16, 16), generator=generator)
    targets = torch.rand((7, 3, 32, 32), generator=generator)
    model = build_model("R0", 16, 2, 8101).eval()
    with torch.no_grad():
        expected = psnr(model(inputs), targets)
    actual, structural = validate_patches(model, inputs, targets, "cpu", batch=batch)
    assert abs(actual - expected) <= 1e-5
    assert structural is None


def test_fixed_final_is_persisted_before_best_postprocessing(tmp_path, monkeypatch) -> None:
    generator = torch.Generator().manual_seed(39)
    inputs = torch.randint(0, 256, (32, 3, 16, 16), generator=generator, dtype=torch.uint8)
    targets = torch.randint(0, 256, (32, 3, 32, 32), generator=generator, dtype=torch.uint8)
    metadata = {
        "trainClips": ["synthetic-train"], "valClips": ["synthetic-val"],
        "structure": "gop", "crfDistribution": "synthetic", "sourceManifest": "synthetic",
        "trainPatches": 32, "corpusDigest": "synthetic", "seed": 39,
        "structureSpec": {"description": "synthetic", "x264params": "synthetic"},
        "crfSpec": {"description": "synthetic"}, "crfHistogram": {}, "preset": "medium",
        "sequences": 1, "sequenceFrames": 24,
    }
    monkeypatch.setattr(train, "load_video_pairs", lambda *args, **kwargs: (metadata, inputs, targets, inputs, targets))
    final_path = tmp_path / "R0-update1.json"

    def validate_before_ssim(*args, include_ssim=False, **kwargs):
        if include_ssim:
            assert final_path.is_file()
            states = torch.load(tmp_path / "R0-states.pt", weights_only=True)
            assert states["fixedFinal"]
            assert json.loads(final_path.read_text())["training"]["checkpointKind"] == "fixed-final"
            raise RuntimeError("synthetic postprocessing failure")
        return 25.0, None

    monkeypatch.setattr(train, "validate_patches", validate_before_ssim)
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--pairs", "synthetic", "--out", str(tmp_path / "R0.json"),
        "--m8-audit", str(tmp_path / "audit.json"), "--m8-registration", "c955743",
        "--m8-smoke", "--max-steps", "1", "--epochs", "1", "--device", "cpu",
    ])
    with pytest.raises(RuntimeError, match="synthetic postprocessing failure"):
        train.main()
    assert final_path.is_file()
    assert not (tmp_path / "R0.json").exists()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_uint8_and_batched_global_psnr_parity() -> None:
    generator = torch.Generator().manual_seed(31)
    inputs = torch.randint(0, 256, (7, 3, 16, 16), generator=generator, dtype=torch.uint8)
    targets = torch.randint(0, 256, (7, 3, 32, 32), generator=generator, dtype=torch.uint8)
    torch.testing.assert_close(patch_batch(inputs, slice(None), "mps").cpu(), inputs.float() / 255,
                               atol=0, rtol=0)
    model = build_model("R0", 16, 2, 8101).eval().to("mps")
    with torch.no_grad():
        expected = psnr(model(patch_batch(inputs, slice(None), "mps")),
                        patch_batch(targets, slice(None), "mps"))
    actual, _ = validate_patches(model, inputs, targets, "mps", batch=3)
    assert abs(actual - expected) <= 1e-5