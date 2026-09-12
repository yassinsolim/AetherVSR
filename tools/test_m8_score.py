"""Synthetic scorer apparatus tests only; never open experimental models or frames."""

import statistics
import hashlib
import copy
import json

import pytest
import torch
from PIL import Image

import m8_score as scorer
from aethersr import AetherSR, export_weights


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def images(tmp_path):
    generator = torch.Generator().manual_seed(17)
    frames = []
    for index in range(1, 9):
        frame = {"id": f"{index:04d}"}
        for kind, size in (("lr", (9, 7)), ("hr", (18, 14))):
            path = tmp_path / kind / f"{kind}_{index:04d}.png"
            path.parent.mkdir(exist_ok=True)
            pixels = torch.randint(0, 256, (size[1], size[0], 3), generator=generator,
                                   dtype=torch.uint8)
            Image.frombytes("RGB", size, bytes(pixels.flatten().tolist())).save(path)
            frame[kind] = str(path.relative_to(tmp_path))
        frames.append(frame)
    cells = [{"clip": "synthetic", "category": "text", "crf": crf, "frames": frames}
             for crf in scorer.CRFS]
    return tmp_path, cells


def test_collect_matches_original_cpu_score_and_keeps_negative_cells(images):
    root, cells = images
    model = torch.nn.Upsample(scale_factor=2, mode="nearest").eval()
    original_cells = [{**cell, "hr": str(root / "hr"), "lr": str(root / "lr")} for cell in cells]
    baseline = scorer.m6.score(None, original_cells)["perCell"]
    result = scorer.collect(root, cells, {"nearest": model}, "cpu")
    assert result["baseline"]["perCell"] == baseline
    expected = []
    for cell in cells:
        values = []
        for frame in cell["frames"]:
            hr, lr = (scorer.m6.load_png(str(root / frame[kind])) for kind in ("hr", "lr"))
            values.append(scorer.metrics(model(lr), hr))
        expected.append({"clip": cell["clip"], "category": cell["category"], "crf": cell["crf"],
                         "psnr": statistics.fmean(value[0] for value in values),
                         "ssim": statistics.fmean(value[1] for value in values)})
    assert result["models"]["nearest"] == scorer.m6.summarise(expected, baseline, scorer.CRFS)
    assert len(result["models"]["nearest"]["perCell"]) == 3
    assert all(cell["delta"] < 0 for cell in result["models"]["nearest"]["perCell"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_frame_rejected(bad):
    with pytest.raises(scorer.experiment.GuardError, match="nonfinite"):
        scorer.metrics(torch.full((1, 3, 14, 18), bad), torch.zeros(1, 3, 14, 18))


def test_refuses_to_overwrite_before_scoring(tmp_path):
    target = tmp_path / "results/m8-scores.json"
    target.parent.mkdir()
    target.write_text("existing evidence")
    assert scorer.main(["--out", str(target)], root=tmp_path) == 1
    assert target.read_text() == "existing evidence"


def test_refuses_incomplete_runs(tmp_path):
    with pytest.raises(ValueError):
        scorer.model_artifacts(tmp_path, {}, {"runs": []})


def test_model_bytes_are_checked(tmp_path):
    target = tmp_path / "model.json"
    target.write_text("model bytes")
    artifacts = {"test": {"path": "model.json", "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}}
    scorer.check_models(tmp_path, artifacts)
    target.write_text("different bytes")
    with pytest.raises(scorer.experiment.GuardError, match="model bytes changed"):
        scorer.check_models(tmp_path, artifacts)


def test_failed_parity_cannot_open_scoring():
    with pytest.raises(scorer.experiment.GuardError, match="passing scoring parity"):
        scorer.validate_parity({"passed": False}, {}, {}, {})


def test_missing_frame_is_not_silently_scored(images):
    root, cells = images
    cells[0]["frames"] = cells[0]["frames"][:-1]
    with pytest.raises(scorer.experiment.GuardError, match="eight scored frames"):
        scorer.collect(root, cells, {}, "cpu")


def test_binding_collection_rejects_unreliable_mps_backend_before_loading(images):
    root, cells = images
    with pytest.raises(scorer.experiment.GuardError, match="CPU reference"):
        scorer.collect(root, cells, {}, "mps")


def test_metadata_different_neural_exports_match_original_and_reversed_order(images):
    root, cells = images
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(18)
        model = AetherSR(channels=16, depth=2).eval()
    payload = {"features": 16, "depth": 2, "parameters": 6291, "scale": 2,
               "weights": export_weights(model)}
    artifacts = {}
    for key in ("best", "final"):
        path = root / f"{key}.json"
        path.write_text(json.dumps({**payload, "checkpointKind": key}))
        artifacts[key] = {"path": path.name, "sha256": scorer.experiment.sha256(path)}
    assert artifacts["best"]["sha256"] != artifacts["final"]["sha256"]
    loaded = scorer.load_models(root, artifacts, "cpu")
    assert scorer.tensor_identity(loaded["best"]) == scorer.tensor_identity(loaded["final"])
    forward = scorer.collect(root, cells, loaded, "cpu")
    reverse = scorer.collect(root, cells, dict(reversed(list(loaded.items()))), "cpu")
    assert forward["models"] == reverse["models"]
    assert forward["frameScores"] == reverse["frameScores"]
    assert forward["models"]["best"] == forward["models"]["final"]
    assert forward["duplicateWeightVerification"]["perFrameComparisons"] == 24
    assert set(forward["duplicateWeightVerification"]["maxima"].values()) == {0.0}
    original_cells = [{**cell, "hr": str(root / "hr"), "lr": str(root / "lr")} for cell in cells]
    original = scorer.m6.score(str(root / "best.json"), original_cells)["perCell"]
    for expected, actual in zip(original, forward["models"]["best"]["perCell"], strict=True):
        assert expected["psnr"] == actual["psnr"]
        assert expected["ssim"] == actual["ssim"]
    frame = cells[0]["frames"][0]
    inputs = scorer.m6.load_png(str(root / frame["lr"]))
    assert list(inputs.shape) == [1, 3, 7, 9]
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), model(inputs.contiguous()), rtol=0, atol=1e-5)


@pytest.mark.parametrize("metric", ["output", "psnr", "ssim"])
def test_inconsistent_identical_weights_fail_closed(metric):
    expected = torch.zeros(1, 3, 14, 18)
    actual = expected + (0.01 if metric == "output" else 0)
    values = (30.1 if metric == "psnr" else 30.0, 0.91 if metric == "ssim" else 0.9)
    with pytest.raises(scorer.experiment.GuardError, match="identical weights"):
        scorer.duplicate_check(expected, actual, (30.0, 0.9), values)


def test_historical_training_hashes_remain_authoritative(monkeypatch, tmp_path):
    original = {"tools/train.py": b"trainer", "tools/m8_score.py": b"old scorer",
                "tools/test_m8_score.py": b"old tests"}
    source = {name: hashlib.sha256(raw).hexdigest() for name, raw in original.items()}
    current = {**source, "tools/m8_score.py": "repaired", "tools/test_m8_score.py": "new tests"}
    manifest = {"executionCommit": "original-commit", "codeSha256": source}
    monkeypatch.setattr(scorer.experiment, "code_hashes", lambda root: current)
    monkeypatch.setattr(scorer.experiment, "git", lambda root, *args: original[args[-1].split(":")[1]])
    assert scorer.training_sources(tmp_path, manifest) == source
    current["tools/train.py"] = "changed trainer"
    with pytest.raises(scorer.experiment.GuardError, match="non-scoring source changed"):
        scorer.training_sources(tmp_path, manifest)
    current["tools/train.py"] = source["tools/train.py"]
    original["tools/train.py"] = b"wrong historical bytes"
    with pytest.raises(scorer.experiment.GuardError, match="historical training source hash"):
        scorer.training_sources(tmp_path, manifest)


def test_withdrawn_mps_parity_is_never_eligible():
    with pytest.raises(scorer.experiment.GuardError, match="withdrawn MPS"):
        scorer.validate_parity({"schema": "aethervsr.m8-scoring-parity/1", "passed": True}, {}, {}, {})


def test_independent_reverse_parity_requires_identical_metrics():
    forward = {"reverseModelOrder": False, "metricDigest": "same", "cells": ["frozen"]}
    reverse = {**forward, "reverseModelOrder": True}
    scorer.validate_parity_pair(forward, reverse)
    changed = copy.deepcopy(reverse)
    changed["metricDigest"] = "different"
    with pytest.raises(scorer.experiment.GuardError, match="CPU parity runs"):
        scorer.validate_parity_pair(forward, changed)


def test_frame_metrics_reproduce_cell_means_and_reject_corruption(images):
    root, cells = images
    result = scorer.collect(root, cells, {}, "cpu")
    scorer.validate_frame_scores(result, cells)
    result["frameScores"]["catmull_rom"][0][0] = (1.0, 0.0)
    with pytest.raises(scorer.experiment.GuardError, match="differs from its eight frame metrics"):
        scorer.validate_frame_scores(result, cells)


def test_publisher_rejects_withdrawn_backend_before_any_evaluation(monkeypatch, tmp_path):
    monkeypatch.setattr(scorer.experiment, "committed_bytes", lambda *args: None)
    monkeypatch.setattr(scorer.experiment, "read_json", lambda path: {"device": "mps"})
    with pytest.raises(scorer.experiment.GuardError, match="withdrawn MPS scores are ineligible"):
        scorer.publish_report(tmp_path, {}, {}, {}, {}, {}, "repair")