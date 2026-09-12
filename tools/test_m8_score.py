"""Synthetic scorer apparatus tests only; never open experimental models or frames."""

import statistics
import hashlib

import pytest
import torch
from PIL import Image

import m8_score as scorer


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