"""Preparation must reject source drift and incomplete sequence coverage."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

SCRIPT = Path(__file__).with_name("corpus-prepare.py")
spec = importlib.util.spec_from_file_location("corpus_prepare", SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


@pytest.fixture
def pinned_input(tmp_path, monkeypatch):
    clip = {
        "id": "pinned-clip", "title": "File:Pinned.webm",
        "source_url": "https://example.invalid/pinned.webm", "category": "nature",
        "commonsSha1": "expected-sha1", "byteLength": 100,
        "prefixSha256": "expected-prefix", "prefixBytes": 100, "pinnedAt": "original",
        "duration": 4.0,
    }
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps({"clips": [clip]}))
    destination = tmp_path / "prepared"
    monkeypatch.setattr(prepare, "api_info", lambda *args: {
        clip["title"]: {"sha1": "expected-sha1", "size": 100},
    })
    monkeypatch.setattr(prepare, "prefix_hash", lambda *args: ("expected-prefix", 100))
    monkeypatch.setattr(prepare.time, "sleep", lambda *args: None)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--manifest", str(source),
                                     "--out", str(destination), "--sequences", "3"])
    return clip, source, destination


@pytest.mark.parametrize("failure", ["prefix-unavailable", "prefix-changed", "api-unavailable", "sha1-changed"])
def test_source_identity_failure_preserves_pins_and_skips_decode(pinned_input, monkeypatch, failure):
    clip, source, destination = pinned_input
    if failure == "prefix-unavailable":
        monkeypatch.setattr(prepare, "prefix_hash", lambda *args: (None, None))
    elif failure == "prefix-changed":
        monkeypatch.setattr(prepare, "prefix_hash", lambda *args: ("replacement-prefix", 100))
    elif failure == "api-unavailable":
        monkeypatch.setattr(prepare, "api_info", lambda *args: {})
    else:
        monkeypatch.setattr(prepare, "api_info", lambda *args: {
            clip["title"]: {"sha1": "replacement-sha1", "size": 100},
        })
    def forbidden_decode(*args, **kwargs):
        pytest.fail("unverified source reached the decoder")
    monkeypatch.setattr(prepare, "extract_hr", forbidden_decode)
    assert prepare.main() == 1
    assert json.loads(source.read_text())["clips"] == [clip]
    metadata = json.loads((destination / "pairs.json").read_text())
    assert metadata["clipsPrepared"] == 0
    assert [record["clip"] for record in metadata["clipsFailed"]] == [clip["id"]]
    assert not (destination / "all.pt").exists()


@pytest.mark.parametrize("successful_sequences", [0, 1, 2, 3])
def test_only_complete_preparation_emits_training_tensors(pinned_input, monkeypatch, successful_sequences):
    clip, source, destination = pinned_input
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    import video_degrade_lib
    # Isolate network/codec availability, not the completion/accounting under test.
    outcomes = iter([24] * successful_sequences + [0] * (3 - successful_sequences))
    monkeypatch.setattr(prepare, "extract_hr", lambda *args: next(outcomes))
    monkeypatch.setattr(prepare, "degrade", lambda *args: {"lrFrames": 24, "bytes": 100, "crf": args[2]})
    def harvest(hr_dir, lr_dir, patches, *args):
        patches["lr"].append(torch.zeros(3, 2, 2, dtype=torch.uint8))
        patches["hr"].append(torch.zeros(3, 4, 4, dtype=torch.uint8))
        return 1
    monkeypatch.setattr(video_degrade_lib, "harvest_patches", harvest)
    result = prepare.main()
    metadata = json.loads((destination / "pairs.json").read_text())
    assert metadata["sequences"] == successful_sequences
    if successful_sequences == 3:
        assert result == 0
        assert metadata["clipsFailed"] == []
        tensors = torch.load(destination / "all.pt", weights_only=True)
        assert tensors["lr"].shape[0] == tensors["hr"].shape[0] == 3
    else:
        assert result == 1
        failure = metadata["clipsFailed"][0]
        assert failure["clip"] == clip["id"]
        assert failure["expectedSequences"] == 3
        assert failure["actualSequences"] == successful_sequences
        assert not (destination / "all.pt").exists()


@pytest.mark.parametrize("duration", [None, 0, float("nan"), float("inf")])
def test_unverified_duration_never_reaches_the_decoder(pinned_input, monkeypatch, duration):
    clip, source, destination = pinned_input
    clip["duration"] = duration
    source.write_text(json.dumps({"clips": [clip]}))
    def forbidden_decode(*args, **kwargs):
        pytest.fail("unverified timing reached the decoder")
    monkeypatch.setattr(prepare, "extract_hr", forbidden_decode)
    assert prepare.main() == 1
    metadata = json.loads((destination / "pairs.json").read_text())
    assert "duration" in metadata["clipsFailed"][0]["reason"]
    assert not (destination / "all.pt").exists()
