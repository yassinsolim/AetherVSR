"""Observable corpus merge contracts: fail closed and preserve patch identity."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

SCRIPT = Path(__file__).with_name("corpus-shard.py")
spec = importlib.util.spec_from_file_location("corpus_shard", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def manifest(root, ids):
    path = root / "manifest.json"
    path.write_text(json.dumps({"clips": [{"id": cid} for cid in ids]}))
    return path


def shard(root, name, clips, failures=(), sequences=1):
    path = root / name
    path.mkdir()
    index, values = [], []
    for cid, seq, value in clips:
        index.append({"clip": cid, "tag": f"{cid}__{seq:02}", "patches": 2})
        values.extend([value, value])
    tensor = torch.tensor(values, dtype=torch.uint8).reshape(-1, 1, 1, 1)
    torch.save({"lr": tensor, "hr": tensor.repeat(1, 1, 2, 2)}, path / "all.pt")
    (path / "pairs.json").write_text(json.dumps({
        "index": index, "clipsFailed": [{"clip": cid} for cid in failures],
        "schema": "aethervsr.video-pairs/2",
        "sequencesPerClip": sequences,
        "encodeEvidence": [{"tag": entry["tag"], "bytes": i + 100} for i, entry in enumerate(index)],
    }))
    return path


def test_incomplete_cli_rejects_before_creating_output(tmp_path):
    source = manifest(tmp_path, ["a", "b"])
    part = shard(tmp_path, "part", [("a", 0, 7)], ["b"])
    destination = tmp_path / "merged"
    result = subprocess.run([
        sys.executable, str(SCRIPT), "merge", "--manifest", str(source),
        "--shard-dirs", str(part), "--out", str(destination),
    ], capture_output=True, text=True)
    assert result.returncode == 1
    assert "missing b" in result.stderr
    assert not destination.exists()


def test_complete_cli_preserves_manifest_and_patch_order(tmp_path):
    source = manifest(tmp_path, ["a", "b", "c"])
    left = shard(tmp_path, "left", [("c", 0, 30), ("a", 0, 10)])
    right = shard(tmp_path, "right", [("b", 0, 20)])
    destination = tmp_path / "merged"
    result = subprocess.run([
        sys.executable, str(SCRIPT), "merge", "--manifest", str(source),
        "--shard-dirs", str(left), str(right), "--out", str(destination),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    metadata = json.loads((destination / "pairs.json").read_text())
    assert metadata["clipsFailed"] == []
    assert "failures" not in metadata
    assert [entry["tag"] for entry in metadata["index"]] == ["a__00", "b__00", "c__00"]
    assert [entry["tag"] for entry in metadata["encodeEvidence"]] == [entry["tag"] for entry in metadata["index"]]
    assert metadata["sourceManifest"] == str(source)
    tensors = torch.load(destination / "all.pt", weights_only=True)
    expected = torch.tensor([10, 10, 20, 20, 30, 30], dtype=torch.uint8)
    assert torch.equal(tensors["lr"][:, 0, 0, 0], expected)
    assert torch.equal(tensors["hr"][:, 0, 1, 1], expected)
    assert sum(entry["patches"] for entry in metadata["index"]) == len(expected)


@pytest.mark.parametrize("case", ["missing", "duplicate-manifest", "duplicate-owner", "unknown", "unknown-failure", "failed-prepared"])
def test_invalid_coverage_never_emits_a_corpus(tmp_path, case):
    source = manifest(tmp_path, ["a", "a"] if case == "duplicate-manifest" else ["a", "b"])
    clips = [("a", 0, 1)]
    if case not in ("missing", "duplicate-manifest"):
        clips.append(("b", 0, 2))
    if case == "unknown":
        clips.append(("alien", 0, 3))
    failures = ["alien"] if case == "unknown-failure" else ["a"] if case == "failed-prepared" else []
    parts = [shard(tmp_path, "first", clips, failures)]
    if case == "duplicate-owner":
        parts.append(shard(tmp_path, "second", [("a", 1, 4)]))
    destination = tmp_path / "merged"
    assert module.merge(str(source), [str(p) for p in parts], str(destination)) == 1
    assert not destination.exists()


def test_existing_destination_is_not_overwritten(tmp_path):
    source = manifest(tmp_path, ["a"])
    part = shard(tmp_path, "part", [("a", 0, 7)])
    destination = tmp_path / "merged"
    destination.mkdir()
    (destination / "pairs.json").write_text("preserve this output")
    assert module.merge(str(source), [str(part)], str(destination)) == 1
    assert (destination / "pairs.json").read_text() == "preserve this output"
    assert not (destination / "all.pt").exists()


@pytest.mark.parametrize("case", ["missing-evidence", "duplicate-evidence", "short-lr", "protocol-drift"])
def test_metadata_or_tensor_drift_prevents_output(tmp_path, case):
    source = manifest(tmp_path, ["a", "b"])
    parts = [shard(tmp_path, "first", [("a", 0, 7)]), shard(tmp_path, "second", [("b", 0, 8)])]
    meta_path = parts[1] / "pairs.json"
    metadata = json.loads(meta_path.read_text())
    if case == "missing-evidence":
        metadata["encodeEvidence"] = []
    elif case == "duplicate-evidence":
        metadata["encodeEvidence"] *= 2
    elif case == "protocol-drift":
        metadata["seed"] = 1234
    else:
        blob = torch.load(parts[1] / "all.pt", weights_only=True)
        blob["lr"] = blob["lr"][:1]
        torch.save(blob, parts[1] / "all.pt")
    meta_path.write_text(json.dumps(metadata))
    destination = tmp_path / "merged"
    assert module.merge(str(source), [str(p) for p in parts], str(destination)) == 1
    assert not destination.exists()


@pytest.mark.parametrize("slots", [[0], [0, 1], [0, 1, 3]])
def test_partial_or_wrong_sequence_slots_never_emit_output(tmp_path, slots):
    source = manifest(tmp_path, ["a"])
    part = shard(tmp_path, "part", [("a", seq, seq + 1) for seq in slots], sequences=3)
    destination = tmp_path / "merged"
    assert module.merge(str(source), [str(part)], str(destination)) == 1
    assert not destination.exists()
