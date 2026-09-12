"""Fail-closed execution and provenance gates for the registered M8 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import time
from datetime import datetime, timezone
from collections import Counter
from contextlib import contextmanager


ROOT = Path(__file__).resolve().parents[1]
PLAN = "results/m8-plan.json"
REGISTRATION = "docs/M8-PREREGISTRATION.md"
REGISTRATION_PREFIX = "c955743"
RUN_ORDER = [[8101, "R0"], [8101, "R3"], [8102, "R3"], [8102, "R0"],
             [8103, "R0"], [8103, "R3"]]
TRAINING = {
    "optimizerUpdates": 81180, "epochsUpperBound": 60, "channels": 16,
    "depth": 2, "scale": 2, "batch": 32, "hrPatch": 128, "lrPatch": 64,
    "trainPatches": 43488, "validationPatches": 3072, "optimizer": "Adam",
    "lr": 0.002, "betas": [0.9, 0.999], "eps": 1e-8, "weightDecay": 0,
    "amsgrad": False, "loss": "L1", "dtype": "float32", "device": "mps",
    "lrSchedule": "cosine over zero-based optimizer updates",
    "lrCheckIndices": [0, 20295, 40590, 60885, 81180],
    "validationEveryUpdates": 1353, "validationDraws": 60,
    "checkpointMetric": "RGB PSNR of global fixed validation-patch MSE",
    "checkpointTieRule": "earliest update on exact tie",
    "snapshotUpdates": [5412, 10824, 16200, 29766, 50061, 64944, 81180],
    "snapshotOnlyUpdates": [16200], "matchedEffectiveInitialization": True,
    "initializationAbsTolerance": 1e-5, "initializationRelTolerance": 0,
    "shuffleRng": "independent CPU torch.Generator seeded by paired seed",
    "augmentationRng": "independent random.Random seeded by paired seed",
    "augmentations": ["horizontal flip p=0.5", "vertical flip p=0.5",
                      "90-degree rotation p=0.5"],
    "diagnosticPrefixBatches": 8, "onlineCrop": False,
    "onlineDegradation": False, "serialExecution": True,
}
INPUT_FREEZE = "results/m8-input-freeze.json"
SMOKE_REPORT = "results/m8-smoke.json"
RUNS_REPORT = "results/m8-runs.json"


class GuardError(RuntimeError):
    """A prerequisite or an evidence check failed; never restart automatically."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GuardError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def local_path(root: Path, name: str) -> Path:
    path = root / name
    require(not Path(name).is_absolute() and ".." not in Path(name).parts,
            f"not a repository-relative path: {name}")
    require(path.resolve().is_relative_to(root.resolve()), f"path escapes repository: {name}")
    return path


def write_new(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def validate_plan(plan: dict) -> None:
    expected = {
        "schema": "aethervsr.m8-plan/1", "registration": REGISTRATION,
        "arms": ["R0", "R3"], "pairedSeeds": [8101, 8102, 8103],
        "runOrder": RUN_ORDER, "smokeSeed": 8099, "training": TRAINING,
        "conditionalFinalSeeds": [8201, 8202, 8203, 8204, 8205],
    }
    for key, value in expected.items():
        require(plan.get(key) == value, f"unregistered plan configuration: {key}")
    for key, value in {"crfs": [18, 26, 34], "frames": 8, "fps": 24,
                       "requestedStartSeconds": 12.0, "clips": 16, "cells": 48}.items():
        require(plan.get("scoring", {}).get(key) == value,
                f"unregistered scoring configuration: {key}")


def git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    require(result.returncode == 0, f"git {' '.join(args)} failed")
    return result.stdout


def committed_bytes(root: Path, commit: str, name: str) -> None:
    require(git(root, "show", f"{commit}:{name}") == local_path(root, name).read_bytes(),
            f"committed bytes changed: {name}")


def registration(root: Path, plan_path: Path) -> dict:
    plan_name = str(plan_path.relative_to(root))
    paths = [REGISTRATION, PLAN]
    additions = []
    for name in paths:
        commits = git(root, "log", "--reverse", "--diff-filter=A", "--format=%H",
                      "HEAD", "--", name).decode().splitlines()
        require(bool(commits), f"no registration addition commit: {name}")
        additions.append(commits[0])
    require(additions[0] == additions[1], "registration files were not added together")
    commit = additions[0]
    require(commit.startswith(REGISTRATION_PREFIX), "unexpected M8 registration commit")
    execution = git(root, "rev-parse", "HEAD").decode().strip()
    git(root, "merge-base", "--is-ancestor", commit, execution)
    for name in paths:
        committed_bytes(root, commit, name)
    require(plan_path.read_bytes() == git(root, "show", f"{commit}:{PLAN}"),
            f"alternate plan differs from registration: {plan_name}")
    validate_plan(read_json(plan_path))
    return {"registrationCommit": commit, "executionCommit": execution,
            "registrationFiles": {name: sha256(root / name) for name in paths}}


def file_record(root: Path, name: str) -> dict:
    path = local_path(root, name)
    require(path.is_file(), f"missing frozen input: {name}")
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def sequence_mapping(meta: dict, train_count: int, val_count: int) -> dict:
    groups = {"train": set(meta["trainClips"]), "val": set(meta["valClips"])}
    require(not groups["train"] & groups["val"], "train/val clip overlap")
    for split in groups:
        require(len(groups[split]) == len(meta[f"{split}Clips"]), "duplicate clip ids")
    offsets = {"train": 0, "val": 0}
    rows = {"train": [], "val": []}
    tags = set()
    for position, entry in enumerate(meta["index"]):
        members = [split for split, clips in groups.items() if entry["clip"] in clips]
        require(len(members) == 1, f"ambiguous index membership at {position}")
        split = members[0]
        require(entry.get("split", split) == split, f"index split mismatch at {position}")
        count, frames = entry["patches"], entry["frames"]
        require(type(count) is int and count > 0 and type(frames) is int and frames > 0
                and count % frames == 0, f"invalid sequence patch count at {position}")
        require(entry["tag"] not in tags, "duplicate sequence tag")
        tags.add(entry["tag"])
        rows[split].append({"indexPosition": position, "startRow": offsets[split],
                            "endRow": offsets[split] + count, "sequence": entry})
        offsets[split] += count
    require(offsets["train"] == train_count == meta["trainPatches"],
            "training index patch count does not match train tensor")
    require(meta["valPatches"] == val_count, "validation patch count mismatch")
    require(offsets["val"] in (0, val_count), "partial validation index")
    for split, entries in rows.items():
        if entries:
            require({row["sequence"]["clip"] for row in entries} == groups[split],
                    f"index omits {split} clips")
    require(meta["sequences"] in (len(rows["train"]), len(meta["index"])),
            "sequence count does not match index")
    return {"train": rows["train"], "validationSequencesIndexed": len(rows["val"]),
            "indexEntries": len(meta["index"]), "trainSequences": len(rows["train"]),
            "trainPatches": offsets["train"],
            "validationIndexAvailable": bool(rows["val"])}


def inspect_corpus(root: Path, plan: dict) -> tuple[dict, dict]:
    import torch

    directory = local_path(root, plan["pairsDirectory"])
    meta = read_json(directory / "pairs.json")
    expected = {"hrSize": [2560, 1440], "lrSize": [1280, 720], "fps": 24,
                "sequenceFrames": 24, "seed": 20260906, "structure": "gop",
                "preset": "medium", "crfDistribution": "uniform [18,36] per sequence"}
    for key, value in expected.items():
        require(meta.get(key) == value, f"frozen corpus protocol mismatch: {key}")
    require(meta["structureSpec"]["x264params"] ==
            "keyint=48:min-keyint=24:scenecut=40:bframes=3:ref=3", "codec settings changed")
    shapes = {}
    for split, count in (("train", 43488), ("val", 3072)):
        blob = torch.load(directory / f"{split}.pt", mmap=True, weights_only=True,
                          map_location="cpu")
        for kind, size in (("lr", 64), ("hr", 128)):
            tensor = blob[kind]
            require(tensor.dtype == torch.uint8 and tuple(tensor.shape) == (count, 3, size, size),
                    f"{split}.{kind}: expected uint8 {(count, 3, size, size)}")
            shapes[f"{split}.{kind}"] = list(tensor.shape)
    mapping = sequence_mapping(meta, 43488, 3072)
    require(len(meta["trainClips"]) == 151 and len(meta["valClips"]) == 16,
            "frozen corpus clip counts changed")
    require(mapping["trainSequences"] == 453, "expected 453 training sequences")
    require(mapping["validationSequencesIndexed"] in (0, 32), "expected 32 validation sequences")
    require(all(row["sequence"]["frames"] == 24 and row["sequence"]["patches"] == 96
                and 18 <= row["sequence"]["crf"] <= 36 for row in mapping["train"]),
            "training sequence geometry or degradation changed")
    return meta, {"shapes": shapes, "trainSequences": mapping["trainSequences"],
                  "indexEntries": mapping["indexEntries"],
                  "validationIndexAvailable": mapping["validationIndexAvailable"],
                  "validationSequencesIndexed": mapping["validationSequencesIndexed"]}


def cache_inventory(root: Path, cache: str, clips: list[dict]) -> tuple[dict, list]:
    from PIL import Image

    directory = local_path(root, cache)
    ids = [clip["id"] for clip in clips]
    require(len(ids) == len(set(ids)) == 16, "validation requires 16 unique clip ids")
    categories = Counter(clip["category"] for clip in clips)
    require(dict(categories) == {"daylight": 1, "faces": 2, "lowlight": 2, "motion": 2,
                                 "nature": 2, "text": 3, "texture": 2, "urban": 2},
            "validation category counts changed")
    require(directory.is_dir(), "missing validation cache")
    require({path.name for path in directory.iterdir() if path.is_dir()} == set(ids),
            "validation cache clip ids differ from manifest")
    files, cells = {}, []
    for clip in clips:
        clip_dir = directory / clip["id"]
        require({path.name for path in clip_dir.iterdir() if path.is_dir()} ==
                {"hr", "crf18", "crf26", "crf34"}, f"missing or extra cache cells: {clip['id']}")
        frame_sets = {}
        for subdir, prefix, size in (("hr", "hr", (2560, 1440)),
                                     ("crf18", "lr", (1280, 720)),
                                     ("crf26", "lr", (1280, 720)),
                                     ("crf34", "lr", (1280, 720))):
            paths = sorted((clip_dir / subdir).glob("*.png"))
            require(len(paths) == 8, f"expected eight PNGs: {clip['id']}/{subdir}")
            frames = {}
            for path in paths:
                match = re.fullmatch(rf"{prefix}_(\d{{4}})\.png", path.name)
                require(match is not None, f"invalid cached frame name: {path.name}")
                name = str(path.relative_to(root))
                with Image.open(path) as image:
                    require(image.format == "PNG" and image.size == size and image.mode == "RGB",
                            f"cached image geometry/format mismatch: {name}")
                    image.verify()
                files[name] = {**file_record(root, name), "dimensions": list(size), "mode": "RGB"}
                frames[match.group(1)] = name
            frame_sets[subdir] = frames
        for crf in (18, 26, 34):
            hr, lr = frame_sets["hr"], frame_sets[f"crf{crf}"]
            require(hr.keys() == lr.keys(), f"mismatched HR/LR frames: {clip['id']}/crf{crf}")
            cells.append({"clip": clip["id"], "category": clip["category"], "crf": crf,
                          "frames": [{"id": frame, "hr": hr[frame], "lr": lr[frame]}
                                     for frame in sorted(hr)]})
    require(len(cells) == 48, "expected 48 validation cells")
    return files, cells


def freeze_inputs(root: Path, plan: dict, registered: dict) -> dict:
    files = {}
    for key in ("trainingManifest", "validationManifest", "productionModel"):
        name = plan[key]
        files[name] = file_record(root, name)
        require(files[name]["sha256"] == plan[f"{key}Sha256"], f"input hash mismatch: {name}")
    for key in ("pairsFreeze", "validationFreeze"):
        name = plan[key]
        committed_bytes(root, registered["registrationCommit"], name)
        files[name] = file_record(root, name)
    pair_freeze = read_json(root / plan["pairsFreeze"])
    require(pair_freeze["pairsDirectory"] == plan["pairsDirectory"], "pairs directory changed")
    for key, value in {"trainClips": 151, "validationClips": 16, "trainSequences": 453,
                       "validationSequences": 32, "trainPatches": 43488,
                       "validationPatches": 3072}.items():
        require(pair_freeze.get(key) == value, f"pairs freeze count changed: {key}")
    for filename in ("pairs.json", "train.pt", "val.pt"):
        name = f"{plan['pairsDirectory']}/{filename}"
        files[name] = file_record(root, name)
        require(files[name] == pair_freeze["files"][filename], f"pairs file hash/size mismatch: {name}")
    meta, corpus = inspect_corpus(root, plan)
    digest = hashlib.sha256(json.dumps(meta["index"], sort_keys=True).encode())
    for filename in ("train.pt", "val.pt"):
        with (root / plan["pairsDirectory"] / filename).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    require(digest.hexdigest() == plan["trainerCorpusDigest"] == pair_freeze["trainerCorpusDigest"],
            "trainer combined corpusDigest mismatch")
    train_clips = read_json(root / plan["trainingManifest"])["clips"]
    val_clips = read_json(root / plan["validationManifest"])["clips"]
    for split, clips in (("train", train_clips), ("val", val_clips)):
        require(set(meta[f"{split}Clips"]) == {clip["id"] for clip in clips},
                f"{split} manifest/pairs membership differs")
    val_freeze = read_json(root / plan["validationFreeze"])
    require(val_freeze["manifestSha256"] == plan["validationManifestSha256"] and
            val_freeze["trainingManifestSha256"] == plan["trainingManifestSha256"] and
            val_freeze["audit"]["overlaps"] == [] and val_freeze["audit"]["schemaProblems"] == [],
            "invalid creator-disjoint validation freeze")
    images, cells = cache_inventory(root, plan["validationCache"], val_clips)
    files.update(images)
    inputs = {"files": files, "corpusDigest": digest.hexdigest(), "corpus": corpus,
              "validationCells": cells}
    return {**inputs, "digest": aggregate(inputs)}


def close_tensors(actual, expected, label: str, *, exact: bool = False) -> float:
    import torch

    require(actual.shape == expected.shape and torch.isfinite(actual).all().item()
            and torch.isfinite(expected).all().item(), f"nonfinite or mismatched tensors: {label}")
    error = (actual - expected).abs().max().item()
    require(error <= (0 if exact else 1e-5), f"{label}: max absolute error {error} exceeds tolerance")
    return error


def diagnostic_inputs():
    import torch

    generator = torch.Generator().manual_seed(20260911)
    return [torch.rand((2, 3, height, width), generator=generator)
            for height, width in ((1, 1), (3, 5), (16, 24))]


def initial_equivalence(seed: int, devices=("cpu", "mps")) -> dict:
    import torch
    from train import build_model

    evidence = {}
    with torch.no_grad():
        for device in devices:
            require(device != "mps" or torch.backends.mps.is_available(), "MPS is unavailable")
            rng = torch.get_rng_state().clone()
            plain = build_model("R0", 16, 2, seed).eval().to(device)
            branched = build_model("R3", 16, 2, seed).eval().to(device)
            require(torch.equal(rng, torch.get_rng_state()), "build_model altered the shared RNG")
            checks = {}
            for name in ("stem", "head"):
                for parameter in ("weight", "bias"):
                    label = f"{name}.{parameter}"
                    checks[label] = close_tensors(getattr(getattr(plain, name), parameter),
                                                  getattr(getattr(branched, name), parameter),
                                                  label, exact=True)
            generator = torch.Generator().manual_seed(20260911)
            for position, (conv, block) in enumerate(zip(plain.body, branched.body, strict=True)):
                weight, bias = block.fused_parameters()
                checks[f"body{position}.weight"] = close_tensors(weight, conv.weight, "fused body weight")
                checks[f"body{position}.bias"] = close_tensors(bias, conv.bias, "fused body bias")
                for height, width in ((1, 1), (3, 5), (16, 24)):
                    inputs = torch.randn((2, 16, height, width), generator=generator).to(device)
                    before, after = conv(inputs), block(inputs)
                    key = f"body{position}.{height}x{width}"
                    checks[f"{key}.preTanh"] = close_tensors(before, after, key)
                    checks[f"{key}.postTanh"] = close_tensors(torch.tanh(before), torch.tanh(after), key)
            for inputs in diagnostic_inputs():
                inputs = inputs.to(device)
                key = f"network.{inputs.shape[-2]}x{inputs.shape[-1]}"
                checks[key] = close_tensors(plain(inputs), branched(inputs), key)
            evidence[device] = checks
    return {"seed": seed, "atol": 1e-5, "rtol": 0, "bordersIncluded": True,
            "beforeOptimizerStep": True, "checks": evidence, "passed": True}


def expected_prefix(seed: int, count: int) -> list[dict]:
    from train import PairedStream

    require(count >= 256, "diagnostic prefix needs eight complete batches")
    stream = PairedStream(seed)
    permutation = stream.permutation(count)
    records = []
    for update in range(8):
        flip_w, flip_h, rotate = stream.transforms()
        records.append({"update": update, "samples": permutation[update * 32:(update + 1) * 32].tolist(),
                        "flipW": flip_w, "flipH": flip_h, "rotate90": rotate})
    return records


def stream_hash(prefix: list[dict]) -> str:
    digest = hashlib.sha256()
    for record in prefix:
        digest.update(json.dumps(record, sort_keys=True).encode())
    return digest.hexdigest()


def prefix_proof(pairs: Path, prefix: list[dict]) -> dict:
    import torch

    meta = read_json(pairs / "pairs.json")
    blob = torch.load(pairs / "train.pt", mmap=True, weights_only=True, map_location="cpu")
    count = len(blob["lr"])
    for key, size in (("lr", 64), ("hr", 128)):
        require(blob[key].dtype == torch.uint8 and tuple(blob[key].shape) == (count, 3, size, size),
                f"prefix tensor geometry mismatch: {key}")
    mapping = sequence_mapping(meta, count, meta["valPatches"])
    require(len(prefix) == 8 and all(len(batch["samples"]) == 32 for batch in prefix),
            "prefix must contain exactly eight complete batches")
    batches = []
    for batch in prefix:
        samples = []
        for row in batch["samples"]:
            require(type(row) is int and 0 <= row < count, "sample row outside frozen tensor")
            entry = next(entry for entry in mapping["train"] if entry["startRow"] <= row < entry["endRow"])
            patches = {key: hashlib.sha256(blob[key][row].contiguous().numpy().tobytes()).hexdigest()
                       for key in ("lr", "hr")}
            samples.append({"row": row, "indexPosition": entry["indexPosition"],
                            "rowWithinSequence": row - entry["startRow"],
                            "sequence": entry["sequence"], "patchSha256": patches,
                            "cachedBounds": {"lr": [0, 0, 64, 64], "hr": [0, 0, 128, 128]},
                            "originalCropCoordinates": None,
                            "originalCropCoordinatesAvailability": "unavailable: not stored in pairs index",
                            "frameIdentity": entry["sequence"].get("frame"),
                            "frameIdentityAvailability": "only where stored in original sequence metadata"})
        batches.append({"stream": batch, "patches": samples})
    proof = {"batches": batches, "trainIndexPatches": mapping["trainPatches"],
             "trainIndexSequences": mapping["trainSequences"], "indexEntries": mapping["indexEntries"],
             "tensorLayout": "uint8 CHW, contiguous raw bytes before transforms",
             "codec": {key: meta[key] for key in ("structure", "structureSpec", "preset", "fps",
                                                 "sequenceFrames", "seed") if key in meta}}
    return {**proof, "digest": aggregate(proof)}


def validate_report(report: dict, plan: dict, registered: dict, seed: int, rung: str,
                    *, smoke: bool) -> None:
    from train import update_lr, validation_updates

    budget = 8 if smoke else plan["training"]["optimizerUpdates"]
    every = 1 if smoke else plan["training"]["validationEveryUpdates"]
    expected = {"schema": "aethervsr.m8-training-run/1", "seed": seed, "rung": rung,
                "smokeOnly": smoke, "stepBudget": budget, "optimizerSteps": budget,
                "registrationCommit": registered["registrationCommit"], "device": "mps",
                "corpusDigest": plan["trainerCorpusDigest"],
                "validationUpdates": validation_updates(budget, every)}
    for key, value in expected.items():
        require(report.get(key) == value, f"trainer report mismatch: {key}")
    history = report["history"]
    require([row["update"] for row in history] == expected["validationUpdates"], "history budget mismatch")
    require([row["psnr"] for row in history] == report["validationPsnr"], "history PSNR mismatch")
    for row in history:
        require(all(math.isfinite(row[key]) for key in ("psnr", "loss", "lr")), "nonfinite loss/history")
        require(row["lr"] == update_lr(0.002, row["update"] - 1, budget), "history LR mismatch")
    require(report["bestUpdate"] == max(history, key=lambda row: row["psnr"])["update"],
            "best checkpoint violates earliest strict-maximum rule")
    indices = (0, budget // 4, budget // 2, 3 * budget // 4, budget)
    require(report["lrAtIndices"] == {str(index): update_lr(0.002, index, budget) for index in indices},
            "LR checkpoints mismatch")
    require(report["diagnosticPrefix"] == expected_prefix(seed, plan["training"]["trainPatches"]),
            "diagnosticPrefix differs from paired eight-batch stream")
    require(re.fullmatch(r"[0-9a-f]{64}", report["streamSha256"]) is not None, "invalid stream hash")
    if smoke:
        require(report["streamSha256"] == stream_hash(report["diagnosticPrefix"]), "smoke stream hash mismatch")
    snapshots = {str(budget)} if smoke else {str(update) for update in plan["training"]["snapshotUpdates"]}
    require(set(report["snapshotPaths"]) == set(report["snapshotSha256"]) == snapshots,
            "snapshot update list mismatch")
    require(set(report["modelPaths"]) == set(report["modelSha256"]) == {"fixedFinal", "bestValidation"},
            "both final and best exports are required")
    require(set(report["fusionMaxAbs"]) == snapshots and
            all(math.isfinite(value) and 0 <= value <= 1e-5 for value in report["fusionMaxAbs"].values()),
            "reported fusion bound failed")
    for key in ("trainingSecondsIncludingValidationAndSnapshots", "processSecondsThroughExport",
                "updatesPerSecondIncludingValidationAndSnapshots"):
        require(math.isfinite(report[key]) and report[key] > 0, f"invalid measured timing: {key}")


def paired_reports(left: dict, right: dict) -> None:
    require({left["rung"], right["rung"]} == {"R0", "R3"}, "paired arms mismatch")
    for key in ("seed", "registrationCommit", "corpusDigest", "smokeOnly", "stepBudget",
                "optimizerSteps", "diagnosticPrefix", "streamSha256", "lrAtIndices", "validationUpdates"):
        require(left[key] == right[key], f"paired report mismatch: {key}")
    require([row["lr"] for row in left["history"]] == [row["lr"] for row in right["history"]],
            "paired history LR mismatch")


def verify_exports(root: Path, report: dict, devices=("cpu", "mps")) -> dict:
    import torch
    from evaluate import load_model
    from train import build_model

    state_path = local_path(root, report["statePath"])
    states = torch.load(state_path, weights_only=True, map_location="cpu")
    state_hash = sha256(state_path)
    optimizer = states["optimizer"]
    parameter_ids = {parameter for group in optimizer["param_groups"] for parameter in group["params"]}
    require(parameter_ids and set(optimizer["state"]) == parameter_ids, "incomplete optimizer state")
    require(all(float(state["step"]) == report["optimizerSteps"] for state in optimizer["state"].values()),
            "optimizer state does not match completed update budget")
    snapshots = {int(update) for update in report["snapshotPaths"]}
    require(set(states["snapshots"]) == snapshots, "state snapshots differ from report")
    checks = {}
    with torch.no_grad():
        initial = build_model(report["rung"], 16, 2, report["seed"])
        for name, tensor in initial.state_dict().items():
            close_tensors(states["initial"][name], tensor, f"saved initial {name}", exact=True)
        exports = [("fixedFinal", report["modelPaths"]["fixedFinal"], states["fixedFinal"],
                    report["optimizerSteps"], "fixed-final", report["modelSha256"]["fixedFinal"]),
                   ("bestValidation", report["modelPaths"]["bestValidation"], states["best"],
                    report["bestUpdate"], "best-validation", report["modelSha256"]["bestValidation"])]
        for update, path in report["snapshotPaths"].items():
            checkpoint = int(update)
            exports.append((f"snapshot{update}", path, states["snapshots"][checkpoint], checkpoint,
                            "fixed-final" if checkpoint == report["optimizerSteps"] else "snapshot",
                            report["snapshotSha256"][update]))
        for name in states["fixedFinal"]:
            close_tensors(states["fixedFinal"][name], states["snapshots"][report["optimizerSteps"]][name],
                          f"fixed-final snapshot {name}", exact=True)
        for label, name, state, update, kind, expected_hash in exports:
            path = local_path(root, name)
            require(sha256(path) == expected_hash, f"export bytes changed: {name}")
            loaded, payload = load_model(str(path))
            require(payload["parameters"] == 6291 and payload["features"] == 16 and payload["depth"] == 2
                    and payload["scale"] == 2, "deployed graph changed")
            for key, value in {"optimizerSteps": report["optimizerSteps"], "stepBudget": report["stepBudget"],
                               "checkpointUpdate": update, "checkpointKind": kind, "rung": report["rung"],
                               "seed": report["seed"], "smokeOnly": report["smokeOnly"],
                               "batch": 32, "trainPatches": 43488,
                               "patchesSeen": report["optimizerSteps"] * 32,
                               "matchedEffectiveInitialization": True,
                               "registrationCommit": report["registrationCommit"],
                               "corpusDigest": report["corpusDigest"]}.items():
                require(payload["training"].get(key) == value, f"export training metadata mismatch: {key}")
            model = build_model(report["rung"], 16, 2, report["seed"]).eval()
            model.load_state_dict(state)
            fused = model.fuse() if hasattr(model, "fuse") else model
            for key, tensor in fused.state_dict().items():
                close_tensors(loaded.state_dict()[key], tensor, f"{label} reloaded weight {key}")
            for device in devices:
                require(device != "mps" or torch.backends.mps.is_available(), "MPS is unavailable")
                model.to(device)
                loaded.to(device)
                errors = [close_tensors(model(inputs.to(device)), loaded(inputs.to(device)),
                                        f"{label} reload {device}") for inputs in diagnostic_inputs()]
                checks[f"{label}.{device}"] = max(errors)
    return {"stateSha256": state_hash, "reloadMaxAbs": checks, "atol": 1e-5, "rtol": 0,
            "bordersIncluded": True, "passed": True}


def code_hashes(root: Path) -> dict:
    names = {f"tools/{name}.py" for name in ("train", "reparam", "aethersr", "dataset", "degrade",
                                            "evaluate", "video-degrade", "test_reparam")}
    for pattern in ("m8*.py", "test_m8*.py"):
        names.update(str(path.relative_to(root)) for path in (root / "tools").glob(pattern))
    require({"tools/m8_experiment.py", "tools/test_m8_experiment.py", "tools/test_m8_training.py"}
            <= names, "missing M8 implementation/tests")
    return {name: sha256(local_path(root, name)) for name in sorted(names)}


def clean_training_sources(root: Path, code: dict) -> None:
    require(not git(root, "status", "--porcelain", "--untracked-files=no").strip(),
            "--train requires a clean tracked worktree")
    for name in code:
        committed_bytes(root, "HEAD", name)


def machine_toolchain() -> dict:
    import platform
    import PIL
    import numpy
    import torch

    def output(*args):
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        require(result.returncode == 0, f"cannot measure toolchain: {args[0]}")
        return result.stdout.strip()

    return {"measuredAt": utc_now(), "cpu": output("sysctl", "-n", "machdep.cpu.brand_string"),
            "memoryBytes": int(output("sysctl", "-n", "hw.memsize")),
            "os": output("sw_vers"), "architecture": platform.machine(),
            "python": platform.python_version(), "pythonExecutable": Path(sys.executable).name,
            "pythonExecutableSha256": sha256(Path(sys.executable)),
            "torch": str(torch.__version__), "numpy": numpy.__version__, "pillow": PIL.__version__,
            "mpsAvailable": torch.backends.mps.is_available()}


def no_concurrent_training(listing: str | None = None, own_pid: int | None = None) -> dict:
    own_pid = os.getpid() if own_pid is None else own_pid
    if listing is None:
        result = subprocess.run(["ps", "-axo", "pid=,ppid=,args="], capture_output=True,
                                text=True, check=False)
        require(result.returncode == 0, "cannot inspect process list; refusing execution")
        listing = result.stdout
    processes = {}
    for line in listing.splitlines():
        fields = line.strip().split(None, 2)
        require(len(fields) >= 2 and fields[0].isdigit() and fields[1].isdigit(),
                "unparseable process listing")
        processes[int(fields[0])] = (int(fields[1]), fields[2] if len(fields) == 3 else "")
    require(own_pid in processes, "own process absent from process listing")
    ancestors = set()
    current = own_pid
    while current in processes and current not in ancestors:
        ancestors.add(current)
        current = processes[current][0]
    pattern = r"(?:^|[/\s])(?:train|m8_experiment|evaluate|m6-validate)\.py(?:\s|$)"
    competing = [pid for pid, (_, command) in processes.items()
                 if pid not in ancestors and re.search(pattern, command)]
    require(not competing, f"concurrent training/evaluation processes: {competing}")
    return {"checkedAt": utc_now(), "competingProcesses": len(competing),
            "selfAndAncestorsExcluded": True}


@contextmanager
def execution_lock(root: Path):
    directory = root / "models"
    directory.mkdir(exist_ok=True)
    lock = directory / ".m8-experiment.lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise GuardError("M8 execution lock exists; investigate it, never silently restart") from error
    try:
        yield
    finally:
        lock.rmdir()


def check_frozen_files(root: Path, inputs: dict) -> None:
    for name, record in inputs["files"].items():
        require(file_record(root, name) == {key: record[key] for key in ("sha256", "bytes")},
                f"frozen input bytes changed: {name}")


def prepared_freeze(root: Path, inputs: dict, registered: dict, *, committed: bool) -> dict:
    if committed:
        committed_bytes(root, "HEAD", INPUT_FREEZE)
    frozen = read_json(root / INPUT_FREEZE)
    require(frozen.get("schema") == "aethervsr.m8-input-freeze/1", "input freeze schema mismatch")
    require(frozen["registrationCommit"] == registered["registrationCommit"], "input freeze registration mismatch")
    require(frozen["inputs"] == inputs, "prepared frozen inputs changed")
    git(root, "merge-base", "--is-ancestor", frozen["executionCommit"], registered["executionCommit"])
    return frozen


def passing_smoke(root: Path, plan: dict, inputs: dict, registered: dict, code: dict) -> dict:
    committed_bytes(root, "HEAD", SMOKE_REPORT)
    smoke = read_json(root / SMOKE_REPORT)
    require(smoke.get("schema") == "aethervsr.m8-smoke/1" and smoke.get("passed") is True
            and smoke.get("smokeOnly") is True, "a committed passing smoke report is required")
    require(smoke["registrationCommit"] == registered["registrationCommit"], "smoke registration mismatch")
    require(smoke["inputDigest"] == inputs["digest"] and smoke["inputFreezeSha256"] == sha256(root / INPUT_FREEZE),
            "smoke input freeze changed")
    require(smoke["codeSha256"] == code, "implementation changed since smoke")
    git(root, "merge-base", "--is-ancestor", smoke["executionCommit"], registered["executionCommit"])
    initial = smoke["initialEquivalence"]
    require(initial["passed"] is True and initial["beforeOptimizerStep"] is True
            and initial["seed"] == 8099 and initial["atol"] == 1e-5 and initial["rtol"] == 0
            and set(initial["checks"]) == {"cpu", "mps"}, "smoke initialization gate missing")
    require(all(checks and all(math.isfinite(error) and 0 <= error <= 1e-5 for error in checks.values())
                for checks in initial["checks"].values()), "smoke initialization bound failed")
    require(len(smoke["runs"]) == len(smoke["chronology"]) == 2, "incomplete paired smoke")
    for rung, report, execution in zip(("R0", "R3"), smoke["runs"], smoke["chronology"], strict=True):
        validate_report(report, plan, registered, 8099, rung, smoke=True)
        verify_execution(root, execution, plan, registered, code, inputs, 8099, rung, smoke=True)
        require(report == read_json(local_path(root, execution["reportPath"])), "committed smoke report changed")
    paired_reports(*smoke["runs"])
    require(smoke["prefixProof"] == prefix_proof(root / plan["pairsDirectory"], smoke["runs"][0]["diagnosticPrefix"]),
            "smoke patch-byte prefix proof changed")
    verify_chronology(smoke["chronology"])
    return smoke


def run_command(plan: dict, registered: dict, seed: int, rung: str, *, smoke: bool,
                root: Path = ROOT) -> list[str]:
    require((smoke and seed == 8099 and rung in ("R0", "R3")) or
            (not smoke and [seed, rung] in RUN_ORDER), "unregistered run")
    training = plan["training"]
    directory = "models/m8-smoke" if smoke else "models/m8"
    stem = f"{directory}/{rung}-seed{seed}"
    interpreter = os.path.relpath(sys.executable, root)
    if "/" not in interpreter:
        interpreter = f"./{interpreter}"
    command = [interpreter, "tools/train.py", "--pairs", plan["pairsDirectory"], "--rung", rung,
               "--epochs", str(1 if smoke else training["epochsUpperBound"]),
               "--max-steps", str(8 if smoke else training["optimizerUpdates"]),
               "--batch", str(training["batch"]), "--lr", str(training["lr"]),
               "--patch", str(training["hrPatch"]), "--seed", str(seed),
               "--channels", str(training["channels"]), "--depth", str(training["depth"]),
               "--device", training["device"], "--out", f"{stem}.json",
               "--m8-audit", f"{stem}-run.json", "--m8-registration", registered["registrationCommit"]]
    return command + (["--m8-smoke"] if smoke else [])


def run_logged(command: list[str], root: Path, stdout: Path, stderr: Path) -> tuple[int, float]:
    started = time.perf_counter()
    with stdout.open("xb") as out_log, stderr.open("xb") as err_log:
        process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env={**os.environ, "PYTHONUNBUFFERED": "1"})
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, (out_log, sys.stdout))
                selector.register(process.stderr, selectors.EVENT_READ, (err_log, sys.stderr))
                while selector.get_map():
                    for key, _ in selector.select():
                        chunk = key.fileobj.read1(65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                            continue
                        log, console = key.data
                        log.write(chunk)
                        log.flush()
                        console.write(chunk.decode("utf-8", errors="replace"))
                        console.flush()
            returncode = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for pipe in (process.stdout, process.stderr):
                if not pipe.closed:
                    pipe.close()
    return returncode, time.perf_counter() - started


def expected_artifacts(command: list[str], report: dict) -> set[str]:
    output = command[command.index("--out") + 1]
    stem = str(Path(output).with_suffix(""))
    require(report["modelPaths"]["bestValidation"] == output and
            report["modelPaths"]["fixedFinal"] == f"{stem}-update{report['optimizerSteps']}.json" and
            report["statePath"] == f"{stem}-states.pt", "trainer output paths changed")
    for update, path in report["snapshotPaths"].items():
        require(path == f"{stem}-update{update}.json", "trainer snapshot path changed")
    return {output, report["statePath"], *report["modelPaths"].values(), *report["snapshotPaths"].values(),
            f"{stem}-run.json", f"{stem}-start.json", f"{stem}-stdout.log", f"{stem}-stderr.log"}


def execute_run(root: Path, plan: dict, registered: dict, inputs: dict, code: dict,
                machine: dict, seed: int, rung: str, *, smoke: bool, run_index: int | None) -> tuple[dict, dict]:
    command = run_command(plan, registered, seed, rung, smoke=smoke, root=root)
    output = command[command.index("--out") + 1]
    stem = str(Path(output).with_suffix(""))
    require(not list((root / stem).parent.glob(Path(stem).name + "*")),
            f"run already has artifacts; refusing overwrite/restart: {stem}")
    require(code_hashes(root) == code, "implementation changed before trainer launch")
    require(git(root, "rev-parse", "HEAD").decode().strip() == registered["executionCommit"],
            "execution commit changed before trainer launch")
    process_check = no_concurrent_training()
    started = time.perf_counter()
    execution = {"schema": "aethervsr.m8-execution/1", "runIndex": run_index, "seed": seed, "rung": rung,
                 "smokeOnly": smoke, "command": command, "startedAt": utc_now(), **registered,
                 "inputDigest": inputs["digest"], "codeSha256": code, "toolchain": machine,
                 "processCheck": process_check, "completed": False, "reportPath": f"{stem}-run.json"}
    write_new(root / f"{stem}-start.json", execution)
    try:
        returncode, seconds = run_logged(command, root, root / f"{stem}-stdout.log", root / f"{stem}-stderr.log")
        execution.update(returncode=returncode, subprocessWallSeconds=seconds,
                         subprocessTimingScope="Popen through wait and log drain; excludes runner verification")
        require(returncode == 0, f"trainer failed with exit code {returncode}; no automatic restart")
        report = read_json(root / execution["reportPath"])
        validate_report(report, plan, registered, seed, rung, smoke=smoke)
        artifacts = expected_artifacts(command, report)
        execution["exportVerification"] = verify_exports(root, report)
        require(code_hashes(root) == code, "code changed during execution")
        require(git(root, "rev-parse", "HEAD").decode().strip() == registered["executionCommit"],
                "execution commit changed during execution")
        check_frozen_files(root, inputs)
        execution["artifacts"] = {name: file_record(root, name) for name in sorted(artifacts)}
        execution["completed"] = True
    except BaseException as error:
        execution["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        execution["finishedAt"] = utc_now()
        execution["executionWallSeconds"] = time.perf_counter() - started
        execution["executionTimingScope"] = "start record through subprocess, artifact and frozen-input verification"
        write_new(root / f"{stem}-execution.json", execution)
    return report, execution


def verify_execution(root: Path, execution: dict, plan: dict, registered: dict, code: dict,
                     inputs: dict, seed: int, rung: str, *, smoke: bool) -> dict:
    require(execution.get("schema") == "aethervsr.m8-execution/1" and execution.get("completed") is True
            and execution.get("returncode") == 0, "incomplete execution; never restart it implicitly")
    require(execution["registrationCommit"] == registered["registrationCommit"]
            and execution["inputDigest"] == inputs["digest"] and execution["codeSha256"] == code,
            "execution provenance changed")
    command = run_command(plan, registered, seed, rung, smoke=smoke, root=root)
    require(execution["command"] == command, "execution command differs from registered run")
    require(execution["reportPath"] == command[command.index("--m8-audit") + 1], "execution report path mismatch")
    report = read_json(local_path(root, execution["reportPath"]))
    validate_report(report, plan, registered, seed, rung, smoke=smoke)
    require(set(execution["artifacts"]) == expected_artifacts(command, report), "missing execution artifacts")
    for name, record in execution["artifacts"].items():
        require(file_record(root, name) == record, f"execution artifact bytes changed: {name}")
    verification = execution["exportVerification"]
    require(verification["passed"] is True and verification["atol"] == 1e-5 and verification["rtol"] == 0
            and verification["stateSha256"] == sha256(local_path(root, report["statePath"]))
            and any(key.endswith(".cpu") for key in verification["reloadMaxAbs"])
            and any(key.endswith(".mps") for key in verification["reloadMaxAbs"])
            and all(math.isfinite(value) and 0 <= value <= 1e-5 for value in verification["reloadMaxAbs"].values()),
            "missing/failed export verification")
    return report


def verify_chronology(executions: list[dict]) -> None:
    previous = None
    for execution in executions:
        started = datetime.fromisoformat(execution["startedAt"])
        finished = datetime.fromisoformat(execution["finishedAt"])
        require(started.tzinfo is not None and finished.tzinfo is not None and finished >= started,
                "invalid execution timestamps")
        require(previous is None or started >= previous, "overlapping or out-of-order executions")
        require(math.isfinite(execution["subprocessWallSeconds"]) and execution["subprocessWallSeconds"] > 0,
                "missing measured subprocess duration")
        previous = finished


def run_directory(root: Path, session: dict, *, smoke: bool, run_index: int | None) -> Path:
    name = "models/m8-smoke" if smoke else "models/m8"
    git(root, "check-ignore", "--quiet", f"{name}/.m8-owner.json")
    directory = local_path(root, name)
    if directory.exists():
        require(not smoke and run_index is not None and run_index > 0,
                f"output directory exists; refusing overwrite: {name}")
        owner = directory / ".m8-owner.json"
        require(owner.is_file() and read_json(owner) == session,
                "existing output directory is not owned by this frozen execution")
    else:
        require(smoke or run_index in (None, 0), "earlier registered runs must complete first")
        directory.mkdir(parents=True)
        write_new(directory / ".m8-owner.json", session)
    return directory


def load_runs(root: Path, plan: dict, registered: dict, inputs: dict, code: dict,
              count: int) -> tuple[list, list]:
    runs, chronology = [], []
    for index, (seed, rung) in enumerate(plan["runOrder"][:count]):
        path = root / f"models/m8/{rung}-seed{seed}-execution.json"
        require(path.is_file(), f"missing preceding completed run {index}")
        execution = read_json(path)
        require(execution["runIndex"] == index, "execution index mismatch")
        runs.append(verify_execution(root, execution, plan, registered, code, inputs, seed, rung, smoke=False))
        chronology.append(execution)
    verify_chronology(chronology)
    for index in range(0, len(runs) - 1, 2):
        paired_reports(runs[index], runs[index + 1])
    return runs, chronology


def main(argv: list[str] | None = None, *, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true", help="inventory existing frozen inputs only")
    action.add_argument("--smoke", action="store_true", help="run the two non-candidate MPS smoke arms")
    action.add_argument("--train", action="store_true", help="run only the six registered paired candidates")
    parser.add_argument("--plan", default=PLAN, help="must be byte-identical to the committed M8 plan")
    parser.add_argument("--run-index", type=int, choices=range(6), help="one serial run; --train only")
    args = parser.parse_args(argv)
    if args.run_index is not None and not args.train:
        parser.error("--run-index requires --train")
    try:
        plan_path = Path(args.plan)
        if plan_path.is_absolute():
            require(plan_path.resolve().is_relative_to(root.resolve()), "plan must be inside repository")
        else:
            plan_path = local_path(root, args.plan)
        registered = registration(root, plan_path)
        plan = read_json(plan_path)
        code = code_hashes(root)
        target = INPUT_FREEZE if args.prepare else SMOKE_REPORT if args.smoke else RUNS_REPORT
        require(not (root / target).exists(), f"refusing to overwrite {target}")
        if args.train:
            clean_training_sources(root, code)
        if args.smoke:
            require(not (root / "models/m8-smoke").exists(), "smoke output directory already exists")
        inputs = freeze_inputs(root, plan, registered)
        machine = machine_toolchain()
        if args.prepare:
            write_new(root / INPUT_FREEZE, {"schema": "aethervsr.m8-input-freeze/1", **registered,
                                          "createdAt": utc_now(), "toolchain": machine,
                                          "codeSha256": code, "inputs": inputs})
            print(f"Prepared {INPUT_FREEZE}: {inputs['digest']}")
            return 0
        prepared_freeze(root, inputs, registered, committed=args.train)
        require(machine["mpsAvailable"], "MPS is unavailable; no CPU fallback")
        if args.train:
            passing_smoke(root, plan, inputs, registered, code)
        with execution_lock(root):
            no_concurrent_training()
            session = {**registered, "inputDigest": inputs["digest"], "codeSha256": code}
            if args.train and args.run_index:
                load_runs(root, plan, registered, inputs, code, args.run_index)
            run_directory(root, session, smoke=args.smoke, run_index=args.run_index)
            if args.smoke:
                initial = initial_equivalence(plan["smokeSeed"])
                runs, chronology = [], []
                for rung in plan["arms"]:
                    require(freeze_inputs(root, plan, registered) == inputs, "inputs changed before smoke launch")
                    report, execution = execute_run(root, plan, registered, inputs, code, machine,
                                                    plan["smokeSeed"], rung, smoke=True, run_index=None)
                    runs.append(report)
                    chronology.append(execution)
                paired_reports(*runs)
                verify_chronology(chronology)
                proof = prefix_proof(root / plan["pairsDirectory"], runs[0]["diagnosticPrefix"])
                write_new(root / SMOKE_REPORT, {"schema": "aethervsr.m8-smoke/1", **registered,
                                               "createdAt": utc_now(), "smokeOnly": True, "passed": True,
                                               "inputDigest": inputs["digest"],
                                               "inputFreezeSha256": sha256(root / INPUT_FREEZE),
                                               "codeSha256": code, "toolchain": machine,
                                               "initialEquivalence": initial, "prefixProof": proof,
                                               "runs": runs, "chronology": chronology})
                print(f"Passing non-candidate smoke: {SMOKE_REPORT}; commit evidence and implementation before --train")
                return 0
            indices = range(6) if args.run_index is None else [args.run_index]
            for index in indices:
                clean_training_sources(root, code)
                require(code_hashes(root) == code, "implementation changed before launch")
                require(freeze_inputs(root, plan, registered) == inputs, "inputs changed before training launch")
                load_runs(root, plan, registered, inputs, code, index)
                seed, rung = plan["runOrder"][index]
                execute_run(root, plan, registered, inputs, code, machine, seed, rung,
                            smoke=False, run_index=index)
                load_runs(root, plan, registered, inputs, code, index + 1)
            if args.run_index in (None, 5):
                runs, chronology = load_runs(root, plan, registered, inputs, code, 6)
                write_new(root / RUNS_REPORT, {"schema": "aethervsr.m8-runs/1", **registered,
                                              "createdAt": utc_now(), "inputDigest": inputs["digest"],
                                              "inputFreezeSha256": sha256(root / INPUT_FREEZE),
                                              "smokeSha256": sha256(root / SMOKE_REPORT),
                                              "codeSha256": code, "runs": runs, "chronology": chronology})
                print(f"Six verified serial runs: {RUNS_REPORT}")
            return 0
    except (GuardError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"M8 blocked: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())