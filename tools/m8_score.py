"""Frozen M8 CPU-reference scoring; supersedes the withdrawn MPS apparatus."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

import evaluate
import m8_experiment as experiment
import m8_report as report


ROOT = Path(__file__).resolve().parents[1]
CRFS = [18, 26, 34]
PARITY = "results/m8-cpu-scoring-parity-forward.json"
REVERSE_PARITY = "results/m8-cpu-scoring-parity-reverse.json"
REPAIR = "docs/M8-SCORING-REPAIR.md"
SCORING_ONLY = {"tools/m8_score.py", "tools/test_m8_score.py"}
CPU_THREADS = 4
SCORES = "results/m8-scores.json"
TOLERANCE = 1e-5
specification = importlib.util.spec_from_file_location("m8_m6_validate", ROOT / "tools/m6-validate.py")
m6 = importlib.util.module_from_spec(specification)
specification.loader.exec_module(m6)
require = experiment.require


def code_hashes(root: Path) -> dict:
    names = ("m8_score.py", "test_m8_score.py", "m6-validate.py", "evaluate.py", "m8_report.py")
    return {**experiment.code_hashes(root), **{
        f"tools/{name}": experiment.sha256(root / "tools" / name) for name in names
    }}


def frozen_inputs(root: Path, plan: dict, registered: dict) -> dict:
    inputs = experiment.freeze_inputs(root, plan, registered)
    experiment.prepared_freeze(root, inputs, registered, committed=True)
    cells = inputs["validationCells"]
    require(len(cells) == 48, "exactly 48 validation cells required")
    expected = set()
    for cell in cells:
        require(len(cell["frames"]) == 8, "exactly eight paired frames required")
        for frame in cell["frames"]:
            expected.update((frame["hr"], frame["lr"]))
    cache = experiment.local_path(root, plan["validationCache"])
    actual = {str(path.relative_to(root)) for path in cache.rglob("*.png")}
    require(actual == expected and len(expected) == 512, "missing or extra frozen images")
    return inputs


def check_models(root: Path, artifacts: dict) -> None:
    for artifact in artifacts.values():
        report.require_hash(artifact["sha256"], "model file hash")
        path = experiment.local_path(root, artifact["path"])
        require(path.is_file() and experiment.sha256(path) == artifact["sha256"],
                f"model bytes changed: {artifact['path']}")


def model_artifacts(root: Path, plan: dict, manifest: dict) -> dict:
    runs = report.validate_runs(manifest)
    artifacts = {report.PRODUCTION_KEY: {
        "path": plan["productionModel"], "sha256": plan["productionModelSha256"],
    }}
    require(plan["productionModelSha256"] == report.PRODUCTION_SHA256, "production hash mismatch")
    for run in runs.values():
        require(run.get("smokeOnly") is False, "smoke model cannot enter binding scores")
        steps = {str(update) for update in report.SNAPSHOT_UPDATES}
        require(set(run.get("snapshotPaths", {})) == set(run.get("snapshotSha256", {})) ==
                set(run.get("snapshotKeys", {})) == steps, "all registered snapshots required")
        require(run["snapshotPaths"]["81180"] == run["modelPaths"]["fixedFinal"] and
                run["snapshotSha256"]["81180"] == run["modelSha256"]["fixedFinal"],
                "fixed-final snapshot mismatch")
        entries = [(run["modelKeys"][kind], run["modelPaths"][kind], run["modelSha256"][kind])
                   for kind in report.CHECKPOINTS]
        entries.extend((run["snapshotKeys"][update], run["snapshotPaths"][update],
                        run["snapshotSha256"][update]) for update in sorted(steps) if update != "81180")
        for key, name, digest in entries:
            require(key not in artifacts and name not in {entry["path"] for entry in artifacts.values()},
                    "duplicate model path/key")
            artifacts[key] = {"path": name, "sha256": digest}
    require(len(artifacts) == 49, "expected 49 explicit model paths, never a directory glob")
    check_models(root, artifacts)
    return artifacts


def training_sources(root: Path, manifest: dict) -> dict:
    source = manifest["codeSha256"]
    current = experiment.code_hashes(root)
    require(set(source) == set(current), "training source inventory changed")
    for name, digest in source.items():
        recorded = experiment.git(root, "show", f"{manifest['executionCommit']}:{name}")
        require(hashlib.sha256(recorded).hexdigest() == digest,
                f"historical training source hash mismatch: {name}")
        require(name in SCORING_ONLY or current[name] == digest,
                f"non-scoring source changed since training: {name}")
    return source


def binding_runs(root: Path, plan: dict, registered: dict, inputs: dict) -> tuple[dict, dict]:
    experiment.committed_bytes(root, "HEAD", experiment.RUNS_REPORT)
    manifest = experiment.read_json(root / experiment.RUNS_REPORT)
    require(manifest["registrationCommit"] == registered["registrationCommit"] and
            manifest["inputDigest"] == inputs["digest"], "training provenance mismatch")
    source = training_sources(root, manifest)
    experiment.passing_smoke(root, plan, inputs, registered, source)
    runs, chronology = experiment.load_runs(root, plan, registered, inputs, source, 6)
    require(manifest["runs"] == runs and manifest["chronology"] == chronology,
            "committed run evidence differs from verified execution")
    for execution in chronology:
        experiment.git(root, "merge-base", "--is-ancestor", registered["registrationCommit"],
                       execution["executionCommit"])
        experiment.git(root, "merge-base", "--is-ancestor", execution["executionCommit"],
                       registered["executionCommit"])
        for name in source:
            original = experiment.git(root, "show", f"{execution['executionCommit']}:{name}")
            require(hashlib.sha256(original).hexdigest() == source[name],
                f"execution source hash mismatch: {name}")
    return manifest, model_artifacts(root, plan, manifest)


def metrics(output: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    require(output.shape == reference.shape and output.dtype == reference.dtype == torch.float32,
            "scoring tensor shape/dtype mismatch")
    require(output.device.type == reference.device.type == "cpu", "metrics require the CPU reference")
    require(torch.isfinite(output).all().item() and torch.isfinite(reference).all().item(),
            "nonfinite scoring tensors")
    values = evaluate.psnr(output, reference), evaluate.ssim(output, reference)
    require(all(math.isfinite(value) for value in values), "nonfinite scoring metrics")
    return values


def tensor_identity(model: torch.nn.Module) -> str:
    digest = hashlib.sha256(type(model).__qualname__.encode())
    for name, tensor in sorted(model.state_dict().items()):
        require(tensor.device.type == "cpu" and tensor.dtype == torch.float32,
                "model tensors require CPU float32")
        digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def duplicate_check(reference: torch.Tensor, actual: torch.Tensor,
                    expected_metrics: tuple[float, float], actual_metrics: tuple[float, float]) -> dict:
    differences = {"outputMaxAbs": (reference - actual).abs().max().item(),
                   "psnrMaxAbs": abs(expected_metrics[0] - actual_metrics[0]),
                   "ssimMaxAbs": abs(expected_metrics[1] - actual_metrics[1])}
    require(all(math.isfinite(value) and value <= TOLERANCE for value in differences.values()),
            f"identical weights produced inconsistent outputs/metrics: {differences}")
    return differences


def collect(root: Path, cells: list[dict], models: dict, device: str) -> dict:
    require(device == "cpu", "binding scores require the CPU reference; MPS scoring is unreliable")
    identities = {key: tensor_identity(model) for key, model in models.items()}
    groups = {identity: [key for key, value in identities.items() if value == identity]
              for identity in set(identities.values())}
    groups = {identity: keys for identity, keys in groups.items() if len(keys) > 1}
    duplicate_maxima = {"outputMaxAbs": 0.0, "psnrMaxAbs": 0.0, "ssimMaxAbs": 0.0}
    comparisons = 0
    grouped = {}
    for position, cell in enumerate(cells):
        for frame in cell["frames"]:
            grouped.setdefault(frame["hr"], []).append((position, frame["lr"]))
    accumulated = {key: [[] for _ in cells] for key in (None, *models)}
    with torch.no_grad():
        for frame_index, (hr_path, pairs) in enumerate(grouped.items()):
            reference = m6.load_png(str(root / hr_path)).to(device)
            for position, lr_path in pairs:
                inputs = m6.load_png(str(root / lr_path)).to(device)
                accumulated[None][position].append(metrics(evaluate.catmull_rom_2x(inputs), reference))
                duplicate_outputs = {}
                for key, model in models.items():
                    output = model(inputs).clamp(0, 1)
                    values = metrics(output, reference)
                    accumulated[key][position].append(values)
                    identity = identities[key]
                    if identity in groups:
                        if identity in duplicate_outputs:
                            first_output, first_metrics = duplicate_outputs[identity]
                            differences = duplicate_check(first_output, output, first_metrics, values)
                            comparisons += 1
                            for metric, value in differences.items():
                                duplicate_maxima[metric] = max(duplicate_maxima[metric], value)
                        else:
                            duplicate_outputs[identity] = (output, values)
            if frame_index % 8 == 7:
                print(f"scored {frame_index + 1}/{len(grouped)} reference frames, {len(models)} models", flush=True)
    rows = {}
    for key, values in accumulated.items():
        rows[key] = []
        for cell, samples in zip(cells, values, strict=True):
            require(len(samples) == 8, "every cell must have exactly eight scored frames")
            rows[key].append({"clip": cell["clip"], "category": cell["category"], "crf": cell["crf"],
                              "psnr": statistics.fmean(value[0] for value in samples),
                              "ssim": statistics.fmean(value[1] for value in samples)})
    baseline = rows.pop(None)
    return {"schema": "aethervsr.m6-validation/1", "crfs": CRFS, "frames": 8,
            "clips": sorted({cell["clip"] for cell in cells}),
            "statisticalUnit": "clip; mean frame PSNR per cell, equal tiers and clips",
            "baseline": {"name": "catmull_rom", "perCell": baseline},
            "models": {key: m6.summarise(value, baseline, CRFS) for key, value in rows.items()},
            "frameScoreMetricOrder": ["psnr", "ssim"],
            "frameScoreCellOrder": [{"clip": cell["clip"], "crf": cell["crf"],
                          "frames": [frame["id"] for frame in cell["frames"]]} for cell in cells],
            "frameScores": {"catmull_rom" if key is None else key: values
                    for key, values in accumulated.items()},
            "tensorSha256": identities,
            "duplicateWeightVerification": {"passed": True, "atol": TOLERANCE, "rtol": 0,
                             "groups": groups, "perFrameComparisons": comparisons,
                             "maxima": duplicate_maxima,
                             "independentlyEvaluated": True}}


def load_models(root: Path, artifacts: dict, device: str) -> dict:
    require(device == "cpu", "models require the CPU reference")
    models = {}
    for key, artifact in artifacts.items():
        model, payload = evaluate.load_model(str(experiment.local_path(root, artifact["path"])))
        require((payload.get("features"), payload.get("depth"), payload.get("parameters"),
                 payload.get("scale")) == (16, 2, 6291, 2), "not a deployed C16D2 model")
        require(all(torch.isfinite(tensor).all().item() for tensor in model.parameters()),
                "nonfinite model parameters")
        models[key] = model.eval().to(device)
    return models


def apparatus_parity(root: Path, inputs: dict, artifacts: dict, *, reverse: bool) -> dict:
    keys = [report.PRODUCTION_KEY, "R0-seed8101", "R0-seed8101-update81180",
            "R0-seed8102", "R0-seed8102-update81180", "R3-seed8101", "R3-seed8101-update10824"]
    selected = {key: artifacts[key] for key in (reversed(keys) if reverse else keys)}
    cells = [cell for cell in inputs["validationCells"] if cell["category"] == "text" or
             (cell["category"] == "motion" and cell["crf"] == 34)]
    require(cells and {cell["crf"] for cell in cells} == set(CRFS), "missing registered parity cells")
    measured = collect(root, cells, load_models(root, selected, "cpu"), "cpu")
    original_cells = [{"clip": cell["clip"], "category": cell["category"], "crf": cell["crf"],
                       "hr": str((root / cell["frames"][0]["hr"]).parent),
                       "lr": str((root / cell["frames"][0]["lr"]).parent)} for cell in cells]
    checks = {}
    for key in ("catmull_rom", *selected):
        original = m6.score(None if key == "catmull_rom" else str(root / selected[key]["path"]),
                            original_cells)["perCell"]
        actual = (measured["baseline"] if key == "catmull_rom" else measured["models"][key])["perCell"]
        differences = {metric: max(abs(left[metric] - right[metric])
                                   for left, right in zip(original, actual, strict=True))
                       for metric in ("psnr", "ssim")}
        require(all(value <= TOLERANCE for value in differences.values()),
                f"CPU collection differs from original M6 evaluator: {key}: {differences}")
        checks[key] = differences
    require(len(measured["duplicateWeightVerification"]["groups"]) == 3,
            "the three known duplicate-weight pairs must be independently checked")
    return {"checks": checks, "passed": True, "atol": TOLERANCE, "rtol": 0,
            "bordersIncluded": True, "reverseModelOrder": reverse,
            "cells": [{key: cell[key] for key in ("clip", "category", "crf")} for cell in cells],
            "models": selected, "duplicateWeightVerification": measured["duplicateWeightVerification"],
            "metricDigest": experiment.aggregate({key: measured[key] for key in
                                                   ("baseline", "models", "frameScores")}),
            "scope": "post-withdrawal CPU apparatus; original M6 evaluator, all text cells and motion CRF34"}


def validate_parity(evidence: dict, registered: dict, inputs: dict, code: dict) -> None:
    require(evidence.get("schema") == "aethervsr.m8-cpu-scoring-parity/1" and evidence.get("passed") is True,
            "passing scoring parity required; withdrawn MPS parity is not eligible")
    require(evidence["registrationCommit"] == registered["registrationCommit"] and
            evidence["preInputDigest"] == evidence["postInputDigest"] == inputs["digest"] and
            evidence["codeSha256"] == code and evidence["productionSha256"] == report.PRODUCTION_SHA256 and
            evidence["device"] == "cpu" and evidence["cpuThreads"] == CPU_THREADS and
            evidence["deterministicAlgorithms"] is True,
            "scoring parity provenance changed")
    require(len(evidence["checks"]) == 8, "incomplete scoring parity")
    require(evidence["atol"] == TOLERANCE and evidence["rtol"] == 0, "parity tolerance changed")
    for check in evidence["checks"].values():
        require(all(math.isfinite(check[key]) and 0 <= check[key] <= TOLERANCE
                    for key in ("psnr", "ssim")), "failed parity check")
    duplicate = evidence["duplicateWeightVerification"]
    require(duplicate["passed"] is True and duplicate["independentlyEvaluated"] is True and
            len(duplicate["groups"]) == 3 and duplicate["perFrameComparisons"] > 0 and
            all(0 <= value <= TOLERANCE for value in duplicate["maxima"].values()),
            "duplicate-weight parity missing or failed")


def validate_parity_pair(forward: dict, reverse: dict) -> None:
    require(forward["reverseModelOrder"] is False and reverse["reverseModelOrder"] is True and
            forward["metricDigest"] == reverse["metricDigest"] and forward["cells"] == reverse["cells"],
            "independent CPU parity runs or reversed model order disagree")


def validate_frame_scores(scores: dict, cells: list[dict]) -> None:
    expected_order = [{"clip": cell["clip"], "crf": cell["crf"],
                       "frames": [frame["id"] for frame in cell["frames"]]} for cell in cells]
    require(scores["frameScoreMetricOrder"] == ["psnr", "ssim"] and
            scores["frameScoreCellOrder"] == expected_order, "frame-score identity/order mismatch")
    require(set(scores["frameScores"]) == {"catmull_rom", *scores["models"]},
            "missing model frame scores")
    for key, frames_by_cell in scores["frameScores"].items():
        rows = (scores["baseline"] if key == "catmull_rom" else scores["models"][key])["perCell"]
        require(len(frames_by_cell) == len(rows) == len(cells), "incomplete frame-score cells")
        for cell, frames, row in zip(cells, frames_by_cell, rows, strict=True):
            require(len(frames) == 8 and all(len(values) == 2 and
                    all(type(value) in (int, float) and math.isfinite(value) for value in values)
                    for values in frames), "incomplete or nonfinite per-frame metrics")
            require(all(row[name] == cell[name] for name in ("clip", "category", "crf")),
                    "per-frame and aggregate cell identities differ")
            for position, metric in enumerate(("psnr", "ssim")):
                require(row[metric] == statistics.fmean(values[position] for values in frames),
                        f"cell {metric} differs from its eight frame metrics")


def publish_report(root: Path, inputs: dict, manifest: dict, artifacts: dict,
           registered: dict, source: dict, repair_sha: str) -> dict:
    experiment.committed_bytes(root, "HEAD", SCORES)
    scores = experiment.read_json(root / SCORES)
    require(scores.get("device") == "cpu" and scores.get("dtype") == "float32" and
        scores.get("cpuThreads") == CPU_THREADS and scores.get("deterministicAlgorithms") is True,
        "publication requires CPU-reference scores; withdrawn MPS scores are ineligible")
    require(scores["registrationCommit"] == registered["registrationCommit"] and
        scores["preInputDigest"] == scores["postInputDigest"] == inputs["digest"] and
        scores["codeSha256"] == source and scores["scoringRepairSha256"] == repair_sha and
        scores["modelArtifacts"] == artifacts and scores["trainingCodeSha256"] == manifest["codeSha256"],
        "CPU score provenance mismatch")
    parity_hashes = {name: experiment.sha256(root / name) for name in (PARITY, REVERSE_PARITY)}
    require(scores["paritySha256"] == parity_hashes, "CPU score parity evidence changed")
    validate_frame_scores(scores, inputs["validationCells"])
    identities = {key: tensor_identity(model) for key, model in load_models(root, artifacts, "cpu").items()}
    require(scores["tensorSha256"] == identities, "scored tensor identities changed")
    duplicate = scores["duplicateWeightVerification"]
    groups = {identity: [key for key, value in identities.items() if value == identity]
          for identity in set(identities.values())}
    groups = {identity: keys for identity, keys in groups.items() if len(keys) > 1}
    require(duplicate["passed"] is True and duplicate["independentlyEvaluated"] is True and
        duplicate["groups"] == groups and duplicate["atol"] == TOLERANCE and duplicate["rtol"] == 0 and
        duplicate["perFrameComparisons"] == sum(len(keys) - 1 for keys in groups.values()) * 48 * 8 and
        all(math.isfinite(value) and 0 <= value <= TOLERANCE for value in duplicate["maxima"].values()),
        "binding duplicate-weight verification mismatch")
    for keys in groups.values():
        for key in keys[1:]:
            require(scores["frameScores"][key] == scores["frameScores"][keys[0]],
                    "identical CPU weights have different frame scores")
    categories = {cell["clip"]: cell["category"] for cell in inputs["validationCells"]}
    result = report.build_report(scores, manifest, categories)
    result["scope"] = "Measured frozen captured CPU validation; provenance and frame aggregates verified before publication."
    result["publication"] = {
    "passed": True, "scores": SCORES, "scoresSha256": experiment.sha256(root / SCORES),
    "runs": experiment.RUNS_REPORT, "runsSha256": experiment.sha256(root / experiment.RUNS_REPORT),
    "paritySha256": parity_hashes, "repairSha256": repair_sha,
    "frameAggregatesVerified": True, "duplicateWeightVerification": duplicate,
    "withdrawnMpsEvidenceEligible": False,
    }
    result["limitations"] = [
    "Three paired seeds cannot establish conventional significance; sign-flip minimum p is 0.25.",
    "Historical MPS patch-validation selections are retained unchanged; best-validation is secondary, not CPU reselected.",
    "Category and horizon comparisons are descriptive, not additional rejection opportunities.",
    "M7 differs in seeds, identity initialization and cosine horizon; no convergence or monotonicity is established.",
    "Recovered training cache is shared across M8 arms but not claimed byte-identical to lost M6 cache.",
    ]
    return result


def main(argv: list[str] | None = None, *, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--parity-out", type=Path)
    action.add_argument("--out", type=Path)
    action.add_argument("--report-out", type=Path)
    parser.add_argument("--reverse-model-order", action="store_true")
    args = parser.parse_args(argv)
    try:
        target = args.parity_out or args.out or args.report_out
        target = target if target.is_absolute() else root / target
        require(target.resolve().is_relative_to(root.resolve()), "output must remain inside repository")
        require(not target.exists(), "refusing to overwrite scoring evidence")
        publishing = args.report_out is not None
        binding = args.out is not None or publishing
        require(not binding or not args.reverse_model_order, "reversal is only a parity diagnostic")
        require(not binding or target == root / ("results/m8-report.json" if publishing else SCORES),
            "binding scores/report require the registered output")
        if not binding:
            require(target == root / (REVERSE_PARITY if args.reverse_model_order else PARITY),
                "parity requires the registered forward/reverse output")
        registered = experiment.registration(root, root / experiment.PLAN)
        plan = experiment.read_json(root / experiment.PLAN)
        source = code_hashes(root)
        experiment.clean_training_sources(root, source)
        experiment.committed_bytes(root, "HEAD", REPAIR)
        repair_sha = experiment.sha256(root / REPAIR)
        torch.set_num_threads(CPU_THREADS)
        torch.use_deterministic_algorithms(True)
        started = time.perf_counter()
        with experiment.execution_lock(root):
            experiment.no_concurrent_training()
            inputs = frozen_inputs(root, plan, registered)
            manifest, artifacts = binding_runs(root, plan, registered, inputs)
            if binding:
                proofs = []
                for name in (PARITY, REVERSE_PARITY):
                    experiment.committed_bytes(root, "HEAD", name)
                    parity = experiment.read_json(root / name)
                    validate_parity(parity, registered, inputs, source)
                    require(parity["scoringRepairSha256"] == repair_sha, "scoring repair changed since parity")
                    proofs.append(parity)
                validate_parity_pair(*proofs)
                if publishing:
                    result = publish_report(root, inputs, manifest, artifacts, registered, source, repair_sha)
                else:
                    models = load_models(root, artifacts, "cpu")
                    result = collect(root, inputs["validationCells"], models, "cpu")
                    validate_frame_scores(result, inputs["validationCells"])
                    categories = {cell["clip"]: cell["category"] for cell in inputs["validationCells"]}
                    report.build_report(result, manifest, categories)
            else:
                result = {"schema": "aethervsr.m8-cpu-scoring-parity/1",
                          **apparatus_parity(root, inputs, artifacts, reverse=args.reverse_model_order),
                          "productionSha256": plan["productionModelSha256"]}
            after = frozen_inputs(root, plan, registered)
            require(after == inputs and code_hashes(root) == source, "inputs/source changed during scoring")
            check_models(root, artifacts)
            require(experiment.git(root, "rev-parse", "HEAD").decode().strip() == registered["executionCommit"],
                    "HEAD changed during scoring")
            experiment.clean_training_sources(root, source)
            result.update(registered)
            result.update(preInputDigest=inputs["digest"], postInputDigest=after["digest"],
                          codeSha256=source, modelArtifacts=artifacts, toolchain=experiment.machine_toolchain(),
                          createdAt=experiment.utc_now(), device="cpu", dtype="float32",
                          cpuThreads=CPU_THREADS, deterministicAlgorithms=True,
                          fallback=False, manifest=plan["validationManifest"],
                          scoringRepair=REPAIR, scoringRepairSha256=repair_sha,
                          trainingExecutionCommit=manifest["executionCommit"],
                          trainingCodeSha256=manifest["codeSha256"],
                          scoringSecondsIncludingVerification=time.perf_counter() - started)
            if binding:
                result["paritySha256"] = {name: experiment.sha256(root / name)
                                         for name in (PARITY, REVERSE_PARITY)}
            experiment.write_new(target, result)
            require(result.get("passed", True), "scoring parity failed; evidence retained")
        kind = "verified report" if publishing else "binding scores" if binding else "apparatus parity"
        print(f"Wrote {kind}: {target}")
        return 0
    except (experiment.GuardError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"M8 scoring blocked: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())