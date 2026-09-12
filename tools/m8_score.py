"""Frozen M8 validation scoring with a pre-candidate CPU/MPS parity gate."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
import statistics
import sys

import torch

import evaluate
import m8_experiment as experiment
import m8_report as report


ROOT = Path(__file__).resolve().parents[1]
CRFS = [18, 26, 34]
PARITY = "results/m8-scoring-parity.json"
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


def binding_runs(root: Path, plan: dict, registered: dict, inputs: dict) -> tuple[dict, dict]:
    experiment.committed_bytes(root, "HEAD", experiment.RUNS_REPORT)
    manifest = experiment.read_json(root / experiment.RUNS_REPORT)
    require(manifest["registrationCommit"] == registered["registrationCommit"] and
            manifest["inputDigest"] == inputs["digest"], "training provenance mismatch")
    source = experiment.code_hashes(root)
    require(manifest["codeSha256"] == source, "source changed since training")
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
            experiment.committed_bytes(root, execution["executionCommit"], name)
    return manifest, model_artifacts(root, plan, manifest)


def metrics(output: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    require(output.shape == reference.shape and output.dtype == reference.dtype == torch.float32,
            "scoring tensor shape/dtype mismatch")
    require(torch.isfinite(output).all().item() and torch.isfinite(reference).all().item(),
            "nonfinite scoring tensors")
    values = evaluate.psnr(output, reference), evaluate.ssim(output, reference)
    require(all(math.isfinite(value) for value in values), "nonfinite scoring metrics")
    return values


def collect(root: Path, cells: list[dict], models: dict, device: str) -> dict:
    grouped = {}
    for position, cell in enumerate(cells):
        for frame in cell["frames"]:
            grouped.setdefault(frame["hr"], []).append((position, frame["lr"]))
    accumulated = {key: [[] for _ in cells] for key in (None, *models)}
    with torch.no_grad():
        for frame_index, (hr_path, pairs) in enumerate(grouped.items()):
            if device == "mps":
                experiment.no_concurrent_training()
            reference = m6.load_png(str(root / hr_path)).to(device)
            for position, lr_path in pairs:
                inputs = m6.load_png(str(root / lr_path)).to(device)
                accumulated[None][position].append(metrics(evaluate.catmull_rom_2x(inputs), reference))
                for key, model in models.items():
                    accumulated[key][position].append(metrics(model(inputs).clamp(0, 1), reference))
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
            "models": {key: m6.summarise(value, baseline, CRFS) for key, value in rows.items()}}


def load_models(root: Path, artifacts: dict, device: str) -> dict:
    models = {}
    for key, artifact in artifacts.items():
        model, payload = evaluate.load_model(str(experiment.local_path(root, artifact["path"])))
        require((payload.get("features"), payload.get("depth"), payload.get("parameters"),
                 payload.get("scale")) == (16, 2, 6291, 2), "not a deployed C16D2 model")
        require(all(torch.isfinite(tensor).all().item() for tensor in model.parameters()),
                "nonfinite model parameters")
        models[key] = model.eval().to(device)
    return models


def parity_case(model, inputs: torch.Tensor, reference: torch.Tensor) -> dict:
    with torch.no_grad():
        original = model.cpu()(inputs).clamp(0, 1) if model is not None else m6.catmull_rom_2x(inputs)
        original_metrics = (m6.psnr(original, reference), m6.ssim(original, reference))
        proposed = (model.to("mps")(inputs.to("mps")).clamp(0, 1) if model is not None
                    else evaluate.catmull_rom_2x(inputs.to("mps")))
        proposed_metrics = metrics(proposed, reference.to("mps"))
        error = (original - proposed.cpu()).abs().max().item()
    differences = [abs(left - right) for left, right in zip(original_metrics, proposed_metrics, strict=True)]
    return {"inputShape": list(inputs.shape), "referenceShape": list(reference.shape),
            "cpu": dict(zip(("psnr", "ssim"), original_metrics)),
            "mps": dict(zip(("psnr", "ssim"), proposed_metrics)), "maxPixelAbs": error,
            "psnrDelta": differences[0], "ssimDelta": differences[1],
            "passed": all(math.isfinite(value) and value <= TOLERANCE for value in (error, *differences))}


def apparatus_parity(root: Path, plan: dict, inputs: dict) -> dict:
    from aethersr import AetherSR

    frame = inputs["validationCells"][0]["frames"][0]
    reference = m6.load_png(str(root / frame["hr"]))
    low_resolution = m6.load_png(str(root / frame["lr"]))
    production, _ = evaluate.load_model(str(root / plan["productionModel"]))
    checks = {"productionFull": parity_case(production, low_resolution, reference),
              "catmullFull": parity_case(None, low_resolution, reference)}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20260911)
        synthetic = AetherSR(channels=4, depth=1).eval()
    generator = torch.Generator().manual_seed(20260911)
    low_resolution = torch.rand((1, 3, 13, 17), generator=generator)
    reference = torch.rand((1, 3, 26, 34), generator=generator)
    checks["syntheticOdd"] = parity_case(synthetic, low_resolution, reference)
    checks["catmullOdd"] = parity_case(None, low_resolution, reference)
    return {"checks": checks, "passed": all(check["passed"] for check in checks.values()),
            "frame": frame, "atol": TOLERANCE, "rtol": 0, "bordersIncluded": True,
            "scope": "apparatus only, production and seeded synthetic model; no M8 candidate"}


def validate_parity(evidence: dict, registered: dict, inputs: dict, code: dict) -> None:
    require(evidence.get("schema") == "aethervsr.m8-scoring-parity/1" and evidence.get("passed") is True,
            "passing scoring parity required")
    require(evidence["registrationCommit"] == registered["registrationCommit"] and
            evidence["preInputDigest"] == evidence["postInputDigest"] == inputs["digest"] and
            evidence["codeSha256"] == code and evidence["productionSha256"] == report.PRODUCTION_SHA256,
            "scoring parity provenance changed")
    require(set(evidence["checks"]) == {"productionFull", "catmullFull", "syntheticOdd", "catmullOdd"},
            "incomplete scoring parity")
    require(evidence["atol"] == TOLERANCE and evidence["rtol"] == 0, "parity tolerance changed")
    for check in evidence["checks"].values():
        require(check["passed"] is True and all(math.isfinite(check[key]) and 0 <= check[key] <= TOLERANCE
                for key in ("maxPixelAbs", "psnrDelta", "ssimDelta")), "failed parity check")


def main(argv: list[str] | None = None, *, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--parity-out", type=Path)
    action.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        target = args.parity_out if args.parity_out is not None else args.out
        target = target if target.is_absolute() else root / target
        require(target.resolve().is_relative_to(root.resolve()), "output must remain inside repository")
        require(not target.exists(), "refusing to overwrite scoring evidence")
        binding = args.out is not None
        require(not binding or target == root / SCORES, "binding scores require the registered output")
        registered = experiment.registration(root, root / experiment.PLAN)
        plan = experiment.read_json(root / experiment.PLAN)
        source = code_hashes(root)
        experiment.clean_training_sources(root, source)
        require(torch.backends.mps.is_available(), "MPS unavailable; no silent device fallback")
        require(all(os.environ.get(name, "0") == "0" for name in
                    ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH")), "no MPS fallback/fast-math")
        with experiment.execution_lock(root):
            experiment.no_concurrent_training()
            inputs = frozen_inputs(root, plan, registered)
            artifacts = {report.PRODUCTION_KEY: {"path": plan["productionModel"],
                                                 "sha256": plan["productionModelSha256"]}}
            if binding:
                experiment.committed_bytes(root, "HEAD", PARITY)
                parity = experiment.read_json(root / PARITY)
                validate_parity(parity, registered, inputs, source)
                manifest, artifacts = binding_runs(root, plan, registered, inputs)
                models = load_models(root, artifacts, "mps")
                result = collect(root, inputs["validationCells"], models, "mps")
                categories = {cell["clip"]: cell["category"] for cell in inputs["validationCells"]}
                report.build_report(result, manifest, categories)
            else:
                check_models(root, artifacts)
                result = {"schema": "aethervsr.m8-scoring-parity/1", **apparatus_parity(root, plan, inputs),
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
                          createdAt=experiment.utc_now(), device="mps", dtype="float32",
                          fallback=False, manifest=plan["validationManifest"])
            experiment.write_new(target, result)
            require(result.get("passed", True), "scoring parity failed; evidence retained")
        print(f"Wrote {'binding scores' if binding else 'apparatus parity'}: {target}")
        return 0
    except (experiment.GuardError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"M8 scoring blocked: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())