#!/usr/bin/env python3
"""Registered M8 paired analysis of supplied scores, without model evaluation.

CLI: python tools/m8_report.py --scores scores.json --runs runs.json --out report.json

Scores use aethervsr.m6-validation/1: models maps scoring keys to objects with
perCell lists of {clip, category, crf, psnr}; clips, crfs and frames are required.
Stored aggregate metrics are ignored. PSNR is already averaged over eight frames.

Run manifest: {"runs": [{rung, seed, optimizerSteps, stepBudget,
validationUpdates, bestUpdate, corpusDigest, registrationCommit,
modelKeys: {fixedFinal, bestValidation}, modelSha256: {fixedFinal, bestValidation},
streamSha256}, ...]}. SHA256 values are lowercase hex. registrationCommit is
the full registered SHA or its exact c955743 abbreviation. Optional validationPsnr
lists 60 patch PSNRs aligned with validationUpdates, enabling bestUpdate checking.
Optional snapshotKeys maps registered update strings to scoring keys; all six
runs must supply the same subset. Unprovided horizons remain not measured.
gap_curve({update: {seed: gap_db}}) describes those horizons without smoothing;
best_update({update: patch_psnr}) checks all eligible draws and exact ties.

build_report(scores, manifest, clip_categories) is pure; tests can supply small
synthetic membership maps. The CLI alone loads and hashes frozen membership.
Neither interface verifies actual model/cache bytes, chronology, initialization,
or training execution. Supplied hashes are declarations, not independent proof.
Synthetic results are apparatus tests, never experimental evidence.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import re
import statistics


SEEDS = (8101, 8102, 8103)
ARMS = ("R0", "R3")
CRFS = (18, 26, 34)
STEP_BUDGET = 81180
VALIDATION_UPDATES = tuple(1353 * draw for draw in range(1, 61))
SNAPSHOT_UPDATES = (5412, 10824, 16200, 29766, 50061, 64944, 81180)
REGISTRATION_COMMIT = "c95574336246ea5114afbec4d2a48e538afd3f7f"
CORPUS_DIGEST = "3f73af7f48ffee2b8efe5e7505dd370944e5614efb9ae4d3c5bff64d3fc563e6"
VALIDATION_SHA256 = "e7a4447fc88c38873826aa616a017aeb2eb8cc6b40257337957f2ea28cb403f1"
PRODUCTION_SHA256 = "d76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a"
PRODUCTION_KEY = "aethersr-c16d2"
CHECKPOINTS = ("fixedFinal", "bestValidation")
TIE_BAND = 0.01
ROOT = Path(__file__).resolve().parents[1]


def finite_number(value: object, label: str) -> float:
    """Reject JSON booleans, strings, missing values and nonfinite numbers."""
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def summarize(differences: dict[int, float]) -> dict:
    """Describe the three equally weighted paired seeds, with ddof=1 SD."""
    if set(differences) != set(SEEDS):
        raise ValueError(f"differences must contain exactly seeds {SEEDS}")
    values = [finite_number(differences[seed], f"seed {seed}") for seed in SEEDS]
    return {
        "perSeed": {str(seed): value for seed, value in zip(SEEDS, values)},
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "sampleSd": statistics.stdev(values),
        "sdDof": 1,
    }


def sign_flip_test(differences: dict[int, float]) -> dict:
    """Enumerate all eight assignments, counting exact absolute-mean ties."""
    summary = summarize(differences)
    values = [Fraction(value) for value in summary["perSeed"].values()]
    observed = abs(sum(values))
    extreme = sum(
        abs(sum(sign * value for sign, value in zip(signs, values))) >= observed
        for signs in product((-1, 1), repeat=3)
    )
    return {
        "test": "exact paired two-sided sign-flip on seed means",
        "pValue": extreme / 8,
        "extremeAssignments": extreme,
        "signAssignments": 8,
        "minimumAttainableP": 0.25,
        "alpha": 0.05,
        "canReachAlpha": False,
        "testIsSignificanceGate": False,
        "assumption": "Sign exchangeability (symmetry under the null), not merely zero mean.",
        "limitation": "Three pairs cannot establish significance at alpha 0.05; minimum p is 0.25.",
    }


def selection(differences: dict[int, float]) -> dict:
    """Apply the fixed-final finite-seed selection rule, not a significance gate."""
    summary = summarize(differences)
    if summary["mean"] > TIE_BAND and summary["min"] > 0:
        label = "PERSISTENT ADVANTAGE"
    elif summary["mean"] > 0:
        label = "SMALL / UNCERTAIN"
    else:
        label = "NO SELECTED ADVANTAGE"
    return {
        "label": label,
        "continuationPermitted": label == "PERSISTENT ADVANTAGE",
        "basis": "fixedFinal only; bestValidation cannot override",
        "signFlip": sign_flip_test(differences),
    }


def best_update(validation_psnr: dict[int, float]) -> int:
    """Choose strictly greater patch PSNR; exact ties retain the earliest draw."""
    if set(validation_psnr) != set(VALIDATION_UPDATES):
        raise ValueError("validation PSNR must contain exactly the 60 eligible updates")
    values = {update: finite_number(validation_psnr[update], "patch PSNR")
              for update in VALIDATION_UPDATES}
    return max(VALIDATION_UPDATES, key=values.__getitem__)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_hash(value: object, label: str) -> None:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"{label} must be a lowercase SHA256")


def validate_runs(manifest: dict) -> dict[tuple[str, int], dict]:
    """Validate declared eligibility and pair identity, without opening models.

    Same-update exports may have different checkpoint metadata and file hashes.
    Tensor equality belongs to the guarded launcher's verify_exports checks
    against saved states, not this report's declared-hash validation.
    """
    require(isinstance(manifest, dict) and isinstance(manifest.get("runs"), list),
            "manifest must contain a runs list")
    indexed = {}
    owners = {}
    for run in manifest["runs"]:
        require(isinstance(run, dict), "each run must be an object")
        rung, seed = run.get("rung"), run.get("seed")
        require(rung in ARMS and type(seed) is int and seed in SEEDS,
                "run must name a registered rung and seed")
        identity = (rung, seed)
        require(identity not in indexed, f"duplicate run {identity}")
        indexed[identity] = run
        for field in ("optimizerSteps", "stepBudget"):
            require(type(run.get(field)) is int and run[field] == STEP_BUDGET,
                    f"{identity}: {field} must be {STEP_BUDGET}")
        updates = run.get("validationUpdates")
        require(isinstance(updates, list) and all(type(update) is int for update in updates)
                and updates == list(VALIDATION_UPDATES),
                f"{identity}: validationUpdates must be the exact 60 registered draws")
        require(type(run.get("bestUpdate")) is int and run["bestUpdate"] in VALIDATION_UPDATES,
                f"{identity}: bestUpdate must be an eligible validation update")
        require(run.get("corpusDigest") == CORPUS_DIGEST,
                f"{identity}: corpusDigest differs from registration")
        require(run.get("registrationCommit") in (REGISTRATION_COMMIT, REGISTRATION_COMMIT[:7]),
                f"{identity}: registrationCommit differs from c955743")
        require_hash(run.get("streamSha256"), f"{identity}: streamSha256")
        for field in ("modelKeys", "modelSha256"):
            require(isinstance(run.get(field), dict) and set(run[field]) == set(CHECKPOINTS),
                    f"{identity}: {field} must contain exactly {CHECKPOINTS}")
        for checkpoint in CHECKPOINTS:
            require_hash(run["modelSha256"][checkpoint], f"{identity}: {checkpoint} hash")
        snapshots = run.get("snapshotKeys", {})
        require(isinstance(snapshots, dict) and all(
            isinstance(update, str) and update in {str(step) for step in SNAPSHOT_UPDATES}
            for update in snapshots), f"{identity}: snapshotKeys contains an unregistered update")
        for key in list(run["modelKeys"].values()) + list(snapshots.values()):
            require(isinstance(key, str) and bool(key) and key != PRODUCTION_KEY,
                    f"{identity}: invalid or reserved scoring key")
            require(key not in owners or owners[key] == identity,
                    f"scoring key {key!r} reused across runs")
            owners[key] = identity
        if run["modelKeys"]["fixedFinal"] == run["modelKeys"]["bestValidation"]:
            require(run["modelSha256"]["fixedFinal"] == run["modelSha256"]["bestValidation"],
                    f"{identity}: aliased scoring key has different hashes")
        if "validationPsnr" in run:
            curve = run["validationPsnr"]
            require(isinstance(curve, list) and len(curve) == len(VALIDATION_UPDATES),
                    f"{identity}: validationPsnr must have 60 values")
            require(best_update(dict(zip(VALIDATION_UPDATES, curve))) == run["bestUpdate"],
                    f"{identity}: bestUpdate violates strict-greater/earliest-tie rule")
    require(set(indexed) == set(product(ARMS, SEEDS)), "runs must contain exactly all six arm/seed keys")
    snapshot_sets = {tuple(sorted(run.get("snapshotKeys", {}))) for run in indexed.values()}
    require(len(snapshot_sets) == 1, "all six runs must have identical snapshot updates")
    for seed in SEEDS:
        require(indexed[("R0", seed)]["streamSha256"] == indexed[("R3", seed)]["streamSha256"],
                f"seed {seed}: paired streamSha256 mismatch")
    return indexed


def validate_scores(scores: dict, clip_categories: dict[str, str]) -> dict:
    """Index every model's complete clip x CRF grid; reject all silent dropping."""
    require(isinstance(clip_categories, dict) and bool(clip_categories) and all(
        isinstance(clip, str) and clip and isinstance(category, str) and category
        for clip, category in clip_categories.items()), "clip_categories must be a nonempty string map")
    require(isinstance(scores, dict) and scores.get("schema") == "aethervsr.m6-validation/1",
            "scores must use aethervsr.m6-validation/1")
    require(type(scores.get("frames")) is int and scores["frames"] == 8, "scores must use eight frames")
    crfs = scores.get("crfs")
    require(isinstance(crfs, list) and all(type(crf) is int for crf in crfs)
            and sorted(crfs) == list(CRFS), "scores crfs must be exactly 18, 26, 34")
    clips = scores.get("clips")
    require(isinstance(clips, list) and all(isinstance(clip, str) for clip in clips)
            and len(clips) == len(clip_categories) and set(clips) == set(clip_categories),
            "scores clips must match frozen membership exactly, without duplicates")
    models = scores.get("models")
    require(isinstance(models, dict) and bool(models), "scores models must be a nonempty map")
    expected = set(product(clip_categories, CRFS))
    indexed = {}
    for key, model in models.items():
        require(isinstance(key, str) and bool(key), "model scoring keys must be nonempty strings")
        require(isinstance(model, dict) and isinstance(model.get("perCell"), list),
                f"{key}: perCell must be a list")
        cells = {}
        for cell in model["perCell"]:
            require(isinstance(cell, dict), f"{key}: cell must be an object")
            clip, crf = cell.get("clip"), cell.get("crf")
            require(isinstance(clip, str) and clip in clip_categories and type(crf) is int and crf in CRFS,
                    f"{key}: unexpected clip or CRF")
            require(cell.get("category") == clip_categories[clip], f"{key}: frozen category mismatch for {clip}")
            require((clip, crf) not in cells, f"{key}: duplicate cell {clip}@{crf}")
            cells[(clip, crf)] = finite_number(cell.get("psnr"), f"{key}: {clip}@{crf} psnr")
        require(set(cells) == expected, f"{key}: missing cells {sorted(expected - set(cells))}")
        indexed[key] = cells
    return indexed


def comparison(left: dict, right: dict, clip_categories: dict[str, str]) -> dict:
    """Describe left-minus-right PSNR using equal tiers, then equal clips."""
    differences = {
        seed: {cell: left[seed][cell] - right[seed][cell] for cell in left[seed]}
        for seed in SEEDS
    }

    def aggregate(clips, crfs):
        return summarize({seed: statistics.mean([
            statistics.mean(differences[seed][(clip, crf)] for crf in crfs)
            for clip in clips]) for seed in SEEDS})

    return {
        "overall": aggregate(sorted(clip_categories), CRFS),
        "byCrf": {str(crf): aggregate(sorted(clip_categories), (crf,)) for crf in CRFS},
        "byCategoryCrf": {
            category: {str(crf): aggregate([
                clip for clip in sorted(clip_categories) if clip_categories[clip] == category], (crf,))
                for crf in CRFS}
            for category in sorted(set(clip_categories.values()))
        },
        "perClip": [{"clip": clip, "category": clip_categories[clip], **aggregate([clip], CRFS)}
                    for clip in sorted(clip_categories)],
        "perCell": [{"clip": clip, "category": clip_categories[clip], "crf": crf,
                     **aggregate([clip], (crf,))}
                    for clip in sorted(clip_categories) for crf in CRFS],
    }


def gap_curve(gaps: dict[int, dict[int, float]]) -> dict:
    """Retain observed crossings and qualify the partial-long versus final gap."""
    require(isinstance(gaps, dict) and all(type(update) is int and update in SNAPSHOT_UPDATES
                                         for update in gaps), "gap curve has unregistered updates")
    points = [{"update": update, **summarize(gaps[update])} for update in sorted(gaps)]

    def sign(value):
        return "positive" if value > 0 else "negative" if value < 0 else "zero"

    def crossings(values):
        previous = None
        zeros = []
        changes = []
        for update, value in values:
            if value == 0:
                zeros.append(update)
                continue
            if previous is not None and sign(previous[1]) != sign(value):
                changes.append({"fromUpdate": previous[0], "toUpdate": update,
                                "fromGapDb": previous[1], "toGapDb": value,
                                "zeroUpdatesBetween": list(zeros)})
            previous = (update, value)
            zeros = []
        return changes

    def endpoint_change(short, final):
        change = final - short
        return {"shortGapDb": short, "finalGapDb": final, "finalMinusShortDb": change,
                "change": "similar" if abs(change) <= TIE_BAND else "increased" if change > 0 else "decreased",
                "shortSign": sign(short), "finalSign": sign(final), "signChanged": sign(short) != sign(final)}

    short_long = {"status": "not measured", "requiredUpdates": [16200, STEP_BUDGET]}
    indexed = {point["update"]: point for point in points}
    if 16200 in indexed and STEP_BUDGET in indexed:
        short, final = indexed[16200], indexed[STEP_BUDGET]
        short_long = {
            "status": "supplied scores", "shortUpdate": 16200, "finalUpdate": STEP_BUDGET,
            "aggregate": endpoint_change(short["mean"], final["mean"]),
            "perSeed": {str(seed): endpoint_change(short["perSeed"][str(seed)], final["perSeed"][str(seed)])
                        for seed in SEEDS},
        }
    return {
        "status": "not measured" if not points else "complete" if len(points) == 7 else "partial",
        "points": points,
        "missingUpdates": [update for update in SNAPSHOT_UPDATES if update not in gaps],
        "crossings": {
            "aggregate": crossings([(point["update"], point["mean"]) for point in points]),
            "perSeed": {str(seed): crossings([(point["update"], point["perSeed"][str(seed)]) for point in points])
                        for seed in SEEDS},
        },
        "shortVsLong": short_long,
        "qualification": "All M8 snapshots share the 81180-update cosine schedule. Update 16200 is a partial "
                         "long schedule, not M7's fully annealed 16200-update schedule. Final-minus-short "
                         "within 0.01 dB is similar. Crossings bracket observations, not interpolated times. "
                         "These points establish neither convergence nor monotonicity or infinite-horizon behavior.",
        "historicalM7": {
            "registeredReportedGapDb": 0.0247,
            "source": "docs/M8-PREREGISTRATION.md#horizon-interpretation",
            "qualification": "Historical corrected short-budget result, not recomputed here and not an M8 "
                             "measurement. M7 used different seeds and uncompensated R3 identity initialization; "
                             "keep separate from this matched-initialization curve.",
        },
    }


def build_report(scores: dict, manifest: dict, clip_categories: dict[str, str]) -> dict:
    """Pure report builder; explicit membership overrides are for apparatus tests."""
    runs = validate_runs(manifest)
    models = validate_scores(scores, clip_categories)
    for run in runs.values():
        keys = run["modelKeys"]
        for key in list(keys.values()) + list(run.get("snapshotKeys", {}).values()):
            require(key in models, f"missing model scoring key {key!r}")
        if run["modelSha256"]["fixedFinal"] == run["modelSha256"]["bestValidation"]:
            require(models[keys["fixedFinal"]] == models[keys["bestValidation"]],
                    "identical checkpoint hashes require identical scores")
        final_snapshot = run.get("snapshotKeys", {}).get(str(STEP_BUDGET))
        if final_snapshot is not None:
            require(models[final_snapshot] == models[keys["fixedFinal"]],
                    "final snapshot scores differ from fixedFinal")

    def arm_scores(rung, checkpoint):
        return {seed: models[runs[(rung, seed)]["modelKeys"][checkpoint]] for seed in SEEDS}

    contrasts = {checkpoint: comparison(arm_scores("R3", checkpoint), arm_scores("R0", checkpoint),
                                        clip_categories) for checkpoint in CHECKPOINTS}
    primary = {int(seed): value for seed, value in contrasts["fixedFinal"]["overall"]["perSeed"].items()}
    snapshots = {}
    for update in sorted(runs[("R0", SEEDS[0])].get("snapshotKeys", {}), key=int):
        snapshots[update] = comparison(
            {seed: models[runs[("R3", seed)]["snapshotKeys"][update]] for seed in SEEDS},
            {seed: models[runs[("R0", seed)]["snapshotKeys"][update]] for seed in SEEDS}, clip_categories)
    horizon = gap_curve({int(update): {int(seed): value for seed, value in result["overall"]["perSeed"].items()}
                         for update, result in snapshots.items()})
    horizon["comparisonsByUpdate"] = snapshots
    production = {"status": "not measured", "requiredKey": PRODUCTION_KEY}
    if PRODUCTION_KEY in models:
        baseline = {seed: models[PRODUCTION_KEY] for seed in SEEDS}
        production = {
            "status": "supplied scores", "key": PRODUCTION_KEY, "frozenSha256": PRODUCTION_SHA256,
            "qualification": "Recovered cache is not claimed byte-identical to the lost M6 cache. "
                             "Baseline key is declared identity; model bytes were not opened.",
            "comparisons": {checkpoint: {
                rung: comparison(arm_scores(rung, checkpoint), baseline, clip_categories) for rung in ARMS
            } for checkpoint in CHECKPOINTS},
        }
    return {
        "schema": "aethervsr.m8-report/1",
        "scope": "Supplied-score arithmetic only; synthetic scores are not experimental evidence.",
        "registrationCommit": REGISTRATION_COMMIT,
        "corpusDigest": CORPUS_DIGEST,
        "metric": "RGB PSNR dB; eight frames per cell, equal tiers per clip, equal clips per seed",
        "clipCategories": dict(sorted(clip_categories.items())),
        "crfs": list(CRFS),
        "runs": [dict(runs[(rung, seed)]) for seed in SEEDS for rung in ARMS],
        "scores": {key: {"perCell": [
            {"clip": clip, "category": clip_categories[clip], "crf": crf, "psnr": cells[(clip, crf)]}
            for clip in sorted(clip_categories) for crf in CRFS]} for key, cells in sorted(models.items())},
        "fixedFinal": contrasts["fixedFinal"],
        "bestValidation": {**contrasts["bestValidation"], "role": "secondary; cannot override fixedFinal"},
        "decision": selection(primary),
        "horizon": horizon,
        "vsProduction": production,
        "checkpointSelectionVerifiedFromSuppliedCurve": all("validationPsnr" in run for run in runs.values()),
        "limitations": [
            "Hashes, training execution, initialization, chronology and cache/frame identity are not independently verified.",
            "Without validationPsnr, bestUpdate eligibility is checked but its optimality and tie handling are not measured.",
            "Category and horizon comparisons are descriptive, not additional rejection opportunities.",
            "Continuation is conditional on independent apparatus and provenance gates; it is not production replacement.",
        ],
    }


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    """Reject duplicate JSON fields before dictionary construction can hide them."""
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def load_validation_categories(root: Path = ROOT) -> dict[str, str]:
    """Load metadata only and fail closed if the registered manifest bytes changed."""
    plan = read_json(root / "results/m8-plan.json")
    expected = {
        "arms": list(ARMS), "pairedSeeds": list(SEEDS), "trainerCorpusDigest": CORPUS_DIGEST,
        "validationManifest": "data/captured-val-m7/manifest.json",
        "validationManifestSha256": VALIDATION_SHA256,
        "productionModel": "public/models/aethersr-c16d2.json", "productionModelSha256": PRODUCTION_SHA256,
    }
    require(all(plan.get(key) == value for key, value in expected.items()), "plan differs from registration")
    training = plan.get("training", {})
    require(training.get("optimizerUpdates") == STEP_BUDGET
            and training.get("validationEveryUpdates") == 1353 and training.get("validationDraws") == 60
            and training.get("snapshotUpdates") == list(SNAPSHOT_UPDATES), "plan training schedule differs")
    scoring = plan.get("scoring", {})
    require(scoring.get("crfs") == list(CRFS) and scoring.get("clips") == 16
            and scoring.get("cells") == 48 and scoring.get("frames") == 8, "plan scoring layout differs")
    raw_manifest = (root / plan["validationManifest"]).read_bytes()
    require(hashlib.sha256(raw_manifest).hexdigest() == VALIDATION_SHA256, "frozen validation manifest hash mismatch")
    manifest = json.loads(raw_manifest, object_pairs_hook=unique_object)
    clips = manifest["clips"]
    categories = {clip["id"]: clip["category"] for clip in clips}
    require(len(clips) == len(categories) == 16 and len(set(categories.values())) == 8,
            "frozen validation membership must have 16 unique IDs and eight categories")
    return categories


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_report(read_json(args.scores), read_json(args.runs), load_validation_categories())
        report["validationManifestSha256"] = VALIDATION_SHA256
        report["inputs"] = {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                            for name, path in (("scores", args.scores), ("runs", args.runs))}
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        with args.out.open("x", encoding="utf-8") as output:
            output.write(payload)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())