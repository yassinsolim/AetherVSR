"""Synthetic apparatus tests only; these scores are not experimental evidence."""

import copy
import json
import math

import pytest

import m8_report
from m8_report import (
    CORPUS_DIGEST, CRFS, PRODUCTION_KEY, REGISTRATION_COMMIT, SEEDS, STEP_BUDGET,
    SNAPSHOT_UPDATES, VALIDATION_UPDATES, best_update, build_report, gap_curve, selection, sign_flip_test, summarize,
)


MEMBERSHIP = {"synthetic-a": "faces", "synthetic-b": "texture", "synthetic-c": "texture"}


def dataset(membership=None, fixed=(0.125, 0.25, 0.375), best=(0.5, 0.5, 0.5)):
    membership = MEMBERSHIP if membership is None else membership
    scores = {"schema": "aethervsr.m6-validation/1", "clips": list(membership),
              "frames": 8, "crfs": list(CRFS), "models": {}}
    runs = []
    for position, seed in enumerate(SEEDS):
        for rung in ("R0", "R3"):
            keys = {checkpoint: f"{rung}-{seed}-{checkpoint}" for checkpoint in ("fixedFinal", "bestValidation")}
            runs.append({
                "rung": rung, "seed": seed, "optimizerSteps": STEP_BUDGET, "stepBudget": STEP_BUDGET,
                "validationUpdates": list(VALIDATION_UPDATES), "bestUpdate": 1353,
                "corpusDigest": CORPUS_DIGEST, "registrationCommit": REGISTRATION_COMMIT,
                "modelKeys": keys, "modelSha256": {"fixedFinal": "a" * 64, "bestValidation": "b" * 64},
                "streamSha256": f"{seed:064x}",
            })
            for checkpoint, gap in (("fixedFinal", fixed[position]), ("bestValidation", best[position])):
                scores["models"][keys[checkpoint]] = {"meanDelta": -999, "perCell": [
                    {"clip": clip, "category": category, "crf": crf, "psnr": 30 + (gap if rung == "R3" else 0)}
                    for clip, category in membership.items() for crf in CRFS
                ]}
    return scores, {"runs": runs}


def report(scores, manifest):
    return build_report(scores, manifest, MEMBERSHIP)


def paired(values):
    return dict(zip(SEEDS, values))


def test_seed_statistics():
    summary = summarize(paired([1, 2, 3]))
    assert summary["mean"] == summary["median"] == 2
    assert summary["min"] == 1
    assert summary["max"] == 3
    assert summary["range"] == 2
    assert summary["sampleSd"] == 1
    assert summary["sdDof"] == 1


@pytest.mark.parametrize("values,expected", [
    ([1, 2, 3], 0.25), ([-1, -2, -3], 0.25),
    ([0, 0, 0], 1), ([1, -1, 0], 1), ([1, 1, 0], 0.5),
    ([1, 1, -1], 1), ([0.1, 0.2, 0.3], 0.25),
])
def test_exact_sign_flip_and_alpha_resolution(values, expected):
    result = sign_flip_test(paired(values))
    assert result["pValue"] == expected
    assert result["signAssignments"] == 8
    assert result["minimumAttainableP"] == 0.25
    assert result["alpha"] == 0.05
    assert result["canReachAlpha"] is False
    assert result["testIsSignificanceGate"] is False


@pytest.mark.parametrize("values,label,allowed", [
    ([0.02] * 3, "PERSISTENT ADVANTAGE", True),
    ([0.01] * 3, "SMALL / UNCERTAIN", False),
    ([0.001] * 3, "SMALL / UNCERTAIN", False),
    ([0.2, 0.2, -0.01], "SMALL / UNCERTAIN", False),
    ([0.2, 0.2, 0], "SMALL / UNCERTAIN", False),
    ([0] * 3, "NO SELECTED ADVANTAGE", False),
    ([-0.1] * 3, "NO SELECTED ADVANTAGE", False),
])
def test_selection_ties_and_inconsistent_signs(values, label, allowed):
    decision = selection(paired(values))
    assert decision["label"] == label
    assert decision["continuationPermitted"] is allowed


def test_best_exact_ties_keep_earliest_update():
    curve = {update: 30.0 for update in reversed(VALIDATION_UPDATES)}
    assert best_update(curve) == 1353
    curve[2706] = math.nextafter(30.0, math.inf)
    assert best_update(curve) == 2706
    curve[16200] = 100
    with pytest.raises(ValueError, match="60 eligible"):
        best_update(curve)


@pytest.mark.parametrize("invalid", [None, True, "30", math.nan, math.inf, -math.inf])
def test_invalid_seed_scores(invalid):
    with pytest.raises(ValueError, match="finite"):
        summarize(paired([0, 0, invalid]))


def test_missing_seed_summary():
    with pytest.raises(ValueError, match="exactly seeds"):
        summarize({8101: 0, 8102: 0})


def test_complete_paired_report_ignores_stored_aggregates_and_order():
    scores, manifest = dataset()
    original = copy.deepcopy((scores, manifest))
    result = report(scores, manifest)
    assert result["fixedFinal"]["overall"]["mean"] == 0.25
    assert result["fixedFinal"]["overall"]["sampleSd"] == 0.125
    assert result["decision"]["continuationPermitted"] is True
    assert len(result["fixedFinal"]["perCell"]) == 9
    assert len(result["fixedFinal"]["perClip"]) == 3
    assert len(result["fixedFinal"]["byCategoryCrf"]) == 2
    assert result["vsProduction"]["status"] == "not measured"
    assert (scores, manifest) == original
    manifest["runs"].reverse()
    for model in scores["models"].values():
        model["perCell"].reverse()
    assert report(scores, manifest) == result


def test_equal_clip_weight_not_equal_category_weight():
    scores, manifest = dataset(fixed=(0, 0, 0))
    for run in manifest["runs"]:
        if run["rung"] == "R3":
            for cell in scores["models"][run["modelKeys"]["fixedFinal"]]["perCell"]:
                cell["psnr"] += {"synthetic-a": 3, "synthetic-b": 0, "synthetic-c": -1}[cell["clip"]]
                cell["psnr"] += {18: 0, 26: 1, 34: 2}[cell["crf"]]
    primary = report(scores, manifest)["fixedFinal"]
    assert primary["overall"]["mean"] == pytest.approx(5 / 3)
    assert primary["byCrf"]["18"]["mean"] == pytest.approx(2 / 3)
    assert primary["byCategoryCrf"]["texture"]["34"]["mean"] == 1.5
    assert primary["perClip"][0]["perSeed"]["8101"] == 4


@pytest.mark.parametrize("fixed", [(0, 0, 0), (-0.125, -0.125, -0.125), (0.125, 0.125, -0.125)])
def test_best_validation_cannot_override(fixed):
    result = report(*dataset(fixed=fixed, best=(1, 1, 1)))
    assert result["bestValidation"]["overall"]["mean"] == 1
    assert result["decision"]["continuationPermitted"] is False


@pytest.mark.parametrize("mutation,match", [
    (lambda runs: runs.pop(), "six arm/seed"),
    (lambda runs: runs.append(copy.deepcopy(runs[0])), "duplicate run"),
    (lambda runs: runs.__setitem__(1, copy.deepcopy(runs[0])), "duplicate run"),
    (lambda runs: runs[0].update(seed=8099), "registered rung and seed"),
    (lambda runs: runs[0].update(rung="R1"), "registered rung and seed"),
    (lambda runs: runs[0].update(optimizerSteps=81179), "optimizerSteps"),
    (lambda runs: runs[0].update(stepBudget=16200), "stepBudget"),
    (lambda runs: runs[0].update(optimizerSteps=81180.0), "optimizerSteps"),
    (lambda runs: runs[0]["validationUpdates"].pop(), "60 registered draws"),
    (lambda runs: runs[0]["validationUpdates"].append(16200), "60 registered draws"),
    (lambda runs: runs[0]["validationUpdates"].__setitem__(0, 2706), "60 registered draws"),
    (lambda runs: runs[0].update(bestUpdate=16200), "eligible validation"),
    (lambda runs: runs[0].update(corpusDigest="c" * 64), "corpusDigest"),
    (lambda runs: runs[0].update(registrationCommit="d" * 40), "registrationCommit"),
    (lambda runs: runs[0].update(streamSha256="e" * 64), "paired stream"),
    (lambda runs: runs[0].pop("streamSha256"), "SHA256"),
    (lambda runs: runs[0]["modelSha256"].update(fixedFinal="not a hash"), "SHA256"),
    (lambda runs: runs[0]["modelSha256"].update(bestValidation="not a hash"), "SHA256"),
    (lambda runs: runs[0]["modelKeys"].pop("bestValidation"), "modelKeys"),
    (lambda runs: runs[0]["modelKeys"].update(fixedFinal=runs[1]["modelKeys"]["fixedFinal"]), "reused across runs"),
    (lambda runs: runs[0]["modelKeys"].update(fixedFinal="absent"), "missing model"),
])
def test_invalid_runs(mutation, match):
    scores, manifest = dataset()
    mutation(manifest["runs"])
    with pytest.raises(ValueError, match=match):
        report(scores, manifest)


def test_short_registration_sha_and_verified_tie_curve():
    scores, manifest = dataset()
    for run in manifest["runs"]:
        run["registrationCommit"] = "c955743"
        run["validationPsnr"] = [30.0] * 60
    assert report(scores, manifest)["checkpointSelectionVerifiedFromSuppliedCurve"] is True
    manifest["runs"][0]["bestUpdate"] = 2706
    with pytest.raises(ValueError, match="earliest-tie"):
        report(scores, manifest)


@pytest.mark.parametrize("mutation,match", [
    (lambda cells: cells.pop(), "missing cells"),
    (lambda cells: cells.append(copy.deepcopy(cells[0])), "duplicate cell"),
    (lambda cells: cells.__setitem__(1, copy.deepcopy(cells[0])), "duplicate cell"),
    (lambda cells: cells[0].update(clip="unregistered"), "unexpected clip"),
    (lambda cells: cells[0].update(category="wrong"), "category mismatch"),
    (lambda cells: cells[0].update(crf=22), "unexpected clip or CRF"),
    (lambda cells: cells[0].update(crf=18.0), "unexpected clip or CRF"),
    (lambda cells: cells[0].update(psnr=math.nan), "finite"),
    (lambda cells: cells[0].update(psnr=math.inf), "finite"),
    (lambda cells: cells[0].update(psnr=True), "finite"),
    (lambda cells: cells[0].pop("psnr"), "finite"),
])
def test_invalid_cells(mutation, match):
    scores, manifest = dataset()
    mutation(next(iter(scores["models"].values()))["perCell"])
    with pytest.raises(ValueError, match=match):
        report(scores, manifest)


@pytest.mark.parametrize("field,value", [
    ("schema", "other"), ("frames", 7), ("frames", True),
    ("crfs", [18, 26, 26]), ("clips", ["synthetic-a"] * 3),
])
def test_invalid_scoring_metadata(field, value):
    scores, manifest = dataset()
    scores[field] = value
    with pytest.raises(ValueError):
        report(scores, manifest)


def test_exact_production_key_only_and_all_models_validated():
    scores, manifest = dataset()
    baseline = copy.deepcopy(next(iter(scores["models"].values())))
    scores["models"]["production-alias"] = baseline
    assert report(scores, manifest)["vsProduction"]["status"] == "not measured"
    scores["models"][PRODUCTION_KEY] = baseline
    relative = report(scores, manifest)["vsProduction"]["comparisons"]
    assert relative["fixedFinal"]["R0"]["overall"]["mean"] == 0
    assert relative["fixedFinal"]["R3"]["overall"]["mean"] == 0.25
    baseline["perCell"].pop()
    with pytest.raises(ValueError, match="missing cells"):
        report(scores, manifest)


def test_json_duplicate_fields_are_rejected(tmp_path):
    source = tmp_path / "duplicates.json"
    source.write_text('{"runs": [], "runs": []}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        m8_report.read_json(source)


def test_cli_with_synthetic_scores_and_real_frozen_membership(tmp_path):
    membership = m8_report.load_validation_categories()
    assert len(membership) == 16
    scores, manifest = dataset(membership)
    source, runs, output = (tmp_path / name for name in ("scores.json", "runs.json", "report.json"))
    source.write_text(json.dumps(scores))
    runs.write_text(json.dumps(manifest))
    arguments = ["--scores", str(source), "--runs", str(runs), "--out", str(output)]
    assert m8_report.main(arguments) == 0
    result = json.loads(output.read_text())
    assert len(result["fixedFinal"]["perCell"]) == 48
    assert sum(len(tiers) for tiers in result["fixedFinal"]["byCategoryCrf"].values()) == 24
    assert "not experimental evidence" in result["scope"]
    with pytest.raises(SystemExit):
        m8_report.main(arguments)
    assert json.loads(output.read_text()) == result


def add_snapshots(scores, manifest, gaps):
    for run in manifest["runs"]:
        snapshots = {}
        for update, gap in gaps.items():
            key = f"{run['rung']}-{run['seed']}-snapshot-{update}"
            snapshots[str(update)] = key
            model = copy.deepcopy(scores["models"][run["modelKeys"]["fixedFinal"]])
            for cell in model["perCell"]:
                cell["psnr"] = 30 + (gap if run["rung"] == "R3" else 0)
            scores["models"][key] = model
        run["snapshotKeys"] = snapshots


def test_snapshot_curves_retain_crossings_without_steering_primary():
    scores, manifest = dataset(fixed=(-0.125,) * 3)
    gaps = dict(zip(SNAPSHOT_UPDATES, (0.25, -0.25, 0.125, 0, -0.25, 0.25, -0.125)))
    add_snapshots(scores, manifest, gaps)
    result = report(scores, manifest)
    horizon = result["horizon"]
    assert horizon["status"] == "complete"
    assert horizon["missingUpdates"] == []
    assert [point["mean"] for point in horizon["points"]] == list(gaps.values())
    assert len(horizon["crossings"]["aggregate"]) == 5
    assert horizon["crossings"]["aggregate"][2]["zeroUpdatesBetween"] == [29766]
    assert len(horizon["crossings"]["perSeed"]["8101"]) == 5
    assert horizon["shortVsLong"]["aggregate"]["change"] == "decreased"
    assert horizon["shortVsLong"]["aggregate"]["signChanged"] is True
    assert horizon["shortVsLong"]["aggregate"]["finalMinusShortDb"] == -0.25
    assert len(horizon["comparisonsByUpdate"]["16200"]["perCell"]) == 9
    assert "partial long schedule" in horizon["qualification"]
    assert "different seeds" in horizon["historicalM7"]["qualification"]
    assert result["decision"]["continuationPermitted"] is False


@pytest.mark.parametrize("short,final,expected", [
    (0, 0.01, "similar"), (0.01, 0, "similar"), (0, 0, "similar"),
    (0, math.nextafter(0.01, math.inf), "increased"),
    (0.125, -0.125, "decreased"), (-0.125, 0.125, "increased"),
])
def test_horizon_exact_similarity_band(short, final, expected):
    horizon = gap_curve({16200: paired([short] * 3), STEP_BUDGET: paired([final] * 3)})
    assert horizon["shortVsLong"]["aggregate"]["change"] == expected
    assert horizon["shortVsLong"]["perSeed"]["8101"]["change"] == expected
    assert horizon["status"] == "partial"


def test_per_seed_crossing_not_hidden_by_positive_mean():
    curve = gap_curve({16200: paired([-0.125, 1, 1]), STEP_BUDGET: paired([0.125, 1, 1])})
    assert curve["crossings"]["aggregate"] == []
    assert len(curve["crossings"]["perSeed"]["8101"]) == 1
    assert curve["shortVsLong"]["perSeed"]["8101"]["signChanged"] is True


def test_missing_horizons_never_become_zero():
    curve = gap_curve({5412: paired([0, 0, 0])})
    assert curve["shortVsLong"]["status"] == "not measured"
    assert gap_curve({})["status"] == "not measured"
    assert report(*dataset())["horizon"]["points"] == []


@pytest.mark.parametrize("mutation,match", [
    (lambda runs: runs[0].pop("snapshotKeys"), "identical snapshot updates"),
    (lambda runs: runs[0]["snapshotKeys"].update({"16201": "wrong"}), "unregistered update"),
    (lambda runs: runs[0]["snapshotKeys"].update({"16200": "absent"}), "missing model"),
    (lambda runs: runs[0]["snapshotKeys"].update({"16200": runs[1]["snapshotKeys"]["16200"]}), "reused across runs"),
])
def test_invalid_snapshot_metadata(mutation, match):
    scores, manifest = dataset()
    add_snapshots(scores, manifest, {16200: 0.125})
    mutation(manifest["runs"])
    with pytest.raises(ValueError, match=match):
        report(scores, manifest)


def test_final_snapshot_must_match_primary_scores():
    scores, manifest = dataset()
    add_snapshots(scores, manifest, {STEP_BUDGET: 99})
    with pytest.raises(ValueError, match="final snapshot scores differ"):
        report(scores, manifest)


def test_same_export_hash_requires_same_scores():
    scores, manifest = dataset(fixed=(0.125,) * 3, best=(0.5,) * 3)
    manifest["runs"][1]["modelSha256"]["bestValidation"] = "a" * 64
    with pytest.raises(ValueError, match="identical checkpoint hashes"):
        report(scores, manifest)


def test_final_best_update_accepts_metadata_distinct_export_hashes():
    scores, manifest = dataset(fixed=(0.125,) * 3, best=(0.125,) * 3)
    for run in manifest["runs"]:
        run["bestUpdate"] = STEP_BUDGET
        run["validationPsnr"] = [30.0] * 59 + [31.0]
        assert run["modelKeys"]["fixedFinal"] != run["modelKeys"]["bestValidation"]
        assert run["modelSha256"]["fixedFinal"] != run["modelSha256"]["bestValidation"]
    result = report(scores, manifest)
    assert result["checkpointSelectionVerifiedFromSuppliedCurve"] is True
    assert result["decision"] == selection(paired([0.125] * 3))


@pytest.mark.parametrize("update", [VALIDATION_UPDATES[0], STEP_BUDGET])
def test_aliased_scoring_key_requires_same_hash(update):
    scores, manifest = dataset()
    run = manifest["runs"][0]
    run["bestUpdate"] = update
    run["modelKeys"]["bestValidation"] = run["modelKeys"]["fixedFinal"]
    with pytest.raises(ValueError, match="aliased scoring key has different hashes"):
        report(scores, manifest)


def test_snapshot_draw_is_not_best_eligible_and_missing_arm_or_seed_rejected():
    for retained in (
        lambda run: run["rung"] == "R0",
        lambda run: run["seed"] != 8101,
    ):
        scores, manifest = dataset()
        manifest["runs"] = [run for run in manifest["runs"] if retained(run)]
        with pytest.raises(ValueError, match="all six"):
            report(scores, manifest)


def test_cli_invalid_inputs_do_not_create_report(tmp_path, monkeypatch):
    scores, manifest = dataset()
    manifest["runs"][0]["validationUpdates"].append(16200)
    source, runs, output = (tmp_path / name for name in ("scores.json", "runs.json", "report.json"))
    source.write_text(json.dumps(scores))
    runs.write_text(json.dumps(manifest))
    monkeypatch.setattr(m8_report, "load_validation_categories", lambda: MEMBERSHIP)
    with pytest.raises(SystemExit) as raised:
        m8_report.main(["--scores", str(source), "--runs", str(runs), "--out", str(output)])
    assert raised.value.code == 2
    assert not output.exists()