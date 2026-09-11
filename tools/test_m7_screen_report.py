"""Regression contracts for Milestone 7 screening eligibility and ranking."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).with_name("m7-screen-report.py")
PILOT = Path(__file__).parents[1] / "results" / "pilot-confounded" / "m7-screening-confounded.json"
CATEGORIES = (
    "daylight", "faces", "lowlight", "motion",
    "nature", "text", "texture", "urban",
)
CLIPS = [(f"{category}-clip-{number}", category)
         for category in CATEGORIES for number in range(2)]


def cells(delta):
    return [
        {
            "clip": clip,
            "category": category,
            "crf": crf,
            "psnr": 30.0 + delta,
        }
        for clip, category in CLIPS
        for crf in (18, 26, 34)
    ]


def corrected_screen():
    seed_deltas = {
        "R0": (0.099, 0.100, 0.101),
        "R1": (0.108, 0.109, 0.110),
        "R2": (0.079, 0.080, 0.081),
        "R3": (0.059, 0.060, 0.061),
    }
    models = {"aethersr-c16d2": {"perCell": cells(0.0)}}
    for rung, deltas in seed_deltas.items():
        for seed, delta in enumerate(deltas, 1):
            models[f"{rung}-seed{seed}"] = {"perCell": cells(delta)}
    return {
        "schema": "aethervsr.m6-validation/1",
        "inputStatus": "corrected",
        "crfs": [18, 26, 34],
        "frames": 8,
        "clips": [clip for clip, _ in CLIPS],
        "categories": {category: 2 for category in CATEGORIES},
        "statisticalUnit": "clip; frames averaged within a clip first",
        "models": models,
    }


def invoke(tmp_path, source, *extra):
    output = tmp_path / "report.json"
    result = subprocess.run([
        sys.executable, str(SCRIPT), "--screening", str(source), "--out", str(output), *extra,
    ], capture_output=True, text=True)
    return result, output


def test_corrected_screen_identifies_missing_and_duplicate_cells(tmp_path):
    data = corrected_screen()
    rows = data["models"]["R0-seed1"]["perCell"]
    rows[-1] = dict(rows[0])
    source = tmp_path / "malformed.json"
    source.write_text(json.dumps(data))

    result, output = invoke(tmp_path, source)

    assert result.returncode == 1
    assert "R0-seed1 has duplicate clip×CRF cell" in result.stderr
    assert "R0-seed1 is missing clip×CRF cells" in result.stderr
    assert not output.exists()


def test_archived_pilot_is_audit_only_even_when_labelled_corrected(tmp_path):
    missing_status, missing_output = invoke(tmp_path, PILOT)
    assert missing_status.returncode == 1
    assert "inputStatus metadata is absent; --input-status is required" in missing_status.stderr
    assert not missing_output.exists()

    result, output = invoke(tmp_path, PILOT, "--input-status", "corrected")

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["inputStatus"] == "withdrawn-confounded"
    assert report["inputStatusSource"] == "known-withdrawn-pilot-path"
    assert report["auditOnly"] is True
    assert "winner" not in report
    assert "selected" not in report
    assert "argmax" not in report
    assert "nullResult" not in report


def test_complete_corrected_screen_uses_the_registered_simplicity_tie_rule(tmp_path):
    source = tmp_path / "corrected.json"
    source.write_text(json.dumps(corrected_screen()))

    result, output = invoke(tmp_path, source)

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["ranking"] == ["R1", "R0", "R2", "R3"]
    assert report["argmax"] == "R1"
    assert report["tiedWithArgmax"] == ["R1", "R0"]
    assert report["winner"] == "R0"
    assert report["arms"]["R0"]["sd"] == pytest.approx(0.001)
    assert len(report["arms"]["R1"]["perCategoryCrfOnValidation"]) == 24
    assert report["historicalConfirmationCells"]["valuesDb"] == {
        "motion@34": -0.0907,
        "texture@34": -0.0069,
    }
