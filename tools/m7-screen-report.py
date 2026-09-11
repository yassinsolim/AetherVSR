#!/usr/bin/env python3
"""Apply the pre-registered Milestone 7 selection rule to a screening run.

The report is intentionally narrow: it computes the clip-first, production-
relative ranking registered in ``docs/M7-PREREGISTRATION.md`` and rejects
incomplete corrected screens.  The withdrawn pilot can still be audited, but
can never acquire a winner from this tool.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
from pathlib import Path
import statistics as st
import sys

TIE_BAND_DB = 0.01
SIMPLICITY_ORDER = ["R0", "R1", "R2", "R3", "R4"]
EXPECTED_RUNGS = tuple(SIMPLICITY_ORDER[:4])
EXPECTED_SEEDS = frozenset({"seed1", "seed2", "seed3"})
EXPECTED_CRFS = (18, 26, 34)
EXPECTED_CLIPS = 16
EXPECTED_CATEGORIES = 8
EXPECTED_FRAMES = 8
EXPECTED_CELLS = EXPECTED_CLIPS * len(EXPECTED_CRFS)
INPUT_STATUS_KEY = "inputStatus"
CORRECTED_STATUS = "corrected"
WITHDRAWN_STATUS = "withdrawn-confounded"
INPUT_STATUSES = (CORRECTED_STATUS, WITHDRAWN_STATUS)
STATISTICAL_UNIT = "clip; frames averaged within a clip first"
HISTORICAL_CONFIRMATION_CELLS = {
    "motion@34": -0.0907,
    "texture@34": -0.0069,
}
KNOWN_WITHDRAWN_PILOT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "pilot-confounded"
)


class ScreeningInputError(ValueError):
    """The input cannot support a trustworthy arithmetic report."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


def cell_name(key: tuple[str, int]) -> str:
    return f"{key[0]}@{key[1]}"


def cell_names(cells: set[tuple[str, int]]) -> str:
    return ", ".join(cell_name(key) for key in sorted(cells))


def is_known_withdrawn_pilot(path: str) -> bool:
    """Whether this is the archived, unselectable confounded pilot."""
    try:
        Path(path).resolve().relative_to(KNOWN_WITHDRAWN_PILOT_DIR)
    except ValueError:
        return False
    return True


def resolve_input_status(data: dict, screening: str,
                         cli_status: str | None) -> tuple[str, str]:
    """Resolve declared eligibility without allowing the archive to be relabelled."""
    if INPUT_STATUS_KEY in data:
        status = data[INPUT_STATUS_KEY]
        if status not in INPUT_STATUSES:
            raise ScreeningInputError([
                f"{INPUT_STATUS_KEY} must be one of {list(INPUT_STATUSES)}, got {status!r}"
            ])
        if cli_status is not None and cli_status != status:
            raise ScreeningInputError([
                f"--input-status {cli_status!r} conflicts with metadata "
                f"{INPUT_STATUS_KEY} {status!r}"
            ])
        source = "metadata"
    else:
        if cli_status is None:
            raise ScreeningInputError([
                f"{INPUT_STATUS_KEY} metadata is absent; --input-status is required"
            ])
        status = cli_status
        source = "cli"

    if is_known_withdrawn_pilot(screening):
        return WITHDRAWN_STATUS, "known-withdrawn-pilot-path"
    return status, source


def validate_layout(data: dict, errors: list[str]) -> tuple[set[tuple[str, int]] | None,
                                                              set[str] | None]:
    """Validate the declared 16-clip, 8-category, three-CRF screen layout."""
    raw_crfs = data.get("crfs")
    if not isinstance(raw_crfs, list):
        errors.append("crfs must be a list containing 18, 26, and 34")
    else:
        seen_crfs: set[int] = set()
        for crf in raw_crfs:
            if not isinstance(crf, int) or isinstance(crf, bool):
                errors.append(f"crfs contains non-integer value {crf!r}")
            elif crf in seen_crfs:
                errors.append(f"crfs contains duplicate CRF {crf}")
            else:
                seen_crfs.add(crf)
        missing_crfs = set(EXPECTED_CRFS) - seen_crfs
        unexpected_crfs = seen_crfs - set(EXPECTED_CRFS)
        if missing_crfs:
            errors.append(f"crfs is missing required values {sorted(missing_crfs)}")
        if unexpected_crfs:
            errors.append(f"crfs contains unregistered values {sorted(unexpected_crfs)}")
        if len(raw_crfs) != len(EXPECTED_CRFS):
            errors.append(
                f"crfs has {len(raw_crfs)} entries; exactly {len(EXPECTED_CRFS)} are required"
            )

    if data.get("frames") != EXPECTED_FRAMES:
        errors.append(f"frames must be {EXPECTED_FRAMES}, got {data.get('frames')!r}")
    if data.get("statisticalUnit") != STATISTICAL_UNIT:
        errors.append(
            f"statisticalUnit must be {STATISTICAL_UNIT!r}, got {data.get('statisticalUnit')!r}"
        )

    clips: list[str] = []
    raw_clips = data.get("clips")
    if not isinstance(raw_clips, list):
        errors.append(f"clips must list exactly {EXPECTED_CLIPS} validation clip ids")
    else:
        for clip in raw_clips:
            if not isinstance(clip, str) or not clip:
                errors.append(f"clips contains invalid clip id {clip!r}")
            elif clip in clips:
                errors.append(f"clips contains duplicate clip id {clip!r}")
            else:
                clips.append(clip)
        if len(raw_clips) != EXPECTED_CLIPS:
            errors.append(
                f"clips has {len(raw_clips)} entries; exactly {EXPECTED_CLIPS} are required"
            )

    categories: set[str] = set()
    raw_categories = data.get("categories")
    if not isinstance(raw_categories, dict):
        errors.append(f"categories must name exactly {EXPECTED_CATEGORIES} validation categories")
    else:
        for category in raw_categories:
            if not isinstance(category, str) or not category:
                errors.append(f"categories contains invalid category {category!r}")
            else:
                categories.add(category)
        if len(categories) != EXPECTED_CATEGORIES:
            errors.append(
                f"categories has {len(categories)} names; exactly {EXPECTED_CATEGORIES} are required"
            )

    expected_cells = None
    if len(clips) == EXPECTED_CLIPS:
        expected_cells = {(clip, crf) for clip in clips for crf in EXPECTED_CRFS}
    return expected_cells, categories or None


def candidate_name(name: str) -> tuple[str, str] | None:
    """Return the registered rung and its non-empty seed label."""
    rung, separator, seed = name.partition("-")
    if separator and rung in EXPECTED_RUNGS and seed:
        return rung, seed
    return None


def validate_roster(models: dict, baseline_key: str, corrected: bool,
                    errors: list[str]) -> dict[str, list[str]]:
    """Return candidate model names by rung after checking the registered roster."""
    if baseline_key not in models:
        errors.append(
            f"frozen production model {baseline_key!r} is not in the run; "
            "the pre-registered metric cannot be computed"
        )
    elif candidate_name(baseline_key) is not None:
        errors.append(
            f"baseline key {baseline_key!r} names a rung seed, not the frozen production model"
        )

    arms: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for name in models:
        if name == baseline_key:
            continue
        parsed = candidate_name(name)
        if parsed is None:
            errors.append(
                f"unexpected model {name!r}; candidates must be R0–R3 with a seed name"
            )
            continue
        rung, seed = parsed
        arms[rung].append((seed, name))

    for rung, runs in arms.items():
        seeds = [seed for seed, _ in runs]
        if len(seeds) != len(set(seeds)):
            errors.append(f"{rung} repeats a seed name: {sorted(seeds)!r}")

    if corrected:
        for rung in EXPECTED_RUNGS:
            seeds = {seed for seed, _ in arms[rung]}
            if seeds != EXPECTED_SEEDS:
                errors.append(
                    f"{rung} has seeds {sorted(seeds)}; exactly {sorted(EXPECTED_SEEDS)} are required"
                )

    return {
        rung: [name for _, name in sorted(runs)]
        for rung, runs in arms.items()
    }


def validate_model_cells(name: str, model: object,
                         expected_cells: set[tuple[str, int]] | None,
                         categories: set[str] | None,
                         errors: list[str]) -> dict[tuple[str, int], dict]:
    """Index one model's cells while retaining every coverage defect as an error."""
    rows: dict[tuple[str, int], dict] = {}
    if not isinstance(model, dict):
        errors.append(f"{name} must be an object with perCell rows")
        return rows

    per_cell = model.get("perCell")
    if not isinstance(per_cell, list):
        errors.append(f"{name}.perCell must be a list")
        return rows
    if len(per_cell) != EXPECTED_CELLS:
        errors.append(
            f"{name} has {len(per_cell)} perCell rows; exactly {EXPECTED_CELLS} are required"
        )

    for index, cell in enumerate(per_cell):
        prefix = f"{name}.perCell[{index}]"
        if not isinstance(cell, dict):
            errors.append(f"{prefix} must be an object")
            continue

        clip = cell.get("clip")
        crf = cell.get("crf")
        category = cell.get("category")
        psnr = cell.get("psnr")
        valid_clip = isinstance(clip, str) and bool(clip)
        valid_crf = isinstance(crf, int) and not isinstance(crf, bool) and crf in EXPECTED_CRFS
        if not valid_clip:
            errors.append(f"{prefix}.clip must be a non-empty string, got {clip!r}")
        if not valid_crf:
            errors.append(f"{prefix}.crf must be one of {list(EXPECTED_CRFS)}, got {crf!r}")
        if not isinstance(category, str) or not category:
            errors.append(f"{prefix}.category must be a non-empty string, got {category!r}")
        elif categories is not None and category not in categories:
            errors.append(f"{prefix}.category {category!r} is not declared in categories")
        if (not isinstance(psnr, (int, float)) or isinstance(psnr, bool)
                or not math.isfinite(psnr)):
            errors.append(f"{prefix}.psnr must be finite, got {psnr!r}")
        if "frames" in cell and cell["frames"] != EXPECTED_FRAMES:
            errors.append(f"{prefix}.frames must be {EXPECTED_FRAMES}, got {cell['frames']!r}")

        if valid_clip and valid_crf:
            key = (clip, crf)
            if key in rows:
                errors.append(f"{name} has duplicate clip×CRF cell {cell_name(key)}")
            else:
                rows[key] = cell

    if expected_cells is not None:
        actual_cells = set(rows)
        missing = expected_cells - actual_cells
        unexpected = actual_cells - expected_cells
        if missing:
            errors.append(f"{name} is missing clip×CRF cells: {cell_names(missing)}")
        if unexpected:
            errors.append(f"{name} has unexpected clip×CRF cells: {cell_names(unexpected)}")
    return rows


def validate_categories(rows_by_model: dict[str, dict[tuple[str, int], dict]],
                        baseline_key: str, expected_cells: set[tuple[str, int]] | None,
                        categories: set[str] | None, errors: list[str]) -> None:
    """Check category identity cell-for-cell rather than pooling mismatched rows."""
    if expected_cells is None or baseline_key not in rows_by_model:
        return
    baseline = rows_by_model[baseline_key]
    baseline_categories: dict[tuple[str, int], str] = {}
    category_by_clip: dict[str, str] = {}
    for key in expected_cells:
        row = baseline.get(key)
        if row is None:
            continue
        category = row.get("category")
        if not isinstance(category, str) or not category:
            continue
        baseline_categories[key] = category
        clip = key[0]
        existing = category_by_clip.setdefault(clip, category)
        if existing != category:
            errors.append(
                f"{baseline_key} assigns clip {clip!r} to both {existing!r} and {category!r}"
            )

    if categories is not None and set(category_by_clip.values()) != categories:
        errors.append(
            f"categories metadata does not match {baseline_key}'s validation category cells"
        )

    for name, rows in rows_by_model.items():
        if name == baseline_key:
            continue
        for key, baseline_category in baseline_categories.items():
            row = rows.get(key)
            if row is None:
                continue
            if row.get("category") != baseline_category:
                errors.append(
                    f"{name} category for {cell_name(key)} is {row.get('category')!r}; "
                    f"production baseline has {baseline_category!r}"
                )


def validate_screening(data: dict, baseline_key: str,
                       corrected: bool) -> dict[str, list[str]]:
    """Reject a malformed screen before any rows can be silently discarded."""
    errors: list[str] = []
    expected_cells, categories = validate_layout(data, errors)
    models = data.get("models")
    if not isinstance(models, dict):
        errors.append("models must be an object keyed by model name")
        raise ScreeningInputError(errors)

    arms = validate_roster(models, baseline_key, corrected, errors)
    rows_by_model = {
        name: validate_model_cells(name, model, expected_cells, categories, errors)
        for name, model in models.items()
    }
    validate_categories(rows_by_model, baseline_key, expected_cells, categories, errors)
    if not arms:
        errors.append("screening has no registered rung seed models to audit")
    if errors:
        raise ScreeningInputError(errors)
    return arms


def per_clip_delta_vs(model: dict,
                      baseline_psnr: dict[tuple[str, int], float]) -> dict[str, float]:
    """Mean over CRFs within a clip, of candidate-minus-baseline PSNR."""
    by_clip: dict[str, list[float]] = collections.defaultdict(list)
    for cell in model["perCell"]:
        key = (cell["clip"], cell["crf"])
        by_clip[cell["clip"]].append(cell["psnr"] - baseline_psnr[key])
    return {clip: st.fmean(values) for clip, values in by_clip.items()}


def exact_permutation_p(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-sided p over all label assignments, with the attainable floor."""
    pool = a + b
    n = len(a)
    obs = abs(st.fmean(b) - st.fmean(a))
    hits = total = 0
    for combo in itertools.combinations(range(len(pool)), n):
        x = [pool[index] for index in combo]
        y = [pool[index] for index in range(len(pool)) if index not in combo]
        total += 1
        if abs(st.fmean(y) - st.fmean(x)) >= obs - 1e-12:
            hits += 1
    return hits / total, 2.0 / total


def summarize_arms(models: dict, arms: dict[str, list[str]],
                   baseline_key: str) -> dict[str, dict]:
    """Calculate only the registered production-relative, clip-first summaries."""
    baseline_psnr = {
        (cell["clip"], cell["crf"]): cell["psnr"]
        for cell in models[baseline_key]["perCell"]
    }
    summary: dict[str, dict] = {}
    for rung in sorted(arms, key=SIMPLICITY_ORDER.index):
        runs = []
        for name in arms[rung]:
            model = models[name]
            deltas = per_clip_delta_vs(model, baseline_psnr)
            by_crf: dict[int, list[float]] = collections.defaultdict(list)
            by_category_crf: dict[tuple[str, int], list[float]] = collections.defaultdict(list)
            for cell in model["perCell"]:
                delta = cell["psnr"] - baseline_psnr[(cell["clip"], cell["crf"])]
                by_crf[cell["crf"]].append(delta)
                by_category_crf[(cell["category"], cell["crf"])].append(delta)
            runs.append({
                "seed": candidate_name(name)[1],
                "meanDelta": st.fmean(deltas.values()),
                "byCrf": {str(crf): st.fmean(values) for crf, values in sorted(by_crf.items())},
                "byCategoryCrf": {
                    f"{category}@{crf}": st.fmean(values)
                    for (category, crf), values in sorted(by_category_crf.items())
                },
            })

        values = [run["meanDelta"] for run in runs]
        per_category_crf = {
            key: st.fmean([run["byCategoryCrf"][key] for run in runs])
            for key in runs[0]["byCategoryCrf"]
        }
        summary[rung] = {
            "seeds": len(values),
            "seedNames": [run["seed"] for run in runs],
            "mean": st.fmean(values),
            "sd": st.stdev(values) if len(values) > 1 else 0.0,
            "perSeed": values,
            "byCrf": {
                key: st.fmean([run["byCrf"][key] for run in runs])
                for key in runs[0]["byCrf"]
            },
            "perCategoryCrfOnValidation": per_category_crf,
        }
    return summary


def comparison_stats(summary: dict[str, dict], ranked: list[str]) -> dict[str, dict]:
    """Disclose the pre-existing three-seed permutation calculation."""
    stats = {}
    if "R0" in summary:
        for rung in ranked:
            if rung == "R0":
                continue
            p, floor = exact_permutation_p(summary["R0"]["perSeed"], summary[rung]["perSeed"])
            stats[f"{rung}_vs_R0"] = {
                "deltaDb": summary[rung]["mean"] - summary["R0"]["mean"],
                "p": p,
                "attainableFloor": floor,
            }
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screening", required=True)
    ap.add_argument("--baseline-key", default="aethersr-c16d2")
    ap.add_argument(
        "--input-status",
        choices=INPUT_STATUSES,
        help=f"required when {INPUT_STATUS_KEY} metadata is absent",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        with open(args.screening, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as error:
        print(f"  could not read screening input: {error}", file=sys.stderr)
        return 1
    if not isinstance(data, dict):
        print("  screening input must be a JSON object", file=sys.stderr)
        return 1

    try:
        status, status_source = resolve_input_status(data, args.screening, args.input_status)
        arms = validate_screening(data, args.baseline_key, status == CORRECTED_STATUS)
    except ScreeningInputError as error:
        for detail in error.errors:
            print(f"  screening input rejected: {detail}", file=sys.stderr)
        return 1

    summary = summarize_arms(data["models"], arms, args.baseline_key)
    ranked = sorted(summary, key=lambda rung: summary[rung]["mean"], reverse=True)
    stats = comparison_stats(summary, ranked)
    report = {
        "schema": "aethervsr.m7-screen/2",
        "inputStatus": status,
        "inputStatusSource": status_source,
        "auditOnly": status == WITHDRAWN_STATUS,
        "metric": "per-clip mean PSNR delta vs the frozen production model, paired on clip x CRF",
        "statisticalUnit": STATISTICAL_UNIT,
        "baseline": {
            "key": args.baseline_key,
            "role": "frozen production model",
        },
        "tieBandDb": TIE_BAND_DB,
        "ranking": ranked,
        "arms": summary,
        "significance": stats,
        "historicalConfirmationCells": {
            "valuesDb": HISTORICAL_CONFIRMATION_CELLS,
            "note": (
                "These are historical Milestone 6 confirmation-corpus values, not "
                "validation cells from this screen. Validation category×CRF values "
                "are reported only under arms.*.perCategoryCrfOnValidation."
            ),
        },
        "provenanceEligibilityPrerequisite": (
            "This report validates declared status and scored-cell coverage only. A corrected "
            "ranking remains conditional on separately verified corrected corpus and validation "
            "provenance, frozen-production-baseline identity, and each model's registered "
            "16,200 optimizer-update training record."
        ),
        "budgetCaveat": (
            "Screening runs the pre-registered 16,200-step budget, which is about a fifth of "
            "the 81,180 steps the shipped model was trained for. A reparameterization benefit "
            "that only appears late in a schedule would not be visible here. Raising the "
            "budget would be a different experiment and must be declared before it is run."
        ),
    }

    if status == CORRECTED_STATUS:
        top = ranked[0]
        tied = [
            rung for rung in ranked
            if summary[top]["mean"] - summary[rung]["mean"] <= TIE_BAND_DB
        ]
        winner = min(tied, key=SIMPLICITY_ORDER.index)
        report.update({
            "argmax": top,
            "tiedWithArgmax": tied,
            "winner": winner,
            "winnerRationale": (
                f"{top} has the highest mean; {sorted(tied)} lie within {TIE_BAND_DB} dB of it, "
                f"and the pre-registered tie-break takes the simplest of those, {winner}."
                if len(tied) > 1
                else f"{top} wins outright, no rung within {TIE_BAND_DB} dB."
            ),
            "nullResult": winner == "R0",
            "whatANullDoesNotMean": (
                "With 3 seeds per arm the exact unpaired permutation test cannot return below "
                f"{2.0 / 20:.3f}, so it can never reach p<0.05 no matter how large the effect. "
                "Selecting R0 therefore means 'no gain was detected at this budget under this "
                "rule'. It is not evidence that the rungs are optimization-equivalent, and it "
                "does not bound how large an undetected effect could be."
            ),
        })
    else:
        report["selectionIneligibility"] = (
            "Withdrawn/confounded pilot input: arithmetic is retained for audit, but this "
            "report intentionally has no winner or null verdict."
        )

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    print(f"  {'rung':<6}{'n':>2}{'mean dB':>10}{'sd':>8}   per-seed", file=sys.stderr)
    for rung in ranked:
        row = summary[rung]
        print(f"  {rung:<6}{row['seeds']:>2}{row['mean']:>10.4f}{row['sd']:>8.4f}   "
              f"{' '.join(f'{value:+.4f}' for value in row['perSeed'])}", file=sys.stderr)
    for name, row in stats.items():
        print(f"  {name}: {row['deltaDb']:+.4f} dB  p={row['p']:.3f} "
              f"(floor {row['attainableFloor']:.3f})", file=sys.stderr)
    if status == CORRECTED_STATUS:
        print(f"\n  winner: {report['winner']}"
              f"{'  -> NULL RESULT, current architecture retained' if report['winner'] == 'R0' else ''}",
              file=sys.stderr)
        print(f"  {report['winnerRationale']}", file=sys.stderr)
    else:
        print("\n  audit only: withdrawn/confounded pilot; no winner or null verdict", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
