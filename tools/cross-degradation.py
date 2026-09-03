#!/usr/bin/env python3
"""Cross-degradation matrix: how each trained model behaves under each condition.

Rows are training conditions, columns are evaluation conditions. The question
this answers is whether training on realistic web-video degradation helps on
compressed input, and what it costs on clean input - which a single aggregate
score cannot show, because "reconstructing detail" and "removing compression
artefacts" are different jobs that happen to share a metric.

Every cell is `neural mean PSNR - Catmull-Rom mean PSNR` on the same images
under the same degradation, so the baseline moves with the condition and the
comparison is never against a fixed reference that flatters one row.

Seeds are aggregated: each cell is a mean over N seeds with its standard
deviation, so a difference can be read against the noise the training procedure
already produces rather than against zero.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys

PY = sys.executable
EVAL = os.path.join(os.path.dirname(__file__), "evaluate.py")


def score(model: str, which: str, profile: str, limit: int, seed: int) -> dict:
    out = subprocess.run(
        [PY, EVAL, "--model", model, "--set", which, "--profile", profile,
         "--limit", str(limit), "--seed", str(seed), "--json"],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise SystemExit(f"evaluate.py failed: {model} / {which} / {profile}\n{out.stderr[-1500:]}")
    return json.loads(out.stdout)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-degradation evaluation matrix.")
    ap.add_argument("--rows", default="box=models/box/seed*.json,realistic=models/realistic/seed*.json")
    ap.add_argument("--columns", default="box,bicubic,lanczos,h264_high,h264_typical,h264_poor")
    ap.add_argument("--set", default="independent")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7, help="fixed seed for randomised degradation, so every row sees identical inputs")
    ap.add_argument("--out", default="results/cross-degradation.json")
    args = ap.parse_args()

    rows = {}
    for spec in args.rows.split(","):
        name, pattern = spec.split("=", 1)
        models = sorted(glob.glob(pattern))
        if not models:
            raise SystemExit(f"no models matched {pattern!r}")
        rows[name] = models

    columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    matrix: dict[str, dict[str, dict]] = {}

    for row_name, models in rows.items():
        matrix[row_name] = {}
        for column in columns:
            deltas, neural, catmull = [], [], []
            for model in models:
                r = score(model, args.set, column, args.limit, args.seed)
                deltas.append(r["neuralOverCatmullRom"]["psnrDb"])
                neural.append(r["summary"]["neural"]["psnr"]["mean"])
                catmull.append(r["summary"]["catmull_rom"]["psnr"]["mean"])
            matrix[row_name][column] = {
                "deltaPsnrDbMean": statistics.fmean(deltas),
                "deltaPsnrDbStd": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "neuralPsnrMean": statistics.fmean(neural),
                "catmullPsnrMean": statistics.fmean(catmull),
                "seeds": len(models),
            }
            cell = matrix[row_name][column]
            print(
                f"  {row_name:>10} x {column:<14} "
                f"neural {cell['neuralPsnrMean']:6.3f}  catmull {cell['catmullPsnrMean']:6.3f}  "
                f"delta {cell['deltaPsnrDbMean']:+6.3f} +/- {cell['deltaPsnrDbStd']:.3f}",
                file=sys.stderr,
            )

    report = {
        "schema": "aethervsr.cross-degradation/1",
        "evaluationSet": args.set,
        "images": args.limit,
        "degradationSeed": args.seed,
        "cell": "neural mean PSNR minus Catmull-Rom mean PSNR on identical inputs, mean +/- std over seeds",
        "rows": {k: [os.path.basename(m) for m in v] for k, v in rows.items()},
        "columns": columns,
        "matrix": matrix,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    width = max(len(c) for c in columns) + 2
    print("\ndelta dB (neural - Catmull-Rom), mean +/- std over seeds\n")
    print("  " + "train\\eval".ljust(12) + "".join(c.ljust(width) for c in columns))
    for row_name in matrix:
        cells = "".join(
            f"{matrix[row_name][c]['deltaPsnrDbMean']:+.3f}".ljust(width) for c in columns
        )
        print("  " + row_name.ljust(12) + cells)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
