#!/usr/bin/env python3
"""Aggregate a set of seed runs into mean / std / min / max.

The point is not to find the best seed. It is to know how big the spread is, so
that a later comparison between two configurations can say whether an observed
difference is larger than the noise the training procedure already produces.

Reports every run. Publishing only the best seed is how a lucky draw becomes a
claimed improvement.
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


def load_runs(pattern: str) -> list[dict]:
    runs = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        runs.append({"path": path, "payload": payload})
    if not runs:
        raise SystemExit(f"no model files matched {pattern!r}")
    return runs


def score(model_path: str, which: str, profile: str, limit: int) -> dict:
    """Runs the evaluator as a subprocess so scoring uses exactly the shipped path."""
    cmd = [
        PY,
        os.path.join(os.path.dirname(__file__), "evaluate.py"),
        "--model", model_path,
        "--set", which,
        "--profile", profile,
        "--limit", str(limit),
        "--json",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise SystemExit(f"evaluate.py failed for {model_path} ({which}):\n{out.stderr[-2000:]}")
    return json.loads(out.stdout)


def stats(values: list[float]) -> dict:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {
        "mean": statistics.fmean(values),
        # Sample standard deviation: these seeds are a sample of the training
        # procedure's output, not the whole population of it.
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate seed runs.")
    ap.add_argument("--models", default="models/box/seed*.json")
    ap.add_argument("--sets", default="val,test,independent")
    ap.add_argument("--profile", default="box")
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--out", default="")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    runs = load_runs(args.models)
    wanted = [s.strip() for s in args.sets.split(",") if s.strip()]

    rows = []
    for run in runs:
        training = run["payload"].get("training", {})
        row = {
            "model": os.path.basename(run["path"]),
            "seed": training.get("seed"),
            "weightsSha256": run["payload"].get("sha256") or run["payload"].get("weightsSha256"),
            "valPsnrDbAtCheckpoint": training.get("valPsnrDb"),
            "valSsimAtCheckpoint": training.get("valSsim"),
            "splitDigest": training.get("splitDigest"),
            "degradation": training.get("degradationProfile"),
            "scores": {},
        }
        for which in wanted:
            result = score(run["path"], which, args.profile, args.limit)
            row["scores"][which] = {
                "neuralPsnr": result["summary"]["neural"]["psnr"]["mean"],
                "neuralSsim": result["summary"]["neural"]["ssim"]["mean"],
                "catmullPsnr": result["summary"]["catmull_rom"]["psnr"]["mean"],
                "catmullSsim": result["summary"]["catmull_rom"]["ssim"]["mean"],
                "bilinearPsnr": result["summary"]["bilinear"]["psnr"]["mean"],
                "deltaPsnrDb": result["neuralOverCatmullRom"]["psnrDb"],
                "deltaSsim": result["neuralOverCatmullRom"]["ssim"],
                "wins": result["neuralOverCatmullRom"]["winsOf"],
                "images": result["images"],
            }
        rows.append(row)
        print(f"scored {row['model']} (seed {row['seed']})", file=sys.stderr)

    aggregate = {}
    for which in wanted:
        aggregate[which] = {
            key: stats([r["scores"][which][key] for r in rows])
            for key in ("neuralPsnr", "neuralSsim", "catmullPsnr", "deltaPsnrDb", "deltaSsim")
        }

    report = {
        "schema": "aethervsr.seed-aggregate/1",
        "models": args.models,
        "degradationProfileAtEval": args.profile,
        "runs": rows,
        "aggregate": aggregate,
        "note": (
            "Every run is listed. The spread here is the noise floor against which "
            "any configuration change must be judged."
        ),
    }

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        print(f"wrote {args.out}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=1))
        return 0

    print(f"\n{len(rows)} runs, evaluated with degradation profile {args.profile!r}\n")
    for which in wanted:
        print(f"--- {which} ---")
        header = f"  {'seed':>6}{'neural PSNR':>13}{'Catmull':>10}{'delta dB':>10}{'SSIM d':>9}  wins"
        print(header)
        for r in rows:
            s = r["scores"][which]
            print(
                f"  {str(r['seed']):>6}{s['neuralPsnr']:>13.3f}{s['catmullPsnr']:>10.3f}"
                f"{s['deltaPsnrDb']:>+10.3f}{s['deltaSsim']:>+9.4f}  {s['wins']}"
            )
        a = aggregate[which]
        print(
            f"  {'mean':>6}{a['neuralPsnr']['mean']:>13.3f}{a['catmullPsnr']['mean']:>10.3f}"
            f"{a['deltaPsnrDb']['mean']:>+10.3f}{a['deltaSsim']['mean']:>+9.4f}"
        )
        print(
            f"  {'std':>6}{a['neuralPsnr']['std']:>13.3f}{'':>10}"
            f"{a['deltaPsnrDb']['std']:>10.3f}{a['deltaSsim']['std']:>9.4f}"
        )
        print(
            f"  {'range':>6}{a['neuralPsnr']['min']:>13.3f}..{a['neuralPsnr']['max']:.3f}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
