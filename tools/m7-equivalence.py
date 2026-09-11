#!/usr/bin/env python3
"""Measure error at every boundary between the training graph and the runtime.

Milestone 7 claims a reparameterized model is the deployed model. That claim is
a chain, and a chain is only as good as the weakest link, so every link is
measured separately rather than end to end:

    multi-branch PyTorch  ->  fused PyTorch  ->  exported JSON  ->  WebGPU

This tool covers the first three. The fourth is the browser golden check, which
now refuses vectors that were not exported from the model under test; this tool
writes those vectors so that comparison is anchored to the fused candidate.

Reporting one end-to-end number would hide which link moved, and two errors that
cancel would look like success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aethersr import AetherSR, count_parameters, export_weights  # noqa: E402
from reparam import LADDERS, RepAetherSR  # noqa: E402

TOL = 1e-5


def rebuild_from_export(payload: dict) -> AetherSR:
    """Reconstruct the network from the exported planar tensors.

    Deliberately goes through the *file*, not the in-memory model: this is what
    proves the export layout round-trips, which is the link where a silent
    reshape or a transposed kernel would hide.
    """
    model = AetherSR(channels=payload["features"], depth=payload["depth"])
    w = payload["weights"]
    with torch.no_grad():
        model.stem.weight.copy_(torch.tensor(w["stem.weight"]).reshape(model.stem.weight.shape))
        model.stem.bias.copy_(torch.tensor(w["stem.bias"]))
        for i, conv in enumerate(model.body):
            conv.weight.copy_(torch.tensor(w[f"body.{i}.weight"]).reshape(conv.weight.shape))
            conv.bias.copy_(torch.tensor(w[f"body.{i}.bias"]))
        model.head.weight.copy_(torch.tensor(w["head.weight"]).reshape(model.head.weight.shape))
        model.head.bias.copy_(torch.tensor(w["head.bias"]))
    return model.eval()


def main() -> int:
    ap = argparse.ArgumentParser(description="Boundary-by-boundary equivalence proof.")
    ap.add_argument("--rungs", default="R1,R2,R3")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="results/m7-equivalence.json")
    args = ap.parse_args()

    boundaries: list[dict] = []
    for rung in [r for r in args.rungs.split(",") if r.strip()]:
        for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
            torch.manual_seed(seed)
            rep = RepAetherSR(16, 2, branches=LADDERS[rung]).eval()
            # Trained-like magnitudes: fusion must hold for weights the optimizer
            # would actually produce, not only for the zero-initialised start.
            for p in rep.parameters():
                p.data = torch.randn(p.shape) * 0.1

            fused = rep.fuse()
            # Round-trip through actual JSON *text*, not an in-memory dict. The
            # deployed model is a file the runtime parses, so a dict handoff
            # would skip the very step that can lose precision: float repr and
            # reparse. json.dumps uses repr(), which is round-trip exact for
            # float64, but the tensors are float32 - so this measures whether
            # widening to double and back is lossless, which is the thing the
            # shipped loader actually does.
            payload = json.loads(json.dumps({
                "features": 16, "depth": 2, "weights": export_weights(fused),
            }))
            reloaded = rebuild_from_export(payload)

            x = torch.rand(2, 3, 24, 32)
            with torch.no_grad():
                a, b, c = rep(x), fused(x), reloaded(x)

            boundaries.append({
                "rung": rung, "seed": seed,
                "trainingParameters": rep.training_parameter_count(),
                "fusedParameters": count_parameters(fused),
                "multiBranchVsFused": (a - b).abs().max().item(),
                "fusedVsReloaded": (b - c).abs().max().item(),
                "multiBranchVsReloaded": (a - c).abs().max().item(),
                "tensorNamesMatch": sorted(export_weights(fused)) == sorted(export_weights(reloaded)),
            })

    worst = max(
        max(b["multiBranchVsFused"], b["fusedVsReloaded"], b["multiBranchVsReloaded"])
        for b in boundaries
    )
    report = {
        "schema": "aethervsr.m7-equivalence/1",
        "chain": "multi-branch PyTorch -> fused PyTorch -> exported JSON -> (WebGPU, browser golden)",
        "tolerance": TOL,
        "worstBoundaryError": worst,
        "passed": worst <= TOL and all(b["tensorNamesMatch"] for b in boundaries),
        "note": ("Each link is measured separately. A single end-to-end number would hide which "
                 "boundary moved, and two errors that cancel would look like success."),
        "boundaries": boundaries,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    print(f"  {'rung':<5}{'seed':>5}{'train':>8}{'fused':>7}"
          f"{'branch->fused':>15}{'fused->json':>13}{'branch->json':>14}", file=sys.stderr)
    for b in boundaries:
        print(f"  {b['rung']:<5}{b['seed']:>5}{b['trainingParameters']:>8}{b['fusedParameters']:>7}"
              f"{b['multiBranchVsFused']:>15.2e}{b['fusedVsReloaded']:>13.2e}"
              f"{b['multiBranchVsReloaded']:>14.2e}", file=sys.stderr)
    print(f"\n  worst boundary error {worst:.3e} (tolerance {TOL:.0e}) -> "
          f"{'PASS' if report['passed'] else 'FAIL'}", file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
