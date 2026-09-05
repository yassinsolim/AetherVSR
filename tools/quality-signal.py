#!/usr/bin/env python3
"""Does a cheap, GPU-computable signal predict where the neural gain disappears?

Milestone 6 may want to switch upscalers based on input quality. Before anyone
builds that selector, this asks whether the switch is even predictable: is there
an observable statistic of the *decoded frame alone* - no reference, no encoder
metadata - that tracks whether the neural stage will beat Catmull-Rom?

Diagnostic only. Nothing here selects a model, and no selector is built.

Every candidate signal is computable from the LR frame in a shader, because a
signal that needs a CPU readback is useless in this pipeline:

  gradientEnergy   mean |grad|, a sharpness proxy
  highFreqEnergy   energy remaining after a 3x3 box blur
  blockiness       8x8 grid discontinuity relative to off-grid, the classic
                   H.264 blocking measure
  lumaVariance     scene-level contrast, as a content control

The last one matters: a signal that only tracks scene content would look
predictive on a corpus where hard scenes happen to be heavily compressed,
without transferring to a new clip. Reporting it alongside makes that visible.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

import torch
import torch.nn.functional as F
from PIL import Image

from evaluate import catmull_rom_2x, load_model, psnr


def load_png(path: str) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        raw = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        return raw.reshape(h, w, 3).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def luma(x: torch.Tensor) -> torch.Tensor:
    return 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]


def gradient_energy(y: torch.Tensor) -> float:
    dx = (y[:, :, :, 1:] - y[:, :, :, :-1]).abs().mean()
    dy = (y[:, :, 1:, :] - y[:, :, :-1, :]).abs().mean()
    return float((dx + dy) / 2)


def high_freq_energy(y: torch.Tensor) -> float:
    blur = F.avg_pool2d(F.pad(y, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    return float((y - blur).abs().mean())


def blockiness(y: torch.Tensor) -> float:
    """Discontinuity on the 8x8 grid relative to off-grid, in the same units.

    H.264 quantises on an 8x8 (and 4x4) grid, so heavy compression leaves steps
    exactly on that lattice. Normalising by the off-grid differences separates
    blocking from ordinary scene detail: a busy scene raises both, blocking
    raises only the first.
    """
    d = (y[:, :, :, 1:] - y[:, :, :, :-1]).abs()[0, 0]
    cols = d.mean(dim=0)
    on = cols[7::8].mean()
    mask = torch.ones(cols.shape[0], dtype=torch.bool)
    mask[7::8] = False
    off = cols[mask].mean()
    return float(on / (off + 1e-8))


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = st.fmean(xs), st.fmean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    return pearson(rank(xs), rank(ys))


def main() -> int:
    ap = argparse.ArgumentParser(description="Correlate cheap frame statistics with neural gain.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--curve", required=True,
                    help="a gop-diagnostic.py report; its decoded frames are reused")
    ap.add_argument("--root", default="data/captured")
    ap.add_argument("--out", default="results/quality-signal.json")
    args = ap.parse_args()

    model, _ = load_model(args.model)
    model.eval()
    with open(args.curve, encoding="utf-8") as fh:
        curve = json.load(fh)
    with open(os.path.join(args.root, "prepared.json"), encoding="utf-8") as fh:
        prepared = {c["id"]: c for c in json.load(fh)["clips"]}

    rows: list[dict] = []
    for rec in curve["perClip"]:
        if rec["structure"] != "gop":
            continue
        cid, crf = rec["clip"], rec["crf"]
        dec = os.path.join(curve.get("workRoot", "data/captured/gopdiag"),
                           cid, f"gop_crf{crf}", "decoded")
        if not os.path.isdir(dec):
            continue
        master = os.path.join(args.root, "clips", cid, "master")
        names = sorted(n for n in os.listdir(dec) if n.endswith(".png"))[:6]
        sig = {"gradientEnergy": [], "highFreqEnergy": [], "blockiness": [], "lumaVariance": []}
        deltas = []
        for n in names:
            lr = load_png(os.path.join(dec, n))
            y = luma(lr)
            sig["gradientEnergy"].append(gradient_energy(y))
            sig["highFreqEnergy"].append(high_freq_energy(y))
            sig["blockiness"].append(blockiness(y))
            sig["lumaVariance"].append(float(y.var()))
            hr_path = os.path.join(master, n)
            if os.path.exists(hr_path):
                hr = load_png(hr_path)
                with torch.no_grad():
                    neural = model(lr).clamp(0, 1)
                deltas.append(psnr(neural, hr) - psnr(catmull_rom_2x(lr), hr))
        if not deltas:
            continue
        rows.append({
            "clip": cid, "category": prepared.get(cid, {}).get("category", "?"), "crf": crf,
            "delta": st.fmean(deltas),
            **{k: st.fmean(v) for k, v in sig.items()},
        })
        print(f"  {cid[:30]:<30} crf{crf:<3} delta {rows[-1]['delta']:+.4f}  "
              f"grad {rows[-1]['gradientEnergy']:.5f}  hf {rows[-1]['highFreqEnergy']:.5f}  "
              f"block {rows[-1]['blockiness']:.3f}", file=sys.stderr)

    signals = ["gradientEnergy", "highFreqEnergy", "blockiness", "lumaVariance"]
    deltas = [r["delta"] for r in rows]
    correlations = {
        s: {"pearson": pearson([r[s] for r in rows], deltas),
            "spearman": spearman([r[s] for r in rows], deltas)}
        for s in signals
    }

    # A signal is only useful if it separates the two regimes it would switch
    # between. Split at the point the gain stops mattering and ask whether the
    # signal distributions actually differ there.
    BENEFIT_DB = 0.15
    beneficial = [r for r in rows if r["delta"] >= BENEFIT_DB]
    marginal = [r for r in rows if r["delta"] < BENEFIT_DB]
    separation = {}
    for s in signals:
        if not beneficial or not marginal:
            continue
        a = [r[s] for r in beneficial]
        b = [r[s] for r in marginal]
        pooled = (st.pstdev(a) + st.pstdev(b)) / 2 or 1e-9
        separation[s] = {
            "beneficialMean": st.fmean(a), "marginalMean": st.fmean(b),
            "standardisedDifference": (st.fmean(a) - st.fmean(b)) / pooled,
            "beneficialN": len(a), "marginalN": len(b),
        }

    report = {
        "schema": "aethervsr.quality-signal/1",
        "diagnosticOnly": ("Reports whether a cheap no-reference signal tracks the neural "
                           "gain. No selector is built and nothing here chooses a model."),
        "model": os.path.basename(args.model),
        "benefitThresholdDb": BENEFIT_DB,
        "cells": len(rows),
        "perCell": rows,
        "correlations": correlations,
        "separation": separation,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    print(f"\n  {'signal':<18}{'pearson':>10}{'spearman':>11}{'std diff':>11}", file=sys.stderr)
    for s in signals:
        c = correlations[s]
        sep = separation.get(s, {}).get("standardisedDifference", float("nan"))
        print(f"  {s:<18}{c['pearson']:>+10.3f}{c['spearman']:>+11.3f}{sep:>+11.3f}", file=sys.stderr)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
