#!/usr/bin/env python3
"""Materialise a training subset as the pair directory train.py already reads.

The corpus is prepared once into a single tensor of patches plus an index that
records which clip each sequence came from. A subset is then a selection over
that index, never a re-preparation: re-cutting patches per subset would give
each scale slightly different crops and the scaling curve would be measuring
crop luck alongside corpus size.

Validation patches are the same for every subset, from the separate validation
corpus, because a scaling curve whose yardstick changes with the thing being
measured is not a curve.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import torch


def patch_labels(index: list[dict]) -> list[str]:
    """Expand the per-sequence index into one clip id per stored patch.

    Patches were appended in index order, so this reconstruction is exact. It is
    rebuilt rather than stored because a stored copy can drift from the tensor
    it describes, and a silent misalignment here would put the wrong clips in
    every subset.
    """
    out: list[str] = []
    for r in index:
        out += [r["clip"]] * r["patches"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Materialise a subset pair directory.")
    ap.add_argument("--pairs", default="/tmp/m6prep/train")
    ap.add_argument("--val-pairs", default="/tmp/m6prep/val")
    ap.add_argument("--subsets", default="data/captured-train-v2/subsets.json")
    ap.add_argument("--select", required=True,
                    help="a scale name such as N48, or 'concentrated' / 'diverse'")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(os.path.join(args.pairs, "pairs.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(args.subsets, encoding="utf-8") as fh:
        subsets = json.load(fh)

    if args.select in subsets["scales"]:
        spec = subsets["scales"][args.select]
    elif args.select in subsets["diversityArms"]:
        spec = subsets["diversityArms"][args.select]
    else:
        raise SystemExit(f"unknown selection {args.select!r}; "
                         f"have {sorted(subsets['scales'])} and "
                         f"{[k for k in subsets['diversityArms'] if k != 'matchedClipCount']}")
    wanted = set(spec["ids"])

    labels = patch_labels(meta["index"])
    blob = torch.load(os.path.join(args.pairs, "all.pt"))
    if len(labels) != blob["hr"].shape[0]:
        raise SystemExit(f"index describes {len(labels)} patches but tensor holds "
                         f"{blob['hr'].shape[0]}; the two have drifted apart")
    keep = [i for i, c in enumerate(labels) if c in wanted]
    if not keep:
        raise SystemExit(f"{args.select} selected no patches")

    os.makedirs(args.out, exist_ok=True)
    torch.save({"lr": blob["lr"][keep], "hr": blob["hr"][keep]},
               os.path.join(args.out, "train.pt"))

    vblob = torch.load(os.path.join(args.val_pairs, "all.pt"))
    torch.save({"lr": vblob["lr"], "hr": vblob["hr"]}, os.path.join(args.out, "val.pt"))

    sel = [r for r in meta["index"] if r["clip"] in wanted]
    out_meta = {
        "schema": "aethervsr.video-pairs/2",
        "subset": args.select,
        "structure": meta["structure"],
        "structureSpec": {"x264params": meta["x264params"],
                          "description": "normal libx264 GOP with I/P/B frames"},
        "crfDistribution": meta["crfDistribution"],
        "crfSpec": {"description": meta["crfDistribution"]},
        "preset": meta["preset"], "seed": meta["seed"], "fps": 24,
        "sequenceFrames": meta["sequenceFrames"],
        "hrSize": meta["hrSize"], "lrSize": meta["lrSize"],
        "sourceManifest": meta["sourceManifest"],
        "splitUnit": "creator; validation is a separate creator-disjoint corpus",
        "trainClips": sorted(wanted),
        "valClips": sorted({r["clip"] for r in json.load(
            open(os.path.join(args.val_pairs, "pairs.json"), encoding="utf-8"))["index"]}),
        "sequences": len(sel),
        "crfHistogram": {str(c): n for c, n in
                         sorted(collections.Counter(r["crf"] for r in sel).items())},
        "categories": dict(collections.Counter(r["category"] for r in sel).most_common()),
        "creators": len({r["creator"] for r in sel}),
        "trainPatches": len(keep), "valPatches": int(vblob["hr"].shape[0]),
        "index": sel, "encodeEvidence": [],
    }
    with open(os.path.join(args.out, "pairs.json"), "w", encoding="utf-8") as fh:
        json.dump(out_meta, fh, indent=1)

    print(f"  {args.select:<14} {len(wanted):>3} clips  {out_meta['creators']:>3} creators  "
          f"{len(keep):>6} train patches  {out_meta['valPatches']} val patches", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
