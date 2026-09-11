#!/usr/bin/env python3
"""Split a corpus manifest for parallel preparation, then merge the results.

Preparation is network-bound: each clip is seeked and decoded straight from
Wikimedia over HTTP, which runs about five minutes per clip. Sequentially that
is half a day for 151 clips, and the machine is idle for almost all of it.

Sharding is safe here for a specific reason, not by luck. `corpus-prepare.py`
derives every per-clip random choice from `rng_for(..., seed, cid)` - keyed by
the clip id, never by position in the list - so a clip yields the same
sequences, the same CRFs and the same patches no matter which shard prepares
it. Merging in manifest order therefore reproduces a sequential run exactly.

Shards must be disjoint manifest slices written to *separate* output
directories. Pointing two processes at one `--out` would have them overwrite
each other's `all.pt` and metadata and collide in `_work`.

    split  - write K manifest shards
    merge  - reassemble one pair directory in original manifest order
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch


def split(manifest_path: str, shards: int, out_dir: str) -> int:
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    clips = manifest["clips"]
    os.makedirs(out_dir, exist_ok=True)

    # Round-robin rather than contiguous blocks: clip duration and resolution
    # vary a lot, and contiguous blocks would leave one shard holding all the
    # long 4K material while the others finished early.
    for k in range(shards):
        shard = dict(manifest)
        shard["clips"] = clips[k::shards]
        with open(os.path.join(out_dir, f"shard{k}.json"), "w", encoding="utf-8") as fh:
            json.dump(shard, fh)
        print(f"  shard{k}: {len(shard['clips'])} clips", file=sys.stderr)
    print(f"  {len(clips)} clips over {shards} shards", file=sys.stderr)
    return 0


def merge(manifest_path: str, shard_dirs: list[str], out_dir: str) -> int:
    with open(manifest_path, encoding="utf-8") as fh:
        order = [c["id"] for c in json.load(fh)["clips"]]

    # tag -> (lr, hr, index entry). Tags are unique across shards because each
    # clip appears in exactly one shard.
    owned: dict[str, tuple[torch.Tensor, torch.Tensor, dict]] = {}
    evidence: list[dict] = []
    failures: list[dict] = []
    meta_seed: dict = {}

    for d in shard_dirs:
        meta_path = os.path.join(d, "pairs.json")
        pt_path = os.path.join(d, "all.pt")
        if not (os.path.exists(meta_path) and os.path.exists(pt_path)):
            print(f"  missing shard output in {d}", file=sys.stderr)
            return 1
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        blob = torch.load(pt_path)
        meta_seed = meta_seed or meta
        evidence.extend(meta.get("evidence", []))
        failures.extend(meta.get("failures", []))

        at = 0
        for entry in meta["index"]:
            n = entry["patches"]
            if entry["tag"] in owned:
                print(f"  duplicate tag {entry['tag']}: shards are not disjoint",
                      file=sys.stderr)
                return 1
            owned[entry["tag"]] = (blob["lr"][at:at + n], blob["hr"][at:at + n], entry)
            at += n
        if at != blob["hr"].shape[0]:
            print(f"  {d}: index accounts for {at} patches, tensor holds "
                  f"{blob['hr'].shape[0]}", file=sys.stderr)
            return 1

    # Emit in original manifest clip order, and within a clip by sequence index,
    # which is exactly the order a single sequential run would have produced.
    lr_parts, hr_parts, index = [], [], []
    for cid in order:
        for tag in sorted(t for t in owned if t.rsplit("__", 1)[0] == cid):
            lr, hr, entry = owned[tag]
            lr_parts.append(lr)
            hr_parts.append(hr)
            index.append(entry)

    if len(index) != len(owned):
        print(f"  {len(owned) - len(index)} sequences did not match a manifest clip",
              file=sys.stderr)
        return 1
    if not hr_parts:
        print("  no patches to merge", file=sys.stderr)
        return 1

    os.makedirs(out_dir, exist_ok=True)
    lr_all, hr_all = torch.cat(lr_parts), torch.cat(hr_parts)
    torch.save({"lr": lr_all, "hr": hr_all}, os.path.join(out_dir, "all.pt"))

    meta = dict(meta_seed)
    meta.update({
        "index": index,
        "evidence": evidence,
        "failures": failures,
        "clipsPrepared": len({e["clip"] for e in index}),
        "sequences": len(index),
        "patches": int(hr_all.shape[0]),
        "assembledFrom": "disjoint manifest shards, merged in manifest order",
    })
    with open(os.path.join(out_dir, "pairs.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)

    print(f"  {meta['clipsPrepared']} clips, {len(index)} sequences, "
          f"{hr_all.shape[0]} patches -> {out_dir}", file=sys.stderr)
    if failures:
        print(f"  {len(failures)} clips failed extraction", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--manifest", required=True)
    s.add_argument("--shards", type=int, default=5)
    s.add_argument("--out", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--manifest", required=True)
    m.add_argument("--shard-dirs", nargs="+", required=True)
    m.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.cmd == "split":
        return split(args.manifest, args.shards, args.out)
    return merge(args.manifest, args.shard_dirs, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
