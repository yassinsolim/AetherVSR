#!/usr/bin/env python3
"""Split a corpus manifest for parallel preparation, then merge the results.

Preparation streams clips over HTTP into independent output directories.
Per-clip RNG seeds do not depend on shard placement, and merge restores manifest
order. This does not guarantee byte-identical rebuilds: extraction failures can
change which sequences and CRF draws are retained. Pin the final cache and reuse
that one cache across every experimental arm.

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
    canonical = set(order)
    if not canonical or len(canonical) != len(order):
        print("  manifest must contain nonempty, unique clip IDs", file=sys.stderr)
        return 1
    if os.path.exists(out_dir):
        print(f"  output already exists: {out_dir}; use a fresh destination", file=sys.stderr)
        return 1

    # tag -> (lr, hr, index entry). Tags are unique across shards because each
    # clip appears in exactly one shard.
    owned: dict[str, tuple[torch.Tensor, torch.Tensor, dict]] = {}
    evidence: list[dict] = []
    failures: list[dict] = []
    meta_seed: dict = {}
    clip_owners: set[str] = set()

    for d in shard_dirs:
        meta_path = os.path.join(d, "pairs.json")
        pt_path = os.path.join(d, "all.pt")
        if not (os.path.exists(meta_path) and os.path.exists(pt_path)):
            print(f"  missing shard output in {d}", file=sys.stderr)
            return 1
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        shard_clips = {entry["clip"] for entry in meta["index"]}
        if shard_clips - canonical or shard_clips & clip_owners:
            print(f"  {d}: unknown or multiply owned clip IDs", file=sys.stderr)
            return 1
        clip_owners.update(shard_clips)
        blob = torch.load(pt_path, weights_only=True)
        if meta_seed:
            protocol_keys = ("schema", "role", "structure", "x264params", "preset",
                             "crfDistribution", "hrSize", "lrSize", "patchHr", "patchLr",
                             "sequencesPerClip", "sequenceFrames", "seed")
            if any(meta.get(key) != meta_seed.get(key) for key in protocol_keys):
                print(f"  {d}: preparation protocol differs from earlier shards", file=sys.stderr)
                return 1
        meta_seed = meta_seed or meta
        evidence.extend(meta["encodeEvidence"])
        # corpus-prepare.py names this key clipsFailed. Reading "failures" here
        # silently dropped every failure record, so a merge that lost 15 of 151
        # clips still reported zero failures - the one number a reader would
        # check to decide whether the corpus was whole.
        failures.extend(meta["clipsFailed"])

        at = 0
        for entry in meta["index"]:
            if entry["tag"].rsplit("__", 1)[0] != entry["clip"]:
                print(f"  {d}: sequence tag does not match clip ID", file=sys.stderr)
                return 1
            n = entry["patches"]
            if not isinstance(n, int) or n <= 0:
                print(f"  {d}: invalid patch count", file=sys.stderr)
                return 1
            if entry["tag"] in owned:
                print(f"  duplicate tag {entry['tag']}: shards are not disjoint",
                      file=sys.stderr)
                return 1
            owned[entry["tag"]] = (blob["lr"][at:at + n], blob["hr"][at:at + n], entry)
            at += n
        if at != blob["hr"].shape[0] or at != blob["lr"].shape[0]:
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

    # Reject partial coverage before creating either output file.
    prepared = {entry["clip"] for entry in index}
    failed = {entry.get("clip") for entry in failures}
    missing = canonical - prepared
    if failed - canonical:
        print("  failure records contain unknown clip IDs", file=sys.stderr)
        return 1
    if missing or failures:
        print(f"  incomplete corpus: {len(prepared)}/{len(canonical)} clips; "
              f"{len(failures)} extraction failures", file=sys.stderr)
        for cid in sorted(missing):
            print(f"    missing {cid}", file=sys.stderr)
        print("  nothing written; recover missing clips before merging", file=sys.stderr)
        return 1

    expected_count = meta_seed["sequencesPerClip"]
    if not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count <= 0:
        print("  invalid sequencesPerClip", file=sys.stderr)
        return 1
    tags_by_clip = {cid: set() for cid in canonical}
    for entry in index:
        tags_by_clip[entry["clip"]].add(entry["tag"])
    partial = {cid: len(tags) for cid, tags in tags_by_clip.items()
               if tags != {f"{cid}__{seq:02d}" for seq in range(expected_count)}}
    if partial:
        for cid, count in sorted(partial.items()):
            print(f"  incomplete clip {cid}: {count}/{expected_count} sequence slots", file=sys.stderr)
        print("  nothing written", file=sys.stderr)
        return 1

    encoded = {entry["tag"]: entry for entry in evidence}
    if len(encoded) != len(evidence) or set(encoded) != set(owned):
        print("  encode evidence must cover each sequence exactly once", file=sys.stderr)
        return 1

    os.makedirs(out_dir, exist_ok=True)
    lr_all, hr_all = torch.cat(lr_parts), torch.cat(hr_parts)
    torch.save({"lr": lr_all, "hr": hr_all}, os.path.join(out_dir, "all.pt"))

    meta = dict(meta_seed)
    meta.update({
        "index": index,
        "encodeEvidence": [encoded[entry["tag"]] for entry in index],
        "clipsFailed": failures,
        "clipsPrepared": len({e["clip"] for e in index}),
        "sequences": len(index),
        "patches": int(hr_all.shape[0]),
        "sourceManifest": manifest_path,
        "shardSources": [os.path.join(d, "pairs.json") for d in shard_dirs],
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
