#!/usr/bin/env python3
"""Source-level dataset splitting for AetherVSR.

The split unit is the **source image**, never the extracted patch.

Milestone 4 split at the patch level: it drew several patches from every
photograph, pooled them, and shuffled the pool. Replaying that split on the
shipped 495-image corpus shows what it produced:

    images contributing to BOTH train and val   231  (46.7% of the corpus)
    val patches whose source also appears in train   297/297  (100.00%)
    val patches from a genuinely unseen photograph   0

Every validation patch came from a photograph the model had trained on, at a
different crop. That measures memorisation of specific images, not
generalisation, so no score from it is evidence about unseen content.

Assignment here is a deterministic function of *content identity* - the
SHA-256 the corpus manifest already records for each file - and not of list
order, filename, or an RNG stream. That means the split survives:

  * a different filesystem or directory ordering
  * repeated runs on any machine
  * changes to patch extraction, patch size or patch count
  * a file being renamed

and it means adding images to the corpus never reshuffles existing
assignments: an image's bucket depends only on its own bytes and the salt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

from PIL import Image
from dataclasses import dataclass

# Bumping this reshuffles every assignment, so it is recorded in the split
# manifest and must never be tuned to move a metric.
DEFAULT_SALT = "aethervsr/split/v2-clustered"

# Two photographs of the same scene are not independent samples even though
# their bytes differ and their content hashes disagree. Splitting per image left
# 36 near-duplicate pairs straddling splits on the 495-image corpus - several at
# dHash distance 2-5, with adjacent filenames and *identical Commons titles*:
# the same manuscript folio scanned twice, the same demonstration photographed
# seconds apart. Inspected visually and confirmed as genuine duplicates.
#
# Images are therefore clustered first, by perceptual similarity and by shared
# source title, and whole clusters are assigned - so a scene lands entirely on
# one side of the split.
DEFAULT_PHASH_THRESHOLD = 10

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitRatios:
    train: float
    val: float
    test: float

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-9:
            raise SystemExit(f"split ratios must sum to 1.0, got {total}")
        for name in SPLITS:
            if getattr(self, name) <= 0:
                raise SystemExit(f"split ratio {name} must be positive")


# 70/15/15 on a 495-image corpus gives roughly 346/74/75.
#
# The reasoning, since the milestone asks for it: the binding constraint here is
# *estimate stability*, not training data. AetherSR C16D2 has 6,291 parameters,
# so 346 photographs at several patches each is not the limiting factor on what
# it can learn - but 50 images would give a validation score noisy enough to
# make checkpoint selection arbitrary. Spending 15% on each held-out split buys
# tighter error bars where they are actually needed. The test split is a second
# held-out set from the *same* corpus and distribution; the independently
# sourced corpus in `data/eval-independent/` is the one that tests whether
# results survive a change of source, and neither substitutes for the other.
DEFAULT_RATIOS = SplitRatios(train=0.70, val=0.15, test=0.15)


def dhash(image, size: int = 8) -> int:
    """64-bit difference hash; survives rescaling and re-encoding."""
    small = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = list(small.getdata())
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | (1 if px[base + col] > px[base + col + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _title_key(entry: dict) -> str:
    """Normalised title, used as a second duplicate signal.

    Two scans of one folio can exceed the perceptual threshold while carrying
    the same Commons title, so title catches what dHash misses and vice versa.
    Trailing plate/page/detail markers are stripped so "... (1)" and
    "... detail" group with their parent.
    """
    title = (entry.get("title") or "").strip().lower()
    title = re.sub(r"^file:", "", title)
    title = re.sub(r"\.(jpe?g|png|tiff?)$", "", title)
    title = re.sub(r"[\s_-]*(\(\d+\)|\bdetail\b|\bplate\b\s*\d*|\bfolio\b\s*\S*|\bp?g?\.?\s*\d+)\s*$", "", title)
    return re.sub(r"[^a-z0-9]+", " ", title).strip()


def cluster_sources(
    entries: list[dict], corpus: str, threshold: int
) -> tuple[dict[str, str], dict[str, int], int]:
    """Union-find over perceptual similarity and shared title.

    Returns identity -> cluster key, the perceptual hashes, and the cluster
    count. The cluster key is the smallest content hash in the group, so it does
    not depend on iteration order. Transitivity is deliberate: if A~B and B~C
    then all three group, even when A and C are further apart than the
    threshold. That errs toward over-grouping, which costs a little training
    data and buys a split that cannot leak a scene.
    """
    hashes: dict[str, int] = {}
    titles: dict[str, str] = {}
    for entry in entries:
        identity = source_identity(entry)
        path = os.path.join(corpus, entry["file"])
        titles[identity] = _title_key(entry)
        if not os.path.exists(path):
            continue
        try:
            with Image.open(path) as im:
                hashes[identity] = dhash(im)
        except Exception:  # noqa: BLE001
            continue

    ids = [source_identity(e) for e in entries]
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_title: dict[str, list[str]] = {}
    for i in ids:
        key = titles.get(i, "")
        if key:
            by_title.setdefault(key, []).append(i)
    for group in by_title.values():
        for other in group[1:]:
            union(group[0], other)

    hashed = [i for i in ids if i in hashes]
    for a in range(len(hashed)):
        for b in range(a + 1, len(hashed)):
            if hamming(hashes[hashed[a]], hashes[hashed[b]]) <= threshold:
                union(hashed[a], hashed[b])

    mapping = {i: find(i) for i in ids}
    return mapping, hashes, len(set(mapping.values()))


def unit_interval(identity: str, salt: str) -> float:
    """Maps a source identity to a stable point in [0, 1).

    Hashing the identity with a salt, rather than using an RNG seeded once and
    consumed in list order, is what makes the assignment independent of
    ordering and stable under corpus growth.
    """
    digest = hashlib.sha256(f"{salt}:{identity}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / (1 << 64)


def assign(identity: str, salt: str, ratios: SplitRatios) -> str:
    u = unit_interval(identity, salt)
    if u < ratios.train:
        return "train"
    if u < ratios.train + ratios.val:
        return "val"
    return "test"


def source_identity(entry: dict) -> str:
    """The content hash, which is what actually identifies a photograph.

    Filenames are assigned by the fetcher and source URLs can differ for the
    same image (thumbnail widths, CDN hosts), so neither is a safe identity.
    """
    sha = (entry.get("sha256") or "").strip().lower()
    if not sha:
        raise SystemExit(f"manifest entry {entry.get('file')!r} has no sha256")
    return sha


def build_split(
    manifest: dict,
    salt: str = DEFAULT_SALT,
    ratios: SplitRatios = DEFAULT_RATIOS,
    corpus: str = "data/corpus",
    threshold: int = DEFAULT_PHASH_THRESHOLD,
) -> dict:
    ratios.validate()
    cluster_of, _hashes, n_clusters = cluster_sources(manifest["images"], corpus, threshold)
    buckets: dict[str, list[dict]] = {name: [] for name in SPLITS}
    seen: dict[str, str] = {}

    for entry in manifest["images"]:
        identity = source_identity(entry)
        # A corpus containing the same photograph twice under two filenames
        # would put identical content in two buckets. Content-addressed
        # assignment makes that impossible - both copies hash to the same
        # bucket - but the duplicate is still worth reporting.
        if identity in seen:
            print(f"  duplicate content: {entry['file']} == {seen[identity]}", file=sys.stderr)
            continue
        seen[identity] = entry["file"]
        # Assign the *cluster*, not the image, so near-duplicates cannot land
        # on opposite sides of the split.
        buckets[assign(cluster_of[identity], salt, ratios)].append(
            {
                "file": entry["file"],
                "sha256": identity,
                "source_url": entry.get("source_url"),
                "licence": entry.get("licence"),
            }
        )

    for name in SPLITS:
        buckets[name].sort(key=lambda e: e["sha256"])

    return {
        "schema": "aethervsr.split/1",
        "salt": salt,
        "unit": "perceptual cluster of source images (dHash <= threshold, or shared normalised title)",
        "clusters": n_clusters,
        "phashThreshold": threshold,
        "ratios": {"train": ratios.train, "val": ratios.val, "test": ratios.test},
        "counts": {name: len(buckets[name]) for name in SPLITS},
        "total": sum(len(buckets[name]) for name in SPLITS),
        "policy": (
            "Validation selects checkpoints and tunes hyperparameters. Test is "
            "evaluated once on a frozen model and never informs any choice. See "
            "DECISIONS.md ADR-0028."
        ),
        "splits": buckets,
    }


def load_split(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def split_files(split: dict, name: str) -> list[str]:
    if name not in SPLITS:
        raise SystemExit(f"unknown split {name!r}; expected one of {SPLITS}")
    return [e["file"] for e in split["splits"][name]]


def split_hashes(split: dict, name: str) -> set[str]:
    return {e["sha256"] for e in split["splits"][name]}


def check_disjoint(split: dict) -> list[str]:
    """Returns a list of problems; empty means the splits are disjoint."""
    problems: list[str] = []
    hashes = {name: split_hashes(split, name) for name in SPLITS}
    files = {name: set(split_files(split, name)) for name in SPLITS}
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for a, b in pairs:
        shared_h = hashes[a] & hashes[b]
        if shared_h:
            problems.append(f"{a} and {b} share {len(shared_h)} content hashes: {sorted(shared_h)[:3]}")
        shared_f = files[a] & files[b]
        if shared_f:
            problems.append(f"{a} and {b} share {len(shared_f)} filenames: {sorted(shared_f)[:3]}")
    counted = sum(len(files[name]) for name in SPLITS)
    if counted != split["total"]:
        problems.append(f"counts do not add up: {counted} files against total {split['total']}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a source-level dataset split.")
    ap.add_argument("--manifest", default="data/corpus/manifest.json")
    ap.add_argument("--out", default="data/splits/corpus-v2.json")
    ap.add_argument("--salt", default=DEFAULT_SALT)
    ap.add_argument("--train", type=float, default=DEFAULT_RATIOS.train)
    ap.add_argument("--val", type=float, default=DEFAULT_RATIOS.val)
    ap.add_argument("--test", type=float, default=DEFAULT_RATIOS.test)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--threshold", type=int, default=DEFAULT_PHASH_THRESHOLD)
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)

    split = build_split(
        manifest, args.salt, SplitRatios(args.train, args.val, args.test), args.corpus, args.threshold
    )
    problems = check_disjoint(split)
    if problems:
        for p in problems:
            print(f"ERROR {p}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=1)

    c = split["counts"]
    print(f"wrote {args.out}")
    print(f"  train {c['train']}  val {c['val']}  test {c['test']}  (total {split['total']})")
    print(f"  {split['clusters']} perceptual clusters from {split['total']} images")
    print("  disjoint at content-hash, filename and perceptual-cluster level: yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
