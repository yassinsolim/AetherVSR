#!/usr/bin/env python3
"""Deterministic nested subsets for the data-scaling curve.

The scaling curve needs N12 ⊂ N24 ⊂ N48 ⊂ N96. Nesting is not a nicety: if the
subsets are drawn independently, a smaller one can happen to contain easier or
better-balanced clips and the curve measures sampling luck instead of scale.

So there is exactly one ordering of the corpus and every subset is a prefix of
it. Nesting then holds by construction, and the only design question is what
order to use.

The order is built greedily to keep every prefix as balanced as the corpus
allows: at each step take the clip whose category is currently most
under-represented relative to its share of the whole corpus, breaking ties
toward creators not yet used. A prefix of this ordering is therefore a
reasonable corpus in its own right, not the first N clips of an alphabetical
list — which would have made N12 an all-`faces` corpus and the curve
meaningless.

Two further orderings are produced for the diversity experiments, both at
matched clip counts so volume is held constant:

  concentrated  as few creators and categories as possible - the shape of the
                v1 corpus, which was half rail footage from eight creators
  diverse       as many creators and categories as possible

Comparing those two answers the question a raw scaling curve cannot: at fixed
data volume, is diversity itself worth anything?
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_schema import normalise_creator  # noqa: E402


def balanced_order(clips: list[dict], seed: int) -> list[dict]:
    """Order so that every prefix is as category- and creator-balanced as possible."""
    import random
    rng = random.Random(seed)
    pool = sorted(clips, key=lambda c: (c["category"], c.get("id", "")))
    rng.shuffle(pool)

    target = collections.Counter(c["category"] for c in pool)
    total = len(pool)
    chosen: list[dict] = []
    have: collections.Counter = collections.Counter()
    used_creators: set[str] = set()

    while pool:
        def deficit(c: dict) -> tuple:
            cat = c["category"]
            # How far below its fair share this category currently sits.
            want = target[cat] / total * (len(chosen) + 1)
            short = want - have[cat]
            new_creator = normalise_creator(c.get("creator")) not in used_creators
            return (-short, not new_creator, c.get("id", ""))

        pick = min(pool, key=deficit)
        pool.remove(pick)
        chosen.append(pick)
        have[pick["category"]] += 1
        used_creators.add(normalise_creator(pick.get("creator")))
    return chosen


def concentrated_order(clips: list[dict], seed: int) -> list[dict]:
    """Deliberately narrow on both axes: few categories, and their most prolific
    creators first.

    This is the control arm for the diversity experiment, and it must resemble
    the corpus this project actually had rather than a random sample, which at
    this size would already be fairly diverse. A first version concentrated on
    creators alone and was not narrow at all: with at most three clips each, no
    one dominates, and the arm still spanned every category. v1 was narrow
    because it was half rail footage AND from eight people, so both axes matter.
    """
    import random
    rng = random.Random(seed)
    # Concentrate on BOTH axes. A creator-only version was not actually narrow:
    # with at most three clips per creator, no one contributes enough to
    # dominate, and the arm still spanned every category - so it differed from
    # the diverse arm in creators alone and understated the contrast.
    # The v1 corpus was narrow because it was half rail footage AND from eight
    # people, so the control reproduces that shape: exhaust the largest
    # categories first, taking each category's most prolific creators.
    by_cat: dict[str, list[dict]] = collections.defaultdict(list)
    for c in clips:
        by_cat[c["category"]].append(c)
    cat_order = sorted(by_cat, key=lambda k: (-len(by_cat[k]), k))
    creator_size = collections.Counter(normalise_creator(c.get("creator")) for c in clips)
    out: list[dict] = []
    for cat in cat_order:
        group = sorted(
            by_cat[cat],
            key=lambda c: (-creator_size[normalise_creator(c.get("creator"))],
                           normalise_creator(c.get("creator")), c.get("id", "")),
        )
        out += group
    rng.random()  # keep the seed consumed identically across orderings
    return out


def diverse_order(clips: list[dict], seed: int) -> list[dict]:
    """Deliberately wide: one clip per creator per round, rotating categories."""
    import random
    rng = random.Random(seed)
    by_creator: dict[str, list[dict]] = collections.defaultdict(list)
    for c in clips:
        by_creator[normalise_creator(c.get("creator"))].append(c)
    for v in by_creator.values():
        rng.shuffle(v)
    out: list[dict] = []
    round_no = 0
    while any(by_creator.values()):
        creators = sorted(k for k, v in by_creator.items() if v)
        rng.shuffle(creators)
        # Rotate categories within a round so a prefix does not fill up with one.
        seen_cat: set[str] = set()
        deferred: list[str] = []
        for k in creators:
            cat = by_creator[k][0]["category"]
            if cat in seen_cat and round_no == 0:
                deferred.append(k)
                continue
            out.append(by_creator[k].pop(0))
            seen_cat.add(cat)
        for k in deferred:
            if by_creator[k]:
                out.append(by_creator[k].pop(0))
        round_no += 1
    return out


def describe(clips: list[dict]) -> dict:
    cats = collections.Counter(c["category"] for c in clips)
    creators = collections.Counter(normalise_creator(c.get("creator")) for c in clips)
    shoots = {c.get("shootId") for c in clips}
    n = len(clips)
    hhi = sum((v / n) ** 2 for v in creators.values()) if n else 0
    return {
        "clips": n,
        "creators": len(creators),
        "shoots": len(shoots),
        "categories": dict(cats.most_common()),
        "largestCreatorShare": max(creators.values()) / n if n else 0,
        "effectiveCreators": 1 / hhi if hhi else 0,
        "durationSeconds": sum(float(c.get("duration") or 0) for c in clips),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build nested and diversity-controlled subsets.")
    ap.add_argument("--manifest", default="data/captured-train-v2/manifest.json")
    ap.add_argument("--scales", default="12,24,48,96")
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--out", default="data/captured-train-v2/subsets.json")
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    clips = manifest["clips"]
    scales = [int(s) for s in args.scales.split(",") if s.strip()]
    scales = [s for s in scales if s <= len(clips)]
    if len(clips) not in scales:
        scales.append(len(clips))

    order = balanced_order(clips, args.seed)
    subsets = {f"N{n}": [c["id"] for c in order[:n]] for n in scales}

    # Diversity arms are sized to the largest scale both orderings can supply,
    # so the comparison is at genuinely matched volume.
    fixed = min(len(clips) // 2, 48)
    conc = concentrated_order(clips, args.seed)[:fixed]
    div = diverse_order(clips, args.seed)[:fixed]

    by_id = {c["id"]: c for c in clips}
    out = {
        "schema": "aethervsr.corpus-subsets/1",
        "sourceManifest": args.manifest,
        "seed": args.seed,
        "nestingProof": "every scale is a prefix of one ordering, so N_k is a subset of N_(k+1)",
        "scales": {k: {"ids": v, **describe([by_id[i] for i in v])} for k, v in subsets.items()},
        "diversityArms": {
            "matchedClipCount": fixed,
            "concentrated": {"ids": [c["id"] for c in conc], **describe(conc)},
            "diverse": {"ids": [c["id"] for c in div], **describe(div)},
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print("  SCALING SUBSETS (each a prefix of one ordering)", file=sys.stderr)
    for k in subsets:
        d = out["scales"][k]
        print(f"    {k:<6} {d['clips']:>3} clips  {d['creators']:>2} creators  "
              f"eff {d['effectiveCreators']:>4.1f}  max share {d['largestCreatorShare']:>5.1%}  "
              f"{d['categories']}", file=sys.stderr)
    prev = None
    for k in subsets:
        cur = set(subsets[k])
        if prev is not None and not prev <= cur:
            raise SystemExit(f"nesting violated at {k}")
        prev = cur
    print("    nesting verified", file=sys.stderr)
    print(f"\n  DIVERSITY ARMS at {fixed} clips each", file=sys.stderr)
    for arm in ("concentrated", "diverse"):
        d = out["diversityArms"][arm]
        print(f"    {arm:<13} {d['creators']:>2} creators  eff {d['effectiveCreators']:>4.1f}  "
              f"max share {d['largestCreatorShare']:>5.1%}  {len(d['categories'])} categories",
              file=sys.stderr)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
