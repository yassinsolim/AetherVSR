#!/usr/bin/env python3
"""Turn discovery pools into three role-separated corpora.

The split order matters and is fixed by the schema's provenance hierarchy:

    clips -> shoots -> creators -> role assignment

Assigning clips first and checking for leaks afterwards is how Milestone 5.5
ended up with two rolls from one afternoon on opposite sides of its
train/validation boundary. Here whole creators are assigned, so the leak is
impossible by construction rather than caught by a test.

Roles are carved in order of how expensive a mistake is:

  confirmation  strictest. Creator-disjoint from training, no exceptions. It
                exists to answer whether the model generalises to footage from
                people it never learned from, and a shared creator quietly
                destroys that.
  validation    creator-disjoint where the corpus can afford it, shoot-disjoint
                at minimum. Whichever holds is recorded, not assumed.
  training      everything left.

Category balance is applied *within* that constraint, never across it: a corpus
that hits its category targets by borrowing a creator from the confirmation set
has traded the only property that makes the confirmation set worth having.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_schema import (  # noqa: E402
    CATEGORIES, SCHEMA, corpus_digest, derive_shoot_id, licence_ok, normalise_creator,
)

# What each role is for, in clips. Confirmation is deliberately small and
# expensive: every clip in it costs a creator that training cannot use.
MAX_CLIPS_PER_CREATOR = 3

TARGETS = {
    "confirmation": {"total": 20, "perCategory": 2},
    "validation": {"total": 16, "perCategory": 2},
}


def slugify(title: str) -> str:
    s = re.sub(r"^File:", "", title)
    s = re.sub(r"\.(webm|mp4|mov|ogv)$", "", s, flags=re.I)
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s[:48] or "clip"


def load_pools(pattern: str) -> list[dict]:
    cands: list[dict] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as fh:
            pool = json.load(fh)
        for c in pool.get("candidates", []):
            c["_pool"] = os.path.basename(path)
            cands.append(c)
    return cands


def dedupe(cands: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drop anything two pools found independently, or that repeats a source.

    Keyed on several identifiers because the pools searched overlapping spaces:
    the same file can arrive under a different title casing, and a re-upload can
    share a URL while differing in title.
    """
    seen: dict[str, dict] = {}
    kept, dropped = [], []
    for c in cands:
        keys = [str(c.get("pageid") or ""), (c.get("source_url") or "").lower(),
                (c.get("title") or "").strip().casefold()]
        keys = [k for k in keys if k]
        hit = next((seen[k] for k in keys if k in seen), None)
        if hit is not None:
            dropped.append({"title": c.get("title"), "reason": "duplicate",
                            "keptFrom": hit.get("_pool"), "droppedFrom": c.get("_pool")})
            continue
        for k in keys:
            seen[k] = c
        kept.append(c)
    return kept, dropped


def gate(cands: list[dict]) -> tuple[list[dict], list[dict]]:
    kept, rejected = [], []
    for c in cands:
        why = None
        if not licence_ok(c.get("licence")):
            why = f"licence {c.get('licence')!r}"
        elif (c.get("width") or 0) < 2560 or (c.get("height") or 0) < 1440:
            why = f"{c.get('width')}x{c.get('height')} below floor"
        elif (c.get("height") or 0) > (c.get("width") or 0):
            why = "portrait source"
        elif c.get("category") not in CATEGORIES:
            why = f"category {c.get('category')!r}"
        elif not c.get("creator"):
            why = "no creator"
        if why:
            rejected.append({"title": c.get("title"), "reason": why})
        else:
            kept.append(c)
    return kept, rejected


def assign(cands: list[dict], seed: int) -> dict[str, list[dict]]:
    """Assign whole creators to roles, smallest role first."""
    import random
    rng = random.Random(seed)

    by_creator: dict[str, list[dict]] = collections.defaultdict(list)
    for c in cands:
        by_creator[normalise_creator(c.get("creator"))].append(c)

    # A creator's usefulness to a small role is the categories they can fill.
    # Sorting by that, then shuffling within ties, keeps the choice deterministic
    # without always handing the same creators to the same role.
    creators = sorted(by_creator, key=lambda k: (-len({x["category"] for x in by_creator[k]}),
                                                 -len(by_creator[k]), k))
    rng.shuffle(creators)

    roles: dict[str, list[dict]] = {"confirmation": [], "validation": [], "training": []}
    taken: set[str] = set()

    # How many clips each small role may take from a category. Without this the
    # greedy pass hands the whole of a scarce category to validation and
    # confirmation - on the first run it left training with zero texture and
    # zero text, which is precisely backwards, since training is the largest
    # consumer and the reason the category was sought.
    available = collections.Counter(c["category"] for c in cands)
    def budget(role: str, category: str) -> int:
        per_role_share = 0.15  # leaves ~70% of any category to training
        cap = TARGETS[role]["perCategory"]
        return max(1, min(cap, round(per_role_share * available[category])))

    for role in ("confirmation", "validation"):
        target = TARGETS[role]
        per_cat: collections.Counter = collections.Counter()
        for creator in creators:
            if creator in taken or len(roles[role]) >= target["total"]:
                continue
            clips = by_creator[creator]
            # Only take a creator if they add a category the role still needs;
            # otherwise they are worth more to training.
            useful = [c for c in clips if per_cat[c["category"]] < budget(role, c["category"])]
            if not useful:
                continue
            for c in useful:
                if len(roles[role]) >= target["total"]:
                    break
                roles[role].append(c)
                per_cat[c["category"]] += 1
            taken.add(creator)

    capped: list[dict] = []
    for creator in creators:
        if creator in taken:
            continue
        # Cap what any one creator contributes to training. One operator means
        # one camera, one encoder and one set of habits, so the fifth clip from
        # them teaches far less than the first from someone new - and a quarter
        # of the v1 corpus sat with a single creator.
        clips = sorted(by_creator[creator], key=lambda x: (x["category"], x.get("title", "")))
        roles["training"] += clips[:MAX_CLIPS_PER_CREATOR]
        capped += clips[MAX_CLIPS_PER_CREATOR:]
    roles["_capped"] = capped
    return roles


def build(role: str, clips: list[dict], notes: dict) -> dict:
    out = []
    for c in sorted(clips, key=lambda x: (x["category"], x.get("title", ""))):
        cid = f"{c['category']}-{slugify(c.get('title',''))}"[:56]
        entry = {
            "id": cid,
            "title": c.get("title"),
            "category": c["category"],
            "creator": c.get("creator"),
            "shootId": c.get("shootId") or derive_shoot_id({"title": c.get("title"),
                                                            "creator": c.get("creator"),
                                                            "id": cid}),
            "source_url": c.get("source_url"),
            "description_url": c.get("description_url"),
            "pageid": c.get("pageid"),
            "licence": c.get("licence"),
            "source_width": c.get("width"),
            "source_height": c.get("height"),
            "source_bytes": c.get("bytes"),
            "duration": c.get("duration"),
            "fps": c.get("fps"),
            "codec": c.get("codec"),
            "hasBurnedInGraphics": bool(c.get("hasBurnedInGraphics")),
            "contentNote": c.get("contentNote"),
            "source_file": f"{cid}.webm",
            # Filled by the fetcher against the bytes actually received.
            "source_sha256": None,
        }
        out.append(entry)
    return {
        "schema": SCHEMA,
        "role": role,
        "clips": out,
        "clipCount": len(out),
        **notes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Assemble role-separated corpora from discovery pools.")
    ap.add_argument("--pools", default="/tmp/m6pool/Pool*.json")
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--out-train", default="data/captured-train-v2/manifest.json")
    ap.add_argument("--out-val", default="data/captured-val/manifest.json")
    ap.add_argument("--out-confirm", default="data/captured-confirm/manifest.json")
    ap.add_argument("--report", default="results/corpus-assembly.json")
    args = ap.parse_args()

    raw = load_pools(args.pools)
    deduped, dups = dedupe(raw)
    gated, rejected = gate(deduped)
    for c in gated:
        c["shootId"] = derive_shoot_id({"title": c.get("title"), "creator": c.get("creator")})
    roles = assign(gated, args.seed)

    # Creator disjointness is a property to verify, never to assume.
    def creators(rs: list[dict]) -> set[str]:
        return {normalise_creator(c.get("creator")) for c in rs}
    tr, va, co = creators(roles["training"]), creators(roles["validation"]), creators(roles["confirmation"])
    checks = {
        "confirmationCreatorDisjointFromTraining": not (co & tr),
        "validationCreatorDisjointFromTraining": not (va & tr),
        "confirmationCreatorDisjointFromValidation": not (co & va),
        "sharedTrainVal": sorted(va & tr),
        "sharedTrainConfirm": sorted(co & tr),
    }
    if not checks["confirmationCreatorDisjointFromTraining"]:
        raise SystemExit(f"confirmation shares creators with training: {sorted(co & tr)}")

    written = {}
    roles.pop("_capped", None) if False else None
    for role, path in (("training", args.out_train), ("validation", args.out_val),
                       ("confirmation", args.out_confirm)):
        notes = {
            "splitPolicy": ("Whole creators are assigned to exactly one role before any clip is "
                            "chosen, so a shoot or creator cannot span roles by construction."),
            "creatorDisjointFromTraining": role == "training" or checks[
                f"{role}CreatorDisjointFromTraining"],
            "assemblySeed": args.seed,
        }
        manifest = build(role, roles[role], notes)
        manifest["contentDigestPending"] = True  # hashes land at fetch time
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1)
        written[role] = {"path": path, "clips": len(manifest["clips"]),
                         "creators": len(creators(roles[role])),
                         "shoots": len({c["shootId"] for c in roles[role]}),
                         "categories": dict(collections.Counter(c["category"] for c in roles[role]))}
        print(f"  {role:<13} {len(manifest['clips']):>3} clips  "
              f"{len(creators(roles[role])):>2} creators  "
              f"{len({c['shootId'] for c in roles[role]}):>2} shoots  "
              f"{written[role]['categories']}", file=sys.stderr)

    report = {
        "schema": "aethervsr.corpus-assembly/1",
        "poolCandidates": len(raw),
        "afterDedupe": len(deduped),
        "afterGate": len(gated),
        "duplicatesDropped": dups,
        "rejected": rejected,
        "roles": written,
        "cappedByCreatorLimit": [
            {"title": c.get("title"), "creator": c.get("creator"), "category": c.get("category")}
            for c in roles.get("_capped", [])
        ],
        "maxClipsPerCreator": MAX_CLIPS_PER_CREATOR,
        "disjointnessChecks": checks,
    }
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"\n  pool {len(raw)} -> dedupe {len(deduped)} -> gate {len(gated)}", file=sys.stderr)
    print(f"  rejected {len(rejected)}, duplicates {len(dups)}", file=sys.stderr)
    print(f"  creator-disjoint: confirm/train {checks['confirmationCreatorDisjointFromTraining']}, "
          f"val/train {checks['validationCreatorDisjointFromTraining']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
