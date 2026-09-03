#!/usr/bin/env python3
"""Fetch an independently sourced CC0 evaluation corpus.

The training corpus comes from Wikimedia Commons' CC-Zero category. Evaluating
on more images from that same category would test a different sample, not a
different source: the same uploaders, the same curation, the same re-encoding
pipeline, often the same photographers.

This fetches from the **Metropolitan Museum of Art Open Access** collection
instead - a different institution, a different imaging pipeline, and images
released CC0 by the museum itself. That makes an overlap with the training
corpus structurally unlikely, and this script then proves it rather than
assuming it.

Three independent overlap checks, all of which must pass for an image to be
accepted:

  1. exact content   SHA-256 of the downloaded bytes against every training hash
  2. source identity source URL against every training URL
  3. perceptual      64-bit difference hash against every training image, which
                     catches the same photograph resized, re-encoded or
                     lightly recoloured - the case a byte hash misses entirely

The result is a test set. Nothing here trains a model, selects an epoch, picks
a seed or tunes a hyperparameter.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image

API = "https://collectionapi.metmuseum.org/public/collection/v1"
UA = "AetherVSR/0.1 (https://github.com/yassinsolim/AetherVSR) independent-eval-fetch"

# The Met's own field. CC0 is asserted per object by the museum; we record the
# object page so the claim is checkable rather than taken on trust.
LICENCE = "CC0"


def api_get(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_bytes(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def dhash(image: Image.Image, size: int = 8) -> int:
    """64-bit difference hash.

    Grayscale, resize to (size+1) x size, then compare horizontally adjacent
    pixels. Robust to rescaling and re-encoding, which is exactly the duplicate
    a SHA-256 comparison cannot see.
    """
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


def training_fingerprints(corpus_dir: str, manifest_path: str) -> tuple[set[str], set[str], list[tuple[int, str]]]:
    """Exact hashes, source URLs and perceptual hashes of the training corpus."""
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    hashes = {(e.get("sha256") or "").lower() for e in manifest["images"]}
    urls = {(e.get("source_url") or "").split("?")[0] for e in manifest["images"]}
    perceptual: list[tuple[int, str]] = []
    missing = 0
    for entry in manifest["images"]:
        path = os.path.join(corpus_dir, entry["file"])
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            with Image.open(path) as im:
                perceptual.append((dhash(im), entry["file"]))
        except Exception:  # noqa: BLE001
            missing += 1
    if missing:
        print(f"  note: {missing} training images unreadable for perceptual hashing", file=sys.stderr)
    return hashes, urls, perceptual


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch an independent CC0 evaluation corpus.")
    ap.add_argument("--out", default="data/eval-independent")
    ap.add_argument("--manifest", default="data/eval-independent/manifest.json")
    ap.add_argument("--train-corpus", default="data/corpus")
    ap.add_argument("--train-manifest", default="data/corpus/manifest.json")
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--min-side", type=int, default=1024)
    ap.add_argument("--hamming-threshold", type=int, default=10, help="dhash distance below which images are treated as duplicates")
    ap.add_argument("--search", default="painting|photograph|textile|landscape|portrait|ceramic")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("fingerprinting the training corpus…")
    train_hashes, train_urls, train_perceptual = training_fingerprints(args.train_corpus, args.train_manifest)
    print(f"  {len(train_hashes)} content hashes, {len(train_urls)} source URLs, {len(train_perceptual)} perceptual hashes")

    candidates: list[int] = []
    for term in args.search.split("|"):
        try:
            q = urllib.parse.urlencode({"q": term, "hasImages": "true", "isPublicDomain": "true"})
            res = api_get(f"{API}/search?{q}")
        except Exception as exc:  # noqa: BLE001
            print(f"  search {term!r} failed: {exc}", file=sys.stderr)
            continue
        ids = res.get("objectIDs") or []
        # Stride rather than take the head: consecutive object IDs are often the
        # same acquisition photographed in one session, which would make the set
        # far less varied than its count suggests.
        candidates.extend(ids[:: max(1, len(ids) // 400)][:400])
        time.sleep(0.2)

    seen_ids: set[int] = set()
    ordered = [i for i in candidates if not (i in seen_ids or seen_ids.add(i))]
    print(f"candidate objects: {len(ordered)}")

    accepted: list[dict] = []
    rejected = {"licence": 0, "small": 0, "error": 0, "exact_dup": 0, "url_dup": 0, "perceptual_dup": 0}
    kept_perceptual: list[int] = []

    for oid in ordered:
        if len(accepted) >= args.count:
            break
        try:
            obj = api_get(f"{API}/objects/{oid}")
        except Exception:  # noqa: BLE001
            rejected["error"] += 1
            continue

        if not obj.get("isPublicDomain"):
            rejected["licence"] += 1
            continue
        url = obj.get("primaryImage") or ""
        if not url:
            rejected["error"] += 1
            continue

        if url.split("?")[0] in train_urls:
            rejected["url_dup"] += 1
            continue

        try:
            payload = fetch_bytes(url)
        except Exception:  # noqa: BLE001
            rejected["error"] += 1
            continue

        sha = hashlib.sha256(payload).hexdigest()
        if sha in train_hashes:
            rejected["exact_dup"] += 1
            continue

        try:
            with Image.open(io.BytesIO(payload)) as im:
                im.load()
                w, h = im.size
                if min(w, h) < args.min_side:
                    rejected["small"] += 1
                    continue
                ph = dhash(im)
        except Exception:  # noqa: BLE001
            rejected["error"] += 1
            continue

        near = next((f for hsh, f in train_perceptual if hamming(ph, hsh) <= args.hamming_threshold), None)
        if near is not None:
            rejected["perceptual_dup"] += 1
            print(f"  perceptual duplicate of training image {near}: object {oid}")
            continue
        # Also guard against duplicates *within* the new set, which the Met has
        # plenty of - the same object photographed from two angles.
        if any(hamming(ph, k) <= args.hamming_threshold for k in kept_perceptual):
            rejected["perceptual_dup"] += 1
            continue
        kept_perceptual.append(ph)

        name = f"ind{len(accepted):04d}_{sha[:10]}.jpg"
        with open(os.path.join(args.out, name), "wb") as fh:
            fh.write(payload)
        accepted.append(
            {
                "file": name,
                "sha256": sha,
                "bytes": len(payload),
                "width": w,
                "height": h,
                "licence": LICENCE,
                "source_url": url,
                "object_url": obj.get("objectURL"),
                "object_id": oid,
                "title": (obj.get("title") or "")[:200],
                "credit": (obj.get("creditLine") or "")[:200],
                "dhash": f"{ph:016x}",
            }
        )
        print(f"  [{len(accepted)}/{args.count}] {name}  {w}x{h}")
        time.sleep(0.15)

    if not accepted:
        raise SystemExit("no images accepted; nothing written")

    manifest = {
        "schema": "aethervsr.eval-independent/1",
        "purpose": (
            "Held-out test corpus, independently sourced. Never used for training, "
            "checkpoint selection, seed selection or hyperparameter tuning."
        ),
        "source": "Metropolitan Museum of Art Open Access",
        "api": API,
        "licence_policy": (
            "Objects the Met marks isPublicDomain, released CC0. Independent of the "
            "Wikimedia CC-Zero category used for training."
        ),
        "independence_checks": {
            "exact_sha256_against_training": True,
            "source_url_against_training": True,
            "perceptual_dhash_against_training": True,
            "perceptual_dhash_within_set": True,
            "hamming_threshold": args.hamming_threshold,
        },
        "min_side": args.min_side,
        "count": len(accepted),
        "rejected": rejected,
        "images": accepted,
    }
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    print(f"\naccepted {len(accepted)}   rejected {rejected}")
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
