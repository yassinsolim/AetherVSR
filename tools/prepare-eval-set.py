#!/usr/bin/env python3
"""Stage a small natural-image evaluation set the dev server can fetch.

The deterministic generated reference in `quality.ts` is excellent for
regression - it is identical on every machine and stresses frequencies right up
to Nyquist - but it is synthetic, and a model that scores well on a zone plate
has not been shown to work on photographs.

This copies a fixed, seed-selected slice of the CC0 training corpus into
`public/eval/`, cropped to an even size, and writes a manifest carrying each
image's licence and source. The images are gitignored; the manifest is not.

**These images come from the training corpus.** They are held out only in the
sense that the evaluation crops differ from the training crops. That is stated
in the manifest and in BENCHMARKS.md rather than glossed: a genuinely
independent test set would need a second corpus, and claiming one we do not
have would be worse than naming the limitation.
"""

from __future__ import annotations

import argparse
import json
import os
import random

from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out", default="public/eval")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    with open(os.path.join(args.corpus, "manifest.json")) as fh:
        corpus = json.load(fh)
    os.makedirs(args.out, exist_ok=True)

    rng = random.Random(args.seed)
    pool = [e for e in corpus["images"] if min(e["width"], e["height"]) >= args.size]
    rng.shuffle(pool)

    entries = []
    for entry in pool:
        if len(entries) >= args.count:
            break
        src = os.path.join(args.corpus, entry["file"])
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                w, h = im.size
                # Centre crop to an exact square, so the 2x downsample is exact
                # and no scaler is advantaged by an odd dimension.
                left = (w - args.size) // 2
                top = (h - args.size) // 2
                crop = im.crop((left, top, left + args.size, top + args.size))
                name = f"eval{len(entries):02d}.png"
                crop.save(os.path.join(args.out, name), "PNG")
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {entry['file']}: {type(exc).__name__} {exc}")
            continue
        entries.append(
            {
                "file": name,
                "size": args.size,
                "licence": entry["licence"],
                "source_url": entry["source_url"],
                "title": entry["title"],
                "corpus_sha256": entry["sha256"],
            }
        )

    manifest = {
        "purpose": "natural-image evaluation set for 2x super-resolution",
        "count": len(entries),
        "size": args.size,
        "seed": args.seed,
        "provenance": (
            "Centre crops of images drawn from the CC0 training corpus. Not an "
            "independent test set: the same photographs were seen during training, "
            "though at different crops. Reported alongside the deterministic "
            "generated reference, never instead of it."
        ),
        "images": entries,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"staged {len(entries)} images -> {args.out}")
    return 0 if entries else 1


if __name__ == "__main__":
    raise SystemExit(main())
