#!/usr/bin/env python3
"""Fetch a CC0-only training corpus from Wikimedia Commons.

Milestone 4 trains its own weights, which makes the training corpus a licence
decision rather than an implementation detail. The conventional
super-resolution corpora do not work here:

  DIV2K   - "made available for academic research purpose only"
  LSDIR   - "academic research purpose only"
  Flickr2K- collected via the Flickr API with no dataset-wide grant; rights
            remain image-specific

This script instead pulls from Commons' CC-Zero category and, critically,
**re-checks the licence of every individual file and refuses anything that is
not CC0**. Category membership alone is not evidence: categories are
user-maintained and occasionally wrong. The per-file `extmetadata` returned with
the image is the authority.

Nothing here is committed to the repository except the manifest. Images land in
a gitignored directory, and the manifest records URL, dimensions, SHA-256,
licence string and author for every one, so the corpus can be audited or
reconstructed without shipping it.

Usage:
    python3 tools/fetch-corpus.py --out data/corpus --count 800
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
UA = "AetherVSR/0.1 (https://github.com/yassinsolim/AetherVSR) training-corpus-fetch"

# CC0 only. A bare "public domain" tag on Commons is a claim about *some*
# jurisdiction - usually expiry of a term, sometimes a US-government work - and
# it is not a dedication the uploader made. Independent review found five such
# files in a corpus the ADR described as CC0, so the filter no longer accepts
# the generic tags it used to.
ACCEPTED_LICENCES = {"cc0", "cc0 1.0", "cc-zero"}


def api_get(params: dict) -> dict:
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as fh:
        return json.load(fh)


def licence_ok(meta: dict) -> tuple[bool, str]:
    """True only when the file's own metadata says CC0."""
    short = (meta.get("LicenseShortName", {}).get("value") or "").strip()
    normalised = short.lower().replace("\u2013", "-")
    if normalised in ACCEPTED_LICENCES:
        return True, short
    # Some CC0 files spell it only in the machine-readable field.
    machine = (meta.get("License", {}).get("value") or "").strip().lower()
    if machine in {"cc0", "cc-zero"}:
        return True, short or machine
    return False, short or machine or "(none)"


def iter_candidates(batch: int, pages: int):
    cont = None
    for _ in range(pages):
        params = {
            "action": "query",
            "format": "json",
            "generator": "categorymembers",
            "gcmtitle": "Category:CC-Zero",
            "gcmtype": "file",
            "gcmlimit": str(batch),
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata|sha1",
            "iiurlwidth": "1400",
        }
        if cont:
            params["gcmcontinue"] = cont
        data = api_get(params)
        for page in data.get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [None])[0]
            if info:
                yield page.get("title", ""), info
        cont = data.get("continue", {}).get("gcmcontinue")
        if not cont:
            return
        time.sleep(0.2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus")
    ap.add_argument("--count", type=int, default=800)
    ap.add_argument("--min-side", type=int, default=640)
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    manifest_path = os.path.join(out, "manifest.json")

    entries: list[dict] = []
    rejected = {"licence": 0, "small": 0, "error": 0, "type": 0}
    seen: set[str] = set()

    for title, info in iter_candidates(50, 200):
        if len(entries) >= args.count:
            break
        ok, licence = licence_ok(info.get("extmetadata", {}))
        if not ok:
            rejected["licence"] += 1
            continue
        if min(info.get("width", 0), info.get("height", 0)) < args.min_side:
            rejected["small"] += 1
            continue
        url = info.get("thumburl") or info.get("url")
        if not url or not url.lower().split("?")[0].endswith((".jpg", ".jpeg", ".png")):
            rejected["type"] += 1
            continue
        if url in seen:
            continue
        seen.add(url)

        name = f"{len(entries):05d}_{hashlib.sha1(title.encode()).hexdigest()[:10]}.jpg"
        dest = os.path.join(out, name)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as fh:
                blob = fh.read()
            with open(dest, "wb") as fh:
                fh.write(blob)
        except Exception as exc:  # noqa: BLE001 - a failed download is just skipped
            rejected["error"] += 1
            print(f"  skip {title[:48]}: {type(exc).__name__}", file=sys.stderr)
            continue

        meta = info.get("extmetadata", {})
        entries.append(
            {
                "file": name,
                "title": title,
                "source_url": url,
                "descriptor_page": meta.get("DescriptionUrl", {}).get("value", ""),
                "width": info.get("thumbwidth", info.get("width")),
                "height": info.get("thumbheight", info.get("height")),
                "licence": licence,
                "artist_html": meta.get("Artist", {}).get("value", ""),
                "sha256": hashlib.sha256(blob).hexdigest(),
                "bytes": len(blob),
            }
        )
        if len(entries) % 25 == 0:
            print(f"  {len(entries)}/{args.count}")

    manifest = {
        "source": "Wikimedia Commons, Category:CC-Zero",
        "api": API,
        "licence_policy": (
            "CC0 only. Generic public-domain tags are not accepted; see ADR-0026. "
            "Every entry's own imageinfo.extmetadata licence field was checked; "
            "category membership alone was not accepted."
        ),
        "accepted": sorted(ACCEPTED_LICENCES),
        "count": len(entries),
        "rejected": rejected,
        "images": entries,
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=1)

    print(f"\nfetched {len(entries)} images -> {out}")
    print(f"rejected: {rejected}")
    print(f"manifest: {manifest_path}")
    return 0 if entries else 1


if __name__ == "__main__":
    raise SystemExit(main())
