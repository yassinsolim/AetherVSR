#!/usr/bin/env python3
"""Fetch the captured-footage test corpus.

Sources are **not redistributed**. The manifest pins each file by URL and
SHA-256; this downloads them into a gitignored working directory and records
the hash actually received, so a later run can prove it got the same bytes.

Licence handling follows the rule the project settled on in Milestone 4.5:
accept only licences read from the file's own metadata, never inferred from a
category or a collection page. Anything ambiguous is rejected rather than
managed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request

UA = "AetherVSR/0.1 (https://github.com/yassinsolim/AetherVSR) captured-footage-fetch"
ACCEPTED = ("cc0", "cc by", "cc by-sa", "public domain")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# NonCommercial and NoDerivatives both disqualify. NC because the project is
# open and may be used commercially; ND because every frame this benchmark
# produces - downscaled, encoded, upscaled - is a derivative work.
FORBIDDEN = re.compile(r"\b(nc|nd)\b|noncommercial|noderiv", re.I)


def licence_ok(licence: str) -> bool:
    low = (licence or "").strip().lower()
    if not low:
        return False
    # Tokenise on the separators CC uses, so "cc by-nd 4.0" yields a bare "nd"
    # that the forbidden pattern can see. Substring checks got this wrong:
    # "-nd" does not appear in "cc by-nd" the way a naive split expects.
    tokens = re.split(r"[\s\-/]+", low)
    if any(FORBIDDEN.fullmatch(t) for t in tokens) or FORBIDDEN.search(low.replace(" ", "")):
        return False
    return any(low.startswith(a) for a in ACCEPTED)


def download(url: str, dest: str, timeout: int = 900) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    total = 0
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            total += len(chunk)
    os.replace(tmp, dest)
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch captured-footage sources.")
    ap.add_argument("--manifest", default="data/captured/manifest.json")
    ap.add_argument("--out", default="data/captured/sources")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    os.makedirs(args.out, exist_ok=True)

    wanted = {c.strip() for c in args.only.split(",") if c.strip()}
    rejected, fetched, reused = [], [], []

    for clip in manifest["clips"]:
        if wanted and clip["id"] not in wanted:
            continue
        if not licence_ok(clip["licence"]):
            rejected.append((clip["id"], clip["licence"]))
            print(f"  REJECT {clip['id']}: licence {clip['licence']!r}", file=sys.stderr)
            continue

        dest = os.path.join(args.out, clip["source_file"])
        if os.path.exists(dest):
            digest = sha256_file(dest)
            recorded = clip.get("source_sha256")
            if recorded and digest != recorded:
                print(f"  HASH MISMATCH {clip['id']}: have {digest[:12]}, manifest {recorded[:12]}", file=sys.stderr)
            reused.append(clip["id"])
            clip["source_sha256"] = digest
            continue

        try:
            size = download(clip["source_url"], dest)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {clip['id']}: {type(exc).__name__} {exc}", file=sys.stderr)
            continue
        digest = sha256_file(dest)
        clip["source_sha256"] = digest
        clip["source_bytes_actual"] = size
        fetched.append(clip["id"])
        print(f"  [{len(fetched)+len(reused)}/{len(manifest['clips'])}] {clip['id']:<44} {size/1e6:7.1f} MB", file=sys.stderr)

    manifest["fetch_summary"] = {
        "fetched": len(fetched), "reused": len(reused), "rejected": rejected,
        "note": "Sources are gitignored and never redistributed; hashes pin what was used.",
    }
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"\nfetched {len(fetched)}, reused {len(reused)}, rejected {len(rejected)}", file=sys.stderr)
    return 0 if not rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
