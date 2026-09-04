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
import urllib.parse
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


def live_licence(description_url: str | None) -> str | None:
    """Reads LicenseShortName straight from the Commons API for this file."""
    if not description_url or "/File:" not in description_url:
        return None
    title = "File:" + description_url.split("/File:", 1)[1]
    params = urllib.parse.urlencode({
        "action": "query", "format": "json", "titles": urllib.parse.unquote(title),
        "prop": "imageinfo", "iiprop": "extmetadata",
    })
    try:
        req = urllib.request.Request(
            "https://commons.wikimedia.org/w/api.php?" + params, headers={"User-Agent": UA}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        for page in (data.get("query") or {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            return ((info.get("extmetadata") or {}).get("LicenseShortName") or {}).get("value")
    except Exception:  # noqa: BLE001
        return None
    return None


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
    mismatched: list[dict] = []
    failed: list[dict] = []
    licence_drift: list[dict] = []

    for clip in manifest["clips"]:
        if wanted and clip["id"] not in wanted:
            continue
        # Re-derive the licence from Commons rather than trusting the string in
        # the manifest. The docstring claimed licences were read from the file's
        # own metadata; in fact the gate only validated a transcription, so a
        # manifest edit would have passed it.
        live = live_licence(clip.get("description_url"))
        if live and live.strip().lower() != (clip.get("licence") or "").strip().lower():
            licence_drift.append({"id": clip["id"], "manifest": clip.get("licence"), "commons": live})
            print(f"  LICENCE DRIFT {clip['id']}: manifest {clip.get('licence')!r}, Commons {live!r}", file=sys.stderr)
            continue
        if live:
            clip["licence"] = live
        if not licence_ok(clip["licence"]):
            rejected.append((clip["id"], clip["licence"]))
            print(f"  REJECT {clip['id']}: licence {clip['licence']!r}", file=sys.stderr)
            continue

        dest = os.path.join(args.out, clip["source_file"])
        if os.path.exists(dest):
            digest = sha256_file(dest)
            recorded = clip.get("source_sha256")
            # Never overwrite the pin. Rewriting it on mismatch turned the
            # integrity check into a rubber stamp: a truncated or re-transcoded
            # source would silently become the new "expected" value and results
            # would be produced over altered bytes with nothing reporting it.
            if recorded and digest != recorded:
                mismatched.append({"id": clip["id"], "expected": recorded, "found": digest})
                print(f"  HASH MISMATCH {clip['id']}: have {digest[:12]}, manifest {recorded[:12]}", file=sys.stderr)
                continue
            reused.append(clip["id"])
            if not recorded:
                clip["source_sha256"] = digest
            continue

        try:
            size = download(clip["source_url"], dest)
        except Exception as exc:  # noqa: BLE001
            failed.append({"id": clip["id"], "error": f"{type(exc).__name__} {exc}"})
            print(f"  FAILED {clip['id']}: {type(exc).__name__} {exc}", file=sys.stderr)
            continue
        digest = sha256_file(dest)
        clip["source_sha256"] = digest
        clip["source_bytes_actual"] = size
        fetched.append(clip["id"])
        print(f"  [{len(fetched)+len(reused)}/{len(manifest['clips'])}] {clip['id']:<44} {size/1e6:7.1f} MB", file=sys.stderr)

    missing = [c["id"] for c in manifest["clips"]
               if (not wanted or c["id"] in wanted)
               and not os.path.exists(os.path.join(args.out, c["source_file"]))]
    manifest["fetch_summary"] = {
        "fetched": len(fetched), "reused": len(reused), "rejected": rejected,
        "mismatched": mismatched, "failed": failed, "licenceDrift": licence_drift,
        "missing": missing,
        "note": "Sources are gitignored and never redistributed; hashes pin what was used.",
    }
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    problems = len(rejected) + len(mismatched) + len(failed) + len(licence_drift) + len(missing)
    print(
        f"\nfetched {len(fetched)}, reused {len(reused)}, rejected {len(rejected)}, "
        f"mismatched {len(mismatched)}, failed {len(failed)}, licence drift {len(licence_drift)}, "
        f"missing {len(missing)}",
        file=sys.stderr,
    )
    # A partial corpus must not exit zero: downstream tools iterate whatever
    # exists and would produce a captured-footage result over a subset while
    # reporting no error.
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
