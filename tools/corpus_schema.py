#!/usr/bin/env python3
"""The captured-corpus manifest schema, and the identity rules that go with it.

Milestone 5.5 split its training corpus by file and leaked: two clips from one
event, one operator, one camera, 36 minutes apart, landed on opposite sides of
the train/validation boundary. Splitting by file is not splitting by anything
the model can distinguish.

So provenance here is a three-level hierarchy, and every manifest carries all
three explicitly rather than leaving them to be inferred at split time:

    creator   who operated the camera
    shoot     one session: one creator, one event, one day, one location
    clip      one file

A split is valid at a level only if no group at that level spans both sides.
`shootId` is mandatory because it is the level that actually bit us; `creator`
is recorded so a stricter creator-disjoint split can be requested when the
corpus is large enough to afford it.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

SCHEMA = "aethervsr.captured-corpus/2"

ROLES = ("training", "validation", "regression-benchmark", "faces-benchmark", "confirmation")

CATEGORIES = ("faces", "nature", "urban", "motion", "texture", "lowlight", "daylight", "text")

# Derivative works are produced from every clip (we degrade and re-encode them),
# so NoDerivatives is disqualifying. NonCommercial is disqualifying because the
# project is Apache-2.0 and cannot inherit a use restriction.
LICENCE_ALLOW = ("cc0", "cc by", "cc-by", "public domain", "pd-", "cc zero", "cc-zero")
LICENCE_DENY = ("nc", "noncommercial", "non-commercial", "nd", "noderiv", "no derivative")

REQUIRED_CLIP_FIELDS = (
    "id", "title", "category", "creator", "shootId",
    "source_url", "description_url", "licence",
    "source_width", "source_height",
)

MIN_WIDTH, MIN_HEIGHT = 2560, 1440


def normalise_creator(name: str | None) -> str:
    """Fold a Commons `Artist` string down to a stable grouping key.

    Unicode-aware on purpose: an earlier version stripped to ASCII and silently
    erased two Japanese uploader names, which un-grouped their clips and
    reintroduced the leak the key exists to close.
    """
    s = unicodedata.normalize("NFKC", (name or "")).casefold()
    s = re.sub(r"\(.*?\)", " ", s)          # "(talk)" and similar decorations
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return " ".join(s.split()) or "unattributed"


def licence_ok(licence: str | None) -> bool:
    """Substring matching is deliberate: 'CC BY-NC-ND 4.0' must fail on both
    counts, and a bare 'CC BY 4.0' must pass without enumerating every version.
    Deny is checked first so a permissive prefix cannot smuggle a restriction.
    """
    low = (licence or "").strip().casefold()
    if not low:
        return False
    # Token-aware so that "nd" inside a word cannot trip the deny list, while
    # "CC BY-NC-ND" still does.
    tokens = re.split(r"[^a-z0-9]+", low)
    for bad in LICENCE_DENY:
        if bad in tokens or bad in low.replace(" ", "-").split("-"):
            return False
    return any(low.startswith(good) or good in low for good in LICENCE_ALLOW)


def derive_shoot_id(clip: dict) -> str:
    """A shoot key for clips whose manifest does not already carry one.

    Creator plus the title's leading words with sequence numbers removed, so
    numbered rolls from one session collapse together ("... C0259", "... C0284",
    "4. Lewis and Clark Bridge", "7. Lewis and Clark Bridge") while unrelated
    uploads by a prolific creator stay apart.
    """
    who = normalise_creator(clip.get("creator"))
    title = unicodedata.normalize("NFKC", clip.get("title") or clip.get("id") or "").casefold()
    title = re.sub(r"^file:", "", title).strip()
    title = re.sub(r"\.(webm|mp4|mov|ogv)$", "", title)
    words = [w for w in re.split(r"[^\w]+", title, flags=re.UNICODE) if w]
    # A leading ordinal numbers the roll within a shoot ("4. Lewis and Clark",
    # "7. Lewis and Clark"), so it must go before the stem is taken or the two
    # halves of one session land in different groups.
    while words and re.fullmatch(r"\d{1,2}", words[0]):
        words.pop(0)
    # Drop sequence markers and dates anywhere: they distinguish rolls, not
    # shoots.
    words = [w for w in words if not re.fullmatch(r"c?\d{3,}|\d{8}|\d{4}", w)]
    stem = " ".join(words[:4])
    return f"{who}::{stem}" if stem else f"{who}::{clip.get('id', '?')}"


def validate_clip(clip: dict) -> list[str]:
    problems: list[str] = []
    for field in REQUIRED_CLIP_FIELDS:
        if not clip.get(field):
            problems.append(f"{clip.get('id', '?')}: missing {field}")
    cat = clip.get("category")
    if cat and cat not in CATEGORIES:
        problems.append(f"{clip.get('id')}: unknown category {cat!r}")
    if not licence_ok(clip.get("licence")):
        problems.append(f"{clip.get('id')}: licence {clip.get('licence')!r} is not permissive")
    # Identity may be pinned either by a full-file SHA-256 (the download path,
    # used for the small benchmark corpora) or by the Commons digest plus our
    # own prefix hash (the streaming path, which never transfers whole files).
    # Both fix the bytes; requiring the first would have forced 52 GB of
    # downloads to use three seconds of each clip.
    has_pin = clip.get("source_sha256") or (clip.get("commonsSha1") and clip.get("prefixSha256"))
    if not has_pin:
        problems.append(f"{clip.get('id')}: no content pin (source_sha256, or commonsSha1+prefixSha256)")
    w, h = clip.get("source_width") or 0, clip.get("source_height") or 0
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        problems.append(f"{clip.get('id')}: {w}x{h} is below the {MIN_WIDTH}x{MIN_HEIGHT} floor")
    if w and h and h > w:
        problems.append(f"{clip.get('id')}: portrait source {w}x{h}; 2x geometry needs landscape")
    return problems


def validate_manifest(manifest: dict) -> list[str]:
    problems: list[str] = []
    if manifest.get("schema") != SCHEMA:
        problems.append(f"schema is {manifest.get('schema')!r}, expected {SCHEMA!r}")
    if manifest.get("role") not in ROLES:
        problems.append(f"role is {manifest.get('role')!r}, expected one of {ROLES}")
    seen_ids: dict[str, int] = {}
    for clip in manifest.get("clips", []):
        problems += validate_clip(clip)
        seen_ids[clip.get("id", "?")] = seen_ids.get(clip.get("id", "?"), 0) + 1
    for cid, n in seen_ids.items():
        if n > 1:
            problems.append(f"duplicate clip id {cid!r} appears {n} times")
    return problems


def corpus_digest(manifest: dict) -> str:
    """Identity of the *content*, not of the file describing it."""
    hashes = sorted(c.get("source_sha256", "") for c in manifest.get("clips", []))
    return hashlib.sha256("".join(hashes).encode()).hexdigest()
