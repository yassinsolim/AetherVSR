"""The M7 validation repair must preserve training and enforce creator isolation."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from corpus_schema import derive_shoot_id, normalise_creator, validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    return json.loads((ROOT / relative).read_text())


def test_decorated_creator_credit_identifies_the_same_operator():
    assert normalise_creator("Foto: PantheraLeo1359531\nSchild: Landmetzgerei Krafft") == normalise_creator("PantheraLeo1359531")


def test_m7_validation_is_disjoint_from_current_and_legacy_training():
    validation = load("data/captured-val-m7/manifest.json")
    current = load("data/captured-train-v2/manifest.json")["clips"]
    legacy = load("data/captured-train/manifest.json")["clips"]
    assert len(current) == 151
    assert len(legacy) == 12
    training = current + legacy
    assert validate_manifest(validation) == []
    creators = {normalise_creator(clip["creator"]) for clip in training}
    shoots = {derive_shoot_id(clip) for clip in training} | {clip["shootId"] for clip in training}
    identity_fields = ("source_url", "title", "pageid", "commonsSha1", "source_sha256")
    identities = {str(clip[key]) for clip in training for key in identity_fields if clip.get(key)}
    for clip in validation["clips"]:
        for alias in clip.get("creatorAliases", [clip["creator"]]):
            assert normalise_creator(alias) not in creators, clip["id"]
        assert clip["shootId"] not in shoots
        assert derive_shoot_id(clip) not in shoots
        assert not {str(clip[key]) for key in identity_fields if clip.get(key)} & identities


def test_revision_replaces_only_the_conflicting_nature_clip():
    original = load("data/captured-val/manifest.json")
    revised = load("data/captured-val-m7/manifest.json")
    old = {clip["id"]: clip for clip in original["clips"]}
    new = {clip["id"]: clip for clip in revised["clips"]}
    assert len(new) == len(old) == 16
    assert set(old) - set(new) == {"nature-knollenteich-mit-enten-20250515-c1288"}
    assert set(new) - set(old) == {"nature-pikktiib-sooritsika-conocephalus-fuscus"}
    assert revised["corpusVersion"] != original.get("corpusVersion")
    assert revised["revisionOf"] == "data/captured-val/manifest.json"
    for cid in set(old) & set(new):
        assert new[cid] == old[cid]
    assert sorted(clip["category"] for clip in new.values()) == sorted(clip["category"] for clip in old.values())


def test_validation_freeze_binds_new_split_and_unchanged_training():
    freeze = load("results/m7-validation-freeze.json")
    for path_key, hash_key in [
        ("manifest", "manifestSha256"),
        ("historicalManifest", "historicalManifestSha256"),
        ("trainingManifest", "trainingManifestSha256"),
    ]:
        assert hashlib.sha256((ROOT / freeze[path_key]).read_bytes()).hexdigest() == freeze[hash_key]
