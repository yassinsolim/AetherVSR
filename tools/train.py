#!/usr/bin/env python3
"""Train AetherVSR's Milestone 4 network and emit a model file.

Deliberately small and reproducible rather than clever. The point is a set of
weights that beats Catmull-Rom on our own evaluation, produced from a corpus
whose licence permits it, with everything recorded that would be needed to do
it again.

    python3 tools/train.py --corpus data/corpus --depth 2 --channels 16 \
        --epochs 60 --out public/models/aethersr-c16d2.json

The corpus manifest is validated before a single patch is read: every entry
must carry a CC0/public-domain licence string, a source URL and a SHA-256, and
the file it names must exist on disk. Training from a partial or unaudited
corpus would make the weights' provenance unprovable, which is the one thing
this model file has to be able to demonstrate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from PIL import Image

from aethersr import AetherSR, box_downsample2, count_parameters, export_weights

ARCHITECTURE_ID = "aethersr-resizeconv"
ARCHITECTURE_VERSION = 2
ACCEPTED_LICENCES = {"cc0", "cc0 1.0", "public domain", "pd", "cc-zero"}


def validate_manifest(corpus: str, minimum: int) -> dict:
    """Refuse to train unless the corpus is complete and provably CC0."""
    path = os.path.join(corpus, "manifest.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} is missing. Run tools/fetch-corpus.py to completion first; "
            "a partial download has no manifest and therefore no provenance."
        )
    with open(path) as fh:
        manifest = json.load(fh)

    images = manifest.get("images", [])
    problems: list[str] = []
    if len(images) < minimum:
        problems.append(f"only {len(images)} images, need at least {minimum}")
    for entry in images:
        name = entry.get("file", "(unnamed)")
        if (entry.get("licence") or "").strip().lower() not in ACCEPTED_LICENCES:
            problems.append(f"{name}: licence {entry.get('licence')!r} is not accepted")
        if not entry.get("source_url"):
            problems.append(f"{name}: no source_url")
        if not entry.get("sha256"):
            problems.append(f"{name}: no sha256")
        if not os.path.exists(os.path.join(corpus, name)):
            problems.append(f"{name}: listed in manifest but absent on disk")
    if problems:
        head = "\n  ".join(problems[:12])
        more = f"\n  … and {len(problems) - 12} more" if len(problems) > 12 else ""
        raise SystemExit(f"corpus failed validation:\n  {head}{more}")

    print(f"corpus validated: {len(images)} images, all CC0, all hashed and present")
    return manifest


def load_patches(
    corpus: str, manifest: dict, patch: int, per_image: int, limit: int, seed: int
) -> torch.Tensor:
    """Sample fixed-size HR patches. Returns float32 NCHW in [0,1]."""
    rng = random.Random(seed)
    patches: list[torch.Tensor] = []
    skipped = 0
    for entry in manifest["images"][:limit]:
        path = os.path.join(corpus, entry["file"])
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                w, h = im.size
                if w < patch or h < patch:
                    skipped += 1
                    continue
                raw = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
                arr = raw.reshape(h, w, 3).permute(2, 0, 1).float().div_(255.0)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(f"  skip {entry['file']}: {type(exc).__name__} {exc}")
            continue
        for _ in range(per_image):
            x = rng.randrange(0, w - patch + 1)
            y = rng.randrange(0, h - patch + 1)
            patches.append(arr[:, y : y + patch, x : x + patch].clone())
    if not patches:
        raise SystemExit("no usable patches; is the corpus populated?")
    if skipped:
        print(f"  {skipped} images skipped (too small or unreadable)")
    return torch.stack(patches)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    if mse <= 0:
        return 99.0
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--per-image", type=int, default=8)
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--min-images", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=20260902)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    manifest = validate_manifest(args.corpus, args.min_images)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    hr = load_patches(args.corpus, manifest, args.patch, args.per_image, args.limit, args.seed)
    print(f"patches: {tuple(hr.shape)}")

    # Held out before training and never trained on.
    n_val = max(1, len(hr) // 10)
    perm = torch.randperm(len(hr), generator=torch.Generator().manual_seed(args.seed))
    val_hr = hr[perm[:n_val]]
    train_hr = hr[perm[n_val:]]
    print(f"train {len(train_hr)}  val {len(val_hr)}")

    model = AetherSR(channels=args.channels, depth=args.depth).to(device)
    print(f"parameters: {count_parameters(model)}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    val_hr_d = val_hr.to(device)
    val_lr_d = box_downsample2(val_hr_d)

    # Bilinear on the same split, so "did it learn anything" has an answer that
    # does not depend on the browser harness.
    with torch.no_grad():
        bilinear = F.interpolate(val_lr_d, scale_factor=2, mode="bilinear", align_corners=False)
        base_psnr = psnr(bilinear.clamp(0, 1), val_hr_d)
    print(f"bilinear baseline on val: {base_psnr:.2f} dB")

    best = -1.0
    best_state = None
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        idx = torch.randperm(len(train_hr))
        total = 0.0
        batches = 0
        for i in range(0, len(train_hr) - args.batch + 1, args.batch):
            batch_hr = train_hr[idx[i : i + args.batch]].to(device)
            # Flips and quarter turns only: they are exact and do not resample,
            # so they cannot contaminate the degradation the model is learning
            # to invert.
            if random.random() < 0.5:
                batch_hr = torch.flip(batch_hr, dims=[3])
            if random.random() < 0.5:
                batch_hr = torch.flip(batch_hr, dims=[2])
            if random.random() < 0.5:
                batch_hr = torch.rot90(batch_hr, 1, dims=[2, 3])
            batch_lr = box_downsample2(batch_hr)

            out = model(batch_lr)
            loss = F.l1_loss(out, batch_hr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
            batches += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            v = psnr(model(val_lr_d), val_hr_d)
        if v > best:
            best = v
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(
                f"  epoch {epoch:3d}  loss {total / max(1, batches):.5f}  val {v:.2f} dB"
                f"  best {best:.2f}"
            )

    print(
        f"trained in {time.time() - start:.0f}s; best val PSNR {best:.2f} dB "
        f"(bilinear {base_psnr:.2f} dB)"
    )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval().cpu()

    corpus_digest = hashlib.sha256(
        "".join(sorted(e["sha256"] for e in manifest["images"])).encode()
    ).hexdigest()

    payload = {
        "architecture": ARCHITECTURE_ID,
        "architectureVersion": ARCHITECTURE_VERSION,
        "scale": 2,
        "inChannels": 3,
        "outChannels": 3,
        "features": args.channels,
        "depth": args.depth,
        "precision": "f32 source weights; runtime may narrow to f16",
        "layers": (
            [
                {
                    "name": "stem",
                    "type": "conv",
                    "kernel": 5,
                    "in": 3,
                    "out": args.channels,
                    "padding": 2,
                    "activation": "tanh",
                }
            ]
            + [
                {
                    "name": f"body.{i}",
                    "type": "conv",
                    "kernel": 3,
                    "in": args.channels,
                    "out": args.channels,
                    "padding": 1,
                    "activation": "tanh",
                }
                for i in range(args.depth)
            ]
            + [
                {"name": "upsample", "type": "nearest", "scale": 2},
                {
                    "name": "head",
                    "type": "conv",
                    "kernel": 3,
                    "in": args.channels,
                    "out": 3,
                    "padding": 1,
                    "activation": "none",
                    "residual": "nearest-upsampled input added before clamp",
                    "clamp": [0, 1],
                },
            ]
        ),
        "normalisation": {"mean": [0, 0, 0], "scale": [1, 1, 1], "range": "[0,1] RGB"},
        "degradation": "box downsample by 2 (avg_pool2d k2 s2), matching quality.ts boxDownsample2",
        "training": {
            "corpus": manifest["source"],
            "corpusImages": manifest["count"],
            "corpusDigest": corpus_digest,
            "corpusLicencePolicy": manifest["licence_policy"],
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "patch": args.patch,
            "seed": args.seed,
            "loss": "L1",
            "valPsnrDb": round(best, 3),
            "bilinearValPsnrDb": round(base_psnr, 3),
        },
        "provenance": (
            "Trained in-house by the AetherVSR project from the architecture in ADR-0022. "
            "Not an upstream checkpoint and not derived from one."
        ),
        "parameters": count_parameters(model),
        "weights": export_weights(model),
    }
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    payload["sha256"] = hashlib.sha256(body.encode()).hexdigest()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"), sort_keys=True)
    print(
        f"wrote {args.out} ({os.path.getsize(args.out) / 1024:.1f} kB), "
        f"sha256 {payload['sha256'][:16]}…"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
