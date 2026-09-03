#!/usr/bin/env python3
"""Train AetherVSR's Milestone 4 network and emit a model file.

Deliberately small and reproducible rather than clever. The point is a set of
weights that beats Catmull-Rom on our own evaluation, produced from a corpus
whose licence permits it, with everything recorded that would be needed to do
it again.

    python3 tools/train.py --corpus data/corpus --depth 2 --channels 16 \
        --epochs 60 --out public/models/aethersr-c16d2.json

The corpus manifest is validated before a single patch is read: every entry
must carry a CC0 licence string, a source URL and a SHA-256, and
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
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image

from aethersr import AetherSR, box_downsample2, count_parameters, export_weights
from dataset import check_disjoint, load_split, split_files
from degrade import degrade_tensor

ARCHITECTURE_ID = "aethersr-resizeconv"
ARCHITECTURE_VERSION = 2
# Must match tools/fetch-corpus.py. This is the enforcement point that an
# edited or re-fetched manifest passes through, so leaving the generic tags
# here would have let exactly the corpus ADR-0026 rejects train and still be
# described as CC0.
ACCEPTED_LICENCES = {"cc0", "cc0 1.0", "cc-zero"}


def _sha256_file(path: str) -> str:
    """Streams the file so a large corpus does not have to fit in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
        declared = entry.get("sha256")
        if not declared:
            problems.append(f"{name}: no sha256")
        path = os.path.join(corpus, name)
        if not os.path.exists(path):
            problems.append(f"{name}: listed in manifest but absent on disk")
        elif declared:
            # The corpus images are gitignored, so the manifest is the only
            # record of what was trained on. Checking the declared hash against
            # the bytes on disk is the difference between a reproducibility
            # claim and a reproducibility hope: without it, edited files train a
            # different model under an unchanged corpus digest.
            actual = _sha256_file(path)
            if actual != declared:
                problems.append(f"{name}: sha256 {actual[:12]}… does not match manifest {declared[:12]}…")
    if problems:
        head = "\n  ".join(problems[:12])
        more = f"\n  … and {len(problems) - 12} more" if len(problems) > 12 else ""
        raise SystemExit(f"corpus failed validation:\n  {head}{more}")

    print(f"corpus validated: {len(images)} images, all CC0, all hashed and present")
    return manifest


def load_patches(
    corpus: str, files: list[str], patch: int, per_image: int, limit: int, seed: int
) -> torch.Tensor:
    """Sample fixed-size HR patches from an explicit file list.

    Takes files rather than the whole manifest because patches must be
    extracted *within* an already-decided source split. Pooling patches and
    splitting the pool is what leaked source identity in Milestone 4: every
    validation patch came from a photograph that was also trained on.
    """
    rng = random.Random(seed)
    patches: list[torch.Tensor] = []
    skipped = 0
    for name in files[:limit]:
        path = os.path.join(corpus, name)
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
            print(f"  skip {name}: {type(exc).__name__} {exc}")
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


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean SSIM over an 11x11 Gaussian window, on luma.

    Matches the convention in `src/bench/quality.ts` so a number here can be
    compared with one from the browser harness rather than merely resembling it.
    """
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=a.device).view(1, 3, 1, 1)
    x = (a * weights).sum(1, keepdim=True)
    y = (b * weights).sum(1, keepdim=True)

    coords = torch.arange(11, dtype=torch.float32, device=a.device) - 5.0
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = (g / g.sum()).view(1, 1, -1)
    kernel = (g.transpose(2, 1) @ g).view(1, 1, 11, 11)

    def blur(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, kernel, padding=0)

    mu_x, mu_y = blur(x), blur(y)
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_xx = blur(x * x) - mu_xx
    sigma_yy = blur(y * y) - mu_yy
    sigma_xy = blur(x * y) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2)
    return float((numerator / denominator).mean())


def make_lr(hr: torch.Tensor, profile: str, seed: int) -> torch.Tensor:
    """HR -> LR under the named degradation profile.

    `box` stays on the exact avg_pool2d path so the clean condition is
    bit-identical to what Milestone 4 trained on and to `quality.ts`.
    """
    if profile == "box":
        return box_downsample2(hr)
    return degrade_tensor(hr, profile, seed).to(hr.device)


def degradation_description(profile: str) -> str:
    """Human-readable degradation, recorded in the model file."""
    if profile == "box":
        return "box downsample by 2 (avg_pool2d k2 s2), matching quality.ts boxDownsample2"
    return f"realistic web-video profile {profile!r}; see tools/degrade.py and BENCHMARKS.md"


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
    ap.add_argument(
        "--split",
        default="data/splits/corpus-v1.json",
        help="source-level split manifest; patches are drawn within each split",
    )
    ap.add_argument(
        "--degradation",
        default="box",
        help="HR->LR degradation: 'box' (clean 2x box downsample) or a realistic "
        "web-video profile name from tools/degrade.py",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    manifest = validate_manifest(args.corpus, args.min_images)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    # The split is decided by source image before a single patch is read, so a
    # photograph cannot contribute to more than one side. `test` is deliberately
    # not loaded here: training must not be able to see it even by accident, and
    # a test score computed every epoch is an invitation to select on it.
    # `tools/evaluate.py` scores a frozen model on any split, once.
    split = load_split(args.split)
    problems = check_disjoint(split)
    if problems:
        for problem in problems:
            print(f"ERROR split not disjoint: {problem}", file=sys.stderr)
        raise SystemExit("refusing to train on a leaking split")

    split_digest = hashlib.sha256(
        json.dumps(split["splits"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    train_files = split_files(split, "train")
    val_files = split_files(split, "val")
    print(f"split {args.split}: train {len(train_files)} images, val {len(val_files)} images")

    train_hr = load_patches(args.corpus, train_files, args.patch, args.per_image, args.limit, args.seed)
    # A fixed seed offset, so validation patches are stable across training
    # seeds: comparing five seeds against five different validation sets would
    # confound seed variance with sampling variance.
    val_hr = load_patches(args.corpus, val_files, args.patch, args.per_image, args.limit, 20260101)
    print(f"patches: train {tuple(train_hr.shape)}  val {tuple(val_hr.shape)}")

    model = AetherSR(channels=args.channels, depth=args.depth).to(device)
    print(f"parameters: {count_parameters(model)}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    val_hr_d = val_hr.to(device)
    # Validation degradation is fixed for the run: a validation set that is
    # re-randomised every epoch makes checkpoint selection partly a draw on the
    # degradation, not on the model.
    val_lr_d = make_lr(val_hr_d, args.degradation, seed=20260101)

    # Bilinear on the same split, so "did it learn anything" has an answer that
    # does not depend on the browser harness.
    with torch.no_grad():
        bilinear = F.interpolate(val_lr_d, scale_factor=2, mode="bilinear", align_corners=False)
        base_psnr = psnr(bilinear.clamp(0, 1), val_hr_d)
    print(f"bilinear baseline on val: {base_psnr:.2f} dB")

    best = -1.0
    best_ssim = float("nan")
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
            # Fresh degradation draw per batch for randomised profiles, so the
            # model sees the distribution rather than one fixed sample of it.
            # `box` ignores the seed and stays exact.
            batch_lr = make_lr(batch_hr, args.degradation, seed=args.seed * 100_003 + epoch * 1_009 + batches)

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
            pred = model(val_lr_d).clamp(0, 1)
            v = psnr(pred, val_hr_d)
        if v > best:
            best = v
            # The SSIM *of the selected checkpoint*, not the best SSIM seen at
            # any epoch: reporting each metric's own maximum describes a model
            # that was never saved.
            with torch.no_grad():
                best_ssim = ssim(pred, val_hr_d)
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

    # Over verified file hashes: validate_manifest has already confirmed each
    # one against the bytes actually read, so this digest identifies the
    # content trained on rather than the strings describing it.
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
        "degradation": degradation_description(args.degradation),
        "training": {
            "corpus": manifest["source"],
            "corpusImages": manifest["count"],
            "corpusDigest": corpus_digest,
            "corpusLicencePolicy": manifest["licence_policy"],
            # Which source images were trainable at all. A model carrying this
            # can be checked against the split it claims to have used, and a
            # score quoted against the wrong split becomes detectable.
            "split": os.path.basename(args.split),
            "splitDigest": split_digest,
            "splitSalt": split["salt"],
            "splitCounts": split["counts"],
            "splitPolicy": (
                "Source-level. Validation selects the checkpoint; test is never "
                "read during training. See ADR-0028."
            ),
            "degradationProfile": args.degradation,
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "patch": args.patch,
            "seed": args.seed,
            "loss": "L1",
            "valPsnrDb": round(best, 3),
            "valSsim": round(best_ssim, 5),
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
