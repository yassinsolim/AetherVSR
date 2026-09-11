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

import math
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
from reparam import LADDERS, RepAetherSR
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


def load_video_pairs(
    pairs_dir: str,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load pre-degraded (LR, HR) patch pairs produced by tools/video-degrade.py.

    Unlike the still-photograph path, the degradation is not applied here and
    cannot be re-drawn per batch: it happened once, inside a real 720p H.264
    encode of a temporally coherent sequence, which is the entire point. The
    tensors are stored uint8 and converted on the way to the device.
    """
    with open(os.path.join(pairs_dir, "pairs.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    out: list[torch.Tensor] = []
    for split in ("train", "val"):
        path = os.path.join(pairs_dir, f"{split}.pt")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}; run tools/video-degrade.py first")
        blob = torch.load(path)
        lr, hr = blob["lr"], blob["hr"]
        if lr.shape[0] != hr.shape[0]:
            raise SystemExit(f"{split}: {lr.shape[0]} lr patches vs {hr.shape[0]} hr patches")
        if hr.shape[-1] != lr.shape[-1] * 2 or hr.shape[-2] != lr.shape[-2] * 2:
            raise SystemExit(f"{split}: hr {tuple(hr.shape)} is not 2x lr {tuple(lr.shape)}")
        out += [lr.float().div_(255.0), hr.float().div_(255.0)]
    return meta, out[0], out[1], out[2], out[3]


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
    ap.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="cap total optimizer updates and stop when reached. Equalises training "
        "compute across corpus sizes, so a dataset-scale comparison measures more "
        "unique data rather than more gradient steps. 0 disables the cap.",
    )
    ap.add_argument(
        "--pairs-hr-only",
        action="store_true",
        help="take only the HR patches from --pairs and degrade them with --degradation "
        "on the fly, exactly as the still-photograph path does. Isolates the corpus: "
        "video frames vs photographs with the degradation pipeline held identical.",
    )
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
        default="data/splits/corpus-v3.json",
        help="source-level split manifest; patches are drawn within each split",
    )
    ap.add_argument(
        "--degradation",
        default="box",
        help="HR->LR degradation: 'box' (clean 2x box downsample) or a realistic "
        "web-video profile name from tools/degrade.py",
    )
    ap.add_argument(
        "--pairs",
        default=None,
        help="directory of pre-degraded video pairs from tools/video-degrade.py. "
        "Mutually exclusive with --corpus/--split/--degradation: the degradation "
        "already happened, at full frame resolution, inside a real video encode.",
    )
    ap.add_argument("--rung", default="R0", choices=sorted(LADDERS),
                    help="training-time body block (Milestone 7). R0 is the ordinary "
                         "network; R1+ add linear branches that fuse back into the "
                         "identical deployed 3x3 convolution before export")
    ap.add_argument("--device", default=None, choices=["cpu", "mps"],
                    help="override device selection; cpu is slower but reproducible "
                         "and does not deadlock when runs are queued")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # MPS is the default because it is faster, but it deadlocks when more than
    # one process holds a Metal context, and this milestone trains twelve models
    # in a row alongside evaluation work. `--device cpu` is the escape hatch,
    # and it is also bit-reproducible, which MPS is not.
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    if args.pairs:
        # Video pairs carry their own source-level split and their own recorded
        # degradation, so none of the still-corpus machinery applies. What must
        # still hold is the thing that machinery existed to guarantee: no source
        # video on both sides.
        pairs_meta, train_lr, train_hr, val_lr, val_hr = load_video_pairs(args.pairs)
        split_digest = hashlib.sha256(
            json.dumps(
                {"train": pairs_meta["trainClips"], "val": pairs_meta["valClips"]},
                sort_keys=True, separators=(",", ":"),
            ).encode()
        ).hexdigest()
        leak = set(pairs_meta["trainClips"]) & set(pairs_meta["valClips"])
        if leak:
            raise SystemExit(f"refusing to train on a leaking split: {sorted(leak)}")
        print(f"pairs {args.pairs}: structure={pairs_meta['structure']} "
              f"crf={pairs_meta['crfDistribution']}")
        print(f"split by source video: train {len(pairs_meta['trainClips'])} clips, "
              f"val {len(pairs_meta['valClips'])} clips")
        print(f"patches: train {tuple(train_hr.shape)}  val {tuple(val_hr.shape)}")
    else:
        manifest = validate_manifest(args.corpus, args.min_images)
        pairs_meta, train_lr, val_lr = None, None, None

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

    # Structural reparameterization (Milestone 7) changes the *training* graph
    # only. R0 is the ordinary network; every other rung adds linear branches
    # that are summed before the existing tanh and fused back into the identical
    # deployed convolution before anything is exported. The training loop below
    # is deliberately untouched, so a rung comparison varies the block and
    # nothing else.
    if args.rung == "R0":
        model = AetherSR(channels=args.channels, depth=args.depth).to(device)
    else:
        model = RepAetherSR(
            channels=args.channels, depth=args.depth, branches=LADDERS[args.rung]
        ).to(device)
    fused_params = (
        model.fused_parameter_count() if hasattr(model, "fused_parameter_count")
        else count_parameters(model)
    )
    print(
        f"rung {args.rung}: {count_parameters(model)} training parameters, "
        f"{fused_params} fused inference parameters"
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    val_hr_d = val_hr.to(device)
    if args.pairs and not args.pairs_hr_only:
        # Already degraded, by a real encoder, once. Nothing to re-draw.
        val_lr_d = val_lr.to(device)
    else:
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

    # Steps, not just epochs. A larger corpus at a fixed epoch count silently
    # receives proportionally more optimizer updates, so "more data helped" and
    # "more updates helped" become indistinguishable - exactly the confound a
    # scaling curve has to avoid. With --max-steps every scale gets the same
    # number of updates and differs only in how much unique data those updates
    # saw.
    steps_per_epoch = max(1, (len(train_hr) - args.batch + 1 + args.batch - 1) // args.batch)
    step_budget = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch
    total_steps = 0

    # The cosine runs on optimizer updates, not epochs, and validation happens on
    # a fixed update cadence rather than at epoch boundaries.
    #
    # Both used to key off epochs, which quietly broke every cross-corpus
    # comparison: at a fixed 16,200-update budget a 12-clip corpus completes 159
    # short epochs while a 151-clip corpus completes 12 long ones, so the two ran
    # different fractions of their schedules (mean multiplier 0.5037 against
    # 0.5428, an 7.8% cumulative learning-rate advantage to the larger corpus)
    # and got 159 against 12 chances to draw a lucky best checkpoint. A scaling
    # curve measured that way varies schedule and selection cadence alongside
    # the thing it claims to isolate.
    VAL_EVERY = max(1, step_budget // 60)

    def lr_at(step: int) -> float:
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, step / step_budget)))

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
            sel = idx[i : i + args.batch]
            batch_hr = train_hr[sel].to(device)
            # Flips and quarter turns only: they are exact and do not resample,
            # so they cannot contaminate the degradation the model is learning
            # to invert.
            flip_w = random.random() < 0.5
            flip_h = random.random() < 0.5
            rot = random.random() < 0.5
            if flip_w:
                batch_hr = torch.flip(batch_hr, dims=[3])
            if flip_h:
                batch_hr = torch.flip(batch_hr, dims=[2])
            if rot:
                batch_hr = torch.rot90(batch_hr, 1, dims=[2, 3])

            if args.pairs and not args.pairs_hr_only:
                # The LR is already fixed to this HR, so it must receive exactly
                # the same geometric transform. Re-drawing per tensor would pair
                # a flipped target with an unflipped input and teach the model a
                # transform it will never see.
                batch_lr = train_lr[sel].to(device)
                if flip_w:
                    batch_lr = torch.flip(batch_lr, dims=[3])
                if flip_h:
                    batch_lr = torch.flip(batch_lr, dims=[2])
                if rot:
                    batch_lr = torch.rot90(batch_lr, 1, dims=[2, 3])
            else:
                # Fresh degradation draw per batch for randomised profiles, so the
                # model sees the distribution rather than one fixed sample of it.
                # `box` ignores the seed and stays exact.
                batch_lr = make_lr(
                    batch_hr, args.degradation, seed=args.seed * 100_003 + epoch * 1_009 + batches
                )

            for group in opt.param_groups:
                group["lr"] = lr_at(total_steps)
            out = model(batch_lr)
            loss = F.l1_loss(out, batch_hr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
            batches += 1
            total_steps += 1

            if total_steps % VAL_EVERY == 0 or total_steps >= step_budget:
                model.eval()
                with torch.no_grad():
                    pred = model(val_lr_d).clamp(0, 1)
                    v = psnr(pred, val_hr_d)
                if v > best:
                    best = v
                    # The SSIM *of the selected checkpoint*, not the best SSIM
                    # seen at any point: reporting each metric's own maximum
                    # describes a model that was never saved.
                    with torch.no_grad():
                        best_ssim = ssim(pred, val_hr_d)
                    best_state = {k: t.detach().cpu().clone()
                                  for k, t in model.state_dict().items()}
                model.train()
            if args.max_steps > 0 and total_steps >= args.max_steps:
                break

        if epoch % max(1, args.epochs // 12) == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  step {total_steps:6d}  "
                  f"loss {total / max(1, batches):.5f}  best {best:.2f} dB")
        if total_steps >= step_budget:
            print(f"  step budget {step_budget} reached during epoch {epoch}", file=sys.stderr)
            break

    print(
        f"trained in {time.time() - start:.0f}s; best val PSNR {best:.2f} dB "
        f"(bilinear {base_psnr:.2f} dB)"
    )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval().cpu()

    # Collapse the training-only branches before anything is measured or
    # exported. Everything downstream - parameter count, weights, golden vectors
    # and the shipped JSON - must see the ordinary deployed network, so the rung
    # cannot leak into production even by accident.
    training_parameters = count_parameters(model)
    rung_branches = list(getattr(model, "branches", ("k3",)))
    if hasattr(model, "fuse"):
        model = model.fuse()

    # Over verified file hashes: validate_manifest has already confirmed each
    # one against the bytes actually read, so this digest identifies the
    # content trained on rather than the strings describing it. The video-pairs
    # path has no still corpus; it is identified by its own split digest and
    # recorded degradation instead.
    corpus_digest = (
        None
        if args.pairs
        else hashlib.sha256(
            "".join(sorted(e["sha256"] for e in manifest["images"])).encode()
        ).hexdigest()
    )

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
        "degradation": (
            f"video {pairs_meta['structure']} / CRF {pairs_meta['crfDistribution']}: "
            f"{pairs_meta['structureSpec']['description']}; {pairs_meta['crfSpec']['description']}"
            if args.pairs
            else degradation_description(args.degradation)
        ),
        "training": {
            "corpus": pairs_meta["sourceManifest"] if args.pairs else manifest["source"],
            "corpusImages": pairs_meta["trainPatches"] if args.pairs else manifest["count"],
            "corpusDigest": None if args.pairs else corpus_digest,
            "corpusLicencePolicy": (
                "CC0/CC BY/CC BY-SA/public domain captured video; see manifest"
                if args.pairs
                else manifest["licence_policy"]
            ),
            # Which source images were trainable at all. A model carrying this
            # can be checked against the split it claims to have used, and a
            # score quoted against the wrong split becomes detectable.
            "split": os.path.basename(args.pairs) if args.pairs else os.path.basename(args.split),
            "splitDigest": split_digest,
            "splitSalt": None if args.pairs else split["salt"],
            "splitCounts": (
                {"train": len(pairs_meta["trainClips"]), "val": len(pairs_meta["valClips"])}
                if args.pairs
                else split["counts"]
            ),
            "splitPolicy": (
                "Source-level. Validation selects the checkpoint; test is never "
                "read during training. See ADR-0028."
            ),
            "degradationProfile": (
                f"video:{pairs_meta['structure']}:{pairs_meta['crfDistribution']}"
                if args.pairs
                else args.degradation
            ),
            # The exact compression the model was trained against, so two
            # differently-degraded models can never be confused for each other.
            "videoDegradation": (
                {
                    "structure": pairs_meta["structure"],
                    "x264params": pairs_meta["structureSpec"]["x264params"],
                    "crfDistribution": pairs_meta["crfDistribution"],
                    "crfHistogram": pairs_meta["crfHistogram"],
                    "preset": pairs_meta["preset"],
                    "sequences": pairs_meta["sequences"],
                    "sequenceFrames": pairs_meta["sequenceFrames"],
                    "trainClips": pairs_meta["trainClips"],
                    "valClips": pairs_meta["valClips"],
                    "pairSeed": pairs_meta["seed"],
                }
                if args.pairs
                else None
            ),
            "epochs": args.epochs,
            # Recorded so a scaling comparison can be checked for the compute
            # confound rather than trusted about it.
            "stepBudget": step_budget,
            "optimizerSteps": total_steps,
            "stepsPerEpoch": steps_per_epoch,
            "lrSchedule": "cosine over optimizer updates (T_max = step budget)",
            "validationEverySteps": VAL_EVERY,
            "validationOpportunities": step_budget // VAL_EVERY,
            "epochsCompleted": epoch + 1,
            "trainPatches": int(len(train_hr)),
            "patchesSeen": int(total_steps * args.batch),
            "batch": args.batch,
            "lr": args.lr,
            "patch": args.patch,
            "seed": args.seed,
            "loss": "L1",
            # Structural reparameterization is a TRAINING-time property. These
            # two numbers differ on purpose, and conflating them would describe
            # a model that was never shipped: `parameters` below is the fused
            # inference count and is what the runtime loads.
            "rung": args.rung,
            "rungBranches": rung_branches,
            "trainingParameters": training_parameters,
            "fusedInferenceParameters": count_parameters(model),
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
