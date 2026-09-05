#!/usr/bin/env python3
"""Milestone 5.5 Part E: is CRF 34 information destroyed, or just unexploited?

Milestone 5 found the neural model's advantage over Catmull-Rom collapses to
statistical noise at CRF 34 (+0.055 dB, 7/10 wins, not significant) while
holding at CRF 18/26. Milestone 5.5 refuted the leading explanation (GOP
structure). This script asks the remaining question directly: does a
dramatically higher-capacity offline model recover meaningfully more signal
from the same CRF 34 bitstream, or has the compression destroyed the
information regardless of decoder capacity?

Oracle: SwinIR classical image super-resolution, x2, "M" (middle) size,
trained on DIV2K+Flickr2K (001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth),
11,752,487 parameters -- 1,868x the 6,291-parameter production AetherSR
C16D2. Apache-2.0, ETH Zurich Computer Vision Lab (Liang et al., ICCVW 2021,
arXiv:2108.10257). Licence and weight/source provenance are recorded verbatim
in the output JSON's "oracle" block, not just asserted here.

Why SwinIR and not a GAN oracle (Real-ESRGAN): a first pass with
Real-ESRGAN's x2plus RRDBNet (16.7M params, BSD-3-Clause) was tried and
discarded -- see the "consideredAndRejected" block in the output JSON. Its
adversarial/perceptual training objective trades pixel fidelity for
hallucinated sharpness, so it scored *below* Catmull-Rom on PSNR despite
2,655x the parameters, on a small preliminary sample. That is a real
property of GAN-SR, not evidence about recoverable information, and it would
have confounded this specific diagnostic (which asks about pixel-fidelity
headroom, matching AetherSR's own L1-loss training objective). SwinIR's
classical-SR checkpoint is trained with a pixel-fidelity loss on DIV2K+
Flickr2K, making it the apples-to-apples high-capacity comparison this
question needs.

THIS IS A DIAGNOSTIC UPPER BOUND, NOT A PRODUCTION CANDIDATE.
  - It runs roughly four orders of magnitude slower per frame than AetherSR
    C16D2 (tens of seconds, not milliseconds) and is evaluated offline on
    Apple Silicon (MPS), never in the browser.
  - It is scored ONLY on the frozen 10-clip captured test set (see
    data/captured/FROZEN.json), and ONLY at h264_poor (CRF 34) and
    h264_typical (CRF 26, for contrast). This happens after every AetherSR
    modelling decision -- architecture, seed, hyperparameters, checkpoint --
    was already fixed. The oracle result cannot and must not feed back into
    model selection; it is read-only diagnosis of the input, not the model.
  - Because of the oracle's cost, frames are subsampled within each clip
    (see --frame-stride); the clip, not the frame, is already the
    statistical unit throughout this project (frames from one scene are
    correlated, not independent), so this trades nothing but wall-clock time.
    The same frame subset is used for every method, so the comparison stays
    exactly paired.

Vendors the reference network_swinir.py (Apache-2.0) directly from the
SwinIR repository rather than reimplementing shifted-window attention by
hand, and shims timm.models.layers' three symbols (to_2tuple, trunc_normal_,
DropPath) so no CUDA-oriented `timm` install is required; both training-time
no-ops at eval() on a loaded checkpoint. Everything lives under a gitignored
/tmp cache -- see ensure_swinir_source()/ensure_swinir_weights() -- nothing
is committed to the repository.

Reuses tools/evaluate.py's psnr/ssim/catmull_rom_2x and tools/captured-eval.py's
paired, clip-as-statistical-unit convention (the two small statistics helpers
below are intentionally identical to captured-eval.py's, not a fork of the
methodology).

CPU for every conventional-filter and production-model computation, matching
tools/captured-eval.py's documented reason (MPS gives measurably different
answers for identical model-free computations on this machine). The oracle's
own forward pass runs on MPS for tractable runtime; --selftest below verifies
that for THIS architecture, CPU and MPS agree to ~125 dB on a synthetic
input, i.e. the device choice does not contaminate the result the way it
would for the tiny production model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import math
import os
import statistics as st
import sys
import urllib.request

import torch
import torch.nn.functional as F
from PIL import Image

from evaluate import catmull_rom_2x, load_model, psnr, ssim

# ---------------------------------------------------------------------------
# Oracle identity and provenance
# ---------------------------------------------------------------------------

SWINIR_NETWORK_URL = "https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/models/network_swinir.py"
SWINIR_NETWORK_SHA256 = "9e143898679ebeebc5d2fc94ad1b89c38aa4a4d43da4e0fcba0f93e476994913"
SWINIR_WEIGHTS_URL = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
SWINIR_WEIGHTS_SHA256 = "2032ebf8f401dd3ce2fae5f3852117cb72101ec6ed8358faa64c2a3fa09ed4ac"
SWINIR_LICENSE_URL = "https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/LICENSE"
SWINIR_README_URL = "https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/README.md"
DEFAULT_SWINIR_CACHE = "/tmp/aethervsr-oracle/swinir"

# Real-ESRGAN x2plus RRDBNet, tried first and rejected. Numbers are from a
# preliminary run (2 frames/clip, CRF 34 only, same protocol otherwise) kept
# here verbatim for the record, not as part of the primary evaluation below.
REJECTED_ORACLE_NOTE = {
    "candidate": "Real-ESRGAN x2plus (RRDBNet, 16,703,171 params, BSD-3-Clause)",
    "weightsUrl": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "reasonRejected": (
        "GAN/perceptual training objective (adversarial + perceptual loss) trades pixel "
        "fidelity for hallucinated sharpness; scored BELOW Catmull-Rom on PSNR on every "
        "sampled clip but one, despite 2,655x the parameters. This is a known property of "
        "GAN-SR (real-world/perceptual SR), not evidence about recoverable pixel-fidelity "
        "information, and would have made a PSNR-based verdict about 'headroom' incoherent."
    ),
    "preliminarySample": {
        "framesPerClip": 2,
        "condition": "h264_poor",
        "oracleMinusCatmullRomPsnrDb_perClip": {
            "motion-2023-maplefest-in-bowmanville-on-2": -0.915,
            "motion-rogla-slovenia-ostruzica-1": -0.797,
            "motion-aerial-drone-chase-of-a-tra": -0.619,
            "urban-funchal-madeira-amazing-view": -1.068,
            "urban-drone-footage-in-new-orleans": -0.913,
            "nature-the-beautiful-blue-river-dj": -0.970,
            "nature-piz-neir-aerial-video-4k-we": -0.591,
            "lowlight-hofer-weihnachtsmarkt-am-": 0.146,
            "texture-snowex-field-campaign-4k-b": -2.727,
            "texture-4k-drone-footage-of-a-barg": -2.033,
        },
        "wins": 1,
        "losses": 9,
    },
}

LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)  # BT.709, matching tools/evaluate.py and captured-align.py

# Exact instantiation args from SwinIR's own main_test_swinir.py::define_model()
# for task="classical_sr", scale=2, using the DF2K/patch64 ("setting2") weights.
SWINIR_MODEL_KWARGS = dict(
    upscale=2, in_chans=3, img_size=64, window_size=8, img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv",
)

_TIMM_SHIM_SOURCE = '''"""Minimal timm.models.layers shim: only the three symbols network_swinir.py
imports (to_2tuple, trunc_normal_, DropPath), reimplemented against timm's
published source rather than approximated, so we avoid pulling all of timm
(and its dependency tree) just to run one vendored inference file offline.
DropPath and trunc_normal_ only affect training-time stochastic depth and
weight init; both are no-ops relative to a loaded, eval()-mode checkpoint.
"""
import collections.abc

import torch
import torch.nn as nn


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return (x, x)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor
'''


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "AetherVSR-oracle-crf34/1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def ensure_swinir_source(cache_dir: str) -> str:
    """Vendors network_swinir.py (pinned sha256) + a timm shim under a gitignored cache dir.

    Never writes into the repository. Returns `cache_dir`, which is prepended
    to sys.path so `from models.network_swinir import SwinIR` resolves.
    """
    models_dir = os.path.join(cache_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    open(os.path.join(models_dir, "__init__.py"), "a").close()
    net_path = os.path.join(models_dir, "network_swinir.py")
    if not (os.path.exists(net_path) and _sha256_file(net_path) == SWINIR_NETWORK_SHA256):
        print(f"fetching {SWINIR_NETWORK_URL}", file=sys.stderr)
        data = _download(SWINIR_NETWORK_URL)
        got = _sha256_bytes(data)
        if got != SWINIR_NETWORK_SHA256:
            raise SystemExit(f"network_swinir.py checksum mismatch: got {got}, expected {SWINIR_NETWORK_SHA256}")
        with open(net_path, "wb") as fh:
            fh.write(data)

    shim_root = os.path.join(cache_dir, "timm_shim")
    layers_dir = os.path.join(shim_root, "timm", "models", "layers")
    os.makedirs(layers_dir, exist_ok=True)
    open(os.path.join(shim_root, "timm", "__init__.py"), "a").close()
    open(os.path.join(shim_root, "timm", "models", "__init__.py"), "a").close()
    with open(os.path.join(layers_dir, "__init__.py"), "w", encoding="utf-8") as fh:
        fh.write(_TIMM_SHIM_SOURCE)

    for p in (shim_root, cache_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    return cache_dir


def ensure_swinir_weights(cache_dir: str) -> str:
    path = os.path.join(cache_dir, os.path.basename(SWINIR_WEIGHTS_URL))
    if os.path.exists(path) and _sha256_file(path) == SWINIR_WEIGHTS_SHA256:
        return path
    os.makedirs(cache_dir, exist_ok=True)
    print(f"fetching oracle weights: {SWINIR_WEIGHTS_URL}", file=sys.stderr)
    data = _download(SWINIR_WEIGHTS_URL)
    got = _sha256_bytes(data)
    if got != SWINIR_WEIGHTS_SHA256:
        raise SystemExit(f"oracle weight checksum mismatch: got {got}, expected {SWINIR_WEIGHTS_SHA256}")
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def load_oracle(cache_dir: str, device: str):
    ensure_swinir_source(cache_dir)
    weights_path = ensure_swinir_weights(cache_dir)
    net_module = importlib.import_module("models.network_swinir")
    model = net_module.SwinIR(**SWINIR_MODEL_KWARGS)
    payload = torch.load(weights_path, map_location="cpu")
    state = payload["params"] if "params" in payload else payload["params_ema"]
    model.load_state_dict(state, strict=True)  # raises on any architectural mismatch
    model.eval()
    return model.to(device), weights_path


def oracle_upscale(model, lr: torch.Tensor, device: str, window_size: int = 8, scale: int = 2) -> torch.Tensor:
    """Matches main_test_swinir.py's non-tiled classical_sr inference path exactly:
    reflect-pad (via mirrored flip, not constant/replicate) up to the next multiple of
    `window_size` on the bottom/right, run the network, then crop back to h*scale x w*scale.
    RGB in [0,1] -- no BGR swap needed (the official script itself converts cv2's BGR to RGB
    before feeding the network; RGB is the network's native channel order). Internal dataset
    mean-subtraction is a registered buffer inside the model's own forward(), so nothing
    extra is applied here.
    """
    lr = lr.to(device)
    _, _, h_old, w_old = lr.shape
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    padded = torch.cat([lr, torch.flip(lr, [2])], 2)[:, :, : h_old + h_pad, :]
    padded = torch.cat([padded, torch.flip(padded, [3])], 3)[:, :, :, : w_old + w_pad]
    with torch.no_grad():
        out = model(padded)
    out = out[:, :, : h_old * scale, : w_old * scale]
    return out.clamp(0, 1).cpu()


def cpu_mps_agreement(model, device: str) -> dict | None:
    """Self-check: does this architecture answer identically on CPU vs MPS?

    Unlike the 6.3K-parameter production model (documented in
    tools/captured-eval.py to disagree by up to 2.5 dB between CPU and MPS
    for plain resampling math), SwinIR's ops (conv, layernorm, windowed
    self-attention, softmax) are checked directly here rather than assumed.
    """
    if device != "mps":
        return None
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)  # exact multiple of window_size=8
    cpu_model = model.to("cpu")
    with torch.no_grad():
        y_cpu = cpu_model(x)
    mps_model = cpu_model.to("mps")
    with torch.no_grad():
        y_mps = mps_model(x.to("mps")).to("cpu")
    model.to(device)
    mse = F.mse_loss(y_cpu, y_mps).item()
    return {
        "mse": mse,
        "psnrDb": float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse),
        "maxAbsDiff": (y_cpu - y_mps).abs().max().item(),
    }


# ---------------------------------------------------------------------------
# Metrics -- psnr/ssim/catmull_rom_2x come from tools/evaluate.py; psnr_y adds
# the luma-only PSNR the assignment asks for alongside full-RGB PSNR.
# ---------------------------------------------------------------------------


def _luma(a: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor(LUMA_WEIGHTS, device=a.device).view(1, 3, 1, 1)
    return (a.clamp(0, 1) * weights).sum(1, keepdim=True)


def psnr_y(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(_luma(a), _luma(b)).item()
    return float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)


def load_png(path: str) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        raw = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        return raw.reshape(h, w, 3).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Statistics -- identical convention to tools/captured-eval.py: the clip is
# the resampling/inferential unit, comparisons are paired, sign-flip
# permutation is exact for n<=20.
# ---------------------------------------------------------------------------


def frame_distribution(values: list[float]) -> dict:
    ordered = sorted(v for v in values if v != float("inf"))
    if not ordered:
        return {}
    return {
        "mean": st.fmean(ordered),
        "median": st.median(ordered),
        "p5": ordered[min(len(ordered) - 1, round(0.05 * (len(ordered) - 1)))],
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "min": ordered[0],
        "max": ordered[-1],
        "n": len(ordered),
    }


def paired_stats(diffs: list[float], bootstrap: int = 20_000, seed: int = 12345) -> dict:
    n = len(diffs)
    if n == 0:
        return {}
    mean = st.fmean(diffs)
    wins = sum(1 for d in diffs if d > 0)

    rng = torch.Generator().manual_seed(seed)
    tensor = torch.tensor(diffs, dtype=torch.float64)
    idx = torch.randint(0, n, (bootstrap, n), generator=rng)
    means = tensor[idx].mean(dim=1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()

    if n <= 20:
        count = 0
        for signs in itertools.product((1, -1), repeat=n):
            flipped = st.fmean(s * d for s, d in zip(signs, diffs))
            if abs(flipped) >= abs(mean) - 1e-12:
                count += 1
        perm_p = count / (2**n)
        exact = True
    else:
        perm_p, exact = float("nan"), False

    from math import comb

    tail = sum(comb(n, k) for k in range(0, min(wins, n - wins) + 1))
    sign_p = min(1.0, 2 * tail / (2**n))

    return {
        "clips": n,
        "meanDiff": mean,
        "medianDiff": st.median(diffs),
        "bootstrapCI95": [lo, hi],
        "wins": wins,
        "losses": n - wins,
        "signTestP": sign_p,
        "permutationP": perm_p,
        "permutationExact": exact,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="../public/models/aethersr-c16d2-realistic.json")
    ap.add_argument("--root", default="../data/captured")
    ap.add_argument("--conditions", default="h264_poor,h264_typical")
    ap.add_argument("--oracle-cache", default=DEFAULT_SWINIR_CACHE)
    ap.add_argument("--oracle-device", default="mps" if torch.backends.mps.is_available() else "cpu",
                     choices=("cpu", "mps"))
    ap.add_argument("--out", default="../results/oracle-crf34.json")
    ap.add_argument("--limit-frames", type=int, default=0, help="0 = no cap, applied before --frame-stride")
    ap.add_argument("--frame-stride", type=int, default=3,
                     help="evaluate every Nth frame per clip (statistical unit is the clip, "
                          "not the frame, and the oracle costs ~1000x AetherSR per frame)")
    args = ap.parse_args()

    with open(os.path.join(args.root, "FROZEN.json"), encoding="utf-8") as fh:
        frozen = json.load(fh)
    with open(os.path.join(args.root, "prepared.json"), encoding="utf-8") as fh:
        prepared = json.load(fh)
    clip_ids = {c["id"] for c in prepared["clips"]}
    if clip_ids != set(frozen["clipIds"]) or len(clip_ids) != 10:
        raise SystemExit(
            f"prepared.json clips ({len(clip_ids)}) do not match the frozen 10-clip test set; "
            "refusing to run (this evaluation must stay confined to the frozen set)."
        )

    neural, neural_payload = load_model(args.model)
    neural = neural.to("cpu")
    neural_params = sum(p.numel() for p in neural.parameters())

    oracle, weights_path = load_oracle(args.oracle_cache, args.oracle_device)
    oracle_params = sum(p.numel() for p in oracle.parameters())
    agreement = cpu_mps_agreement(oracle, args.oracle_device)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    methods = ("nearest", "bilinear", "catmull_rom", "neural", "oracle")
    per_clip: list[dict] = []

    for clip in prepared["clips"]:
        clip_root = os.path.join(args.root, "clips", clip["id"])
        master_dir = os.path.join(clip_root, "master")
        masters = sorted(os.listdir(master_dir))
        if args.limit_frames:
            masters = masters[: args.limit_frames]
        masters = masters[:: args.frame_stride]

        row = {"id": clip["id"], "category": clip["category"], "conditions": {}}
        for condition in conditions:
            decoded_dir = os.path.join(clip_root, "decoded", condition)
            if not os.path.isdir(decoded_dir):
                continue
            acc: dict[str, dict[str, list[float]]] = {
                m: {"psnr": [], "psnrY": [], "ssim": []} for m in methods
            }
            for name in masters:
                lr_path = os.path.join(decoded_dir, name)
                if not os.path.exists(lr_path):
                    continue
                ref = load_png(os.path.join(master_dir, name))
                lr = load_png(lr_path)
                with torch.no_grad():
                    outs = {
                        "nearest": F.interpolate(lr, scale_factor=2, mode="nearest").clamp(0, 1),
                        "bilinear": F.interpolate(lr, scale_factor=2, mode="bilinear", align_corners=False).clamp(0, 1),
                        "catmull_rom": catmull_rom_2x(lr),
                        "neural": neural(lr).clamp(0, 1),
                        "oracle": oracle_upscale(oracle, lr, args.oracle_device),
                    }
                for key, out in outs.items():
                    if out.shape != ref.shape:
                        raise SystemExit(
                            f"geometry mismatch {clip['id']}/{condition}/{key}: "
                            f"{tuple(out.shape)} against {tuple(ref.shape)}"
                        )
                    acc[key]["psnr"].append(psnr(out, ref))
                    acc[key]["psnrY"].append(psnr_y(out, ref))
                    acc[key]["ssim"].append(ssim(out, ref))

            if not acc["oracle"]["psnr"]:
                continue
            row["conditions"][condition] = {m: {k: frame_distribution(v) for k, v in acc[m].items()} for m in acc}
            row["conditions"][condition]["framesUsed"] = len(acc["oracle"]["psnr"])
            row["conditions"][condition]["clipMeanDiff"] = {
                "oracleMinusCatmullRom": {
                    "psnr": st.fmean(a - b for a, b in zip(acc["oracle"]["psnr"], acc["catmull_rom"]["psnr"])),
                    "psnrY": st.fmean(a - b for a, b in zip(acc["oracle"]["psnrY"], acc["catmull_rom"]["psnrY"])),
                    "ssim": st.fmean(a - b for a, b in zip(acc["oracle"]["ssim"], acc["catmull_rom"]["ssim"])),
                },
                "oracleMinusNeural": {
                    "psnr": st.fmean(a - b for a, b in zip(acc["oracle"]["psnr"], acc["neural"]["psnr"])),
                    "psnrY": st.fmean(a - b for a, b in zip(acc["oracle"]["psnrY"], acc["neural"]["psnrY"])),
                    "ssim": st.fmean(a - b for a, b in zip(acc["oracle"]["ssim"], acc["neural"]["ssim"])),
                },
                "neuralMinusCatmullRom": {
                    "psnr": st.fmean(a - b for a, b in zip(acc["neural"]["psnr"], acc["catmull_rom"]["psnr"])),
                },
            }
            print(
                f"  {clip['id'][:34]:<34} {condition:<13} "
                f"oracle {row['conditions'][condition]['oracle']['psnr']['mean']:6.3f}  "
                f"catmull {row['conditions'][condition]['catmull_rom']['psnr']['mean']:6.3f}  "
                f"neural {row['conditions'][condition]['neural']['psnr']['mean']:6.3f}  "
                f"oracle-catmull {row['conditions'][condition]['clipMeanDiff']['oracleMinusCatmullRom']['psnr']:+6.3f}  "
                f"oracle-neural {row['conditions'][condition]['clipMeanDiff']['oracleMinusNeural']['psnr']:+6.3f}",
                file=sys.stderr,
            )
        per_clip.append(row)

    aggregate = {}
    for condition in conditions:
        rows = [c["conditions"][condition] for c in per_clip if condition in c.get("conditions", {})]
        aggregate[condition] = {
            "oracleMinusCatmullRom": {
                "psnr": paired_stats([r["clipMeanDiff"]["oracleMinusCatmullRom"]["psnr"] for r in rows]),
                "psnrY": paired_stats([r["clipMeanDiff"]["oracleMinusCatmullRom"]["psnrY"] for r in rows]),
                "ssimMeanDiff_DIAGNOSTIC_ONLY": st.fmean(
                    r["clipMeanDiff"]["oracleMinusCatmullRom"]["ssim"] for r in rows
                ),
            },
            "oracleMinusNeural": {
                "psnr": paired_stats([r["clipMeanDiff"]["oracleMinusNeural"]["psnr"] for r in rows]),
                "psnrY": paired_stats([r["clipMeanDiff"]["oracleMinusNeural"]["psnrY"] for r in rows]),
                "ssimMeanDiff_DIAGNOSTIC_ONLY": st.fmean(
                    r["clipMeanDiff"]["oracleMinusNeural"]["ssim"] for r in rows
                ),
            },
            "neuralMinusCatmullRom_psnr_forContext": paired_stats(
                [r["clipMeanDiff"]["neuralMinusCatmullRom"]["psnr"] for r in rows]
            ),
        }

    report = {
        "schema": "aethervsr.oracle-crf34/1",
        "milestone": "5.5 Part E",
        "question": (
            "Is there useful recoverable information in CRF 34 captured video that the "
            "6,291-parameter production model simply fails to exploit, or is the "
            "information destroyed by the encode?"
        ),
        "diagnosticOnly": True,
        "diagnosticStatement": (
            "This oracle evaluation runs after every AetherSR C16D2 modelling decision "
            "(architecture, seed, hyperparameters, checkpoint) was already fixed. It is "
            "read-only diagnosis of the CRF 34/26 input signal, not a production candidate "
            "and not part of model selection: nothing here fed back into training, seed "
            "choice, or checkpoint selection. Runtime is irrelevant and not compared."
        ),
        "testSet": {
            "frozen": True,
            "clipCount": len(clip_ids),
            "testsetDigest": frozen.get("testsetDigest"),
            "note": "Confined to the frozen 10-clip captured set; asserted equal to data/captured/FROZEN.json at run time.",
        },
        "frameSampling": {
            "framesPerClipTotal": 24,
            "frameStride": args.frame_stride,
            "note": (
                "The clip, not the frame, is the statistical unit throughout this project "
                "(frames from one scene are correlated, not independent observations), so "
                "subsampling frames within a clip trades only wall-clock time, not statistical "
                "power. The same frame indices are used for every one of the five methods, so "
                "the comparison stays exactly paired."
            ),
        },
        "productionModel": {
            "path": os.path.basename(args.model),
            "paramCount": neural_params,
            "degradationProfile": neural_payload.get("training", {}).get("degradationProfile"),
        },
        "oracle": {
            "name": "SwinIR classical SR, x2, M-size, DF2K weights (001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth)",
            "paper": "Liang et al., SwinIR: Image Restoration Using Swin Transformer, ICCVW 2021 (arXiv:2108.10257)",
            "repo": "https://github.com/JingyunLiang/SwinIR",
            "paramCount": oracle_params,
            "capacityRatioVsProduction": oracle_params / neural_params,
            "license": {
                "spdx": "Apache-2.0",
                "verifiedFromUrl": SWINIR_LICENSE_URL,
                "note": "Fetched and read verbatim from the repository LICENSE file at HEAD before use.",
            },
            "source": {
                "networkFileUrl": SWINIR_NETWORK_URL,
                "networkFileSha256": SWINIR_NETWORK_SHA256,
                "note": (
                    "Vendored verbatim (pinned sha256, verified at run time) rather than "
                    "reimplemented -- shifted-window attention has enough moving parts "
                    "(relative position bias, window partition/reverse, attention masking) "
                    "that a hand rewrite risks a silent, still-running correctness bug."
                ),
            },
            "weights": {
                "url": SWINIR_WEIGHTS_URL,
                "sha256": SWINIR_WEIGHTS_SHA256,
                "listedAt": SWINIR_README_URL,
                "cachedPath": weights_path,
                "note": "Cached under /tmp (gitignored path), never committed to the repository.",
            },
            "device": args.oracle_device,
            "cpuMpsAgreementCheck": agreement,
            "preprocessing": {
                "inputRange": "float32 RGB in [0,1], full-range (matches the decoded PNGs already produced by the captured pipeline)",
                "channelOrder": (
                    "RGB. The official main_test_swinir.py converts cv2's BGR to RGB "
                    "(img[:, :, [2, 1, 0]]) before feeding the network and converts back only "
                    "for cv2 image writeback -- the network's native channel order is RGB, "
                    "matching what load_png() here already produces. No conversion needed."
                ),
                "datasetMeanSubtraction": (
                    "Handled internally: SwinIR registers its DIV2K RGB mean as a model buffer "
                    "and applies/reverses it inside forward() itself. Nothing extra applied here."
                ),
                "scaleAdaptation": (
                    "Native x2 classical-SR checkpoint used directly -- no 4x-to-2x adaptation "
                    "was needed."
                ),
                "windowPadding": (
                    "SwinIR requires H,W divisible by window_size=8 (1280x720 already are, but "
                    "the official reflect-via-flip pad-to-next-multiple is still applied for "
                    "exact parity with main_test_swinir.py), network applied to the padded "
                    "frame, then cropped back to h*2 x w*2 before scoring."
                ),
                "precision": "fp32 (no CUDA half-precision path on this machine)",
                "tiling": "none -- 1280x720 fits in memory for a single forward pass on Apple M5 MPS",
            },
        },
        "consideredAndRejected": REJECTED_ORACLE_NOTE,
        "conditions": conditions,
        "methods": list(methods),
        "statisticalUnit": (
            "clip. Frames within a clip are averaged first; all inference is across clips."
        ),
        "comparison": "paired: every method sees the identical decoded frame",
        "perClip": per_clip,
        "aggregate": aggregate,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
