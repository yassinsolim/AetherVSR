#!/usr/bin/env python3
"""Export deterministic test vectors from the reference implementation.

The WGSL graph is checked against PyTorch, not against itself. This writes a
small JSON file holding a fixed input and the reference's per-stage outputs, so
the browser can compare stage by stage and say *which* stage diverged rather
than only that the picture is wrong.

    python3 tools/export-golden.py --model public/models/aethersr-c16d2.json \
        --out public/models/golden-c16d2.json

Deliberately small - 24x16 by default. Enough pixels to exercise interior
arithmetic and all four borders, small enough to embed and to compare exhaustively.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from aethersr import AetherSR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--height", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    with open(args.model) as fh:
        doc = json.load(fh)

    model = AetherSR(channels=doc["features"], depth=doc["depth"])
    state = {}
    w = doc["weights"]
    state["stem.weight"] = torch.tensor(w["stem.weight"]).reshape(doc["features"], 3, 5, 5)
    state["stem.bias"] = torch.tensor(w["stem.bias"])
    for i in range(doc["depth"]):
        state[f"body.{i}.weight"] = torch.tensor(w[f"body.{i}.weight"]).reshape(
            doc["features"], doc["features"], 3, 3
        )
        state[f"body.{i}.bias"] = torch.tensor(w[f"body.{i}.bias"])
    state["head.weight"] = torch.tensor(w["head.weight"]).reshape(3, doc["features"], 3, 3)
    state["head.bias"] = torch.tensor(w["head.bias"])
    model.load_state_dict(state)
    model.eval()

    # Deterministic input covering [0,1], including exact 0 and 1 so the clamp
    # and the border padding are both exercised.
    g = torch.Generator().manual_seed(args.seed)
    lr = torch.rand(1, 3, args.height, args.width, generator=g)
    lr[0, :, 0, 0] = 0.0
    lr[0, :, -1, -1] = 1.0

    with torch.no_grad():
        # Intermediates are recomputed here so each can be exported, but the
        # *output* comes from the model's own forward. Re-deriving it by hand
        # is how the global residual went missing from an earlier export, which
        # made the WGSL graph look wrong when it was the reference that was.
        stem = torch.tanh(model.stem(lr))
        stages = {"stem": stem}
        x = stem
        for i, conv in enumerate(model.body):
            x = torch.tanh(conv(x))
            stages[f"body.{i}"] = x
        out = model(lr)

        # Guard: the hand-computed trunk must agree with the model's own, or the
        # exported intermediates describe a different network from the output.
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = F.interpolate(lr, scale_factor=2, mode="nearest")
        recomputed = (model.head(up) + skip).clamp(0.0, 1.0)
        drift = (recomputed - out).abs().max().item()
        if drift > 1e-6:
            raise SystemExit(f"exporter disagrees with model.forward by {drift:.3e}")

    payload = {
        "model": os.path.basename(args.model),
        "modelSha256": doc.get("sha256", ""),
        "features": doc["features"],
        "depth": doc["depth"],
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "note": (
            "Reference activations from tools/aethersr.py in float32. Stage tensors are "
            "planar [channel][y][x]; the output is interleaved RGB [y][x][c] after clamping."
        ),
        # Planar, matching how the verifier unpacks the GPU's grouped layout.
        "input": lr.reshape(-1).tolist(),
        "stages": {name: t.reshape(-1).tolist() for name, t in stages.items()},
        "output": out[0].permute(1, 2, 0).reshape(-1).tolist(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1024:.1f} kB)")
    print(f"  input {tuple(lr.shape)}  stages {list(stages)}  output {tuple(out.shape)}")
    print(f"  output range [{out.min():.4f}, {out.max():.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
