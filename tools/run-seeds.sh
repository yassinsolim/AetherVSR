#!/usr/bin/env bash
# Train N seeds of one configuration, identical in every respect except the seed.
#
# Milestone 4 shipped a single seed, and a 1% corpus change then moved the
# synthetic score by 1.1 dB - which told us the seed-to-seed spread was unknown,
# not that the corpus mattered. This exists so that spread is measured before
# any two models are compared.
#
# Usage: tools/run-seeds.sh <profile> <outdir> [seeds...]
set -euo pipefail

PY="${AETHER_PY:-/tmp/aethertrain/bin/python}"
PROFILE="${1:?usage: run-seeds.sh <profile> <outdir> [seeds...]}"
OUTDIR="${2:?usage: run-seeds.sh <profile> <outdir> [seeds...]}"
shift 2
SEEDS=("${@:-1 2 3 4 5}")
if [ "$#" -eq 0 ]; then SEEDS=(1 2 3 4 5); fi

mkdir -p "$OUTDIR"

# Every knob fixed except --seed. Matching the Milestone 4 shipped
# configuration, so the variance measured here is the variance that model was
# drawn from.
for seed in "${SEEDS[@]}"; do
  out="$OUTDIR/seed${seed}.json"
  if [ -f "$out" ]; then
    echo "== seed $seed already trained, skipping =="
    continue
  fi
  echo "== training seed $seed (profile $PROFILE) =="
  "$PY" tools/train.py \
    --corpus data/corpus \
    --split data/splits/corpus-v1.json \
    --degradation "$PROFILE" \
    --out "$out" \
    --channels 16 \
    --depth 2 \
    --patch 128 \
    --per-image 6 \
    --epochs 60 \
    --batch 32 \
    --lr 3e-3 \
    --seed "$seed"
done

echo "== all seeds done: $OUTDIR =="
