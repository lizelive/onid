#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/full_dense_online

for architecture in dense-residual dense-pyramid dense-bottleneck; do
  .venv/bin/python -m onid.train \
    --train-split train \
    --val-split validation \
    --resolution 256 \
    --train-samples 0 \
    --val-samples 0 \
    --output-dir "outputs/full_dense_online/${architecture}" \
    --embedding-kind dense \
    --decoder-architecture "$architecture" \
    --epochs 3 \
    --batch-size 8 \
    --auto-batch-size \
    --max-batch-size 256 \
    --eval-batch-size 8 \
    --checkpoint-interval-steps 500 \
    --online-shuffle-buffer 128 \
    --resume
done