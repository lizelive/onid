# ONID: Inverting DINOv3 Embeddings Through FLUX VAE Latents

This repository implements a reconstruction experiment for the pipeline:

`image -> DINOv3 -> learned conv decoder -> FLUX VAE latents -> FLUX VAE decode -> image`

The target stack follows the requested gated Hugging Face assets:

- `facebook/dinov3-convnext-large-pretrain-lvd1689m`
- `black-forest-labs/FLUX.2-klein-9B` (VAE component only)
- `ILSVRC/imagenet-1k`

## Important constraints

- All three assets are gated. The current Hugging Face account must have accepted access terms.
- `dense` DINO embedding caching at ImageNet-1k scale is multi-terabyte. The scalable dense path is fully online supervision: compute DINO embeddings and FLUX latents on demand during training.
- The included smoke experiment is designed to run end to end on one RTX 4090 with CPU offload available.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

## Project layout

- `src/onid/pairs.py`: ImageNet loading plus paired DINO and VAE latent precomputation
- `src/onid/models.py`: latent decoder architectures
- `src/onid/train.py`: training loop, validation, and sample reconstruction export
- `src/onid/smoke.py`: minimal end-to-end experiment runner
- `paper/report.md`: paper draft with method, limitations, and results section

## Commands

Precompute a smoke subset with pooled embeddings:

```bash
python -m onid.pairs \
  --output-dir artifacts/smoke_pairs \
  --split train \
  --embedding-kind pooled \
  --max-samples 128 \
  --shard-size 32 \
  --resolution 256
```

Precompute validation pairs:

```bash
python -m onid.pairs \
  --output-dir artifacts/smoke_pairs_val \
  --split validation \
  --embedding-kind pooled \
  --max-samples 32 \
  --shard-size 32 \
  --resolution 256
```

Train the decoder:

```bash
python -m onid.train \
  --train-dir artifacts/smoke_pairs \
  --val-dir artifacts/smoke_pairs_val \
  --output-dir outputs/smoke_run \
  --embedding-kind pooled \
  --epochs 3 \
  --batch-size 8
```

Run the full smoke workflow:

```bash
python -m onid.smoke --output-root outputs/smoke_e2e
```

Train dense decoders directly from ImageNet with online DINO embeddings, online FLUX latent targets, automatic batch sizing, and resumable checkpoints:

```bash
python -m onid.train \
  --train-split train \
  --val-split validation \
  --resolution 256 \
  --output-dir outputs/full_dense_online/dense-residual \
  --embedding-kind dense \
  --decoder-architecture dense-residual \
  --epochs 3 \
  --auto-batch-size \
  --max-batch-size 256 \
  --compile \
  --compile-mode max-autotune \
  --checkpoint-interval-steps 500 \
  --online-shuffle-buffer 128 \
  --resume
```

Launch the full latent-cache plus dense architecture sweep in a detached process:

```bash
nohup bash scripts/run_full_dense_experiments.sh > outputs/full_dense_online/pipeline.log 2>&1 < /dev/null &
```

## Experiment variants

- `pooled`: regress FLUX VAE latents from DINO pooled output. This is the scalable baseline for ImageNet-1k.
- `dense`: regress FLUX VAE latents from DINO patch tokens reshaped as a spatial tensor. Full-scale training should stream ImageNet directly and compute both DINO embeddings and FLUX latents online.

Dense decoder architecture options:

- `dense-residual`: constant-width residual upsampling baseline.
- `dense-pyramid`: wider early stages with a tapered channel schedule.
- `dense-bottleneck`: higher-width bottleneck residual blocks for a stronger capacity baseline.

## Expected storage

At `256x256` resolution:

- FLUX VAE latents are cheap enough to compute online on a 4090, which avoids writing dataset-scale artifacts.
- DINO pooled embeddings are modest and fit comfortably on disk.
- DINO dense patch-token tensors dominate storage and are not practical to cache for the full dataset on a 2 TB workspace.

## Recovery

- `onid.train --resume` writes `last.pt` and `state.json` during training, so interrupted long runs continue from the latest checkpointed step without rebuilding any dataset cache.
- Online ImageNet validation now defaults to the fitted train batch size and prints its own progress bar, so long validation passes stay visible in `pipeline.log`.

