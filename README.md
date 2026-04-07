# ONID: Inverting DINOv3 Embeddings Through FLUX VAE Latents

This repository implements a reconstruction experiment for the pipeline:

`image -> DINOv3 -> learned conv decoder -> FLUX VAE latents -> FLUX VAE decode -> image`

The target stack follows the requested gated Hugging Face assets:

- `facebook/dinov3-vit7b16-pretrain-lvd1689m`
- `black-forest-labs/FLUX.2-klein-9B` (VAE component only)
- `ILSVRC/imagenet-1k`

## Important constraints

- All three assets are gated. The current Hugging Face account must have accepted access terms.
- `dense` DINO patch-token caching at ImageNet-1k scale is multi-terabyte. The scalable path is `pooled` embeddings or online DINO extraction.
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

## Experiment variants

- `pooled`: regress FLUX VAE latents from DINO pooled output. This is the scalable baseline for ImageNet-1k.
- `dense`: regress FLUX VAE latents from DINO patch tokens reshaped as a spatial tensor. This preserves dense information but is too large to cache across full ImageNet-1k without aggressive compression or online extraction.

## Expected storage

At `256x256` resolution:

- FLUX VAE latents are small enough for full-dataset caching.
- DINO pooled embeddings are modest and fit comfortably on disk.
- DINO dense patch-token tensors dominate storage and are not practical to cache for the full dataset on a 2 TB workspace.
