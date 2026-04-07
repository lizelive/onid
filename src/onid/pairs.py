from __future__ import annotations

import argparse
import bisect
import json
import random
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import center_crop, normalize, resize, to_tensor
from tqdm import tqdm

from onid.hf_models import (
    DINO_MODEL_NAME,
    FLUX_MODEL_NAME,
    extract_embedding,
    load_dino_encoder,
    load_flux_vae,
    module_device,
    preferred_dtype,
)


IMAGENET_DATASET_NAME = "ILSVRC/imagenet-1k"


@torch.inference_mode()
def preprocess_image(image: Image.Image, resolution: int) -> torch.Tensor:
    image = image.convert("RGB")
    image = resize(image, resolution, interpolation=InterpolationMode.BICUBIC, antialias=True)
    image = center_crop(image, [resolution, resolution])
    return to_tensor(image)


@torch.inference_mode()
def encode_sample(
    image: Image.Image,
    resolution: int,
    processor: Any,
    dino_model: torch.nn.Module,
    vae: torch.nn.Module,
    embedding_kind: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_tensor = preprocess_image(image, resolution)

    dino_tensor = normalize(
        image_tensor,
        mean=processor.image_mean,
        std=processor.image_std,
    ).unsqueeze(0)
    dino_tensor = dino_tensor.to(module_device(dino_model), dtype=dtype)
    dino_outputs = dino_model(pixel_values=dino_tensor)
    embedding = extract_embedding(dino_outputs, dino_model, embedding_kind)

    vae_pixels = ((image_tensor * 2.0) - 1.0).unsqueeze(0)
    vae_pixels = vae_pixels.to(module_device(vae), dtype=dtype)
    latent_dist = vae.encode(vae_pixels).latent_dist
    scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
    latents = latent_dist.mean * scaling_factor

    return embedding.squeeze(0).cpu().to(torch.bfloat16), latents.squeeze(0).cpu().to(torch.bfloat16)


@torch.inference_mode()
def encode_batch(
    images: list[Image.Image],
    resolution: int,
    processor: Any,
    dino_model: torch.nn.Module,
    vae: torch.nn.Module,
    embedding_kind: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_tensors = [preprocess_image(image, resolution) for image in images]
    pixel_batch = torch.stack(image_tensors, dim=0)

    mean = torch.tensor(processor.image_mean, dtype=pixel_batch.dtype).view(1, -1, 1, 1)
    std = torch.tensor(processor.image_std, dtype=pixel_batch.dtype).view(1, -1, 1, 1)
    dino_tensor = (pixel_batch - mean) / std
    dino_tensor = dino_tensor.to(module_device(dino_model), dtype=dtype)
    dino_outputs = dino_model(pixel_values=dino_tensor)
    embeddings = extract_embedding(dino_outputs, dino_model, embedding_kind)

    vae_pixels = ((pixel_batch * 2.0) - 1.0).to(module_device(vae), dtype=dtype)
    latent_dist = vae.encode(vae_pixels).latent_dist
    scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
    latents = latent_dist.mean * scaling_factor

    return embeddings.cpu().to(torch.bfloat16), latents.cpu().to(torch.bfloat16)


@torch.inference_mode()
def encode_latents_batch(
    images: list[Image.Image],
    resolution: int,
    vae: torch.nn.Module,
    dtype: torch.dtype,
) -> torch.Tensor:
    image_tensors = [preprocess_image(image, resolution) for image in images]
    pixel_batch = torch.stack(image_tensors, dim=0)
    vae_pixels = ((pixel_batch * 2.0) - 1.0).to(module_device(vae), dtype=dtype)
    latent_dist = vae.encode(vae_pixels).latent_dist
    scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
    latents = latent_dist.mean * scaling_factor
    return latents.cpu().to(torch.bfloat16)


def atomic_write_text(path: Path, content: str) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(content)
    temp_path.replace(path)


def save_shard(
    output_dir: Path,
    shard_index: int,
    tensors: dict[str, list[torch.Tensor]],
) -> dict[str, Any]:
    shard_name = f"shard-{shard_index:05d}.pt"
    shard_path = output_dir / shard_name
    shard_data = {name: torch.stack(values, dim=0) for name, values in tensors.items()}
    torch.save(shard_data, shard_path)
    first_key = next(iter(tensors))
    return {
        "file": shard_name,
        "count": len(tensors[first_key]),
    }


def iter_imagenet_samples(
    split: str,
    max_samples: int,
    shuffle: bool,
    seed: int,
    start_index: int = 0,
) -> Any:
    remote_files = [
        file_name
        for file_name in list_repo_files(IMAGENET_DATASET_NAME, repo_type="dataset")
        if file_name.startswith(f"data/{split}-") and file_name.endswith(".parquet")
    ]
    remote_files.sort()
    if shuffle:
        random.Random(seed).shuffle(remote_files)

    sample_limit = None if max_samples <= 0 else max_samples
    if sample_limit is not None and start_index >= sample_limit:
        return

    emitted = 0
    remaining_skip = start_index

    for remote_file in remote_files:
        local_file = hf_hub_download(
            repo_id=IMAGENET_DATASET_NAME,
            filename=remote_file,
            repo_type="dataset",
        )
        shard_dataset = load_dataset("parquet", data_files={"data": local_file}, split="data")
        if shuffle:
            shard_dataset = shard_dataset.shuffle(seed=seed)
        if remaining_skip >= len(shard_dataset):
            remaining_skip -= len(shard_dataset)
            continue

        start_row = remaining_skip
        remaining_skip = 0
        for row_index in range(start_row, len(shard_dataset)):
            sample = shard_dataset[row_index]
            yield sample
            emitted += 1
            total_seen = start_index + emitted
            if sample_limit is not None and total_seen >= sample_limit:
                return


def load_manifest(root_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(root_dir) / "manifest.json").read_text())


def iter_sharded_tensors(root_dir: str | Path, key: str) -> Any:
    manifest = load_manifest(root_dir)
    root_path = Path(root_dir)
    for shard in manifest["shards"]:
        shard_file = root_path / shard["file"]
        shard_data = torch.load(shard_file, map_location="cpu")
        for tensor in shard_data[key]:
            yield tensor.float()


def precompute_pairs(
    output_dir: str | Path,
    split: str,
    embedding_kind: str,
    resolution: int,
    max_samples: int,
    shard_size: int,
    encode_batch_size: int = 4,
    seed: int = 0,
    streaming: bool = True,
    shuffle: bool = False,
    cache_mode: str = "pairs",
    resume: bool = False,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dtype = preferred_dtype()
    processor = None
    dino_model = None
    if cache_mode == "pairs":
        processor, dino_model = load_dino_encoder(dtype=dtype)
    vae = load_flux_vae(dtype=dtype)

    embeddings: list[torch.Tensor] = []
    latents: list[torch.Tensor] = []
    manifest_path = output_path / "manifest.json"
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        shards = manifest["shards"]
        embedding_shape = manifest.get("embedding_shape")
        latent_shape = manifest.get("latent_shape")
    else:
        manifest = {
            "dataset": IMAGENET_DATASET_NAME,
            "split": split,
            "resolution": resolution,
            "cache_mode": cache_mode,
            "embedding_kind": embedding_kind if cache_mode == "pairs" else None,
            "embedding_shape": None,
            "latent_shape": None,
            "num_samples": 0,
            "dino_model": DINO_MODEL_NAME if cache_mode == "pairs" else None,
            "flux_model": FLUX_MODEL_NAME,
            "shards": [],
        }
        shards = manifest["shards"]
        embedding_shape = None
        latent_shape = None

    completed_samples = sum(shard["count"] for shard in shards)
    remaining_samples = max_samples if max_samples > 0 else None
    if remaining_samples is not None and completed_samples >= remaining_samples:
        return manifest_path

    image_batch: list[Image.Image] = []

    def flush_batch() -> None:
        nonlocal embedding_shape, latent_shape
        if not image_batch:
            return

        batch_embeddings = None
        if cache_mode == "pairs":
            batch_embeddings, batch_latents = encode_batch(
                images=image_batch,
                resolution=resolution,
                processor=processor,
                dino_model=dino_model,
                vae=vae,
                embedding_kind=embedding_kind,
                dtype=dtype,
            )
        else:
            batch_latents = encode_latents_batch(
                images=image_batch,
                resolution=resolution,
                vae=vae,
                dtype=dtype,
            )

        for sample_index, latent in enumerate(batch_latents):
            if batch_embeddings is not None:
                embedding = batch_embeddings[sample_index]
                embeddings.append(embedding)
            latents.append(latent)

            if embedding_shape is None and batch_embeddings is not None:
                embedding_shape = list(batch_embeddings[sample_index].shape)
                latent_shape = list(latent.shape)
            elif latent_shape is None:
                latent_shape = list(latent.shape)

            sample_count = len(latents) if cache_mode == "latents" else len(embeddings)
            if sample_count == shard_size:
                tensor_map: dict[str, list[torch.Tensor]] = {"latents": latents}
                if cache_mode == "pairs":
                    tensor_map["embedding"] = embeddings
                shards.append(save_shard(output_path, len(shards), tensor_map))
                if cache_mode == "pairs":
                    embeddings.clear()
                latents.clear()
                manifest["embedding_shape"] = embedding_shape
                manifest["latent_shape"] = latent_shape
                manifest["num_samples"] = sum(shard["count"] for shard in shards)
                atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            progress.update(1)
        image_batch.clear()

    progress_total = None if max_samples <= 0 else max_samples
    progress_label = cache_mode if cache_mode == "latents" else embedding_kind
    progress = tqdm(total=progress_total, desc=f"precompute:{split}:{progress_label}")
    if completed_samples:
        progress.update(completed_samples)

    for sample in iter_imagenet_samples(
        split=split,
        max_samples=max_samples,
        shuffle=shuffle,
        seed=seed,
        start_index=completed_samples,
    ):
        image_batch.append(sample["image"])
        if len(image_batch) == encode_batch_size:
            flush_batch()

    flush_batch()

    if embeddings:
        shards.append(save_shard(output_path, len(shards), {"embedding": embeddings, "latents": latents}))
    elif latents:
        shards.append(save_shard(output_path, len(shards), {"latents": latents}))

    progress.close()

    manifest["embedding_shape"] = embedding_shape
    manifest["latent_shape"] = latent_shape
    manifest["num_samples"] = sum(shard["count"] for shard in shards)
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
    return manifest_path


class ShardedPairDataset(Dataset):
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.manifest = json.loads((self.root_dir / "manifest.json").read_text())
        self.shards = self.manifest["shards"]
        self.cumulative_counts: list[int] = []
        running_total = 0
        for shard in self.shards:
            running_total += shard["count"]
            self.cumulative_counts.append(running_total)
        self._cached_shard_index: int | None = None
        self._cached_shard: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        return self.manifest["num_samples"]

    def _load_shard(self, shard_index: int) -> dict[str, torch.Tensor]:
        if self._cached_shard_index == shard_index and self._cached_shard is not None:
            return self._cached_shard
        shard_file = self.root_dir / self.shards[shard_index]["file"]
        self._cached_shard = torch.load(shard_file, map_location="cpu")
        self._cached_shard_index = shard_index
        return self._cached_shard

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_index = bisect.bisect_right(self.cumulative_counts, index)
        shard_start = 0 if shard_index == 0 else self.cumulative_counts[shard_index - 1]
        local_index = index - shard_start
        shard = self._load_shard(shard_index)
        return {
            "embedding": shard["embedding"][local_index].float(),
            "latents": shard["latents"][local_index].float(),
        }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute DINOv3 and FLUX VAE pairs from ImageNet-1k")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--cache-mode", default="pairs", choices=["pairs", "latents"])
    parser.add_argument("--embedding-kind", default="pooled", choices=["pooled", "dense"])
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    precompute_pairs(
        output_dir=args.output_dir,
        split=args.split,
        cache_mode=args.cache_mode,
        embedding_kind=args.embedding_kind,
        resolution=args.resolution,
        max_samples=args.max_samples,
        shard_size=args.shard_size,
        encode_batch_size=args.encode_batch_size,
        seed=args.seed,
        streaming=args.streaming,
        shuffle=args.shuffle,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
