from __future__ import annotations

import argparse
import bisect
import json
import math
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

from onid.hf_models import DINO_MODEL_NAME, FLUX_MODEL_NAME, load_dino_encoder, load_flux_vae, module_device, preferred_dtype


IMAGENET_DATASET_NAME = "ILSVRC/imagenet-1k"


@torch.inference_mode()
def preprocess_image(image: Image.Image, resolution: int) -> torch.Tensor:
    image = image.convert("RGB")
    image = resize(image, resolution, interpolation=InterpolationMode.BICUBIC, antialias=True)
    image = center_crop(image, [resolution, resolution])
    return to_tensor(image)


@torch.inference_mode()
def extract_dense_embedding(outputs: Any, num_register_tokens: int) -> torch.Tensor:
    token_offset = 1 + num_register_tokens
    patch_tokens = outputs.last_hidden_state[:, token_offset:, :]
    grid_size = int(math.sqrt(patch_tokens.shape[1]))
    if grid_size * grid_size != patch_tokens.shape[1]:
        raise ValueError(f"Patch token count {patch_tokens.shape[1]} is not a square number")
    patch_tokens = patch_tokens.reshape(patch_tokens.shape[0], grid_size, grid_size, patch_tokens.shape[-1])
    return patch_tokens.permute(0, 3, 1, 2).contiguous()


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

    if embedding_kind == "pooled":
        embedding = dino_outputs.pooler_output
    elif embedding_kind == "dense":
        embedding = extract_dense_embedding(dino_outputs, dino_model.config.num_register_tokens)
    else:
        raise ValueError(f"Unsupported embedding_kind: {embedding_kind}")

    vae_pixels = ((image_tensor * 2.0) - 1.0).unsqueeze(0)
    vae_pixels = vae_pixels.to(module_device(vae), dtype=dtype)
    latent_dist = vae.encode(vae_pixels).latent_dist
    scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
    latents = latent_dist.mean * scaling_factor

    return embedding.squeeze(0).cpu().to(torch.bfloat16), latents.squeeze(0).cpu().to(torch.bfloat16)


def save_shard(
    output_dir: Path,
    shard_index: int,
    embeddings: list[torch.Tensor],
    latents: list[torch.Tensor],
) -> dict[str, Any]:
    shard_name = f"shard-{shard_index:05d}.pt"
    shard_path = output_dir / shard_name
    torch.save(
        {
            "embedding": torch.stack(embeddings, dim=0),
            "latents": torch.stack(latents, dim=0),
        },
        shard_path,
    )
    return {
        "file": shard_name,
        "count": len(embeddings),
    }


def iter_imagenet_samples(split: str, max_samples: int, shuffle: bool, seed: int) -> Any:
    remote_files = [
        file_name
        for file_name in list_repo_files(IMAGENET_DATASET_NAME, repo_type="dataset")
        if file_name.startswith(f"data/{split}-") and file_name.endswith(".parquet")
    ]
    remote_files.sort()
    if shuffle:
        random.Random(seed).shuffle(remote_files)

    yielded = 0
    sample_limit = None if max_samples <= 0 else max_samples

    for remote_file in remote_files:
        local_file = hf_hub_download(
            repo_id=IMAGENET_DATASET_NAME,
            filename=remote_file,
            repo_type="dataset",
        )
        shard_dataset = load_dataset("parquet", data_files={"data": local_file}, split="data")
        if shuffle:
            shard_dataset = shard_dataset.shuffle(seed=seed)
        for sample in shard_dataset:
            yield sample
            yielded += 1
            if sample_limit is not None and yielded >= sample_limit:
                return


def precompute_pairs(
    output_dir: str | Path,
    split: str,
    embedding_kind: str,
    resolution: int,
    max_samples: int,
    shard_size: int,
    seed: int = 0,
    streaming: bool = True,
    shuffle: bool = False,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dtype = preferred_dtype()
    processor, dino_model = load_dino_encoder(dtype=dtype)
    vae = load_flux_vae(dtype=dtype)

    embeddings: list[torch.Tensor] = []
    latents: list[torch.Tensor] = []
    shards: list[dict[str, Any]] = []
    embedding_shape: list[int] | None = None
    latent_shape: list[int] | None = None

    progress_total = max_samples if max_samples > 0 else None
    progress = tqdm(total=progress_total, desc=f"precompute:{split}:{embedding_kind}")
    for sample in iter_imagenet_samples(split=split, max_samples=max_samples, shuffle=shuffle, seed=seed):

        embedding, latent = encode_sample(
            image=sample["image"],
            resolution=resolution,
            processor=processor,
            dino_model=dino_model,
            vae=vae,
            embedding_kind=embedding_kind,
            dtype=dtype,
        )
        embeddings.append(embedding)
        latents.append(latent)

        if embedding_shape is None:
            embedding_shape = list(embedding.shape)
            latent_shape = list(latent.shape)

        if len(embeddings) == shard_size:
            shards.append(save_shard(output_path, len(shards), embeddings, latents))
            embeddings.clear()
            latents.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        progress.update(1)

    if embeddings:
        shards.append(save_shard(output_path, len(shards), embeddings, latents))

    progress.close()

    manifest = {
        "dataset": IMAGENET_DATASET_NAME,
        "split": split,
        "resolution": resolution,
        "embedding_kind": embedding_kind,
        "embedding_shape": embedding_shape,
        "latent_shape": latent_shape,
        "num_samples": sum(shard["count"] for shard in shards),
        "dino_model": DINO_MODEL_NAME,
        "flux_model": FLUX_MODEL_NAME,
        "shards": shards,
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
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
    parser.add_argument("--embedding-kind", default="pooled", choices=["pooled", "dense"])
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    precompute_pairs(
        output_dir=args.output_dir,
        split=args.split,
        embedding_kind=args.embedding_kind,
        resolution=args.resolution,
        max_samples=args.max_samples,
        shard_size=args.shard_size,
        seed=args.seed,
        streaming=args.streaming,
        shuffle=args.shuffle,
    )


if __name__ == "__main__":
    main()
