from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from onid.hf_models import extract_embedding, load_dino_encoder, load_flux_vae, preferred_dtype
from onid.models import build_decoder
from onid.pairs import IMAGENET_DATASET_NAME, ShardedPairDataset, iter_imagenet_samples, iter_sharded_tensors, load_manifest, preprocess_image


IMAGENET_SPLIT_SIZES = {
    "train": 1_281_167,
    "validation": 50_000,
}


def configure_tf32(device: torch.device, enabled: bool = True) -> bool:
    if device.type != "cuda":
        return False

    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    torch.set_float32_matmul_precision("high" if enabled else "highest")
    return enabled


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2))
    temp_path.replace(path)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def autocast_context(device: torch.device) -> Any:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def supports_torch_compile(device: torch.device) -> bool:
    if not hasattr(torch, "compile"):
        return False
    if device.type != "cuda":
        return False
    return torch.cuda.get_device_capability(device) >= (7, 0)


def maybe_compile(
    target: Any,
    device: torch.device,
    enabled: bool,
    mode: str,
    label: str,
) -> Any:
    if not enabled or not supports_torch_compile(device):
        return target

    compiled_target = torch.compile(target, fullgraph=False, mode=mode)
    compile_failed = False

    def compiled_or_eager(*args: Any, **kwargs: Any) -> Any:
        nonlocal compile_failed
        if compile_failed:
            return target(*args, **kwargs)
        try:
            return compiled_target(*args, **kwargs)
        except Exception as error:
            compile_failed = True
            print(f"torch.compile fallback for {label}: {type(error).__name__}: {error}")
            return target(*args, **kwargs)

    return compiled_or_eager


def build_optimizer_step(
    optimizer: AdamW,
    device: torch.device,
    enabled: bool,
    mode: str,
) -> Callable[[], None]:
    if not enabled or not supports_torch_compile(device):
        return optimizer.step

    def eager_optimizer_step() -> None:
        optimizer.step()

    return maybe_compile(eager_optimizer_step, device=device, enabled=True, mode=mode, label="optimizer.step")


def mark_compile_step_begin(enabled: bool) -> None:
    if not enabled:
        return
    compiler = getattr(torch, "compiler", None)
    if compiler is None:
        return
    marker = getattr(compiler, "cudagraph_mark_step_begin", None)
    if marker is not None:
        marker()


def set_module_mode(module: nn.Module, compiled_module: Any, training: bool) -> None:
    module.train(training)
    if compiled_module is not module and hasattr(compiled_module, "train"):
        compiled_module.train(training)


def resolve_sample_count(split: str, max_samples: int) -> int:
    if split not in IMAGENET_SPLIT_SIZES:
        raise ValueError(f"Unsupported ImageNet split: {split}")
    total = IMAGENET_SPLIT_SIZES[split]
    if max_samples <= 0:
        return total
    return min(max_samples, total)


@torch.inference_mode()
def decode_latents(
    vae: nn.Module,
    latents: torch.Tensor,
    vae_decode: Callable[..., Any] | None = None,
) -> torch.Tensor:
    scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
    vae_param = next(vae.parameters())
    latents = latents.to(device=vae_param.device, dtype=vae_param.dtype)
    decode_fn = vae_decode or vae.decode
    images = decode_fn(latents / scaling_factor).sample
    return images.add(1.0).div(2.0).clamp(0.0, 1.0)


class OnlineEmbeddingEncoder:
    def __init__(
        self,
        embedding_kind: str,
        resolution: int,
        device: torch.device,
        enable_compile: bool = False,
        compile_mode: str = "max-autotune",
    ) -> None:
        self.embedding_kind = embedding_kind
        self.resolution = resolution
        self.device = device
        self.dtype = preferred_dtype()
        self.processor, self.model = load_dino_encoder(device=device, dtype=self.dtype)
        self.model_forward = maybe_compile(
            self.model,
            device=device,
            enabled=enable_compile,
            mode=compile_mode,
            label="dino.encoder",
        )
        self.image_mean = torch.tensor(self.processor.image_mean, device=device, dtype=torch.float32).view(1, -1, 1, 1)
        self.image_std = torch.tensor(self.processor.image_std, device=device, dtype=torch.float32).view(1, -1, 1, 1)

    @torch.no_grad()
    def infer_embedding_shape(self) -> list[int]:
        dummy = torch.zeros(1, 3, self.resolution, self.resolution, device=self.device, dtype=torch.float32)
        embedding = self.encode(dummy)
        return list(embedding.shape[1:])

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        normalized = images.to(self.device, dtype=torch.float32)
        normalized = (normalized - self.image_mean) / self.image_std
        normalized = normalized.to(dtype=self.dtype)
        with autocast_context(self.device):
            outputs = self.model_forward(pixel_values=normalized)
            embeddings = extract_embedding(outputs, self.model, self.embedding_kind)
        return embeddings.clone()


class OnlineSupervisionEncoder(OnlineEmbeddingEncoder):
    def __init__(
        self,
        embedding_kind: str,
        resolution: int,
        device: torch.device,
        enable_compile: bool = False,
        compile_mode: str = "max-autotune",
    ) -> None:
        super().__init__(
            embedding_kind=embedding_kind,
            resolution=resolution,
            device=device,
            enable_compile=enable_compile,
            compile_mode=compile_mode,
        )
        self.vae = load_flux_vae(device=device, dtype=self.dtype)
        self.vae_encode = self.vae.encode
        self.vae_decode = self.vae.decode

    @torch.no_grad()
    def infer_shapes(self) -> tuple[list[int], list[int]]:
        dummy = torch.zeros(1, 3, self.resolution, self.resolution, device=self.device, dtype=torch.float32)
        embeddings, latents = self.encode_supervision(dummy)
        return list(embeddings.shape[1:]), list(latents.shape[1:])

    @torch.no_grad()
    def encode_supervision(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images = images.to(self.device, dtype=torch.float32)
        embeddings = self.encode(images)
        vae_pixels = ((images * 2.0) - 1.0).to(dtype=self.dtype)
        with autocast_context(self.device):
            latent_dist = self.vae_encode(vae_pixels).latent_dist
            scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
            latents = latent_dist.mean * scaling_factor
        return embeddings, latents.clone()


def iter_online_pairs(
    latent_dir: str | Path,
    seed: int,
    shuffle_buffer: int,
) -> Iterator[dict[str, torch.Tensor]]:
    manifest = load_manifest(latent_dir)
    image_iter = iter_imagenet_samples(
        split=manifest["split"],
        max_samples=manifest["num_samples"],
        shuffle=False,
        seed=seed,
    )
    latent_iter = iter_sharded_tensors(latent_dir, "latents")

    buffer: list[dict[str, torch.Tensor]] = []
    rng = random.Random(seed)
    for sample, latent in zip(image_iter, latent_iter):
        pair = {
            "image": preprocess_image(sample["image"], manifest["resolution"]),
            "latents": latent,
        }
        if shuffle_buffer <= 0:
            yield pair
            continue
        if len(buffer) < shuffle_buffer:
            buffer.append(pair)
            continue
        swap_index = rng.randrange(len(buffer))
        yield buffer[swap_index]
        buffer[swap_index] = pair

    while buffer:
        yield buffer.pop(rng.randrange(len(buffer)))


def iter_online_batches(
    latent_dir: str | Path,
    batch_size: int,
    seed: int,
    shuffle_buffer: int,
    start_step: int = 0,
) -> Iterator[dict[str, torch.Tensor]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    pair_iter = iter_online_pairs(latent_dir=latent_dir, seed=seed, shuffle_buffer=shuffle_buffer)
    samples_to_skip = start_step * batch_size
    for _ in range(samples_to_skip):
        try:
            next(pair_iter)
        except StopIteration:
            return

    images: list[torch.Tensor] = []
    latents: list[torch.Tensor] = []
    for pair in pair_iter:
        images.append(pair["image"])
        latents.append(pair["latents"])
        if len(images) == batch_size:
            yield {
                "image": torch.stack(images, dim=0),
                "latents": torch.stack(latents, dim=0),
            }
            images.clear()
            latents.clear()

    if images:
        yield {
            "image": torch.stack(images, dim=0),
            "latents": torch.stack(latents, dim=0),
        }


def iter_imagenet_batches(
    split: str,
    resolution: int,
    batch_size: int,
    seed: int,
    max_samples: int,
    start_step: int = 0,
    shuffle: bool = True,
) -> Iterator[dict[str, torch.Tensor]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    sample_iter = iter_imagenet_samples(
        split=split,
        max_samples=max_samples,
        shuffle=shuffle,
        seed=seed,
        start_index=start_step * batch_size,
    )
    images: list[torch.Tensor] = []
    for sample in sample_iter:
        images.append(preprocess_image(sample["image"], resolution))
        if len(images) == batch_size:
            yield {"image": torch.stack(images, dim=0)}
            images.clear()

    if images:
        yield {"image": torch.stack(images, dim=0)}


def gather_probe_batch(
    data_mode: str,
    batch_limit: int,
    train_dir: str | Path | None,
    train_split: str | None,
    resolution: int,
    train_samples: int,
    shuffle_buffer: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    if data_mode == "latents":
        batch_iter = iter_online_batches(
            latent_dir=train_dir,
            batch_size=batch_limit,
            seed=seed,
            shuffle_buffer=shuffle_buffer,
        )
        return next(batch_iter)
    if data_mode == "imagenet":
        batch_iter = iter_imagenet_batches(
            split=train_split,
            resolution=resolution,
            batch_size=batch_limit,
            seed=seed,
            max_samples=min(train_samples, batch_limit),
            shuffle=True,
        )
        return next(batch_iter)

    dataset = ShardedPairDataset(train_dir)
    probe_samples = [dataset[index] for index in range(min(batch_limit, len(dataset)))]
    return {
        "embedding": torch.stack([sample["embedding"] for sample in probe_samples], dim=0),
        "latents": torch.stack([sample["latents"] for sample in probe_samples], dim=0),
    }


def try_batch_size(
    batch: dict[str, torch.Tensor],
    batch_size: int,
    data_mode: str,
    encoder: OnlineEmbeddingEncoder | None,
    supervisor: OnlineSupervisionEncoder | None,
    model_forward: Callable[[torch.Tensor], torch.Tensor],
    optimizer: AdamW,
    device: torch.device,
) -> bool:
    try:
        optimizer.zero_grad(set_to_none=True)
        if data_mode == "pairs":
            embeddings = batch["embedding"][:batch_size].to(device)
            latents = batch["latents"][:batch_size].to(device)
        elif data_mode == "latents":
            if encoder is None:
                raise ValueError("encoder is required for latent-cached probing")
            embeddings = encoder.encode(batch["image"][:batch_size])
            latents = batch["latents"][:batch_size].to(device)
        else:
            if supervisor is None:
                raise ValueError("supervisor is required for online ImageNet probing")
            embeddings, latents = supervisor.encode_supervision(batch["image"][:batch_size])

        with autocast_context(device):
            predictions = model_forward(embeddings).clone()
        loss = F.mse_loss(predictions.float(), latents.float())
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
        return True
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False


def auto_select_batch_size(
    data_mode: str,
    train_dir: str | Path | None,
    train_split: str | None,
    resolution: int,
    train_samples: int,
    requested_batch_size: int,
    max_batch_size: int,
    encoder: OnlineEmbeddingEncoder | None,
    supervisor: OnlineSupervisionEncoder | None,
    model_forward: Callable[[torch.Tensor], torch.Tensor],
    optimizer: AdamW,
    device: torch.device,
    shuffle_buffer: int,
    seed: int,
) -> int:
    if device.type != "cuda":
        return requested_batch_size

    upper_bound = min(train_samples, max(requested_batch_size, max_batch_size))
    probe_batch = gather_probe_batch(
        data_mode=data_mode,
        batch_limit=upper_bound,
        train_dir=train_dir,
        train_split=train_split,
        resolution=resolution,
        train_samples=train_samples,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )

    good = 0
    bad = upper_bound + 1
    candidate = 1
    while candidate <= upper_bound:
        if try_batch_size(probe_batch, candidate, data_mode, encoder, supervisor, model_forward, optimizer, device):
            good = candidate
            candidate *= 2
            continue
        bad = candidate
        break

    if good == 0:
        raise RuntimeError("Unable to fit even batch size 1 on the current GPU")
    if bad == upper_bound + 1:
        return upper_bound

    low = good + 1
    high = bad - 1
    while low <= high:
        mid = (low + high) // 2
        if try_batch_size(probe_batch, mid, data_mode, encoder, supervisor, model_forward, optimizer, device):
            good = mid
            low = mid + 1
        else:
            high = mid - 1
    return good


def evaluate_cached(
    model: nn.Module,
    model_forward: Callable[[torch.Tensor], torch.Tensor],
    vae: nn.Module,
    vae_decode: Callable[..., Any],
    dataloader: DataLoader,
    device: torch.device,
    image_metric_batches: int,
    output_dir: Path,
    epoch: int,
) -> dict[str, float]:
    set_module_mode(model, model_forward, training=False)
    latent_mse_sum = 0.0
    image_mse_sum = 0.0
    image_psnr_sum = 0.0
    steps = 0
    image_steps = 0

    for batch_index, batch in enumerate(dataloader):
        embeddings = batch["embedding"].to(device)
        latents = batch["latents"].to(device)
        with autocast_context(device):
            predictions = model_forward(embeddings).clone()
        latent_mse = F.mse_loss(predictions.float(), latents.float())
        latent_mse_sum += latent_mse.item()
        steps += 1

        if batch_index < image_metric_batches:
            pred_images = decode_latents(vae, predictions, vae_decode=vae_decode)
            target_images = decode_latents(vae, latents, vae_decode=vae_decode)
            image_mse = F.mse_loss(pred_images, target_images)
            image_psnr = -10.0 * math.log10(max(image_mse.item(), 1.0e-8))
            image_mse_sum += image_mse.item()
            image_psnr_sum += image_psnr
            image_steps += 1

            if batch_index == 0:
                sample_grid = make_grid(torch.cat([target_images[:4], pred_images[:4]], dim=0), nrow=4)
                save_image(sample_grid, output_dir / f"epoch-{epoch:03d}-recon.png")

    return {
        "latent_mse": latent_mse_sum / max(steps, 1),
        "image_mse": image_mse_sum / max(image_steps, 1),
        "image_psnr": image_psnr_sum / max(image_steps, 1),
    }


def evaluate_latent_online(
    model: nn.Module,
    model_forward: Callable[[torch.Tensor], torch.Tensor],
    vae: nn.Module,
    vae_decode: Callable[..., Any],
    encoder: OnlineEmbeddingEncoder,
    latent_dir: str | Path,
    batch_size: int,
    device: torch.device,
    image_metric_batches: int,
    output_dir: Path,
    epoch: int,
) -> dict[str, float]:
    set_module_mode(model, model_forward, training=False)
    latent_mse_sum = 0.0
    image_mse_sum = 0.0
    image_psnr_sum = 0.0
    steps = 0
    image_steps = 0

    for batch_index, batch in enumerate(
        iter_online_batches(latent_dir=latent_dir, batch_size=batch_size, seed=0, shuffle_buffer=0)
    ):
        latents = batch["latents"].to(device)
        embeddings = encoder.encode(batch["image"])
        with autocast_context(device):
            predictions = model_forward(embeddings).clone()
        latent_mse = F.mse_loss(predictions.float(), latents.float())
        latent_mse_sum += latent_mse.item()
        steps += 1

        if batch_index < image_metric_batches:
            pred_images = decode_latents(vae, predictions, vae_decode=vae_decode)
            target_images = decode_latents(vae, latents, vae_decode=vae_decode)
            image_mse = F.mse_loss(pred_images, target_images)
            image_psnr = -10.0 * math.log10(max(image_mse.item(), 1.0e-8))
            image_mse_sum += image_mse.item()
            image_psnr_sum += image_psnr
            image_steps += 1

            if batch_index == 0:
                sample_grid = make_grid(torch.cat([target_images[:4], pred_images[:4]], dim=0), nrow=4)
                save_image(sample_grid, output_dir / f"epoch-{epoch:03d}-recon.png")

    return {
        "latent_mse": latent_mse_sum / max(steps, 1),
        "image_mse": image_mse_sum / max(image_steps, 1),
        "image_psnr": image_psnr_sum / max(image_steps, 1),
    }


def evaluate_imagenet_online(
    model: nn.Module,
    model_forward: Callable[[torch.Tensor], torch.Tensor],
    supervisor: OnlineSupervisionEncoder,
    split: str,
    resolution: int,
    sample_count: int,
    batch_size: int,
    device: torch.device,
    image_metric_batches: int,
    output_dir: Path,
    epoch: int,
) -> dict[str, float]:
    set_module_mode(model, model_forward, training=False)
    latent_mse_sum = 0.0
    image_mse_sum = 0.0
    image_psnr_sum = 0.0
    steps = 0
    image_steps = 0

    for batch_index, batch in enumerate(
        iter_imagenet_batches(
            split=split,
            resolution=resolution,
            batch_size=batch_size,
            seed=0,
            max_samples=sample_count,
            shuffle=False,
        )
    ):
        embeddings, latents = supervisor.encode_supervision(batch["image"])
        with autocast_context(device):
            predictions = model_forward(embeddings).clone()
        latent_mse = F.mse_loss(predictions.float(), latents.float())
        latent_mse_sum += latent_mse.item()
        steps += 1

        if batch_index < image_metric_batches:
            pred_images = decode_latents(supervisor.vae, predictions, vae_decode=supervisor.vae_decode)
            target_images = decode_latents(supervisor.vae, latents, vae_decode=supervisor.vae_decode)
            image_mse = F.mse_loss(pred_images, target_images)
            image_psnr = -10.0 * math.log10(max(image_mse.item(), 1.0e-8))
            image_mse_sum += image_mse.item()
            image_psnr_sum += image_psnr
            image_steps += 1

            if batch_index == 0:
                sample_grid = make_grid(torch.cat([target_images[:4], pred_images[:4]], dim=0), nrow=4)
                save_image(sample_grid, output_dir / f"epoch-{epoch:03d}-recon.png")

    return {
        "latent_mse": latent_mse_sum / max(steps, 1),
        "image_mse": image_mse_sum / max(image_steps, 1),
        "image_psnr": image_psnr_sum / max(image_steps, 1),
    }


def train_experiment(
    train_dir: str | Path | None,
    val_dir: str | Path | None,
    output_dir: str | Path,
    embedding_kind: str,
    epochs: int,
    batch_size: int,
    learning_rate: float = 1.0e-4,
    weight_decay: float = 1.0e-4,
    num_workers: int = 4,
    seed: int = 0,
    image_metric_batches: int = 2,
    decoder_architecture: str | None = None,
    resume: bool = False,
    auto_batch_size: bool = False,
    max_batch_size: int = 64,
    eval_batch_size: int | None = None,
    checkpoint_interval_steps: int = 500,
    online_shuffle_buffer: int = 0,
    train_split: str | None = None,
    val_split: str | None = None,
    resolution: int = 256,
    train_samples: int = 0,
    val_samples: int = 0,
    enable_compile: bool = True,
    compile_mode: str = "max-autotune",
) -> Path:
    set_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf32_enabled = configure_tf32(device, enabled=True)
    compile_enabled = enable_compile and supports_torch_compile(device)
    data_mode: str
    train_descriptor: dict[str, Any]
    val_descriptor: dict[str, Any]
    encoder: OnlineEmbeddingEncoder | None = None
    supervisor: OnlineSupervisionEncoder | None = None

    if train_dir is not None and val_dir is not None:
        train_manifest = load_manifest(train_dir)
        val_manifest = load_manifest(val_dir)
        if train_manifest.get("cache_mode", "pairs") == "latents":
            if val_manifest.get("cache_mode", "pairs") != "latents":
                raise ValueError("train_dir and val_dir must use the same cache mode")
            data_mode = "latents"
            encoder = OnlineEmbeddingEncoder(
                embedding_kind=embedding_kind,
                resolution=train_manifest["resolution"],
                device=device,
                enable_compile=compile_enabled,
                compile_mode=compile_mode,
            )
            embedding_shape = encoder.infer_embedding_shape()
            latent_shape = train_manifest["latent_shape"]
        else:
            data_mode = "pairs"
            embedding_shape = train_manifest["embedding_shape"]
            latent_shape = train_manifest["latent_shape"]
        train_descriptor = train_manifest
        val_descriptor = val_manifest
        train_samples = train_manifest["num_samples"]
        val_samples = val_manifest["num_samples"]
        resolution = train_manifest["resolution"]
    else:
        if train_split is None or val_split is None:
            raise ValueError("Either train_dir/val_dir or train_split/val_split must be provided")
        data_mode = "imagenet"
        train_samples = resolve_sample_count(train_split, train_samples)
        val_samples = resolve_sample_count(val_split, val_samples)
        supervisor = OnlineSupervisionEncoder(
            embedding_kind=embedding_kind,
            resolution=resolution,
            device=device,
            enable_compile=compile_enabled,
            compile_mode=compile_mode,
        )
        embedding_shape, latent_shape = supervisor.infer_shapes()
        train_descriptor = {
            "dataset": IMAGENET_DATASET_NAME,
            "split": train_split,
            "resolution": resolution,
            "num_samples": train_samples,
            "cache_mode": "none",
        }
        val_descriptor = {
            "dataset": IMAGENET_DATASET_NAME,
            "split": val_split,
            "resolution": resolution,
            "num_samples": val_samples,
            "cache_mode": "none",
        }

    model = build_decoder(
        embedding_kind=embedding_kind,
        embedding_shape=embedding_shape,
        latent_shape=latent_shape,
        architecture=decoder_architecture,
    ).to(device)
    model_forward = maybe_compile(
        model,
        device=device,
        enabled=compile_enabled,
        mode=compile_mode,
        label="decoder.forward",
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_step = build_optimizer_step(optimizer, device=device, enabled=compile_enabled, mode=compile_mode)

    checkpoint_path = output_path / "last.pt"
    metrics_path = output_path / "metrics.json"
    state_path = output_path / "state.json"
    metrics: dict[str, Any] = {
        "data_mode": data_mode,
        "embedding_kind": embedding_kind,
        "decoder_architecture": decoder_architecture or ("dense-residual" if embedding_kind == "dense" else None),
        "train_samples": train_samples,
        "val_samples": val_samples,
        "epochs": [],
    }
    start_epoch = 1
    start_step = 0
    best_val = float("inf")

    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        metrics = checkpoint["metrics"]
        best_val = checkpoint["best_val"]
        start_epoch = checkpoint["epoch"]
        start_step = checkpoint["step_in_epoch"]
        batch_size = checkpoint.get("run_config", {}).get("batch_size", batch_size)

    optimizer_step = build_optimizer_step(optimizer, device=device, enabled=compile_enabled, mode=compile_mode)

    if auto_batch_size and not checkpoint_path.exists():
        batch_size = auto_select_batch_size(
            data_mode=data_mode,
            train_dir=train_dir,
            train_split=train_descriptor["split"],
            resolution=resolution,
            train_samples=train_samples,
            requested_batch_size=batch_size,
            max_batch_size=max_batch_size,
            encoder=encoder,
            supervisor=supervisor,
            model_forward=model_forward,
            optimizer=optimizer,
            device=device,
            shuffle_buffer=online_shuffle_buffer,
            seed=seed,
        )

    eval_batch_size = eval_batch_size or min(batch_size, 8)
    run_config = {
        "data_mode": data_mode,
        "embedding_kind": embedding_kind,
        "decoder_architecture": decoder_architecture,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "online_shuffle_buffer": online_shuffle_buffer,
        "checkpoint_interval_steps": checkpoint_interval_steps,
        "compile_enabled": compile_enabled,
        "compile_mode": compile_mode,
        "tf32_enabled": tf32_enabled,
        "train": train_descriptor,
        "val": val_descriptor,
    }
    atomic_write_json(output_path / "run_config.json", run_config)

    if start_epoch > epochs:
        return metrics_path

    if data_mode == "imagenet":
        if supervisor is None:
            raise ValueError("supervisor is required for online ImageNet mode")
        vae = supervisor.vae
        vae_decode = supervisor.vae_decode
    else:
        vae = load_flux_vae(device=device)
        vae_decode = vae.decode

    if data_mode == "pairs":
        train_dataset = ShardedPairDataset(train_dir)
        val_dataset = ShardedPairDataset(val_dir)
        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
    else:
        steps_per_epoch = math.ceil(train_samples / batch_size)

    def save_checkpoint(epoch_value: int, step_value: int) -> None:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "best_val": best_val,
            "epoch": epoch_value,
            "step_in_epoch": step_value,
            "run_config": run_config,
        }
        atomic_torch_save(checkpoint, checkpoint_path)
        atomic_write_json(
            state_path,
            {
                "epoch": epoch_value,
                "step_in_epoch": step_value,
                "best_val": best_val,
                "batch_size": batch_size,
                "data_mode": data_mode,
            },
        )

    for epoch in range(start_epoch, epochs + 1):
        set_module_mode(model, model_forward, training=True)
        train_loss_sum = 0.0
        train_steps = 0
        epoch_seed = seed + epoch

        if data_mode == "pairs":
            generator = torch.Generator()
            generator.manual_seed(epoch_seed)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                generator=generator,
            )
            train_iter: Iterator[dict[str, torch.Tensor]] = iter(train_loader)
            for _ in range(start_step):
                next(train_iter)
        elif data_mode == "latents":
            train_iter = iter_online_batches(
                latent_dir=train_dir,
                batch_size=batch_size,
                seed=epoch_seed,
                shuffle_buffer=online_shuffle_buffer,
                start_step=start_step,
            )
        else:
            train_iter = iter_imagenet_batches(
                split=train_descriptor["split"],
                resolution=resolution,
                batch_size=batch_size,
                seed=epoch_seed,
                max_samples=train_samples,
                start_step=start_step,
                shuffle=True,
            )

        progress = tqdm(total=steps_per_epoch - start_step, desc=f"train:epoch={epoch}")
        for local_step, batch in enumerate(train_iter, start=start_step + 1):
            mark_compile_step_begin(compile_enabled)
            if data_mode == "pairs":
                embeddings = batch["embedding"].to(device)
                latents = batch["latents"].to(device)
            elif data_mode == "latents":
                if encoder is None:
                    raise ValueError("encoder is required for latent-cached mode")
                embeddings = encoder.encode(batch["image"])
                latents = batch["latents"].to(device)
            else:
                if supervisor is None:
                    raise ValueError("supervisor is required for online ImageNet mode")
                embeddings, latents = supervisor.encode_supervision(batch["image"])

            with autocast_context(device):
                predictions = model_forward(embeddings).clone()
            loss = F.mse_loss(predictions.float(), latents.float())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer_step()

            train_loss_sum += loss.item()
            train_steps += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if checkpoint_interval_steps > 0 and local_step % checkpoint_interval_steps == 0:
                save_checkpoint(epoch, local_step)

        progress.close()

        if data_mode == "pairs":
            val_metrics = evaluate_cached(
                model=model,
                model_forward=model_forward,
                vae=vae,
                vae_decode=vae_decode,
                dataloader=val_loader,
                device=device,
                image_metric_batches=image_metric_batches,
                output_dir=output_path,
                epoch=epoch,
            )
        elif data_mode == "latents":
            if encoder is None:
                raise ValueError("encoder is required for latent-cached validation")
            val_metrics = evaluate_latent_online(
                model=model,
                model_forward=model_forward,
                vae=vae,
                vae_decode=vae_decode,
                encoder=encoder,
                latent_dir=val_dir,
                batch_size=eval_batch_size,
                device=device,
                image_metric_batches=image_metric_batches,
                output_dir=output_path,
                epoch=epoch,
            )
        else:
            if supervisor is None:
                raise ValueError("supervisor is required for online ImageNet validation")
            val_metrics = evaluate_imagenet_online(
                model=model,
                model_forward=model_forward,
                supervisor=supervisor,
                split=val_descriptor["split"],
                resolution=resolution,
                sample_count=val_samples,
                batch_size=eval_batch_size,
                device=device,
                image_metric_batches=image_metric_batches,
                output_dir=output_path,
                epoch=epoch,
            )

        epoch_metrics = {
            "epoch": epoch,
            "train_latent_mse": train_loss_sum / max(train_steps, 1),
            **val_metrics,
        }
        metrics["epochs"].append(epoch_metrics)
        atomic_write_json(metrics_path, metrics)

        if val_metrics["latent_mse"] < best_val:
            best_val = val_metrics["latent_mse"]
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": metrics,
                    "best_val": best_val,
                    "epoch": epoch + 1,
                    "step_in_epoch": 0,
                    "run_config": run_config,
                },
                output_path / "best.pt",
            )

        save_checkpoint(epoch + 1, 0)
        start_step = 0

    return metrics_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the DINO-to-FLUX latent decoder")
    parser.add_argument("--train-dir")
    parser.add_argument("--val-dir")
    parser.add_argument("--train-split", choices=["train", "validation"])
    parser.add_argument("--val-split", choices=["train", "validation"])
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--val-samples", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-kind", default="pooled", choices=["pooled", "dense"])
    parser.add_argument("--decoder-architecture", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--auto-batch-size", action="store_true")
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-metric-batches", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-interval-steps", type=int, default=500)
    parser.add_argument("--online-shuffle-buffer", type=int, default=0)
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--compile", dest="enable_compile", action="store_true")
    parser.add_argument("--no-compile", dest="enable_compile", action="store_false")
    parser.set_defaults(enable_compile=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if (args.train_dir is None) != (args.val_dir is None):
        raise ValueError("train-dir and val-dir must be provided together")
    if args.train_dir is None and (args.train_split is None or args.val_split is None):
        raise ValueError("Provide either train-dir/val-dir or train-split/val-split")

    train_experiment(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        embedding_kind=args.embedding_kind,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        image_metric_batches=args.image_metric_batches,
        decoder_architecture=args.decoder_architecture,
        resume=args.resume,
        auto_batch_size=args.auto_batch_size,
        max_batch_size=args.max_batch_size,
        eval_batch_size=args.eval_batch_size,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        online_shuffle_buffer=args.online_shuffle_buffer,
        train_split=args.train_split,
        val_split=args.val_split,
        resolution=args.resolution,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        enable_compile=args.enable_compile,
        compile_mode=args.compile_mode,
    )


if __name__ == "__main__":
    main()
