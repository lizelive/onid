from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from onid.hf_models import load_flux_vae
from onid.models import build_decoder
from onid.pairs import ShardedPairDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def decode_latents(vae: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
    vae_param = next(vae.parameters())
    latents = latents.to(device=vae_param.device, dtype=vae_param.dtype)
    images = vae.decode(latents / scaling_factor).sample
    return images.add(1.0).div(2.0).clamp(0.0, 1.0)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    vae: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    image_metric_batches: int,
    output_dir: Path,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    latent_mse_sum = 0.0
    image_mse_sum = 0.0
    image_psnr_sum = 0.0
    steps = 0
    image_steps = 0

    for batch_index, batch in enumerate(dataloader):
        embeddings = batch["embedding"].to(device)
        latents = batch["latents"].to(device)
        predictions = model(embeddings)
        latent_mse = F.mse_loss(predictions, latents)
        latent_mse_sum += latent_mse.item()
        steps += 1

        if batch_index < image_metric_batches:
            pred_images = decode_latents(vae, predictions)
            target_images = decode_latents(vae, latents)
            image_mse = F.mse_loss(pred_images, target_images)
            image_psnr = -10.0 * math.log10(max(image_mse.item(), 1.0e-8))
            image_mse_sum += image_mse.item()
            image_psnr_sum += image_psnr
            image_steps += 1

            if batch_index == 0:
                sample_grid = make_grid(
                    torch.cat([target_images[:4], pred_images[:4]], dim=0),
                    nrow=4,
                )
                save_image(sample_grid, output_dir / f"epoch-{epoch:03d}-recon.png")

    return {
        "latent_mse": latent_mse_sum / max(steps, 1),
        "image_mse": image_mse_sum / max(image_steps, 1),
        "image_psnr": image_psnr_sum / max(image_steps, 1),
    }


def train_experiment(
    train_dir: str | Path,
    val_dir: str | Path,
    output_dir: str | Path,
    embedding_kind: str,
    epochs: int,
    batch_size: int,
    learning_rate: float = 1.0e-4,
    weight_decay: float = 1.0e-4,
    num_workers: int = 4,
    seed: int = 0,
    image_metric_batches: int = 2,
) -> Path:
    set_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ShardedPairDataset(train_dir)
    val_dataset = ShardedPairDataset(val_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_decoder(
        embedding_kind=embedding_kind,
        embedding_shape=train_dataset.manifest["embedding_shape"],
        latent_shape=train_dataset.manifest["latent_shape"],
    ).to(device)
    vae = load_flux_vae(device=device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    metrics: dict[str, Any] = {
        "embedding_kind": embedding_kind,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs": [],
    }

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"train:epoch={epoch}")
        train_loss_sum = 0.0
        train_steps = 0

        for batch in progress:
            embeddings = batch["embedding"].to(device)
            latents = batch["latents"].to(device)

            predictions = model(embeddings)
            loss = F.mse_loss(predictions, latents)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_metrics = evaluate(
            model=model,
            vae=vae,
            dataloader=val_loader,
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

        checkpoint = {
            "model": model.state_dict(),
            "metrics": metrics,
            "manifest": train_dataset.manifest,
        }
        torch.save(checkpoint, output_path / "last.pt")
        if val_metrics["latent_mse"] < best_val:
            best_val = val_metrics["latent_mse"]
            torch.save(checkpoint, output_path / "best.pt")

        (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return output_path / "metrics.json"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the DINO-to-FLUX latent decoder")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-kind", default="pooled", choices=["pooled", "dense"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-metric-batches", type=int, default=2)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
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
    )


if __name__ == "__main__":
    main()
