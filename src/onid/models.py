from __future__ import annotations

import math

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = math.gcd(channels, 32) or 1
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = math.gcd(out_channels, 32) or 1
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            ResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PooledLatentDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        latent_channels: int,
        latent_grid: int,
        base_channels: int = 128,
    ) -> None:
        super().__init__()
        if latent_grid % 4 != 0:
            raise ValueError(f"Expected latent_grid divisible by 4, got {latent_grid}")

        self.latent_grid = latent_grid
        self.base_grid = latent_grid // 4
        self.fc = nn.Linear(embedding_dim, base_channels * self.base_grid * self.base_grid)
        self.net = nn.Sequential(
            UpsampleBlock(base_channels, base_channels),
            UpsampleBlock(base_channels, base_channels),
            ResidualBlock(base_channels),
            nn.GroupNorm(math.gcd(base_channels, 32) or 1, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        batch_size = embedding.shape[0]
        x = self.fc(embedding)
        x = x.view(batch_size, -1, self.base_grid, self.base_grid)
        return self.net(x)


class DenseLatentDecoder(nn.Module):
    def __init__(
        self,
        embedding_channels: int,
        latent_channels: int,
        embedding_grid: int,
        latent_grid: int,
        hidden_channels: int = 256,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = [
            nn.Conv2d(embedding_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
        ]

        current_grid = embedding_grid
        while current_grid < latent_grid:
            blocks.append(UpsampleBlock(hidden_channels, hidden_channels))
            current_grid *= 2

        if current_grid != latent_grid:
            raise ValueError(
                f"Cannot map dense embedding grid {embedding_grid} to latent grid {latent_grid} with x2 upsampling"
            )

        blocks.extend(
            [
            nn.GroupNorm(math.gcd(hidden_channels, 32) or 1, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1),
            ]
        )
        self.net = nn.Sequential(*blocks)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


def build_decoder(
    embedding_kind: str,
    embedding_shape: list[int],
    latent_shape: list[int],
) -> nn.Module:
    if embedding_kind == "pooled":
        return PooledLatentDecoder(
            embedding_dim=embedding_shape[0],
            latent_channels=latent_shape[0],
            latent_grid=latent_shape[-1],
        )
    if embedding_kind == "dense":
        return DenseLatentDecoder(
            embedding_channels=embedding_shape[0],
            latent_channels=latent_shape[0],
            embedding_grid=embedding_shape[-1],
            latent_grid=latent_shape[-1],
        )
    raise ValueError(f"Unsupported embedding_kind: {embedding_kind}")
