from __future__ import annotations

import math
from typing import Any, Tuple

import torch
from diffusers import AutoencoderKLFlux2
from transformers import AutoImageProcessor, AutoModel


DINO_MODEL_NAME = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
FLUX_MODEL_NAME = "black-forest-labs/FLUX.2-klein-9B"


def preferred_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def load_dino_encoder(
    model_name: str = DINO_MODEL_NAME,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tuple[AutoImageProcessor, torch.nn.Module]:
    dtype = dtype or preferred_dtype()
    device = device or default_device()

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.requires_grad_(False)
    model.eval()
    return processor, model


def dense_token_offset(model: torch.nn.Module) -> int:
    return 1 + (getattr(model.config, "num_register_tokens", 0) or 0)


def extract_dense_embedding(outputs: Any, model: torch.nn.Module) -> torch.Tensor:
    token_offset = dense_token_offset(model)
    patch_tokens = outputs.last_hidden_state[:, token_offset:, :]
    grid_size = int(math.sqrt(patch_tokens.shape[1]))
    if grid_size * grid_size != patch_tokens.shape[1]:
        raise ValueError(f"Patch token count {patch_tokens.shape[1]} is not a square number")
    patch_tokens = patch_tokens.reshape(patch_tokens.shape[0], grid_size, grid_size, patch_tokens.shape[-1])
    return patch_tokens.permute(0, 3, 1, 2).contiguous()


def extract_embedding(outputs: Any, model: torch.nn.Module, embedding_kind: str) -> torch.Tensor:
    if embedding_kind == "pooled":
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return pooled
        patch_tokens = outputs.last_hidden_state[:, dense_token_offset(model) :, :]
        return patch_tokens.mean(dim=1)
    if embedding_kind == "dense":
        return extract_dense_embedding(outputs, model)
    raise ValueError(f"Unsupported embedding_kind: {embedding_kind}")


def load_flux_vae(
    model_name: str = FLUX_MODEL_NAME,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> AutoencoderKLFlux2:
    dtype = dtype or preferred_dtype()
    device = device or default_device()

    vae = AutoencoderKLFlux2.from_pretrained(
        model_name,
        subfolder="vae",
        torch_dtype=dtype,
    )
    vae = vae.to(device)
    vae.requires_grad_(False)
    vae.enable_slicing()
    vae.eval()
    return vae
