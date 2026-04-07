from __future__ import annotations

from typing import Tuple

import torch
from diffusers import AutoencoderKLFlux2
from transformers import AutoImageProcessor, AutoModel


DINO_MODEL_NAME = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
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
    model.eval()
    return processor, model


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
    vae.enable_slicing()
    vae.eval()
    return vae
