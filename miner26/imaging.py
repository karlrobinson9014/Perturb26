"""Image conversions that reproduce the validator's decode path exactly."""
from __future__ import annotations

import base64
import io

import numpy as np
import torch
from PIL import Image

from miner26 import _upstream  # noqa: F401
from perturbnet.image_io import encode_image_b64


def task_b64_from_bytes(raw: bytes) -> str:
    """Same transform as upstream ``image_url_to_b64`` after the HTTP GET."""
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def b64_to_u8(image_b64: str) -> torch.Tensor:
    """uint8 CHW tensor of a base64 image (RGB)."""
    raw = base64.b64decode(image_b64)
    arr = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous()


def u8_to_b64(image_u8: torch.Tensor) -> str:
    """PNG base64 via upstream's encoder (lossless round trip for uint8 input)."""
    return encode_image_b64(image_u8.float() / 255.0)
