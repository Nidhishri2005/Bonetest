"""Image preprocessing for inference and training."""

from __future__ import annotations

import io
from typing import Tuple

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from app.core.config import Settings, get_settings


def letterbox_resize(
    image: NDArray[np.uint8], target_size: int
) -> NDArray[np.uint8]:
    """Resize preserving aspect ratio with zero padding."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def normalize_image(
    image: NDArray[np.uint8],
    mean: float | None = None,
    std: float | None = None,
) -> NDArray[np.float32]:
    settings = get_settings()
    mean = mean if mean is not None else settings.norm_mean
    std = std if std is not None else settings.norm_std

    normalized = image.astype(np.float32) / 255.0
    normalized = (normalized - mean) / max(std, 1e-6)
    return normalized


def bytes_to_grayscale(image_bytes: bytes) -> NDArray[np.uint8]:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Unable to decode image. Supported formats: PNG, JPEG, BMP.")
    return image


def preprocess_for_inference(
    image_bytes: bytes,
    settings: Settings | None = None,
) -> Tuple[torch.Tensor, NDArray[np.uint8]]:
    """Return (1,1,H,W) tensor and original grayscale for overlay."""
    """Return (1,1,H,W) tensor and letterbox canvas (H,W) for exact Grad-CAM overlay."""
    settings = settings or get_settings()
    raw = bytes_to_grayscale(image_bytes)
    original_display = raw.copy()
    resized = letterbox_resize(raw, settings.image_size)
    normalized = normalize_image(resized)
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).float()
    return tensor, original_display
    return tensor, resized


def tensor_to_display_image(tensor: torch.Tensor) -> NDArray[np.uint8]:
    """Denormalize single-channel tensor to uint8 for visualization."""
    settings = get_settings()
    arr = tensor.squeeze().cpu().numpy()
    arr = arr * settings.norm_std + settings.norm_mean
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def overlay_heatmap(
    base: NDArray[np.uint8],
    heatmap: NDArray[np.float32],
    alpha: float = 0.45,
) -> NDArray[np.uint8]:
    """Apply JET colormap heatmap over grayscale X-ray."""
    if base.ndim == 2:
        base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        base_bgr = base.copy()

    heatmap_norm = np.clip(heatmap, 0, 1)
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base_bgr, 1 - alpha, colored, alpha, 0)
    return overlay


def encode_image_base64(image: NDArray[np.uint8], fmt: str = ".png") -> str:
    import base64

    success, buffer = cv2.imencode(fmt, image)
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer).decode("utf-8")
