"""Image and target preprocessing utilities for bone age prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def letterbox_resize(
    image: np.ndarray,
    target_size: int = 512,
    fill_value: int = 0,
) -> np.ndarray:
    """Resize an image preserving its aspect ratio by padding with fill_value."""
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)

    if len(image.shape) == 2:
        canvas = np.full((target_size, target_size), fill_value, dtype=image.dtype)
        top = (target_size - nh) // 2
        left = (target_size - nw) // 2
        canvas[top : top + nh, left : left + nw] = resized
    else:
        canvas = np.full(
            (target_size, target_size, image.shape[2]),
            fill_value,
            dtype=image.dtype,
        )
        top = (target_size - nh) // 2
        left = (target_size - nw) // 2
        canvas[top : top + nh, left : left + nw, :] = resized

    return canvas


def normalize_image(
    image: np.ndarray,
    mean: float = 0.4523,
    std: float = 0.2118,
) -> np.ndarray:
    """Normalize pixel values: [0, 255] -> [0.0, 1.0] -> z-score standardisation."""
    img = image.astype(np.float32) / 255.0
    return (img - mean) / std


def denormalize_image(
    image: np.ndarray,
    mean: float = 0.4523,
    std: float = 0.2118,
) -> np.ndarray:
    """Invert z-score standardisation and return uint8 image [0, 255]."""
    img = image * std + mean
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def standardize_target(
    target: float | np.ndarray | torch.Tensor,
    mean: float,
    std: float,
) -> float | np.ndarray | torch.Tensor:
    """Standardize bone age target: (y - mean) / std."""
    return (target - mean) / std


def inverse_transform_target(
    pred: float | np.ndarray | torch.Tensor,
    mean: float,
    std: float,
) -> float | np.ndarray | torch.Tensor:
    """Convert standardized prediction back to original units (months)."""
    return pred * std + mean


def compute_dataset_statistics(
    data_dir: Path,
    image_size: int = 512,
) -> dict:
    """Compute dataset pixel mean and std."""
    data_dir = Path(data_dir)
    possible_csv = [
        data_dir / "boneage-training-dataset.csv",
        data_dir / "train.csv",
    ]

    csv_path = None
    for p in possible_csv:
        if p.exists():
            csv_path = p
            break

    if csv_path is None:
        raise FileNotFoundError("Training CSV not found.")

    df = pd.read_csv(csv_path)

    possible_dirs = [
        data_dir / "boneage-training-dataset" / "boneage-training-dataset",
        data_dir / "boneage-training-dataset",
        data_dir / "train",
        data_dir,
    ]

    image_dir = None
    for d in possible_dirs:
        if d.exists() and d.is_dir():
            if any(d.glob("*.png")) or any(d.glob("*.jpg")):
                image_dir = d
                break

    if image_dir is None:
        raise FileNotFoundError("Image directory not found.")

    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    total_pixels = 0

    print("Computing dataset statistics...")
    for image_id in tqdm(df["id"]):
        image_path = image_dir / f"{image_id}.png"
        if not image_path.exists():
            image_path = image_dir / f"{image_id}.jpg"

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        image = letterbox_resize(image, image_size)
        image = image.astype(np.float32) / 255.0

        pixel_sum += float(image.sum())
        pixel_sq_sum += float(np.square(image).sum())
        total_pixels += image.size

    mean = pixel_sum / total_pixels
    std = float(np.sqrt((pixel_sq_sum / total_pixels) - (mean ** 2)))

    return {
        "mean": float(mean),
        "std": float(std),
    }


def save_normalization_stats(
    stats: dict,
    output_path: Path,
) -> None:
    """Save normalization and target statistics to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)


def load_normalization_stats(path: Path) -> dict:
    """Load normalization statistics."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
