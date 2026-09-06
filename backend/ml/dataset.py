"""RSNA Pediatric Bone Age dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from ml.preprocessing import letterbox_resize, normalize_image


class BoneAgeDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path,
        image_size: int = 512,
        transform: Optional[Callable] = None,
        norm_mean: float = 0.4523,
        norm_std: float = 0.2118,
        target_mean: Optional[float] = None,
        target_std: Optional[float] = None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.transform = transform
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row["id"])

        possible_paths = [
            self.image_dir / f"{image_id}.png",
            self.image_dir / f"{image_id}.jpg",
            self.image_dir / f"{image_id}.jpeg",
        ]

        image = None
        for path in possible_paths:
            if path.exists():
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    break

        if image is None:
            raise FileNotFoundError(
                f"Image {image_id} not found in {self.image_dir}"
            )

        image = letterbox_resize(image, self.image_size)

        if self.transform:
            image = self.transform(image=image)["image"]

        image = normalize_image(
            image,
            self.norm_mean,
            self.norm_std,
        )

        image = torch.from_numpy(image).unsqueeze(0).float()

        raw_age = float(row["boneage"])
        age_tensor = torch.tensor(raw_age, dtype=torch.float32)

        # Standardized target if mean and std are provided
        if self.target_mean is not None and self.target_std is not None:
            norm_target = (raw_age - self.target_mean) / max(self.target_std, 1e-6)
            target_tensor = torch.tensor(norm_target, dtype=torch.float32)
        else:
            target_tensor = age_tensor

        # Handle male column robustly (boolean, string, or numeric)
        male_val = row.get("male", 0)
        if isinstance(male_val, (bool, np.bool_)):
            male_float = 1.0 if male_val else 0.0
        elif isinstance(male_val, str):
            male_float = 1.0 if male_val.strip().lower() in ("true", "1", "male", "m") else 0.0
        else:
            male_float = 1.0 if float(male_val) > 0.5 else 0.0

        male_tensor = torch.tensor(male_float, dtype=torch.float32)

        return {
            "image": image,
            "bone_age": age_tensor,        # Clinical ground-truth in months
            "target": target_tensor,        # Optimization target (standardized or raw)
            "male": male_tensor,
            "id": image_id,
        }


def stratified_split(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    df = df.copy()

    # Remove rows with missing bone age
    df = df.dropna(subset=["boneage"])

    # Create age bins covering all pediatric ranges (0 to 240 months)
    df["age_bin"] = pd.cut(
        df["boneage"],
        bins=[-1, 48, 96, 144, 192, 300],
        labels=["infant_toddler", "early_childhood", "middle_childhood", "adolescence", "late_adolescence"],
    )

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
        stratify=df["age_bin"],
    )

    return (
        train_df.drop(columns=["age_bin"]),
        val_df.drop(columns=["age_bin"]),
    )


def find_image_directory(data_dir: Path) -> Path:
    """Automatically locate the RSNA training image folder."""
    data_dir = Path(data_dir)
    if data_dir.is_file():
        data_dir = data_dir.parent

    possible_dirs = [
        data_dir / "boneage-training-dataset" / "boneage-training-dataset",
        data_dir / "boneage-training-dataset",
        data_dir / "train",
        data_dir / "images",
        data_dir,
    ]

    for d in possible_dirs:
        if d.exists() and d.is_dir():
            has_images = any(d.glob("*.png")) or any(d.glob("*.jpg")) or any(d.glob("*.jpeg"))
            if has_images:
                return d

    # Deep search for any directory with radiograph images
    if data_dir.exists() and data_dir.is_dir():
        for sub in data_dir.glob("**/"):
            if sub.is_dir():
                if any(sub.glob("*.png")) or any(sub.glob("*.jpg")):
                    return sub

    raise FileNotFoundError(f"Could not locate RSNA images in {data_dir}")


def create_dataloaders(
    data_dir: Path,
    csv_name="boneage-training-dataset.csv",
    batch_size=16,
    image_size=512,
    num_workers=2,
    train_transform=None,
    norm_mean=0.4523,
    norm_std=0.2118,
    normalize_target: bool = True,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    data_dir = Path(data_dir)
    if data_dir.is_file() and data_dir.suffix.lower() == ".csv":
        csv_path = data_dir
        data_dir = data_dir.parent
    else:
        possible_csv = [
            data_dir / "boneage-training-dataset.csv",
            data_dir / "train.csv",
            data_dir / csv_name,
        ]
        csv_path = None
        for p in possible_csv:
            if p.exists() and p.is_file():
                csv_path = p
                break

        if csv_path is None and data_dir.exists():
            for p in data_dir.glob("**/*.csv"):
                if "boneage" in p.name.lower() or "train" in p.name.lower():
                    csv_path = p
                    break

    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(f"No training CSV found in {data_dir}. Please check your dataset path.")

    df = pd.read_csv(csv_path)
    image_dir = find_image_directory(data_dir)

    train_df, val_df = stratified_split(df, val_ratio=val_ratio, seed=seed)

    # Calculate target statistics ONLY on training split to prevent data leakage
    if normalize_target:
        target_mean = float(train_df["boneage"].mean())
        target_std = float(train_df["boneage"].std())
    else:
        target_mean = None
        target_std = None

    train_ds = BoneAgeDataset(
        train_df,
        image_dir,
        image_size=image_size,
        transform=train_transform,
        norm_mean=norm_mean,
        norm_std=norm_std,
        target_mean=target_mean,
        target_std=target_std,
    )

    val_ds = BoneAgeDataset(
        val_df,
        image_dir,
        image_size=image_size,
        transform=None,
        norm_mean=norm_mean,
        norm_std=norm_std,
        target_mean=target_mean,
        target_std=target_std,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        train_df,
        val_df,
        target_mean,
        target_std,
    )
