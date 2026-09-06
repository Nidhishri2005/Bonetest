"""Training-time image augmentation."""

from __future__ import annotations

import cv2
import numpy as np


class TrainAugmentation:
    """Medically appropriate augmentation pipeline for pediatric hand X-rays."""

    def __init__(
        self,
        flip_prob: float = 0.5,
        rotate_limit: int = 10,
        brightness_limit: float = 0.12,
        contrast_limit: float = 0.12,
        shift_limit: float = 0.04,
        scale_limit: float = 0.05,
        noise_std: float = 0.005,
    ) -> None:
        self.flip_prob = flip_prob
        self.rotate_limit = rotate_limit
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.noise_std = noise_std

    def __call__(self, image: np.ndarray) -> dict:
        img = image.copy()
        h, w = img.shape[:2]

        # 1. Horizontal flip (clinically symmetric for bone maturation)
        if self.flip_prob > 0 and np.random.random() < self.flip_prob:
            img = cv2.flip(img, 1)

        # 2. Small rotation and subtle translation with black border (never reflect bone borders)
        if self.rotate_limit > 0 or self.scale_limit > 0 or self.shift_limit > 0:
            angle = float(np.random.uniform(-self.rotate_limit, self.rotate_limit)) if self.rotate_limit > 0 else 0.0
            scale = float(1.0 + np.random.uniform(-self.scale_limit, self.scale_limit)) if self.scale_limit > 0 else 1.0
            matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)

            if self.shift_limit > 0:
                matrix[0, 2] += float(np.random.uniform(-self.shift_limit, self.shift_limit) * w)
                matrix[1, 2] += float(np.random.uniform(-self.shift_limit, self.shift_limit) * h)

            img = cv2.warpAffine(
                img,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        # 3. Subtle brightness and contrast variation
        alpha = float(1.0 + np.random.uniform(-self.contrast_limit, self.contrast_limit))
        beta = float(np.random.uniform(-self.brightness_limit, self.brightness_limit) * 255.0)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # 4. Mild Gaussian noise
        if self.noise_std > 0 and np.random.random() < 0.3:
            noise = np.random.normal(0, self.noise_std * 255, img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return {"image": img}
