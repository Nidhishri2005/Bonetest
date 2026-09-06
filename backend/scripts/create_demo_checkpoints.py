"""Create untrained demo checkpoints for local inference testing.

These weights are randomly initialized — predictions will not be clinically
accurate. Replace with trained checkpoints from ml.train for production use.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.models import MODEL_REGISTRY, SUPPORTED_MODELS
from app.models.cnn_rf import CNNFeatureExtractor


def main() -> None:
    ckpt_dir = BACKEND_ROOT / "checkpoints"
    rf_dir = BACKEND_ROOT / "rf_models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rf_dir.mkdir(parents=True, exist_ok=True)

    for model_type in SUPPORTED_MODELS:
        if model_type == "cnn_rf":
            continue
        meta = MODEL_REGISTRY[model_type]
        model = meta["class"](pretrained=False)
        # Initialize final bias to clinical mean to prevent near-zero outputs
        if hasattr(model, "backbone") and hasattr(model.backbone, "fc"):
            torch.nn.init.constant_(model.backbone.fc.bias, 0.0)
        elif hasattr(model, "head") and hasattr(model.head[-1], "bias"):
            torch.nn.init.constant_(model.head[-1].bias, 0.0)
        elif hasattr(model, "fusion_head") and hasattr(model.fusion_head[-1], "bias"):
            torch.nn.init.constant_(model.fusion_head[-1].bias, 0.0)

        path = ckpt_dir / f"{model_type}_best.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_type": model_type,
                "note": "Demo checkpoint — train with ml.train for real weights",
                "target_mean": 127.32,
                "target_std": 41.18,
                "norm_mean": 0.4523,
                "norm_std": 0.2118,
                "note": "Demo checkpoint — train with ml.train for trained weights",
            },
            path,
        )
        print(f"Created {path}")

    # CNN + RF
    cnn = CNNFeatureExtractor(pretrained=False)
    torch.save(
        {"model_state_dict": cnn.state_dict(), "model_type": "cnn_rf"},
        ckpt_dir / "cnn_rf_best.pt",
    )

    rng = np.random.RandomState(42)
    X_dummy = rng.randn(100, 512)
    y_dummy = rng.uniform(24, 180, 100)
    rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X_dummy, y_dummy)
    joblib.dump(rf, rf_dir / "cnn_rf.joblib")
    print(f"Created demo RF model at {rf_dir / 'cnn_rf.joblib'}")
    print("\nDemo checkpoints ready. Start the API with: uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
