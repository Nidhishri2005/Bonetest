"""Bone age inference service."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from app.models.cnn_rf import CNNWithRFWrapper
from app.models.multimodal_cnn import MultimodalCNN
from app.services.model_registry import ModelRegistry, get_registry
from app.services.preprocessing import preprocess_for_inference

# Validation RMSE used for confidence calibration (from sample metrics)
MODEL_RMSE: dict[str, float] = {
    "cnn": 8.42,
    "cnn_dnn": 7.89,
    "multimodal_cnn": 7.15,
    "cnn_rf": 8.01,
}


def _compute_confidence(prediction: float, model_type: str) -> float:
    rmse = MODEL_RMSE.get(model_type, 10.0)
    # Higher confidence when prediction is within typical pediatric range
    rmse = MODEL_RMSE.get(model_type, 11.5)
    # Higher confidence when prediction is within typical pediatric range (1-228 months)
    range_penalty = 0.0
    if prediction < 12 or prediction > 228:
        range_penalty = 0.15
    base = max(0.0, 1.0 - rmse / 20.0)
    return float(np.clip(base - range_penalty, 0.55, 0.98))
    if prediction < 1 or prediction > 228:
        range_penalty = 0.20
    base = max(0.0, 1.0 - rmse / 25.0)
    return float(np.clip(base - range_penalty, 0.50, 0.98))


class InferenceService:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or get_registry()

    def predict(
        self,
        image_bytes: bytes,
        model_type: str,
        gender: Optional[str] = None,
    ) -> dict:
        start = time.perf_counter()
        tensor, _ = preprocess_for_inference(image_bytes)
        tensor = tensor.to(self.registry.device)

        model = self.registry.load(model_type)

        with torch.no_grad():
            if isinstance(model, CNNWithRFWrapper):
                features = model.cnn(tensor)
                pred = model.predict_numpy(features.cpu().numpy().flatten())
            elif isinstance(model, MultimodalCNN):
                gender_val = 1.0 if gender == "male" else 0.0
                gender_tensor = torch.tensor(
                    [[gender_val]], device=self.registry.device, dtype=tensor.dtype
                )
                pred = model(tensor, gender_tensor).item()
                target_mean, target_std = self.registry.get_target_scaler(model_type)
                if target_mean is not None and target_std is not None:
                    pred = pred * target_std + target_mean
            else:
                pred = model(tensor).item()
                target_mean, target_std = self.registry.get_target_scaler(model_type)
                if target_mean is not None and target_std is not None:
                    pred = pred * target_std + target_mean

        elapsed_ms = (time.perf_counter() - start) * 1000
        confidence = _compute_confidence(pred, model_type)

        return {
            "model_type": model_type,
            "bone_age_months": round(pred, 1),
            "confidence": round(confidence, 3),
            "gender_used": gender if model_type == "multimodal_cnn" else None,
            "processing_time_ms": round(elapsed_ms, 2),
        }


_inference_service: InferenceService | None = None


def get_inference_service() -> InferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service
