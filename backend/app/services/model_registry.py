"""Lazy-loading model registry for inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import torch

from app.core.config import Settings, get_settings
from app.models import MODEL_REGISTRY, SUPPORTED_MODELS
from app.models.cnn_rf import CNNFeatureExtractor, CNNWithRFWrapper

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._models: Dict[str, Any] = {}
        self._target_scalers: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        self._device = torch.device(
            self.settings.device
            if torch.cuda.is_available() or self.settings.device == "cpu"
            else "cpu"
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def checkpoint_path(self, model_type: str) -> Path:
        return self.settings.checkpoint_dir / f"{model_type}_best.pt"

    def rf_path(self) -> Path:
        return self.settings.rf_models_dir / "cnn_rf.joblib"

    def is_checkpoint_available(self, model_type: str) -> bool:
        if model_type not in SUPPORTED_MODELS:
            return False
        if not self.checkpoint_path(model_type).exists():
            return False
        if model_type == "cnn_rf" and not self.rf_path().exists():
            return False
        return True

    def get_target_scaler(self, model_type: str) -> Tuple[Optional[float], Optional[float]]:
        return self._target_scalers.get(model_type, (None, None))

    def list_models(self) -> list[dict]:
        result = []
        for model_id, meta in MODEL_REGISTRY.items():
            result.append(
                {
                    "id": model_id,
                    "display_name": meta["display_name"],
                    "description": meta["description"],
                    "requires_gender": meta["requires_gender"],
                    "supports_gradcam": meta["supports_gradcam"],
                    "checkpoint_available": self.is_checkpoint_available(model_id),
                }
            )
        return result

    def load(self, model_type: str) -> Any:
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}")

        if model_type in self._models:
            return self._models[model_type]

        ckpt_path = self.checkpoint_path(model_type)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for '{model_type}' at {ckpt_path}. "
                "Train models first or copy checkpoints into backend/checkpoints/."
            )

        meta = MODEL_REGISTRY[model_type]
        model_class = meta["class"]
        model = model_class(pretrained=False)

        checkpoint = torch.load(ckpt_path, map_location=self._device, weights_only=False)
        backbone = checkpoint.get("backbone", "resnet18")

        try:
            model = model_class(pretrained=False, backbone_name=backbone)
        except TypeError:
            model = model_class(pretrained=False)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.to(self._device)
        model.eval()

        target_mean = checkpoint.get("target_mean", None)
        target_std = checkpoint.get("target_std", None)

        if target_mean is None:
            norm_json = self.settings.checkpoint_dir / "normalization.json"
            if norm_json.exists():
                try:
                    with open(norm_json) as f:
                        stats = json.load(f)
                    target_mean = stats.get("target_mean")
                    target_std = stats.get("target_std")
                except Exception:
                    pass

        self._target_scalers[model_type] = (target_mean, target_std)

        if model_type == "cnn_rf":
            rf_path = self.rf_path()
            if not rf_path.exists():
                raise FileNotFoundError(f"Random Forest model not found at {rf_path}")
            rf_model = joblib.load(rf_path)
            wrapper = CNNWithRFWrapper(model, rf_model)
            wrapper = CNNWithRFWrapper(model, rf_model, target_mean=target_mean, target_std=target_std)
            self._models[model_type] = wrapper
            logger.info("Loaded model: %s (CNN + RF)", model_type)
            return wrapper

        self._models[model_type] = model
        logger.info("Loaded model: %s", model_type)
        return model

    def get_pytorch_module(self, model_type: str) -> torch.nn.Module:
        loaded = self.load(model_type)
        if isinstance(loaded, CNNWithRFWrapper):
            return loaded.cnn
        return loaded


_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
