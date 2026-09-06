"""Grad-CAM explainability for CNN-based models."""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.models.cnn_rf import CNNWithRFWrapper
from app.models.multimodal_cnn import MultimodalCNN
from app.services.inference_service import InferenceService, get_inference_service
from app.services.model_registry import get_registry
from app.services.preprocessing import (
    encode_image_base64,
    overlay_heatmap,
    preprocess_for_inference,
)


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hooks = []

        def forward_hook(_module, _input, output) -> None:
            self.activations = output.detach()

        def backward_hook(_module, _grad_input, grad_output) -> None:
            self.gradients = grad_output[0].detach()

        self._hooks.append(target_layer.register_forward_hook(forward_hook))
        self._hooks.append(
            target_layer.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()

    def generate(self, input_tensor: torch.Tensor, gender_tensor=None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        input_tensor = input_tensor.clone().requires_grad_(True)

        if isinstance(self.model, MultimodalCNN):
            output = self.model(input_tensor, gender_tensor)
        else:
            output = self.model(input_tensor)

        if output.dim() == 0:
            output = output.unsqueeze(0)

        self.model.zero_grad(set_to_none=True)
        output.backward(torch.ones_like(output), retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.astype(np.float32)


class GradCamService:
    def __init__(self, inference: InferenceService | None = None) -> None:
        self.inference = inference or get_inference_service()
        self.registry = get_registry()

    def generate(
        self,
        image_bytes: bytes,
        model_type: str,
        gender: Optional[str] = None,
    ) -> dict:
        start = time.perf_counter()
        meta = self.registry.list_models()
        model_meta = next((m for m in meta if m["id"] == model_type), None)
        if model_meta and not model_meta["supports_gradcam"]:
            raise ValueError(f"Model '{model_type}' does not support Grad-CAM")

        tensor, display = preprocess_for_inference(image_bytes)
        tensor = tensor.to(self.registry.device)

        pytorch_module = self.registry.get_pytorch_module(model_type)
        target_layer = pytorch_module.get_gradcam_target_layer()

        gender_tensor = None
        if isinstance(pytorch_module, MultimodalCNN):
            gender_val = 1.0 if gender == "male" else 0.0
            gender_tensor = torch.tensor(
                [[gender_val]], device=self.registry.device, dtype=tensor.dtype
            )

        pytorch_module.eval()
        gradcam = GradCAM(pytorch_module, target_layer)

        try:
            heatmap = gradcam.generate(tensor, gender_tensor)
        finally:
            gradcam.remove_hooks()

        prediction = self.inference.predict(image_bytes, model_type, gender)

        resized_display = cv2.resize(
            display, (tensor.shape[3], tensor.shape[2])
        )
        if display.shape[:2] != (tensor.shape[2], tensor.shape[3]):
            resized_display = cv2.resize(
                display, (tensor.shape[3], tensor.shape[2])
            )
        else:
            resized_display = display
        overlay = overlay_heatmap(resized_display, heatmap)
        heatmap_color = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "model_type": model_type,
            "bone_age_months": prediction["bone_age_months"],
            "confidence": prediction["confidence"],
            "heatmap_base64": encode_image_base64(heatmap_color),
            "overlay_base64": encode_image_base64(overlay),
            "processing_time_ms": round(elapsed_ms, 2),
        }


_gradcam_service: GradCamService | None = None


def get_gradcam_service() -> GradCamService:
    global _gradcam_service
    if _gradcam_service is None:
        _gradcam_service = GradCamService()
    return _gradcam_service
