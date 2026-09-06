import torch
import torch.nn as nn
from torchvision import models

from app.models.cnn import _adapt_resnet_input_conv


class CNNFeatureExtractor(nn.Module):
    """CNN backbone that outputs feature vectors for Random Forest regression."""

    def __init__(self, pretrained: bool = False, backbone_name: str = "resnet18") -> None:
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
        else:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)

        _adapt_resnet_input_conv(backbone)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        return features

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.backbone.layer4[-1]


class CNNWithRFWrapper:
    """Model D: PyTorch feature extractor + sklearn Random Forest."""

    def __init__(
        self,
        cnn: CNNFeatureExtractor,
        rf_model,
        target_mean: float | None = None,
        target_std: float | None = None,
    ) -> None:
        self.cnn = cnn
        self.rf_model = rf_model
        self.target_mean = target_mean
        self.target_std = target_std

    def predict_numpy(self, features) -> float:
        raw_pred = float(self.rf_model.predict(features.reshape(1, -1))[0])
        if self.target_mean is not None and self.target_std is not None:
            return raw_pred * self.target_std + self.target_mean
        return raw_pred
