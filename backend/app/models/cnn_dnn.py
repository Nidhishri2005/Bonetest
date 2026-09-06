import torch
import torch.nn as nn
from torchvision import models

from app.models.cnn import _adapt_resnet_input_conv


class CNNDNN(nn.Module):
    """Model B: ResNet feature extractor + deep neural network head."""

    def __init__(
        self,
        pretrained: bool = False,
        dropout: float = 0.25,
        backbone_name: str = "resnet18",
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
        else:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)

        _adapt_resnet_input_conv(backbone)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        return self.head(features).squeeze(-1)

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.backbone.layer4[-1]
