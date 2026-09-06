import torch
import torch.nn as nn
from torchvision import models

from app.models.cnn import _adapt_resnet_input_conv


class MultimodalCNN(nn.Module):
    """Model C: Image CNN branch fused with gender metadata."""

    def __init__(
        self,
        pretrained: bool = False,
        meta_dim: int = 1,
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
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_backbone = backbone

        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        fusion_in = self.feature_dim + 32
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(
        self, x: torch.Tensor, gender: torch.Tensor | None = None
    ) -> torch.Tensor:
        image_features = self.image_backbone(x)
        if image_features.dim() > 2:
            image_features = torch.flatten(image_features, 1)

        if gender is None:
            gender = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        elif gender.dim() == 1:
            gender = gender.unsqueeze(-1)
        gender = gender.float()

        meta_features = self.meta_encoder(gender)
        fused = torch.cat([image_features, meta_features], dim=1)
        return self.fusion_head(fused).squeeze(-1)

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.image_backbone.layer4[-1]
