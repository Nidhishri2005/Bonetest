import torch
import torch.nn as nn
from torchvision import models


def _adapt_resnet_input_conv(resnet: nn.Module) -> None:
    """Replace first conv layer for single-channel grayscale X-rays while preserving ImageNet weights."""
    old_conv = resnet.conv1
    new_conv = nn.Conv2d(
        1,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()
    resnet.conv1 = new_conv


class CNNBaseline(nn.Module):
    """Model A: ResNet backbone with linear regression head."""

    def __init__(self, pretrained: bool = False, backbone_name: str = "resnet18") -> None:
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        else:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)

        _adapt_resnet_input_conv(self.backbone)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.backbone.layer4[-1]

