import torch.nn as nn
from torchvision.models import resnet18, resnet34


class _DashboardResNet(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool = False, dropout: float = 0.2):
        super().__init__()
        if backbone_name == "resnet18":
            self.backbone = resnet18(pretrained=pretrained)
        elif backbone_name == "resnet34":
            self.backbone = resnet34(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


class Resnet18_dashboard(_DashboardResNet):
    def __init__(self, num_classes: int, pretrained: bool = False, dropout: float = 0.2):
        super().__init__(
            backbone_name="resnet18",
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )


class Resnet34_dashboard(_DashboardResNet):
    def __init__(self, num_classes: int, pretrained: bool = False, dropout: float = 0.2):
        super().__init__(
            backbone_name="resnet34",
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )
