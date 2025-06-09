import torchvision.models as models
import torch.nn as nn
import torch
from torchvision.models import ResNet50_Weights

class PhaseResNet50Model(nn.Module):
    """
    ResNet-50 backbone for phase recognition on APTOS.

    Extracts 2048-dim features for each frame and predicts logits for 35 phases.
    """
    def __init__(self,
                 pretrained: bool = True,
                 num_classes: int = 35):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone # (N, 2048)
        self.fc_phase = nn.Linear(2048, num_classes) #(N, 35)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.fc_phase(features)
        return features, logits