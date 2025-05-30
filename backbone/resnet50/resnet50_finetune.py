import torchvision.models as models
import torch.nn as nn
import torch
from torchsummary import summary
from torchvision.models import ResNet50_Weights
# feel free to add to the list: https://pytorch.org/docs/stable/torchvision/models.html
from datetime import datetime

### TWO HEAD MODELS ###

class PhaseResNet50Model(nn.Module):
    """
    ResNet-50 backbone for phase recognition on APTOS.

    Extracts 2048-dim features for each frame and predicts logits for 35 phases.
    """
    def __init__(self,
                 pretrained: bool = True,
                 num_classes: int = 35):
        super().__init__()
        # Load a ResNet-50 pretrained on ImageNet
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        # Remove original fully-connected layer
        backbone.fc = nn.Identity()
        self.backbone = backbone # (N, 2048)
        # New classification head for phases
        self.fc_phase = nn.Linear(2048, num_classes) #(N, 35)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (Tensor): batch of images, shape (N, 3, 224, 224)
        Returns:
            features (Tensor): extracted features, shape (N, 2048)
            logits   (Tensor): class logits,      shape (N, num_classes)
        """
        features = self.backbone(x)
        logits = self.fc_phase(features)
        return features, logits