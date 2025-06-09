import torchvision.models as models
import torch.nn as nn

from torchvision.models import ResNet50_Weights
class PhaseResNet50Model(nn.Module):
    def __init__(self, pretrained=True, num_classes=35, dropout_p=0.5):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc_phase = nn.Linear(2048, num_classes)
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.fc_phase(features)
        return features, logits