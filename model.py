import torch
import torch.nn as nn
from torchvision import models


class PlantNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
    ):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self._setup_frozen_layers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

    def _setup_frozen_layers(self):
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
