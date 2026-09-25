import torch
import torch.nn as nn
from torchvision.models import resnet18


class CIFARResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        backbone = resnet18(weights=None)

        backbone.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        backbone.maxpool = nn.Identity()

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward_until_layer2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        return x

    def forward_from_layer2(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.fc(features)

        return logits, features

    def forward(self, x, return_features=False):
        x = self.forward_until_layer2(x)
        logits, features = self.forward_from_layer2(x)

        if return_features:
            return logits, features

        return logits