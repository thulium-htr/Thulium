import torch.nn as nn
import torch

class ResNetBackbone(nn.Module):
    """
    Standard ResNet-style backbone for extracting features from line images.
    """
    def __init__(self, in_channels=3, output_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Simplified blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, output_channels, 2, stride=2)

    def _make_layer(self, in_planes, planes, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            features: (B, output_channels, H', W')
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
