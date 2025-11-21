# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def depthwise_separable_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class MiniXception(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

        self.block1 = depthwise_separable_conv(8, 16)
        self.block2 = depthwise_separable_conv(16, 32)
        self.block3 = depthwise_separable_conv(32, 64)
        self.block4 = depthwise_separable_conv(64, 128)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
