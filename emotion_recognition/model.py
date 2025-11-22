import torch
import torch.nn as nn

class MiniXceptionResNet(nn.Module):
    """Legacy placeholder retained for backwards compatibility.
    Internally delegates to CNNEmotionNet architecture requested by user.
    """
    def __init__(self, num_classes=7, in_channels=1):
        super().__init__()
        self.model = CNNEmotionNet(num_classes=num_classes, in_channels=in_channels)

    def forward(self, x):
        return self.model(x)

class CNNEmotionNet(nn.Module):
    """CNN architecture matching provided Keras summary with dynamic FC sizing.

    Handles arbitrary square input >=48 by inferring flattened feature size at runtime.
    This avoids shape mismatch (e.g. 64x64 -> 4608 features vs 48x48 -> 2048).
    """
    def __init__(self, num_classes=7, in_channels=1, dropout_conv=0.25, dropout_mid=0.25, dropout_fc=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(dropout_conv)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=0)
        self.pool3 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(dropout_mid)

        # Defer creation of fc1 until first forward pass (lazy) to match input size.
        self.fc1 = None
        self.drop_fc = nn.Dropout(dropout_fc)
        self.fc_out = nn.Linear(1024, num_classes)

        self.act = nn.ReLU(inplace=True)

    def _ensure_fc1(self, flattened_dim, device):
        if self.fc1 is None:
            self.fc1 = nn.Linear(flattened_dim, 1024).to(device)

    def forward(self, x):
        x = self.act(self.conv1(x))      # (N,32,H-2,H-2)
        x = self.act(self.conv2(x))      # (N,64,H-4,H-4)
        x = self.pool1(x)                # (N,64,(H-4)/2,(H-4)/2)
        x = self.drop1(x)
        x = self.act(self.conv3(x))      # (N,128, ... )
        x = self.pool2(x)
        x = self.act(self.conv4(x))
        x = self.pool3(x)
        x = self.drop2(x)
        x = torch.flatten(x, 1)
        self._ensure_fc1(x.size(1), x.device)
        x = self.act(self.fc1(x))
        x = self.drop_fc(x)
        return self.fc_out(x)
