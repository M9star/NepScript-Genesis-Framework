import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    """Simple CNN for 32x32 grayscale digit classification (10 classes)."""
    def __init__(self, num_classes: int = 10):
        super(CNNClassifier, self).__init__()
        # Convolutional blocks
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Fully‑connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # after two poolings, 32->16->8
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Input shape: (B, 1, 32, 32)
        x = F.relu(self.bn1(self.conv1(x)) )
        x = F.max_pool2d(x, 2)               # (B, 32, 16, 16)
        x = F.relu(self.bn2(self.conv2(x)) )
        x = F.max_pool2d(x, 2)               # (B, 64, 8, 8)
        x = x.view(x.size(0), -1)            # flatten
        x = self.dropout(F.relu(self.fc1(x)))
        logits = self.fc2(x)
        return logits
