import torch
import torch.nn as nn
import torch.nn.functional as F



class CNNClassifier(nn.Module):
    """Minimal CNN for 32x32 grayscale digit classification."""
    def __init__(self, num_classes: int = 10):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)   # single conv, 8 filters
        self.fc1 = nn.Linear(8 * 14 * 14, 32)         # very small hidden
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)        # (B, 8, 14, 14)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


