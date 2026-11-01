"""Small CNN shared by training and inference. No Qt dependency."""

import torch
from torch import nn
from torch.nn import functional as F


class DigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.hidden = nn.Linear(32 * 7 * 7, 64)
        self.output = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.15)

    def forward_with_features(self, x: torch.Tensor):
        first = F.max_pool2d(F.relu(self.conv1(x)), 2)
        second = F.max_pool2d(F.relu(self.conv2(first)), 2)
        hidden = F.relu(self.hidden(second.flatten(1)))
        logits = self.output(self.dropout(hidden))
        return logits, first, second, hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(x)[0]
