"""
Defines the CNN architecture used for MNIST digit classification
across all federated clients and the central server.

Architecture:
  Conv2d(1→32) → ReLU → MaxPool
  Conv2d(32→64) → ReLU → MaxPool
  Flatten
  Linear(1600→128) → ReLU → Dropout(0.5)
  Linear(128→10)  ← 10 digit classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """
    A lightweight CNN for MNIST digit classification.
    Shared across all FL clients — every client trains this
    same architecture on its local data partition.
    """

    def __init__(self) -> None:
        super(MNISTNet, self).__init__()

        # --- Feature extractor ---
        # Block 1: 1×28×28 → 32×12×12
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=5, padding=2
        )  # keeps spatial size at 28×28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # → 32×14×14

        # Block 2: 32×14×14 → 64×7×7
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, padding=2
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # → 64×7×7

        # --- Classifier head ---
        # 64 × 7 × 7 = 3136 features after flatten
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=0.5)   # regularisation
        self.fc2 = nn.Linear(128, 10)      # 10 output classes (digits 0-9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input x shape: (batch, 1, 28, 28)."""
        # Feature extraction
        x = self.pool1(F.relu(self.conv1(x)))   # → (B, 32, 14, 14)
        x = self.pool2(F.relu(self.conv2(x)))   # → (B, 64, 7, 7)

        # Flatten spatial dimensions
        x = x.view(-1, 64 * 7 * 7)             # → (B, 3136)

        # Classification
        x = F.relu(self.fc1(x))                 # → (B, 128)
        x = self.dropout(x)
        x = self.fc2(x)                          # → (B, 10) — raw logits
        return x


def get_model() -> MNISTNet:
    """Factory helper: returns a freshly initialised MNISTNet."""
    return MNISTNet()
