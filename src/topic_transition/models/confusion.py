"""Definition for the Confusion models."""
import torch
import torch.nn as nn


class FFConfusion(nn.Module):
    """Simple confusion model."""

    def __init__(self, input_dim: int, categories: int):
        """Init Simplest confusion model."""
        super(FFConfusion, self).__init__()
        self.fc1 = nn.Linear(input_dim, categories)

    def forward(self, input_features: torch.Tensor):
        """Forward."""
        output = torch.sigmoid(self.fc1(input_features))
        return output
