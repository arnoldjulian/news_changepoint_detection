"""Definition for the Confusion model."""
import torch
import torch.nn as nn


class FFConfusion(nn.Module):
    """Feedforward confusion model."""

    def __init__(self, embedding_dim: int, grid_size: int):
        """Create model layers.

        Parameters
        ----------
        embedding_dim
            Size of the input embedding.
        grid_size
            Number of discrete values for the tuning parameter in the confusion method.

        """
        super(FFConfusion, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, grid_size)

    def forward(self, input_features: torch.Tensor):
        """Perform a forward pass and return a prediction for each value of the tuning parameter."""
        output = torch.sigmoid(self.fc1(input_features))
        return output
