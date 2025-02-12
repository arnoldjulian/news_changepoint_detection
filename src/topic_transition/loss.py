"""Loss-related tools."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class BCEWithWeights(nn.Module):
    """Loss with weights."""

    def __init__(self, data: pd.DataFrame, dtype: torch.dtype, device: torch.device):
        """
        Initialize an instance of the BCEWithWeights class.

        Parameters
        ----------
        data : pd.DataFrame
            The input data as a pandas DataFrame.
        dtype : torch.dtype
            The data type for calculations.
        device : torch.device
            The device (CPU or GPU) to perform calculations on.
        """
        super(BCEWithWeights, self).__init__()
        self.dtype = dtype
        self.device = device
        labels = np.stack(data["label"].values)  # type: ignore
        self.num_splits = labels.shape[1]
        self.pos_ratios = torch.empty((self.num_splits,), device=device, dtype=dtype)
        for i in range(self.num_splits):
            pos_train_num = np.sum(labels[:, i] == 1)
            neg_train_num = np.sum(labels[:, i] == 0)
            self.pos_ratios[i] = torch.tensor(neg_train_num / pos_train_num, dtype=dtype, device=device)

    def __call__(self, model_outputs: torch.Tensor, targets: torch.Tensor, **kwargs: dict) -> torch.Tensor:
        """
        Calculate loss.

        Parameters
        ----------
        model_outputs : torch.Tensor
            The output tensor from the model, representing the predicted probabilities.
        targets : torch.Tensor
            The tensor containing the true target values.
        **kwargs : dict
            Additional optional arguments.

        Returns
        -------
        torch.Tensor
            The mean loss calculated using binary cross-entropy.
        """
        if "split_idx" in kwargs:
            split_idx = int(kwargs["split_idx"])  # type: ignore
            losses = -1 * (
                self.pos_ratios[split_idx] * targets * torch.log(model_outputs)
                + (1 - targets) * torch.log(1 - model_outputs)
            )
        else:
            losses = -1 * (
                self.pos_ratios.reshape(1, self.num_splits) * targets * torch.log(model_outputs)
                + (1 - targets) * torch.log(1 - model_outputs)
            )
        if "loss_mask" in kwargs:
            loss_mask = kwargs["loss_mask"]  # type: ignore
            losses = losses * loss_mask
        aggregate = True
        if "aggregate" in kwargs:
            aggregate = kwargs["aggregate"]  # type: ignore
        if aggregate:
            return losses.mean()
        else:
            return losses
