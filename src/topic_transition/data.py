"""Data functions."""
import random

import pandas as pd

random.seed(42)


def get_splits(df: pd.DataFrame, shuffle: bool = True, train_ratio: float = 0.8) -> tuple[list[int], list[int]]:
    """
    Split the DataFrame indices into training and validation sets based on unique dates.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be split.
    shuffle : bool, optional
        Whether to shuffle the indices within each date (default is True).
    train_ratio : float, optional
        The ratio of samples to be included in the training set (default is 0.8).

    Returns
    -------
    tuple[list[int], list[int]]
        A tuple containing two lists of indices: train_idxs (indices for the training set)
        and val_idxs (indices for the validation set).
    """
    train_idxs = []
    val_idxs = []
    days = df["date"].unique()
    for day in days:
        day_idxs = list(df[df["date"] == day].index)
        if shuffle:
            random.shuffle(day_idxs)
        train_size = int(len(day_idxs) * train_ratio)
        val_size = len(day_idxs) - train_size
        if train_size == 0 or val_size == 0:
            train_list = []
            val_list = []
            for item in day_idxs:
                if random.random() < train_ratio:
                    train_list.append(item)
                else:
                    val_list.append(item)
        else:
            train_list = day_idxs[:train_size]
            val_list = day_idxs[train_size:]
        train_idxs.extend(train_list)
        val_idxs.extend(val_list)
    return train_idxs, val_idxs
