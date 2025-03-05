"""Tools for training confusion model."""
import os
import pickle
from datetime import date, datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from scipy.sparse import csr_matrix
from torch import Tensor
from torch import nn as nn
from torch import optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from topic_transition.data import get_splits
from topic_transition.loss import BCEWithWeights
from topic_transition.models.confusion import FFConfusion
from topic_transition.utils import get_dates_for_interval
from topic_transition.vectorizers import get_vectorizer


def get_feedforward_dataloader(
    data: pd.DataFrame, idxs: list[int], batch_size: int, shuffle: bool, dtype: torch.dtype
) -> DataLoader:
    """
    Create dataloader.

    Parameters
    ----------
    dtype
    data : pd.DataFrame
        The input data containing vectors, labels and dates.
    idxs : list[int]
        The list of indices to select from the data.
    batch_size : int
        The batch size for the training dataloader.
    shuffle : bool
        Whether to shuffle the data during training.

    Returns
    -------
    torch.utils.data.DataLoader
        The training dataloader containing selected tensors, labels and dates.
    """
    if isinstance(data.iloc[0]["vector"], csr_matrix):
        selected_vectors = scipy.sparse.vstack(data.loc[idxs, "vector"].values)
        selected_tensors = torch.tensor(selected_vectors.toarray(), dtype=dtype)
    else:
        selected_vectors = np.stack(data.loc[idxs, "vector"].values)  # type: ignore
        selected_tensors = torch.tensor(selected_vectors, dtype=dtype)

    selected_labels = np.stack(data.loc[idxs, "label"].values)  # type: ignore
    selected_labels = torch.tensor(selected_labels, dtype=dtype)

    # Extract dates
    selected_dates = data.loc[idxs, "date"]
    timestamps = torch.tensor([pd.to_datetime(dt).timestamp() for dt in selected_dates])
    dataset = TensorDataset(selected_tensors, selected_labels, timestamps)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_feedforward_loaders(
    batch_size: int,
    data: pd.DataFrame,
    torch_dtype: torch.dtype,
    train_idxs: list[int],
    val_idxs: list[int],
    config: dict,
    dataset_path: str,
    vectorizer_type: str,
) -> tuple[DataLoader, DataLoader | None, int]:
    """
    Get dataloaders for non-llm model.

    Parameters
    ----------
    batch_size : int
        The batch size for the data loaders.
    data : pd.DataFrame
        The input data containing the full text.
    torch_dtype : torch.dtype
        The torch data type to be used.
    train_idxs : list
        The list of indices for training data.
    val_idxs : list
        The list of indices for validation data.
    config : dict
        The configuration settings for training.
    vectorizer_config : dict
        The configuration settings for the vectorizer.

    Returns
    -------
    Tuple[DataLoader, Union[DataLoader, None], sparse matrix]
        A tuple containing the training data loader, validation data loader (if applicable), and the vectorized data.
    """
    vectors_path = dataset_path.replace("pkl", "tfidf")
    if vectorizer_type == "tfidf" and os.path.exists(vectors_path):
        with open(vectors_path, "rb") as f:
            vectors = pickle.load(f)
    else:
        vectorizer = get_vectorizer(vectorizer_type)
        vectors = vectorizer.fit_transform(data["full_text"])
        if vectorizer_type == "tfidf":
            with open(vectors_path, "wb") as f:
                pickle.dump(vectors, f)
    if vectorizer_type == "tfidf":
        data["vector"] = [vectors.getrow(i) for i in range(vectors.shape[0])]  # type: ignore
    else:
        data["vector"] = [vectors[i] for i in range(vectors.shape[0])]
    train_loader = get_feedforward_dataloader(data, train_idxs, batch_size, shuffle=True, dtype=torch_dtype)
    if 1.0 > config["train_ratio"] > 0.0:
        val_loader = get_feedforward_dataloader(data, val_idxs, batch_size, shuffle=False, dtype=torch_dtype)
    else:
        raise ValueError("Invalid train ratio: {config['train_ratio']}. Must be between 0.0 and 1.0.")
    return train_loader, val_loader, int(vectors.shape[1])


def set_labels(data: pd.DataFrame, first_split_idx: int):
    """
    Set labels for each article according to the confusion scheme.

    It is based on this paper:
    https://arxiv.org/abs/1610.02048

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data.

    Returns
    -------
    None
    """
    start_date = data["date"].iloc[0]
    end_date = data["date"].iloc[-1]
    dates = get_dates_for_interval(start_date, end_date)
    # We cannot have a split on the very edges.
    labels = np.empty((len(data), len(dates) - 2 * first_split_idx))
    for article_idx, row in data.iterrows():
        for tr_idx, transition_date in enumerate(dates[first_split_idx:-first_split_idx]):
            article_date = row["date"]
            if article_date < transition_date:
                labels[article_idx, tr_idx] = 0
            elif article_date > transition_date:
                labels[article_idx, tr_idx] = 1
            else:
                labels[article_idx, tr_idx] = 0.5
    data["label"] = [labels[article_idx, :] for article_idx in range(len(data))]


def calculate_indicators(
    dataloader,
    model: FFConfusion,
    device: torch.device,
    dates: list[date],
    first_split_idx: int,
    split_distance: int | None,
):
    """
    Calculate indicator values with a specific model.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader object that provides the input data and labels.
    model : FFConfusion
        The model used for calculating the indicators.
    device : torch.device
        The device where the model and data will be loaded.
    dates : list[date]
        The list of dates for which indicators are calculated.

    Returns
    -------
    pandas.DataFrame
        A dataframe object containing the calculated indicators for each date.

    """

    def get_output_label_ff(batch):
        inputs, labels, timestamps = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        return model(inputs), labels, timestamps

    model.eval()
    with torch.no_grad():
        split_left_accuracies: list[list] = [[] for _ in dates[first_split_idx:-first_split_idx]]
        split_right_accuracies: list[list] = [[] for _ in dates[first_split_idx:-first_split_idx]]
        for batch in dataloader:
            outputs, labels, timestamps = get_output_label_ff(batch)
            timestamp_list = timestamps.tolist()
            date_list = [datetime.fromtimestamp(ts).date() for ts in timestamp_list]
            predictions = torch.round(outputs)

            for split_day_idx, split_date in enumerate(dates[first_split_idx:-first_split_idx]):
                if split_distance is None:
                    split_labels = labels[:, split_day_idx]
                    split_predictions = predictions[:, split_day_idx]
                else:
                    mask = [
                        (
                            split_date - split_distance * timedelta(days=1)
                            <= dt
                            <= split_date + split_distance * timedelta(days=1)
                        )
                        for dt in date_list
                    ]
                    split_labels = labels[mask, split_day_idx]
                    split_predictions = predictions[mask, split_day_idx]
                split_neg_accuracies = calculate_batch_side_accuracy(split_labels, split_predictions, 0.0)
                split_pos_accuracies = calculate_batch_side_accuracy(split_labels, split_predictions, 1.0)
                split_left_accuracies[split_day_idx].append(split_neg_accuracies)
                split_right_accuracies[split_day_idx].append(split_pos_accuracies)
    accuracy_indicators = []
    for split_day_idx in range(len(dates[first_split_idx:-first_split_idx])):
        left_accuracy_arr = np.concatenate(split_left_accuracies[split_day_idx])
        right_accuracy_arr = np.concatenate(split_right_accuracies[split_day_idx])
        if left_accuracy_arr.shape[0] == 0 and right_accuracy_arr.shape[0] == 0:
            accuracy_indicator = np.nan
        elif left_accuracy_arr.shape[0] == 0:
            accuracy_indicator = np.mean(right_accuracy_arr)
        elif right_accuracy_arr.shape[0] == 0:
            accuracy_indicator = np.mean(left_accuracy_arr)
        else:
            mean_left_accuracy = np.mean(left_accuracy_arr)
            mean_right_accuracy = np.mean(right_accuracy_arr)
            accuracy_indicator = np.mean([mean_left_accuracy, mean_right_accuracy])
        accuracy_indicators.append(accuracy_indicator)

    indicators_df = pd.DataFrame.from_dict({"date": [], "indicator_value": []})
    indicators_df["date"] = dates[first_split_idx:-first_split_idx]
    indicators_df["indicator_value"] = accuracy_indicators
    return indicators_df


def calculate_batch_side_accuracy(split_labels: Tensor, split_predictions: Tensor, label: float) -> np.ndarray:
    """
    Calculate accuracy for one type of class label.

    Parameters
    ----------
    split_labels : Tensor
        Tensor containing the labels for each data point in the batch.

    split_predictions : Tensor
        Tensor containing the predicted labels for each data point in the batch.

    label : float
        The label for which the side accuracy needs to be calculated.

    Returns
    -------
    np.ndarray
        Numpy array containing the batch side accuracies, where each
        accuracy value corresponds to a data point in the batch.

    """
    side_label_filt = split_labels == label
    batch_neg_accuracies = ((split_predictions == split_labels)[side_label_filt]).cpu().numpy()
    return batch_neg_accuracies


def train_confusion(
    data: pd.DataFrame, train_out: str, config: dict, dataset_path: str, vectorizer_type: str
) -> None:
    """
    Train a model with confusion scheme.

    Parameters
    ----------
    data
        The input data containing the text data to be trained on.
    train_out
        The path to the directory where the trained model and other output files will be saved.
    config
        A dictionary containing the configuration parameters for training the model.
    vectorizer_type
        A dictionary containing the configuration parameters for the vectorizer used to transform the text data.

    Returns
    -------
    None

    """
    if "first_split_date" in config:
        first_split_date = config["first_split_date"]
        start_date = data["date"].iloc[0]
        end_date = data["date"].iloc[-1]
        dates = get_dates_for_interval(start_date, end_date)
        dates_mm_dd = [date.strftime("%m-%d") for date in dates]
        if first_split_date not in dates_mm_dd:
            raise ValueError(f"Invalid first split date: {first_split_date}")
        first_split_idx = dates_mm_dd.index(first_split_date)
    else:
        first_split_idx = config["first_split_idx"]

    split_distance = config["split_distance"]
    set_labels(data, first_split_idx)
    start_date = data["date"].iloc[0]
    end_date = data["date"].iloc[-1]
    dates = get_dates_for_interval(start_date, end_date)
    split_dates = dates[first_split_idx:-first_split_idx]
    num_splits = len(split_dates)
    train_idxs, val_idxs = get_splits(data, train_ratio=config["train_ratio"])
    batch_size = config["batch_size"]
    torch_dtype = getattr(torch, config["dtype"])
    torch.set_default_dtype(torch_dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, input_dim = get_feedforward_loaders(
        batch_size, data, torch_dtype, train_idxs, val_idxs, config, dataset_path, vectorizer_type
    )
    model = FFConfusion(input_dim, num_splits).type(torch_dtype).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    criterion = BCEWithWeights(data, torch_dtype, device)
    model.to(device)
    criterion.to(device)
    min_loss = np.inf
    per_epoch_losses: dict[str, list] = {"epoch": [], "train": [], "val": []}  # type: ignore
    for epoch in tqdm(range(config["max_epochs"]), desc="Epoch"):
        epoch_train_loss, epoch_val_loss = train_epoch(
            criterion, device, model, optimizer, train_loader, val_loader, split_dates, split_distance
        )
        per_epoch_losses["epoch"].append(epoch)
        per_epoch_losses["train"].append(epoch_train_loss)
        per_epoch_losses["val"].append(epoch_val_loss)
        epoch_loss = epoch_val_loss
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save(model, os.path.join(train_out, "best_model_checkpoint.pth"))
        else:
            if epoch >= config["min_epochs"]:
                break
    loss_df = pd.DataFrame.from_dict(per_epoch_losses)
    loss_df.to_csv(os.path.join(train_out, "per_epoch_losses.csv"), index=False)
    indicators_path = os.path.join(train_out, "indicator_values.csv")
    model = torch.load(os.path.join(train_out, "best_model_checkpoint.pth"), weights_only=False)
    indicator_df = calculate_indicators(
        val_loader, model, device, dates, first_split_idx, split_distance
    )
    indicator_df.to_csv(indicators_path, index=False)
    loss_df["train"] = loss_df["train"] / loss_df["train"].iloc[0]
    loss_df["val"] = loss_df["val"] / loss_df["val"].iloc[0]
    loss_df.plot(x="epoch", y=["train", "val"], kind="line")
    plt.title("Training and Validation Loss Over Time (Normalized)")
    plt.savefig(os.path.join(train_out, "losses.png"))


def train_epoch(
    criterion: BCEWithWeights | nn.BCELoss,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    split_dates: list[date],
    split_distance: int | None,
) -> tuple[float, float | None]:
    """
    Train a single epoch of a neural network model.

    Parameters
    ----------
    criterion : torch.nn.Module
        The loss function used to compute the training loss.
    device : torch.device
        The device on which the model and tensors should be placed.
    model : torch.nn.Module
        The neural network model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer used to update the model's parameters.
    train_loader : torch.utils.data.DataLoader
        The data loader for the training dataset.
    val_loader : torch.utils.data.DataLoader or None
        The data loader for the validation dataset. Pass None if not performing validation.

    Returns
    -------
    epoch_train_loss : float
        The average training loss for the epoch.
    epoch_val_loss : float or None
        The average validation loss for the epoch, or None if `val_loader` is None.
    """

    def calculate_loss_mask(timestamps):
        np_dates = np.array([np.datetime64(int(t), "s") for t in timestamps])
        split_np_dates = np.array([np.datetime64(d) for d in split_dates])
        batch_size = len(np_dates)
        num_splits = len(split_np_dates)
        sample_dates = np.tile(np_dates, (num_splits, 1)).T
        split_dates_2D = np.tile(split_np_dates, (batch_size, 1))
        distances = np.abs((sample_dates - split_dates_2D).astype("timedelta64[D]").astype(int))
        return torch.tensor((distances <= split_distance).astype(int), device=device)

    def apply_batch_ff(batch):
        inputs, labels, timestamps = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        if split_distance is None:
            return criterion(outputs, labels)
        else:
            loss_mask = calculate_loss_mask(timestamps)
            return criterion(outputs, labels, loss_mask=loss_mask)  # type: ignore

    epoch_train_loss = 0.0
    model.train()
    for batch in train_loader:
        batch_size = batch[1].shape[0]
        loss = apply_batch_ff(batch)
        epoch_train_loss += float(loss)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * batch_size
    epoch_train_loss /= len(train_loader.dataset)  # type: ignore
    model.eval()
    with torch.no_grad():
        if val_loader is None:
            epoch_val_loss = None
        else:
            epoch_val_loss = 0.0
            for batch in val_loader:
                batch_size = batch[1].shape[0]
                loss = apply_batch_ff(batch)
                epoch_val_loss += loss.item() * batch_size
            epoch_val_loss /= len(val_loader.dataset)  # type: ignore
    return epoch_train_loss, epoch_val_loss
