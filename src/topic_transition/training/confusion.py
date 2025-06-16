"""Tools for training confusion model."""
import os
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
from topic_transition.model import FFConfusion
from topic_transition.utils import get_dates_for_interval
from topic_transition.vectorizers import get_vectorizer


def get_single_dataloader(
    data: pd.DataFrame, idxs: list[int], batch_size: int, shuffle: bool, dtype: torch.dtype
) -> DataLoader:
    """Create dataloader."""
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


def get_dataloaders(
    batch_size: int,
    data: pd.DataFrame,
    torch_dtype: torch.dtype,
    train_idxs: list[int],
    val_idxs: list[int],
    config: dict,
    vectorizer_type: str,
) -> tuple[DataLoader, DataLoader | None, int]:
    """Get dataloaders for non-llm model."""
    vectorizer = get_vectorizer(vectorizer_type)
    vectors = vectorizer.fit_transform(data["full_text"])
    data["vector"] = [vectors[i] for i in range(vectors.shape[0])]
    train_loader = get_single_dataloader(data, train_idxs, batch_size, shuffle=True, dtype=torch_dtype)
    if 1.0 > config["train_ratio"] > 0.0:
        val_loader = get_single_dataloader(data, val_idxs, batch_size, shuffle=False, dtype=torch_dtype)
    else:
        val_loader = train_loader
    return train_loader, val_loader, int(vectors.shape[1])


def set_labels(data: pd.DataFrame, split_dates: list[date] | None = None) -> None:
    """Set labels for each article according to the confusion scheme."""
    labels = np.empty((len(data), len(split_dates)))
    for article_idx, row in data.iterrows():
        for tr_idx, transition_date in enumerate(split_dates):
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
    """Calculate indicator values with a specific model."""

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
    """Calculate accuracy for one type of class label."""
    side_label_filt = split_labels == label
    batch_neg_accuracies = ((split_predictions == split_labels)[side_label_filt]).cpu().numpy()
    return batch_neg_accuracies


def train_confusion(data: pd.DataFrame, train_out: str, config: dict, vectorizer_type: str) -> None:
    """Train a model with confusion scheme."""
    split_distance = config["split_distance"]
    if "first_split_date" in config:
        first_split_date = config["first_split_date"]
        start_date = data["date"].iloc[0]
        end_date = data["date"].iloc[-1]
        dates = get_dates_for_interval(start_date, end_date)
        dates_mm_dd = [date.strftime("%m-%d") for date in dates]
        if first_split_date not in dates_mm_dd:
            raise ValueError(f"Invalid first split date: {first_split_date}")
        first_split_idx = dates_mm_dd.index(first_split_date)
    elif "first_split_idx" in config:
        first_split_idx = config["first_split_idx"]
    else:
        first_split_idx = split_distance

    start_date = data["date"].iloc[0]
    end_date = data["date"].iloc[-1]
    dates = get_dates_for_interval(start_date, end_date)
    if first_split_idx > 0:
        split_dates = dates[first_split_idx:-first_split_idx]
    else:
        split_dates = dates
    set_labels(data, split_dates)
    num_splits = len(split_dates)
    train_idxs, val_idxs = get_splits(data, train_ratio=config["train_ratio"])
    batch_size = config["batch_size"]
    torch_dtype = getattr(torch, config["dtype"])
    torch.set_default_dtype(torch_dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, input_dim = get_dataloaders(
        batch_size, data, torch_dtype, train_idxs, val_idxs, config, vectorizer_type
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
    model_path = os.path.join(train_out, "best_model_checkpoint.pth")
    model = torch.load(os.path.join(train_out, "best_model_checkpoint.pth"), weights_only=False)
    indicator_df = calculate_indicators(val_loader, model, device, dates, first_split_idx, split_distance)
    indicator_df.to_csv(indicators_path, index=False)
    loss_df["train"] = loss_df["train"] / loss_df["train"].iloc[0]
    loss_df["val"] = loss_df["val"] / loss_df["val"].iloc[0]
    loss_df.plot(x="epoch", y=["train", "val"], kind="line")
    plt.title("Training and Validation Loss Over Time (Normalized)")
    plt.savefig(os.path.join(train_out, "losses.png"))
    os.remove(model_path)


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
    """Train a single epoch of a neural network model."""

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
