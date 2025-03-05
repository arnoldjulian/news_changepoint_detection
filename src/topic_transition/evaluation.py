"""Tools for evaluation."""
import os

import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def load_dataset(
    training_path: str | None = None, dataset_base_path: str | None = None, dataset_path: str | None = None
) -> dict[str, list]:
    """
    Load dataset.

    Parameters
    ----------
    training_path
        The file path where the training data is stored. If specified, this and `dataset_base_path`
        are used to construct the `dataset_path` dynamically.
    dataset_base_path
        The base directory path where datasets are stored. Used in conjunction with `training_path`
        to build the full `dataset_path`.
    dataset_path
        The full file path to the dataset. If specified, it is directly used to load the dataset.

    Returns
    -------
    dict[str, list]
        A dictionary representation of the dataset, with keys as column names and values as lists
        of column data. The dataset includes a new 'date' column derived from the 'webPublicationDate'
        column.
    """
    if training_path is not None and dataset_base_path is not None:
        year, section_id = training_path.split(os.path.sep)[-3:-1]
        dataset_path = os.path.join(dataset_base_path, year)
        dataset_path = os.path.join(dataset_path, f"{section_id}.pkl")
    elif dataset_path is not None:
        dataset_path = pd.read_pickle(dataset_path)
    else:
        raise ValueError("No dataset path provided!")
    dataset = pd.read_pickle(dataset_path)  # type: ignore
    dataset["date"] = pd.to_datetime(dataset["webPublicationDate"]).dt.date
    return dataset


def get_events_from_path(events_path: str):
    """Get events with significance scores."""
    events = pd.read_csv(events_path)
    events["date"] = pd.to_datetime(events["date"]).dt.date
    return events
