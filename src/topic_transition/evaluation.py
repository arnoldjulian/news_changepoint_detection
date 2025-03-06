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
    """Load dataset."""
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


def get_events_from_path(events_path: str) -> pd.DataFrame:
    """Get events with significance scores."""
    events = pd.read_csv(events_path)
    events["date"] = pd.to_datetime(events["date"]).dt.date
    return events
