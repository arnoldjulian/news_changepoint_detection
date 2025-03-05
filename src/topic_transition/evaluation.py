"""Tools for evaluation."""
import os

import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


def get_events_with_scores(training_path: str, dataset_base_path: str, events_base_path: str) -> pd.DataFrame:
    """
    Load events and calculate their significance scores.

    Parameters
    ----------
    training_path
        The file path to the training data, used to determine `year` and `section_id`.
    dataset_base_path
        The base directory path where the dataset files are stored.
    events_base_path
        The base directory path where the events data is stored.
    """
    time_interval, section_id = training_path.split(os.path.sep)[-3:-1]
    dataset_path = os.path.join(dataset_base_path, time_interval)
    dataset_path = os.path.join(dataset_path, f"{section_id}.pkl")
    dataset = pd.read_pickle(dataset_path)
    dataset["date"] = pd.to_datetime(dataset["webPublicationDate"]).dt.date
    return get_events_with_significance_scores(time_interval, section_id, events_base_path)


def get_events_with_significance_scores(
    time_interval: str,
    section_id: str,
    events_base_path: str,
) -> pd.DataFrame:
    """
    Calculate significance scores.

    Parameters
    ----------
    dataset : pd.DataFrame
        The primary dataset for which significance scores are to be calculated.
    time_interval
        The specific year for the events to be considered.
    section_id
        Identifier for a particular section to fetch corresponding events.
    events_base_path
        Base directory path where event files are stored.
    """
    events_dir = os.path.join(events_base_path, time_interval)
    events_path = os.path.join(events_dir, f"{section_id}.csv")
    if os.path.exists(events_path):
        events = pd.read_csv(events_path)
    else:
        events = pd.read_csv(os.path.join(events_dir, "world.csv"))
    events["date"] = pd.to_datetime(events["date"]).dt.date
    return events


def get_events_from_path(events_path: str):
    """Get events with significance scores."""
    events = pd.read_csv(events_path)
    events["date"] = pd.to_datetime(events["date"]).dt.date
    return events


def calculate_avg_indicators_for_dataset(indicators: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average indicators.

    Parameters
    ----------
    training_path : str
        The file path to the directory containing the indicators data file.

    events : pd.DataFrame
        A DataFrame containing the event data, which includes a date column.
    """
    # Ensure dates are comparable
    events["date"] = pd.to_datetime(events["date"]).dt.date

    # Merge indicators with events
    merged_df = pd.merge(indicators, events, on="date", how="inner")

    # Normalize the indicator values to sum to 1.0
    total_indicator_value = merged_df["indicator_value"].sum()
    if total_indicator_value != 0:
        merged_df["normalized_indicator_value"] = merged_df["indicator_value"] / total_indicator_value
    else:
        merged_df["normalized_indicator_value"] = 0

    # Calculate averages
    avg_indicator_value = merged_df["normalized_indicator_value"].mean()

    # Calculate correlation
    if len(merged_df["indicator_value"]) < 2:
        correlation = np.nan
    else:
        correlation = merged_df["indicator_value"].corr(merged_df["yearly_avg_tfidf_score"])

    # Create a DataFrame with the results
    result_df = pd.DataFrame(
        {
            "indicators_average": [avg_indicator_value],
            "indicator_significance_correlation": [correlation],
        }
    )

    return result_df
