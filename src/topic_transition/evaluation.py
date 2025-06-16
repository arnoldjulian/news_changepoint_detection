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
    else:
        raise ValueError("No dataset path provided!")
    if dataset_path.endswith(".csv"):
        dataset = pd.read_csv(dataset_path)  # type: ignore
    else:
        dataset = pd.read_pickle(dataset_path)  # type: ignore

    dataset["date"] = pd.to_datetime(dataset["webPublicationDate"]).dt.date
    return dataset


def get_events_from_path(events_path: str) -> pd.DataFrame:
    """Get events with significance scores."""
    events = pd.read_csv(events_path)
    events["date"] = pd.to_datetime(events["date"]).dt.date
    return events


def calculate_deltas_for_dataset(
    training_path: str,
    events: pd.DataFrame,
    evaluation_path: str,
    model_name: str,
) -> pd.DataFrame:
    """Calculate top-k deltas for a selected evaluation."""
    deltas: dict[str, list] = {
        "model_name": [],
        "delta": [],
        "date": [],
        "description": [],
        "evaluation_path": [],
        "time_interval": [],
        "section_id": [],
    }

    year_or_month, section_id = training_path.split(os.path.sep)[-3:-1]

    indicators_path = os.path.join(training_path, "indicator_values.csv")

    if not os.path.exists(indicators_path):
        return pd.DataFrame.from_dict(deltas)

    indicators = pd.read_csv(indicators_path)
    indicators["date"] = pd.to_datetime(indicators["date"]).dt.date
    indicator_max_date = indicators.loc[indicators["indicator_value"].argmax()]["date"]  # type: ignore

    def get_closest(indicator_max_date, events):
        closest_event = None
        min_delta = None
        for _, event in events.iterrows():
            delta = abs((event["date"] - indicator_max_date).days)
            if (closest_event is None) or (delta < min_delta):
                closest_event = event
                min_delta = delta
        return closest_event, min_delta

    indicators.to_csv(os.path.join(evaluation_path, f"{year_or_month}_{section_id}_indicators.csv"), index=False)
    closest_event, min_delta = get_closest(indicator_max_date, events)
    deltas["model_name"].append(model_name)
    deltas["delta"].append(min_delta)
    deltas["date"].append(closest_event["date"])
    deltas["description"].append(closest_event["description"])
    deltas["evaluation_path"].append(evaluation_path)
    deltas["time_interval"].append(year_or_month)
    deltas["section_id"].append(section_id)
    return pd.DataFrame.from_dict(deltas)
