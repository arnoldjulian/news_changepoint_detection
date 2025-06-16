"""Tools for training with different models."""
import logging
import os
import random
import re
from datetime import date, timedelta, datetime

import pandas as pd
import yaml
from dateutil.relativedelta import relativedelta

from topic_transition.training.confusion import train_confusion
from topic_transition.utils import get_dates_for_interval, get_training_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


MAX_CHUNK_SIZE = {
    "all-MiniLM-L6-v2": 256,
    "all-MiniLM-L12-v2": 256,
    "sentence-t5-base": 512,
    "all-mpnet-base-v2": 384,
    "all-distilroberta-v1": 128,
    "BAAI/llm-embedder": 512,
}


def month_day_format(date_string: str) -> bool:
    """Determine if this date matches MM-DD format."""
    if re.match(r"^\d{2}-\d{2}$", date_string):
        return True
    else:
        return False


def chunk_data(data: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    """Preprocess the dataframe by splitting the 'full_text' column into chunks if necessary."""
    if not chunk_size:
        return data

    new_rows = []

    for _, row in data.iterrows():
        words = row["full_text"].split()

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])  # noqa: E203

            new_rows.append(
                {
                    "date": row["date"],
                    "webTitle": row["webTitle"],
                    "full_text": chunk,
                }
            )

    return pd.DataFrame(new_rows)


def prepare_training_directory(
    config: dict, time_interval: str, section: str, max_iterations: int = 5
) -> tuple[str | None, str | None]:
    """Prepare directories and paths required for model training."""
    train_out = os.path.join(config["trainings_root"], time_interval)
    train_out = os.path.join(train_out, section)
    training_name = config["name"]
    train_out = os.path.join(train_out, training_name)
    overwrite = config["overwrite_trainings"]
    assert overwrite in ["overwrite", "skip", "new"]
    indicators_path = os.path.join(train_out, "indicator_values.csv")
    if os.path.exists(indicators_path):
        if overwrite == "skip":
            print(f"Skipping training {train_out} because it already exists")
            return None, None
        elif overwrite == "new":
            train_out = get_training_path(train_out, max_iterations=max_iterations)
            if train_out is None:
                return None, None
    os.makedirs(train_out, exist_ok=True)
    with open(os.path.join(train_out, "config.yml"), "w") as file:
        yaml.dump(config, file)
    indicators_path = os.path.join(train_out, "indicator_values.csv")
    return train_out, indicators_path


def generate_random_indicators_for_dataset(config: dict) -> None:
    """Train a single model."""
    data, section, time_interval = get_dataset(config)
    train_out, indicators_path = prepare_training_directory(config, time_interval, section)
    if not train_out:
        return None

    logger.info("Preparing datasets")
    start_date = data["date"].iloc[0]
    end_date = data["date"].iloc[-1]
    dates = get_dates_for_interval(start_date, end_date)
    first_split_idx = config["first_split_idx"]
    if first_split_idx > 0:
        dates = dates[first_split_idx:-first_split_idx]

    random_split_date = random.choice(dates)

    indicator_values = pd.DataFrame({"date": dates})
    indicator_values["indicator_value"] = 0
    indicator_values.loc[indicator_values["date"] == random_split_date, "indicator_value"] = 1
    indicator_values.to_csv(indicators_path, index=False)


def get_dataset(config: dict):
    """Fetch and process the dataset based on the provided configuration."""
    dataset_path = config["dataset"]["path"]
    if dataset_path.endswith(".csv"):
        data = pd.read_csv(dataset_path)
    else:
        data = pd.read_pickle(dataset_path)
    data["date"] = pd.to_datetime(data["webPublicationDate"]).dt.date
    dataset_config = config["dataset"]
    path_parts = dataset_config["path"].split(os.path.sep)
    section = path_parts[-1][:-4]
    if "selected_month" in config and config["selected_month"] is not None:
        split_distance = config["split_distance"]
        start_date = date(config["selected_year"], config["selected_month"], 1)
        end_date = start_date + relativedelta(months=1)
        start_date = start_date - timedelta(days=split_distance)
        end_date = end_date + timedelta(days=split_distance)
        data = data[(data["date"] >= start_date) & (data["date"] < end_date)]
        assert len(data) > 0
        month = str(config["selected_month"]).zfill(2)
        time_interval = f"{config['selected_year']}-{month}"
    else:
        time_interval = path_parts[-2]
    return data, section, time_interval


def generate_constant_indicators_for_dataset(config: dict, prediction_day: int) -> None:
    """Train a single model with constant split indicator."""
    data, section, time_interval = get_dataset(config)
    train_out, indicators_path = prepare_training_directory(config, time_interval, section, 365)
    if not train_out:
        return None

    logger.info("Preparing datasets")
    start_date = data["date"].iloc[0]
    end_date = data["date"].iloc[-1]
    dates = get_dates_for_interval(start_date, end_date)

    if "first_split_date" in config and "last_split_date" in config:
        year = config["dataset"]["path"].split(os.sep)[-2]
        first_split_date = datetime.strptime(f'{year}-{config["first_split_date"]}', "%Y-%m-%d").date()
        last_split_date = datetime.strptime(f'{year}-{config["last_split_date"]}', "%Y-%m-%d").date()
        first_split_idx = dates.index(first_split_date)
        last_split_idx = dates.index(last_split_date) + 1
        dates = dates[first_split_idx:last_split_idx]
    elif "first_split_idx" in config:
        first_split_idx = config["first_split_idx"]
        if first_split_idx > 0:
            dates = dates[first_split_idx:-first_split_idx]
    else:
        raise ValueError(
            "Either 'first_split_date' and 'last_split_date' or 'first_split_idx' must be provided in the config."
        )

    if len(dates) <= prediction_day:
        return None

    indicator_values = pd.DataFrame({"date": dates})
    indicator_values["indicator_value"] = 0
    indicator_values.loc[indicator_values["date"] == dates[prediction_day], "indicator_value"] = 1

    indicator_values.to_csv(indicators_path, index=False)


def train_on_single_dataset(config: dict) -> int | None:
    """Train a single model."""
    data, section, time_interval = get_dataset(config)
    train_out, indicators_path = prepare_training_directory(config, time_interval, section)
    if not train_out:
        return None

    logger.info("Preparing datasets")
    data.reset_index(drop=True, inplace=True)
    data["webTitle"] = data["webTitle"].fillna("")
    data["text"] = data["text"].fillna("")
    data = data[~((data["webTitle"] == "") & (data["text"] == ""))]
    data["full_text"] = data["webTitle"] + "\n" + data["text"]
    data = chunk_data(data, MAX_CHUNK_SIZE[config["vectorizer"]])
    return train_confusion(data, train_out, config, config["vectorizer"])
