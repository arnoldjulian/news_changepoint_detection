"""Tools for training with different models."""
import logging
import os
import random
import re
from datetime import date, timedelta

import pandas as pd
import yaml
from dateutil.relativedelta import relativedelta

from topic_transition.training.confusion import train_confusion
from topic_transition.utils import get_dates_for_interval, increment_path_number

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
    """
    Preprocess the dataframe by splitting the 'full_text' column into chunks if necessary.

    Args:
        data (pd.DataFrame): Input dataframe with 'full_text' and 'date' columns.
        vectorizer_config (dict): Configuration dict containing chunk_size.

    Returns:
        pd.DataFrame: Processed dataframe with 'full_text' split into chunks if applicable.
    """
    # If chunk_size is not set, return the original dataframe
    if not chunk_size:
        return data

    # List to collect new rows
    new_rows = []

    # Iterate over each row in the dataframe
    for _, row in data.iterrows():
        # Split the full_text into words
        words = row["full_text"].split()

        # Calculate the number of chunks
        for i in range(0, len(words), chunk_size):
            # Create text chunk
            chunk = " ".join(words[i : i + chunk_size])  # noqa: E203

            # Create a new row and append to the new_rows list
            new_rows.append(
                {
                    "date": row["date"],  # Keep the date
                    "webTitle": row["webTitle"],
                    "full_text": chunk,  # The current chunk
                }
            )

    # Create a new dataframe from the collected rows
    new_data = pd.DataFrame(new_rows)

    return new_data


def single_random(config: dict) -> None:
    """Train a single model."""
    general_config = config["general"]
    training_config = config["training"]
    dataset_config = config["dataset"]
    data = pd.read_pickle(dataset_config["path"])
    data["date"] = pd.to_datetime(data["webPublicationDate"]).dt.date
    path_parts = dataset_config["path"].split(os.path.sep)
    if "selected_month" in config and config["selected_month"] is not None:
        split_distance = training_config["confusion"]["split_distance"]
        start_date = date(config["selected_year"], config["selected_month"], 1)
        end_date = start_date + relativedelta(months=1)
        start_date = start_date - timedelta(days=split_distance)
        end_date = end_date + timedelta(days=split_distance)
        data = data[(data["date"] >= start_date) & (data["date"] < end_date)]
        time_interval = f"{start_date}-{end_date}"
    else:
        time_interval = path_parts[-2]
    section = path_parts[-1][:-4]

    print(general_config["trainings_root"])
    train_out = os.path.join(general_config["trainings_root"], time_interval)
    train_out = os.path.join(train_out, section)
    training_name = training_config["name"]
    train_out = os.path.join(train_out, training_name)

    overwrite = config["overwrite_trainings"]
    assert overwrite in ["overwrite", "skip", "new"]
    indicators_path = os.path.join(train_out, "indicator_values.csv")
    if os.path.exists(indicators_path):
        if overwrite == "skip":
            return None
        elif overwrite == "new":
            train_out = increment_path_number(train_out)
    os.makedirs(train_out, exist_ok=True)
    indicators_path = os.path.join(train_out, "indicator_values.csv")

    with open(os.path.join(train_out, "config.yml"), "w") as file:
        yaml.dump(config, file)
    logger.info("Preparing datasets")
    if "first_split_date" in training_config:
        first_split_date = training_config["first_split_date"]
        start_date = data["date"].iloc[0]
        end_date = data["date"].iloc[-1]
        dates = get_dates_for_interval(start_date, end_date)
        dates_mm_dd = [date.strftime("%m-%d") for date in dates]
        if first_split_date not in dates_mm_dd:
            raise ValueError(f"Invalid first split date: {first_split_date}")
        if "first_split_idx" not in training_config:
            first_split_idx = dates_mm_dd.index(first_split_date)
    elif "first_split_idx" in training_config:
        first_split_idx = training_config["first_split_idx"]
        if first_split_idx is None:
            first_split_idx = training_config["confusion"]["split_distance"]
    else:
        first_split_idx = training_config["confusion"]["split_distance"]

    start_date = data["date"].iloc[0]
    end_date = data["date"].iloc[-1]
    dates = get_dates_for_interval(start_date, end_date)
    split_dates = dates[first_split_idx:-first_split_idx]

    # Randomly select a split date
    if not split_dates:
        raise ValueError("No valid split dates available.")
    random_split_date = random.choice(split_dates)

    # Generate the indicator values DataFrame
    indicator_values = pd.DataFrame({"date": split_dates})
    indicator_values["indicator_value"] = 0
    indicator_values.loc[indicator_values["date"] == random_split_date, "indicator_value"] = 1

    # Save the indicator values to CSV
    indicator_values.to_csv(indicators_path, index=False)


def single_train(config: dict) -> int | None:
    """Train a single model."""
    dataset_config = config["dataset"]
    data = pd.read_pickle(dataset_config["path"])
    data["date"] = pd.to_datetime(data["webPublicationDate"]).dt.date
    path_parts = dataset_config["path"].split(os.path.sep)
    if "selected_month" in config and config["selected_month"] is not None:
        split_distance = config["split_distance"]
        start_date = date(config["selected_year"], config["selected_month"], 1)
        end_date = start_date + relativedelta(months=1)
        start_date = start_date - timedelta(days=split_distance)
        end_date = end_date + timedelta(days=split_distance)
        data = data[(data["date"] >= start_date) & (data["date"] < end_date)]
        assert len(data) > 0
        time_interval = f"{start_date}-{end_date}"
    else:
        time_interval = path_parts[-2]
    section = path_parts[-1][:-4]
    train_out = os.path.join(config["trainings_root"], time_interval)
    train_out = os.path.join(train_out, section)
    training_name = config["name"]
    train_out = os.path.join(train_out, training_name)
    indicators_path = os.path.join(train_out, "indicator_values.csv")
    overwrite = config["overwrite_trainings"]
    assert overwrite in ["overwrite", "skip", "new"]
    if os.path.exists(indicators_path):
        if overwrite == "skip":
            return None
        elif overwrite == "new":
            train_out = increment_path_number(train_out)
            if train_out is None:
                return None
    os.makedirs(train_out, exist_ok=True)
    with open(os.path.join(train_out, "config.yml"), "w") as file:
        yaml.dump(config, file)
    logger.info("Preparing datasets")
    data.reset_index(drop=True, inplace=True)
    data["full_text"] = data["webTitle"] + "\n" + data["text"]
    if config["vectorizer"] != "tfidf":
        data = chunk_data(data, MAX_CHUNK_SIZE[config["vectorizer"]])
    return train_confusion(data, train_out, config, dataset_config["path"], config["vectorizer"])
