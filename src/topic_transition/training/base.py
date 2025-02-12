"""Tools for training with different models."""
import logging
import os
import re

import pandas as pd
import yaml

from topic_transition.training.confusion import train_confusion
from topic_transition.utils import increment_path_number

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def month_day_format(date_string: str) -> bool:
    """Determine if this date matches MM-DD format."""
    if re.match(r"^\d{2}-\d{2}$", date_string):
        return True
    else:
        return False


def single_train(config: dict) -> None:
    """Train a single model."""
    general_config = config["general"]
    training_config = config["training"]
    vectorizer_config = training_config["vectorizer"]
    dataset_config = config["dataset"]
    data = pd.read_pickle(dataset_config["path"])
    path_parts = dataset_config["path"].split(os.path.sep)
    data["date"] = data["webPublicationDate"].apply(lambda x: pd.to_datetime(x).to_pydatetime().date())
    time_interval = path_parts[-2]
    section = path_parts[-1][:-4]
    train_out = os.path.join(general_config["trainings_root"], time_interval)
    train_out = os.path.join(train_out, section)
    training_name = training_config["name"]
    train_out = os.path.join(train_out, training_name)
    indicators_path = os.path.join(train_out, "indicator_values.csv")
    overwrite = config["overwrite_trainings"]
    assert overwrite in ["overwrite", "skip", "new"]
    if os.path.exists(indicators_path):
        if overwrite == "skip":
            return
        elif overwrite == "new":
            train_out = increment_path_number(train_out)
    os.makedirs(train_out, exist_ok=True)
    with open(os.path.join(train_out, "config.yml"), "w") as file:
        yaml.dump(config, file)
    logger.info("Preparing datasets")
    data.reset_index(drop=True, inplace=True)
    data["full_text"] = data["webTitle"] + "\n" + data["text"]
    train_confusion(data, train_out, training_config, dataset_config, vectorizer_config)
