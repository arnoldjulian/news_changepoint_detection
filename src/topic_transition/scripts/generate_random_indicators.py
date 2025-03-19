"""Script for training a model on all available datasets."""
import argparse
import logging

import yaml

from topic_transition.training import generate_random_indicators_for_dataset
from topic_transition.utils import set_random_seed

logger = logging.getLogger("train_with_single_dataset")
logger.setLevel(logging.INFO)


def process_training(dataset_path, selected_month, base_config=None):
    """Do a training on a single dataset or a month in the data."""
    config = base_config.copy()
    config["dataset"] = {"path": dataset_path}
    if selected_month is not None:
        selected_year, selected_month = selected_month.split("-")
        config["selected_year"] = int(selected_year)
        config["selected_month"] = int(selected_month)
    generate_random_indicators_for_dataset(config)


def main(base_config) -> None:
    """Do all trainings for all months or years."""
    dataset_paths = base_config["datasets"]
    set_random_seed(base_config["deterministic"])
    if "selected_months" in base_config:
        selected_months = base_config["selected_months"]
    else:
        selected_months = [None for _ in dataset_paths]

    for dataset_path, selected_month in zip(dataset_paths, selected_months):
        process_training(dataset_path, selected_month, base_config=base_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and validate .pkl files containing dataframes")
    parser.add_argument("configuration", help="Configuration for the model and training parameters.")
    args = parser.parse_args()

    with open(args.configuration, "r") as file:
        main(yaml.safe_load(file))
