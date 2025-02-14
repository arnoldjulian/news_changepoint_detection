"""Script for training a model on all available datasets."""
import argparse
import logging

import yaml

from topic_transition.training import single_train
from topic_transition.utils import set_random_seed

logger = logging.getLogger("train_with_single_dataset")
logger.setLevel(logging.INFO)


def main(config) -> None:
    """Do all trainings for all months or years."""
    datasets = config["datasets"]
    set_random_seed(config["deterministic"])
    for dataset_path in datasets:
        config["dataset"] = {"path": dataset_path}
        single_train(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and validate .pkl files containing dataframes")
    parser.add_argument("configuration", help="Configuration for the model and training parameters.")
    args = parser.parse_args()

    with open(args.configuration, "r") as file:
        config = yaml.safe_load(file)

    main(config)
