"""Train a model on all available datasets."""
import argparse
import logging

import torch
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
    parser.add_argument("--log_file_path", help="File to log to. If no path is given, log to console.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        total_memory_gb = total_memory / (1024**3)
        print(f"Total GPU Memory: {total_memory_gb:.2f} GB")
    else:
        print("No GPU available")

    if args.log_file_path:
        file_handler = logging.FileHandler(args.log_file_path)
        logger.addHandler(file_handler)

    with open(args.configuration, "r") as file:
        config = yaml.safe_load(file)

    main(config)
