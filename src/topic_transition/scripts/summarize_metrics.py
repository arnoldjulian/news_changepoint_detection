"""Script for aggregating the metrics of different evaluations."""
import argparse

import yaml

from topic_transition.summary_metrics import summarize_metrics


def main(config: dict) -> None:
    """
    Run script.

    Parameters
    ----------
    config
        A dictionary containing configuration settings, including:
            - "evaluations": A dictionary where keys are model names and values
                are lists of paths to directories containing evaluation data.
            - "summary_path": Path to the directory where the summary CSV file will be saved.

    """
    summarize_metrics(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Summarize metrics from different evaluations")
    parser.add_argument("configuration", help="Configuration file path for summary.")

    args = parser.parse_args()

    with open(args.configuration, "r") as file:
        config = yaml.safe_load(file)

    main(config)
