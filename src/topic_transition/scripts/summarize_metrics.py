"""Script for aggregating the metrics of different evaluations."""
import argparse

import yaml

from topic_transition.summary_metrics import summarize_metrics


def main(config: dict) -> None:
    """Aggregate delta metrics from different evaluations."""
    summarize_metrics(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize metrics from different evaluations")
    parser.add_argument("configuration", help="Configuration file path for summary.")

    args = parser.parse_args()

    with open(args.configuration, "r") as file:
        config = yaml.safe_load(file)

    main(config)
