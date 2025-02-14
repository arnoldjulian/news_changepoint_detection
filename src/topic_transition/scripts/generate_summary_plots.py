"""Script for generating indicator value plots over multiple trainings."""
import argparse

import yaml

from topic_transition.summary_plots import generate_summary_plots


def main(config: dict) -> None:
    """Generate summary plots."""
    generate_summary_plots(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configuration", help="Configuration file path for summary.")

    args = parser.parse_args()

    with open(args.configuration, "r") as file:
        config = yaml.safe_load(file)

    main(config)
