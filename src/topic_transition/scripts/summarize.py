"""Tools for summarizing the results of different evaluations."""
import argparse

import yaml

from topic_transition.summary_metrics import summarize_metrics
from topic_transition.summary_plots import generate_summary_plots

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configuration", help="Configuration file path for summary.")

    args = parser.parse_args()

    with open(args.configuration, "r") as file:
        config = yaml.safe_load(file)

    summarize_metrics(config)
    #generate_summary_plots(config)
