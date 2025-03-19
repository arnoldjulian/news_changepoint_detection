"""Tools for evaluating Total Variational Distance with Latent Dirichlet Allocation."""
import argparse
import copy
import os
import re

import nltk
import yaml
from nltk.corpus import stopwords

from topic_transition.tvd import get_tvd_metrics, load_all_events

nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def sanity_check(base_config):
    # Extract lists from config
    datasets = base_config["selected_datasets"]
    events = base_config["selected_events"]
    lda_paths = base_config["lda_paths"]
    months = base_config["selected_months"]

    # Check that all lists are of the same length
    assert len(datasets) == len(events), (
        f"Mismatch in list sizes: datasets({len(datasets)}) and events({len(events)})"
    )
    assert len(datasets) == len(lda_paths), (
        f"Mismatch in list sizes: datasets({len(datasets)}) and lda_paths({len(lda_paths)})"
    )
    assert len(datasets) == len(months), (
        f"Mismatch in list sizes: datasets({len(datasets)}) and months({len(months)})"
    )

    # Iterate over all the items and check year, month, and category consistency
    for idx, (dataset, event, lda, month) in enumerate(zip(datasets, events, lda_paths, months)):
        # Extract fields
        dataset_match = re.match(r".*/(\d{4})/([\w\-]+).csv", dataset)
        event_match = re.match(r".*/(\d{4})/([\w\-]+).csv", event)
        lda_match = re.match(r".*/(\d{4})-(\d{2})/([\w\-]+)", lda)
        month_match = re.match(r"(\d{4})-(\d{2})", month)

        # Assert valid formatting
        assert dataset_match, f"Invalid dataset format at index {idx}: {dataset}"
        assert event_match, f"Invalid event format at index {idx}: {event}"
        assert lda_match, f"Invalid lda_path format at index {idx}: {lda}"
        assert month_match, f"Invalid month format at index {idx}: {month}"

        # Extract details
        dataset_year, dataset_category = dataset_match.groups()
        event_year, event_category = event_match.groups()
        lda_year, lda_month, lda_category = lda_match.groups()
        month_year, month_month = month_match.groups()

        # Assert year consistency
        assert dataset_year == event_year, (
            f"Year mismatch at index {idx}: dataset({dataset_year}), event({event_year})"
        )
        assert dataset_year == lda_year, (
            f"Year mismatch at index {idx}: dataset({dataset_year}), lda({lda_year})"
        )
        assert dataset_year == month_year, (
            f"Year mismatch at index {idx}: dataset({dataset_year}), month({month_year})"
        )

        # Assert month consistency
        assert lda_month == month_month, (
            f"Month mismatch at index {idx}: lda({lda_month}), month({month_month})"
        )

        # Assert category consistency
        assert dataset_category == event_category, (
            f"Category mismatch at index {idx}: dataset({dataset_category}), event({event_category})"
        )
        assert dataset_category == lda_category, (
            f"Category mismatch at index {idx}: dataset({dataset_category}), lda({lda_category})"
        )


def main(base_config: dict) -> None:
    """Evaluate TVD on selected datasets."""
    sanity_check(base_config)
    training_name = base_config["model_name"]
    repeat_evaluations = base_config["repeat_evaluations"]
    if repeat_evaluations:
        training_names = [training_name]
        for i in range(1, repeat_evaluations):
            training_names.append(f"{training_name}_num_{i}")
    else:
        training_names = [training_name]

    evaluations_base_path = base_config["evaluation_base_path"]
    lda_config = base_config["lda"]
    dataset_paths = base_config["selected_datasets"]
    years_sections = [dataset_path.split(os.path.sep)[-2:] for dataset_path in dataset_paths]
    section_ids = [value[1][:-4] for value in years_sections]
    if "selected_months" in base_config:
        selected_months = base_config["selected_months"]
        selected_data_intervals = []
        for selected_month in selected_months:
            selected_year, selected_month = selected_month.split("-")
            time_interval = f"{selected_year}-{selected_month}"
            selected_data_intervals.append(time_interval)
    else:
        selected_data_intervals = [value[0] for value in years_sections]

    all_events = load_all_events(base_config)

    for i, training_name in enumerate(training_names):
        config = copy.deepcopy(base_config)
        evaluation_path = os.path.join(evaluations_base_path, training_name)

        os.makedirs(evaluation_path, exist_ok=True)

        if i > 0:
            lda_paths = config["lda_paths"]
            for j, lda_path in enumerate(lda_paths):
                lda_paths[j] = f"{lda_path}_num_{i}"

        deltas_df, tvds = get_tvd_metrics(all_events, config, dataset_paths, lda_config)

        for tvd, selected_data_interval, section_id in zip(tvds, selected_data_intervals, section_ids):
            indicator_path = os.path.join(evaluation_path, f"{selected_data_interval}_{section_id}_indicators.csv")
            tvd.to_csv(indicator_path, index=False)

        deltas_df.to_csv(os.path.join(evaluation_path, "deltas.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configuration", help="Configuration for evaluation.")

    args = parser.parse_args()

    with open(args.configuration, "r") as file:
        main(yaml.safe_load(file))
