"""Tools for evaluating a specific trained model."""
import argparse
import os
import re
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from topic_transition.evaluation import calculate_deltas_for_dataset


def find_files_with_prefixes(
    selected_trainings: list[str], all_events: list[pd.DataFrame], event_paths: list[str], evaluation_base_path: str
) -> tuple[list, list[pd.DataFrame], list[str]]:
    """Find files with given prefixes and adjust the `all_events` list."""
    adjusted_trainings = []
    adjusted_events = []
    adjusted_event_paths = []
    evaluation_paths = []
    for selected_training, event, event_path in zip(selected_trainings, all_events, event_paths, strict=True):
        time_interval = selected_training.split(os.path.sep)[-3]
        if re.match(r"\d{4}-\d{2}", time_interval):
            month = int(time_interval.split("-")[1])
            event["month"] = event["date"].apply(lambda dt: dt.month)
            event = event[event["month"] == month]
            assert len(event) > 0, f"No events for month {month} in {event_path}"
        abs_training = os.path.abspath(selected_training)
        adjusted_trainings.append(abs_training)
        adjusted_events.append(event)
        adjusted_event_paths.append(event_path)
        base_path = Path(abs_training).parent
        evaluation_path = os.path.join(evaluation_base_path, Path(selected_training).name)
        os.makedirs(evaluation_path, exist_ok=True)
        evaluation_paths.append(evaluation_path)
        for f in os.listdir(base_path):
            path = os.path.join(base_path, f)
            if path.startswith(f"{abs_training}_num"):
                adjusted_trainings.append(path)
                adjusted_events.append(event)
                adjusted_event_paths.append(event_path)
                evaluation_path = os.path.join(evaluation_base_path, f)
                os.makedirs(evaluation_path, exist_ok=True)
                evaluation_paths.append(evaluation_path)

    return adjusted_trainings, adjusted_events, evaluation_paths


def sanity_check(config):
    """Check the consistency of configuration data used for processing."""
    if "selected_trainings" and "selected_events" in config:
        trainings = config["selected_trainings"]
        events = config["selected_events"]

        # Check list size
        assert len(trainings) == len(
            events
        ), f"List size mismatch: trainings({len(trainings)}) and events({len(events)})"


def main(config: dict) -> None:
    """Evaluate all trainings of a selected model type, including multiple iterations of the same training."""
    sanity_check(config)
    model_name = config["model_name"]
    evaluations_base_path = config["evaluation_base_path"]
    if "selected_trainings" and "selected_events" in config:
        selected_trainings = config["selected_trainings"]
        selected_events = config["selected_events"]
    elif "selected_trainings_events" in config:
        selected_trainings_events = config["selected_trainings_events"]
        selected_trainings = [pair[0] for pair in selected_trainings_events]
        selected_events = [pair[1] for pair in selected_trainings_events]
    else:
        raise ValueError("No selected trainings or events found in config.")
    dataset_base_path = config["dataset_base_path"]

    all_events = []
    for events_path in selected_events:
        year_or_month, section_id = events_path.split(os.path.sep)[-2:]
        section_id = section_id.split(".")[0]
        if year_or_month.isdigit():
            dataset_path = os.path.join(dataset_base_path, year_or_month, f"{section_id}.csv")
        else:
            dataset_path = os.path.join(dataset_base_path, year_or_month.split("-")[0], f"{section_id}.csv")
        if dataset_path.endswith(".csv"):
            dataset = pd.read_csv(dataset_path)
        else:
            dataset = pd.read_pickle(dataset_path)
        dataset["date"] = pd.to_datetime(dataset["webPublicationDate"]).dt.date
        events = pd.read_csv(events_path)
        events["date"] = pd.to_datetime(events["date"]).dt.date
        all_events.append(events)

    selected_trainings, all_events, evaluation_paths = find_files_with_prefixes(
        selected_trainings, all_events, selected_events, evaluations_base_path
    )

    with Pool(processes=config["processes"]) as pool:
        func = partial(calculate_deltas_for_dataset, model_name=model_name)
        results = tqdm(pool.starmap(func, zip(selected_trainings, all_events, evaluation_paths)))

    all_deltas_df = pd.concat(results, ignore_index=True)

    unique_evaluation_paths = sorted(set(evaluation_paths))
    for evaluation_path in unique_evaluation_paths:
        deltas_df = all_deltas_df[all_deltas_df["evaluation_path"] == evaluation_path]
        deltas_df.to_csv(os.path.join(evaluation_path, "deltas.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configuration", help="Configuration for evaluation.")

    args = parser.parse_args()

    with open(args.configuration, "r") as file:
        main(yaml.safe_load(file))
