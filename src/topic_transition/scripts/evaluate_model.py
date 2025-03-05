"""Tools for evaluating a specific trained model."""
import argparse
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm


def calculate_deltas_for_dataset(
    training_path: str,
    events: pd.DataFrame,
    evaluation_path: str,
    model_name: str,
) -> pd.DataFrame:
    """
    Calculate top-k deltas for a selected evaluation.

    Parameters
    ----------
    training_path
        The path to the training data directory.
    events:
        List of events with dates and descriptions.
    evaluation_path
        Directory path where we save the results.
    topn_events
        The number of top events to consider for top-k delta.
    model_name
        The name of the model.
    """
    deltas: dict[str, list] = {
        "model_name": [],
        "delta": [],
        "date": [],
        "description": [],
        "evaluation_path": [],
        "year": [],
        "section_id": [],
    }

    year, section_id = training_path.split(os.path.sep)[-3:-1]

    indicators_path = os.path.join(training_path, "indicator_values.csv")
    indicators = pd.read_csv(indicators_path)
    indicators["date"] = pd.to_datetime(indicators["date"]).dt.date
    indicator_max_date = indicators.loc[indicators["indicator_value"].argmax()]["date"]  # type: ignore

    def get_closest(indicator_max_date, events):
        closest_event = None
        min_delta = None
        for _, event in events.iterrows():
            delta = abs((event["date"] - indicator_max_date).days)
            if (closest_event is None) or (delta < min_delta):
                closest_event = event
                min_delta = delta
        return closest_event, min_delta

    indicators.to_csv(os.path.join(evaluation_path, f"{year}_{section_id}_indicators.csv"), index=False)
    closest_event, min_delta = get_closest(indicator_max_date, events)
    deltas["model_name"].append(model_name)
    deltas["delta"].append(min_delta)
    deltas["date"].append(closest_event["date"])
    deltas["description"].append(closest_event["description"])
    deltas["evaluation_path"].append(evaluation_path)
    deltas["year"].append(year)
    deltas["section_id"].append(section_id)
    return pd.DataFrame.from_dict(deltas)


def find_files_with_prefixes(
    selected_trainings: list[str], all_events: list[pd.DataFrame], event_paths: list[str], evaluation_base_path: str
) -> tuple[list[str], list[str], list[str]]:
    """
    Find files with given prefixes and adjust the `all_events` list.

    Parameters
    ----------
    selected_trainings
        A list of training file path prefixes.
    all_events
        A list of event paths, one for each prefix in `selected_trainings`.
    event_paths
        A list of paths for event csvs.
    evaluation_base_path
        The root directory for all evaluation results.
    """
    adjusted_trainings = []
    adjusted_events = []
    adjusted_event_paths = []
    evaluation_paths = []
    for selected_training, event, event_path in zip(selected_trainings, all_events, event_paths, strict=True):
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


def main(config: dict) -> None:
    """
    Evaluate all trainings of a selected model type, including multiple iterations of the same training.

    Parameters
    ----------
    config
        A dictionary containing the configuration for the main function.
    """
    model_name = config["model_name"]
    evaluations_base_path = config["evaluation_base_path"]
    selected_trainings = config["selected_trainings"]
    dataset_base_path = config["dataset_base_path"]
    selected_events = config["selected_events"]

    all_events = []
    for events_path in selected_events:
        year, section_id = events_path.split(os.path.sep)[-2:]
        section_id = section_id.split(".")[0]
        dataset_path = os.path.join(dataset_base_path, year, f"{section_id}.pkl")
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
        config = yaml.safe_load(file)

    main(config)
