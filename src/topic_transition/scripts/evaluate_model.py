"""Tools for evaluating a specific trained model."""
import argparse
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from topic_transition.evaluation import calculate_avg_indicators_for_dataset, calculate_significance_scores, get_events_with_scores


def calculate_deltas_for_dataset(
    training_path: str,
    events: pd.DataFrame,
    evaluation_path: str,
    topn_events: int,
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
        "top_n": [],
        "evaluation_path": [],
        "year": [],
        "section_id": [],
    }

    year, section_id = training_path.split(os.path.sep)[-3:-1]

    significant_events = []

    sorted_events = events.sort_values(by="significance_score", ascending=False)
    most_significant_events = sorted_events.iloc[:topn_events]
    significant_events.append(most_significant_events)

    significant_events_df = pd.concat(significant_events)
    significant_events_df.reset_index(inplace=True, drop=True)
    significant_events_df["year"] = pd.to_datetime(significant_events_df["date"]).dt.year

    indicators_path = os.path.join(training_path, "indicator_values.csv")
    indicators = pd.read_csv(indicators_path)
    indicators["date"] = pd.to_datetime(indicators["date"]).dt.date
    indicator_max_date = indicators.loc[indicators["indicator_value"].argmax()]["date"]  # type: ignore
    sorted_events = events.sort_values(by="significance_score", ascending=False)

    def get_closest_topk(indicator_max_date, topk_events):
        closest_event = None
        min_delta = None
        for _, event in topk_events.iterrows():
            delta = abs((event["date"] - indicator_max_date).days)
            if (closest_event is None) or (delta < min_delta):
                closest_event = event
                min_delta = delta
        return closest_event, min_delta

    indicators.to_csv(os.path.join(evaluation_path, f"{year}_{section_id}_indicators.csv"), index=False)
    for topk in range(1, topn_events + 1):
        topk_events = sorted_events.iloc[:topk]
        assert "description" in topk_events.columns
        closest_event, min_delta = get_closest_topk(indicator_max_date, topk_events)
        deltas["model_name"].append(model_name)
        deltas["delta"].append(min_delta)
        deltas["date"].append(closest_event["date"])
        deltas["description"].append(closest_event["description"])
        deltas["top_n"].append(topk)
        deltas["evaluation_path"].append(evaluation_path)
        deltas["year"].append(year)
        deltas["section_id"].append(section_id)
    return pd.DataFrame.from_dict(deltas)


def calculate_avg_indicators_for_dataset_from_training(training_path: str, events: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average indicators.

    Parameters
    ----------
    training_path
        The file path to the directory containing the indicators csv file.
    events
        A DataFrame containing the event data.
    """
    indicators_path = os.path.join(training_path, "indicator_values.csv")
    indicators = pd.read_csv(indicators_path)
    indicators["date"] = pd.to_datetime(indicators["date"]).dt.date
    return calculate_avg_indicators_for_dataset(indicators, events)


def get_events_from_training(
    selected_training: str, dataset_base_path: str, events_base_path: str
) -> pd.DataFrame[str]:
    """
    Get events based on paths to training results.

    Parameters
    ----------
    selected_training
        The path or identifier for the selected training dataset.
    dataset_base_path
        The base path where all datasets are stored.
    events_base_path
        The base path where event data related to the trainings are stored.
    """
    return get_events_with_scores(
        training_path=selected_training, dataset_base_path=dataset_base_path, events_base_path=events_base_path
    )


def find_files_with_prefixes(
    selected_trainings: list[str], all_events: list[str], event_paths: list[str], evaluation_base_path: str
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


def determine_training_type(selected_trainings: str):
    """
    Determine the training type based on the training paths in selected_trainings.

    Parameters
    ----------
    selected_trainings
        A list of training paths.
    """
    TRAINING_TYPES = {"generated", "artificial_split", "guardian"}
    detected_types = set()

    for training_path in selected_trainings:
        # Check if the training path contains one of the valid types
        matches = [typ for typ in TRAINING_TYPES if typ in training_path]
        if len(matches) != 1:
            raise ValueError(
                f"Training path '{training_path}' must contain exactly one valid training type: {TRAINING_TYPES}"
            )
        detected_types.add(matches[0])

    if len(detected_types) > 1:
        raise ValueError(
            f"Training paths contain multiple training types: {detected_types}. Ensure all paths are consistent."
        )

    return detected_types.pop()


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
    topn_events = config["topn_events"]
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
        calculate_significance_scores(dataset, events, "yearly")
        all_events.append(events)

    selected_trainings, all_events, evaluation_paths = find_files_with_prefixes(
        selected_trainings, all_events, selected_events, evaluations_base_path
    )

    with Pool(processes=config["processes"]) as pool:
        func = partial(
            calculate_deltas_for_dataset,
            topn_events=topn_events,
            model_name=model_name,
        )
        results = tqdm(pool.starmap(func, zip(selected_trainings, all_events, evaluation_paths)))

    all_deltas_df = pd.concat(results, ignore_index=True)

    unique_evaluation_paths = sorted(set(evaluation_paths))
    for evaluation_path in unique_evaluation_paths:
        deltas_df = all_deltas_df[all_deltas_df["evaluation_path"] == evaluation_path]
        deltas_df.to_csv(os.path.join(evaluation_path, "deltas.csv"), index=False)
        grouped = deltas_df.groupby(["model_name", "top_n"])["delta"].mean().reset_index()
        pivot_df = grouped.pivot(index="model_name", columns="top_n", values="delta").reset_index()
        pivot_df.columns = ["model_name"] + [f"top_{int(col)}_delta" for col in pivot_df.columns[1:]]  # type: ignore
        pivot_df.to_csv(os.path.join(evaluation_path, "deltas.csv"), index=False)

        with Pool(processes=config["processes"]) as pool:
            func = partial(calculate_avg_indicators_for_dataset_from_training)
            averages = tqdm(pool.starmap(func, zip(selected_trainings, all_events)))

        concatenated_df = pd.concat(averages, ignore_index=True)
        mean_values = concatenated_df.mean()
        average_of_averages_df = pd.DataFrame(mean_values).T
        average_of_averages_df.to_csv(os.path.join(evaluation_path, "average_of_averages.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configuration", help="Configuration for evaluation.")

    args = parser.parse_args()

    with open(args.configuration, "r") as file:
        config = yaml.safe_load(file)

    main(config)
