"""Tools for evaluating a specific model."""

import argparse
import copy
import os
from functools import partial
from multiprocessing import Pool

import nltk
import pandas as pd
import yaml
from nltk.corpus import stopwords
from tqdm import tqdm

from topic_transition.evaluation import calculate_avg_indicators_for_dataset
from topic_transition.tvd import get_tvd_metrics, load_all_events

nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def main(base_config: dict) -> None:
    """
    Evaluate TVD on selected datasets.

    Parameters
    ----------
    base_config : dict
        A dictionary containing the configuration for the main function.

    Returns
    -------
    None
        This function does not return any value, it performs calculations and saves the results.
    """
    training_name = base_config["training_name"]
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
    all_events, topn_events = load_all_events(base_config)

    for i, training_name in enumerate(training_names):
        config = copy.deepcopy(base_config)
        evaluation_path = os.path.join(evaluations_base_path, training_name)

        os.makedirs(evaluation_path, exist_ok=True)

        if i > 0:
            lda_paths = config["lda_paths"]
            for j, lda_path in enumerate(lda_paths):
                lda_paths[j] = f"{lda_path}_num_{i}"

        deltas_df, pivot_df, tvds = get_tvd_metrics(all_events, config, dataset_paths, lda_config, topn_events)

        for tvd, dataset_path in zip(tvds, dataset_paths):
            year, secion_id = dataset_path.split(os.path.sep)[-2:]
            secion_id = secion_id[:-4]
            indicator_path = os.path.join(evaluation_path, f"{year}_{secion_id}_indicators.csv")
            tvd.to_csv(indicator_path, index=False)

        deltas_df.to_csv(os.path.join(evaluation_path, "deltas.csv"), index=False)
        pivot_df.to_csv(os.path.join(evaluation_path, "deltas.csv"), index=False)

        with Pool(processes=config["processes"]) as pool:
            func = partial(calculate_avg_indicators_for_dataset)
            averages = tqdm(pool.starmap(func, zip(tvds, all_events)))

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
