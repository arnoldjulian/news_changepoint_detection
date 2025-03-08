"""Tools for evaluating Total Variational Distance with Latent Dirichlet Allocation."""
import argparse
import copy
import os
from datetime import date, timedelta

import nltk
import yaml
from dateutil.relativedelta import relativedelta
from nltk.corpus import stopwords

from topic_transition.tvd import get_tvd_metrics, load_all_events

nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def main(base_config: dict) -> None:
    """Evaluate TVD on selected datasets."""
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
        split_distance = base_config["tvd_l"]
        selected_data_intervals = []
        for selected_month in selected_months:
            selected_year, selected_month = selected_month.split("-")
            start_date = date(int(selected_year), int(selected_month), 1)
            end_date = start_date + relativedelta(months=1)
            start_date = start_date - timedelta(days=split_distance)
            end_date = end_date + timedelta(days=split_distance)
            time_interval = f"{start_date}-{end_date}"
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
