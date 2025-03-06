"""Tools for calculating Total Variation Distance and."""
import os
from datetime import date, timedelta
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from topic_transition.evaluation import get_events_from_path
from topic_transition.utils import get_dates_for_interval, load_or_train_lda, total_variation_distance


def calculate_dataset_tvd(
    dataset: pd.DataFrame, dist: float = 1, first_split_date: str | None = None, first_split_idx: int | None = None
) -> pd.DataFrame:
    """Calculate Total Variation Distances for each date of the dataset."""
    start_date = dataset["date"].iloc[0]
    end_date = dataset["date"].iloc[-1]
    dates = get_dates_for_interval(start_date, end_date)

    if first_split_idx is None:
        if first_split_date is None:
            first_split_idx = dist
        else:
            dates_mm_dd = [date.strftime("%m-%d") for date in dates]
            if first_split_date not in dates_mm_dd:
                raise ValueError(f"Invalid first split date: {first_split_date}")
            first_split_idx = dates_mm_dd.index(first_split_date)

    if first_split_idx > 0:
        dates = dates[first_split_idx:-first_split_idx]

    tvds: dict[str, list] = {"date": [], "indicator_value": []}
    for dt in dates:
        tvds["date"].append(dt)
        if dist == np.inf:
            iv_start = start_date
            iv_end = end_date
        else:
            iv_start = dt - timedelta(days=dist)
            if iv_start < start_date:
                iv_start = start_date
            iv_end = dt + timedelta(days=dist)
            if iv_end > end_date:
                iv_end = end_date
        left_data = dataset[(dataset["date"] >= iv_start) & (dt > dataset["date"])]
        if len(left_data) > 0:
            left_dists = np.stack(left_data["topic_distribution"])  # type: ignore
            left_dist = left_dists.mean(axis=0)
        else:
            tvds["indicator_value"].append(0)
            continue
        right_data = dataset[(dataset["date"] <= iv_end) & (dt < dataset["date"])]
        if len(right_data) > 0:
            right_dists = np.stack(right_data["topic_distribution"])  # type: ignore
            right_dist = right_dists.mean(axis=0)
        else:
            tvds["indicator_value"].append(0)
            continue
        tvd = total_variation_distance(left_dist, right_dist)
        tvds["indicator_value"].append(tvd)
    return pd.DataFrame.from_dict(tvds)


def calculate_topict_distribution_tvd(
    dataset_path: str,
    lda_path: str,
    selected_month: str | None,
    lda_config: dict,
    tvd_l: float,
    first_split_date: str,
    force_new_tvd: bool = False,
    first_split_idx: int | None = None,
):
    """Calculate deltas for all possible methods."""
    dataset = pd.read_pickle(dataset_path)
    dataset["date"] = pd.to_datetime(dataset["webPublicationDate"]).dt.date

    if selected_month is not None:
        selected_year, selected_month = selected_month.split("-")

        start_date = date(int(selected_year), int(selected_month), 1)
        end_date = start_date + relativedelta(months=1)
        start_date = start_date - timedelta(days=tvd_l)
        end_date = end_date + timedelta(days=tvd_l)
        dataset = dataset[(dataset["date"] >= start_date) & (dataset["date"] < end_date)]

    os.makedirs(lda_path, exist_ok=True)
    corpus, lda_model = load_or_train_lda(dataset, force_new_tvd, lda_config, lda_path)

    dataset["topic_distribution"] = [
        np.array(lda_model.get_document_topics(bow, minimum_probability=0))[:, 1] for bow in corpus
    ]
    return calculate_dataset_tvd(
        dataset, dist=tvd_l, first_split_date=first_split_date, first_split_idx=first_split_idx
    )


def get_tvd_metrics(all_events, config, dataset_paths, lda_config):
    """Calculate quality metrics for TVD for specific datasets."""
    if config["tvd_l"] == "inf":
        tvd_l = np.inf
    else:
        tvd_l = int(config["tvd_l"])
    first_split_date = config["first_split_date"] if "first_split_date" in config else None
    first_split_idx = config["first_split_idx"] if "first_split_idx" in config else None
    with Pool(processes=config["processes"]) as pool:
        lda_paths = config["lda_paths"]
        func = partial(
            calculate_topict_distribution_tvd,
            lda_config=lda_config,
            tvd_l=tvd_l,
            first_split_date=first_split_date,
            force_new_tvd=config["force_new_tvd"],
            first_split_idx=first_split_idx,
        )
        if "selected_months" in config:
            selected_months = config["selected_months"]
        else:
            selected_months = [None for _ in range(len(dataset_paths))]

        tvds = tqdm(pool.starmap(func, zip(dataset_paths, lda_paths, selected_months)), "Calculating TVDs")
    with Pool(processes=config["processes"]) as pool:
        func = partial(calculate_deltas_for_dataset, tvd_l=config["tvd_l"])
        path_parts = [dataset_path.split(os.path.sep)[-2:] for dataset_path in dataset_paths]
        years = [parts[0] for parts in path_parts]
        section_ids = [parts[1].split(".")[0] for parts in path_parts]
        results = tqdm(pool.starmap(func, zip(years, section_ids, all_events, tvds)), "Calculating deltas")
    deltas_df = pd.concat(results, ignore_index=True)
    return deltas_df, tvds


def load_all_events(config):
    """Load all event DataFrames for each dataset."""
    with Pool(processes=config["processes"]) as pool:
        events_paths = config["selected_events"]
        all_events = pool.starmap(get_events_from_path, zip(events_paths))
    return all_events


def calculate_topic_distribution_from_base_path(
    dataset_path,
    lda_base_path,
    selected_month,
    lda_config,
    tvd_l,
    first_split_date,
    force_new_tvd,
    first_split_idx: int | None = None,
):
    """Calculate LDA toopic distributions using dataset and root path to ldas."""
    year, section_id = dataset_path.split(os.path.sep)[-2:]
    section_id = section_id[:-4]
    lda_path = os.path.join(lda_base_path, year)
    lda_path = os.path.join(lda_path, section_id)
    return calculate_topict_distribution_tvd(
        dataset_path,
        lda_path,
        selected_month,
        lda_config,
        tvd_l,
        first_split_date,
        force_new_tvd,
        first_split_idx=first_split_idx,
    )


def calculate_deltas_for_dataset(
    year: str, section_id: str, events: pd.DataFrame, tvd: pd.DataFrame, tvd_l: int
) -> pd.DataFrame:
    """Calculate deltas for all possible methods."""
    deltas: dict[str, list] = {
        "model_name": [],
        "delta": [],
        "date": [],
        "description": [],
        "year": [],
        "section_id": [],
    }

    def get_closest(indicator_max_date, topk_events):
        closest_event = None
        min_delta = None
        for _, event in topk_events.iterrows():
            delta = abs((event["date"] - indicator_max_date).days)
            if (closest_event is None) or (delta < min_delta):
                closest_event = event
                min_delta = delta
        return closest_event, min_delta

    indicator_max_idx = tvd["indicator_value"].argmax()
    indicator_max_date = tvd.loc[indicator_max_idx, "date"]
    closest_event, min_delta = get_closest(indicator_max_date, events)
    deltas["model_name"].append(f"tvd_{tvd_l}")
    deltas["delta"].append(min_delta)
    deltas["date"].append(closest_event["date"])
    deltas["description"].append(closest_event["description"])
    deltas["year"].append(year)
    deltas["section_id"].append(section_id)
    return pd.DataFrame.from_dict(deltas)
