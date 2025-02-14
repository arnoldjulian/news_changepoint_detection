"""Tools for generating summary plots."""
import os
from glob import glob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter


def stde(arr: np.array) -> float:
    """Get standard error."""
    return np.std(arr) / np.sqrt(len(arr))


def plot_indicators(events: pd.DataFrame, grouped: pd.api.typing.DataFrameGroupBy,
                    indicators: pd.DataFrame, output_root_path: str, show_max_tvd: bool, xticks_format: str = "YY-MM"):
    """Plot indicators for all times, sections and models."""
    for key, group in list(grouped):
        agg_indicators = indicators["_".join(key[:3])]
        plt.figure(figsize=(10, 6))
        plt.plot(agg_indicators["date"], agg_indicators["mean"], label="Mean", color="blue")
        plt.fill_between(
            agg_indicators["date"],
            agg_indicators["mean"] - agg_indicators["stde"],
            agg_indicators["mean"] + agg_indicators["stde"],
            color="blue",
            alpha=0.2,
            label="± Std Err",
        )

        plt.xlabel("Date")
        plt.ylabel("Indicator Value")

        ax = plt.gca()

        if xticks_format == "dd":
            ax.xaxis.set_major_formatter(DateFormatter("%d"))  # Show only day of the month
        elif xticks_format == "YY-MM":
            ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))  # Show year and month
        else:
            ax.xaxis.set_major_formatter(DateFormatter("%Y"))

        max_mean_date = agg_indicators.loc[agg_indicators["mean"].idxmax(), "date"]

        ax.axvline(
            max_mean_date,
            color="blue",
            linestyle="--",
            label="Max Mean indicator value",
        )

        if show_max_tvd:
            baseline_model = group["baseline_model"].iloc[0]
            if not pd.isna(baseline_model):
                baseline_key = f"{key[0]}_{key[1]}_{baseline_model}"
                assert baseline_key in indicators
                baseline_agg_indicators = indicators[baseline_key]
                max_mean_baseline_date = baseline_agg_indicators.loc[baseline_agg_indicators["mean"].idxmax(), "date"]
                ax.axvline(
                    max_mean_baseline_date,
                    color="red",
                    linestyle="dotted",
                    label="Max Mean TVD indicator value",
                )

        if events is not None:
            event_date = group["event_date"].iloc[0]
            if not pd.isna(event_date):
                ax.axvline(
                    event_date,
                    color="green",
                    linestyle=":",
                    label="Event date",
                )

        output_path = os.path.join(output_root_path, key[2])
        os.makedirs(output_path, exist_ok=True)
        plt.legend()
        plt.savefig(os.path.join(output_path, f"{key[0]}_{key[1]}_{key[2]}.jpg"))


def aggregate_indicators(grouped: pd.api.typing.DataFrameGroupBy, rescale_models: bool) -> dict:
    """Calculate mean and stde indicators for all iterations."""
    group_sizes = grouped.size()

    assert np.all((group_sizes == 5).values)
    indicators = {}
    for key, group in grouped:
        dfs = []
        for _, row in group.iterrows():
            dfs.append(pd.read_csv(row["indicator_path"]))
        agg_df = pd.concat(dfs, ignore_index=True)

        agg_df["date"] = pd.to_datetime(agg_df["date"]).dt.date

        if rescale_models and "TVD" not in key[2]:
            agg_df["rescaled_value"] = 2 * agg_df["indicator_value"] - 1
            agg_indicators = agg_df.groupby("date")["rescaled_value"].agg(["mean", stde]).reset_index()
        else:
            agg_indicators = agg_df.groupby("date")["indicator_value"].agg(["mean", stde]).reset_index()

        indicators["_".join(key[:3])] = agg_indicators
    return indicators


def get_indicators(config: dict, events: pd.DataFrame) -> pd.DataFrame:
    """Get DataFrame with all indicators."""
    indicators_data = {
        "time_interval": [],
        "year": [],
        "section": [],
        "model": [],
        "baseline_model": [],
        "indicator_path": [],
        "indicator_filename": [],
    }
    for model in config["evaluations"]:
        for iteration_path in config["evaluations"][model]:
            indicator_paths = glob(os.path.join(iteration_path, "*_indicators.csv"))
            for indicator_path in indicator_paths:
                filename = os.path.basename(indicator_path)[:-4]
                parts = filename.split("_")

                if len(parts) == 5:
                    interval = parts[0]
                    year = parts[0]
                    section = "_".join(parts[1:3])
                elif len(parts) == 4:
                    interval = "_".join(parts[:2])
                    year = int(parts[0].split("-")[0]) + 1
                    section = parts[2]
                elif len(parts) == 3:
                    interval = parts[0]
                    year = interval
                    section = parts[-2]
                else:
                    raise ValueError(f"Invalid indicators file: {indicator_path}")

                indicators_data["time_interval"].append(interval)
                indicators_data["year"].append(int(year))
                indicators_data["section"].append(section)
                indicators_data["model"].append(model)
                if "TVD" in model:
                    baseline_model = pd.NA
                else:
                    L = model.split(",")[-1].split("=")[-1]
                    baseline_model = f"TVD,L={L}"
                indicators_data["baseline_model"].append(baseline_model)
                indicators_data["indicator_path"].append(indicator_path)
                indicators_data["indicator_filename"].append(filename[: -len("_indicators")])
    indicator_df = pd.DataFrame.from_dict(indicators_data)
    if events is not None:
        indicator_df = indicator_df.merge(
            events.rename(columns={"date": "event_date"}), on=["year", "section", "time_interval"], how="left"
        )
    indicator_df["indicator_fname"] = indicator_df["indicator_path"].apply(lambda path: path.split(os.path.sep)[-1])
    indicator_df.drop_duplicates("indicator_path", inplace=True)
    return indicator_df


def get_events(manual_events_path: str, significant_events_path: str) -> pd.DataFrame | None:
    """Get all events if a path is given."""
    if manual_events_path:
        events = pd.read_csv(manual_events_path)
        events["date"] = pd.to_datetime(events["date"])
        events["year"] = events["date"].dt.year
        events["date"] = events["date"].dt.date
        events["time_interval"] = events["time_interval"].astype("string")
    elif significant_events_path:
        top1_events = pd.read_csv(significant_events_path)
        top1_events = top1_events[top1_events["top_n"] == 1]
        top1_events.drop(["model_name", "score_type", "evaluation_path", "top_n"], axis=1, inplace=True)
        top1_events.reset_index(inplace=True, drop=True)
        top1_events.rename({"section_id": "section", "year": "time_interval"}, axis=1, inplace=True)
        top1_events["time_interval"] = top1_events["time_interval"].astype("string")
        top1_events["date"] = pd.to_datetime(top1_events["date"])
        top1_events["year"] = top1_events["date"].dt.year
        top1_events["date"] = top1_events["date"].dt.date
        top1_events.drop_duplicates(subset=["date", "section"], inplace=True, ignore_index=True)
        # In the artificial split data the sections have a special format that is incompatible
        if np.all(top1_events["section"].apply(lambda section: len(section.split("_"))) == 3):
            top1_events["section"] = top1_events["section"].apply(lambda section: "_".join(section.split("_")[:2]))
        events = top1_events
    else:
        events = None
    return events


def generate_summary_plots(config: dict):
    """Plot aggregate indicator values."""
    summary_path = config["summary_path"]
    manual_events_path = config["manual_events_path"] if "manual_events_path" in config else None
    significant_events_path = config["significant_events_path"] if "significant_events_path" in config else None
    rescale_models = config["rescale_models"] if "rescale_models" in config else False
    show_max_tvd = config["show_max_tvd"] if "show_max_tvd" in config else False
    events = get_events(manual_events_path, significant_events_path)
    dir_parts = ["max_ind"]
    if significant_events_path is not None:
        dir_parts.append("significant_events")
    elif manual_events_path is not None:
        dir_parts.append("manual_events")
    if show_max_tvd:
        dir_parts.append("max_tvd")
    output_path = os.path.join(summary_path, "+".join(dir_parts))
    os.makedirs(output_path, exist_ok=True)
    indicator_df = get_indicators(config, events)
    grouped = indicator_df.groupby(["time_interval", "section", "model", "indicator_fname"])
    indicators = aggregate_indicators(grouped, rescale_models)
    plot_indicators(events, grouped, indicators, output_path, show_max_tvd, config["xticks_format"])
