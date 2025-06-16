"""Tools for generating summary plots."""
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter


def stde(x):
    """Get standard error."""
    return np.std(x) / np.sqrt(len(x))


def plot_indicators(grouped, indicators, output_root_path, xticks_format="YY-MM"):
    """Plot indicators for all times, sections and models."""
    for key, group in list(grouped):
        agg_indicators = indicators["_".join(key[:3])]
        plt.figure(figsize=(10, 6))
        agg_indicators["datetime"] = pd.to_datetime(agg_indicators["date"])
        plt.plot(agg_indicators["datetime"], agg_indicators["mean"], label="Model Indicator Mean", color="blue")
        ax = plt.gca()
        plt.fill_between(
            agg_indicators["date"],
            agg_indicators["mean"] - agg_indicators["stde"],
            agg_indicators["mean"] + agg_indicators["stde"],
            color="blue",
            alpha=0.2,
            label="± Std Err",
        )
        max_idx = agg_indicators["mean"].argmax()
        model_max_date = agg_indicators["datetime"].iloc[max_idx]

        if xticks_format == "dd":
            display_delta = pd.Timedelta(hours=3)
        else:
            display_delta = pd.Timedelta(days=2)

        displayed_max_date = model_max_date + display_delta
        ax.axvline(displayed_max_date, color="blue", linestyle="--", label="Model Prediction")

        model_closest_event_date = group["event_date"].iloc[0]
        if not pd.isna(model_closest_event_date):
            ax.axvline(
                model_closest_event_date,
                color="purple",
                linestyle=":",
                label="Model Event",
            )

        baseline_model = group["baseline_model"].iloc[0]
        if not pd.isna(baseline_model):
            baseline_key = f"{key[0]}_{key[1]}_{baseline_model}"
            baseline_agg_indicators = indicators[baseline_key]
            baseline_agg_indicators["datetime"] = pd.to_datetime(baseline_agg_indicators["date"])
            model_mean = agg_indicators["mean"].mean()
            baseline_mean = baseline_agg_indicators["mean"].mean()
            baseline_agg_indicators["mean"] = baseline_agg_indicators["mean"] + model_mean - baseline_mean
            agg_indicators = baseline_agg_indicators
            plt.plot(agg_indicators["datetime"], agg_indicators["mean"], label="TVD Indicator Mean", color="red")
            plt.fill_between(
                agg_indicators["date"],
                agg_indicators["mean"] - agg_indicators["stde"],
                agg_indicators["mean"] + agg_indicators["stde"],
                color="red",
                alpha=0.2,
                label="± Std Err",
            )
            max_idx = agg_indicators["mean"].argmax()
            baseline_max_date = agg_indicators["datetime"].iloc[max_idx]
            baseline_max_date -= display_delta
            ax.axvline(baseline_max_date, color="red", linestyle="-")

            baseline_group_key = (key[0], key[1], baseline_model, key[3])
            baseline_closest_event_date = grouped.get_group(baseline_group_key)["event_date"].iloc[0]
            if not pd.isna(baseline_closest_event_date) and baseline_closest_event_date != model_closest_event_date:
                ax.axvline(
                    baseline_closest_event_date,
                    color="orange",
                    linestyle=":",
                    label="TVD Event",
                )

        plt.xlabel("Date")
        plt.ylabel("Indicator Value")

        if xticks_format == "dd":
            ax.xaxis.set_major_formatter(DateFormatter("%d"))  # Show only day of the month
        elif xticks_format == "YY-MM":
            ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))  # Show year and month
        else:
            ax.xaxis.set_major_formatter(DateFormatter("%Y"))

        output_path = os.path.join(output_root_path, key[2])
        os.makedirs(output_path, exist_ok=True)
        plt.legend()
        plt.savefig(os.path.join(output_path, f"{key[0]}_{key[1]}_{key[2]}.jpg"))
        agg_indicators.to_csv(os.path.join(output_path, f"{key[0]}_{key[1]}_{key[2]}.csv"), index=False)


def aggregate_indicators(grouped):
    """Calculate mean and stde indicators for all 5 iterations."""
    indicators = {}
    for key, group in grouped:
        dfs = []
        for _, row in group.iterrows():
            dfs.append(pd.read_csv(row["indicator_path"]))
        agg_df = pd.concat(dfs, ignore_index=True)

        agg_df["date"] = pd.to_datetime(agg_df["date"]).dt.date

        agg_indicators = agg_df.groupby("date")["indicator_value"].agg(["mean", stde]).reset_index()

        indicators["_".join(key[:3])] = agg_indicators
    return indicators


def get_all_iteration_deltas(config):
    """Get DataFrame with all indicators."""
    all_deltas = []
    for model in config["evaluations"]:
        for iteration_path in config["evaluations"][model]:
            event_deltas = load_event_deltas(iteration_path)
            dirname = iteration_path.split(os.path.sep)[-1]
            parts = dirname.split("_num_")
            if len(parts) > 1:
                iteration = int(parts[-1])
            else:
                iteration = 0
            event_deltas["iteration"] = iteration
            event_deltas["model"] = model

            all_deltas.append(event_deltas)
    return pd.concat(all_deltas, ignore_index=True)


def load_event_deltas(evaluation_path):
    """Load events with deltas."""
    events_path = os.path.join(evaluation_path, "deltas.csv")
    events = pd.read_csv(events_path)
    events.drop(["model_name"], axis=1, inplace=True)
    if "evaluation_path" in events.columns:
        events.drop(["evaluation_path"], axis=1, inplace=True)
    events.reset_index(inplace=True, drop=True)
    events.rename({"section_id": "section"}, axis=1, inplace=True)
    events["time_interval"] = events["time_interval"].astype("string")
    events["date"] = pd.to_datetime(events["date"])
    events["year"] = events["date"].dt.year
    events["event_date"] = events["date"].dt.date
    events.drop_duplicates(subset=["date", "section"], inplace=True, ignore_index=True)
    # In the artificial split data the sections have a special format that is incompatible
    if np.all(events["section"].apply(lambda section: len(section.split("_"))) == 3):
        events["section"] = events["section"].apply(lambda section: "_".join(section.split("_")[:2]))
    events["indicator_path"] = (
        evaluation_path + os.path.sep + events["time_interval"] + "_" + events["section"] + "_indicators.csv"
    )
    return events


def generate_summary_plots(config):
    """Plot aggregate indicator values."""
    summary_path = config["summary_path"]

    output_path = summary_path

    os.makedirs(output_path, exist_ok=True)
    indicator_df = get_all_iteration_deltas(config)
    grouped = indicator_df.groupby(["time_interval", "section", "model", "iteration"])
    indicators = aggregate_indicators(grouped)
    plot_indicators(grouped, indicators, output_path, config["xticks_format"])
