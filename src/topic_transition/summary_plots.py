"""Tools for generating summary plots."""
import os
from glob import glob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter


def stde(x):
    """Get standard error."""
    return np.std(x) / np.sqrt(len(x))


def plot_indicators(grouped, indicators, output_root_path, xticks_format="YY-MM", show_closest_events=False):
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

        # Add vertical lines with slight offsets
        if xticks_format == "dd":
            display_delta = pd.Timedelta(hours=3)
        else:
            display_delta = pd.Timedelta(days=2)

        displayed_max_date = model_max_date + display_delta
        ax.axvline(displayed_max_date, color="blue", linestyle="--", label="Model Prediction")

        #if group is not None:
        model_closest_event_date = group["event_date"].iloc[0]
        if not pd.isna(model_closest_event_date):
            ax.axvline(
                model_closest_event_date,
                color="purple",
                linestyle=":",
                label="Model Event",
            )

        # Show TVD baseline indicator plot
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
            ax.axvline(
                baseline_max_date,
                color="red",
                linestyle="-"
            )

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
        #plt.ylim(0.0, 1.0)  # Set y-axis limits

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


def aggregate_indicators(grouped, rescale_models):
    """Calculate mean and stde indicators for all 5 iterations."""
    group_sizes = grouped.size()

    #assert np.all((group_sizes == 5).values)
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


def get_indicators_with_event(config):
    """Get DataFrame with all indicators."""

    all_indicators = []
    for model in config["evaluations"]:
        first_iter_path = config["evaluations"][model][0]
        events = load_events(first_iter_path)

        indicators_data = {
            "time_interval": [],
            "section": [],
            "model": [],
            "baseline_model": [],
            "indicator_path": [],
            "indicator_filename": [],
        }

        for iteration_path in config["evaluations"][model]:
            indicator_paths = glob(os.path.join(iteration_path, "*_indicators.csv"))
            for indicator_path in indicator_paths:
                filename = os.path.basename(indicator_path)[:-4]
                parts = filename.split("_")

                if len(parts) == 5:
                    interval = parts[0]
                    section = "_".join(parts[1:3])
                elif len(parts) == 4:
                    interval = "_".join(parts[:2])
                    section = parts[2]
                elif len(parts) == 3:
                    interval = parts[0]
                    section = parts[-2]
                else:
                    raise ValueError(f"Invalid indicators file: {indicator_path}")

                indicators_data["time_interval"].append(interval)
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
                events.rename(columns={"date": "event_date"}), on=["section", "time_interval"], how="inner"
            )
        all_indicators.append(indicator_df)
    indicator_df = pd.concat(all_indicators, ignore_index=True)
    indicator_df["indicator_fname"] = indicator_df["indicator_path"].apply(lambda path: path.split(os.path.sep)[-1])
    indicator_df.drop_duplicates("indicator_path", inplace=True)
    return indicator_df


def load_events(first_iter_path):
    events_path = os.path.join(first_iter_path, "deltas.csv")
    events = pd.read_csv(events_path)
    events.drop(["model_name"], axis=1, inplace=True)
    if "evaluation_path" in events.columns:
        events.drop(["evaluation_path"], axis=1, inplace=True)
    events.reset_index(inplace=True, drop=True)
    events.rename({"section_id": "section", "year": "time_interval"}, axis=1, inplace=True)
    events["time_interval"] = events["time_interval"].astype("string")
    events["date"] = pd.to_datetime(events["date"])
    events["year"] = events["date"].dt.year
    events["date"] = events["date"].dt.date
    events.drop_duplicates(subset=["date", "section"], inplace=True, ignore_index=True)
    # In the artificial split data the sections have a special format that is incompatible
    if np.all(events["section"].apply(lambda section: len(section.split("_"))) == 3):
        events["section"] = events["section"].apply(lambda section: "_".join(section.split("_")[:2]))
    return events


def get_top1_significant_events(manual_events_path, significant_events_path):
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


def generate_summary_plots(config):
    """Plot aggregate indicator values."""
    summary_path = config["summary_path"]
    rescale_models = config["rescale_models"] if "rescale_models" in config else False

    output_path = summary_path

    os.makedirs(output_path, exist_ok=True)
    indicator_df = get_indicators_with_event(config)
    grouped = indicator_df.groupby(["time_interval", "section", "model", "indicator_fname"])
    indicators = aggregate_indicators(grouped, rescale_models)
    plot_indicators(grouped, indicators, output_path, config["xticks_format"])
