"""Tools for generating summary metrics."""
import os

import numpy as np
import pandas as pd

from topic_transition.utils import find_matching_directories


def bold_max_min_in_column(mean, stde, optimal_value):
    """
    Return bolded value.

    Parameters
    ----------
    mean : float
        The numerical value to format.
    stde:
        Standard Error.
    optimal_value : float
        The maximum value in the column.
    """
    formatted_value = f"{mean:.2f} ± {stde:.2f}"
    return f"\\textbf{{{formatted_value}}}" if mean == optimal_value else formatted_value


def split_model_name(model_name: str) -> tuple[str, str, float | int]:
    """
    Split the model_name2 into base model and L value.

    Parameters
    ----------
    model_name
        The complete model name.
    """
    parts = model_name.split(",L=")
    base_model = parts[0]

    if len(parts) > 1:
        l_value = parts[1]
        if l_value == "inf":
            l_value_weight = float("inf")
        else:
            l_value_weight = int(l_value)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return base_model, l_value, l_value_weight


def generate_delta_latex_table(summary: dict, output_path: str):
    """
    Save latex snippet of the indicator deltas table.

    Parameters
    ----------
    summary
        DataFrame containing delta metrics for different models.
    output_path
        Path where the generated LaTeX table will be saved.
    """
    table_content = "\\begin{table}[htbp]\n\\centering\n"
    table_content += "\\caption{Delta metrics for different models}\n\\small\n"
    table_content += "\\begin{tabular}{lc}\n\\toprule\n"
    table_content += "model name &"
    table_content += "delta ± stde"
    table_content += "\\\\\n\\midrule\n"

    for _, row in summary.iterrows():
        table_content += f"{row['model_name2']}"
        mean = row["delta_mean"]
        stde = row["delta_stde"]
        table_content += f" & {bold_max_min_in_column(mean, stde, summary['delta_mean'].min())}"
        table_content += "\\\\\n"

    table_content += "\\bottomrule\n\\end{tabular}\n\\medskip\n\\end{table}\n"

    with open(output_path, "w") as file:
        file.write(table_content)


def summarize_metrics(config: dict):
    """Generate the delta metrics table."""
    evaluated_models = config["evaluations"]
    summary_path = config["summary_path"]
    os.makedirs(summary_path, exist_ok=True)
    all_summaries = []
    for model_name, paths in evaluated_models.items():
        if not isinstance(paths, list):
            paths = find_matching_directories(paths)
        list_of_dfs = []
        for path in paths:
            df = pd.read_csv(os.path.join(path, "deltas.csv"))
            list_of_dfs.append(df)
        delta_df = pd.concat(list_of_dfs)
        delta_columns = [col for col in delta_df.columns if "delta" in col]

        def stde(x):
            return np.std(x) / np.sqrt(len(x))

        agg_dict = {column: ["mean", stde] for column in delta_columns}
        summary_df = delta_df.groupby("model_name").agg(agg_dict).reset_index()  # type: ignore
        summary_df.columns = [
            "_".join(col).strip() if col[1] else col[0] for col in summary_df.columns.values
        ]  # type: ignore
        summary_df["model_name2"] = model_name
        all_summaries.append(summary_df)
        model_summary_path = os.path.join(summary_path, model_name)
        os.makedirs(model_summary_path, exist_ok=True)
    summary = pd.concat(all_summaries, ignore_index=True)
    summary.to_csv(os.path.join(summary_path, "metrics.csv"), index=False)
    latex_output_2 = os.path.join(summary_path, "delta_table.tex")
    split_results = summary["model_name2"].apply(split_model_name)
    summary[["base_model", "l_value", "l_value_weight"]] = pd.DataFrame(split_results.tolist(), index=summary.index)
    summary = summary.sort_values(by=["base_model", "l_value_weight"])
    generate_delta_latex_table(summary, latex_output_2)
