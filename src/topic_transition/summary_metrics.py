"""Tools for generating summary metrics."""
import os

import numpy as np
import pandas as pd

from topic_transition.utils import find_matching_directories


def bold_max_min_in_column(value, std, max_value):
    """
    Return bolded value.

    Parameters
    ----------
    value : float
        The numerical value to format.
    max_value : float
        The maximum value in the column.
    is_average : bool
        Flag indicating if the value is an average value.

    Returns
    -------
    str
        The formatted value; bold if it matches the max_value.
    """
    formatted_value = f"{value:.2f} ± {std:.2f}"
    return f"\\textbf{{{formatted_value}}}" if value == max_value else formatted_value


def split_model_name(model_name2):
    """
    Split the model_name2 into base model and L value.

    Parameters
    ----------
    model_name2 : str
        The complete model name.

    Returns
    -------
    tuple
        A tuple containing the base model and L value.
    """
    parts = model_name2.split(",L=")
    base_model = parts[0]

    if len(parts) > 1:
        l_value = parts[1]
        if l_value == "inf":
            l_value_weight = float("inf")
        else:
            l_value_weight = int(l_value)
    else:
        l_value_weight = float("inf")  # Default to inf if no L value

    return base_model, l_value, l_value_weight


def generate_delta_latex_table(summary, output_path, top_n_delta):
    """
    Save latex snippet of the indicator deltas table.

    Parameters
    ----------
    summary : pandas.DataFrame
        DataFrame containing delta metrics for different models.
    output_path : str
        Path where the generated LaTeX table will be saved.
    """
    min_top_deltas = [summary[f"top_{i}_delta_mean"].min() for i in range(1, top_n_delta + 1)]

    table_content = "\\begin{table}[htbp]\n\\centering\n"
    table_content += "\\caption{Delta metrics for different models}\n\\label{tab:model_deltas}\n\\small\n"
    column_alignments = "l" + "".join(["c" for _ in range(top_n_delta)])
    table_content += "\\begin{tabular}{" + column_alignments + "}\n\\toprule\n"
    table_content += "model name &"

    for i in range(1, top_n_delta + 1):
        table_content += f"top {i} delta±stde"
    table_content += "\\\\\n\\midrule\n"

    for _, row in summary.iterrows():
        table_content += f"{row['model_name2']}"
        for i in range(1, top_n_delta + 1):
            mean = row[f"top_{i}_delta_mean"]
            stde = row[f"top_{i}_delta_stde"]
            table_content += f" & {bold_max_min_in_column(mean, stde,min_top_deltas[i-1])}"
        table_content += "\\\\\n"

    table_content += "\\bottomrule\n\\end{tabular}\n\\medskip\n\\end{table}\n"

    with open(output_path, "w") as file:
        file.write(table_content)


def summarize_metrics(config):
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
            df = pd.read_csv(os.path.join(path, "avg_tfidf_score_deltas.csv"))
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
    # Sort the DataFrame by base_model and l_value_weight
    summary = summary.sort_values(by=["base_model", "l_value_weight"])
    generate_delta_latex_table(summary, latex_output_2, config["top_n_delta"])
