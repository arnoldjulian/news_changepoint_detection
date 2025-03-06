import os

import pandas as pd
import pytest

from topic_transition.summary_metrics import summarize_metrics


@pytest.fixture
def mock_config(tmp_path):
    model_1_path = tmp_path / "random,L=180"
    model_2_path = tmp_path / "TVD,L=180"
    summary_path = tmp_path / "summary"
    os.makedirs(model_1_path, exist_ok=True)
    os.makedirs(model_2_path, exist_ok=True)

    deltas_1 = pd.DataFrame({"model_name": ["random,L=180"], "delta": [1]})
    deltas_2 = pd.DataFrame({"model_name": ["TVD,L=180"], "delta": [2]})
    deltas_1.to_csv(model_1_path / "deltas.csv", index=False)
    deltas_2.to_csv(model_2_path / "deltas.csv", index=False)

    return {
        "evaluations": {"random,L=180": [str(model_1_path)], "TVD,L=180": [str(model_2_path)]},
        "summary_path": str(summary_path),
    }


def test_summarize_metrics_creates_summary_file(mock_config):
    summarize_metrics(mock_config)
    summary_file = os.path.join(mock_config["summary_path"], "metrics.csv")

    assert os.path.exists(summary_file)

    summary_df = pd.read_csv(summary_file)
    assert not summary_df.empty
    assert "delta_mean" in summary_df.columns
    assert "delta_stde" in summary_df.columns


def test_summarize_metrics_creates_model_folders(mock_config):
    summarize_metrics(mock_config)

    model_1_path = os.path.join(mock_config["summary_path"], "random,L=180")
    model_2_path = os.path.join(mock_config["summary_path"], "TVD,L=180")

    assert os.path.exists(model_1_path)
    assert os.path.exists(model_2_path)


def test_summarize_metrics_creates_latex_table(mock_config):
    summarize_metrics(mock_config)
    latex_file = os.path.join(mock_config["summary_path"], "delta_table.tex")

    assert os.path.exists(latex_file)
    with open(latex_file, "r") as f:
        content = f.read()
        assert "\\begin{tabular}" in content
