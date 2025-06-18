import os

import pandas as pd
import pytest

from topic_transition.evaluation import calculate_deltas_for_dataset


@pytest.fixture
def mock_events():
    data = {
        "date": list(pd.to_datetime(["2023-01-01", "2023-01-06", "2023-01-10"]).date),
        "description": ["Event 1", "Event 2", "Event 3"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_indicators_csv(tmp_path):
    data = {
        "date": list(pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-07"]).date),
        "indicator_value": [0, 1, 0],
    }
    csv_path = tmp_path / "indicator_values.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_training_path(tmp_path, mock_indicators_csv):
    year_path = tmp_path / "2023"
    section_path = year_path / "section_1"
    section_path.mkdir(parents=True, exist_ok=True)
    mock_indicators_csv.rename(section_path / "indicator_values.csv")
    return section_path


@pytest.fixture
def mock_evaluation_path(tmp_path):
    eval_path = tmp_path / "evaluation"
    eval_path.mkdir(parents=True, exist_ok=True)
    return eval_path


def test_calculate_deltas_for_dataset_generates_expected_output(mock_training_path, mock_events, mock_evaluation_path):
    result_df = calculate_deltas_for_dataset(
        training_path=str(mock_training_path),
        events=mock_events,
        evaluation_path=str(mock_evaluation_path),
        model_name="test_model",
    )

    assert not result_df.empty
    assert "model_name" in result_df.columns
    assert result_df.loc[0, "model_name"] == "test_model"
    assert "delta" in result_df.columns
    assert result_df.loc[0, "delta"] == 2
