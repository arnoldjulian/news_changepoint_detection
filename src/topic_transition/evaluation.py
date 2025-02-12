"""Tools for evaluation."""
import os
from datetime import date

import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from plotly import graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from topic_transition.utils import get_last_day_of_month

nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def load_dataset(
    training_path: str | None = None, dataset_base_path: str | None = None, dataset_path: str | None = None
) -> dict[str, list]:
    """
    Load dataset.

    Parameters
    ----------
    training_path : str or None, optional
        The file path where the training data is stored. If specified, this and `dataset_base_path`
        are used to construct the `dataset_path` dynamically.
    dataset_base_path : str or None, optional
        The base directory path where datasets are stored. Used in conjunction with `training_path`
        to build the full `dataset_path`.
    dataset_path : str or None, optional
        The full file path to the dataset. If specified, it is directly used to load the dataset.

    Returns
    -------
    dict[str, list]
        A dictionary representation of the dataset, with keys as column names and values as lists
        of column data. The dataset includes a new 'date' column derived from the 'webPublicationDate'
        column.
    """
    if training_path is not None and dataset_base_path is not None:
        year, section_id = training_path.split(os.path.sep)[-3:-1]
        dataset_path = os.path.join(dataset_base_path, year)
        dataset_path = os.path.join(dataset_path, f"{section_id}.pkl")
    elif dataset_path is not None:
        dataset_path = pd.read_pickle(dataset_path)
    else:
        raise ValueError("No dataset path provided!")
    dataset = pd.read_pickle(dataset_path)  # type: ignore
    dataset["date"] = pd.to_datetime(dataset["webPublicationDate"]).dt.date
    return dataset


def get_events(training_path: str, dataset_base_path: str, events_base_path: str) -> pd.DataFrame:
    """
    Calculate significance scores.

    Parameters
    ----------
    training_path : str
        The file path to the training data, used to determine `year` and `section_id`.
    dataset_base_path : str
        The base directory path where the dataset files are stored.
    events_base_path : str
        The base directory path where the events data is stored.
    """
    time_interval, section_id = training_path.split(os.path.sep)[-3:-1]
    dataset_path = os.path.join(dataset_base_path, time_interval)
    dataset_path = os.path.join(dataset_path, f"{section_id}.pkl")
    dataset = pd.read_pickle(dataset_path)
    dataset["date"] = pd.to_datetime(dataset["webPublicationDate"]).dt.date
    return get_events_with_significance_scores(dataset, time_interval, section_id, events_base_path)


def get_events_with_significance_scores(
    dataset: pd.DataFrame | str,
    time_interval: str,
    section_id: str,
    events_base_path: str,
):
    """
    Calculate significance scores.

    Parameters
    ----------
    dataset : pd.DataFrame
        The primary dataset for which significance scores are to be calculated.
    time_interval : str
        The specific year for the events to be considered.
    section_id : str
        Identifier for a particular section to fetch corresponding events.
    events_base_path : str
        Base directory path where event files are stored.

    Returns
    -------
    events : pd.DataFrame
        DataFrame containing the events data enriched with significance scores.
    """
    if isinstance(dataset, str):
        dataset = pd.read_pickle(dataset)
    events_dir = os.path.join(events_base_path, time_interval)
    events_path = os.path.join(events_dir, f"{section_id}.csv")
    if os.path.exists(events_path):
        events = pd.read_csv(events_path)
    else:
        events = pd.read_csv(os.path.join(events_dir, "world.csv"))
    events["date"] = pd.to_datetime(events["date"]).dt.date
    calculate_significance_scores(dataset, events, "yearly")  # type: ignore
    return events


def get_events_from_path(dataset: pd.DataFrame | str, events_path: str):
    """Get events with significance scores."""
    if isinstance(dataset, str):
        dataset = pd.read_pickle(dataset)
    events = pd.read_csv(events_path)
    events["date"] = pd.to_datetime(events["date"]).dt.date
    calculate_significance_scores(dataset, events, "yearly")  # type: ignore
    return events


def calculate_significance_scores(articles: pd.DataFrame, events: pd.DataFrame, time_interval: str = "yearly"):
    """
    Calculate significance scores for all score types for a specific dataset.

    Parameters
    ----------
    articles : pd.DataFrame
        DataFrame containing articles data with columns "webTitle" and "tokenized".

    events : pd.DataFrame
        DataFrame containing events data with columns "description" and "tokenized".

    time_interval : str, optional
        The time interval for calculating significance scores. Default is "yearly".

    """

    def process_text(text):
        tokens = word_tokenize(text)
        return [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]

    articles["tokenized"] = articles["webTitle"].apply(process_text)
    events["description"] = events["description"].fillna("")
    if len(events) == 1:
        events[f"{time_interval}_avg_tfidf_score"] = 1.0
    else:
        events["tokenized"] = events["description"].apply(process_text)
        vectorizer = TfidfVectorizer()
        vectorizer.fit(list(events["description"]) + list(articles["webTitle"]))
        news_vectors = vectorizer.transform(articles["webTitle"]).toarray()
        event_vectors = vectorizer.transform(events["description"]).toarray()
        tfidf_mtx = cosine_similarity(event_vectors, news_vectors)
        events[f"{time_interval}_avg_tfidf_score"] = tfidf_mtx.mean(axis=1)


def plot_indicators(
    indicators: pd.DataFrame,
    most_significant_event: pd.Series,
    year: int | str,
    section_id: int | str,
    evaluation_path: str,
    significance_score: str,
    topk: int,
) -> None:
    """
    Plot indicator values with most significant event(s) displayed as a red line.

    Parameters
    ----------
    indicators : pd.DataFrame
        A pandas DataFrame containing the indicator values. It should have columns "date" and "indicator_value".

    most_significant_event : pd.Series
        A pandas Series representing the most significant event. It should have columns "date" and "description".

    year : int or str
        The year of the evaluation.

    section_id : int or str
        The ID of the section being evaluated.

    evaluation_path : str
        The path where the evaluation file will be saved.

    Returns
    -------
    None

    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=indicators["date"],
            y=indicators["indicator_value"],
            mode="lines",
            name="Indicator values",
        )
    )
    fig.update_yaxes(range=[0.5, max(indicators["indicator_value"])])

    first_date = indicators["date"].min()
    last_date = indicators["date"].max()
    min_date = date(year=first_date.year, month=first_date.month, day=1)
    max_date = get_last_day_of_month(last_date)
    fig.update_xaxes(range=[min_date, max_date])

    significant_date = most_significant_event["date"]
    fig.add_shape(
        type="line",
        x0=significant_date,
        y0=0,
        x1=significant_date,
        y1=max(indicators["indicator_value"]),
        line={"color": "green"},
        name="Significant Event",
    )

    max_idx = indicators["indicator_value"].argmax()
    max_date = indicators["date"][max_idx]  # type: ignore

    if abs((max_date - significant_date).days) < 2:
        # Plot one of the lines as a dashed line if they are too close.
        fig.add_shape(
            type="line",
            x0=max_date,
            y0=0,
            x1=max_date,
            y1=max(indicators["indicator_value"]),
            line={"color": "red", "dash": "dash"},
            name="Max Indicator Value",
        )
    else:
        fig.add_shape(
            type="line",
            x0=max_date,
            y0=0,
            x1=max_date,
            y1=max(indicators["indicator_value"]),
            line={"color": "red"},
            name="Max Indicator Value",
        )

    fig.update_layout(
        title=f"{year} / {section_id}",
        xaxis_title="Date",
        yaxis_title="Indicator value",
        autosize=False,
        width=800,
        height=500,
    )

    indicators_file_base = os.path.join(
        evaluation_path, f"{year}_{section_id}_{significance_score}_indicators_top{topk}"
    )
    html_filename = f"{indicators_file_base}.html"
    fig.write_html(html_filename)
    jpg_filename = f"{indicators_file_base}.jpg"
    fig.write_image(jpg_filename)


def calculate_avg_indicators_for_dataset(indicators: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average indicators.

    Parameters
    ----------
    training_path : str
        The file path to the directory containing the indicators data file.

    events : pd.DataFrame
        A DataFrame containing the event data, which includes a date column.
    """
    # Ensure dates are comparable
    events["date"] = pd.to_datetime(events["date"]).dt.date

    # Normalize the yearly_avg_tfidf_score in the events dataframe
    total_tfidf_score = events["yearly_avg_tfidf_score"].sum()
    if total_tfidf_score != 0:
        events["normalized_tfidf_score"] = events["yearly_avg_tfidf_score"] / total_tfidf_score
    else:
        events["normalized_tfidf_score"] = 0

    # Merge indicators with events
    merged_df = pd.merge(indicators, events, on="date", how="inner")

    # Normalize the indicator values to sum to 1.0
    total_indicator_value = merged_df["indicator_value"].sum()
    if total_indicator_value != 0:
        merged_df["normalized_indicator_value"] = merged_df["indicator_value"] / total_indicator_value
    else:
        merged_df["normalized_indicator_value"] = 0

    # Calculate averages
    avg_indicator_value = merged_df["normalized_indicator_value"].mean()

    # Calculate weighted average
    weighted_avg = merged_df["normalized_indicator_value"] * merged_df["normalized_tfidf_score"]
    weighted_avg = weighted_avg.mean()  # type: ignore

    # Calculate correlation
    if len(merged_df["indicator_value"]) < 2:
        correlation = np.nan
    else:
        correlation = merged_df["indicator_value"].corr(merged_df["yearly_avg_tfidf_score"])

    # Create a DataFrame with the results
    result_df = pd.DataFrame(
        {
            "indicators_average": [avg_indicator_value],
            "weighted_indicators_average": [weighted_avg],
            "indicator_significance_correlation": [correlation],
        }
    )

    return result_df
