"""Data processing-related functions."""
import random

import pandas as pd
import spacy

LANGUAGE_MODEL = spacy.load("en_core_web_sm")


random.seed(42)


def get_splits(articles: pd.DataFrame, shuffle: bool = True, train_ratio: float = 0.8) -> tuple[list[int], list[int]]:
    """
    Split the DataFrame indices into training and validation sets based on unique dates.

    Parameters
    ----------
    articles
        The DataFrame containing the data to be split.
    shuffle
        Whether to shuffle the indices within each date (default is True).
    train_ratio
        The ratio of samples to be included in the training set (default is 0.8).
    """
    train_idxs = []
    val_idxs = []
    days = articles["date"].unique()
    for day in days:
        day_idxs = list(articles[articles["date"] == day].index)
        if shuffle:
            random.shuffle(day_idxs)
        train_size = int(len(day_idxs) * train_ratio)
        val_size = len(day_idxs) - train_size
        if train_size == 0 or val_size == 0:
            train_list = []
            val_list = []
            for item in day_idxs:
                if random.random() < train_ratio:
                    train_list.append(item)
                else:
                    val_list.append(item)
        else:
            train_list = day_idxs[:train_size]
            val_list = day_idxs[train_size:]
        train_idxs.extend(train_list)
        val_idxs.extend(val_list)
    return train_idxs, val_idxs


def preprocess_text(text: str) -> str:
    """
    Preprocesses text.

    Parameters
    ----------
    text
        Unprocessed article text.
    """
    text = text.lower()
    doc = LANGUAGE_MODEL(text)
    tokens = [
        token.lemma_ for token in doc if not token.is_stop and token.is_alpha and not token.is_punct and len(token) > 2
    ]

    processed_text = " ".join(tokens)
    return processed_text
