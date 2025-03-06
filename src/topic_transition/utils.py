"""Utility functions."""
import calendar
import datetime
import glob
import os
import pickle
import random
import re
import time
from datetime import date, timedelta

import numpy as np
import torch
from gensim import corpora
from gensim.models import LdaModel

from topic_transition.data import preprocess_text


def get_dates_for_interval(start_date: date, end_date: date) -> list[date]:
    """Get a complete list of dates for a given time interval."""
    return [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]


def get_fischer_information_change(x_data: np.ndarray) -> np.ndarray:
    """Calculate the differences in Fischer Information."""
    eps = np.finfo(x_data.dtype).eps
    x_data += eps
    dp1 = 1
    der_sq = ((np.log(x_data) - np.roll(np.log(x_data), -1, axis=0)) / (2 * dp1)) ** 2
    der_sq[np.isnan(der_sq)] = 0
    tot = x_data * der_sq
    tot[np.isnan(tot)] = 0
    tot[np.isinf(tot)] = 0
    tot_sum = np.sum(tot[:-1], axis=1)
    return tot_sum


def format_time_period(start_date: str | datetime.date, end_date: str | datetime.date) -> str:
    """Format time period."""
    if isinstance(start_date, str):
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    else:
        start = start_date
    if isinstance(end_date, str):
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    else:
        end = end_date

    if start.year == end.year and start.month == 1 and start.day == 1 and end.month == 12 and end.day == 31:
        return start.strftime("%Y")

    last_day = calendar.monthrange(start.year, start.month)[1]
    if start.year == end.year and start.month == end.month and start.day == 1 and end.day == last_day:
        return start.strftime("%Y-%m")

    return f"{start_date}_{end_date}"


def get_last_day_of_month(dt: datetime.date) -> datetime.date:
    """Get last day of the month for a given date."""
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    return datetime.date(dt.year, dt.month, last_day)


def total_variation_distance(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Calculate Total Variation Distance between 2 arrays."""
    if len(arr1.shape) == 1:
        arr1 = arr1 / np.sum(arr1, keepdims=True)
        arr2 = arr2 / np.sum(arr2, keepdims=True)
        abs_diff = np.abs(arr1 - arr2)
        sum_abs_diff = np.sum(abs_diff)
        return sum_abs_diff / 2
    else:
        arr1 = arr1 / np.sum(arr1, axis=1, keepdims=True)
        arr2 = arr2 / np.sum(arr2, axis=1, keepdims=True)
        abs_diff = np.abs(arr1 - arr2)
        sum_abs_diff = np.sum(abs_diff, axis=1)
        return sum_abs_diff / 2


def get_f1_scores(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Calculate f1 scores for different texts."""
    common_tokens = pred & ref
    len_arr = np.vectorize(len)
    # Use np.errstate to handle warnings for division by zero and invalid values
    with np.errstate(divide="ignore", invalid="ignore"):
        precisions = len_arr(common_tokens) / len_arr(pred)
        recalls = len_arr(common_tokens) / len_arr(ref)
    precisions[np.where(np.isnan(precisions))] = 0
    recalls[np.where(np.isnan(recalls))] = 0
    denoms = precisions + recalls
    denoms[np.where(denoms == 0)] = 1
    f1s = 2 * precisions * recalls / denoms
    return f1s


def increment_path_number(path: str) -> str:
    """Create a new path with an incremented numerical suffix for 'num_<number>'."""
    training_paths = glob.glob(path + "_num_*")
    training_nums = [int(re.match(r".*_num_(\d+)$", tr_path).group(1)) for tr_path in training_paths]
    new_training_num = max(training_nums) + 1 if training_nums else 1
    new_path = f"{path}_num_{new_training_num}"

    return new_path


def find_matching_directories(base_path: str) -> list:
    """Find directories that match the given base path pattern."""
    search_pattern = base_path + "_*"
    matching_paths = glob.glob(search_pattern)

    matching_dirs = [path for path in matching_paths if os.path.isdir(path)]
    return matching_dirs


def load_or_train_lda(dataset, force_new_tvd, lda_config, lda_path):
    """Load an LDA model if exists. Otherwise train and save a new model."""
    corpus_path = os.path.join(lda_path, "corpus.pkl")
    dictionary_path = os.path.join(lda_path, "dictionary.pkl")
    if os.path.exists(corpus_path) and os.path.exists(dictionary_path):
        with open(corpus_path, "rb") as fr:
            corpus = pickle.load(fr)
        with open(dictionary_path, "rb") as fr:
            dictionary = pickle.load(fr)
    else:
        dataset["preprocessed_text"] = dataset["text"].apply(preprocess_text)
        corpus = dataset["preprocessed_text"].apply(lambda doc: doc.split())
        dataset.drop("preprocessed_text", inplace=True, axis=1)
        dictionary = corpora.Dictionary(corpus)
        dictionary.filter_extremes(no_below=10)
        corpus = [dictionary.doc2bow(text) for text in corpus]

        with open(corpus_path, "wb") as fw:
            pickle.dump(corpus, fw)
        with open(dictionary_path, "wb") as fw:
            pickle.dump(dictionary, fw)
    if os.path.exists(lda_path) and force_new_tvd:
        lda_path = increment_path_number(lda_path)
    os.makedirs(lda_path, exist_ok=True)
    lda_model_path = os.path.join(lda_path, "lda.model")
    if os.path.exists(lda_model_path):
        lda_model = LdaModel.load(lda_model_path)
    else:
        lda_config.update({"id2word": dictionary})
        lda_model = LdaModel(corpus, **lda_config)
        lda_model.save(lda_model_path)
    return corpus, lda_model


def set_random_seed(deterministic: bool):
    """Set the random seed for reproducibility in both deterministic and non-deterministic modes."""
    seed = 42 if deterministic else int(time.time())  # Use fixed seed or system time
    print(f"Setting random seed: {seed}")  # Optional: For debugging
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
