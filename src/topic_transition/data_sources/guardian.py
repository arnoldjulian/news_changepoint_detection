"""Tools for fetching article data from Guardian."""

import functools
import logging
import math
import os
import time
from datetime import date
from typing import Any, Callable

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from httpx import RequestError
from tqdm import tqdm

from topic_transition.utils import get_dates_for_interval

logger = logging.getLogger("httpx")
logger.setLevel(logging.WARNING)

logger = logging.getLogger("guardian")
logger.setLevel(logging.INFO)


MAX_RESULTS_PER_PAGE = 200
VALID_SECTION_IDS = [
    "business",
    "global",
    "global-development",
    "politics",
    "technology",
    "uk-news",
    "us-news",
    "world",
]
COLUMNS = [
    "id",
    "type",
    "sectionId",
    "sectionName",
    "webPublicationDate",
    "webTitle",
    "webUrl",
    "apiUrl",
    "isHosted",
    "pillarId",
    "pillarName",
    "text",
]


total_requests = 0
max_requests = 500
wait_after_request = 1


def get_article_text(text_url: str) -> str | None:
    """
    Retrieve and parse the text content of an article from a given URL.

    Parameters
    ----------
    text_url : str
        The URL of the article's text content.

    Returns
    -------
    str
        The parsed text content of the article.
    """
    try:
        response = get_with_waiting(text_url).text
        soup = BeautifulSoup(response, "html.parser")
        paragraphs = soup.find_all("p")
        return "\n".join([p.get_text() for p in paragraphs])
    except RuntimeError:
        return None


def get_page_metadata(
    start_date: date, end_date: date, sections: list[str], api_key: str, page: int = 1
) -> tuple[list, int] | None:
    """
    Retrieve metadata for articles published on a specific date and within specified sections.

    Parameters
    ----------
    start_date : date
        The start date for the time interval.
    end_date : date
        The end date for the time interval.

    sections : list of str
        A list of section names for which metadata is to be retrieved.
    api_key : str
        The API key required for accessing the source of metadata.
    page : int, optional
        The page number of the metadata to retrieve (default is 1).

    Returns
    -------
    tuple[list, dict]
        A tuple containing two elements:
        1. A list of metadata for articles published on the specified date and within the specified sections.
        2. Additional metadata or information related to the retrieval process.
    """
    endpoint_url = f"https://content.guardianapis.com/search?section={'|'.join(sections)}"
    params = {
        "page": page,
        "from-date": str(start_date),
        "to-date": str(end_date),
        "api-key": api_key,
        "page-size": MAX_RESULTS_PER_PAGE,
    }
    try:
        response = get_with_waiting(endpoint_url, params).json()
        response = response["response"]
        return response["results"], response["pages"]
    except RuntimeError:
        return None


def rate_limited(limit: int) -> Callable:
    """
    Decorate a function to limit the number of requests.

    Parameters
    ----------
    limit : int
        The maximum number of times the decorated function can be called before hitting the rate limit.

    Returns
    -------
    callable
        A decorator function that can be applied to other functions.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: list, **kwargs: dict) -> Any:
            global total_requests
            if total_requests >= limit:
                raise RuntimeError("Rate limit exceeded")
            result = func(*args, **kwargs)
            total_requests += 1
            return result

        return wrapper

    return decorator


def get_with_waiting(
    endpoint: str, params: dict | None = None, retry_times: int = 5, wait: float = 30
) -> httpx.Response:
    """
    Attempt to send a GET request to the specified endpoint.

    Parameters
    ----------
    endpoint : str
        The URL of the endpoint to send the GET request to.
    params : dict, optional
        The parameters to include in the GET request (default is None).
    retry_times : int, optional
        The number of times to retry the request in case of failure (default is 5).
    wait : float, optional
        The waiting period (in seconds) between retries (default is 30).

    Returns
    -------
    dict
        The JSON response received from the server.

    Raises
    ------
    RuntimeError
        If all retry attempts fail, or if the rate limit is exceeded.
    """
    for _ in range(retry_times):
        response = httpx.get(endpoint, params=params)
        time.sleep(wait_after_request)
        if response.status_code == 200:
            time.sleep(wait_after_request)
            return response
        if response.status_code == 429:
            logger.warning(f"429: Too many requests: {response.text}. Endpoint: {endpoint}. Retrying.")
            time.sleep(wait)
            continue
        else:
            raise RuntimeError(f"Failed to fetch data: {response.text}. Endpoint: {endpoint}")
    raise RuntimeError(f"Out of retries. Endpoint: {endpoint}")


def get_interval_metadata(start_date: date, end_date: date, sections: list[str], guardian_api_key: str) -> list[str]:
    """
    Retrieve metadata for articles published on a specific date within specified sections.

    Parameters
    ----------
    start_date : date
        The start date for the time interval.
    end_date : date
        The end date for the time interval.
    sections : list of str
        The list of section names for which to retrieve metadata.
    guardian_api_key : str
        The API key for accessing the Guardian API.

    Returns
    -------
    list
        A list of metadata objects for articles published on the specified date and sections.
    """
    articles_data = []
    response = get_page_metadata(start_date, end_date, sections, guardian_api_key, page=1)
    if response is None:
        return []
    results, total_pages = response
    articles_data.extend(results)
    for page in range(2, total_pages + 1):
        response = get_page_metadata(start_date, end_date, sections, guardian_api_key, page=page)
        if response is None:
            continue
        results, _ = response
        articles_data.extend(results)
    return articles_data


def fetch_data_for_interval(
    guardian_api_key: str, output_dir: str, sections: list[str], start_date, end_date, day_delta: int
) -> None:
    """
    Fetch data from the Guardian API for articles published within specified sections for a given year.

    Parameters
    ----------
    guardian_api_key : str
        The API key for accessing the Guardian API.
    output_dir : str
        The directory where the fetched data will be stored.
    sections : list of str
        The list of section names for which to fetch data.
    year : int
        The year for which to fetch data.
    day_delta : int
        How many days should one request cover.

    Returns
    -------
    None
     This function does not return any value; it fetches and saves data to the specified directory.
    """
    year = start_date.year
    if start_date.month == 1 and start_date.day == 1 and end_date.month == 12 and end_date.day == 31:
        dates_dir = os.path.join(output_dir, str(year))
    else:
        dates_dir = os.path.join(
            output_dir,
            f"{start_date.year}-{start_date.month}-{start_date.day}_{end_date.year}-{end_date.month}-{end_date.day}",
        )
    filtered_sections = []
    for section in sections:
        pkl_path = os.path.join(dates_dir, section + ".pkl")
        if not os.path.isfile(pkl_path):
            filtered_sections.append(section)
    sections = filtered_sections
    os.makedirs(dates_dir, exist_ok=True)

    for section in sections:
        interval_dates = get_dates_for_interval(start_date, end_date)
        num_intervals = max(1, math.ceil(len(interval_dates) // day_delta))
        progress_bar = tqdm(total=num_intervals, desc=f"Processing articles for {year} / {section}")
        all_interval_paths = []
        all_intervals = []
        for i in range(num_intervals):
            interval = interval_dates[i * day_delta : (i + 1) * day_delta]  # noqa
            if len(interval) == 0:
                continue
            start_date = interval[0]
            end_date = interval[-1]
            dt_interval = f"{str(start_date)}-{str(end_date)}"
            interval_path = os.path.join(dates_dir, f"{section}_{dt_interval}.pkl")
            all_interval_paths.append(interval_path)
            if os.path.isfile(interval_path):
                logger.warning(f"Loading data for {year} / {dt_interval} from disk")
                interval_data = pd.read_pickle(interval_path)
            else:
                try:
                    md = get_interval_metadata(start_date, end_date, [section], guardian_api_key)
                except RequestError as e:
                    logger.error(f"Failed at interval {start_date}-{end_date} / {str(section)} due to {str(e)}")
                    raise e
                if len(md) > 0:
                    interval_data = pd.DataFrame(md)
                    interval_data["text"] = interval_data["webUrl"].apply(lambda url: get_article_text(url))
                else:
                    interval_data = pd.DataFrame(columns=COLUMNS)
                interval_data.to_pickle(interval_path)
            all_intervals.append(interval_data)

            progress_bar.update(1)
        logger.info("Saving data")
        section_data = pd.concat(all_intervals)
        if len(section_data) > 0:
            section_data["date"] = pd.to_datetime(section_data["webPublicationDate"])
            section_data.sort_values(by=["date"], inplace=True)
            section_data.drop("date", axis=1, inplace=True)
            section_path = os.path.join(dates_dir, f"{section}.pkl")
            section_data.to_pickle(section_path)
        else:
            logger.warning(f"No data for {year} / {section}")
        logger.info("Cleaning up interval files.")
        for interval_path in all_interval_paths:
            os.remove(interval_path)
