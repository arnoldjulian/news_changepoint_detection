"""Fetch all data (metadata and text) of Guardian newspaper articles."""
import argparse
import logging
import os
from datetime import date, datetime

from topic_transition.guardian import VALID_SECTION_IDS, fetch_data_for_interval

logger = logging.getLogger("guardian")
logger.setLevel(logging.INFO)


def main(
    start_date: date, end_date: date, sections: list[str], output_dir: str, guardian_api_key: str, day_delta: int
) -> None:
    """Fetch Guardian data for a time interval."""
    fetch_data_for_interval(guardian_api_key, output_dir, sections, start_date, end_date, day_delta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and validate .pkl files containing dataframes")
    parser.add_argument("sections", help="Download all articles with this 'sectionId'.")
    parser.add_argument(
        "--output_dir", default="dvc/datasets/guardian", help="Output directory to save merged dataframe"
    )
    parser.add_argument("--log_file_path", help="File to log to. If no path is given, log to console.")
    parser.add_argument("--day_delta", help="How many days should one request cover.", type=int, default=10)
    parser.add_argument("--start_date", help="Download articles starting from this date.")
    parser.add_argument("--end_date", help="Download articles ending at this date.")
    args = parser.parse_args()

    guardian_api_key = os.environ["GUARDIAN_API_KEY"]
    if guardian_api_key is None:
        raise ValueError("GUARDIAN_API_KEY is not specified.")
    sections = args.sections.split(",")
    for section in sections:
        if section not in VALID_SECTION_IDS:
            raise ValueError(f"Invalid section name: {section}")

    if args.log_file_path:
        file_handler = logging.FileHandler(args.log_file_path)
        logger.addHandler(file_handler)

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    main(start_date, end_date, sections, args.output_dir, guardian_api_key, args.day_delta)
